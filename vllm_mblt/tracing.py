"""NPU event tracing for the MBLT platform.

qbruntime records NPU activity as a Chrome Tracing JSON log through
`start_tracing_events(path)` / `stop_tracing_events()`, which can be opened in
https://ui.perfetto.dev/. This module wraps that pair so `MbltWorker` can serve
vLLM's worker profiler hook (`Worker.profile(is_start)`) with it, the way
out-of-tree platforms wire in their own device profiler.

Two properties of the qbruntime tracer shape the code here:

- The log is buffered in the process and written only when
  `stop_tracing_events()` returns, so a trace has to be a bounded window
  (`/start_profile` .. `/stop_profile`) rather than something left on for the
  life of a server, and an unstopped trace is a lost trace.
- Tracing is a process-global facility with no handle to scope it and no way
  to ask who owns it, so a second `start` while one is already recording would
  silently take over the first one's window. Starts issued through this module
  are tracked and refused; an already-imported `mblt_model_zoo` benchmark
  helper is consulted too, which is as far as ownership can be established
  without an ownership query in qbruntime itself.
"""

import os
import socket
import sys
import threading
import time
from typing import Any, Optional

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# vLLM's own switch for worker profiling. Reused rather than replaced with an
# MBLT-specific variable so `/start_profile`, `LLM.start_profile()` and
# `vllm bench serve --profile` keep working unchanged: the OpenAI server only
# registers the profile routes when this is set.
TRACE_DIR_ENV_VAR = "VLLM_TORCH_PROFILER_DIR"

# Names the profiler a trace came from, the way vLLM's own front-end traces
# carry `async_llm`, so both are recognizable in one directory listing.
TRACE_NAME = "mblt_npu"

# Process-global record of the tracer that owns the running qbruntime trace.
_ACTIVE_TRACER: Optional["MbltTracer"] = None
_ACTIVE_TRACER_LOCK = threading.Lock()

# `mblt_model_zoo`'s benchmark helpers trace through the same process-global
# qbruntime facility and record their ownership in this module attribute.
_EXTERNAL_TRACE_OWNERS = (("mblt_model_zoo.hf_transformers.utils.benchmark_utils", "_ACTIVE_QBRUNTIME_TRACE_HANDLE"),)


def _external_trace_owner() -> Optional[str]:
    """Name a non-MBLT holder of the process-global qbruntime trace, if any.

    qbruntime exposes no way to ask whether a trace is running or who started
    it, so ownership can only be read from the clients that track it
    themselves. Only modules already imported are inspected -- one that was
    never imported cannot have started a trace -- so this costs no import and
    degrades to "unknown owner" if a client renames its bookkeeping.
    """
    for module_name, attr_name in _EXTERNAL_TRACE_OWNERS:
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, attr_name, None) is not None:
            return module_name
    return None


def resolve_trace_dir() -> Optional[str]:
    """Return the directory traces are written to, or None if tracing is off.

    The value is whatever vLLM resolved the variable to -- it expands `~` and
    makes the path absolute -- rather than a second normalization of the raw
    environment variable, so the NPU trace lands beside the front-end CPU
    trace instead of wherever this module happened to resolve it to.
    """
    trace_dir = envs.VLLM_TORCH_PROFILER_DIR
    if not trace_dir:
        return None
    if "://" in trace_dir:
        # vLLM passes a remote destination such as `gs://bucket/traces`
        # straight through for its own traces. qbruntime writes through a plain
        # file path and cannot reach one, and treating it as a local path would
        # silently create a directory named after the URI.
        raise RuntimeError(f"NPU event tracing needs a local directory, but {TRACE_DIR_ENV_VAR} is {trace_dir!r}.")
    return trace_dir


def trace_filename(rank: int) -> str:
    """Name a trace file the way torch names its own.

    `torch.profiler.tensorboard_trace_handler` builds
    `{hostname}_{pid}.{worker_name}.{time_ns}.pt.trace.json`, noting that the
    nanosecond is there "to avoid naming clash when exporting the trace"; vLLM
    and vllm-ascend both hand it a rank-derived worker name and let it append
    the timestamp. This follows that convention instead of inventing one.

    Every component earns its place. The timestamp separates windows of one
    tracer, tracers built one after another in a process, and a pid reused
    after a restart. The pid separates concurrent processes on one host. The
    hostname separates processes that share a mounted trace directory but not
    a pid namespace, where two containers can hold the same pid.
    """
    return f"{socket.gethostname()}_{os.getpid()}.{TRACE_NAME}_rank{rank}.{time.time_ns()}.json"


def load_backend() -> Any:
    """Import the qbruntime module that provides the tracing entry points."""
    try:
        import qbruntime  # type: ignore
    except Exception as e:  # pragma: no cover - qbruntime is a hard dependency
        raise RuntimeError("NPU event tracing requires qbruntime to be available.") from e
    return qbruntime


class MbltTracer:
    """Record one qbruntime trace file per start/stop window."""

    def __init__(self, trace_dir: str, *, rank: int = 0, backend: Any = None) -> None:
        self.trace_dir = trace_dir
        self.rank = rank
        self._backend = backend
        self._path: Optional[str] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether qbruntime is currently recording into this tracer's file."""
        return self._running

    @property
    def path(self) -> Optional[str]:
        """The file the current or most recent window writes to."""
        return self._path

    def _get_backend(self) -> Any:
        if self._backend is None:
            self._backend = load_backend()
        return self._backend

    def start(self) -> Optional[str]:
        """Begin a trace window and return the destination path.

        Returns None when there is nothing to do because a trace is already
        recording -- a repeated `/start_profile`, or a window owned by another
        client in this process. Anything that means the requested trace will
        not exist raises instead, so the caller does not report success for a
        trace it is not going to get.
        """
        global _ACTIVE_TRACER

        with _ACTIVE_TRACER_LOCK:
            if self._running:
                logger.warning("MBLT NPU trace is already recording into %s; ignoring start.", self._path)
                return None
            # Probed before the test so the owner is known either way; it is
            # a dict lookup, not an import.
            external_owner = _external_trace_owner()
            if _ACTIVE_TRACER is not None or external_owner is not None:
                # Another owner in this process holds the global tracer. Taking
                # it over would cut their window short and drop their file.
                logger.warning(
                    "A qbruntime trace started by %s is already active in this process; ignoring start. "
                    "Stop that trace before starting one through the vLLM profiler hook.",
                    external_owner or "another MBLT tracer",
                )
                return None

            try:
                os.makedirs(self.trace_dir, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare the MBLT NPU trace directory {self.trace_dir}.") from e

            path = os.path.join(self.trace_dir, trace_filename(self.rank))
            try:
                started = self._get_backend().start_tracing_events(path)
            except Exception as e:
                raise RuntimeError(f"Failed to start an MBLT NPU trace at {path}.") from e

            if started is False:
                raise RuntimeError(f"qbruntime refused to start an MBLT NPU trace at {path}.")

            self._path = path
            self._running = True
            _ACTIVE_TRACER = self
            logger.info("Started MBLT NPU trace; the log is written to %s on stop.", path)
            return path

    def stop(self) -> Optional[str]:
        """End the current trace window, write its file and return the path.

        Returns None when no trace of ours was running, which is the harmless
        case of a `/stop_profile` that does not follow a start. A backend that
        fails to write the log raises, because the window is then lost and the
        caller would otherwise be told the trace is on disk.
        """
        global _ACTIVE_TRACER

        with _ACTIVE_TRACER_LOCK:
            if not self._running:
                logger.warning("MBLT NPU trace was not started, nothing to stop.")
                return None

            path = self._path
            try:
                self._get_backend().stop_tracing_events()
            except Exception as e:
                raise RuntimeError(f"Failed to write the MBLT NPU trace for {path}.") from e
            finally:
                # The buffered log is gone either way, and holding the running
                # flag or the global claim would block every later window too.
                self._running = False
                if _ACTIVE_TRACER is self:
                    _ACTIVE_TRACER = None

            logger.info("Stopped MBLT NPU trace; wrote %s. View it at https://ui.perfetto.dev/.", path)
            return path
