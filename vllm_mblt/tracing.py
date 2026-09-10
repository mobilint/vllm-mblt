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
- Tracing is a process-global facility with no handle to scope it, so a second
  `start` while one is already recording would silently take over the first
  one's window. `mblt_model_zoo`'s benchmark helpers guard the same way.
"""

import os
import threading
from typing import Any, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# vLLM's own switch for worker profiling. Reused rather than replaced with an
# MBLT-specific variable so `/start_profile`, `LLM.start_profile()` and
# `vllm bench serve --profile` keep working unchanged: the OpenAI server only
# registers the profile routes when this is set.
TRACE_DIR_ENV_VAR = "VLLM_TORCH_PROFILER_DIR"

TRACE_FILENAME_PREFIX = "mblt_trace"

# Process-global record of the tracer that owns the running qbruntime trace.
_ACTIVE_TRACER: Optional["MbltTracer"] = None
_ACTIVE_TRACER_LOCK = threading.Lock()


def resolve_trace_dir() -> Optional[str]:
    """Return the absolute directory traces are written to, or None if unset."""
    value = os.getenv(TRACE_DIR_ENV_VAR)
    if value is None or not value.strip():
        return None
    return os.path.abspath(os.path.expanduser(value.strip()))


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
        self._window = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether qbruntime is currently recording into this tracer's file."""
        return self._running

    def _get_backend(self) -> Any:
        if self._backend is None:
            self._backend = load_backend()
        return self._backend

    def trace_path(self, window: Optional[int] = None) -> str:
        """Return the file the given window (default: the next one) writes to."""
        if window is None:
            window = self._window
        return os.path.join(self.trace_dir, f"{TRACE_FILENAME_PREFIX}_{self.rank}_{window}.json")

    def start(self) -> Optional[str]:
        """Begin a trace window. Returns the destination path, or None if not started."""
        global _ACTIVE_TRACER

        with _ACTIVE_TRACER_LOCK:
            if self._running:
                logger.warning("MBLT NPU trace is already recording into %s; ignoring start.", self.trace_path())
                return None
            if _ACTIVE_TRACER is not None:
                # Another owner in this process (for example the mblt_model_zoo
                # benchmark helpers) holds the global tracer. Taking it over
                # would cut their window short and drop their file.
                logger.warning(
                    "Another qbruntime trace is already active in this process; ignoring start. "
                    "Stop that trace before starting one through the vLLM profiler hook."
                )
                return None

            path = self.trace_path()
            try:
                os.makedirs(self.trace_dir, exist_ok=True)
                started = self._get_backend().start_tracing_events(path)
            except Exception as e:
                logger.warning("Failed to start MBLT NPU trace at %s: %s", path, e)
                return None

            if started is False:
                logger.warning("qbruntime refused to start an NPU trace at %s.", path)
                return None

            self._running = True
            _ACTIVE_TRACER = self
            logger.info("Started MBLT NPU trace; the log is written to %s on stop.", path)
            return path

    def stop(self) -> Optional[str]:
        """End the current trace window and write its file. Returns the path written."""
        global _ACTIVE_TRACER

        with _ACTIVE_TRACER_LOCK:
            if not self._running:
                logger.warning("MBLT NPU trace was not started, nothing to stop.")
                return None

            path = self.trace_path()
            try:
                self._get_backend().stop_tracing_events()
            except Exception as e:
                # The buffered log is gone either way; leaving _running set
                # would only block every later window as well.
                logger.warning("Failed to stop MBLT NPU trace for %s: %s", path, e)
                return None
            finally:
                self._running = False
                if _ACTIVE_TRACER is self:
                    _ACTIVE_TRACER = None
                self._window += 1

            logger.info("Stopped MBLT NPU trace; wrote %s. View it at https://ui.perfetto.dev/.", path)
            return path
