import os
import socket
import sys
from types import SimpleNamespace

import pytest

import vllm_mblt.tracing as tracing
from vllm_mblt.mblt_worker import MbltWorker
from vllm_mblt.tracing import TRACE_DIR_ENV_VAR, MbltTracer, resolve_trace_dir


class FakeQbRuntime:
    """Stand-in for the qbruntime module's process-global tracing entry points."""

    def __init__(
        self,
        *,
        start_result: object = True,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self.started_paths: list[str] = []
        self.stop_count = 0
        self._start_result = start_result
        self._start_error = start_error
        self._stop_error = stop_error

    def start_tracing_events(self, path: str) -> object:
        if self._start_error is not None:
            raise self._start_error
        self.started_paths.append(path)
        return self._start_result

    def stop_tracing_events(self) -> None:
        self.stop_count += 1
        if self._stop_error is not None:
            raise self._stop_error


@pytest.fixture(autouse=True)
def _clear_active_tracer():
    tracing._ACTIVE_TRACER = None
    yield
    tracing._ACTIVE_TRACER = None


class TestResolveTraceDir:
    def test_resolve_trace_dir_returns_none_when_env_var_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(TRACE_DIR_ENV_VAR, raising=False)
        assert resolve_trace_dir() is None

    def test_resolve_trace_dir_returns_absolute_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, str(tmp_path))
        resolved = resolve_trace_dir()
        assert resolved == os.path.abspath(str(tmp_path))
        assert os.path.isabs(resolved)

    def test_resolve_trace_dir_makes_a_relative_path_absolute(self, monkeypatch) -> None:
        # vLLM resolves the variable itself; reading its value rather than the
        # raw environment keeps the NPU trace beside the front-end CPU trace.
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, "relative/traces")
        resolved = resolve_trace_dir()
        assert os.path.isabs(resolved)
        assert resolved.endswith(os.path.join("relative", "traces"))

    def test_resolve_trace_dir_rejects_a_remote_destination(self, monkeypatch) -> None:
        # vLLM passes gs:// through for its own traces, but qbruntime writes
        # through a plain file path and would otherwise create a local
        # directory named after the URI.
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, "gs://bucket/traces")
        with pytest.raises(RuntimeError, match="local directory"):
            resolve_trace_dir()


class TestTraceFilename:
    """The name follows torch's own convention; see `tracing.trace_filename`."""

    def test_name_carries_hostname_pid_profiler_rank_and_timestamp(self) -> None:
        host_pid, name, timestamp, extension = tracing.trace_filename(rank=3).split(".")

        assert host_pid == f"{socket.gethostname()}_{os.getpid()}"
        assert name == f"{tracing.TRACE_NAME}_rank3"
        assert timestamp.isdigit()
        assert extension == "json"

    def test_the_timestamp_separates_back_to_back_names(self) -> None:
        # This is what makes a name unique without a counter or a probe, and
        # it is why torch appends a nanosecond rather than a second.
        first = tracing.trace_filename(rank=0)
        second = tracing.trace_filename(rank=0)

        assert first != second
        assert int(second.split(".")[2]) > int(first.split(".")[2])


class TestMbltTracer:
    def test_start_creates_trace_dir_and_passes_a_conventional_path(self, tmp_path) -> None:
        backend = FakeQbRuntime()
        trace_dir = tmp_path / "traces"
        tracer = MbltTracer(str(trace_dir), rank=2, backend=backend)

        path = tracer.start()

        assert trace_dir.is_dir()
        assert backend.started_paths == [path]
        assert os.path.dirname(path) == str(trace_dir)
        host, pid = os.path.basename(path).split(".")[0].rsplit("_", 1)
        assert host == socket.gethostname()
        assert pid == str(os.getpid())
        assert os.path.basename(path).split(".")[1] == "mblt_npu_rank2"
        assert tracer.is_running

    def test_stop_writes_and_the_next_window_gets_its_own_file(self, tmp_path) -> None:
        backend = FakeQbRuntime()
        tracer = MbltTracer(str(tmp_path), backend=backend)

        first = tracer.start()
        assert tracer.stop() == first
        assert not tracer.is_running
        assert backend.stop_count == 1

        second = tracer.start()
        assert second != first
        assert backend.started_paths == [first, second]

    def test_repeated_start_does_not_raise_or_restart_the_running_trace(self, tmp_path) -> None:
        backend = FakeQbRuntime()
        tracer = MbltTracer(str(tmp_path), backend=backend)

        tracer.start()
        assert tracer.start() is None
        assert len(backend.started_paths) == 1
        assert tracer.is_running

    def test_stop_without_start_is_a_no_op(self, tmp_path) -> None:
        backend = FakeQbRuntime()
        tracer = MbltTracer(str(tmp_path), backend=backend)

        assert tracer.stop() is None
        assert backend.stop_count == 0

    def test_start_is_refused_while_another_tracer_holds_the_process_trace(self, tmp_path) -> None:
        # qbruntime tracing is process-global: taking it over would cut the
        # first owner's window short and drop its file.
        backend = FakeQbRuntime()
        owner = MbltTracer(str(tmp_path), rank=0, backend=backend)
        other = MbltTracer(str(tmp_path), rank=1, backend=backend)

        owner.start()
        assert other.start() is None
        assert not other.is_running
        assert len(backend.started_paths) == 1

        owner.stop()
        assert other.start() is not None

    def test_a_second_tracer_in_one_process_does_not_reuse_the_first_path(self, tmp_path) -> None:
        # Two sequential offline LLM instances build one tracer each, with the
        # same rank, pid and hostname; only the timestamp separates them.
        first = MbltTracer(str(tmp_path), backend=FakeQbRuntime())
        first_path = first.start()
        first.stop()

        second = MbltTracer(str(tmp_path), backend=FakeQbRuntime())
        second_path = second.start()

        assert second_path != first_path

    def test_start_raises_when_the_trace_dir_cannot_be_prepared(self, tmp_path) -> None:
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("")
        tracer = MbltTracer(str(blocker / "traces"), backend=FakeQbRuntime())

        with pytest.raises(RuntimeError, match="Failed to prepare"):
            tracer.start()
        assert not tracer.is_running

    def test_a_refused_start_raises_without_claiming_the_trace(self, tmp_path) -> None:
        # Reporting success for a trace that will not exist is worse than
        # failing the profile request.
        backend = FakeQbRuntime(start_result=False)
        tracer = MbltTracer(str(tmp_path), backend=backend)

        with pytest.raises(RuntimeError, match="refused"):
            tracer.start()
        assert not tracer.is_running
        assert tracing._ACTIVE_TRACER is None

    def test_start_failure_raises_and_leaves_the_next_start_usable(self, tmp_path) -> None:
        backend = FakeQbRuntime(start_error=RuntimeError("no device"))
        tracer = MbltTracer(str(tmp_path), backend=backend)

        with pytest.raises(RuntimeError, match="Failed to start"):
            tracer.start()
        assert not tracer.is_running

        tracer._backend = FakeQbRuntime()
        assert tracer.start() is not None

    def test_stop_failure_raises_but_releases_the_trace_for_later_windows(self, tmp_path) -> None:
        backend = FakeQbRuntime(stop_error=RuntimeError("write failed"))
        tracer = MbltTracer(str(tmp_path), backend=backend)

        tracer.start()
        with pytest.raises(RuntimeError, match="Failed to write"):
            tracer.stop()
        assert not tracer.is_running
        assert tracing._ACTIVE_TRACER is None
        assert tracer.start() is not None

    def test_start_defers_to_a_trace_owned_by_an_external_client(self, monkeypatch, tmp_path) -> None:
        # mblt_model_zoo's benchmark helpers drive the same process-global
        # qbruntime tracer and record their ownership in a module attribute.
        module_name, attr_name = tracing._EXTERNAL_TRACE_OWNERS[0]
        monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{attr_name: object()}))
        backend = FakeQbRuntime()
        tracer = MbltTracer(str(tmp_path), backend=backend)

        assert tracer.start() is None
        assert not tracer.is_running
        assert backend.started_paths == []

    def test_start_proceeds_when_the_external_client_holds_no_trace(self, monkeypatch, tmp_path) -> None:
        module_name, attr_name = tracing._EXTERNAL_TRACE_OWNERS[0]
        monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{attr_name: None}))
        tracer = MbltTracer(str(tmp_path), backend=FakeQbRuntime())

        assert tracer.start() is not None

    def test_external_owner_probe_never_imports_the_module(self, monkeypatch) -> None:
        # Inspecting sys.modules only: a module that was never imported cannot
        # have started a trace, and importing it here would pull in the
        # benchmark helpers' heavy dependencies.
        module_name = tracing._EXTERNAL_TRACE_OWNERS[0][0]
        monkeypatch.delitem(sys.modules, module_name, raising=False)

        def fail_on_import(name, *args, **kwargs):
            raise AssertionError(f"unexpected import of {name}")

        monkeypatch.setattr("builtins.__import__", fail_on_import)
        assert tracing._external_trace_owner() is None

    def test_backend_is_imported_lazily(self, tmp_path, monkeypatch) -> None:
        backend = FakeQbRuntime()
        monkeypatch.setattr(tracing, "load_backend", lambda: backend)
        tracer = MbltTracer(str(tmp_path))

        assert tracer._backend is None
        tracer.start()
        assert tracer._backend is backend


class TestMbltWorkerProfileHook:
    def _make_worker(self, *, rank: int = 0) -> MbltWorker:
        worker = MbltWorker.__new__(MbltWorker)
        worker.rank = rank
        worker._tracer = None
        return worker

    def test_profile_reports_how_to_enable_tracing_when_the_env_var_is_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(TRACE_DIR_ENV_VAR, raising=False)
        worker = self._make_worker()

        with pytest.raises(RuntimeError, match=TRACE_DIR_ENV_VAR):
            worker.profile(is_start=True)

    def test_profile_starts_and_stops_a_trace_in_the_configured_dir(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, str(tmp_path))
        backend = FakeQbRuntime()
        monkeypatch.setattr(tracing, "load_backend", lambda: backend)
        worker = self._make_worker(rank=3)

        worker.profile(is_start=True)
        assert worker._tracer is not None
        assert worker._tracer.is_running
        assert len(backend.started_paths) == 1
        started = backend.started_paths[0]
        assert os.path.dirname(started) == os.path.abspath(str(tmp_path))
        assert f".{tracing.TRACE_NAME}_rank3." in os.path.basename(started)

        worker.profile(is_start=False)
        assert not worker._tracer.is_running
        assert backend.stop_count == 1

    def test_profile_reuses_one_tracer_across_windows(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, str(tmp_path))
        monkeypatch.setattr(tracing, "load_backend", lambda: FakeQbRuntime())
        worker = self._make_worker()

        worker.profile(is_start=True)
        tracer = worker._tracer
        worker.profile(is_start=False)
        worker.profile(is_start=True)

        assert worker._tracer is tracer
        worker.profile(is_start=False)

    def test_shutdown_stops_a_trace_left_running(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, str(tmp_path))
        backend = FakeQbRuntime()
        monkeypatch.setattr(tracing, "load_backend", lambda: backend)
        worker = self._make_worker()
        worker.model = None
        worker.cache_model = None
        worker.input_embeddings = None
        worker._infer_output_buffers = None
        worker.runtime_cache = SimpleNamespace(reset=lambda: None)

        worker.profile(is_start=True)
        worker.shutdown()

        assert backend.stop_count == 1
        assert not worker._tracer.is_running

    def test_shutdown_without_tracing_touches_no_tracer(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, str(tmp_path))
        worker = self._make_worker()
        worker.model = None
        worker.cache_model = None
        worker.input_embeddings = None
        worker._infer_output_buffers = None
        worker.runtime_cache = SimpleNamespace(reset=lambda: None)

        worker.shutdown()

        assert worker._tracer is None
