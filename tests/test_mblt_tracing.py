import os
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

    def test_resolve_trace_dir_treats_blank_env_var_as_unset(self, monkeypatch) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, "   ")
        assert resolve_trace_dir() is None

    def test_resolve_trace_dir_returns_absolute_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(TRACE_DIR_ENV_VAR, f"  {tmp_path}  ")
        resolved = resolve_trace_dir()
        assert resolved == os.path.abspath(str(tmp_path))
        assert os.path.isabs(resolved)


class TestMbltTracer:
    def test_start_creates_trace_dir_and_passes_rank_qualified_path(self, tmp_path) -> None:
        backend = FakeQbRuntime()
        trace_dir = tmp_path / "traces"
        tracer = MbltTracer(str(trace_dir), rank=2, backend=backend)

        path = tracer.start()

        assert trace_dir.is_dir()
        assert backend.started_paths == [path]
        assert os.path.basename(path) == "mblt_trace_2_0.json"
        assert tracer.is_running

    def test_stop_writes_and_advances_to_the_next_window(self, tmp_path) -> None:
        backend = FakeQbRuntime()
        tracer = MbltTracer(str(tmp_path), backend=backend)

        first = tracer.start()
        assert tracer.stop() == first
        assert not tracer.is_running
        assert backend.stop_count == 1

        second = tracer.start()
        assert second != first
        assert os.path.basename(second) == "mblt_trace_0_1.json"
        assert backend.started_paths == [first, second]

    def test_repeated_start_does_not_restart_the_running_trace(self, tmp_path) -> None:
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

    def test_start_reports_failure_without_claiming_the_trace(self, tmp_path) -> None:
        backend = FakeQbRuntime(start_result=False)
        tracer = MbltTracer(str(tmp_path), backend=backend)

        assert tracer.start() is None
        assert not tracer.is_running
        assert tracing._ACTIVE_TRACER is None

    def test_start_failure_leaves_the_next_start_usable(self, tmp_path) -> None:
        backend = FakeQbRuntime(start_error=RuntimeError("no device"))
        tracer = MbltTracer(str(tmp_path), backend=backend)

        assert tracer.start() is None
        assert not tracer.is_running

        tracer._backend = FakeQbRuntime()
        assert tracer.start() is not None

    def test_stop_failure_releases_the_trace_instead_of_blocking_later_windows(self, tmp_path) -> None:
        backend = FakeQbRuntime(stop_error=RuntimeError("write failed"))
        tracer = MbltTracer(str(tmp_path), backend=backend)

        tracer.start()
        assert tracer.stop() is None
        assert not tracer.is_running
        assert tracing._ACTIVE_TRACER is None
        assert tracer.start() is not None

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
        assert backend.started_paths == [os.path.join(os.path.abspath(str(tmp_path)), "mblt_trace_3_0.json")]

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
