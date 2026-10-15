from types import SimpleNamespace

import pytest

from vllm_mblt import host_threads
from vllm_mblt.mblt_worker import MbltWorker


@pytest.fixture
def helper_calls(monkeypatch):
    calls = []

    def fake_apply(n_engines_on_host=None):
        calls.append(n_engines_on_host)
        return 5

    monkeypatch.setattr(host_threads, "apply_host_thread_defaults", fake_apply)
    return calls


@pytest.mark.parametrize(
    ("vllm_config", "expected"),
    [
        (SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=3, data_parallel_size_local=3)), 3),
        # Multi-node: only this host's ranks share its CPUs.
        (SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=4, data_parallel_size_local=2)), 2),
        (SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_size_local=1)), 1),
        (SimpleNamespace(parallel_config=SimpleNamespace()), 1),
        (SimpleNamespace(), 1),
    ],
)
def test_local_engine_count_uses_local_data_parallel_size(vllm_config, expected):
    assert host_threads.local_engine_count(vllm_config) == expected


def test_apply_passes_local_engine_count_to_transformers_mblt(helper_calls):
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=3))

    assert host_threads.apply_engine_host_thread_defaults(config) == 5
    assert helper_calls == [3]


def test_apply_reports_kept_environment(monkeypatch):
    monkeypatch.setattr(host_threads, "apply_host_thread_defaults", lambda n_engines_on_host=None: None)

    assert host_threads.apply_engine_host_thread_defaults(SimpleNamespace()) is None


def test_worker_init_device_applies_thread_defaults(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "vllm_mblt.mblt_worker.apply_engine_host_thread_defaults",
        lambda vllm_config: seen.append(vllm_config),
    )
    worker = MbltWorker.__new__(MbltWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=2))
    worker.rank = 0
    worker.local_rank = 0

    worker.init_device()

    assert seen == [worker.vllm_config]
