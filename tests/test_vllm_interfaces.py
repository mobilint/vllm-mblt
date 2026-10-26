"""Worker and platform hooks whose contract changed in vLLM 0.32 and that no serving test reaches without an NPU."""

import inspect
from types import SimpleNamespace

import pytest
from vllm.platforms.interface import Platform
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_mblt.mblt_platform import MbltPlatform
from vllm_mblt.mblt_worker import MbltWorker


def test_compile_or_warm_up_model_returns_compilation_times() -> None:
    # The executor reads `.language_model` from every worker's result.
    worker = MbltWorker.__new__(MbltWorker)
    worker._log_init_stage = lambda *args, **kwargs: None

    assert worker.compile_or_warm_up_model() == CompilationTimes(language_model=0.0, encoder=0.0)


def test_manual_seed_all_is_implemented() -> None:
    # vLLM's set_random_seed calls it; the base implementation raises NotImplementedError.
    MbltPlatform.manual_seed_all(1234)


def test_load_model_refuses_dummy_weights() -> None:
    worker = MbltWorker.__new__(MbltWorker)
    worker.model_config = SimpleNamespace(model="mobilint/Qwen3-0.6B")

    with pytest.raises(NotImplementedError, match="dummy weights"):
        worker.load_model(load_dummy_weights=True)


def test_validate_request_takes_vllm_032_arguments() -> None:
    expected = list(inspect.signature(Platform.validate_request).parameters)
    assert list(inspect.signature(MbltPlatform.validate_request).parameters) == expected


def test_supported_kv_cache_layouts_need_no_attention_backend() -> None:
    # The base implementation looks up attention backends, which this platform has none of.
    worker = MbltWorker.__new__(MbltWorker)

    layouts = worker.get_supported_kv_cache_layouts()

    assert layouts and all(isinstance(name, str) for name in layouts)
