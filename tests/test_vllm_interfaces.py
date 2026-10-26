"""Worker and platform hooks whose contract changed in vLLM 0.32 and that no serving test reaches without an NPU."""

import inspect
from types import SimpleNamespace

import pytest
from vllm.platforms.interface import Platform
from vllm.v1.worker.worker_base import CompilationTimes

from tests.test_mblt_platform_prefill import _make_vllm_config
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


def test_async_scheduling_is_turned_off() -> None:
    # vLLM 0.32 enables it by default; the engine would then call sample_tokens(), which this worker leaves
    # unimplemented because execute_model() already samples.
    config = _make_vllm_config(64)
    config.scheduler_config.async_scheduling = True

    MbltPlatform.check_and_update_config(config)

    assert config.scheduler_config.async_scheduling is False


def test_sampling_helpers_compile_without_a_host_toolchain() -> None:
    # vLLM's prompt-logprob rank count is torch.compile'd with this backend; inductor needs a working C++ toolchain.
    assert MbltPlatform.simple_compile_backend == "eager"


def test_top_k_top_p_runs_on_host_logits_without_triton() -> None:
    # The worker module turns vLLM's Triton top-k/top-p path off: the logits are host tensors.
    import torch
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

    assert topk_topp_sampler.HAS_TRITON is False
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    out = topk_topp_sampler.apply_top_k_top_p(logits.clone(), torch.tensor([2]), torch.tensor([0.9]))
    assert torch.isinf(out[0, 0]) and torch.isinf(out[0, 3]) and not torch.isinf(out[0, 1])
