"""Worker and platform hooks whose contract changed in vLLM 0.32 and that no serving test reaches without an NPU."""

import inspect
from types import SimpleNamespace

import pytest
from vllm.platforms.interface import Platform
from vllm.v1.worker.worker_base import CompilationTimes

from tests.test_mblt_platform_prefill import _make_vllm_config
from vllm_mblt.mblt_platform import MbltPlatform
from vllm_mblt.mblt_worker import MbltWorker


class _StaticInputProcessor:
    """Stand-in for vLLM's InputProcessor, which calls the platform with the engine's config."""

    def __init__(self, config) -> None:
        self.vllm_config = config
        self.model_config = config.model_config

    def process(self, inputs) -> None:
        MbltPlatform.validate_request(inputs, SimpleNamespace())


def test_static_batch_frontend_rejects_incompatible_image_layout() -> None:
    import torch
    from vllm.exceptions import VLLMValidationError
    from vllm.multimodal.inputs import PlaceholderRange

    config = _make_vllm_config(
        128,
        hf_config=SimpleNamespace(model_type="mobilint-qwen3_vl", dynamic_vision=False),
        loader_extra_config={
            "text_max_batch_size": 16,
            "static_image_token_offset": 4,
            "static_image_token_length": 64,
        },
    )
    processor = _StaticInputProcessor(config)
    valid = {"type": "multimodal", "mm_placeholders": {"image": [PlaceholderRange(offset=4, length=64)]}}
    processor.process(valid)
    for inputs in (
        {"type": "token"},
        {"type": "multimodal", "mm_placeholders": {"image": [PlaceholderRange(offset=5, length=64)]}},
        {"type": "multimodal", "mm_placeholders": {"image": [PlaceholderRange(offset=4, length=63)]}},
        {
            "type": "multimodal",
            "mm_placeholders": {
                "image": [PlaceholderRange(offset=4, length=64, is_embed=torch.zeros(64, dtype=torch.bool))]
            },
        },
    ):
        with pytest.raises(VLLMValidationError, match="one image at the compiled placeholder"):
            processor.process(inputs)
    processor.process(valid)  # a rejected request does not affect subsequent validation


def test_dynamic_qwen3_vl_frontend_does_not_use_static_layout() -> None:
    config = _make_vllm_config(
        128,
        hf_config=SimpleNamespace(model_type="mobilint-qwen3_vl", dynamic_vision=True),
        loader_extra_config={"text_max_batch_size": 16},
    )
    _StaticInputProcessor(config).process({"type": "token"})


@pytest.mark.parametrize("dynamic_flag", [None, False, True])
def test_qwen3_vl_without_static_position_does_not_assume_static_layout(dynamic_flag) -> None:
    hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
    if dynamic_flag is not None:
        hf_config.dynamic_vision = dynamic_flag
    config = _make_vllm_config(128, hf_config=hf_config, loader_extra_config={"text_max_batch_size": 16})
    _StaticInputProcessor(config).process({"type": "token"})


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


def test_transformers_5_vision_outputs_unpack_to_embeddings_and_deepstack() -> None:
    # transformers 5's Qwen3-VL get_image_features returns this ModelOutput where 4.x returned
    # ((per-image embeddings...), [deepstack per layer]).
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

    first, second = torch.ones(2, 4), torch.full((3, 4), 2.0)
    deepstack = [torch.zeros(5, 4), torch.ones(5, 4)]
    output = BaseModelOutputWithDeepstackFeatures(pooler_output=(first, second), deepstack_features=deepstack)

    embeds = MbltWorker._normalize_multimodal_embeddings(output)
    assert embeds.shape == (5, 4) and embeds[2:].eq(2.0).all()
    assert MbltWorker._extract_deepstack_embeddings(output) == deepstack
    plain = BaseModelOutputWithDeepstackFeatures(pooler_output=(first,))
    assert MbltWorker._normalize_multimodal_embeddings(plain) is first
    assert MbltWorker._extract_deepstack_embeddings(plain) is None


@pytest.mark.parametrize(("world_size", "backend", "expected"), [(1, "mp", "uni"), (1, "uni", "uni"), (2, "mp", "mp")])
def test_single_worker_engines_run_in_process(world_size, backend, expected) -> None:
    # vLLM 0.32 picks the multiprocessing executor for data parallel engines.
    config = _make_vllm_config(64)
    config.parallel_config.world_size = world_size
    config.parallel_config.distributed_executor_backend = backend

    MbltPlatform.check_and_update_config(config)

    assert config.parallel_config.distributed_executor_backend == expected


def test_new_requests_keep_the_data_of_prefix_cached_multimodal_items() -> None:
    """vLLM 0.32 strips covered items for GPU workers; this worker re-embeds the whole prompt, so it keeps them."""
    import vllm.v1.core.sched.output as sched_output
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

    import vllm_mblt.mblt_worker  # noqa: F401

    feature = MultiModalFeatureSpec(
        data={"input_features": object()},
        modality="audio",
        identifier="clip",
        mm_position=PlaceholderRange(offset=1, length=4),
    )

    assert sched_output.strip_covered_mm_data([feature], num_computed_tokens=16) == [feature]
