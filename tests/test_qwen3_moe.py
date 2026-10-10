from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import ModelRegistry

from tests.test_mblt_platform_prefill import _make_vllm_config
from vllm_mblt import register_model
from vllm_mblt.mblt_platform import MbltPlatform
from vllm_mblt.mblt_worker import MbltWorker
from vllm_mblt.moe_runtime import MbltMoECacheModel, is_mixture_of_experts_model
from vllm_mblt.runtime_cache import MbltRuntimeCacheManager


class _FakeSharedMxq:
    def __init__(self, index: int, max_cache_size: int = 4096) -> None:
        self.index = index
        self.max_cache_size = max_cache_size
        self.loaded: list[tuple[object, int]] = []

    def get_input_buffer_info(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(max_cache_size=self.max_cache_size)]

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        return [f"layer{self.index}-slot{cache_id}".encode()]

    def load_cache_memory(self, blobs: object, cache_id: int = 0) -> None:
        self.loaded.append((blobs, cache_id))


class _FakeMoECache:
    def __init__(self) -> None:
        self.seq_length = 0

    def set_seq_length(self, seq_length: int) -> None:
        self.seq_length = seq_length

    def get_seq_length(self) -> int:
        return self.seq_length


class _FakeMoEModel:
    """Duck-types the transformers-mblt MoE surface the adapter relies on."""

    def __init__(self, num_layers: int = 3, vocab_size: int = 11, max_cache_sizes: tuple[int, ...] = ()) -> None:
        self.config = SimpleNamespace(vocab_size=vocab_size, model_type="mobilint-qwen3_moe")
        sizes = max_cache_sizes or (4096,) * num_layers
        self.shared = [_FakeSharedMxq(index, sizes[index]) for index in range(num_layers)]
        self.calls: list[dict[str, object]] = []
        self.built_caches: list[int] = []

    def get_shared_mxq_models(self) -> list[_FakeSharedMxq]:
        return self.shared

    def build_mobilint_cache(self, batch_size: int = 1) -> _FakeMoECache:
        self.built_caches.append(batch_size)
        return _FakeMoECache()

    def __call__(self, *, inputs_embeds, past_key_values, use_cache, logits_to_keep):
        self.calls.append(
            {
                "inputs_embeds": inputs_embeds,
                "cache_size": past_key_values.get_seq_length(),
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
            }
        )
        logits = torch.full((1, 1, self.config.vocab_size), float(inputs_embeds.shape[1]))
        return SimpleNamespace(logits=logits)

    def eval(self) -> None:
        return None

    def get_input_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace()


def test_register_model_registers_qwen3_moe_by_import_path() -> None:
    register_model()

    registered = ModelRegistry.models["MobilintQwen3MoEForCausalLM"]

    # Lazy: transformers-mblt is imported only when a MoE model is served.
    assert registered.module_name == "vllm_mblt.models.modeling_qwen3_moe"
    assert registered.class_name == "MobilintQwen3MoEForCausalLM"


def test_moe_model_detection_requires_shared_mxqs_and_cache_builder() -> None:
    assert is_mixture_of_experts_model(_FakeMoEModel())
    assert not is_mixture_of_experts_model(SimpleNamespace(get_cache_mxq_model=lambda: object()))


def test_moe_cache_model_rejects_dense_models() -> None:
    with pytest.raises(TypeError):
        MbltMoECacheModel(SimpleNamespace(get_cache_mxq_model=lambda: object()))


def test_moe_cache_model_infer_delegates_forward_at_the_requested_cache_size() -> None:
    model = _FakeMoEModel()
    cache_model = MbltMoECacheModel(model)
    embeds = np.ones((1, 5, 4), dtype=np.float32)

    outputs = cache_model.infer(embeds, outputs=[np.empty((1, 1, 11), dtype=np.float32)], cache_size=7)

    assert model.built_caches == [1]
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["cache_size"] == 7
    assert call["use_cache"] is True
    assert call["logits_to_keep"] == 1
    assert tuple(call["inputs_embeds"].shape) == (1, 5, 4)
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 1, 11)
    assert float(outputs[0][0, 0, 0]) == 5.0

    # The worker passes the [1, seq, hidden] array inside a list when the MXQ has extra inputs.
    cache_model.infer([np.ones((1, 1, 4), dtype=np.float32)], cache_size=12)
    assert model.calls[-1]["cache_size"] == 12
    assert tuple(model.calls[-1]["inputs_embeds"].shape) == (1, 1, 4)


def test_moe_cache_model_reports_last_token_logits() -> None:
    assert MbltMoECacheModel(_FakeMoEModel(vocab_size=17)).get_model_output_shape() == [(1, 1, 17)]


def test_moe_cache_model_snapshots_every_shared_mxq_in_layer_order() -> None:
    model = _FakeMoEModel(num_layers=3)
    cache_model = MbltMoECacheModel(model)

    blobs = cache_model.dump_cache_memory()
    assert blobs == [[b"layer0-slot0"], [b"layer1-slot0"], [b"layer2-slot0"]]

    cache_model.load_cache_memory(blobs)
    assert [shared.loaded for shared in model.shared] == [
        [([b"layer0-slot0"], 0)],
        [([b"layer1-slot0"], 0)],
        [([b"layer2-slot0"], 0)],
    ]


def test_moe_cache_model_rejects_snapshot_with_wrong_layer_count() -> None:
    cache_model = MbltMoECacheModel(_FakeMoEModel(num_layers=3))

    with pytest.raises(ValueError, match="2 layers"):
        cache_model.load_cache_memory([[b"a"], [b"b"]])


def _make_loading_worker(max_batch_size: int, max_seq_len: int = 4096) -> MbltWorker:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_seq_len = max_seq_len
    worker.rank = 0
    worker.local_rank = 0
    worker.model = None
    worker.cache_model = None
    worker._infer_output_buffers = None
    worker.max_batch_size = max_batch_size
    worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=max_batch_size, block_size=128)
    worker.load_config = SimpleNamespace(model_loader_extra_config={})
    worker.model_config = SimpleNamespace(
        model="mobilint/Qwen3-30B-A3B",
        hf_config=SimpleNamespace(model_type="mobilint-qwen3_moe"),
        model_kwargs={},
        hf_overrides={},
    )
    worker.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(model_loader_extra_config={}), model_config=worker.model_config
    )
    worker._calibrate_prefix_cache_prefill_costs = lambda: None
    return worker


def test_load_model_wraps_moe_model_in_single_slot_cache_model(monkeypatch) -> None:
    worker = _make_loading_worker(max_batch_size=1)
    fake_model = _FakeMoEModel()
    monkeypatch.setattr(
        "vllm_mblt.mblt_worker.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: fake_model
    )

    worker.load_model()

    assert isinstance(worker.cache_model, MbltMoECacheModel)
    assert worker.cache_model.model is fake_model
    assert worker.cache_backend is None
    assert worker._get_cache_models() == [worker.cache_model]


def test_load_model_refuses_batch_serving_for_moe(monkeypatch) -> None:
    worker = _make_loading_worker(max_batch_size=2)
    monkeypatch.setattr(
        "vllm_mblt.mblt_worker.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: _FakeMoEModel()
    )

    with pytest.raises(RuntimeError, match="non-batch"):
        worker.load_model()


def test_moe_cache_model_reports_the_smallest_shared_kv_capacity() -> None:
    model = _FakeMoEModel(num_layers=3, max_cache_sizes=(4096, 2048, 4096))

    assert MbltMoECacheModel(model).max_cache_size == 2048


def test_load_model_refuses_max_model_len_above_moe_kv_capacity(monkeypatch) -> None:
    worker = _make_loading_worker(max_batch_size=1, max_seq_len=8192)
    monkeypatch.setattr(
        "vllm_mblt.mblt_worker.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: _FakeMoEModel()
    )

    with pytest.raises(RuntimeError, match="--max-model-len 4096"):
        worker.load_model()


def test_load_model_forwards_moe_role_placement_kwargs(monkeypatch) -> None:
    worker = _make_loading_worker(max_batch_size=1)
    worker.load_config = SimpleNamespace(
        model_loader_extra_config={
            "shared_target_cores": ["0:0:0", "1:0:0"],
            "expert_target_cores": ["0:0:0", "0:0:1"],
            "lm_head_target_cores": ["1:0:0"],
            "num_expert_workers": 8,
            "unrelated": "ignored",
        }
    )
    calls = []

    def from_pretrained(*args, **kwargs):
        calls.append(kwargs)
        return _FakeMoEModel()

    monkeypatch.setattr("vllm_mblt.mblt_worker.AutoModelForCausalLM.from_pretrained", from_pretrained)

    worker.load_model()

    assert calls == [
        {
            "trust_remote_code": True,
            "shared_target_cores": ["0:0:0", "1:0:0"],
            "expert_target_cores": ["0:0:0", "0:0:1"],
            "lm_head_target_cores": ["1:0:0"],
            "num_expert_workers": 8,
        }
    ]


def test_platform_refuses_batch_configuration_for_moe() -> None:
    vllm_config = _make_vllm_config(
        {"single": 128},
        hf_model_type="mobilint-qwen3_moe",
        loader_extra_config={"max_batch_size": 4},
    )
    vllm_config.model_config.max_model_len = 4096

    with pytest.raises(ValueError, match="non-batch"):
        MbltPlatform.check_and_update_config(vllm_config)


def test_platform_clamps_moe_max_model_len_to_kv_capacity() -> None:
    vllm_config = _make_vllm_config({"single": 128}, hf_model_type="mobilint-qwen3_moe", max_batch_size=1)
    vllm_config.model_config.max_model_len = 40960

    MbltPlatform.check_and_update_config(vllm_config)

    assert vllm_config.model_config.max_model_len == 4096


def test_platform_keeps_smaller_moe_max_model_len() -> None:
    vllm_config = _make_vllm_config({"single": 128}, hf_model_type="mobilint-qwen3_moe", max_batch_size=1)
    vllm_config.model_config.max_model_len = 2048

    MbltPlatform.check_and_update_config(vllm_config)

    assert vllm_config.model_config.max_model_len == 2048


def test_platform_serves_moe_one_sequence_at_a_time() -> None:
    # MobilintMixtureOfExpertsConfigMixin.max_batch_size is fixed at 1.
    vllm_config = _make_vllm_config(
        {"single": 128},
        hf_model_type="mobilint-qwen3_moe",
        max_batch_size=1,
        scheduler_max_num_seqs=256,
    )
    vllm_config.model_config.max_model_len = 4096

    MbltPlatform.check_and_update_config(vllm_config)

    assert vllm_config.scheduler_config.max_num_seqs == 1
    assert vllm_config.scheduler_config.max_num_batched_tokens == 128
