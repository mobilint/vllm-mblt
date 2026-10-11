from types import SimpleNamespace

import pytest
from vllm import ModelRegistry

from tests.test_mblt_platform_prefill import _make_vllm_config
from vllm_mblt import register_model
from vllm_mblt.mblt_platform import MbltPlatform
from vllm_mblt.mblt_worker import MbltWorker
from vllm_mblt.moe_runtime import MbltMoECacheModel, is_mixture_of_experts_model
from vllm_mblt.request_cache import RequestCacheSlots
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
        self.built_caches: list[int] = []

    def get_shared_mxq_models(self) -> list[_FakeSharedMxq]:
        return self.shared

    def build_mobilint_cache(self, batch_size: int = 1) -> _FakeMoECache:
        self.built_caches.append(batch_size)
        return _FakeMoECache()

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


def test_moe_cache_model_reports_last_token_logits() -> None:
    assert MbltMoECacheModel(_FakeMoEModel(vocab_size=17)).get_model_output_shape() == [(1, 1, 17)]


def test_moe_snapshot_size_spans_every_shared_mxq() -> None:
    model = _FakeMoEModel(num_layers=3)
    for shared in model.shared:
        shared.get_cache_infos = lambda: [SimpleNamespace(size=4 * 2**20)]

    assert RequestCacheSlots([], moe=MbltMoECacheModel(model)).kv_nbytes_per_cache() == 3 * 4 * 2**20


def _make_loading_worker(max_batch_size: int, max_seq_len: int = 4096) -> MbltWorker:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_seq_len = max_seq_len
    worker.rank = 0
    worker.local_rank = 0
    worker.model = None
    worker.cache_model = None
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
