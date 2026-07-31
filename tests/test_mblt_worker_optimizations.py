from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logprobs import Logprob
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.logprobs import LogprobsProcessor, create_prompt_logprobs
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.sampler import Sampler

from vllm_mblt.mblt_worker import (
    InferenceLogits,
    MbltWorker,
    PrefixCacheCostModel,
    RequestState,
    _is_multimodal_hf_config,
    _is_qwen3_vl_hf_config,
)
from vllm_mblt.runtime_cache import MbltRuntimeCacheManager, RuntimeCacheRequest


class TestMbltWorkerOptimizations:
    def test_multimodal_hf_config_detects_mobilint_qwen_vl_model_types(self) -> None:
        assert _is_multimodal_hf_config(SimpleNamespace(model_type="mobilint-qwen2_vl"))
        assert _is_multimodal_hf_config(SimpleNamespace(model_type="mobilint-qwen3_vl"))
        assert not _is_multimodal_hf_config(SimpleNamespace(model_type="qwen2_vl"))
        assert not _is_multimodal_hf_config(SimpleNamespace(model_type="qwen3_vl"))
        assert not _is_multimodal_hf_config(SimpleNamespace(model_type="qwen2"))
        assert not _is_multimodal_hf_config(SimpleNamespace(model_type=None))
        assert not _is_multimodal_hf_config(SimpleNamespace())

    def test_multimodal_hf_config_ignores_architecture_heuristics(self) -> None:
        assert not _is_multimodal_hf_config(
            SimpleNamespace(model_type="some_vision_encoder", architectures=["SomeVisionModel"])
        )
        assert not _is_multimodal_hf_config(
            SimpleNamespace(model_type="custom_text_model", architectures=["FooVLForConditionalGeneration"])
        )
        assert not _is_multimodal_hf_config(
            SimpleNamespace(model_type="vision_with_text_tower", vision_config=SimpleNamespace())
        )

    def test_qwen3_vl_hf_config_detects_only_qwen3_vl(self) -> None:
        assert _is_qwen3_vl_hf_config(SimpleNamespace(model_type="mobilint-qwen3_vl"))
        assert not _is_qwen3_vl_hf_config(SimpleNamespace(model_type="qwen3_vl"))
        assert not _is_qwen3_vl_hf_config(SimpleNamespace(model_type="qwen2_vl"))
        assert not _is_qwen3_vl_hf_config(SimpleNamespace(architectures=["MobilintQwen3VLForConditionalGeneration"]))

    def test_multimodal_model_detection_uses_mobilint_model_type_only(self) -> None:
        worker = self._make_worker()
        assert not worker._is_multimodal_model()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        assert worker._is_multimodal_model()
        worker.model_config.hf_config = SimpleNamespace(model_type="qwen2_vl")
        worker.model.config = SimpleNamespace(model_type="mobilint-qwen3_vl", vocab_size=32000)
        assert worker._is_multimodal_model()

    def test_mobilint_vlm_request_constraints_allow_text_only(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        worker._validate_mobilint_vlm_request_constraints(None, session_id="session-a")
        worker._validate_mobilint_vlm_request_constraints([], session_id="session-a")
        assert worker._vlm_image_positions_by_session == {}

    def test_mobilint_vlm_request_constraints_allow_one_image_and_fix_position(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        feature = self._make_mm_feature("image", offset=4, length=2)
        worker._validate_mobilint_vlm_request_constraints([feature], session_id="session-a")
        assert worker._vlm_image_positions_by_session["session-a"] == (4, 2, None)

    def test_mobilint_vlm_request_constraints_reject_multiple_images(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        with pytest.raises(RuntimeError, match="exactly one image"):
            worker._validate_mobilint_vlm_request_constraints(
                [
                    self._make_mm_feature("image", offset=1, length=2),
                    self._make_mm_feature("image", offset=3, length=2),
                ],
                session_id="session-a",
            )

    def test_mobilint_vlm_request_constraints_reject_video(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        with pytest.raises(RuntimeError, match="does not support video"):
            worker._validate_mobilint_vlm_request_constraints(
                [self._make_mm_feature("video", offset=1, length=2)], session_id="session-a"
            )

    def test_mobilint_vlm_request_constraints_reject_image_position_change_in_same_session(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        worker._validate_mobilint_vlm_request_constraints(
            [self._make_mm_feature("image", offset=1, length=2)], session_id="session-a"
        )
        with pytest.raises(RuntimeError, match="fixed image-token position"):
            worker._validate_mobilint_vlm_request_constraints(
                [self._make_mm_feature("image", offset=2, length=2)], session_id="session-a"
            )

    def test_mobilint_vlm_request_constraints_allow_different_positions_across_sessions(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        worker._validate_mobilint_vlm_request_constraints(
            [self._make_mm_feature("image", offset=1, length=2)], session_id="session-a"
        )
        worker._validate_mobilint_vlm_request_constraints(
            [self._make_mm_feature("image", offset=4, length=3)], session_id="session-b"
        )
        assert worker._vlm_image_positions_by_session["session-a"] == (1, 2, None)
        assert worker._vlm_image_positions_by_session["session-b"] == (4, 3, None)

    def test_mobilint_vlm_request_constraints_include_is_embed_in_fixed_position(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        worker._validate_mobilint_vlm_request_constraints(
            [self._make_mm_feature("image", offset=1, length=3, is_embed=torch.tensor([True, False, True]))],
            session_id="session-a",
        )
        assert worker._vlm_image_positions_by_session["session-a"] == (1, 3, (True, False, True))

    def test_mobilint_vlm_request_constraints_are_noop_for_non_mobilint_models(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="qwen2_vl")
        worker._validate_mobilint_vlm_request_constraints(
            [self._make_mm_feature("video", offset=1, length=2), self._make_mm_feature("image", offset=3, length=2)],
            session_id="session-a",
        )
        assert getattr(worker, "_vlm_image_positions_by_session", {}) == {}

    def test_vlm_multimodal_cache_identity_includes_image_content_fingerprint(self) -> None:
        image_a = self._make_mm_feature(
            "image",
            offset=1,
            length=2,
            data={
                "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "image_grid_thw": torch.tensor([1, 1, 2]),
            },
        )
        image_b = self._make_mm_feature(
            "image",
            offset=1,
            length=2,
            data={
                "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
                "image_grid_thw": torch.tensor([1, 1, 2]),
            },
        )

        identity_a = MbltWorker._build_vlm_multimodal_cache_identity("session-a", [image_a])
        identity_b = MbltWorker._build_vlm_multimodal_cache_identity("session-a", [image_b])

        assert identity_a != identity_b

    def test_vlm_multimodal_cache_identity_is_stable_for_same_image_content(self) -> None:
        data = {
            "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "image_grid_thw": torch.tensor([1, 1, 2]),
        }

        identity_a = MbltWorker._build_vlm_multimodal_cache_identity(
            "session-a",
            [self._make_mm_feature("image", offset=1, length=2, data=data)],
        )
        identity_b = MbltWorker._build_vlm_multimodal_cache_identity(
            "session-a",
            [
                self._make_mm_feature(
                    "image",
                    offset=1,
                    length=2,
                    data={key: value.clone() for key, value in data.items()},
                )
            ],
        )

        assert identity_a == identity_b

    def test_vlm_multimodal_cache_identity_marks_unfingerprintable_feature_unresolved(self) -> None:
        identity = MbltWorker._build_vlm_multimodal_cache_identity(
            "session-a",
            [
                self._make_mm_feature(
                    "image",
                    offset=1,
                    length=2,
                    data={"pixel_values": object(), "image_grid_thw": torch.tensor([1, 1, 2])},
                )
            ],
        )

        assert identity == ("vlm", "session-a", (("image", (1, 2, None), None),))

    def _make_worker(self) -> MbltWorker:
        worker = MbltWorker.__new__(MbltWorker)
        worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen2"))
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            model_config=worker.model_config,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        )
        worker.model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
        worker.cache_model = None
        worker.max_batch_size = 1
        worker.empty_logits_processors = LogitsProcessors(None)
        worker.empty_prompt_token_ids = torch.empty((0, 0), dtype=torch.int64)
        worker.sampler = Sampler(logprobs_mode="raw_logits")
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
        worker.prefix_cache_cost_model = PrefixCacheCostModel(block_size=128)
        worker.input_embeddings = SimpleNamespace()
        worker.print_debug = False
        worker._warned_last_logit_prompt_logprobs = False
        worker._vlm_image_positions_by_session = {}
        return worker

    def _make_scheduler_output(self, num_scheduled_tokens: dict[str, int]) -> SimpleNamespace:
        return SimpleNamespace(
            finished_req_ids=[],
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                num_computed_tokens=[],
                num_output_tokens=[],
                new_block_ids=[],
                resumed_req_ids=set(),
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            kv_connector_metadata=None,
        )

    def _make_new_request(
        self,
        req_id: str,
        prompt_embeds: torch.Tensor,
        mm_features: list[SimpleNamespace],
        *,
        num_computed_tokens: int = 0,
        block_ids: tuple[list[int], ...] | None = None,
        block_hashes: tuple[object, ...] | None = None,
        session_id: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            req_id=req_id,
            sampling_params=SamplingParams.from_optional(temperature=0.0),
            prompt_token_ids=list(range(int(prompt_embeds.shape[0]))),
            prompt_embeds=prompt_embeds,
            mm_features=mm_features,
            block_ids=block_ids or ([11],),
            block_hashes=block_hashes,
            num_computed_tokens=num_computed_tokens,
            session_id=session_id,
        )

    def _make_request_state(
        self,
        worker: MbltWorker,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int],
        *,
        output_token_ids: list[int] | None = None,
    ) -> RequestState:
        return RequestState(
            is_prefill=False,
            output_token_ids=output_token_ids or [],
            sampling_params=sampling_params,
            cached_sampling_state=worker._make_cached_sampling_state(sampling_params, prompt_token_ids),
            block_ids=([],),
            first_seq_blocks=(),
            first_seq_block_hashes=None,
            num_computed_tokens=0,
            num_output_tokens=0,
            prompt_embeds=np.empty((0, 1), dtype=np.float32),
            prompt_deepstack_embeds=None,
            is_multimodal=False,
            prompt_len=0,
            prompt_token_ids=prompt_token_ids,
            cache_slot_id=None,
            vlm_session_id=None,
        )

    def _make_mm_feature(
        self,
        modality: str = "image",
        *,
        offset: int = 1,
        length: int = 2,
        is_embed: torch.Tensor | None = None,
        data: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            modality=modality,
            data=data or {},
            mm_position=SimpleNamespace(offset=offset, length=length, is_embed=is_embed),
        )

    def test_snapshot_index_can_prefer_shallower_prefix_with_more_tokens(self) -> None:
        worker = self._make_worker()
        worker.runtime_cache.store_snapshot(
            req_id="short_shared",
            blobs=[],
            block_ids=([1, 2, 8],),
            first_seq_blocks=(1, 2, 8),
            num_tokens=384,
        )
        short_shared = worker.runtime_cache.get_snapshot("short_shared")
        assert short_shared is not None
        worker.runtime_cache.store_snapshot(
            req_id="deep_but_short",
            blobs=[],
            block_ids=([1, 2, 3, 7],),
            first_seq_blocks=(1, 2, 3, 7),
            num_tokens=100,
        )
        req_state = SimpleNamespace(num_computed_tokens=300, first_seq_blocks=(1, 2, 3, 9))
        match = worker.runtime_cache.choose_snapshot(
            RuntimeCacheRequest(
                req_id="",
                block_ids=(),
                first_seq_blocks=req_state.first_seq_blocks,
                num_computed_tokens=req_state.num_computed_tokens,
            )
        )
        snapshot, matched_tokens = match
        assert snapshot is short_shared
        assert matched_tokens == 256

    def test_llm_prefix_cache_does_not_load_reused_physical_blocks_for_different_prompt_tokens(self) -> None:
        worker = self._make_worker()
        worker.runtime_cache.store_snapshot(
            req_id="old-content",
            blobs=["stale-runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            num_tokens=4,
            cache_token_ids=(101, 102, 103, 104),
        )
        load_calls = []
        worker.runtime_cache.set_io_adapters(
            load_runtime_cache=lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )
        req_state = self._make_request_state(
            worker,
            SamplingParams.from_optional(),
            [201, 202, 203, 204],
        )
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)
        req_state.num_computed_tokens = 4

        cache_size = worker._load_snapshot_if_needed("new-content", req_state)

        assert cache_size == 0
        assert load_calls == []
        assert worker.runtime_cache.loaded_req_id is None

    def test_short_prefix_hit_skips_load_and_keeps_snapshot(self) -> None:
        worker = self._make_worker()
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            first_seq_block_hashes=("shared-block",),
            num_tokens=16,
            cache_token_ids=tuple(range(16)),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model.observe_prefill(128, 20.0)
        worker.prefix_cache_cost_model.observe_load(None, 5.0)
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(32)))
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)
        req_state.first_seq_block_hashes = ("shared-block",)
        req_state.num_computed_tokens = 16

        cache_size = worker._load_snapshot_if_needed("request", req_state)

        assert cache_size == 0
        assert load_calls == []
        assert worker.runtime_cache.get_snapshot("shared") is not None
        assert worker.runtime_cache.loaded_req_id is None

    def test_long_prefix_hit_loads_and_returns_suffix_start(self) -> None:
        worker = self._make_worker()
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            first_seq_block_hashes=("shared-block",),
            num_tokens=128,
            cache_token_ids=tuple(range(128)),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model.observe_prefill(128, 20.0)
        worker.prefix_cache_cost_model.observe_load(None, 5.0)
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(160)))
        req_state.block_ids = ([10, 11],)
        req_state.first_seq_blocks = (10, 11)
        req_state.first_seq_block_hashes = ("shared-block", "suffix-block")
        req_state.num_computed_tokens = 128

        cache_size = worker._load_snapshot_if_needed("request", req_state)

        assert cache_size == 128
        assert load_calls == [(["runtime-cache"], None)]
        assert worker.runtime_cache.loaded_req_id == "request"

    def test_explicit_prompt_embeds_with_same_tokens_but_different_content_do_not_load(self) -> None:
        worker = self._make_worker()
        old_embeds = np.zeros((128, 4), dtype=np.float32)
        new_embeds = old_embeds.copy()
        new_embeds[17, 2] = 1.0
        old_req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(128)))
        old_req_state.prompt_embeds = old_embeds
        old_req_state.prompt_len = 128
        old_req_state.explicit_prompt_embeds = True
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            first_seq_block_hashes=("shared-block",),
            num_tokens=128,
            cache_token_ids=tuple(range(128)),
            prompt_embed_cache_identity=worker._make_prompt_embed_cache_identity(old_req_state),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model.observe_prefill(128, 20.0)
        worker.prefix_cache_cost_model.observe_load(None, 5.0)
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(128)))
        req_state.prompt_embeds = new_embeds
        req_state.prompt_len = 128
        req_state.explicit_prompt_embeds = True
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)
        req_state.first_seq_block_hashes = ("shared-block",)
        req_state.num_computed_tokens = 128

        cache_size = worker._load_snapshot_if_needed("request", req_state)

        assert cache_size == 0
        assert load_calls == []
        assert worker.runtime_cache.loaded_req_id is None

    def test_explicit_prompt_embeds_with_identical_content_load(self) -> None:
        worker = self._make_worker()
        prompt_embeds = np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
        old_req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(128)))
        old_req_state.prompt_embeds = prompt_embeds
        old_req_state.prompt_len = 128
        old_req_state.explicit_prompt_embeds = True
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            first_seq_block_hashes=("shared-block",),
            num_tokens=128,
            cache_token_ids=tuple(range(128)),
            prompt_embed_cache_identity=worker._make_prompt_embed_cache_identity(old_req_state),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model.observe_prefill(128, 20.0)
        worker.prefix_cache_cost_model.observe_load(None, 5.0)
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(128)))
        req_state.prompt_embeds = prompt_embeds.copy()
        req_state.prompt_len = 128
        req_state.explicit_prompt_embeds = True
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)
        req_state.first_seq_block_hashes = ("shared-block",)
        req_state.num_computed_tokens = 128

        cache_size = worker._load_snapshot_if_needed("request", req_state)

        assert cache_size == 128
        assert load_calls == [(["runtime-cache"], None)]

    def test_short_explicit_prompt_embed_hit_skips_fingerprint_when_cost_policy_skips(
        self,
        monkeypatch,
    ) -> None:
        worker = self._make_worker()
        calls = []

        def fingerprint(prompt_embeds, num_tokens):
            calls.append((prompt_embeds.shape, int(num_tokens)))
            return ("unexpected",)

        monkeypatch.setattr(worker, "_fingerprint_prompt_embed_prefix", fingerprint)
        old_req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(16)))
        old_req_state.prompt_embeds = np.zeros((16, 4), dtype=np.float32)
        old_req_state.prompt_len = 16
        old_req_state.explicit_prompt_embeds = True
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            first_seq_block_hashes=("shared-block",),
            num_tokens=16,
            cache_token_ids=tuple(range(16)),
            prompt_embed_cache_identity=worker._make_prompt_embed_cache_identity(old_req_state),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model.observe_prefill(128, 20.0)
        worker.prefix_cache_cost_model.observe_load(None, 5.0)
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(16)))
        req_state.prompt_embeds = np.zeros((16, 4), dtype=np.float32)
        req_state.prompt_len = 16
        req_state.explicit_prompt_embeds = True
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)
        req_state.first_seq_block_hashes = ("shared-block",)
        req_state.num_computed_tokens = 16

        cache_size = worker._load_snapshot_if_needed("request", req_state)

        assert cache_size == 0
        assert load_calls == []
        assert calls == []

    def test_explicit_prompt_embed_generated_suffix_reuses_after_prompt_boundary(self) -> None:
        worker = self._make_worker()
        prompt_embeds = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        old_req_state = self._make_request_state(
            worker,
            SamplingParams.from_optional(),
            [101, 102, 103, 104],
            output_token_ids=[201, 202],
        )
        old_req_state.prompt_embeds = prompt_embeds
        old_req_state.prompt_len = 4
        old_req_state.explicit_prompt_embeds = True
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10, 11],),
            first_seq_blocks=(10, 11),
            first_seq_block_hashes=("shared-a", "shared-b"),
            num_tokens=6,
            cache_token_ids=worker._cache_token_ids(old_req_state, 6),
            prompt_embed_cache_identity=worker._make_prompt_embed_cache_identity(old_req_state),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model = PrefixCacheCostModel(
            block_size=128,
            auto_threshold_enabled=False,
        )
        req_state = self._make_request_state(
            worker,
            SamplingParams.from_optional(),
            [301, 302, 303, 304],
            output_token_ids=[201, 202],
        )
        req_state.prompt_embeds = prompt_embeds.copy()
        req_state.prompt_len = 4
        req_state.explicit_prompt_embeds = True
        req_state.block_ids = ([10, 11],)
        req_state.first_seq_blocks = (10, 11)
        req_state.first_seq_block_hashes = ("shared-a", "shared-b")
        req_state.num_computed_tokens = 6

        cache_size = worker._load_snapshot_if_needed("request", req_state)

        assert cache_size == 6
        assert load_calls == [(["runtime-cache"], None)]

    def test_batch_prefix_cache_threshold_applies_per_cache_id(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=2, block_size=128)
        worker.prefix_cache_cost_model = PrefixCacheCostModel(block_size=128)
        worker.runtime_cache.store_snapshot(
            req_id="shared",
            blobs=["runtime-cache"],
            block_ids=([10],),
            first_seq_blocks=(10,),
            num_tokens=16,
            cache_token_ids=tuple(range(16)),
        )
        load_calls = []
        worker._load_runtime_cache = lambda blobs, slot_id=None: load_calls.append((blobs, slot_id)) or True
        worker.prefix_cache_cost_model.observe_prefill(128, 20.0)
        worker.prefix_cache_cost_model.observe_load(1, 5.0)
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), list(range(32)))
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)
        req_state.first_seq_block_hashes = ("shared-block",)
        req_state.num_computed_tokens = 16
        req_state.cache_slot_id = 1

        cache_size = worker._load_snapshot_if_needed("request", req_state, slot_id=1)

        assert cache_size == 0
        assert load_calls == []
        assert worker.runtime_cache.live_slot_owner(1) == "request"
        assert worker.runtime_cache.get_snapshot("shared") is not None

    def test_prefix_cache_manual_override_and_auto_disable(self, monkeypatch) -> None:
        cost_model = PrefixCacheCostModel(
            block_size=128,
            auto_threshold_enabled=True,
            manual_min_hit_tokens=64,
        )
        cost_model.observe_prefill(128, 20.0)
        cost_model.observe_load(None, 1.0)
        assert not cost_model.should_load(matched_tokens=16)
        assert cost_model.should_load(matched_tokens=128)

        cost_model = PrefixCacheCostModel(
            block_size=128,
            auto_threshold_enabled=False,
            manual_min_hit_tokens=None,
        )
        cost_model.observe_prefill(128, 1.0)
        cost_model.observe_load(None, 100.0)
        assert cost_model.should_load(matched_tokens=16)

        config_model = PrefixCacheCostModel.from_config(
            SimpleNamespace(
                load_config=SimpleNamespace(
                    model_loader_extra_config={
                        "prefix_cache_auto_threshold": "0",
                        "prefix_cache_min_hit_tokens": "32",
                        "prefix_cache_load_margin": "0.8",
                    }
                ),
                model_config=SimpleNamespace(model_kwargs={}, hf_overrides={}),
            ),
            block_size=128,
        )
        assert not config_model.auto_threshold_enabled
        assert config_model.manual_min_hit_tokens == 32
        assert config_model.margin == 0.8

        monkeypatch.setenv("VLLM_MBLT_PREFIX_CACHE_MIN_HIT_TOKENS", "96")
        env_model = PrefixCacheCostModel.from_config(SimpleNamespace(), block_size=128)
        assert env_model.manual_min_hit_tokens == 96

    def test_prefix_cache_calibration_uses_one_active_batch_cache_id(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 4
        worker.max_seq_len = 128
        worker.input_embeddings = torch.nn.Embedding(1, 4)
        calls = []

        def infer(inputs, *, params):
            calls.extend((int(param.sequence_length), int(param.cache_size), int(param.cache_id)) for param in params)
            total_tokens = sum(int(param.sequence_length) for param in params)
            return [np.zeros((1, total_tokens, 8), dtype=np.float32)]

        worker.cache_model = SimpleNamespace(infer=infer)
        worker.prefix_cache_cost_model = PrefixCacheCostModel(block_size=128)

        worker._calibrate_prefix_cache_prefill_costs()

        assert calls
        assert all(call == (128, 0, 1) for call in calls)
        assert worker.prefix_cache_cost_model.prefill_ms(128) is not None

    def test_llm_prefix_cache_dump_saves_prompt_and_generated_token_identity(self) -> None:
        worker = self._make_worker()
        worker.runtime_cache.set_io_adapters(dump_runtime_cache=lambda slot_id: ["runtime-cache"])
        req_state = self._make_request_state(
            worker,
            SamplingParams.from_optional(),
            [101, 102],
            output_token_ids=[201, 202],
        )
        req_state.block_ids = ([10],)
        req_state.first_seq_blocks = (10,)

        dumped = worker._dump_snapshot("req", req_state, next_num_tokens=4)

        assert dumped
        snapshot = worker.runtime_cache.get_snapshot("req")
        assert snapshot is not None
        assert snapshot.cache_token_ids == (101, 102, 201, 202)

    def test_sampling_metadata_reuses_request_generator_and_enables_penalties(self) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(seed=123, frequency_penalty=0.5, top_k=20)
        req_state = self._make_request_state(worker, sampling_params, [11, 12, 13], output_token_ids=[21, 22])
        metadata_first = worker._make_sampling_metadata([req_state])
        metadata_second = worker._make_sampling_metadata([req_state])
        assert not metadata_first.no_penalties
        assert metadata_first.generators[0] is metadata_second.generators[0]
        assert metadata_first.prompt_token_ids.tolist() == [[11, 12, 13]]
        assert metadata_first.top_k.tolist() == [20]
        assert metadata_first.frequency_penalties.tolist() == [0.5]

    def test_sampling_metadata_skips_prompt_tensor_when_penalties_disabled(self) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional()
        req_state = self._make_request_state(worker, sampling_params, [1, 2, 3])
        metadata = worker._make_sampling_metadata([req_state])
        assert metadata.no_penalties
        assert metadata.prompt_token_ids is None

    @pytest.mark.parametrize("prompt_token_ids", ([15339, 1917], [128000, 15339, 1917]))
    def test_prompt_logprobs_for_echo_start_after_first_prompt_token(self, prompt_token_ids: list[int]) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)

        vocab_size = max(prompt_token_ids + [42]) + 1
        sequence_logits = np.full((len(prompt_token_ids), vocab_size), -10.0, dtype=np.float32)
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            sequence_logits[prompt_pos - 1, token_id] = 10.0

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=sequence_logits,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        assert prompt_logprobs_tensors is not None
        assert prompt_logprobs_tensors.logprob_token_ids.shape[0] == len(prompt_token_ids) - 1
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=prompt_logprobs_tensors)
        )
        assert processor.prompt_logprobs is not None
        assert len(processor.prompt_logprobs) == len(prompt_token_ids)
        assert processor.prompt_logprobs[0] is None
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert token_id in processor.prompt_logprobs[prompt_pos]

        generated_token_id = 42
        generated_logprobs = {generated_token_id: Logprob(logprob=-0.5, rank=1, decoded_token=None)}
        completion_logprobs = OpenAIServingCompletion._create_completion_logprobs(
            OpenAIServingCompletion.__new__(OpenAIServingCompletion),
            token_ids=[*prompt_token_ids, generated_token_id],
            top_logprobs=[*processor.prompt_logprobs, generated_logprobs],
            num_output_top_logprobs=1,
            tokenizer=SimpleNamespace(decode=lambda token_id: f"token:{token_id}"),
            return_as_token_id=True,
        )
        assert completion_logprobs.token_logprobs[0] is None
        assert completion_logprobs.token_logprobs[1] is not None
        assert completion_logprobs.token_logprobs[-1] == -0.5

    @pytest.mark.parametrize("prompt_token_ids", ([15339, 1917], [128000, 15339, 1917]))
    def test_prompt_logprobs_include_actual_prompt_token_when_not_topk(
        self, prompt_token_ids: list[int]
    ) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)

        top_token_id = 42
        vocab_size = max(prompt_token_ids + [top_token_id]) + 1
        sequence_logits = np.full((len(prompt_token_ids), vocab_size), -10.0, dtype=np.float32)
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            sequence_logits[prompt_pos - 1, top_token_id] = 10.0
            sequence_logits[prompt_pos - 1, token_id] = -5.0

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=sequence_logits,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=prompt_logprobs_tensors)
        )

        assert processor.prompt_logprobs is not None
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert processor.prompt_logprobs[prompt_pos] is not None
            assert token_id in processor.prompt_logprobs[prompt_pos]
            assert top_token_id in processor.prompt_logprobs[prompt_pos]

        generated_token_id = 43
        generated_logprobs = {generated_token_id: Logprob(logprob=-0.5, rank=1, decoded_token=None)}
        completion_logprobs = OpenAIServingCompletion._create_completion_logprobs(
            OpenAIServingCompletion.__new__(OpenAIServingCompletion),
            token_ids=[*prompt_token_ids, generated_token_id],
            top_logprobs=[*processor.prompt_logprobs, generated_logprobs],
            num_output_top_logprobs=1,
            tokenizer=SimpleNamespace(decode=lambda token_id: f"token:{token_id}"),
            return_as_token_id=True,
        )
        assert completion_logprobs.token_logprobs[0] is None
        assert all(logprob is not None for logprob in completion_logprobs.token_logprobs[1:])

    @pytest.mark.parametrize("prompt_token_ids", ([64], [15339], [128000]))
    def test_echo_logprobs_one_token_prompt_serializer_keeps_first_prompt_logprob_null(
        self, prompt_token_ids: list[int]
    ) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=np.empty((0, max(prompt_token_ids) + 1), dtype=np.float32),
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        assert prompt_logprobs_tensors is not None

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=create_prompt_logprobs(),
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=prompt_logprobs_tensors)
        )
        assert processor.prompt_logprobs == [None]

        generated_token_id = 42
        generated_logprobs = {generated_token_id: Logprob(logprob=-0.25, rank=1, decoded_token=None)}
        completion_logprobs = OpenAIServingCompletion._create_completion_logprobs(
            OpenAIServingCompletion.__new__(OpenAIServingCompletion),
            token_ids=[*prompt_token_ids, generated_token_id],
            top_logprobs=[*processor.prompt_logprobs, generated_logprobs],
            num_output_top_logprobs=1,
            tokenizer=SimpleNamespace(decode=lambda token_id: f"token:{token_id}"),
            return_as_token_id=True,
        )
        assert completion_logprobs.token_logprobs == [None, -0.25]

    def test_prompt_logprobs_include_chunk_boundary_prompt_token(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103, 104]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        vocab_size = max(prompt_token_ids) + 1

        first_chunk_logits = np.full((2, vocab_size), -10.0, dtype=np.float32)
        first_chunk_logits[0, prompt_token_ids[1]] = 10.0
        first_chunk_logits[1, prompt_token_ids[2]] = 10.0

        first_chunk_prompt_logprobs = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=first_chunk_logits,
            start_idx=0,
            scheduled_end=2,
        )

        second_chunk_logits = np.full((2, vocab_size), -10.0, dtype=np.float32)
        second_chunk_logits[0, prompt_token_ids[3]] = 10.0

        second_chunk_prompt_logprobs = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=second_chunk_logits,
            start_idx=2,
            scheduled_end=len(prompt_token_ids),
        )

        assert first_chunk_prompt_logprobs is not None
        assert second_chunk_prompt_logprobs is not None
        assert first_chunk_prompt_logprobs.logprob_token_ids.shape[0] == 2
        assert second_chunk_prompt_logprobs.logprob_token_ids.shape[0] == 1
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=first_chunk_prompt_logprobs)
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=second_chunk_prompt_logprobs)
        )

        assert processor.prompt_logprobs is not None
        assert len(processor.prompt_logprobs) == len(prompt_token_ids)
        assert processor.prompt_logprobs[0] is None
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert processor.prompt_logprobs[prompt_pos] is not None
            assert token_id in processor.prompt_logprobs[prompt_pos]

    def test_prompt_logprobs_are_buffered_until_prefill_completion_for_scheduler(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103, 104]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        vocab_size = max(prompt_token_ids) + 1

        first_chunk_logits = np.full((2, vocab_size), -10.0, dtype=np.float32)
        first_chunk_logits[0, prompt_token_ids[1]] = 10.0
        first_chunk_logits[1, prompt_token_ids[2]] = 10.0

        first_scheduler_prompt_logprobs = worker._get_completed_prompt_logprobs_tensors_for_scheduler(
            req_state=req_state,
            sequence_logits=first_chunk_logits,
            start_idx=0,
            scheduled_end=2,
        )

        assert first_scheduler_prompt_logprobs is None
        assert req_state.in_progress_prompt_logprobs is not None
        assert req_state.next_prompt_logprob_pos == 3

        second_chunk_logits = np.full((2, vocab_size), -10.0, dtype=np.float32)
        second_chunk_logits[0, prompt_token_ids[3]] = 10.0

        completed_prompt_logprobs = worker._get_completed_prompt_logprobs_tensors_for_scheduler(
            req_state=req_state,
            sequence_logits=second_chunk_logits,
            start_idx=2,
            scheduled_end=len(prompt_token_ids),
        )

        assert completed_prompt_logprobs is not None
        assert completed_prompt_logprobs.logprob_token_ids.shape[0] == len(prompt_token_ids) - 1
        assert req_state.in_progress_prompt_logprobs is None
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=completed_prompt_logprobs)
        )

        assert processor.prompt_logprobs is not None
        assert len(processor.prompt_logprobs) == len(prompt_token_ids)
        assert processor.prompt_logprobs[0] is None
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert processor.prompt_logprobs[prompt_pos] is not None
            assert token_id in processor.prompt_logprobs[prompt_pos]

    def test_completed_prompt_logprobs_wait_for_scheduler_emitted_output(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103, 104]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        vocab_size = max(prompt_token_ids) + 1

        logits = np.full((len(prompt_token_ids) - 1, vocab_size), -10.0, dtype=np.float32)
        for row, token_id in enumerate(prompt_token_ids[1:]):
            logits[row, token_id] = 10.0

        non_emitting_step_prompt_logprobs = worker._get_completed_prompt_logprobs_tensors_for_scheduler(
            req_state=req_state,
            sequence_logits=logits,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
            can_emit_output=False,
        )

        assert non_emitting_step_prompt_logprobs is None
        assert req_state.in_progress_prompt_logprobs is not None
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

        emitting_step_prompt_logprobs = worker._get_completed_prompt_logprobs_tensors_for_scheduler(
            req_state=req_state,
            sequence_logits=None,
            start_idx=len(prompt_token_ids),
            scheduled_end=len(prompt_token_ids) + 1,
            can_emit_output=True,
        )

        assert emitting_step_prompt_logprobs is not None
        assert emitting_step_prompt_logprobs.logprob_token_ids.shape[0] == len(prompt_token_ids) - 1
        assert req_state.in_progress_prompt_logprobs is None

    def test_prompt_logprobs_replay_does_not_duplicate_emitted_positions(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103, 104]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        vocab_size = max(prompt_token_ids) + 1

        first_chunk_logits = np.full((2, vocab_size), -10.0, dtype=np.float32)
        first_chunk_logits[0, prompt_token_ids[1]] = 10.0
        first_chunk_logits[1, prompt_token_ids[2]] = 10.0
        first_chunk_prompt_logprobs = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=first_chunk_logits,
            start_idx=0,
            scheduled_end=2,
        )
        assert first_chunk_prompt_logprobs is not None
        assert first_chunk_prompt_logprobs.logprob_token_ids.shape[0] == 2

        replay_logits = np.full((3, vocab_size), -10.0, dtype=np.float32)
        replay_logits[0, prompt_token_ids[1]] = 10.0
        replay_logits[1, prompt_token_ids[2]] = 10.0
        replay_logits[2, prompt_token_ids[3]] = 10.0
        replay_prompt_logprobs = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=replay_logits,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert replay_prompt_logprobs is not None
        assert replay_prompt_logprobs.logprob_token_ids.shape[0] == 1
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=first_chunk_prompt_logprobs)
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=replay_prompt_logprobs)
        )

        assert processor.prompt_logprobs is not None
        assert len(processor.prompt_logprobs) == len(prompt_token_ids)
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert processor.prompt_logprobs[prompt_pos] is not None
            assert token_id in processor.prompt_logprobs[prompt_pos]

    def test_prompt_logprobs_prefix_cache_hit_recomputes_from_start_for_contiguous_positions(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103, 104]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.num_computed_tokens = 2

        cache_size = worker._load_snapshot_if_needed("req", req_state)
        assert cache_size == 0

        vocab_size = max(prompt_token_ids) + 1
        sequence_logits = np.full((len(prompt_token_ids), vocab_size), -10.0, dtype=np.float32)
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            sequence_logits[prompt_pos - 1, token_id] = 10.0

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=sequence_logits,
            start_idx=cache_size,
            scheduled_end=len(prompt_token_ids),
        )

        assert prompt_logprobs_tensors is not None
        assert prompt_logprobs_tensors.logprob_token_ids.shape[0] == len(prompt_token_ids) - 1
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)
        assert not worker._should_recompute_prompt_logprobs_from_start(req_state)

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=prompt_logprobs_tensors)
        )

        assert processor.prompt_logprobs is not None
        assert len(processor.prompt_logprobs) == len(prompt_token_ids)
        assert processor.prompt_logprobs[0] is None
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert processor.prompt_logprobs[prompt_pos] is not None
            assert token_id in processor.prompt_logprobs[prompt_pos]

    @pytest.mark.parametrize("prompt_token_ids", ([], [101]))
    def test_prompt_logprobs_terminal_for_empty_or_one_token_prompt(self, prompt_token_ids: list[int]) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=np.empty((0, 1), dtype=np.float32),
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        assert prompt_logprobs_tensors is not None
        assert prompt_logprobs_tensors.logprob_token_ids.shape[0] == 0

        req_state.num_computed_tokens = 1
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)
        assert not worker._should_recompute_prompt_logprobs_from_start(req_state)

    def test_prompt_logprobs_unavailable_logits_terminal_during_decode(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.num_computed_tokens = 2

        assert worker._should_recompute_prompt_logprobs_from_start(req_state)
        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=None,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert prompt_logprobs_tensors is None
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)
        assert not worker._should_recompute_prompt_logprobs_from_start(req_state)

    @pytest.mark.parametrize("prompt_token_ids", ([15339, 1917], [128000, 15339, 1917]))
    def test_prompt_logprobs_fallback_for_last_token_only_runtime(self, prompt_token_ids: list[int]) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        vocab_size = max(prompt_token_ids + [42]) + 1
        calls: list[tuple[int, int]] = []

        def infer_logits(input_embeds, deepstack_embeds, cache_size):
            assert deepstack_embeds is None
            calls.append((int(input_embeds.shape[0]), int(cache_size)))
            prompt_pos = int(cache_size) + int(input_embeds.shape[0])
            logits = np.full((1, vocab_size), -10.0, dtype=np.float32)
            logits[0, prompt_token_ids[prompt_pos]] = 10.0
            return logits

        worker._infer_logits = infer_logits

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors_with_fallback(
            req_state=req_state,
            sequence_logits=None,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert prompt_logprobs_tensors is not None
        assert prompt_logprobs_tensors.logprob_token_ids.shape[0] == len(prompt_token_ids) - 1
        assert calls == [(1, prompt_pos - 1) for prompt_pos in range(1, len(prompt_token_ids))]
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

        processor = LogprobsProcessor(
            tokenizer=None,
            logprobs=[],
            prompt_logprobs=[None],
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=1,
        )
        processor.update_from_output(
            SimpleNamespace(new_logprobs=None, new_prompt_logprobs_tensors=prompt_logprobs_tensors)
        )
        assert processor.prompt_logprobs is not None
        assert len(processor.prompt_logprobs) == len(prompt_token_ids)
        assert processor.prompt_logprobs[0] is None
        for prompt_pos, token_id in enumerate(prompt_token_ids[1:], start=1):
            assert processor.prompt_logprobs[prompt_pos] is not None
            assert token_id in processor.prompt_logprobs[prompt_pos]

        generated_token_id = 42
        generated_logprobs = {generated_token_id: Logprob(logprob=-0.25, rank=1, decoded_token=None)}
        completion_logprobs = OpenAIServingCompletion._create_completion_logprobs(
            OpenAIServingCompletion.__new__(OpenAIServingCompletion),
            token_ids=[*prompt_token_ids, generated_token_id],
            top_logprobs=[*processor.prompt_logprobs, generated_logprobs],
            num_output_top_logprobs=1,
            tokenizer=SimpleNamespace(decode=lambda token_id: f"token:{token_id}"),
            return_as_token_id=True,
        )
        assert completion_logprobs.token_logprobs[0] is None
        assert all(logprob is not None for logprob in completion_logprobs.token_logprobs[1:])
        assert completion_logprobs.token_logprobs[-1] == -0.25

    @pytest.mark.parametrize("prompt_token_ids", ([], [15339]))
    def test_prompt_logprobs_fallback_returns_none_for_empty_or_one_token_prompt(
        self, prompt_token_ids: list[int]
    ) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        def infer_logits(input_embeds, deepstack_embeds, cache_size):
            raise AssertionError("short prompts have no prompt positions that require fallback replay")

        worker._infer_logits = infer_logits

        fallback_logits = worker._compute_prompt_logprobs_sequence_logits_fallback(
            req_state=req_state,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors_with_fallback(
            req_state=req_state,
            sequence_logits=None,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert fallback_logits is None
        assert prompt_logprobs_tensors is None

    def test_prompt_logprobs_fallback_returns_none_after_all_prompt_positions_emitted(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.next_prompt_logprob_pos = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        def infer_logits(input_embeds, deepstack_embeds, cache_size):
            raise AssertionError("completed prompt logprobs should not trigger fallback replay")

        worker._infer_logits = infer_logits

        fallback_logits = worker._compute_prompt_logprobs_sequence_logits_fallback(
            req_state=req_state,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors_with_fallback(
            req_state=req_state,
            sequence_logits=None,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert fallback_logits is None
        assert prompt_logprobs_tensors is None

    def test_prompt_logprobs_fallback_preserves_multitoken_rows_for_needed_positions(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        vocab_size = max(prompt_token_ids) + 1
        calls: list[tuple[int, int]] = []

        def infer_logits(input_embeds, deepstack_embeds, cache_size):
            assert deepstack_embeds is None
            calls.append((int(input_embeds.shape[0]), int(cache_size)))
            prompt_pos = int(cache_size) + int(input_embeds.shape[0])
            logits = np.full((1, vocab_size), -10.0, dtype=np.float32)
            logits[0, prompt_token_ids[prompt_pos]] = 10.0
            return logits

        worker._infer_logits = infer_logits

        fallback_logits = worker._compute_prompt_logprobs_sequence_logits_fallback(
            req_state=req_state,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        assert fallback_logits is not None
        assert fallback_logits.shape == (len(prompt_token_ids) - 1, vocab_size)
        assert calls == [(1, prompt_pos - 1) for prompt_pos in range(1, len(prompt_token_ids))]

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=fallback_logits,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )
        assert prompt_logprobs_tensors is not None
        assert prompt_logprobs_tensors.logprob_token_ids.shape[0] == len(prompt_token_ids) - 1
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)

    def test_prompt_logprobs_fallback_replay_work_is_linear_for_long_prompt(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = list(range(101, 133))
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        vocab_size = max(prompt_token_ids) + 1
        calls: list[tuple[int, int]] = []

        def infer_logits(input_embeds, deepstack_embeds, cache_size):
            assert deepstack_embeds is None
            submitted_tokens = int(input_embeds.shape[0])
            cache_size = int(cache_size)
            calls.append((submitted_tokens, cache_size))
            prompt_pos = cache_size + submitted_tokens
            logits = np.full((1, vocab_size), -10.0, dtype=np.float32)
            logits[0, prompt_token_ids[prompt_pos]] = 10.0
            return logits

        worker._infer_logits = infer_logits

        fallback_logits = worker._compute_prompt_logprobs_sequence_logits_fallback(
            req_state=req_state,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert fallback_logits is not None
        assert fallback_logits.shape == (len(prompt_token_ids) - 1, vocab_size)
        assert calls == [(1, prompt_pos - 1) for prompt_pos in range(1, len(prompt_token_ids))]
        assert sum(submitted_tokens for submitted_tokens, _cache_size in calls) == len(prompt_token_ids) - 1
        assert sum(submitted_tokens for submitted_tokens, _cache_size in calls) < len(prompt_token_ids) * 2
        assert sum(submitted_tokens for submitted_tokens, _cache_size in calls) != sum(
            range(1, len(prompt_token_ids))
        )

    def test_prompt_logprobs_fallback_uses_incremental_batch_cache_slot(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103, 104]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        vocab_size = max(prompt_token_ids) + 1
        calls: list[tuple[int, int, int]] = []

        def infer_logits_batch(input_embeds_batch, cache_sizes, cache_ids):
            assert len(input_embeds_batch) == len(cache_sizes) == len(cache_ids) == 1
            submitted_tokens = int(input_embeds_batch[0].shape[0])
            cache_size = int(cache_sizes[0])
            cache_id = int(cache_ids[0])
            calls.append((submitted_tokens, cache_size, cache_id))
            prompt_pos = cache_size + submitted_tokens
            logits = np.full((vocab_size,), -10.0, dtype=np.float32)
            logits[prompt_token_ids[prompt_pos]] = 10.0
            return [logits]

        worker._infer_logits_batch = infer_logits_batch

        fallback_logits = worker._compute_prompt_logprobs_sequence_logits_fallback(
            req_state=req_state,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
            cache_id=7,
        )

        assert fallback_logits is not None
        assert fallback_logits.shape == (len(prompt_token_ids) - 1, vocab_size)
        assert calls == [(1, prompt_pos - 1, 7) for prompt_pos in range(1, len(prompt_token_ids))]

    def test_prompt_logprobs_fallback_batches_replay_across_requests(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        prompt_token_ids_by_cache_id = {
            11: [101, 102, 103, 104],
            12: [201, 202, 203],
            13: [301, 302, 303, 304],
        }
        req_states: list[RequestState] = []
        for prompt_token_ids in prompt_token_ids_by_cache_id.values():
            req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
            req_state.prompt_len = len(prompt_token_ids)
            req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)
            req_states.append(req_state)

        vocab_size = 400
        calls: list[tuple[tuple[int, int, int], ...]] = []

        def infer_logits_batch(input_embeds_batch, cache_sizes, cache_ids):
            calls.append(
                tuple(
                    (int(input_embeds.shape[0]), int(cache_size), int(cache_id))
                    for input_embeds, cache_size, cache_id in zip(input_embeds_batch, cache_sizes, cache_ids)
                )
            )
            logits_batch = []
            for input_embeds, cache_size, cache_id in zip(input_embeds_batch, cache_sizes, cache_ids):
                prompt_pos = int(cache_size) + int(input_embeds.shape[0])
                logits = np.full((vocab_size,), -10.0, dtype=np.float32)
                logits[prompt_token_ids_by_cache_id[int(cache_id)][prompt_pos]] = 10.0
                logits_batch.append(logits)
            return logits_batch

        worker._infer_logits_batch = infer_logits_batch

        fallback_logits = worker._compute_prompt_logprobs_sequence_logits_fallback_batch(
            [
                (0, req_states[0], 0, req_states[0].prompt_len, 11),
                (1, req_states[1], 0, req_states[1].prompt_len, 12),
                (2, req_states[2], 0, req_states[2].prompt_len, 13),
            ]
        )

        assert set(fallback_logits) == {0, 1, 2}
        assert fallback_logits[0].shape == (3, vocab_size)
        assert fallback_logits[1].shape == (2, vocab_size)
        assert fallback_logits[2].shape == (3, vocab_size)
        assert calls == [
            ((1, 0, 11), (1, 0, 12)),
            ((1, 0, 13),),
            ((1, 1, 11), (1, 1, 12)),
            ((1, 1, 13),),
            ((1, 2, 11), (1, 2, 13)),
        ]

        for output_index, req_state in enumerate(req_states):
            prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
                req_state=req_state,
                sequence_logits=fallback_logits[output_index],
                start_idx=0,
                scheduled_end=req_state.prompt_len,
            )
            assert prompt_logprobs_tensors is not None
            assert prompt_logprobs_tensors.logprob_token_ids.shape[0] == len(req_state.prompt_token_ids) - 1
            assert req_state.next_prompt_logprob_pos == len(req_state.prompt_token_ids)

    def test_prompt_logprob_microsteps_do_not_submit_long_batch_param(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        prompt_token_ids = list(range(300))
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)

        calls: list[tuple[int, int, int]] = []
        vocab_size = 320

        def infer(_inputs, *, params):
            calls.extend(
                (int(param.sequence_length), int(param.cache_size), int(param.cache_id))
                for param in params
            )
            return [np.zeros((len(params), vocab_size), dtype=np.float32)]

        worker.cache_model = SimpleNamespace(
            infer=infer,
            get_model_output_shape=lambda: [(2, vocab_size)],
        )

        outputs = worker._run_prompt_logprob_microsteps_batch(
            [(0, "req", req_state, 256, 258, 7)]
        )

        assert set(outputs) == {0}
        assert calls == [(1, 256, 7), (1, 257, 7)]
        assert all(sequence_length == 1 for sequence_length, _cache_size, _cache_id in calls)

    def test_prompt_logprob_microsteps_batch_requests_up_to_max_batch_size(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_states: list[RequestState] = []
        for base in (10, 20, 30):
            prompt_token_ids = [base, base + 1, base + 2]
            req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
            req_state.prompt_len = len(prompt_token_ids)
            req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)
            req_states.append(req_state)

        calls: list[tuple[tuple[int, int, int], ...]] = []
        vocab_size = 64

        def infer(_inputs, *, params):
            calls.append(
                tuple(
                    (int(param.sequence_length), int(param.cache_size), int(param.cache_id))
                    for param in params
                )
            )
            return [np.zeros((len(params), vocab_size), dtype=np.float32)]

        worker.cache_model = SimpleNamespace(
            infer=infer,
            get_model_output_shape=lambda: [(2, vocab_size)],
        )

        outputs = worker._run_prompt_logprob_microsteps_batch(
            [
                (0, "req-a", req_states[0], 0, 2, 11),
                (1, "req-b", req_states[1], 0, 2, 12),
                (2, "req-c", req_states[2], 0, 2, 13),
            ]
        )

        assert set(outputs) == {0, 1, 2}
        assert calls == [
            ((1, 0, 11), (1, 0, 12)),
            ((1, 0, 13),),
            ((1, 1, 11), (1, 1, 12)),
            ((1, 1, 13),),
        ]

    def test_normal_long_prefill_is_submitted_in_per_request_chunks(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 4
        worker.vllm_config.scheduler_config.long_prefill_token_threshold = 128
        worker.req_states = {}
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=4, block_size=128)
        worker.print_debug = False
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(4, 16)])

        sampling_params = SamplingParams.from_optional(temperature=0.0)
        req_state = self._make_request_state(worker, sampling_params, list(range(300)))
        req_state.prompt_len = 300
        req_state.prompt_embeds = np.ones((300, 4), dtype=np.float32)
        req_state.cache_slot_id = 0
        worker.req_states = {"long": req_state}

        calls: list[tuple[tuple[int, int, int], ...]] = []

        def infer_chunk(input_embeds_batch, cache_sizes, cache_ids):
            calls.append(
                tuple(
                    (int(input_embeds.shape[0]), int(cache_size), int(cache_id))
                    for input_embeds, cache_size, cache_id in zip(input_embeds_batch, cache_sizes, cache_ids)
                )
            )
            return [
                InferenceLogits(
                    last_token_logits=np.full(16, int(cache_size) + int(input_embeds.shape[0]), dtype=np.float32),
                    full_sequence_logits=np.zeros((int(input_embeds.shape[0]), 16), dtype=np.float32),
                )
                for input_embeds, cache_size in zip(input_embeds_batch, cache_sizes)
            ]

        worker._load_snapshot_if_needed = lambda _req_id, req_state, **_kwargs: int(req_state.num_computed_tokens)
        worker._infer_logits_batch_with_sequence = infer_chunk
        worker._make_sampling_metadata = lambda _states: None
        worker._sample_next_token = lambda _logits, _metadata: SimpleNamespace(
            sampled_token_ids=torch.tensor([[9]], dtype=torch.int64),
            logprobs_tensors=None,
        )

        output = worker.execute_model(self._make_scheduler_output({"long": 300}))

        assert output is not None
        assert calls == [((128, 0, 0),), ((128, 128, 0),), ((44, 256, 0),)]
        assert max(sequence_length for call in calls for sequence_length, _cache_size, _cache_id in call) <= 128
        assert req_state.num_computed_tokens == 300
        assert output.sampled_token_ids[output.req_id_to_index["long"]].tolist() == [9]

    def test_multiple_normal_long_prefills_are_grouped_per_chunk(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        worker.vllm_config.scheduler_config.long_prefill_token_threshold = 128
        worker.req_states = {}
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=2, block_size=128)
        worker.print_debug = False
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(2, 16)])

        sampling_params = SamplingParams.from_optional(temperature=0.0)
        req_a_state = self._make_request_state(worker, sampling_params, list(range(260)))
        req_b_state = self._make_request_state(worker, sampling_params, list(range(1000, 1260)))
        for slot_id, req_state in enumerate((req_a_state, req_b_state)):
            req_state.prompt_len = 260
            req_state.prompt_embeds = np.ones((260, 4), dtype=np.float32)
            req_state.cache_slot_id = slot_id
        worker.req_states = {"req-a": req_a_state, "req-b": req_b_state}

        calls: list[tuple[tuple[int, int, int], ...]] = []

        def infer_chunk(input_embeds_batch, cache_sizes, cache_ids):
            calls.append(
                tuple(
                    (int(input_embeds.shape[0]), int(cache_size), int(cache_id))
                    for input_embeds, cache_size, cache_id in zip(input_embeds_batch, cache_sizes, cache_ids)
                )
            )
            return [
                InferenceLogits(
                    last_token_logits=np.full(16, int(cache_id), dtype=np.float32),
                    full_sequence_logits=None,
                )
                for cache_id in cache_ids
            ]

        worker._load_snapshot_if_needed = lambda _req_id, req_state, **_kwargs: int(req_state.num_computed_tokens)
        worker._infer_logits_batch_with_sequence = infer_chunk
        worker._make_sampling_metadata = lambda _states: None
        worker._sample_next_token = lambda _logits, _metadata: SimpleNamespace(
            sampled_token_ids=torch.tensor([[7], [8]], dtype=torch.int64),
            logprobs_tensors=None,
        )

        output = worker.execute_model(self._make_scheduler_output({"req-a": 260, "req-b": 260}))

        assert output is not None
        assert calls == [
            ((128, 0, 0), (128, 0, 1)),
            ((128, 128, 0), (128, 128, 1)),
            ((4, 256, 0), (4, 256, 1)),
        ]
        assert req_a_state.num_computed_tokens == 260
        assert req_b_state.num_computed_tokens == 260
        assert output.sampled_token_ids[output.req_id_to_index["req-a"]].tolist() == [7]
        assert output.sampled_token_ids[output.req_id_to_index["req-b"]].tolist() == [8]

    def test_mixed_batch_keeps_normal_chunks_and_microsteps_prompt_logprobs_in_order(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        worker.vllm_config.scheduler_config.long_prefill_token_threshold = 128
        worker.req_states = {}
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=2, block_size=128)
        worker.print_debug = False
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(2, 16)])

        normal_params = SamplingParams.from_optional(temperature=0.0)
        prompt_params = SamplingParams.from_optional(temperature=0.0, prompt_logprobs=1)
        normal_state = self._make_request_state(worker, normal_params, list(range(260)))
        prompt_state = self._make_request_state(worker, prompt_params, [4, 5, 6])
        normal_state.prompt_len = 260
        normal_state.prompt_embeds = np.ones((260, 4), dtype=np.float32)
        prompt_state.prompt_len = 3
        prompt_state.prompt_embeds = np.ones((3, 4), dtype=np.float32)
        for slot_id, req_state in enumerate((normal_state, prompt_state)):
            req_state.cache_slot_id = slot_id
        worker.req_states = {"normal": normal_state, "prompt": prompt_state}

        chunk_calls: list[tuple[int, int, int]] = []
        micro_calls: list[tuple[tuple[int, int, int], ...]] = []

        def load_snapshot(_req_id, req_state, **_kwargs):
            return int(req_state.num_computed_tokens)

        def infer_chunk(input_embeds_batch, cache_sizes, cache_ids):
            chunk_calls.extend(
                (int(input_embeds.shape[0]), int(cache_size), int(cache_id))
                for input_embeds, cache_size, cache_id in zip(input_embeds_batch, cache_sizes, cache_ids)
            )
            return [
                InferenceLogits(last_token_logits=np.zeros(16, dtype=np.float32), full_sequence_logits=None)
                for _input_embeds in input_embeds_batch
            ]

        def infer_micro(input_embeds_batch, cache_sizes, cache_ids):
            micro_calls.append(
                tuple(
                    (int(input_embeds.shape[0]), int(cache_size), int(cache_id))
                    for input_embeds, cache_size, cache_id in zip(input_embeds_batch, cache_sizes, cache_ids)
                )
            )
            return [np.zeros(16, dtype=np.float32) for _ in input_embeds_batch]

        worker._load_snapshot_if_needed = load_snapshot
        worker._infer_logits_batch_with_sequence = infer_chunk
        worker._infer_logits_batch = infer_micro
        worker._make_sampling_metadata = lambda _states: None
        worker._sample_next_token = lambda _logits, _metadata: SimpleNamespace(
            sampled_token_ids=torch.tensor([[7], [8]], dtype=torch.int64),
            logprobs_tensors=None,
        )

        output = worker.execute_model(
            SimpleNamespace(
                finished_req_ids=[],
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                    new_block_ids=[],
                    resumed_req_ids=set(),
                ),
                num_scheduled_tokens={"normal": 260, "prompt": 3},
                kv_connector_metadata=None,
            )
        )

        assert output is not None
        assert chunk_calls == [(128, 0, 0), (128, 128, 0), (4, 256, 0)]
        assert micro_calls == [((1, 0, 1),), ((1, 1, 1),), ((1, 2, 1),)]
        assert normal_state.num_computed_tokens == 260
        assert prompt_state.num_computed_tokens == 3
        assert "prompt" in output.prompt_logprobs_dict
        assert output.req_ids == ["normal", "prompt"]
        assert output.sampled_token_ids[output.req_id_to_index["normal"]].tolist() == [7]
        assert output.sampled_token_ids[output.req_id_to_index["prompt"]].tolist() == [8]

    def test_mixed_batch_logprobs_align_with_full_req_ids(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 3
        worker.req_states = {}
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=3, block_size=128)
        worker.print_debug = False
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(3, 16)])

        sampling_params = SamplingParams.from_optional(temperature=0.0, logprobs=1)
        prefill_state = self._make_request_state(worker, sampling_params, [1, 2, 3, 4, 5])
        sample_a_state = self._make_request_state(worker, sampling_params, [6, 7, 8])
        sample_b_state = self._make_request_state(worker, sampling_params, [9, 10, 11])
        for slot_id, req_state in enumerate((prefill_state, sample_a_state, sample_b_state)):
            req_state.prompt_embeds = np.ones((len(req_state.prompt_token_ids), 4), dtype=np.float32)
            req_state.prompt_len = len(req_state.prompt_token_ids)
            req_state.cache_slot_id = slot_id
        worker.req_states = {
            "prefill": prefill_state,
            "sample-a": sample_a_state,
            "sample-b": sample_b_state,
        }

        worker._load_snapshot_if_needed = lambda _req_id, req_state, **_kwargs: int(req_state.num_computed_tokens)
        worker._infer_logits_batch_with_sequence = lambda input_embeds_batch, cache_sizes, cache_ids: [
            InferenceLogits(last_token_logits=np.zeros(16, dtype=np.float32), full_sequence_logits=None)
            for _ in input_embeds_batch
        ]
        worker._make_sampling_metadata = lambda _states: None
        worker._sample_next_token = lambda _logits, _metadata: SimpleNamespace(
            sampled_token_ids=torch.tensor([[12], [13]], dtype=torch.int64),
            logprobs_tensors=LogprobsTensors(
                logprob_token_ids=torch.tensor([[12, 1], [13, 2]], dtype=torch.int32),
                logprobs=torch.tensor([[-0.1, -1.0], [-0.2, -2.0]], dtype=torch.float32),
                selected_token_ranks=torch.tensor([1, 1], dtype=torch.int32),
            ),
        )

        output = worker.execute_model(
            SimpleNamespace(
                finished_req_ids=[],
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                    new_block_ids=[],
                    resumed_req_ids=set(),
                ),
                num_scheduled_tokens={"prefill": 3, "sample-a": 3, "sample-b": 3},
                kv_connector_metadata=None,
            )
        )

        assert output is not None
        assert output.req_ids == ["prefill", "sample-a", "sample-b"]
        assert output.logprobs is not None
        assert output.logprobs.cu_num_generated_tokens == [0, 0, 1, 2]
        assert output.logprobs.slice(0, 1).logprob_token_ids.shape[0] == 0
        assert output.logprobs.slice(1, 2).logprob_token_ids.tolist() == [[12, 1]]
        assert output.logprobs.slice(2, 3).logprob_token_ids.tolist() == [[13, 2]]

    def test_last_logit_prompt_logprob_microstep_warning_emitted_once(self, caplog) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 1
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, [1, 2])
        req_state.prompt_len = 2
        req_state.prompt_embeds = np.ones((2, 4), dtype=np.float32)

        worker.cache_model = SimpleNamespace(
            infer=lambda _inputs, *, params: [np.zeros((len(params), 8), dtype=np.float32)],
            get_model_output_shape=lambda: [(1, 8)],
        )

        caplog.set_level("WARNING")
        worker._run_prompt_logprob_microsteps_batch([(0, "req", req_state, 0, 1, 0)])
        worker._run_prompt_logprob_microsteps_batch([(0, "req", req_state, 1, 2, 0)])

        warning_records = [
            record
            for record in caplog.records
            if "Prompt logprobs on last-logit MBLT/MXQ outputs use a slower 1-token microstep path"
            in record.getMessage()
        ]
        assert len(warning_records) == 1

    def test_prompt_logprobs_unsupported_logits_shape_terminal_during_decode(self) -> None:
        worker = self._make_worker()
        prompt_token_ids = [101, 102, 103]
        sampling_params = SamplingParams.from_optional(logprobs=1, prompt_logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.num_computed_tokens = 2

        assert worker._should_recompute_prompt_logprobs_from_start(req_state)
        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=np.empty((1, 1, 1), dtype=np.float32),
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert prompt_logprobs_tensors is None
        assert req_state.next_prompt_logprob_pos == len(prompt_token_ids)
        assert not worker._should_recompute_prompt_logprobs_from_start(req_state)

    def test_2d_single_row_logits_are_not_treated_as_full_sequence_logits(self) -> None:
        vocab_size = 8
        last_token_logits = np.zeros((1, vocab_size), dtype=np.float32)

        assert MbltWorker._normalize_sequence_logits(last_token_logits, expected_seq_len=2) is None
        sequence_logits = MbltWorker._normalize_sequence_logits(last_token_logits, expected_seq_len=1)
        assert sequence_logits is last_token_logits

    def test_runtime_last_token_shape_is_not_prompt_sequence_logits_for_batch1_single_core(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 1
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(1, 8)])
        last_token_logits = np.zeros((1, 8), dtype=np.float32)

        assert worker._runtime_output_logits_mode(input_seq_len=1) == "last_token"
        assert worker._normalize_runtime_sequence_logits(last_token_logits, expected_seq_len=1) is None

    def test_runtime_full_sequence_shape_is_preserved_for_prompt_logprobs(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 1
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(-1, 8)])
        sequence_logits = np.zeros((3, 8), dtype=np.float32)

        normalized = worker._normalize_runtime_sequence_logits(sequence_logits, expected_seq_len=3)

        assert normalized is sequence_logits

    def test_batch1_single_core_echo_prompt_logprobs_microstep_live_cache_not_final_row(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 1
        worker.req_states = {}
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
        worker.print_debug = False
        worker._infer_output_buffers = None

        # The simple prompt mirrors "The capital of France is" tokenized into a
        # short sequence for this worker-level regression.  Requesting both
        # logprobs and prompt_logprobs corresponds to OpenAI completions
        # echo=true, logprobs=5, max_tokens=1.
        prompt_token_ids = [101, 102, 103, 104]
        generated_token_id = 7
        vocab_size = 128
        sampling_params = SamplingParams.from_optional(temperature=0.0, logprobs=5, prompt_logprobs=5)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)
        worker.req_states = {"france": req_state}

        infer_calls: list[tuple[int, int]] = []

        def infer(inputs, *, cache_size, outputs=None):
            input_embeds = np.asarray(inputs[0] if isinstance(inputs, list) else inputs)
            sequence_length = int(input_embeds.shape[1])
            cache_size = int(cache_size)
            infer_calls.append((sequence_length, cache_size))
            prompt_pos = cache_size + sequence_length
            if prompt_pos >= len(prompt_token_ids):
                logits = np.full((1, vocab_size), -40.0, dtype=np.float32)
                logits[0, generated_token_id] = 40.0
                # This is the repeated final-position top token observed in the
                # bug.  It must remain only the generated-token distribution and
                # must not be copied to every echoed prompt position.
                logits[0, 10] = 39.0
                return [logits]

            logits = np.full((1, vocab_size), -40.0, dtype=np.float32)
            logits[0, prompt_token_ids[prompt_pos]] = 40.0
            logits[0, 20 + prompt_pos] = 39.0
            return [logits]

        worker.cache_model = SimpleNamespace(
            infer=infer,
            get_model_output_shape=lambda: [(1, vocab_size)],
            get_num_model_variants=lambda: 0,
        )
        worker._load_snapshot_if_needed = lambda _req_id, req_state, **_kwargs: int(req_state.num_computed_tokens)

        output = worker.execute_model(self._make_scheduler_output({"france": len(prompt_token_ids)}))

        assert output is not None
        assert infer_calls == [(1, pos) for pos in range(len(prompt_token_ids))]
        assert all(sequence_length == 1 for sequence_length, _cache_size in infer_calls)
        assert output.sampled_token_ids[output.req_id_to_index["france"]].tolist() == [generated_token_id]
        assert output.logprobs is not None
        assert output.prompt_logprobs_dict.keys() == {"france"}

        prompt_logprobs_tensors = output.prompt_logprobs_dict["france"]
        prompt_top_ids = prompt_logprobs_tensors.logprob_token_ids[:, 0].tolist()
        assert prompt_top_ids == prompt_token_ids[1:]
        assert len({tuple(row.tolist()) for row in prompt_logprobs_tensors.logprob_token_ids}) > 1
        assert all(10 not in row.tolist() for row in prompt_logprobs_tensors.logprob_token_ids)
        assert output.logprobs.logprob_token_ids.tolist()[0][0] == generated_token_id

    def test_chat_template_echo_prompt_logprobs_do_not_reuse_repeated_final_top_token(self) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(logprobs=5, prompt_logprobs=5)
        # Token IDs stand in for the Llama chat-template repro prompt with
        # special tokens.  The regression checks prompt-position alignment, not
        # tokenizer-specific decoding.
        prompt_token_ids = [128000, 128006, 9125, 128007, 271, 387, 4320, 128009, 128006, 882, 128007, 271, 9906, 1917]
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        vocab_size = max(prompt_token_ids) + 32
        repeated_final_top_id = 271

        prompt_rows = np.full((len(prompt_token_ids) - 1, vocab_size), -40.0, dtype=np.float32)
        for row, token_id in enumerate(prompt_token_ids[1:]):
            prompt_rows[row, repeated_final_top_id] = -5.0
            prompt_rows[row, token_id] = 5.0 + row

        prompt_logprobs_tensors = worker._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=prompt_rows,
            start_idx=0,
            scheduled_end=len(prompt_token_ids),
        )

        assert prompt_logprobs_tensors is not None
        top_token_rows = [tuple(row.tolist()) for row in prompt_logprobs_tensors.logprob_token_ids]
        assert len(set(top_token_rows)) > 1
        assert prompt_logprobs_tensors.logprob_token_ids[:, 0].tolist() == prompt_token_ids[1:]
        repeated_final_top_count = sum(
            repeated_final_top_id == int(row[0]) for row in prompt_logprobs_tensors.logprob_token_ids
        )
        assert repeated_final_top_count < len(prompt_token_ids) - 1

    @pytest.mark.parametrize(
        ("name", "prompt_token_ids", "repeated_final_top_id"),
        [
            ("simple_fact", [101, 102, 103, 104, 105, 106], 201),
            ("repeat", [301, 302, 303, 304, 305, 306, 307, 308], 303),
            (
                "llama_chat",
                [128000, 128006, 9125, 128007, 271, 387, 4320, 128009, 128006, 882, 128007, 271, 9906, 1917],
                9906,
            ),
        ],
    )
    def test_echo_prompt_logprob_microsteps_copy_rows_from_reused_runtime_output_buffer(
        self,
        name: str,
        prompt_token_ids: list[int],
        repeated_final_top_id: int,
    ) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 1
        worker.req_states = {}
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
        worker.print_debug = False
        worker._infer_output_buffers = None

        generated_token_id = 7
        vocab_size = max(max(prompt_token_ids), repeated_final_top_id, generated_token_id) + 16
        sampling_params = SamplingParams.from_optional(temperature=0.0, logprobs=5, prompt_logprobs=5)
        req_state = self._make_request_state(worker, sampling_params, prompt_token_ids)
        req_state.prompt_len = len(prompt_token_ids)
        req_state.prompt_embeds = np.ones((len(prompt_token_ids), 4), dtype=np.float32)
        worker.req_states = {name: req_state}

        infer_calls: list[tuple[int, int, bool]] = []

        def make_logits(cache_size: int, sequence_length: int) -> np.ndarray:
            prompt_pos = cache_size + sequence_length
            logits = np.full((1, vocab_size), -40.0, dtype=np.float32)
            if prompt_pos >= len(prompt_token_ids):
                # Final prefill/assistant-first distribution.  If prompt rows are
                # views into a reused backend output buffer, this row is later
                # observed at most/all echo prompt positions.
                logits[0, generated_token_id] = 40.0
                logits[0, repeated_final_top_id] = 39.0
            else:
                # Position-specific prefill distribution that predicts prompt
                # token i from logits row i - 1.
                logits[0, prompt_token_ids[prompt_pos]] = 40.0
                logits[0, (20 + prompt_pos) % vocab_size] = 39.0
            return logits

        def infer(inputs, *, cache_size, outputs=None):
            input_embeds = np.asarray(inputs[0] if isinstance(inputs, list) else inputs)
            sequence_length = int(input_embeds.shape[1])
            cache_size = int(cache_size)
            infer_calls.append((sequence_length, cache_size, outputs is not None))
            logits = make_logits(cache_size, sequence_length)
            if outputs is not None:
                outputs[0][...] = logits
                return None
            return [logits]

        worker.cache_model = SimpleNamespace(
            infer=infer,
            get_model_output_shape=lambda: [(1, vocab_size)],
            get_num_model_variants=lambda: 0,
        )
        worker._load_snapshot_if_needed = lambda _req_id, req_state, **_kwargs: int(req_state.num_computed_tokens)

        output = worker.execute_model(self._make_scheduler_output({name: len(prompt_token_ids)}))

        assert output is not None
        assert infer_calls == [(1, pos, pos > 0) for pos in range(len(prompt_token_ids))]
        assert output.sampled_token_ids[output.req_id_to_index[name]].tolist() == [generated_token_id]
        assert output.logprobs is not None
        assert output.logprobs.logprob_token_ids.tolist()[0][0] == generated_token_id
        assert output.prompt_logprobs_dict.keys() == {name}

        prompt_logprobs_tensors = output.prompt_logprobs_dict[name]
        prompt_top_ids = prompt_logprobs_tensors.logprob_token_ids[:, 0].tolist()

        assert prompt_top_ids == prompt_token_ids[1:]
        assert len(set(prompt_top_ids)) > 1
        assert prompt_top_ids != [generated_token_id] * (len(prompt_token_ids) - 1)
        assert prompt_top_ids != [repeated_final_top_id] * (len(prompt_token_ids) - 1)
        final_top_count = sum(token_id in (generated_token_id, repeated_final_top_id) for token_id in prompt_top_ids)
        assert final_top_count < len(prompt_top_ids) - 1

    def test_batch_2d_output_shape_is_treated_as_last_token_even_when_seq_len_matches_batch(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 2
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(2, 8)])

        assert worker._runtime_output_logits_mode(input_seq_len=2) == "last_token"

    def test_batch_3d_output_shape_does_not_force_full_sequence_mode(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 16

        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(1, 128, 8)])
        assert worker._runtime_output_logits_mode(input_seq_len=128) == "unknown"

        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(1, -1, 8)])
        assert worker._runtime_output_logits_mode(input_seq_len=128) == "unknown"

    def test_batch_3d_prompt_logprobs_use_microsteps(self) -> None:
        worker = self._make_worker()
        worker.max_batch_size = 16
        worker.cache_model = SimpleNamespace(get_model_output_shape=lambda: [(1, 128, 8)])
        req_state = self._make_request_state(
            worker,
            SamplingParams.from_optional(prompt_logprobs=1),
            list(range(512)),
        )

        assert worker._needs_last_logit_prompt_logprob_microsteps(req_state, sequence_length=128)

    def test_sampling_penalties_can_be_forced_off_for_non_cuda_runtime(self) -> None:
        worker = self._make_worker()
        worker.enable_sampling_penalties = False
        worker._warned_penalties_disabled = False
        sampling_params = SamplingParams.from_optional(
            frequency_penalty=0.5, presence_penalty=0.2, repetition_penalty=1.1
        )
        req_state = self._make_request_state(worker, sampling_params, [1, 2, 3])
        metadata = worker._make_sampling_metadata([req_state])
        assert metadata.no_penalties
        assert metadata.prompt_token_ids is None
        assert metadata.frequency_penalties.tolist() == [0.0]
        assert metadata.presence_penalties.tolist() == [0.0]
        assert metadata.repetition_penalties.tolist() == [1.0]

    def test_prefill_completion_step_is_sampled(self) -> None:
        worker = self._make_worker()
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), [1, 2, 3, 4])
        req_state.prompt_len = 4
        assert worker._should_sample_after_step(req_state, 4, 4)
        assert worker._should_sample_after_step(req_state, 5, 1)
        assert not worker._should_sample_after_step(req_state, 3, 3)
        assert not worker._should_sample_after_step(req_state, 4, 0)

    def test_generated_token_logprobs_are_log_softmax_normalized(self) -> None:
        worker = self._make_worker()
        sampling_params = SamplingParams.from_optional(temperature=0.0, logprobs=1)
        req_state = self._make_request_state(worker, sampling_params, [1, 2])
        sampling_metadata = worker._make_sampling_metadata([req_state])
        logits = torch.tensor([[0.0, 5.0, 1.0]], dtype=torch.float32)

        sampler_output = worker._sample_next_token(logits=logits, sampling_metadata=sampling_metadata)
        assert sampler_output.logprobs_tensors is not None

        logprobs_tensors = sampler_output.logprobs_tensors
        expected_logprobs = torch.log_softmax(logits, dim=-1)
        selected_token_id = int(sampler_output.sampled_token_ids[0, 0].item())

        assert selected_token_id == 1
        assert torch.all(logprobs_tensors.logprobs <= 0)
        torch.testing.assert_close(
            logprobs_tensors.logprobs[0, 0],
            expected_logprobs[0, selected_token_id],
        )
        torch.testing.assert_close(
            logprobs_tensors.logprobs[0, 1],
            expected_logprobs[0, 1],
        )

    def test_normalize_multimodal_embeddings_accepts_tensor_outputs(self) -> None:
        embeddings = torch.randn(2, 4)
        assert MbltWorker._normalize_multimodal_embeddings(embeddings) is embeddings
        assert MbltWorker._normalize_multimodal_embeddings((embeddings,)) is embeddings

    def test_normalize_multimodal_embeddings_accepts_qwen3_vl_outputs(self) -> None:
        first_image = torch.randn(2, 4)
        second_image = torch.randn(3, 4)
        deepstack_features = [torch.randn(5, 4)]
        single = MbltWorker._normalize_multimodal_embeddings(((first_image,), deepstack_features))
        multiple = MbltWorker._normalize_multimodal_embeddings(((first_image, second_image), deepstack_features))
        assert single is first_image
        torch.testing.assert_close(multiple, torch.cat((first_image, second_image), dim=0))

    def test_scatter_deepstack_embeddings_aligns_to_prompt_positions(self) -> None:
        prompt_embeds = torch.zeros(6, 4)
        placeholder = SimpleNamespace(offset=2, length=2, is_embed=None)
        layer0 = torch.ones(2, 4)
        layer1 = torch.full((2, 4), 2.0)
        deepstack = MbltWorker._scatter_deepstack_embeddings(None, prompt_embeds, placeholder, [layer0, layer1])
        assert deepstack is not None
        assert deepstack is not None
        assert tuple(deepstack.shape) == (2, 6, 4)
        torch.testing.assert_close(deepstack[0, 2:4], layer0)
        torch.testing.assert_close(deepstack[1, 2:4], layer1)
        torch.testing.assert_close(deepstack[:, :2], torch.zeros(2, 2, 4))
        torch.testing.assert_close(deepstack[:, 4:], torch.zeros(2, 2, 4))

    def test_build_prompt_embeds_keeps_prompt_embeds_without_mm_features(self) -> None:
        worker = self._make_worker()
        prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        merged, deepstack = worker._build_prompt_embeds(
            prompt_token_ids=None, prompt_embeds=prompt_embeds, mm_features=None
        )
        assert merged is not prompt_embeds
        torch.testing.assert_close(merged, prompt_embeds)
        assert deepstack is None

    def test_build_prompt_embeds_scatters_mm_features_into_prompt_embeds(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        base_prompt_embeds = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        original_prompt_embeds = base_prompt_embeds.clone()
        image_embeds = torch.full((2, 4), 7.0)
        deepstack_layers = [torch.full((2, 4), 3.0), torch.full((2, 4), 5.0)]
        captured = {}

        def get_image_features(**kwargs):
            captured.update(kwargs)
            return ((image_embeds,), deepstack_layers)

        worker.model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000), get_image_features=get_image_features)
        feature = SimpleNamespace(
            modality="image",
            data={"pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.tensor([1, 1, 1])},
            mm_position=SimpleNamespace(offset=1, length=2, is_embed=None),
        )
        merged, deepstack = worker._build_prompt_embeds(
            prompt_token_ids=None, prompt_embeds=base_prompt_embeds, mm_features=[feature]
        )
        assert merged is not base_prompt_embeds
        assert tuple(merged.shape) == (5, 4)
        torch.testing.assert_close(merged[:1], original_prompt_embeds[:1])
        torch.testing.assert_close(merged[1:3], image_embeds)
        torch.testing.assert_close(merged[3:], original_prompt_embeds[3:])
        torch.testing.assert_close(base_prompt_embeds, original_prompt_embeds)
        assert tuple(captured["image_grid_thw"].shape) == (1, 3)
        assert deepstack is not None
        assert deepstack is not None
        assert tuple(deepstack.shape) == (2, 5, 4)
        torch.testing.assert_close(deepstack[0, 1:3], deepstack_layers[0])
        torch.testing.assert_close(deepstack[1, 1:3], deepstack_layers[1])
        torch.testing.assert_close(deepstack[:, :1], torch.zeros(2, 1, 4))
        torch.testing.assert_close(deepstack[:, 3:], torch.zeros(2, 2, 4))

    def test_build_prompt_embeds_ignores_deepstack_outputs_for_non_qwen3_vl(self) -> None:
        worker = self._make_worker()
        base_prompt_embeds = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        image_embeds = torch.full((2, 4), 7.0)
        deepstack_layers = [torch.full((2, 4), 3.0)]
        worker.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=32000, model_type="qwen2_vl"),
            get_image_features=lambda **_kwargs: ((image_embeds,), deepstack_layers),
        )
        feature = SimpleNamespace(
            modality="image",
            data={"pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.tensor([1, 1, 1])},
            mm_position=SimpleNamespace(offset=1, length=2, is_embed=None),
        )
        merged, deepstack = worker._build_prompt_embeds(
            prompt_token_ids=None, prompt_embeds=base_prompt_embeds, mm_features=[feature]
        )
        assert merged is not base_prompt_embeds
        torch.testing.assert_close(merged[1:3], image_embeds)
        torch.testing.assert_close(base_prompt_embeds, torch.arange(20, dtype=torch.float32).reshape(5, 4))
        assert deepstack is None

    def test_single_request_vlm_reuses_lm_kv_prefix_and_runs_suffix_only(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        worker.model.config = SimpleNamespace(model_type="mobilint-qwen2_vl", vocab_size=32000)
        worker.req_states = {}
        worker._infer_output_buffers = None
        worker.runtime_cache.set_io_adapters(
            dump_runtime_cache=lambda _slot_id: ["unused"],
            load_runtime_cache=lambda blobs, _slot_id: blobs == ["lm-kv-prefix"],
        )
        base_prompt_embeds = torch.arange(130 * 4, dtype=torch.float32).reshape(130, 4)
        feature = SimpleNamespace(
            modality="image",
            data={"pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.tensor([1, 1, 1])},
            mm_position=SimpleNamespace(offset=4, length=2, is_embed=None),
        )
        worker.runtime_cache.store_snapshot(
            req_id="shared-text-prefix",
            blobs=["lm-kv-prefix"],
            block_ids=([11],),
            first_seq_blocks=(11,),
            num_tokens=128,
            cache_token_ids=tuple(range(128)),
            multimodal_cache_identity=MbltWorker._build_vlm_multimodal_cache_identity(
                "vlm-session",
                [feature],
            ),
        )

        image_feature_calls: list[dict[str, torch.Tensor]] = []
        image_embeds = torch.full((2, 4), 9.0)

        def get_image_features(**kwargs):
            image_feature_calls.append(kwargs)
            return image_embeds

        infer_calls: list[tuple[int, int, np.ndarray]] = []

        def infer(inputs, *, cache_size, outputs=None):
            input_embeds = np.asarray(inputs[0] if isinstance(inputs, list) else inputs)
            infer_calls.append((int(input_embeds.shape[1]), int(cache_size), input_embeds.copy()))
            return [np.full((1, 16), 1.0, dtype=np.float32)]

        worker.model.get_image_features = get_image_features
        worker.cache_model = SimpleNamespace(
            infer=infer,
            get_model_output_shape=lambda: [(1, 16)],
            get_num_model_variants=lambda: 0,
        )
        worker._make_sampling_metadata = lambda _states: None
        worker._sample_next_token = lambda _logits, _metadata: SimpleNamespace(
            sampled_token_ids=torch.tensor([[7]], dtype=torch.int64),
            logprobs_tensors=None,
        )

        new_req = self._make_new_request(
            "vlm-hit",
            base_prompt_embeds,
            [feature],
            num_computed_tokens=128,
            block_ids=([11, 12],),
            session_id="vlm-session",
        )
        scheduler_output = self._make_scheduler_output({"vlm-hit": 2})
        scheduler_output.scheduled_new_reqs = [new_req]

        output = worker.execute_model(scheduler_output)

        assert output is not None
        assert image_feature_calls and tuple(image_feature_calls[0]["image_grid_thw"].shape) == (1, 3)
        assert [(sequence_length, cache_size) for sequence_length, cache_size, _inputs in infer_calls] == [(2, 128)]
        np.testing.assert_array_equal(infer_calls[0][2][0], base_prompt_embeds[128:130].numpy())
        assert worker.req_states["vlm-hit"].num_computed_tokens == 130
        assert worker.runtime_cache.loaded_req_id == "vlm-hit"

    def test_repeated_image_vlm_requests_still_rebuild_vision_embeddings(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen2_vl")
        worker.model.config = SimpleNamespace(model_type="mobilint-qwen2_vl", vocab_size=32000)
        worker.req_states = {}
        worker._infer_output_buffers = None
        worker.runtime_cache.set_io_adapters(
            dump_runtime_cache=lambda _slot_id: ["live-kv"],
            load_runtime_cache=lambda _blobs, _slot_id: True,
        )

        image_feature_call_count = 0

        def get_image_features(**_kwargs):
            nonlocal image_feature_call_count
            image_feature_call_count += 1
            return torch.full((2, 4), float(image_feature_call_count))

        infer_inputs: list[np.ndarray] = []

        def infer(inputs, *, cache_size, outputs=None):
            del cache_size, outputs
            infer_inputs.append(np.asarray(inputs[0] if isinstance(inputs, list) else inputs).copy())
            return [np.full((1, 16), 1.0, dtype=np.float32)]

        worker.model.get_image_features = get_image_features
        worker.cache_model = SimpleNamespace(
            infer=infer,
            get_model_output_shape=lambda: [(1, 16)],
            get_num_model_variants=lambda: 0,
        )
        worker._make_sampling_metadata = lambda _states: None
        worker._sample_next_token = lambda _logits, _metadata: SimpleNamespace(
            sampled_token_ids=torch.tensor([[7]], dtype=torch.int64),
            logprobs_tensors=None,
        )

        for index, req_id in enumerate(("vlm-first", "vlm-second"), start=1):
            prompt_embeds = torch.zeros(6, 4)
            feature = SimpleNamespace(
                modality="image",
                data={"pixel_values": torch.full((1, 3), float(index)), "image_grid_thw": torch.tensor([1, 1, 1])},
                mm_position=SimpleNamespace(offset=2, length=2, is_embed=None),
            )
            scheduler_output = self._make_scheduler_output({req_id: 6})
            scheduler_output.scheduled_new_reqs = [
                self._make_new_request(
                    req_id,
                    prompt_embeds,
                    [feature],
                    block_ids=([20 + index],),
                    session_id=f"vlm-session-{index}",
                )
            ]

            assert worker.execute_model(scheduler_output) is not None

        assert image_feature_call_count == 2
        assert len(infer_inputs) == 2
        np.testing.assert_array_equal(infer_inputs[0][0, 2:4], np.full((2, 4), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(infer_inputs[1][0, 2:4], np.full((2, 4), 2.0, dtype=np.float32))

    def test_build_deepstack_input_embeds_pads_decode_tokens(self) -> None:
        worker = self._make_worker()
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), [1, 2])
        req_state.prompt_len = 4
        req_state.prompt_deepstack_embeds = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        sliced = worker._build_deepstack_input_embeds(req_state, 2, 6)
        assert sliced is not None
        assert sliced is not None
        assert tuple(sliced.shape) == (2, 4, 3)
        np.testing.assert_array_equal(sliced[:, :2, :], req_state.prompt_deepstack_embeds[:, 2:4, :])
        np.testing.assert_array_equal(sliced[:, 2:, :], np.zeros((2, 2, 3), dtype=np.float32))

    def test_build_infer_inputs_adds_zero_deepstack_for_qwen3_vl_dual_input_model(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        worker.cache_model = SimpleNamespace(
            get_num_model_variants=lambda: 1,
            get_model_variant_handle=lambda _idx: SimpleNamespace(
                get_model_input_shape=lambda: [(1, -1, 4), (3, -1, 4)]
            ),
        )
        input_embeds = np.ones((5, 4), dtype=np.float32)
        infer_inputs = worker._build_infer_inputs(input_embeds, None)
        assert isinstance(infer_inputs, list)
        assert isinstance(infer_inputs, list)
        assert tuple(infer_inputs[0].shape) == (1, 5, 4)
        assert tuple(infer_inputs[1].shape) == (3, 5, 4)
        np.testing.assert_array_equal(infer_inputs[1], np.zeros((3, 5, 4), dtype=np.float32))

    def test_build_infer_inputs_ignores_dual_input_shape_for_non_qwen3_vl_model(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="qwen2_vl")
        worker.cache_model = SimpleNamespace(
            get_num_model_variants=lambda: 1,
            get_model_variant_handle=lambda _idx: SimpleNamespace(
                get_model_input_shape=lambda: [(1, -1, 4), (3, -1, 4)]
            ),
        )
        input_embeds = np.ones((5, 4), dtype=np.float32)
        infer_inputs = worker._build_infer_inputs(input_embeds, None)
        assert isinstance(infer_inputs, np.ndarray)
        assert tuple(infer_inputs.shape) == (1, 5, 4)

    def test_build_infer_inputs_rejects_explicit_deepstack_for_non_qwen3_vl_model(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="qwen2_vl")
        worker.cache_model = SimpleNamespace(
            get_num_model_variants=lambda: 1,
            get_model_variant_handle=lambda _idx: SimpleNamespace(
                get_model_input_shape=lambda: [(1, -1, 4), (3, -1, 4)]
            ),
        )
        input_embeds = np.ones((5, 4), dtype=np.float32)
        deepstack_embeds = np.zeros((3, 5, 4), dtype=np.float32)
        with pytest.raises(RuntimeError, match="only supported for Qwen3-VL"):
            worker._build_infer_inputs(input_embeds, deepstack_embeds)

    def test_build_infer_inputs_rejects_invalid_deepstack_shape(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        worker.cache_model = SimpleNamespace(
            get_num_model_variants=lambda: 1,
            get_model_variant_handle=lambda _idx: SimpleNamespace(
                get_model_input_shape=lambda: [(1, -1, 4), (3, -1, 8)]
            ),
        )
        input_embeds = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(RuntimeError, match="hidden dimension mismatch"):
            worker._build_infer_inputs(input_embeds, None)

    def test_build_infer_inputs_rejects_mismatched_deepstack_embeds(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        worker.cache_model = SimpleNamespace(
            get_num_model_variants=lambda: 1,
            get_model_variant_handle=lambda _idx: SimpleNamespace(
                get_model_input_shape=lambda: [(1, -1, 4), (3, -1, 4)]
            ),
        )
        input_embeds = np.ones((5, 4), dtype=np.float32)
        deepstack_embeds = np.ones((2, 5, 4), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Deepstack embedding shape mismatch"):
            worker._build_infer_inputs(input_embeds, deepstack_embeds)

    def test_infer_logits_passes_deepstack_to_dual_input_model(self) -> None:
        worker = self._make_worker()
        worker.model_config.hf_config = SimpleNamespace(model_type="mobilint-qwen3_vl")
        captured = {}

        def infer(inputs, **kwargs):
            captured["inputs"] = inputs
            captured["kwargs"] = kwargs
            return [np.arange(1 * 2 * 5, dtype=np.float32).reshape(1, 2, 5)]

        worker.cache_model = SimpleNamespace(
            get_num_model_variants=lambda: 1,
            get_model_variant_handle=lambda _idx: SimpleNamespace(
                get_model_input_shape=lambda: [(1, -1, 4), (2, -1, 4)]
            ),
            infer=infer,
            get_model_output_shape=lambda: [],
        )
        worker._infer_output_buffers = None
        input_embeds = np.ones((2, 4), dtype=np.float32)
        deepstack_embeds = np.full((2, 2, 4), 3.0, dtype=np.float32)
        logits = worker._infer_logits(input_embeds, deepstack_embeds, cache_size=7)
        assert tuple(logits.shape) == (1, 5)
        assert "inputs" in captured
        infer_inputs = captured["inputs"]
        assert isinstance(infer_inputs, list)
        assert isinstance(infer_inputs, list)
        assert len(infer_inputs) == 2
        np.testing.assert_array_equal(infer_inputs[0], np.expand_dims(input_embeds, axis=0))
        np.testing.assert_array_equal(infer_inputs[1], deepstack_embeds)
        assert captured["kwargs"] == {"cache_size": 7}

    def test_batch_vlm_fails_fast_until_artifacts_are_available(self) -> None:
        worker = self._make_worker()
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), [1, 2])
        req_state.is_multimodal = True
        req_state.prompt_deepstack_embeds = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(RuntimeError, match="VLM batch execution is not supported"):
            worker._ensure_batch_vlm_supported(req_state)

    def test_batch_vlm_without_deepstack_fails_fast_until_artifacts_are_available(self) -> None:
        worker = self._make_worker()
        req_state = self._make_request_state(worker, SamplingParams.from_optional(), [1, 2])
        req_state.is_multimodal = True
        req_state.prompt_deepstack_embeds = None
        with pytest.raises(RuntimeError, match="VLM batch execution is not supported"):
            worker._ensure_batch_vlm_supported(req_state)

    def test_init_uses_text_runtime_max_batch_size_for_cache_slots(self, monkeypatch) -> None:
        def worker_base_init(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False):
            self.vllm_config = vllm_config
            self.local_rank = local_rank
            self.rank = rank

        monkeypatch.setattr("vllm_mblt.mblt_worker.WorkerBase.__init__", worker_base_init)
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            load_config=SimpleNamespace(
                model_loader_extra_config={"core_mode": "global4", "max_batch_size": 16, "text_max_batch_size": 4}
            ),
            model_config=SimpleNamespace(max_model_len=1024, hf_config=SimpleNamespace(model_type="mobilint-qwen3_vl")),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=128),
        )

        worker = MbltWorker(config, local_rank=0, rank=0, distributed_init_method="env://")

        assert worker.max_batch_size == 4
        assert worker.runtime_cache.free_slots == [0, 1, 2, 3]

    def test_init_uses_shared_text_only_max_batch_size_for_cache_slots(self, monkeypatch) -> None:
        def worker_base_init(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False):
            self.vllm_config = vllm_config
            self.local_rank = local_rank
            self.rank = rank

        monkeypatch.setattr("vllm_mblt.mblt_worker.WorkerBase.__init__", worker_base_init)
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            load_config=SimpleNamespace(
                model_loader_extra_config={"core_mode": "global4", "max_batch_size": 16, "text_max_batch_size": 4}
            ),
            model_config=SimpleNamespace(max_model_len=1024, hf_config=SimpleNamespace(model_type="qwen2")),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=128),
        )

        worker = MbltWorker(config, local_rank=0, rank=0, distributed_init_method="env://")

        assert worker.max_batch_size == 16
        assert worker.runtime_cache.free_slots == list(range(16))

    def test_load_model_passes_text_runtime_layout_kwargs_to_from_pretrained(self, monkeypatch) -> None:
        worker = MbltWorker.__new__(MbltWorker)
        worker.rank = 0
        worker.local_rank = 0
        worker.model = None
        worker.cache_model = None
        worker._infer_output_buffers = None
        worker.max_batch_size = 1
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
        worker.load_config = SimpleNamespace(
            model_loader_extra_config={
                "dev_no": 2,
                "mxq_path": "/tmp/model.mxq",
                "max_batch_size": 8,
                "npu_prefill_chunk_size": {"global8": 512},
                "target_cores": ["1:0"],
                "target_clusters": [0, 1],
                "core_mode": "global8",
                "text_core_mode": "single",
                "text_target_cores": ["0:0"],
                "vision_mxq_path": "/tmp/vision.mxq",
                "vision_core_mode": "global4",
                "vision_target_clusters": [0],
                "unrelated": "ignored",
            }
        )
        worker.model_config = SimpleNamespace(
            model="mobilint/test-model", hf_config=SimpleNamespace(model_type="qwen2"), model_kwargs={}, hf_overrides={}
        )
        worker.vllm_config = SimpleNamespace(
            load_config=SimpleNamespace(model_loader_extra_config={}), model_config=worker.model_config
        )
        fake_model = SimpleNamespace(
            eval=lambda: None,
            get_input_embeddings=lambda: SimpleNamespace(),
            get_cache_mxq_model=lambda: SimpleNamespace(),
        )
        calls = []

        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return fake_model

        monkeypatch.setattr("vllm_mblt.mblt_worker.AutoModelForCausalLM.from_pretrained", from_pretrained)

        worker.load_model()

        assert calls == [
            (
                ("mobilint/test-model",),
                {
                    "trust_remote_code": True,
                    "dev_no": 2,
                    "mxq_path": "/tmp/model.mxq",
                    "max_batch_size": 8,
                    "npu_prefill_chunk_size": {"global8": 512},
                    "target_cores": ["1:0"],
                    "target_clusters": [0, 1],
                    "core_mode": "global8",
                    "text_core_mode": "single",
                    "text_target_cores": ["0:0"],
                    "vision_mxq_path": "/tmp/vision.mxq",
                    "vision_core_mode": "global4",
                    "vision_target_clusters": [0],
                },
            )
        ]

    def test_load_model_expands_vlm_runtime_layout_kwargs_to_subconfigs(self, monkeypatch) -> None:
        worker = MbltWorker.__new__(MbltWorker)
        worker.rank = 0
        worker.local_rank = 0
        worker.model = None
        worker.cache_model = None
        worker._infer_output_buffers = None
        worker.max_batch_size = 1
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
        worker.load_config = SimpleNamespace(
            model_loader_extra_config={
                "dev_no": 0,
                "mxq_path": "/tmp/ignored-top-level.mxq",
                "max_batch_size": 4,
                "npu_prefill_chunk_size": {"global4": 256},
                "target_cores": ["1:0"],
                "target_clusters": [0, 1],
                "core_mode": "global4",
                "text_core_mode": "single",
                "text_target_cores": ["0:0"],
                "vision_mxq_path": "/tmp/vision.mxq",
                "vision_core_mode": "multi",
                "vision_target_clusters": [0],
            }
        )
        worker.model_config = SimpleNamespace(
            model="mobilint/Qwen3-VL-2B-Instruct",
            hf_config=SimpleNamespace(model_type="mobilint-qwen3_vl"),
            model_kwargs={},
            hf_overrides={},
        )
        worker.vllm_config = SimpleNamespace(
            load_config=SimpleNamespace(model_loader_extra_config={}), model_config=worker.model_config
        )
        fake_model = SimpleNamespace(
            eval=lambda: None,
            get_input_embeddings=lambda: SimpleNamespace(),
            get_cache_mxq_model=lambda: SimpleNamespace(),
        )
        calls = []

        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return fake_model

        monkeypatch.setattr("vllm_mblt.mblt_worker.AutoModelForImageTextToText.from_pretrained", from_pretrained)

        worker.load_model()

        assert calls == [
            (
                ("mobilint/Qwen3-VL-2B-Instruct",),
                {
                    "trust_remote_code": True,
                    "vision_dev_no": 0,
                    "text_dev_no": 0,
                    "vision_max_batch_size": 4,
                    "text_max_batch_size": 4,
                    "vision_npu_prefill_chunk_size": {"global4": 256},
                    "text_npu_prefill_chunk_size": {"global4": 256},
                    "vision_target_cores": ["1:0"],
                    "text_target_cores": ["0:0"],
                    "vision_target_clusters": [0],
                    "text_target_clusters": [0, 1],
                    "vision_core_mode": "multi",
                    "text_core_mode": "single",
                    "vision_mxq_path": "/tmp/vision.mxq",
                },
            )
        ]
