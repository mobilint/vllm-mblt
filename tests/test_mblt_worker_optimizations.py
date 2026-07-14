from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logprobs import Logprob
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.sampler import Sampler

from vllm_mblt.mblt_worker import (
    MbltWorker,
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

    def _make_worker(self) -> MbltWorker:
        worker = MbltWorker.__new__(MbltWorker)
        worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen2"))
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128), model_config=worker.model_config
        )
        worker.model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
        worker.empty_logits_processors = LogitsProcessors(None)
        worker.empty_prompt_token_ids = torch.empty((0, 0), dtype=torch.int64)
        worker.sampler = Sampler(logprobs_mode="raw_logits")
        worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
        worker._vlm_image_positions_by_session = {}
        return worker

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
        self, modality: str = "image", *, offset: int = 1, length: int = 2, is_embed: torch.Tensor | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            modality=modality, data={}, mm_position=SimpleNamespace(offset=offset, length=length, is_embed=is_embed)
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
