import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintGenerationMixin
from qbruntime import BatchParam
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, MLAAttentionSpec
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.worker_base import WorkerBase

from vllm_mblt.mblt_platform import resolve_model_max_batch_size
from vllm_mblt.runtime_cache import (
    KVBlockIds,
    MbltRuntimeCacheManager,
    RuntimeCacheRequest,
    append_block_ids,
    first_seq_blocks,
    normalize_block_ids,
)

logger = init_logger(__name__)


_MULTIMODAL_HF_MODEL_TYPES = frozenset(
    {
        "mobilint-qwen2_vl",
        "mobilint-qwen3_vl",
    }
)

_MBLT_BACKEND_KWARG_FIELDS = frozenset(
    {
        "mxq_path",
        "dev_no",
        "max_batch_size",
        "core_mode",
        "target_cores",
        "target_clusters",
        "npu_prefill_chunk_size",
    }
)
_MBLT_BACKEND_KWARG_PREFIXES = ("", "text_", "vision_", "encoder_", "decoder_", "base_", "draft_", "fc_")
_MBLT_BACKEND_KWARG_NAMES = frozenset(
    f"{prefix}{field}" for prefix in _MBLT_BACKEND_KWARG_PREFIXES for field in _MBLT_BACKEND_KWARG_FIELDS
)
_MULTIMODAL_SHARED_BACKEND_KWARG_FIELDS = _MBLT_BACKEND_KWARG_FIELDS - {"mxq_path"}
_MULTIMODAL_BACKEND_PREFIXES = ("vision_", "text_")


def _is_multimodal_hf_config(hf_config: object) -> bool:
    model_type = getattr(hf_config, "model_type", None)
    return isinstance(model_type, str) and model_type in _MULTIMODAL_HF_MODEL_TYPES


def _is_qwen3_vl_hf_config(hf_config: object) -> bool:
    model_type = getattr(hf_config, "model_type", None)
    return model_type == "mobilint-qwen3_vl"


def _normalize_model_kwargs_for_hf_config(
    model_kwargs: dict[str, object],
    hf_config: object,
) -> dict[str, object]:
    if not _is_multimodal_hf_config(hf_config):
        return model_kwargs

    normalized: dict[str, object] = {}
    shared: dict[str, object] = {}
    for key, value in model_kwargs.items():
        if any(key.startswith(prefix) for prefix in _MULTIMODAL_BACKEND_PREFIXES):
            normalized[key] = value
        elif key in _MULTIMODAL_SHARED_BACKEND_KWARG_FIELDS:
            shared[key] = value

    for field, value in shared.items():
        for prefix in _MULTIMODAL_BACKEND_PREFIXES:
            normalized.setdefault(f"{prefix}{field}", value)

    return normalized


@dataclass
class RequestState:
    is_prefill: bool
    output_token_ids: list[int]
    sampling_params: SamplingParams
    cached_sampling_state: "CachedSamplingState"
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_computed_tokens: int
    num_output_tokens: int
    prompt_embeds: np.ndarray
    prompt_deepstack_embeds: Optional[np.ndarray]
    is_multimodal: bool
    prompt_len: int
    prompt_token_ids: list[int]
    cache_slot_id: Optional[int]
    vlm_session_id: Optional[str]
    next_prompt_logprob_pos: int = 1
    in_progress_prompt_logprobs: Optional[LogprobsTensors] = None


@dataclass
class CachedSamplingState:
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    generator: Optional[torch.Generator]
    max_num_logprobs: Optional[int]
    bad_words_token_ids: Optional[list[list[int]]]
    prompt_token_ids: torch.Tensor
    has_penalties: bool


@dataclass
class InferenceLogits:
    last_token_logits: np.ndarray
    full_sequence_logits: Optional[np.ndarray]


@dataclass
class PromptLogprobFallbackReplayState:
    output_index: int
    req_state: RequestState
    start_idx: int
    prompt_end: int
    cache_id: int
    current_prompt_pos: int
    fallback_cache_size: int
    rows: list[np.ndarray]


@dataclass
class PromptLogprobMicrostepState:
    output_index: int
    req_id: str
    req_state: RequestState
    start_idx: int
    scheduled_end: int
    cache_id: int
    cache_size: int
    prompt_logits_end: int
    rows: list[np.ndarray]
    last_token_logits: Optional[np.ndarray] = None


@dataclass
class NormalBatchChunkState:
    output_index: int
    input_embeds: np.ndarray
    cache_id: int
    cache_size: int
    offset: int = 0
    full_sequence_logits: Optional[list[np.ndarray]] = None
    last_token_logits: Optional[np.ndarray] = None


class MbltWorker(WorkerBase):
    MAX_FINISHED_CACHE_SNAPSHOTS = 16

    def __init__(
        self,
        vllm_config: "VllmConfig",
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)

        self.model: Optional[MobilintGenerationMixin] = None
        self.input_embeddings: Optional[nn.Module] = None
        self.cache_model: Optional[Any] = None
        self._infer_output_buffers: Optional[list[np.ndarray]] = None

        self.req_states: Dict[str, RequestState] = {}
        self._warned_batch_cache_snapshot_unsupported = False
        self._warned_last_logit_prompt_logprobs = False
        self._vlm_image_positions_by_session: dict[str, tuple[int, int, Optional[tuple[bool, ...]]]] = {}

        self.max_batch_size = resolve_model_max_batch_size(self.vllm_config) or 1
        self.max_seq_len = self.vllm_config.model_config.max_model_len
        self.runtime_cache = MbltRuntimeCacheManager(
            max_batch_size=self.max_batch_size,
            block_size=self._kv_block_size(),
            max_finished_snapshots=self.MAX_FINISHED_CACHE_SNAPSHOTS,
            dump_runtime_cache=self._dump_runtime_cache,
            load_runtime_cache=self._load_runtime_cache,
        )
        self.sampler = Sampler(logprobs_mode="raw_logits")
        self.empty_logits_processors = LogitsProcessors(None)
        self.empty_prompt_token_ids = torch.empty((0, 0), dtype=torch.int64)
        self.enable_chunked_prefill = self.vllm_config.scheduler_config.enable_chunked_prefill
        self.max_num_batched_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # Disabled by default to avoid per-token stdout spam in production runs.
        self.print_debug = os.getenv("VLLM_MBLT_DEBUG", "0") in {"1", "true", "TRUE", "True"}
        penalties_env = os.getenv("VLLM_MBLT_ENABLE_SAMPLING_PENALTIES")
        if penalties_env is None:
            self.enable_sampling_penalties = torch.cuda.is_available()
        else:
            self.enable_sampling_penalties = penalties_env in {"1", "true", "TRUE", "True"}
        self._warned_penalties_disabled = False

    def _log_init_stage(self, stage: str, start_time: Optional[float] = None, **fields: object) -> None:
        payload = {
            "pid": os.getpid(),
            "rank": self.rank,
            "local_rank": self.local_rank,
            **fields,
        }
        suffix = ""
        if start_time is not None:
            suffix = f" elapsed={time.perf_counter() - start_time:.2f}s"
        details = " ".join(f"{key}={value!r}" for key, value in payload.items())
        logger.info("[mblt-init] %s %s%s", stage, details, suffix)

    def _reset_cache_slots(self) -> None:
        self.runtime_cache.set_io_adapters(
            dump_runtime_cache=self._dump_runtime_cache,
            load_runtime_cache=self._load_runtime_cache,
        )
        self.runtime_cache.reset_slots(max_batch_size=self.max_batch_size)

    def _is_batch_model(self) -> bool:
        return self.max_batch_size > 1

    def _kv_block_size(self) -> int:
        configured = self.vllm_config.cache_config.block_size
        if configured is None:
            return 128
        return int(configured)

    def _num_blocks_per_request(self) -> int:
        return max(1, math.ceil(self.max_seq_len / self._kv_block_size()))

    def _normalize_block_ids(self, block_ids: KVBlockIds) -> KVBlockIds:
        return normalize_block_ids(block_ids)

    def _append_block_ids(
        self,
        current_block_ids: KVBlockIds,
        new_block_ids: KVBlockIds,
    ) -> KVBlockIds:
        return append_block_ids(current_block_ids, new_block_ids)

    def _first_seq_blocks(self, block_ids: KVBlockIds) -> tuple[int, ...]:
        return first_seq_blocks(block_ids)

    def _get_cache_model(self) -> Any:
        if self.cache_model is None:
            if self.model is None:
                raise RuntimeError("Model is not initialized.")
            self.cache_model = self.model.get_cache_mxq_model()
        return self.cache_model

    def _supports_deepstack_input(self) -> bool:
        hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
        if _is_qwen3_vl_hf_config(hf_config):
            return True

        vllm_model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        hf_config = getattr(vllm_model_config, "hf_config", None)
        if _is_qwen3_vl_hf_config(hf_config):
            return True

        model_config = getattr(getattr(self, "model", None), "config", None)
        return _is_qwen3_vl_hf_config(model_config)

    def _is_multimodal_model(self) -> bool:
        hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
        if _is_multimodal_hf_config(hf_config):
            return True

        vllm_model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        hf_config = getattr(vllm_model_config, "hf_config", None)
        if _is_multimodal_hf_config(hf_config):
            return True

        model_config = getattr(getattr(self, "model", None), "config", None)
        return _is_multimodal_hf_config(model_config)

    @staticmethod
    def _multimodal_position_signature(
        placeholder: PlaceholderRange,
    ) -> tuple[int, int, Optional[tuple[bool, ...]]]:
        is_embed = getattr(placeholder, "is_embed", None)
        if is_embed is None:
            embed_signature = None
        elif isinstance(is_embed, torch.Tensor):
            embed_signature = tuple(bool(value) for value in is_embed.detach().cpu().bool().tolist())
        else:
            embed_signature = tuple(bool(value) for value in is_embed)
        return (int(placeholder.offset), int(placeholder.length), embed_signature)

    @staticmethod
    def _get_vlm_session_id(new_req: object) -> str:
        for attr in ("session_id", "conversation_id"):
            value = getattr(new_req, attr, None)
            if value is not None:
                return str(value)

        metadata = getattr(new_req, "metadata", None)
        if isinstance(metadata, dict):
            for key in ("session_id", "conversation_id"):
                value = metadata.get(key)
                if value is not None:
                    return str(value)

        return str(getattr(new_req, "req_id"))

    def _validate_mobilint_vlm_request_constraints(
        self,
        mm_features: Optional[list[MultiModalFeatureSpec]],
        session_id: str,
    ) -> None:
        if not self._is_multimodal_model():
            return
        if not mm_features:
            return

        image_features = []
        for feature in mm_features:
            modality = getattr(feature, "modality", "")
            if modality.startswith("video"):
                raise RuntimeError(
                    "Mobilint Qwen2/3-VL on NPU does not support video inputs. "
                    "Only one initial image is supported; subsequent turns must be text-only."
                )
            if modality.startswith("image"):
                image_features.append(feature)
            else:
                raise RuntimeError(f"Unsupported multimodal modality for Mobilint Qwen2/3-VL on NPU: {modality}")

        if len(image_features) != 1:
            raise RuntimeError(
                "Mobilint Qwen2/3-VL on NPU supports exactly one image in the initial "
                f"request, but got {len(image_features)} image features. Subsequent turns "
                "must be text-only."
            )

        position = self._multimodal_position_signature(image_features[0].mm_position)
        positions_by_session = getattr(self, "_vlm_image_positions_by_session", None)
        if positions_by_session is None:
            positions_by_session = {}
            self._vlm_image_positions_by_session = positions_by_session
        fixed_position = positions_by_session.get(session_id)
        if fixed_position is not None and position != fixed_position:
            raise RuntimeError(
                "Mobilint Qwen2/3-VL on NPU requires a fixed image-token position. "
                f"session_id={session_id}, expected={fixed_position}, got={position}."
            )

        positions_by_session[session_id] = position

    def _get_cache_slot(self, req_id: str) -> int:
        return self.runtime_cache.get_slot(req_id)

    def _assign_cache_slot(self, req_id: str) -> int:
        return self.runtime_cache.assign_slot(req_id)

    def _release_cache_slot(self, req_id: str) -> None:
        self.runtime_cache.release_slot(req_id)

    def _dump_runtime_cache(self, slot_id: Optional[int] = None) -> Optional[list[Any]]:
        cache_model = self._get_cache_model()
        if slot_id is None:
            return cache_model.dump_cache_memory()
        return cache_model.dump_cache_memory(cache_id=slot_id)

    def _load_runtime_cache(self, blobs: list[Any], slot_id: Optional[int] = None) -> bool:
        cache_model = self._get_cache_model()
        if slot_id is None:
            cache_model.load_cache_memory(blobs)
        else:
            cache_model.load_cache_memory(blobs, cache_id=slot_id)
        return True

    def _make_batch_params(
        self,
        sequence_lengths: list[int],
        cache_sizes: list[int],
        cache_ids: list[int],
    ) -> list[BatchParam]:
        if not (len(sequence_lengths) == len(cache_sizes) == len(cache_ids)):
            raise RuntimeError(
                "BatchParam inputs must have identical lengths: "
                f"sequence_lengths={len(sequence_lengths)}, "
                f"cache_sizes={len(cache_sizes)}, cache_ids={len(cache_ids)}"
            )

        return [
            BatchParam(
                sequence_length=sequence_length,
                cache_size=cache_size,
                cache_id=cache_id,
            )
            for sequence_length, cache_size, cache_id in zip(
                sequence_lengths,
                cache_sizes,
                cache_ids,
            )
        ]

    def _normal_batch_chunk_token_cap(self) -> Optional[int]:
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        cap = getattr(scheduler_config, "long_prefill_token_threshold", None)
        if cap is None:
            return None
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            return None
        return cap if cap > 0 else None

    def _to_cpu_float32_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach()
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            tensor = tensor.to(dtype=torch.float32, device="cpu")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor.numpy()

    def _embed_token_ids(self, token_ids: list[int]) -> np.ndarray:
        if not token_ids:
            raise RuntimeError("Cannot embed an empty token slice.")
        if self.input_embeddings is None:
            raise RuntimeError("Input embeddings are not initialized.")
        token_tensor = torch.as_tensor(token_ids, dtype=torch.long)
        token_embeds = self.input_embeddings(token_tensor)
        return self._to_cpu_float32_numpy(token_embeds)

    @staticmethod
    def _extract_multimodal_value(feature: MultiModalFeatureSpec, key: str) -> object:
        item = feature.data
        if item is None or key not in item:
            return None
        value = item[key]
        return getattr(value, "data", value)

    @staticmethod
    def _to_torch_tensor(value: object, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _normalize_grid_thw(value: object | None) -> Optional[torch.Tensor]:
        if value is None:
            return None

        grid_thw = MbltWorker._to_torch_tensor(value, dtype=torch.long)
        if grid_thw.ndim == 1:
            if grid_thw.numel() != 3:
                raise RuntimeError(
                    "Multimodal grid_thw must contain exactly 3 values when "
                    f"1-dimensional, but got shape={tuple(grid_thw.shape)}."
                )
            grid_thw = grid_thw.unsqueeze(0)
        elif grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise RuntimeError(
                f"Multimodal grid_thw must have shape (3,) or (N, 3), but got shape={tuple(grid_thw.shape)}."
            )

        return grid_thw

    @staticmethod
    def _normalize_multimodal_embeddings(embeddings: object) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        pooler_output = getattr(embeddings, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        if isinstance(embeddings, (list, tuple)) and embeddings:
            first = embeddings[0]
            if isinstance(first, torch.Tensor):
                return first
            if isinstance(first, (list, tuple)) and first:
                if not all(isinstance(item, torch.Tensor) for item in first):
                    raise RuntimeError(
                        f"Unsupported nested multimodal embedding output: {[type(item).__name__ for item in first]!r}"
                    )
                if len(first) == 1:
                    return first[0]
                return torch.cat(tuple(first), dim=0)
        raise RuntimeError(f"Unsupported multimodal embedding output: {type(embeddings)!r}")

    @staticmethod
    def _extract_deepstack_embeddings(embeddings: object) -> Optional[list[torch.Tensor]]:
        if not isinstance(embeddings, (list, tuple)) or len(embeddings) < 2:
            return None
        first = embeddings[0]
        deepstack = embeddings[1]
        if not isinstance(first, (list, tuple)):
            return None
        if deepstack is None:
            return None
        if not isinstance(deepstack, (list, tuple)):
            raise RuntimeError(f"Unsupported deepstack multimodal embedding output: {type(deepstack)!r}")
        if not all(isinstance(item, torch.Tensor) for item in deepstack):
            raise RuntimeError(
                f"Unsupported deepstack multimodal embedding tensors: {[type(item).__name__ for item in deepstack]!r}"
            )
        return list(deepstack)

    @staticmethod
    def _scatter_deepstack_embeddings(
        deepstack_prompt_embeds: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        placeholder: PlaceholderRange,
        deepstack_embeds: Optional[list[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if not deepstack_embeds:
            return deepstack_prompt_embeds

        num_layers = len(deepstack_embeds)
        prompt_len = int(prompt_embeds.shape[0])
        hidden_size = int(prompt_embeds.shape[-1])
        if deepstack_prompt_embeds is None:
            deepstack_prompt_embeds = torch.zeros(
                (num_layers, prompt_len, hidden_size),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )

        if int(deepstack_prompt_embeds.shape[0]) != num_layers:
            raise RuntimeError(
                f"Deepstack layer-count mismatch: current={deepstack_prompt_embeds.shape[0]}, new={num_layers}"
            )

        start = int(placeholder.offset)
        end = start + int(placeholder.length)
        if placeholder.is_embed is None:
            target_indices = torch.arange(start, end, device=prompt_embeds.device)
        else:
            mask = placeholder.is_embed.to(device=prompt_embeds.device, dtype=torch.bool)
            target_indices = torch.arange(start, end, device=prompt_embeds.device)[mask]

        expected = int(target_indices.numel())
        for layer_idx, layer_embeds in enumerate(deepstack_embeds):
            layer_embeds = layer_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            if int(layer_embeds.shape[0]) != expected:
                raise RuntimeError(
                    f"Deepstack placeholder length mismatch: expected={expected}, embeds={layer_embeds.shape[0]}"
                )
            deepstack_prompt_embeds[layer_idx, target_indices, :] = layer_embeds

        return deepstack_prompt_embeds

    @staticmethod
    def _scatter_multimodal_embeddings(
        prompt_embeds: torch.Tensor,
        placeholder: PlaceholderRange,
        multimodal_embeds: torch.Tensor,
    ) -> None:
        start = int(placeholder.offset)
        end = start + int(placeholder.length)
        target = prompt_embeds[start:end]
        multimodal_embeds = multimodal_embeds.to(
            device=target.device,
            dtype=target.dtype,
        )

        if placeholder.is_embed is None:
            if multimodal_embeds.shape[0] != target.shape[0]:
                raise RuntimeError(
                    "Multimodal placeholder length mismatch: "
                    f"placeholder={target.shape[0]}, embeds={multimodal_embeds.shape[0]}"
                )
            target.copy_(multimodal_embeds)
            return

        mask = placeholder.is_embed.to(device=target.device, dtype=torch.bool)
        expected = int(mask.sum().item())
        if multimodal_embeds.shape[0] != expected:
            raise RuntimeError(
                f"Multimodal placeholder embed-count mismatch: expected={expected}, embeds={multimodal_embeds.shape[0]}"
            )
        target[mask] = multimodal_embeds

    def _build_prompt_embeds(
        self,
        prompt_token_ids: Optional[list[int]],
        prompt_embeds: Optional[torch.Tensor],
        mm_features: Optional[list[MultiModalFeatureSpec]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if prompt_embeds is not None:
            merged_prompt_embeds = prompt_embeds.clone()
        else:
            if prompt_token_ids is None:
                raise RuntimeError("prompt_token_ids or prompt_embeds must be provided.")
            if self.input_embeddings is None:
                raise RuntimeError("Input embeddings are not initialized.")

            token_tensor = torch.as_tensor(prompt_token_ids, dtype=torch.long)
            merged_prompt_embeds = self.input_embeddings(token_tensor)

        if not mm_features:
            return merged_prompt_embeds, None
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        get_image_features = getattr(self.model, "get_image_features", None)
        get_video_features = getattr(self.model, "get_video_features", None)
        supports_deepstack_input = self._supports_deepstack_input()
        deepstack_prompt_embeds: Optional[torch.Tensor] = None

        for feature in mm_features:
            if feature.data is None:
                continue

            modality = feature.modality
            if modality.startswith("image"):
                if not callable(get_image_features):
                    raise RuntimeError(f"Model {type(self.model).__name__} does not expose get_image_features().")
                pixel_values = self._extract_multimodal_value(feature, "pixel_values")
                image_grid_thw = self._extract_multimodal_value(feature, "image_grid_thw")
                if pixel_values is None:
                    raise RuntimeError("Image multimodal feature is missing pixel_values.")
                image_features = get_image_features(
                    pixel_values=self._to_torch_tensor(pixel_values, dtype=torch.float32),
                    image_grid_thw=self._normalize_grid_thw(image_grid_thw),
                )
                image_embeds = self._normalize_multimodal_embeddings(image_features)
                self._scatter_multimodal_embeddings(
                    merged_prompt_embeds,
                    feature.mm_position,
                    image_embeds,
                )
                if supports_deepstack_input:
                    deepstack_prompt_embeds = self._scatter_deepstack_embeddings(
                        deepstack_prompt_embeds,
                        merged_prompt_embeds,
                        feature.mm_position,
                        self._extract_deepstack_embeddings(image_features),
                    )
            elif modality.startswith("video"):
                if not callable(get_video_features):
                    raise RuntimeError(f"Model {type(self.model).__name__} does not expose get_video_features().")
                pixel_values_videos = self._extract_multimodal_value(feature, "pixel_values_videos")
                video_grid_thw = self._extract_multimodal_value(feature, "video_grid_thw")
                if pixel_values_videos is None:
                    raise RuntimeError("Video multimodal feature is missing pixel_values_videos.")
                video_features = get_video_features(
                    pixel_values_videos=self._to_torch_tensor(pixel_values_videos, dtype=torch.float32),
                    video_grid_thw=self._normalize_grid_thw(video_grid_thw),
                )
                video_embeds = self._normalize_multimodal_embeddings(video_features)
                self._scatter_multimodal_embeddings(
                    merged_prompt_embeds,
                    feature.mm_position,
                    video_embeds,
                )
                if supports_deepstack_input:
                    deepstack_prompt_embeds = self._scatter_deepstack_embeddings(
                        deepstack_prompt_embeds,
                        merged_prompt_embeds,
                        feature.mm_position,
                        self._extract_deepstack_embeddings(video_features),
                    )
            else:
                raise NotImplementedError(f"Unsupported multimodal modality for MBLT worker: {modality}")

        return merged_prompt_embeds, deepstack_prompt_embeds

    def _dump_loaded_request_before_switch(
        self,
        next_req_id: str,
        print_debug: bool = False,
    ) -> None:
        loaded_req_id = self.runtime_cache.loaded_req_id
        if loaded_req_id is None or loaded_req_id == next_req_id:
            return
        loaded_req_state = self.req_states.get(loaded_req_id)
        if loaded_req_state is None:
            return
        if not self.runtime_cache.should_dump_snapshot_after_step(
            loaded_req_id,
            loaded_req_state.num_computed_tokens,
        ):
            return
        self._dump_snapshot(
            req_id=loaded_req_id,
            req_state=loaded_req_state,
            next_num_tokens=loaded_req_state.num_computed_tokens,
            print_debug=print_debug,
        )
        if print_debug:
            print(f"[cache] req={loaded_req_id} dump-before-switch next={next_req_id}")

    def _build_input_embeds(
        self,
        req_state: RequestState,
        start: int,
        end: int,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if end < start:
            raise RuntimeError(f"Invalid token slice: start={start}, end={end}")
        if end == start:
            hidden_size = req_state.prompt_embeds.shape[-1]
            return np.empty((0, hidden_size), dtype=np.float32)

        prompt_len = req_state.prompt_len
        pieces: list[np.ndarray] = []

        prompt_start = min(start, prompt_len)
        prompt_end = min(end, prompt_len)
        if prompt_end > prompt_start:
            pieces.append(req_state.prompt_embeds[prompt_start:prompt_end])

        decode_start = max(start - prompt_len, 0)
        decode_end = max(end - prompt_len, 0)
        if decode_end > decode_start:
            token_ids = req_state.output_token_ids[decode_start:decode_end]
            expected = decode_end - decode_start
            if len(token_ids) != expected:
                raise RuntimeError(
                    "Insufficient decode tokens to rebuild cache miss: "
                    f"expected={expected}, got={len(token_ids)}, "
                    f"start={start}, end={end}, prompt_len={prompt_len}"
                )
            pieces.append(self._embed_token_ids(token_ids))

        if not pieces:
            hidden_size = req_state.prompt_embeds.shape[-1]
            return np.empty((0, hidden_size), dtype=np.float32)
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=0)

    def _build_deepstack_input_embeds(
        self,
        req_state: RequestState,
        start: int,
        end: int,
    ) -> Optional[np.ndarray]:
        prompt_deepstack = req_state.prompt_deepstack_embeds
        if prompt_deepstack is None:
            return None
        if end < start:
            raise RuntimeError(f"Invalid token slice: start={start}, end={end}")
        if end == start:
            return np.empty(
                (prompt_deepstack.shape[0], 0, prompt_deepstack.shape[-1]),
                dtype=np.float32,
            )

        prompt_len = req_state.prompt_len
        pieces: list[np.ndarray] = []
        prompt_start = min(start, prompt_len)
        prompt_end = min(end, prompt_len)
        if prompt_end > prompt_start:
            pieces.append(prompt_deepstack[:, prompt_start:prompt_end, :])

        decode_tokens = max(end - max(start, prompt_len), 0)
        if decode_tokens > 0:
            pieces.append(
                np.zeros(
                    (prompt_deepstack.shape[0], decode_tokens, prompt_deepstack.shape[-1]),
                    dtype=np.float32,
                )
            )

        if not pieces:
            return np.empty(
                (prompt_deepstack.shape[0], 0, prompt_deepstack.shape[-1]),
                dtype=np.float32,
            )
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=1)

    @staticmethod
    def _ensure_batch_vlm_supported(req_state: RequestState) -> None:
        # VLM batch execution is intentionally not implemented yet. Mobilint
        # batch-compiled VLM artifacts are not available at the moment, so the
        # batch path below only supports text-only language-model inputs. Keep
        # this fail-fast guard near the batch scheduling path to avoid silently
        # calling qbruntime with missing or unsupported VLM-specific inputs.
        if getattr(req_state, "is_multimodal", False):
            raise RuntimeError(
                "VLM batch execution is not supported yet. "
                "Batch-compiled VLM artifacts are not available, so run VLM "
                "models with max_batch_size=1 until batch VLM support is added."
            )

    @staticmethod
    def _cache_model_input_shapes(cache_model: Any) -> list[tuple[int, ...]]:
        try:
            if cache_model.get_num_model_variants() <= 0:
                return []
            handle = cache_model.get_model_variant_handle(0)
            return [tuple(shape) for shape in handle.get_model_input_shape()]
        except Exception:
            return []

    def _build_infer_inputs(
        self,
        input_embeds: np.ndarray,
        deepstack_embeds: Optional[np.ndarray],
    ) -> np.ndarray | list[np.ndarray]:
        cache_model = self._get_cache_model()
        input_shapes = self._cache_model_input_shapes(cache_model)
        batched_input = np.expand_dims(input_embeds, axis=0)
        if len(input_shapes) < 2:
            return batched_input
        if not self._supports_deepstack_input():
            if deepstack_embeds is not None:
                raise RuntimeError("Deepstack embeddings are only supported for Qwen3-VL models.")
            return batched_input

        deepstack_shape = input_shapes[1]
        if len(deepstack_shape) != 3:
            raise RuntimeError(
                "Dual-input model deepstack input must have rank 3 "
                f"(layers, sequence, hidden), but got shape={deepstack_shape}."
            )

        expected_layers, expected_seq_len, expected_hidden = deepstack_shape
        if expected_layers <= 0:
            raise RuntimeError(
                "Dual-input model deepstack layer dimension must be fixed and positive, "
                f"but got shape={deepstack_shape}."
            )

        input_seq_len = int(input_embeds.shape[0])
        input_hidden = int(input_embeds.shape[-1])
        if expected_seq_len > 0 and expected_seq_len != input_seq_len:
            raise RuntimeError(
                "Dual-input model deepstack sequence dimension mismatch: "
                f"expected={expected_seq_len}, input_seq_len={input_seq_len}, "
                f"shape={deepstack_shape}."
            )
        if expected_hidden > 0 and expected_hidden != input_hidden:
            raise RuntimeError(
                "Dual-input model deepstack hidden dimension mismatch: "
                f"expected={expected_hidden}, input_hidden={input_hidden}, "
                f"shape={deepstack_shape}."
            )

        if deepstack_embeds is None:
            deepstack_embeds = np.zeros(
                (int(expected_layers), input_seq_len, input_hidden),
                dtype=np.float32,
            )
        else:
            expected_shape = (int(expected_layers), input_seq_len, input_hidden)
            if tuple(deepstack_embeds.shape) != expected_shape:
                raise RuntimeError(
                    "Deepstack embedding shape mismatch for dual-input model: "
                    f"expected={expected_shape}, got={tuple(deepstack_embeds.shape)}."
                )
        return [batched_input, deepstack_embeds.astype(np.float32, copy=False)]

    @staticmethod
    def _last_token_logits(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits)
        if logits.ndim == 3:
            return logits[:, -1, :]
        return logits

    @staticmethod
    def _normalize_sequence_logits(logits: np.ndarray, expected_seq_len: int) -> Optional[np.ndarray]:
        logits = np.asarray(logits)
        if logits.ndim == 3:
            if logits.shape[0] != 1:
                return None
            if logits.shape[1] != expected_seq_len:
                return None
            return logits[0]
        if logits.ndim == 2:
            if logits.shape[0] != expected_seq_len:
                return None
            return logits
        return None

    @staticmethod
    def _shape_dim_matches_sequence(dim: int, input_seq_len: int) -> bool:
        return dim == -1 or dim == input_seq_len

    def _runtime_output_logits_mode(self, input_seq_len: int) -> str:
        """Best-effort classify MXQ logits as full-sequence, last-token, or unknown."""

        cache_model = self._get_cache_model()
        get_output_shape = getattr(cache_model, "get_model_output_shape", None)
        if not callable(get_output_shape):
            return "unknown"
        try:
            output_shapes = get_output_shape()
        except Exception:
            return "unknown"
        if not output_shapes:
            return "unknown"

        output_shape = tuple(output_shapes[0])
        if len(output_shape) == 3:
            if self._is_batch_model():
                return "unknown"
            if self._shape_dim_matches_sequence(int(output_shape[1]), input_seq_len):
                return "full_sequence"
            if input_seq_len > 1 and int(output_shape[1]) == 1:
                return "last_token"
            return "unknown"
        if len(output_shape) == 2:
            if self._is_batch_model():
                return "last_token"
            sequence_dim = int(output_shape[0])
            if sequence_dim == -1:
                return "full_sequence"
            if sequence_dim == 1:
                return "last_token"
            if sequence_dim == input_seq_len:
                return "full_sequence"
            return "last_token"
        return "unknown"

    def _normalize_runtime_sequence_logits(self, logits: np.ndarray, expected_seq_len: int) -> Optional[np.ndarray]:
        """Return per-position logits only when the MXQ output is known to contain them.

        OpenAI completions with ``echo=true`` and ``logprobs=N`` require prompt
        token ``i`` to be scored from logits at prompt position ``i - 1``.  A
        last-token-only MXQ can return a 2-D tensor with shape ``(1, vocab)``;
        for Batch1/single-core requests this may accidentally match a
        one-token scheduled/microstep input and looks like valid sequence
        logits if we check the tensor shape alone.  Treat known last-token
        runtime outputs as unavailable for prompt-logprob sequence scoring so
        callers use the dedicated prompt-logprob microstep/fallback path instead
        of reusing the same final-position distribution for prompt tokens.
        """

        mode = self._runtime_output_logits_mode(expected_seq_len)
        if mode != "full_sequence":
            return None
        return self._normalize_sequence_logits(logits, expected_seq_len=expected_seq_len)

    def _needs_last_logit_prompt_logprob_microsteps(
        self,
        req_state: RequestState,
        sequence_length: int,
    ) -> bool:
        if self._num_prompt_logprobs(req_state.sampling_params) is None:
            return False
        if sequence_length <= 0:
            return False
        mode = self._runtime_output_logits_mode(sequence_length)
        return mode != "full_sequence"

    def _warn_last_logit_prompt_logprobs_once(self) -> None:
        if getattr(self, "_warned_last_logit_prompt_logprobs", False):
            return
        logger.warning(
            "Prompt logprobs on last-logit MBLT/MXQ outputs use a slower 1-token microstep path. "
            "Compile or use a full-logits MXQ for faster prompt logprob serving."
        )
        self._warned_last_logit_prompt_logprobs = True

    @staticmethod
    def _can_reuse_output_buffers(
        cache_model: Any,
        output_buffers: list[np.ndarray],
        input_seq_len: int,
    ) -> bool:
        if not output_buffers:
            return False
        get_output_shape = getattr(cache_model, "get_model_output_shape", None)
        if not callable(get_output_shape):
            return True
        try:
            output_shapes = get_output_shape()
        except Exception:
            return True
        if not output_shapes:
            return True

        output_shape = tuple(output_shapes[0])
        if any(dim == -1 for dim in output_shape):
            return False
        if len(output_shape) != output_buffers[0].ndim:
            return True

        expected_shape = tuple(input_seq_len if dim == -1 else dim for dim in output_shape)
        return output_buffers[0].shape == expected_shape

    def _infer_logits(
        self,
        input_embeds: np.ndarray,
        deepstack_embeds: Optional[np.ndarray],
        cache_size: int,
    ) -> np.ndarray:
        cache_model = self._get_cache_model()
        infer_inputs = self._build_infer_inputs(input_embeds, deepstack_embeds)
        output_buffers = self._infer_output_buffers

        if output_buffers is not None and self._can_reuse_output_buffers(
            cache_model,
            output_buffers,
            input_seq_len=int(input_embeds.shape[0]),
        ):
            infer_output = cache_model.infer(
                infer_inputs,
                outputs=output_buffers,
                cache_size=cache_size,
            )
            if infer_output is None:
                return self._last_token_logits(output_buffers[0])
            return self._last_token_logits(infer_output[0])

        infer_output = cache_model.infer(infer_inputs, cache_size=cache_size)
        if infer_output is None:
            raise RuntimeError("mxq infer result is None!")

        logits = infer_output[0]
        self._infer_output_buffers = [np.empty_like(logits)]
        return self._last_token_logits(logits)

    def _infer_logits_with_sequence(
        self,
        input_embeds: np.ndarray,
        deepstack_embeds: Optional[np.ndarray],
        cache_size: int,
    ) -> InferenceLogits:
        cache_model = self._get_cache_model()
        infer_inputs = self._build_infer_inputs(input_embeds, deepstack_embeds)
        output_buffers = self._infer_output_buffers

        if output_buffers is not None and self._can_reuse_output_buffers(
            cache_model,
            output_buffers,
            input_seq_len=int(input_embeds.shape[0]),
        ):
            infer_output = cache_model.infer(
                infer_inputs,
                outputs=output_buffers,
                cache_size=cache_size,
            )
            logits = output_buffers[0] if infer_output is None else infer_output[0]
        else:
            infer_output = cache_model.infer(infer_inputs, cache_size=cache_size)
            if infer_output is None:
                raise RuntimeError("mxq infer result is None!")
            logits = infer_output[0]
            self._infer_output_buffers = [np.empty_like(logits)]

        logits_np = np.asarray(logits)
        return InferenceLogits(
            last_token_logits=self._last_token_logits(logits_np),
            full_sequence_logits=self._normalize_runtime_sequence_logits(
                logits_np, expected_seq_len=int(input_embeds.shape[0])
            ),
        )

    def _infer_logits_batch(
        self,
        input_embeds_batch: list[np.ndarray],
        cache_sizes: list[int],
        cache_ids: list[int],
    ) -> list[np.ndarray]:
        if not input_embeds_batch:
            return []

        cache_model = self._get_cache_model()
        batch_size = len(input_embeds_batch)
        params = self._make_batch_params(
            sequence_lengths=[int(input_embeds.shape[0]) for input_embeds in input_embeds_batch],
            cache_sizes=cache_sizes,
            cache_ids=cache_ids,
        )

        concat_input = np.concatenate(input_embeds_batch, axis=0).astype(
            np.float32,
            copy=False,
        )
        while concat_input.ndim < 4:
            concat_input = np.expand_dims(concat_input, axis=0)

        infer_output = cache_model.infer([concat_input], params=params)

        logits = infer_output[0] if isinstance(infer_output, (list, tuple)) else infer_output
        logits_np = np.asarray(logits)
        if logits_np.ndim == 3:
            offset = 0
            last_token_logits: list[np.ndarray] = []
            for input_embeds in input_embeds_batch:
                seq_len = int(input_embeds.shape[0])
                if seq_len <= 0:
                    raise RuntimeError("Batched infer received an empty input embedding slice.")
                last_token_logits.append(logits_np[0, offset + seq_len - 1, :])
                offset += seq_len
            if offset != logits_np.shape[1]:
                raise RuntimeError(
                    "Batched infer returned logits with unexpected sequence length: "
                    f"shape={logits_np.shape}, expected_tokens={offset}"
                )
            return last_token_logits
        if logits_np.size % batch_size != 0:
            raise RuntimeError(
                f"Batched infer returned logits with unexpected shape: shape={logits_np.shape}, batch_size={batch_size}"
            )
        logits_np = logits_np.reshape(batch_size, -1)
        return [logits_np[i] for i in range(batch_size)]

    def _infer_logits_batch_with_sequence(
        self,
        input_embeds_batch: list[np.ndarray],
        cache_sizes: list[int],
        cache_ids: list[int],
    ) -> list[InferenceLogits]:
        if not input_embeds_batch:
            return []

        cache_model = self._get_cache_model()
        batch_size = len(input_embeds_batch)
        params = self._make_batch_params(
            sequence_lengths=[int(input_embeds.shape[0]) for input_embeds in input_embeds_batch],
            cache_sizes=cache_sizes,
            cache_ids=cache_ids,
        )

        concat_input = np.concatenate(input_embeds_batch, axis=0).astype(
            np.float32,
            copy=False,
        )
        while concat_input.ndim < 4:
            concat_input = np.expand_dims(concat_input, axis=0)

        infer_output = cache_model.infer([concat_input], params=params)
        logits = infer_output[0] if isinstance(infer_output, (list, tuple)) else infer_output
        logits_np = np.asarray(logits)
        if logits_np.ndim == 3:
            offset = 0
            outputs: list[InferenceLogits] = []
            for input_embeds in input_embeds_batch:
                seq_len = int(input_embeds.shape[0])
                if seq_len <= 0:
                    raise RuntimeError("Batched infer received an empty input embedding slice.")
                sequence_logits = logits_np[0, offset : offset + seq_len, :]
                outputs.append(
                    InferenceLogits(
                        last_token_logits=sequence_logits[-1, :],
                        full_sequence_logits=sequence_logits,
                    )
                )
                offset += seq_len
            if offset != logits_np.shape[1]:
                raise RuntimeError(
                    "Batched infer returned logits with unexpected sequence length: "
                    f"shape={logits_np.shape}, expected_tokens={offset}"
                )
            return outputs

        if logits_np.size % batch_size != 0:
            raise RuntimeError(
                f"Batched infer returned logits with unexpected shape: shape={logits_np.shape}, batch_size={batch_size}"
            )
        logits_np = logits_np.reshape(batch_size, -1)
        return [InferenceLogits(last_token_logits=logits_np[i], full_sequence_logits=None) for i in range(batch_size)]

    def _infer_normal_logits_batch_chunked(
        self,
        output_indices: list[int],
        input_embeds_batch: list[np.ndarray],
        cache_sizes: list[int],
        cache_ids: list[int],
    ) -> dict[int, InferenceLogits]:
        if not output_indices:
            return {}
        if not (
            len(output_indices) == len(input_embeds_batch) == len(cache_sizes) == len(cache_ids)
        ):
            raise RuntimeError(
                "Normal batch chunk inputs must have identical lengths: "
                f"indices={len(output_indices)}, input_embeds={len(input_embeds_batch)}, "
                f"cache_sizes={len(cache_sizes)}, cache_ids={len(cache_ids)}"
            )

        token_cap = self._normal_batch_chunk_token_cap()
        states = [
            NormalBatchChunkState(
                output_index=output_index,
                input_embeds=input_embeds,
                cache_size=cache_size,
                cache_id=cache_id,
                full_sequence_logits=[],
            )
            for output_index, input_embeds, cache_size, cache_id in zip(
                output_indices,
                input_embeds_batch,
                cache_sizes,
                cache_ids,
            )
        ]

        while True:
            active_states = [state for state in states if state.offset < int(state.input_embeds.shape[0])]
            if not active_states:
                break

            for batch_start in range(0, len(active_states), self.max_batch_size):
                batch_states = active_states[batch_start : batch_start + self.max_batch_size]
                chunk_embeds_batch: list[np.ndarray] = []
                chunk_cache_sizes: list[int] = []
                chunk_cache_ids: list[int] = []
                for state in batch_states:
                    remaining = int(state.input_embeds.shape[0]) - state.offset
                    chunk_len = remaining if token_cap is None else min(remaining, token_cap)
                    if chunk_len <= 0:
                        continue
                    chunk_start = state.offset
                    chunk_end = chunk_start + chunk_len
                    chunk_embeds_batch.append(state.input_embeds[chunk_start:chunk_end])
                    chunk_cache_sizes.append(state.cache_size)
                    chunk_cache_ids.append(state.cache_id)

                if not chunk_embeds_batch:
                    continue

                logits_batch = self._infer_logits_batch_with_sequence(
                    input_embeds_batch=chunk_embeds_batch,
                    cache_sizes=chunk_cache_sizes,
                    cache_ids=chunk_cache_ids,
                )
                for state, chunk_embeds, inference_logits in zip(batch_states, chunk_embeds_batch, logits_batch):
                    chunk_len = int(chunk_embeds.shape[0])
                    state.offset += chunk_len
                    state.cache_size += chunk_len
                    state.last_token_logits = inference_logits.last_token_logits
                    if inference_logits.full_sequence_logits is None:
                        state.full_sequence_logits = None
                    elif state.full_sequence_logits is not None:
                        state.full_sequence_logits.append(inference_logits.full_sequence_logits)

        outputs: dict[int, InferenceLogits] = {}
        for state in states:
            if state.last_token_logits is None:
                raise RuntimeError(f"Missing normal batched logits for output_index={state.output_index}.")
            full_sequence_logits = None
            if state.full_sequence_logits is not None:
                full_sequence_logits = (
                    np.concatenate(state.full_sequence_logits, axis=0)
                    if state.full_sequence_logits
                    else np.empty((0, int(state.last_token_logits.shape[0])), dtype=state.last_token_logits.dtype)
                )
            outputs[state.output_index] = InferenceLogits(
                last_token_logits=state.last_token_logits,
                full_sequence_logits=full_sequence_logits,
            )
        return outputs

    def _num_prompt_logprobs(self, sampling_params: SamplingParams) -> Optional[int]:
        num_prompt_logprobs = sampling_params.prompt_logprobs
        if num_prompt_logprobs is None:
            return None
        if num_prompt_logprobs == -1:
            if self.model is None:
                raise RuntimeError("Model is not initialized.")
            return int(self.model.config.vocab_size)
        return int(num_prompt_logprobs)

    def _should_recompute_prompt_logprobs_from_start(self, req_state: RequestState) -> bool:
        return (
            self._num_prompt_logprobs(req_state.sampling_params) is not None
            and req_state.next_prompt_logprob_pos <= 1
            and req_state.next_prompt_logprob_pos < len(req_state.prompt_token_ids)
            and req_state.num_computed_tokens > 0
        )

    def _get_prompt_logprobs_tensors(
        self,
        req_state: RequestState,
        sequence_logits: Optional[np.ndarray],
        start_idx: int,
        scheduled_end: int,
    ):
        num_prompt_logprobs = self._num_prompt_logprobs(req_state.sampling_params)
        if num_prompt_logprobs is None:
            return None

        prompt_token_ids = req_state.prompt_token_ids
        num_prompt_tokens = len(prompt_token_ids)
        if num_prompt_tokens <= 1:
            req_state.next_prompt_logprob_pos = num_prompt_tokens
            from vllm.v1.outputs import LogprobsTensors

            return LogprobsTensors.empty_cpu(0, num_prompt_logprobs + 1)

        if sequence_logits is None:
            logger.warning_once(
                "Prompt logprobs were requested, but the MBLT runtime returned only last-token logits. "
                "Use _get_prompt_logprobs_tensors_with_fallback to compute prompt token logprobs."
            )
            req_state.next_prompt_logprob_pos = num_prompt_tokens
            return None

        sequence_logits = np.asarray(sequence_logits)
        if sequence_logits.ndim != 2:
            logger.warning_once(
                "Prompt logprobs were requested, but MBLT logits have unsupported shape %s. "
                "Prompt token logprobs will be omitted for this request.",
                sequence_logits.shape,
            )
            req_state.next_prompt_logprob_pos = num_prompt_tokens
            return None

        prompt_end = min(scheduled_end + 1, num_prompt_tokens)
        # For prompt position i > 0, use logits produced while processing token
        # i - 1. There is intentionally no logprob for prompt token 0; vLLM's
        # LogprobsProcessor seeds prompt_logprobs with a leading None.
        first_prompt_pos = max(1, start_idx + 1, req_state.next_prompt_logprob_pos)
        if prompt_end <= first_prompt_pos:
            return None

        logits_start = first_prompt_pos - start_idx - 1
        logits_end = logits_start + (prompt_end - first_prompt_pos)
        if logits_start < 0 or logits_end > sequence_logits.shape[0]:
            logger.warning_once(
                "Prompt logprobs were requested, but MBLT logits shape %s does not cover prompt positions "
                "[%s, %s). Prompt token logprobs will be omitted for this request.",
                sequence_logits.shape,
                first_prompt_pos,
                prompt_end,
            )
            req_state.next_prompt_logprob_pos = num_prompt_tokens
            return None

        prompt_logits = torch.from_numpy(sequence_logits[logits_start:logits_end]).to(dtype=torch.float32)
        target_token_ids = torch.as_tensor(prompt_token_ids[first_prompt_pos:prompt_end], dtype=torch.int64)
        logprobs = self.sampler.compute_logprobs(prompt_logits)
        req_state.next_prompt_logprob_pos = prompt_end
        return self.sampler.gather_logprobs(logprobs, num_prompt_logprobs, target_token_ids)

    def _compute_prompt_logprobs_sequence_logits_fallback(
        self,
        req_state: RequestState,
        start_idx: int,
        scheduled_end: int,
        cache_id: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Compute prompt-position logits when the runtime exposes only last-token logits.

        vLLM/OpenAI completions with ``echo=true`` and ``logprobs=N`` need
        log P(t_i | t_0...t_{i-1}) for prompt tokens as well as generated
        tokens.  Some MBLT artifacts return only the last-token logits for an
        inference call, so a normal full-prompt prefill cannot directly provide
        all prompt positions.  This correctness-first fallback incrementally
        replays the prompt in an empty/isolated fallback cache and uses each
        step's last-token logits as the distribution for the next prompt token.

        The caller still runs the scheduled prefill/decode afterwards, so the
        generation cache and generated-token logits are kept on the existing
        fast path.  This method is invoked only for requests that explicitly
        ask for prompt logprobs and only when full sequence logits are absent.
        """

        if self._num_prompt_logprobs(req_state.sampling_params) is None:
            return None

        prompt_token_ids = req_state.prompt_token_ids
        num_prompt_tokens = len(prompt_token_ids)
        if req_state.next_prompt_logprob_pos >= num_prompt_tokens:
            return None
        if num_prompt_tokens <= 1:
            return None

        prompt_end = min(scheduled_end + 1, num_prompt_tokens)
        first_prompt_pos = max(1, start_idx + 1, req_state.next_prompt_logprob_pos)
        if prompt_end <= first_prompt_pos:
            return None

        # _get_prompt_logprobs_tensors maps prompt position i to row
        # i - start_idx - 1.  Therefore the fallback sequence logits must start
        # at prompt position start_idx + 1, not first_prompt_pos.  Some prompt
        # positions in that span may already have been emitted by an earlier
        # chunk/replay, but we still include their rows so the tensor layout is
        # identical to full-sequence runtime output.  _get_prompt_logprobs_tensors
        # will slice away already-emitted positions using next_prompt_logprob_pos.
        #
        # Feed only new embeddings after the first fallback step and advance the
        # runtime cache size.  This computes log P(t_i | t_0...t_{i-1}) with
        # O(N) submitted embeddings instead of replaying every full prefix.
        rows: list[np.ndarray] = []
        fallback_cache_size = 0
        for prompt_pos in range(start_idx + 1, prompt_end):
            if fallback_cache_size == 0:
                # Prime the fallback cache with the prefix that predicts
                # prompt_token_ids[prompt_pos].  For the common start_idx == 0
                # case this is exactly one token; for a nonzero start_idx it is
                # one linear prefix-fill call, not one call per prefix length.
                input_start = 0
                input_end = prompt_pos
            else:
                # The fallback cache already contains prompt_pos - 1 tokens;
                # feed only the next new token to obtain logits for prompt_pos.
                input_start = prompt_pos - 1
                input_end = prompt_pos

            input_embeds = req_state.prompt_embeds[input_start:input_end]
            input_deepstack = (
                req_state.prompt_deepstack_embeds[:, input_start:input_end, :]
                if req_state.prompt_deepstack_embeds is not None
                else None
            )
            if cache_id is not None:
                if input_deepstack is not None:
                    raise RuntimeError("Batch prompt-logprob fallback does not support deepstack embeddings.")
                logits = self._infer_logits_batch(
                    input_embeds_batch=[input_embeds],
                    cache_sizes=[fallback_cache_size],
                    cache_ids=[cache_id],
                )[0]
            else:
                logits = self._infer_logits(input_embeds, input_deepstack, cache_size=fallback_cache_size)
            # _infer_logits may return a view into qbruntime's reusable output
            # buffer.  Prompt logprob replay/microstep paths keep one row per
            # prompt position, so each row must be detached immediately.  If we
            # store the view, later infer() calls overwrite the same memory and
            # every echoed prompt position can become identical to the final
            # prefill/decode distribution (for example repeated "Hello" or
            # " Paris" top-1 rows).
            rows.append(np.asarray(self._last_token_logits(logits)).reshape(-1).copy())
            fallback_cache_size += int(input_embeds.shape[0])

        if not rows:
            return None
        return np.stack(rows, axis=0)

    def _make_prompt_logprobs_fallback_replay_state(
        self,
        output_index: int,
        req_state: RequestState,
        start_idx: int,
        scheduled_end: int,
        cache_id: int,
    ) -> Optional[PromptLogprobFallbackReplayState]:
        if self._num_prompt_logprobs(req_state.sampling_params) is None:
            return None

        prompt_token_ids = req_state.prompt_token_ids
        num_prompt_tokens = len(prompt_token_ids)
        if req_state.next_prompt_logprob_pos >= num_prompt_tokens:
            return None
        if num_prompt_tokens <= 1:
            return None

        prompt_end = min(scheduled_end + 1, num_prompt_tokens)
        first_prompt_pos = max(1, start_idx + 1, req_state.next_prompt_logprob_pos)
        if prompt_end <= first_prompt_pos:
            return None
        if req_state.prompt_deepstack_embeds is not None:
            raise RuntimeError("Batch prompt-logprob fallback does not support deepstack embeddings.")

        return PromptLogprobFallbackReplayState(
            output_index=output_index,
            req_state=req_state,
            start_idx=start_idx,
            prompt_end=prompt_end,
            cache_id=cache_id,
            current_prompt_pos=start_idx + 1,
            fallback_cache_size=0,
            rows=[],
        )

    def _compute_prompt_logprobs_sequence_logits_fallback_batch(
        self,
        requests: list[tuple[int, RequestState, int, int, int]],
    ) -> dict[int, np.ndarray]:
        """Batch prompt-logprob fallback replay across requests.

        Each active request owns an isolated qbruntime cache_id.  The replay
        cache starts empty and advances one prompt position at a time, but the
        work for different requests is submitted together through BatchParam.
        """

        states = [
            state
            for output_index, req_state, start_idx, scheduled_end, cache_id in requests
            if (
                state := self._make_prompt_logprobs_fallback_replay_state(
                    output_index=output_index,
                    req_state=req_state,
                    start_idx=start_idx,
                    scheduled_end=scheduled_end,
                    cache_id=cache_id,
                )
            )
            is not None
        ]
        if not states:
            return {}

        while True:
            active_states = [state for state in states if state.current_prompt_pos < state.prompt_end]
            if not active_states:
                break

            for batch_start in range(0, len(active_states), self.max_batch_size):
                batch_states = active_states[batch_start : batch_start + self.max_batch_size]
                input_embeds_batch: list[np.ndarray] = []
                cache_sizes: list[int] = []
                cache_ids: list[int] = []

                for state in batch_states:
                    prompt_pos = state.current_prompt_pos
                    if state.fallback_cache_size == 0:
                        input_start = 0
                        input_end = prompt_pos
                    else:
                        input_start = prompt_pos - 1
                        input_end = prompt_pos

                    input_embeds = state.req_state.prompt_embeds[input_start:input_end]
                    input_embeds_batch.append(input_embeds)
                    cache_sizes.append(state.fallback_cache_size)
                    cache_ids.append(state.cache_id)

                logits_batch = self._infer_logits_batch(
                    input_embeds_batch=input_embeds_batch,
                    cache_sizes=cache_sizes,
                    cache_ids=cache_ids,
                )

                for state, input_embeds, logits in zip(batch_states, input_embeds_batch, logits_batch):
                    # Detach from any backend-owned/reused logits buffer before
                    # the next microstep overwrites it.
                    state.rows.append(np.asarray(self._last_token_logits(logits)).reshape(-1).copy())
                    state.fallback_cache_size += int(input_embeds.shape[0])
                    state.current_prompt_pos += 1

        return {
            state.output_index: np.stack(state.rows, axis=0)
            for state in states
            if state.rows
        }

    def _run_prompt_logprob_microsteps_batch(
        self,
        requests: list[tuple[int, str, RequestState, int, int, int]],
    ) -> dict[int, InferenceLogits]:
        """Run scheduled prompt-logprob requests as 1-token batched steps.

        Last-token-only MXQs cannot produce the per-position prompt logits from
        a multi-token prefill.  This path advances the live request cache one
        token at a time, preserving the final sampling logits while collecting
        prompt-position logits from the same scheduled range.
        """

        states = [
            PromptLogprobMicrostepState(
                output_index=output_index,
                req_id=req_id,
                req_state=req_state,
                start_idx=start_idx,
                scheduled_end=scheduled_end,
                cache_id=cache_id,
                cache_size=start_idx,
                prompt_logits_end=min(scheduled_end + 1, req_state.prompt_len),
                rows=[],
            )
            for output_index, req_id, req_state, start_idx, scheduled_end, cache_id in requests
            if scheduled_end > start_idx
        ]
        if not states:
            return {}

        if self.print_debug:
            logger.info(
                "[mblt-batch-debug] prompt-logprob microsteps: state_count=%d "
                "max_batch_size=%d ranges=%s cache_sizes=%s",
                len(states),
                self.max_batch_size,
                [
                    (state.req_id, state.start_idx, state.scheduled_end, state.prompt_logits_end)
                    for state in states
                ],
                [state.cache_size for state in states],
            )
        self._warn_last_logit_prompt_logprobs_once()

        while True:
            active_states = [state for state in states if state.cache_size < state.scheduled_end]
            if not active_states:
                break

            for batch_start in range(0, len(active_states), self.max_batch_size):
                batch_states = active_states[batch_start : batch_start + self.max_batch_size]
                input_embeds_batch = [
                    self._build_input_embeds(state.req_state, state.cache_size, state.cache_size + 1)
                    for state in batch_states
                ]
                cache_sizes = [state.cache_size for state in batch_states]
                cache_ids = [state.cache_id for state in batch_states]

                logits_batch = self._infer_logits_batch(
                    input_embeds_batch=input_embeds_batch,
                    cache_sizes=cache_sizes,
                    cache_ids=cache_ids,
                )

                for state, logits in zip(batch_states, logits_batch):
                    # Keep a copy of the position-specific logits row.  Some
                    # runtimes reuse the same output buffer for every infer()
                    # call; storing a view would broadcast the last step's
                    # logits across all prompt echo logprob positions.
                    logits_row = np.asarray(self._last_token_logits(logits)).reshape(-1).copy()
                    prompt_pos = state.cache_size + 1
                    if prompt_pos < state.prompt_logits_end:
                        state.rows.append(logits_row)
                    state.last_token_logits = logits_row
                    state.cache_size += 1

        outputs: dict[int, InferenceLogits] = {}
        for state in states:
            if state.last_token_logits is None:
                continue
            full_sequence_logits = (
                np.stack(state.rows, axis=0)
                if state.rows
                else np.empty((0, int(state.last_token_logits.shape[0])), dtype=state.last_token_logits.dtype)
            )
            outputs[state.output_index] = InferenceLogits(
                last_token_logits=state.last_token_logits,
                full_sequence_logits=full_sequence_logits,
            )
        return outputs

    def _run_prompt_logprob_microsteps_single(
        self,
        req_id: str,
        req_state: RequestState,
        start_idx: int,
        scheduled_end: int,
    ) -> Optional[InferenceLogits]:
        """Run a nonbatch scheduled range as 1-token steps for prompt logprobs.

        This is the single-core equivalent of
        :meth:`_run_prompt_logprob_microsteps_batch`.  Last-token-only MXQs
        cannot return per-position prompt logits from a multi-token prefill, so
        OpenAI ``echo=true`` prompt logprobs must advance the live request cache
        one token at a time and collect the logits from token ``i - 1`` to score
        prompt token ``i``.  The final microstep's logits are also the normal
        sampling logits, which preserves generated-token logprob behavior.
        """

        if scheduled_end <= start_idx:
            return None

        cache_size = start_idx
        prompt_logits_end = min(scheduled_end + 1, req_state.prompt_len)
        rows: list[np.ndarray] = []
        last_token_logits: Optional[np.ndarray] = None

        if self.print_debug:
            logger.info(
                "[mblt-debug] prompt-logprob single microsteps: req_id=%s range=(%d,%d) "
                "prompt_logits_end=%d next_prompt_logprob_pos=%d",
                req_id,
                start_idx,
                scheduled_end,
                prompt_logits_end,
                req_state.next_prompt_logprob_pos,
            )
        self._warn_last_logit_prompt_logprobs_once()

        while cache_size < scheduled_end:
            input_embeds = self._build_input_embeds(req_state, cache_size, cache_size + 1)
            deepstack_embeds = self._build_deepstack_input_embeds(req_state, cache_size, cache_size + 1)
            logits = self._infer_logits(input_embeds, deepstack_embeds, cache_size=cache_size)
            # Keep a copy of the position-specific logits row.  _infer_logits
            # can return a view into self._infer_output_buffers/qbruntime output
            # memory, which is overwritten by the next infer() call.
            logits_row = np.asarray(self._last_token_logits(logits)).reshape(-1).copy()

            # The logits emitted after consuming token at cache_size predict the
            # next token at prompt_pos=cache_size + 1.  There is no prompt
            # logprob for prompt token 0, and logits that predict the first
            # generated token are kept only as last_token_logits for sampling.
            prompt_pos = cache_size + 1
            if prompt_pos < prompt_logits_end:
                rows.append(logits_row)
            last_token_logits = logits_row
            cache_size += 1

        if last_token_logits is None:
            return None
        full_sequence_logits = (
            np.stack(rows, axis=0)
            if rows
            else np.empty((0, int(last_token_logits.shape[0])), dtype=last_token_logits.dtype)
        )
        return InferenceLogits(
            last_token_logits=last_token_logits,
            full_sequence_logits=full_sequence_logits,
        )

    def _get_prompt_logprobs_tensors_with_fallback(
        self,
        req_state: RequestState,
        sequence_logits: Optional[np.ndarray],
        start_idx: int,
        scheduled_end: int,
        cache_id: Optional[int] = None,
    ):
        if sequence_logits is None:
            fallback_logits = self._compute_prompt_logprobs_sequence_logits_fallback(
                req_state=req_state,
                start_idx=start_idx,
                scheduled_end=scheduled_end,
                cache_id=cache_id,
            )
            if fallback_logits is None:
                return None
            return self._get_prompt_logprobs_tensors(
                req_state=req_state,
                sequence_logits=fallback_logits,
                start_idx=start_idx,
                scheduled_end=scheduled_end,
            )

        prompt_logprobs_tensors = self._get_prompt_logprobs_tensors(
            req_state=req_state,
            sequence_logits=sequence_logits,
            start_idx=start_idx,
            scheduled_end=scheduled_end,
        )
        return prompt_logprobs_tensors

    def _accumulate_prompt_logprobs_tensors(
        self,
        req_state: RequestState,
        prompt_logprobs_tensors: Optional[LogprobsTensors],
        first_prompt_pos: int,
    ) -> None:
        if prompt_logprobs_tensors is None:
            return

        num_prompt_logprobs = self._num_prompt_logprobs(req_state.sampling_params)
        if num_prompt_logprobs is None:
            return

        num_rows = int(prompt_logprobs_tensors.logprob_token_ids.shape[0])
        if req_state.in_progress_prompt_logprobs is None:
            req_state.in_progress_prompt_logprobs = LogprobsTensors.empty_cpu(
                max(0, len(req_state.prompt_token_ids) - 1),
                num_prompt_logprobs + 1,
            )
        if num_rows <= 0:
            return

        dst_start = first_prompt_pos - 1
        dst_end = dst_start + num_rows
        in_progress = req_state.in_progress_prompt_logprobs
        in_progress.logprob_token_ids[dst_start:dst_end].copy_(prompt_logprobs_tensors.logprob_token_ids)
        in_progress.logprobs[dst_start:dst_end].copy_(prompt_logprobs_tensors.logprobs)
        in_progress.selected_token_ranks[dst_start:dst_end].copy_(prompt_logprobs_tensors.selected_token_ranks)

    def _get_completed_prompt_logprobs_tensors_for_scheduler(
        self,
        req_state: RequestState,
        sequence_logits: Optional[np.ndarray],
        start_idx: int,
        scheduled_end: int,
        cache_id: Optional[int] = None,
        can_emit_output: bool = True,
    ) -> Optional[LogprobsTensors]:
        num_prompt_logprobs = self._num_prompt_logprobs(req_state.sampling_params)
        if num_prompt_logprobs is None:
            return None

        first_prompt_pos = max(1, start_idx + 1, req_state.next_prompt_logprob_pos)
        prompt_logprobs_tensors = self._get_prompt_logprobs_tensors_with_fallback(
            req_state=req_state,
            sequence_logits=sequence_logits,
            start_idx=start_idx,
            scheduled_end=scheduled_end,
            cache_id=cache_id,
        )
        self._accumulate_prompt_logprobs_tensors(
            req_state=req_state,
            prompt_logprobs_tensors=prompt_logprobs_tensors,
            first_prompt_pos=first_prompt_pos,
        )

        if scheduled_end < req_state.prompt_len:
            return None

        return self._pop_completed_prompt_logprobs_tensors_for_scheduler(
            req_state=req_state,
            can_emit_output=can_emit_output,
        )

    def _pop_completed_prompt_logprobs_tensors_for_scheduler(
        self,
        req_state: RequestState,
        can_emit_output: bool,
    ) -> Optional[LogprobsTensors]:
        if not can_emit_output:
            return None
        if self._num_prompt_logprobs(req_state.sampling_params) is None:
            return None
        if req_state.next_prompt_logprob_pos < len(req_state.prompt_token_ids):
            return None
        completed_prompt_logprobs = req_state.in_progress_prompt_logprobs
        req_state.in_progress_prompt_logprobs = None
        return completed_prompt_logprobs

    @staticmethod
    def _should_sample_after_step(
        req_state: RequestState,
        scheduled_end: int,
        sequence_length: int,
    ) -> bool:
        if sequence_length <= 0:
            return False
        return scheduled_end >= req_state.prompt_len

    def _sample_next_token(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata):
        # OpenAI-compatible logprobs must be log-softmax-normalized
        # probabilities, not raw logits. Keep the sampler's raw-logits default
        # for generation behavior, but request normalized raw logprobs for the
        # optional generated-token logprob tensors. The sampler only computes
        # these when max_num_logprobs is not None, so normal generation without
        # logprobs does not pay the log_softmax cost.
        return self.sampler.forward(
            logits=logits,
            sampling_metadata=sampling_metadata,
            logprobs_mode_override="raw_logprobs",
        )

    def _load_snapshot_if_needed(
        self,
        req_id: str,
        req_state: RequestState,
        slot_id: Optional[int] = None,
        print_debug: bool = False,
    ) -> int:
        if self._is_batch_model():
            return self._load_snapshot_for_batch_slot(
                req_id=req_id,
                req_state=req_state,
                slot_id=slot_id,
                print_debug=print_debug,
            )

        self._dump_loaded_request_before_switch(
            next_req_id=req_id,
            print_debug=print_debug,
        )

        if self._should_recompute_prompt_logprobs_from_start(req_state):
            self.runtime_cache.clear_loaded_request()
            if print_debug:
                print(f"[cache] req={req_id} bypass-prefix-cache-for-prompt-logprobs")
            return 0

        target_tokens = req_state.num_computed_tokens
        result = self.runtime_cache.load_for_request(
            RuntimeCacheRequest(
                req_id=req_id,
                block_ids=req_state.block_ids,
                first_seq_blocks=req_state.first_seq_blocks,
                num_computed_tokens=target_tokens,
                cache_slot_id=slot_id,
                cache_token_ids=self._cache_token_ids(req_state, target_tokens),
            )
        )
        if print_debug:
            if result.action == "skip-empty":
                print(f"[cache] req={req_id} skip-load target_tokens=0")
            elif result.action == "reuse-live":
                print(f"[cache] req={req_id} reuse-live-cache matched={target_tokens}/{target_tokens}")
            elif result.action == "load-own":
                print(f"[cache] req={req_id} load-own matched={result.matched_tokens}/{target_tokens}")
            elif result.action == "load-shared":
                print(f"[cache] req={req_id} load-shared matched={result.matched_tokens}/{target_tokens}")
            else:
                print(f"[cache] req={req_id} cache-miss fallback matched=0/{target_tokens}")
        return result.matched_tokens

    def _load_snapshot_for_batch_slot(
        self,
        req_id: str,
        req_state: RequestState,
        slot_id: Optional[int],
        print_debug: bool = False,
    ) -> int:
        if slot_id is None:
            raise RuntimeError(f"Batch execution requires a cache slot for req_id={req_id}.")

        if self._should_recompute_prompt_logprobs_from_start(req_state):
            if print_debug:
                print(f"[cache] req={req_id} slot={slot_id} bypass-prefix-cache-for-prompt-logprobs")
            return 0

        target_tokens = req_state.num_computed_tokens
        result = self.runtime_cache.load_for_slot(
            RuntimeCacheRequest(
                req_id=req_id,
                block_ids=req_state.block_ids,
                first_seq_blocks=req_state.first_seq_blocks,
                num_computed_tokens=target_tokens,
                cache_slot_id=slot_id,
                cache_token_ids=self._cache_token_ids(req_state, target_tokens),
            ),
            slot_id=slot_id,
        )
        if print_debug:
            if result.action == "reuse-live":
                print(f"[cache] req={req_id} slot={slot_id} reuse-live-cache matched={target_tokens}/{target_tokens}")
            elif result.action == "load-own":
                print(f"[cache] req={req_id} slot={slot_id} load-own matched={result.matched_tokens}/{target_tokens}")
            elif result.action == "load-shared":
                print(
                    f"[cache] req={req_id} slot={slot_id} "
                    f"load-shared matched={result.matched_tokens}/{target_tokens}"
                )
            else:
                print(f"[cache] req={req_id} slot={slot_id} cache-miss fallback matched=0/{target_tokens}")
        return result.matched_tokens

    def _dump_snapshot(
        self,
        req_id: str,
        req_state: RequestState,
        next_num_tokens: int,
        slot_id: Optional[int] = None,
        print_debug: bool = False,
    ) -> bool:
        snapshot = self.runtime_cache.dump_and_store_snapshot(
            req_id=req_id,
            block_ids=req_state.block_ids,
            first_seq_blocks=req_state.first_seq_blocks,
            num_tokens=next_num_tokens,
            slot_id=slot_id,
            cache_token_ids=self._cache_token_ids(req_state, next_num_tokens),
        )
        if snapshot is None:
            if self._is_batch_model() and not self._warned_batch_cache_snapshot_unsupported:
                logger.warning(
                    "Batch cache runtime does not expose slot-scoped dump/load APIs. "
                    "Finished-request prefix snapshots are disabled for batch-compiled models."
                )
                self._warned_batch_cache_snapshot_unsupported = True
            return False
        if print_debug:
            num_blocks = len(req_state.first_seq_blocks)
            print(
                f"[cache] req={req_id} dump tokens={max(0, int(next_num_tokens))} "
                f"blocks={num_blocks} snapshots={self.runtime_cache.snapshot_count()}"
            )
        return True

    @staticmethod
    def _cache_token_ids(req_state: RequestState, num_tokens: int) -> tuple[int, ...]:
        token_ids = list(req_state.prompt_token_ids) + list(req_state.output_token_ids)
        return tuple(token_ids[: max(0, int(num_tokens))])

    def _finalize_finished_request(
        self,
        req_id: str,
        print_debug: bool = False,
    ) -> None:
        finished_req_state = self.req_states.pop(req_id, None)
        finished_slot_id = finished_req_state.cache_slot_id if finished_req_state is not None else None
        if finished_req_state is not None:
            should_dump = self.runtime_cache.should_dump_snapshot_after_step(
                req_id,
                finished_req_state.num_computed_tokens,
            )
            if self._is_batch_model():
                if (
                    should_dump
                    and self._dump_snapshot(
                        req_id=req_id,
                        req_state=finished_req_state,
                        next_num_tokens=finished_req_state.num_computed_tokens,
                        slot_id=finished_slot_id,
                        print_debug=print_debug,
                    )
                    and print_debug
                ):
                    print(f"[cache] req={req_id} slot={finished_slot_id} dump-on-finish")
            elif (
                self.runtime_cache.loaded_req_id == req_id
                and should_dump
                and self._dump_snapshot(
                    req_id=req_id,
                    req_state=finished_req_state,
                    next_num_tokens=finished_req_state.num_computed_tokens,
                    print_debug=print_debug,
                )
                and print_debug
            ):
                print(f"[cache] req={req_id} dump-on-finish")
        for evicted_req_id in self.runtime_cache.mark_snapshot_finished(req_id):
            if print_debug:
                print(f"[cache] evict-finished req={evicted_req_id} reason=lru-cap")
        self.runtime_cache.clear_loaded_request(req_id)
        if self._is_batch_model():
            self._release_cache_slot(req_id)

    def init_device(self) -> None:
        self._log_init_stage("init_device")
        return

    def load_model(self) -> None:
        self._log_init_stage("load_model:start", model=self.model_config.model)
        model_kwargs: Dict[str, object] = {}

        def _merge_kwargs(value: object) -> None:
            if isinstance(value, str):
                try:
                    import json

                    value = json.loads(value)
                except Exception:
                    return
            if isinstance(value, dict):
                for key in _MBLT_BACKEND_KWARG_NAMES:
                    if key in value:
                        model_kwargs[key] = value[key]

        for source in ("model_loader_extra_config",):
            _merge_kwargs(getattr(self.load_config, source, None))
            _merge_kwargs(getattr(self.vllm_config.load_config, source, None))

        for source in ("model_kwargs", "hf_overrides"):
            _merge_kwargs(getattr(self.model_config, source, None))
            _merge_kwargs(getattr(self.vllm_config.model_config, source, None))

        hf_config = getattr(self.model_config, "hf_config", None)
        model_kwargs = _normalize_model_kwargs_for_hf_config(model_kwargs, hf_config)

        start = time.perf_counter()
        self._log_init_stage(
            "load_model:before_from_pretrained",
            model=self.model_config.model,
            model_kwargs=model_kwargs,
        )
        auto_model_cls = AutoModelForImageTextToText if _is_multimodal_hf_config(hf_config) else AutoModelForCausalLM
        self.model = auto_model_cls.from_pretrained(
            self.model_config.model,
            trust_remote_code=True,
            **model_kwargs,
        )
        self._log_init_stage(
            "load_model:after_from_pretrained",
            start,
            model_type=type(self.model).__name__,
        )

        start = time.perf_counter()
        self._log_init_stage("load_model:before_eval")
        self.model.eval()
        self._log_init_stage("load_model:after_eval", start)

        model_max_batch_size = resolve_model_max_batch_size(self.vllm_config)
        if model_max_batch_size is not None:
            self.max_batch_size = model_max_batch_size
        start = time.perf_counter()
        self._log_init_stage("load_model:before_get_input_embeddings")
        self.input_embeddings = self.model.get_input_embeddings()
        self._log_init_stage(
            "load_model:after_get_input_embeddings",
            start,
            embedding_type=type(self.input_embeddings).__name__,
        )

        start = time.perf_counter()
        self._log_init_stage("load_model:before_get_cache_mxq_model")
        self.cache_model = self.model.get_cache_mxq_model()
        self._log_init_stage(
            "load_model:after_get_cache_mxq_model",
            start,
            cache_model_type=type(self.cache_model).__name__,
        )
        self._reset_cache_slots()
        self._infer_output_buffers = None
        self._log_init_stage("load_model:done")
        return

    def _make_cached_sampling_state(
        self,
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[list[int]],
    ) -> CachedSamplingState:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        enable_sampling_penalties = getattr(self, "enable_sampling_penalties", True)
        requested_frequency_penalty = float(sampling_params.frequency_penalty)
        requested_presence_penalty = float(sampling_params.presence_penalty)
        requested_repetition_penalty = float(sampling_params.repetition_penalty)
        penalties_requested = (
            requested_frequency_penalty != 0.0
            or requested_presence_penalty != 0.0
            or requested_repetition_penalty != 1.0
        )

        if enable_sampling_penalties:
            frequency_penalty = requested_frequency_penalty
            presence_penalty = requested_presence_penalty
            repetition_penalty = requested_repetition_penalty
        else:
            if penalties_requested and not getattr(self, "_warned_penalties_disabled", False):
                logger.warning(
                    "Sampling penalties are disabled for non-CUDA MBLT runtime. "
                    "Ignoring frequency_penalty=%s, presence_penalty=%s, repetition_penalty=%s.",
                    requested_frequency_penalty,
                    requested_presence_penalty,
                    requested_repetition_penalty,
                )
                self._warned_penalties_disabled = True
            frequency_penalty = 0.0
            presence_penalty = 0.0
            repetition_penalty = 1.0

        generator = None
        if sampling_params.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(sampling_params.seed)

        max_num_logprobs = None
        if sampling_params.logprobs is not None:
            max_num_logprobs = sampling_params.logprobs
            if max_num_logprobs < 0:
                max_num_logprobs = 0

        return CachedSamplingState(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k if sampling_params.top_k > 0 else self.model.config.vocab_size),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            generator=generator,
            max_num_logprobs=max_num_logprobs,
            bad_words_token_ids=sampling_params._bad_words_token_ids or None,
            prompt_token_ids=torch.as_tensor(prompt_token_ids or [], dtype=torch.int64),
            has_penalties=(frequency_penalty != 0.0 or presence_penalty != 0.0 or repetition_penalty != 1.0),
        )

    def _pack_prompt_token_ids(
        self,
        prompt_token_ids_list: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if not prompt_token_ids_list:
            return self.empty_prompt_token_ids

        max_prompt_len = max(token_ids.numel() for token_ids in prompt_token_ids_list)
        if max_prompt_len == 0:
            return torch.empty((len(prompt_token_ids_list), 0), dtype=torch.int64)

        prompt_token_ids = torch.full(
            (len(prompt_token_ids_list), max_prompt_len),
            fill_value=self.model.config.vocab_size,
            dtype=torch.int64,
        )
        for row, token_ids in enumerate(prompt_token_ids_list):
            if token_ids.numel() > 0:
                prompt_token_ids[row, : token_ids.numel()] = token_ids
        return prompt_token_ids

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {
            "mblt": MLAAttentionSpec(block_size=self._kv_block_size(), num_kv_heads=1, head_size=1, dtype=torch.int8)
        }

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_cache_config = kv_cache_config
        self._log_init_stage(
            "initialize_from_config",
            block_size=getattr(kv_cache_config, "block_size", None),
            num_groups=len(getattr(kv_cache_config, "groups", {}) or {}),
        )

    def compile_or_warm_up_model(self) -> None:
        self._log_init_stage("compile_or_warm_up_model")
        pass

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        expected = (self.max_batch_size + 1) * self._num_blocks_per_request()
        self._log_init_stage(
            "initialize_cache",
            expected_gpu_blocks=expected,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )
        assert num_gpu_blocks == expected, f"GPU Blocks mismatch: expected {expected}, got {num_gpu_blocks}"

    def determine_available_memory(self) -> int:
        spec = self.get_kv_cache_spec()["mblt"]
        total_blocks = (self.max_batch_size + 1) * self._num_blocks_per_request()
        available_memory = total_blocks * spec.page_size_bytes
        self._log_init_stage(
            "determine_available_memory",
            total_blocks=total_blocks,
            page_size_bytes=spec.page_size_bytes,
            available_memory=available_memory,
        )
        return available_memory

    def check_health(self) -> None:
        self._log_init_stage(
            "check_health",
            model_initialized=self.model is not None,
            cache_model_initialized=self.cache_model is not None,
        )
        if self.model is None or self.cache_model is None:
            raise RuntimeError("MBLT Accelerator/Model is not initialized.")

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if self.input_embeddings is None:
            raise RuntimeError("Input embeddings are not initialized.")

        print_debug = self.print_debug

        if print_debug:
            print("new: ", scheduler_output.scheduled_new_reqs)
            print("cached: ", scheduler_output.scheduled_cached_reqs)
            print("finished: ", scheduler_output.finished_req_ids)
            print("scheduled: ", scheduler_output.num_scheduled_tokens)
            print("metadata: ", scheduler_output.kv_connector_metadata)

        for req_id in scheduler_output.finished_req_ids:
            self._finalize_finished_request(req_id, print_debug=print_debug)

        # Add new requests to req_states
        for new_req in scheduler_output.scheduled_new_reqs:
            sampling_params = new_req.sampling_params or SamplingParams.from_optional()
            vlm_session_id = self._get_vlm_session_id(new_req)
            self._validate_mobilint_vlm_request_constraints(
                new_req.mm_features,
                session_id=vlm_session_id,
            )
            prompt_embeds, prompt_deepstack_embeds = self._build_prompt_embeds(
                new_req.prompt_token_ids,
                new_req.prompt_embeds,
                new_req.mm_features,
            )

            normalized_block_ids = self._normalize_block_ids(new_req.block_ids)
            prompt_embeds_np = self._to_cpu_float32_numpy(prompt_embeds)
            prompt_deepstack_embeds_np = (
                self._to_cpu_float32_numpy(prompt_deepstack_embeds) if prompt_deepstack_embeds is not None else None
            )
            cache_slot_id = self._assign_cache_slot(new_req.req_id) if self._is_batch_model() else None

            self.req_states[new_req.req_id] = RequestState(
                is_prefill=True,
                output_token_ids=[],
                sampling_params=sampling_params,
                cached_sampling_state=self._make_cached_sampling_state(
                    sampling_params,
                    new_req.prompt_token_ids,
                ),
                block_ids=normalized_block_ids,
                first_seq_blocks=self._first_seq_blocks(normalized_block_ids),
                num_computed_tokens=new_req.num_computed_tokens,
                num_output_tokens=0,
                prompt_embeds=prompt_embeds_np,
                prompt_deepstack_embeds=prompt_deepstack_embeds_np,
                is_multimodal=bool(new_req.mm_features),
                prompt_len=int(prompt_embeds_np.shape[0]),
                prompt_token_ids=new_req.prompt_token_ids or [],
                cache_slot_id=cache_slot_id,
                vlm_session_id=vlm_session_id,
            )

        # Continue cached requests
        for i, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
            cached_request_state = self.req_states[req_id]

            # all_token_ids = scheduler_output.scheduled_cached_reqs.all_token_ids[req_id]
            cached_request_state.num_computed_tokens = scheduler_output.scheduled_cached_reqs.num_computed_tokens[i]
            cached_request_state.num_output_tokens = scheduler_output.scheduled_cached_reqs.num_output_tokens[i]

            new_block_ids = scheduler_output.scheduled_cached_reqs.new_block_ids[i]
            if new_block_ids is not None:
                if req_id in scheduler_output.scheduled_cached_reqs.resumed_req_ids:
                    cached_request_state.block_ids = self._normalize_block_ids(new_block_ids)
                else:
                    cached_request_state.block_ids = self._append_block_ids(
                        cached_request_state.block_ids,
                        new_block_ids,
                    )
                cached_request_state.first_seq_blocks = self._first_seq_blocks(cached_request_state.block_ids)

        batch_size = len(scheduler_output.num_scheduled_tokens)

        if batch_size <= 0:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        scheduled_end_positions: list[int] = []
        next_cache_sizes: list[int] = []
        sequence_lengths: list[int] = []
        logits_batch: list[torch.Tensor] = []
        req_states_for_sampling: list[RequestState] = []
        sampling_req_ids: list[str] = []
        prompt_logprobs_dict = {}

        if self._is_batch_model():
            if print_debug:
                logger.info(
                    "[mblt-batch-debug] scheduler_output: batch_size=%d max_batch_size=%d "
                    "max_num_batched_tokens=%s scheduled_tokens=%s cached_req_ids=%s "
                    "new_req_count=%d finished_req_count=%d",
                    batch_size,
                    self.max_batch_size,
                    getattr(self, "max_num_batched_tokens", None),
                    dict(scheduler_output.num_scheduled_tokens),
                    list(getattr(scheduler_output.scheduled_cached_reqs, "req_ids", []) or []),
                    len(getattr(scheduler_output, "scheduled_new_reqs", []) or []),
                    len(getattr(scheduler_output, "finished_req_ids", []) or []),
                )
            if batch_size > self.max_batch_size:
                raise RuntimeError(
                    "Scheduled batch exceeds compiled batch capacity: "
                    f"scheduled={batch_size}, max_batch_size={self.max_batch_size}"
                )

            input_embeds_batch: list[np.ndarray] = []
            cache_sizes: list[int] = []
            cache_ids: list[int] = []
            microstep_indices: list[int] = []
            normal_indices: list[int] = []

            for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
                req_state = self.req_states[req_id]
                slot_id = req_state.cache_slot_id
                if slot_id is None:
                    slot_id = self._assign_cache_slot(req_id)
                    req_state.cache_slot_id = slot_id

                self._ensure_batch_vlm_supported(req_state)

                scheduled_end = req_state.num_computed_tokens + num_scheduled_token
                cache_size = self._load_snapshot_if_needed(
                    req_id,
                    req_state,
                    slot_id=slot_id,
                    print_debug=print_debug,
                )
                input_embeds = self._build_input_embeds(req_state, cache_size, scheduled_end)
                sequence_length = int(input_embeds.shape[0])
                next_cache_size = cache_size + sequence_length
                req_state.is_prefill = scheduled_end < req_state.prompt_len

                req_ids.append(req_id)
                req_id_to_index[req_id] = len(req_ids) - 1
                scheduled_end_positions.append(scheduled_end)
                next_cache_sizes.append(next_cache_size)
                sequence_lengths.append(sequence_length)
                input_embeds_batch.append(input_embeds)
                cache_sizes.append(cache_size)
                cache_ids.append(slot_id)

                if self._needs_last_logit_prompt_logprob_microsteps(req_state, sequence_length):
                    microstep_indices.append(len(req_ids) - 1)
                else:
                    normal_indices.append(len(req_ids) - 1)

            if print_debug:
                logger.info(
                    "[mblt-batch-debug] prepared batch: req_count=%d normal_count=%d "
                    "microstep_count=%d sequence_lengths=%s cache_sizes=%s "
                    "scheduled_end_positions=%s prompt_lens=%s next_prompt_logprob_pos=%s",
                    len(req_ids),
                    len(normal_indices),
                    len(microstep_indices),
                    sequence_lengths,
                    cache_sizes,
                    scheduled_end_positions,
                    [self.req_states[req_id].prompt_len for req_id in req_ids],
                    [self.req_states[req_id].next_prompt_logprob_pos for req_id in req_ids],
                )
            batched_logits: list[Optional[InferenceLogits]] = [None for _ in req_ids]

            if normal_indices:
                normal_logits = self._infer_normal_logits_batch_chunked(
                    output_indices=normal_indices,
                    input_embeds_batch=[input_embeds_batch[i] for i in normal_indices],
                    cache_sizes=[cache_sizes[i] for i in normal_indices],
                    cache_ids=[cache_ids[i] for i in normal_indices],
                )
                for i, inference_logits in normal_logits.items():
                    batched_logits[i] = inference_logits

            if microstep_indices:
                microstep_logits = self._run_prompt_logprob_microsteps_batch(
                    [
                        (
                            i,
                            req_ids[i],
                            self.req_states[req_ids[i]],
                            cache_sizes[i],
                            scheduled_end_positions[i],
                            cache_ids[i],
                        )
                        for i in microstep_indices
                    ]
                )
                for i, inference_logits in microstep_logits.items():
                    batched_logits[i] = inference_logits

            for i, req_id in enumerate(req_ids):
                req_state = self.req_states[req_id]
                inference_logits = batched_logits[i]
                if inference_logits is None:
                    raise RuntimeError(f"Missing batched logits for req_id={req_id}.")
                sequence_logits = inference_logits.full_sequence_logits
                self._get_completed_prompt_logprobs_tensors_for_scheduler(
                    req_state=req_state,
                    sequence_logits=sequence_logits,
                    start_idx=cache_sizes[i],
                    scheduled_end=scheduled_end_positions[i],
                    cache_id=cache_ids[i],
                    can_emit_output=False,
                )

            for i, req_id in enumerate(req_ids):
                req_state = self.req_states[req_id]
                req_state.num_computed_tokens = next_cache_sizes[i]
                self.runtime_cache.mark_slot_owner(cache_ids[i], req_id)
                if self._should_sample_after_step(
                    req_state,
                    scheduled_end_positions[i],
                    sequence_lengths[i],
                ):
                    prompt_logprobs_tensors = self._pop_completed_prompt_logprobs_tensors_for_scheduler(
                        req_state=req_state,
                        can_emit_output=True,
                    )
                    if prompt_logprobs_tensors is not None:
                        prompt_logprobs_dict[req_id] = prompt_logprobs_tensors
                    inference_logits = batched_logits[i]
                    if inference_logits is None:
                        raise RuntimeError(f"Missing sampling logits for req_id={req_id}.")
                    logits_batch.append(torch.from_numpy(inference_logits.last_token_logits).reshape(1, -1))
                    req_states_for_sampling.append(req_state)
                    sampling_req_ids.append(req_id)
        else:
            for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
                req_state = self.req_states[req_id]
                scheduled_end = req_state.num_computed_tokens + num_scheduled_token

                req_ids.append(req_id)
                req_id_to_index[req_id] = len(req_ids) - 1
                scheduled_end_positions.append(scheduled_end)

            for i in range(len(req_ids)):
                req_id = req_ids[i]
                req_state = self.req_states[req_id]
                cache_size = self._load_snapshot_if_needed(
                    req_id,
                    req_state,
                    print_debug=print_debug,
                )
                scheduled_end = scheduled_end_positions[i]
                input_embeds = self._build_input_embeds(req_state, cache_size, scheduled_end)
                deepstack_embeds = self._build_deepstack_input_embeds(
                    req_state,
                    cache_size,
                    scheduled_end,
                )
                sequence_length = int(input_embeds.shape[0])
                next_cache_size = cache_size + sequence_length
                req_state.is_prefill = scheduled_end < req_state.prompt_len
                next_cache_sizes.append(next_cache_size)
                sequence_lengths.append(sequence_length)

                if self._needs_last_logit_prompt_logprob_microsteps(req_state, sequence_length):
                    inference_logits = self._run_prompt_logprob_microsteps_single(
                        req_id=req_id,
                        req_state=req_state,
                        start_idx=cache_size,
                        scheduled_end=scheduled_end,
                    )
                    if inference_logits is None:
                        raise RuntimeError(f"Missing single prompt-logprob microstep logits for req_id={req_id}.")
                else:
                    inference_logits = self._infer_logits_with_sequence(
                        input_embeds,
                        deepstack_embeds,
                        cache_size=cache_size,
                    )
                prompt_logprob_pos_before = req_state.next_prompt_logprob_pos
                self._get_completed_prompt_logprobs_tensors_for_scheduler(
                    req_state=req_state,
                    sequence_logits=inference_logits.full_sequence_logits,
                    start_idx=cache_size,
                    scheduled_end=scheduled_end,
                    can_emit_output=False,
                )
                prompt_logprob_fallback_replayed = (
                    inference_logits.full_sequence_logits is None
                    and req_state.next_prompt_logprob_pos > prompt_logprob_pos_before
                )
                if prompt_logprob_fallback_replayed:
                    # Last-token-only prompt-logprob fallback replays prompt prefixes from
                    # an empty runtime cache.  Re-run this request from the beginning of
                    # the scheduled prefix afterwards so live KV state and generated-token
                    # logits correspond to this request even when the original step had
                    # loaded a prefix snapshot/cache_size > 0.
                    input_embeds = self._build_input_embeds(req_state, 0, scheduled_end)
                    deepstack_embeds = self._build_deepstack_input_embeds(
                        req_state,
                        0,
                        scheduled_end,
                    )
                    sequence_length = int(input_embeds.shape[0])
                    next_cache_size = sequence_length
                    next_cache_sizes[-1] = next_cache_size
                    sequence_lengths[-1] = sequence_length
                    inference_logits = self._infer_logits_with_sequence(
                        input_embeds,
                        deepstack_embeds,
                        cache_size=0,
                    )
                # The live accelerator KV now belongs to this request at
                # next_cache_size tokens, so later same-request decode can reuse it
                # without forcing a block-boundary snapshot dump.
                req_state.num_computed_tokens = next_cache_size
                self.runtime_cache.mark_loaded_request(req_id)
                if self._should_sample_after_step(
                    req_state,
                    scheduled_end,
                    sequence_length,
                ):
                    prompt_logprobs_tensors = self._pop_completed_prompt_logprobs_tensors_for_scheduler(
                        req_state=req_state,
                        can_emit_output=True,
                    )
                    if prompt_logprobs_tensors is not None:
                        prompt_logprobs_dict[req_id] = prompt_logprobs_tensors
                    logits_batch.append(torch.from_numpy(inference_logits.last_token_logits).reshape(1, -1))
                    req_states_for_sampling.append(req_state)
                    sampling_req_ids.append(req_id)

        sampled_token_ids: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in req_ids]
        logprobs = None

        if logits_batch:
            logits = torch.cat(logits_batch, dim=0)
            sampling_metadata = self._make_sampling_metadata(req_states_for_sampling)
            sampler_output = self._sample_next_token(logits, sampling_metadata)
            sampled_token_ids_int: list[list[int]] = sampler_output.sampled_token_ids.tolist()
            generated_lengths_by_req_id: dict[str, int] = {}
            for i, req_id in enumerate(sampling_req_ids):
                self.req_states[req_id].output_token_ids.extend(sampled_token_ids_int[i])
                generated_lengths_by_req_id[req_id] = len(sampled_token_ids_int[i])
                sampled_token_ids[req_id_to_index[req_id]] = np.asarray(
                    sampled_token_ids_int[i],
                    dtype=np.int64,
                )

            if sampler_output.logprobs_tensors is not None:
                cu_num_generated_tokens = [0]
                for req_id in req_ids:
                    cu_num_generated_tokens.append(
                        cu_num_generated_tokens[-1] + generated_lengths_by_req_id.get(req_id, 0)
                    )
                logprobs = sampler_output.logprobs_tensors.tolists(cu_num_generated_tokens)

        if print_debug:
            print(req_ids, req_id_to_index, sampled_token_ids)

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

    def _make_sampling_metadata(
        self,
        request_states: List[RequestState],
    ) -> SamplingMetadata:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        num_requests = len(request_states)
        temperatures = torch.empty(num_requests, dtype=torch.float)
        top_ps = torch.empty(num_requests, dtype=torch.float)
        top_ks = torch.empty(num_requests, dtype=torch.int)
        frequency_penalties = torch.empty(num_requests, dtype=torch.float)
        presence_penalties = torch.empty(num_requests, dtype=torch.float)
        repetition_penalties = torch.empty(num_requests, dtype=torch.float)
        output_token_ids: list[list[int]] = []
        prompt_token_ids_list: list[torch.Tensor] = []

        generators: dict[int, torch.Generator] = {}
        bad_words_token_ids: dict[int, list[list[int]]] = {}

        max_num_logprobs = None
        any_penalties = False
        all_greedy = True
        all_random = True

        for i, state in enumerate(request_states):
            cached_sampling = state.cached_sampling_state

            temperatures[i] = cached_sampling.temperature
            top_ps[i] = cached_sampling.top_p
            top_ks[i] = cached_sampling.top_k
            frequency_penalties[i] = cached_sampling.frequency_penalty
            presence_penalties[i] = cached_sampling.presence_penalty
            repetition_penalties[i] = cached_sampling.repetition_penalty

            output_token_ids.append(state.output_token_ids)
            prompt_token_ids_list.append(cached_sampling.prompt_token_ids)

            if cached_sampling.generator is not None:
                generators[i] = cached_sampling.generator

            if cached_sampling.max_num_logprobs is not None:
                if max_num_logprobs is None:
                    max_num_logprobs = cached_sampling.max_num_logprobs
                else:
                    max_num_logprobs = max(
                        max_num_logprobs,
                        cached_sampling.max_num_logprobs,
                    )

            if cached_sampling.bad_words_token_ids is not None:
                bad_words_token_ids[i] = cached_sampling.bad_words_token_ids

            any_penalties = any_penalties or cached_sampling.has_penalties
            # vLLM treats temperature=0 as greedy decoding regardless of
            # top_k.  The OpenAI API commonly sends temperature=0 with
            # top_k unset, which we normalize to vocab_size above.  Checking
            # only top_k would incorrectly mark such requests as random and
            # route them through top-k/top-p sampling with temp=0, which can
            # repeatedly sample token id 0 ("!") even when argmax logits are
            # correct.  Keep top_k==1 as an additional greedy signal for
            # non-zero-temperature callers that explicitly request it.
            is_greedy = cached_sampling.temperature < 1e-5 or cached_sampling.top_k == 1
            all_greedy = all_greedy and is_greedy
            all_random = all_random and not is_greedy

        prompt_token_ids = self._pack_prompt_token_ids(prompt_token_ids_list) if any_penalties else None

        return SamplingMetadata(
            temperature=temperatures,
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=top_ps,
            top_k=top_ks,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
            no_penalties=not any_penalties,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            output_token_ids=output_token_ids,
            allowed_token_ids_mask=None,
            bad_words_token_ids=bad_words_token_ids,
            logitsprocs=self.empty_logits_processors,
            spec_token_ids=None,
        )

    def sample_tokens(self, grammar_output: GrammarOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        raise NotImplementedError

    def shutdown(self) -> None:
        if self.model:
            dispose = getattr(self.model, "dispose", None)
            if callable(dispose):
                dispose()
        self.cache_model = None
        self.input_embeddings = None
        self._infer_output_buffers = None
        self.runtime_cache.reset()
