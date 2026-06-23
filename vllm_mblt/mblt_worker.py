import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
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
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.worker_base import WorkerBase

from vllm_mblt.mblt_platform import resolve_model_max_batch_size

logger = init_logger(__name__)


_MULTIMODAL_HF_MODEL_TYPES = frozenset(
    {
        "mobilint-qwen2_vl",
        "mobilint-qwen3_vl",
    }
)


def _is_multimodal_hf_config(hf_config: object) -> bool:
    model_type = getattr(hf_config, "model_type", None)
    return isinstance(model_type, str) and model_type in _MULTIMODAL_HF_MODEL_TYPES


def _is_qwen3_vl_hf_config(hf_config: object) -> bool:
    model_type = getattr(hf_config, "model_type", None)
    return model_type == "mobilint-qwen3_vl"


@dataclass
class RequestState:
    is_prefill: bool
    output_token_ids: list[int]
    sampling_params: SamplingParams
    cached_sampling_state: "CachedSamplingState"
    block_ids: tuple[list[int], ...]
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


@dataclass
class CacheSnapshot:
    blobs: list[Any]
    block_ids: tuple[list[int], ...]
    first_seq_blocks: tuple[int, ...]
    num_tokens: int


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
class SnapshotIndexNode:
    children: dict[int, "SnapshotIndexNode"] = field(default_factory=dict)
    best_req_id: Optional[str] = None
    best_num_tokens: int = 0


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
        self.snapshot_index_root = SnapshotIndexNode()

        self.req_states: Dict[str, RequestState] = {}
        self.cache_snapshots: Dict[str, CacheSnapshot] = {}
        self.finished_snapshot_lru: OrderedDict[str, None] = OrderedDict()
        self.loaded_cache_req_id: Optional[str] = None
        self.req_to_cache_slot: dict[str, int] = {}
        self.cache_slot_to_req: dict[int, str] = {}
        self.free_cache_slots: list[int] = []
        self._warned_batch_cache_snapshot_unsupported = False
        self._vlm_image_positions_by_session: dict[str, tuple[int, int, Optional[tuple[bool, ...]]]] = {}

        self.max_batch_size = resolve_model_max_batch_size(self.vllm_config) or 1
        self._reset_cache_slots()
        self.max_seq_len = self.vllm_config.model_config.max_model_len
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
        self.req_to_cache_slot = {}
        self.cache_slot_to_req = {}
        self.free_cache_slots = list(range(max(0, self.max_batch_size)))

    def _is_batch_model(self) -> bool:
        return self.max_batch_size > 1

    def _kv_block_size(self) -> int:
        configured = self.vllm_config.cache_config.block_size
        if configured is None:
            return 128
        return int(configured)

    def _num_blocks_per_request(self) -> int:
        return max(1, math.ceil(self.max_seq_len / self._kv_block_size()))

    def _normalize_block_ids(self, block_ids: tuple[list[int], ...]) -> tuple[list[int], ...]:
        return tuple(list(seq) for seq in block_ids)

    def _append_block_ids(
        self,
        current_block_ids: tuple[list[int], ...],
        new_block_ids: tuple[list[int], ...],
    ) -> tuple[list[int], ...]:
        if not current_block_ids:
            return self._normalize_block_ids(new_block_ids)
        if len(current_block_ids) != len(new_block_ids):
            raise RuntimeError(
                f"KV block_ids layout mismatch: current={len(current_block_ids)} seqs, new={len(new_block_ids)} seqs"
            )
        merged_block_ids: list[list[int]] = []
        for current_seq_blocks, new_seq_blocks in zip(current_block_ids, new_block_ids):
            merged_block_ids.append(list(current_seq_blocks) + list(new_seq_blocks))
        return tuple(merged_block_ids)

    def _first_seq_blocks(self, block_ids: tuple[list[int], ...]) -> tuple[int, ...]:
        if len(block_ids) == 0:
            return ()
        return tuple(block_ids[0])

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
        slot_id = self.req_to_cache_slot.get(req_id)
        if slot_id is None:
            raise RuntimeError(f"No accelerator cache slot is assigned for req_id={req_id}.")
        return slot_id

    def _assign_cache_slot(self, req_id: str) -> int:
        slot_id = self.req_to_cache_slot.get(req_id)
        if slot_id is not None:
            return slot_id
        if not self.free_cache_slots:
            raise RuntimeError(
                "No free accelerator cache slots remain for batch execution. "
                f"req_id={req_id}, max_batch_size={self.max_batch_size}"
            )
        slot_id = self.free_cache_slots.pop(0)
        self.req_to_cache_slot[req_id] = slot_id
        self.cache_slot_to_req[slot_id] = req_id
        return slot_id

    def _release_cache_slot(self, req_id: str) -> None:
        slot_id = self.req_to_cache_slot.pop(req_id, None)
        if slot_id is None:
            return
        owner = self.cache_slot_to_req.get(slot_id)
        if owner == req_id:
            self.cache_slot_to_req.pop(slot_id, None)
        if slot_id not in self.free_cache_slots:
            self.free_cache_slots.append(slot_id)
            self.free_cache_slots.sort()

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

    def _touch_finished_snapshot(self, req_id: str) -> None:
        if req_id in self.finished_snapshot_lru:
            self.finished_snapshot_lru.move_to_end(req_id)
        else:
            self.finished_snapshot_lru[req_id] = None

    @staticmethod
    def _update_snapshot_index_node(
        node: SnapshotIndexNode,
        req_id: str,
        num_tokens: int,
    ) -> None:
        if num_tokens >= node.best_num_tokens:
            node.best_req_id = req_id
            node.best_num_tokens = num_tokens

    def _rebuild_snapshot_index(self) -> None:
        root = SnapshotIndexNode()
        for req_id, snapshot in self.cache_snapshots.items():
            node = root
            self._update_snapshot_index_node(node, req_id, snapshot.num_tokens)
            for block_id in snapshot.first_seq_blocks:
                node = node.children.setdefault(block_id, SnapshotIndexNode())
                self._update_snapshot_index_node(node, req_id, snapshot.num_tokens)
        self.snapshot_index_root = root

    def _evict_old_finished_snapshots(self, print_debug: bool = False) -> None:
        evicted = False
        while len(self.finished_snapshot_lru) > self.MAX_FINISHED_CACHE_SNAPSHOTS:
            evicted_req_id, _ = self.finished_snapshot_lru.popitem(last=False)
            self.cache_snapshots.pop(evicted_req_id, None)
            evicted = True
            if self.loaded_cache_req_id == evicted_req_id:
                self.loaded_cache_req_id = None
            if print_debug:
                print(f"[cache] evict-finished req={evicted_req_id} reason=lru-cap")
        if evicted:
            self._rebuild_snapshot_index()

    def _should_dump_snapshot_after_step(
        self,
        req_id: str,
        next_num_tokens: int,
    ) -> bool:
        snapshot = self.cache_snapshots.get(req_id)
        if snapshot is None:
            return True
        if next_num_tokens <= snapshot.num_tokens:
            return False
        return self._required_blocks(next_num_tokens) > self._required_blocks(snapshot.num_tokens)

    def _dump_loaded_request_before_switch(
        self,
        next_req_id: str,
        print_debug: bool = False,
    ) -> None:
        loaded_req_id = self.loaded_cache_req_id
        if loaded_req_id is None or loaded_req_id == next_req_id:
            return
        loaded_req_state = self.req_states.get(loaded_req_id)
        if loaded_req_state is None:
            return
        if not self._should_dump_snapshot_after_step(
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

    def _required_blocks(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return math.ceil(num_tokens / self._kv_block_size())

    def _prefix_compatible_tokens(
        self,
        target_blocks: tuple[int, ...],
        target_tokens: int,
        snapshot_blocks: tuple[int, ...],
        snapshot_tokens: int,
    ) -> int:
        if target_tokens <= 0 or snapshot_tokens <= 0:
            return 0

        needed_target_blocks = self._required_blocks(target_tokens)
        common_blocks = 0
        for i in range(min(needed_target_blocks, len(target_blocks), len(snapshot_blocks))):
            if target_blocks[i] != snapshot_blocks[i]:
                break
            common_blocks += 1

        if common_blocks == 0:
            return 0

        if common_blocks >= needed_target_blocks:
            return min(target_tokens, snapshot_tokens)

        return min(snapshot_tokens, common_blocks * self._kv_block_size(), target_tokens)

    def _choose_snapshot(self, req_state: RequestState) -> tuple[Optional[CacheSnapshot], int]:
        target_tokens = req_state.num_computed_tokens
        if target_tokens <= 0:
            return None, 0

        target_blocks = req_state.first_seq_blocks
        if not target_blocks:
            return None, 0

        best_snapshot: Optional[CacheSnapshot] = None
        best_tokens = 0
        node = self.snapshot_index_root
        for depth, block_id in enumerate(target_blocks, start=1):
            node = node.children.get(block_id)
            if node is None or node.best_req_id is None:
                break

            snapshot = self.cache_snapshots.get(node.best_req_id)
            if snapshot is None:
                continue

            matched_tokens = min(
                snapshot.num_tokens,
                depth * self._kv_block_size(),
                target_tokens,
            )
            if matched_tokens > best_tokens:
                best_tokens = matched_tokens
                best_snapshot = snapshot

        return best_snapshot, best_tokens

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

    @staticmethod
    def _should_sample_after_step(
        req_state: RequestState,
        scheduled_end: int,
        sequence_length: int,
    ) -> bool:
        if sequence_length <= 0:
            return False
        return scheduled_end >= req_state.prompt_len

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

        target_tokens = req_state.num_computed_tokens
        self._dump_loaded_request_before_switch(
            next_req_id=req_id,
            print_debug=print_debug,
        )

        if target_tokens <= 0:
            self.loaded_cache_req_id = None
            if print_debug:
                print(f"[cache] req={req_id} skip-load target_tokens=0")
            return 0

        # If the active accelerator cache already belongs to this request,
        # it already contains the up-to-date KV state from previous steps.
        if self.loaded_cache_req_id == req_id:
            if print_debug:
                print(f"[cache] req={req_id} reuse-live-cache matched={target_tokens}/{target_tokens}")
            return target_tokens

        own_snapshot = self.cache_snapshots.get(req_id)
        if own_snapshot is not None:
            matched_tokens = self._prefix_compatible_tokens(
                target_blocks=req_state.first_seq_blocks,
                target_tokens=target_tokens,
                snapshot_blocks=own_snapshot.first_seq_blocks,
                snapshot_tokens=own_snapshot.num_tokens,
            )
            if matched_tokens >= target_tokens:
                if self.loaded_cache_req_id != req_id:
                    cache_model = self._get_cache_model()
                    # load_cache_memory(...) must happen before infer(..., cache_size=...)
                    cache_model.load_cache_memory(own_snapshot.blobs)
                    self.loaded_cache_req_id = req_id
                    if print_debug:
                        print(f"[cache] req={req_id} load-own matched={matched_tokens}/{target_tokens}")
                elif print_debug:
                    print(f"[cache] req={req_id} reuse-loaded-own matched={matched_tokens}/{target_tokens}")
                return target_tokens

        snapshot, matched_tokens = self._choose_snapshot(req_state)
        if snapshot is None or matched_tokens <= 0:
            self.loaded_cache_req_id = None
            if print_debug:
                print(f"[cache] req={req_id} cache-miss fallback matched=0/{target_tokens}")
            return 0

        cache_model = self._get_cache_model()
        # load_cache_memory(...) must happen before infer(..., cache_size=...)
        cache_model.load_cache_memory(snapshot.blobs)
        self.loaded_cache_req_id = req_id
        if print_debug:
            print(f"[cache] req={req_id} load-shared matched={matched_tokens}/{target_tokens}")
        return matched_tokens

    def _load_snapshot_for_batch_slot(
        self,
        req_id: str,
        req_state: RequestState,
        slot_id: Optional[int],
        print_debug: bool = False,
    ) -> int:
        if slot_id is None:
            raise RuntimeError(f"Batch execution requires a cache slot for req_id={req_id}.")

        target_tokens = req_state.num_computed_tokens
        if target_tokens <= 0:
            return 0

        live_owner = self.cache_slot_to_req.get(slot_id)
        if live_owner == req_id:
            if print_debug:
                print(f"[cache] req={req_id} slot={slot_id} reuse-live-cache matched={target_tokens}/{target_tokens}")
            return target_tokens

        own_snapshot = self.cache_snapshots.get(req_id)
        if own_snapshot is not None:
            matched_tokens = self._prefix_compatible_tokens(
                target_blocks=req_state.first_seq_blocks,
                target_tokens=target_tokens,
                snapshot_blocks=own_snapshot.first_seq_blocks,
                snapshot_tokens=own_snapshot.num_tokens,
            )
            if matched_tokens >= target_tokens:
                if self._load_runtime_cache(own_snapshot.blobs, slot_id=slot_id):
                    self.cache_slot_to_req[slot_id] = req_id
                    if print_debug:
                        print(f"[cache] req={req_id} slot={slot_id} load-own matched={matched_tokens}/{target_tokens}")
                    return target_tokens

        snapshot, matched_tokens = self._choose_snapshot(req_state)
        if snapshot is not None and matched_tokens > 0:
            if self._load_runtime_cache(snapshot.blobs, slot_id=slot_id):
                self.cache_slot_to_req[slot_id] = req_id
                if print_debug:
                    print(f"[cache] req={req_id} slot={slot_id} load-shared matched={matched_tokens}/{target_tokens}")
                return matched_tokens

        self.cache_slot_to_req[slot_id] = req_id
        if print_debug:
            print(f"[cache] req={req_id} slot={slot_id} cache-miss fallback matched=0/{target_tokens}")
        return 0

    def _dump_snapshot(
        self,
        req_id: str,
        req_state: RequestState,
        next_num_tokens: int,
        slot_id: Optional[int] = None,
        print_debug: bool = False,
    ) -> bool:
        # Active request snapshots are not part of the finished-session LRU pool.
        self.finished_snapshot_lru.pop(req_id, None)
        blobs = self._dump_runtime_cache(slot_id=slot_id)
        if blobs is None:
            if self._is_batch_model() and not self._warned_batch_cache_snapshot_unsupported:
                logger.warning(
                    "Batch cache runtime does not expose slot-scoped dump/load APIs. "
                    "Finished-request prefix snapshots are disabled for batch-compiled models."
                )
                self._warned_batch_cache_snapshot_unsupported = True
            return False
        stored_tokens = max(0, int(next_num_tokens))
        self.cache_snapshots[req_id] = CacheSnapshot(
            blobs=blobs,
            block_ids=self._normalize_block_ids(req_state.block_ids),
            first_seq_blocks=req_state.first_seq_blocks,
            num_tokens=stored_tokens,
        )
        self._rebuild_snapshot_index()
        if print_debug:
            num_blocks = len(req_state.first_seq_blocks)
            print(
                f"[cache] req={req_id} dump tokens={stored_tokens} "
                f"blocks={num_blocks} snapshots={len(self.cache_snapshots)}"
            )
        return True

    def _finalize_finished_request(
        self,
        req_id: str,
        print_debug: bool = False,
    ) -> None:
        finished_req_state = self.req_states.pop(req_id, None)
        finished_slot_id = finished_req_state.cache_slot_id if finished_req_state is not None else None
        if finished_req_state is not None:
            should_dump = self._should_dump_snapshot_after_step(
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
                self.loaded_cache_req_id == req_id
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
        if req_id in self.cache_snapshots:
            self._touch_finished_snapshot(req_id)
        self._evict_old_finished_snapshots(print_debug=print_debug)
        if self.loaded_cache_req_id == req_id:
            self.loaded_cache_req_id = None
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
                for key in (
                    "dev_no",
                    "target_cores",
                    "target_clusters",
                    "core_mode",
                ):
                    if key in value:
                        model_kwargs[key] = value[key]

        for source in ("model_loader_extra_config",):
            _merge_kwargs(getattr(self.load_config, source, None))
            _merge_kwargs(getattr(self.vllm_config.load_config, source, None))

        for source in ("model_kwargs", "hf_overrides"):
            _merge_kwargs(getattr(self.model_config, source, None))
            _merge_kwargs(getattr(self.vllm_config.model_config, source, None))

        start = time.perf_counter()
        self._log_init_stage(
            "load_model:before_from_pretrained",
            model=self.model_config.model,
            model_kwargs=model_kwargs,
        )
        hf_config = getattr(self.model_config, "hf_config", None)
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

        if self._is_batch_model():
            if batch_size > self.max_batch_size:
                raise RuntimeError(
                    "Scheduled batch exceeds compiled batch capacity: "
                    f"scheduled={batch_size}, max_batch_size={self.max_batch_size}"
                )

            input_embeds_batch: list[np.ndarray] = []
            cache_sizes: list[int] = []
            cache_ids: list[int] = []

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

            batched_logits = self._infer_logits_batch(
                input_embeds_batch=input_embeds_batch,
                cache_sizes=cache_sizes,
                cache_ids=cache_ids,
            )

            for i, req_id in enumerate(req_ids):
                req_state = self.req_states[req_id]
                req_state.num_computed_tokens = next_cache_sizes[i]
                self.cache_slot_to_req[cache_ids[i]] = req_id
                if self._should_sample_after_step(
                    req_state,
                    scheduled_end_positions[i],
                    sequence_lengths[i],
                ):
                    logits_batch.append(torch.from_numpy(batched_logits[i]).reshape(1, -1))
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

                logits = self._infer_logits(
                    input_embeds,
                    deepstack_embeds,
                    cache_size=cache_size,
                )
                # The live accelerator KV now belongs to this request at
                # next_cache_size tokens, so later same-request decode can reuse it
                # without forcing a block-boundary snapshot dump.
                req_state.num_computed_tokens = next_cache_size
                self.loaded_cache_req_id = req_id
                if self._should_sample_after_step(
                    req_state,
                    scheduled_end,
                    sequence_length,
                ):
                    logits_batch.append(torch.from_numpy(logits).reshape(1, -1))
                    req_states_for_sampling.append(req_state)
                    sampling_req_ids.append(req_id)

        sampled_token_ids: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in req_ids]
        logprobs = None

        if logits_batch:
            logits = torch.cat(logits_batch, dim=0)
            sampling_metadata = self._make_sampling_metadata(req_states_for_sampling)
            sampler_output = self.sampler.forward(logits=logits, sampling_metadata=sampling_metadata)
            sampled_token_ids_int: list[list[int]] = sampler_output.sampled_token_ids.tolist()
            for i, req_id in enumerate(sampling_req_ids):
                self.req_states[req_id].output_token_ids.extend(sampled_token_ids_int[i])
                sampled_token_ids[req_id_to_index[req_id]] = np.asarray(
                    sampled_token_ids_int[i],
                    dtype=np.int64,
                )

            logprobs = (
                sampler_output.logprobs_tensors.tolists() if sampler_output.logprobs_tensors is not None else None
            )

        if print_debug:
            print(req_ids, req_id_to_index, sampled_token_ids)

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict={},
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
        self.loaded_cache_req_id = None
        self._reset_cache_slots()
        self.snapshot_index_root = SnapshotIndexNode()
