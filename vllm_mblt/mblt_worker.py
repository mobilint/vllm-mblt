import hashlib
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from mblt_model_zoo.hf_transformers.utils.generation_utils import MobilintGenerationMixin
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers_mblt.utils.cache_utils import MobilintCache, MobilintDeepStackCache
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

from vllm_mblt.eagle3_runtime import Eagle3Runner, is_eagle3_model
from vllm_mblt.host_threads import apply_engine_host_thread_defaults
from vllm_mblt.mblt_platform import resolve_model_max_batch_size
from vllm_mblt.models.modeling_vl_utils import QWEN3_VL_MAX_VISION_TOKENS
from vllm_mblt.moe_runtime import MbltMoECacheModel, is_mixture_of_experts_model
from vllm_mblt.request_cache import RequestCacheSlots, join_request_caches
from vllm_mblt.runtime_cache import (
    KVBlockIds,
    MbltRuntimeCacheManager,
    PromptEmbedCacheIdentity,
    RuntimeCacheLoadResult,
    RuntimeCacheRequest,
    RuntimeCacheSnapshot,
    RuntimeCacheSnapshotMatch,
    append_block_ids,
    first_seq_blocks,
    normalize_block_ids,
)
from vllm_mblt.tracing import TRACE_DIR_ENV_VAR, MbltTracer, resolve_trace_dir

logger = init_logger(__name__)


_TRUE_ENV_VALUES = frozenset({"1", "true", "TRUE", "True"})


def _resolve_enable_sampling_penalties(env_value: Optional[str]) -> bool:
    """Decide whether frequency/presence/repetition penalties are applied.

    vLLM applies these with a pure-torch fallback when the fused CUDA kernel is
    unavailable, so the CPU-hosted MBLT sampler can honour them and the default
    is on regardless of CUDA. The released MBLT packages ship
    ``repetition_penalty`` in ``generation_config.json``; dropping it silently
    would generate NPU output under different sampling than the GPU reference it
    is compared against. Setting VLLM_MBLT_ENABLE_SAMPLING_PENALTIES to a
    non-true value restores the previous ignore-and-warn behaviour.
    """
    if env_value is None:
        return True
    return env_value in _TRUE_ENV_VALUES


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
_MBLT_BACKEND_KWARG_PREFIXES = (
    "",
    "text_",
    "vision_",
    "encoder_",
    "decoder_",
    "base_",
    "draft_",
    "fc_",
    # Mixture-of-Experts roles: per-layer shared MXQs, expert MXQs, and lm_head.
    "shared_",
    "expert_",
    "lm_head_",
)
_MBLT_BACKEND_KWARG_NAMES = frozenset(
    f"{prefix}{field}" for prefix in _MBLT_BACKEND_KWARG_PREFIXES for field in _MBLT_BACKEND_KWARG_FIELDS
) | {"num_expert_workers"}
_MULTIMODAL_SHARED_BACKEND_KWARG_FIELDS = _MBLT_BACKEND_KWARG_FIELDS - {"mxq_path"}
_MULTIMODAL_BACKEND_PREFIXES = ("vision_", "text_")


# NOTE: "mobilint-qwen3_asr" is deliberately NOT a member of _MULTIMODAL_HF_MODEL_TYPES above.
# Its call sites mean "image-text model" (vision_/text_ kwarg normalisation,
# AutoModelForImageTextToText, the single-image constraints), and joining the set would discard
# Qwen3-ASR's encoder_/decoder_ device-placement kwargs. Qwen3-ASR is served at max_batch_size 1:
# the batch path's _ensure_batch_vlm_supported refuses multimodal requests from non-VLM models.


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


def _positive_int_attr(root: object, path: str) -> Optional[int]:
    value = root
    for attr in path.split("."):
        value = getattr(value, attr, None)
        if value is None:
            return None
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return None
    return int_value if int_value > 0 else None


@dataclass
class RequestState:
    is_prefill: bool
    output_token_ids: list[int]
    sampling_params: SamplingParams
    cached_sampling_state: "CachedSamplingState"
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    first_seq_block_hashes: Optional[tuple[Hashable, ...]]
    num_computed_tokens: int
    num_output_tokens: int
    prompt_embeds: np.ndarray
    prompt_deepstack_embeds: Optional[np.ndarray]
    mrope_position_delta: int
    is_multimodal: bool
    prompt_len: int
    prompt_token_ids: list[int]
    cache_slot_id: Optional[int]
    vlm_session_id: Optional[str]
    multimodal_cache_identity: Optional[Hashable] = None
    explicit_prompt_embeds: bool = False
    next_prompt_logprob_pos: int = 1
    in_progress_prompt_logprobs: Optional[LogprobsTensors] = None
    # This request's KV: a transformers-mblt cache on the request's NPU slot (see vllm_mblt.request_cache).
    kv_cache: Optional[Any] = None
    # Qwen3-VL with a RoPE-input text MXQ: the prompt's M-RoPE positions ``(3, prompt_len)``; generated tokens
    # continue at ``index + mrope_position_delta``.
    prompt_position_ids: Optional[torch.Tensor] = None


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
    deepstack_embeds: Optional[np.ndarray]
    cache_id: int
    cache_size: int
    offset: int = 0
    full_sequence_logits: Optional[list[np.ndarray]] = None
    last_token_logits: Optional[np.ndarray] = None


def _prefix_cache_config_value(config: object, name: str) -> object:
    """Read a prefix-cache setting from the model loader extra config or the model kwargs/overrides."""
    for root in (
        getattr(getattr(config, "load_config", None), "model_loader_extra_config", None),
        getattr(config, "model_loader_extra_config", None),
        getattr(getattr(config, "model_config", None), "model_kwargs", None),
        getattr(getattr(config, "model_config", None), "hf_overrides", None),
    ):
        if isinstance(root, dict) and name in root:
            return root[name]
    return None


DEFAULT_PREFIX_CACHE_MAX_HOST_MIB = 4096


def _resolve_prefix_cache_max_host_mib(config: object) -> int:
    """Host memory (MiB) finished-request prefix snapshots may take: ``VLLM_MBLT_PREFIX_CACHE_MAX_HOST_MIB``."""
    value = os.getenv("VLLM_MBLT_PREFIX_CACHE_MAX_HOST_MIB")
    if value is None:
        value = _prefix_cache_config_value(config, "prefix_cache_max_host_mib")
    try:
        return DEFAULT_PREFIX_CACHE_MAX_HOST_MIB if value in (None, "") else max(0, int(value))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring prefix_cache_max_host_mib=%r: not an integer number of MiB. Using %d MiB.",
            value,
            DEFAULT_PREFIX_CACHE_MAX_HOST_MIB,
        )
        return DEFAULT_PREFIX_CACHE_MAX_HOST_MIB


def _resolve_prefix_cache_host_offload(config: object) -> bool:
    """Whether finished prefix snapshots may be saved to host memory: ``VLLM_MBLT_PREFIX_CACHE_HOST_OFFLOAD``.

    Off by default: a finished snapshot is reused only while its KV is still on its NPU slot, and a slot that
    changes hands drops it without a dump, as a GPU prefix cache drops evicted blocks (issue #40).
    """
    value = os.getenv("VLLM_MBLT_PREFIX_CACHE_HOST_OFFLOAD")
    if value is None:
        value = _prefix_cache_config_value(config, "prefix_cache_host_offload")
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class PrefixCacheCostModel:
    """Worker-side policy for deciding if a matched runtime snapshot is worth loading, and worth saving."""

    DEFAULT_MARGIN = 0.9
    DEFAULT_EWMA_ALPHA = 0.3

    def __init__(
        self,
        *,
        block_size: int,
        auto_threshold_enabled: bool = True,
        manual_min_hit_tokens: Optional[int] = None,
        margin: float = DEFAULT_MARGIN,
        ewma_alpha: float = DEFAULT_EWMA_ALPHA,
        host_offload: bool = False,
    ) -> None:
        self.block_size = max(1, int(block_size))
        # Opt-in: save a finished snapshot to host memory when its slot is reused (and the cost test passes).
        self.host_offload = bool(host_offload)
        self.auto_threshold_enabled = bool(auto_threshold_enabled)
        self.manual_min_hit_tokens = max(0, int(manual_min_hit_tokens)) if manual_min_hit_tokens is not None else None
        self.margin = float(margin)
        if self.margin <= 0:
            self.margin = self.DEFAULT_MARGIN
        self.ewma_alpha = min(1.0, max(0.0, float(ewma_alpha)))
        self.prefill_ms_by_tokens: dict[int, float] = {}
        self.load_ms_by_cache_id: dict[int | None, float] = {}
        self.dump_ms_by_cache_id: dict[int | None, float] = {}

    @classmethod
    def from_config(cls, config: object, *, block_size: int) -> "PrefixCacheCostModel":
        auto_value = os.getenv("VLLM_MBLT_PREFIX_CACHE_AUTO_THRESHOLD")
        if auto_value is None:
            auto_value = _prefix_cache_config_value(config, "prefix_cache_auto_threshold")
        auto_enabled = True if auto_value is None else str(auto_value) not in {"0", "false", "FALSE", "False"}

        manual_value = os.getenv("VLLM_MBLT_PREFIX_CACHE_MIN_HIT_TOKENS")
        if manual_value is None:
            manual_value = _prefix_cache_config_value(config, "prefix_cache_min_hit_tokens")
        try:
            manual_min_hit_tokens = None if manual_value in (None, "") else int(manual_value)
        except (TypeError, ValueError):
            manual_min_hit_tokens = None

        margin_value = os.getenv("VLLM_MBLT_PREFIX_CACHE_LOAD_MARGIN")
        if margin_value is None:
            margin_value = _prefix_cache_config_value(config, "prefix_cache_load_margin")
        try:
            margin = cls.DEFAULT_MARGIN if margin_value in (None, "") else float(margin_value)
        except (TypeError, ValueError):
            margin = cls.DEFAULT_MARGIN

        return cls(
            block_size=block_size,
            auto_threshold_enabled=auto_enabled,
            manual_min_hit_tokens=manual_min_hit_tokens,
            margin=margin,
            host_offload=_resolve_prefix_cache_host_offload(config),
        )

    def observe_prefill(self, num_tokens: int, elapsed_ms: float) -> None:
        num_tokens = max(1, int(num_tokens))
        elapsed_ms = max(0.0, float(elapsed_ms))
        previous = self.prefill_ms_by_tokens.get(num_tokens)
        if previous is None:
            self.prefill_ms_by_tokens[num_tokens] = elapsed_ms
        else:
            self.prefill_ms_by_tokens[num_tokens] = self._ewma(previous, elapsed_ms)

    def observe_load(self, cache_id: int | None, elapsed_ms: float) -> None:
        self._observe_cache_io(self.load_ms_by_cache_id, cache_id, elapsed_ms)

    def observe_dump(self, cache_id: int | None, elapsed_ms: float) -> None:
        self._observe_cache_io(self.dump_ms_by_cache_id, cache_id, elapsed_ms)

    def should_load(self, *, matched_tokens: int, cache_id: int | None = None) -> bool:
        matched_tokens = max(0, int(matched_tokens))
        if matched_tokens <= 0:
            return False
        if self.manual_min_hit_tokens is not None and matched_tokens < self.manual_min_hit_tokens:
            return False
        if not self.auto_threshold_enabled:
            return True

        load_ms = self.load_ms(cache_id)
        prefill_ms = self.prefill_ms(matched_tokens)
        if load_ms is None or prefill_ms is None:
            return True
        return load_ms < prefill_ms * self.margin

    def should_take_over(self, *, matched_tokens: int) -> bool:
        """Whether a request should continue on a snapshot still on its own slot: it costs no load or dump.

        Only the manual minimum applies; the measured cost test has nothing to weigh.
        """
        matched_tokens = max(0, int(matched_tokens))
        if matched_tokens <= 0:
            return False
        return self.manual_min_hit_tokens is None or matched_tokens >= self.manual_min_hit_tokens

    def should_save_snapshot(self, *, num_tokens: int) -> bool:
        """Whether a finished snapshot about to lose its slot is saved to host memory instead of dropped.

        Only with host offload on, and then only when :meth:`should_keep_snapshot` expects a load could pay.
        """
        return self.host_offload and self.should_keep_snapshot(num_tokens=num_tokens)

    def should_keep_snapshot(self, *, num_tokens: int) -> bool:
        """Whether a snapshot of ``num_tokens`` tokens is worth saving to host memory for a later load.

        A later load uses at most the snapshot's ``num_tokens`` tokens, and :meth:`should_load` refuses a load
        that does not beat recomputing the tokens it brings. When the estimates say every such load would be
        refused, saving the snapshot only costs the dump (issue #40). The load can go to any slot, so the
        cheapest measured slot is used, not the slot the snapshot sits on. Before any load has been measured, a
        measured dump stands in for it: both move the slot's whole KV buffers. Without either measurement, or
        without a prefill measurement, the snapshot is kept.
        """
        num_tokens = max(0, int(num_tokens))
        if num_tokens <= 0:
            return False
        if self.manual_min_hit_tokens is not None and num_tokens < self.manual_min_hit_tokens:
            return False
        if not self.auto_threshold_enabled:
            return True
        transfer_ms = self._min_cache_io_ms(self.load_ms_by_cache_id)
        if transfer_ms is None:
            transfer_ms = self._min_cache_io_ms(self.dump_ms_by_cache_id)
        prefill_ms = self.max_prefill_ms(num_tokens)
        if transfer_ms is None or prefill_ms is None:
            return True
        return transfer_ms < prefill_ms * self.margin

    def max_prefill_ms(self, num_tokens: int) -> Optional[float]:
        """The highest :meth:`prefill_ms` estimate for any prefix of up to ``num_tokens`` tokens that may load.

        The estimate scales each measured bucket down linearly, so a shorter prefix can be estimated above a
        longer one that falls in the next bucket: the maximum is the estimate at ``num_tokens`` or a measured
        bucket at or below it. Buckets below the manual minimum hit length are skipped, as :meth:`should_load`
        refuses those prefixes.
        """
        estimate = self.prefill_ms(num_tokens)
        if estimate is None:
            return None
        num_tokens = max(1, int(num_tokens))
        min_tokens = self.manual_min_hit_tokens or 1
        return max(
            [estimate]
            + [ms for tokens, ms in self.prefill_ms_by_tokens.items() if min_tokens <= tokens <= num_tokens]
        )

    def prefill_ms(self, matched_tokens: int) -> Optional[float]:
        if not self.prefill_ms_by_tokens:
            return None
        matched_tokens = max(1, int(matched_tokens))
        measured = sorted(self.prefill_ms_by_tokens.items())
        for tokens, elapsed_ms in measured:
            if matched_tokens <= tokens:
                return elapsed_ms * (matched_tokens / tokens)
        tokens, elapsed_ms = measured[-1]
        return elapsed_ms * (matched_tokens / tokens)

    def load_ms(self, cache_id: int | None = None) -> Optional[float]:
        return self._cache_io_ms(self.load_ms_by_cache_id, cache_id)

    @staticmethod
    def _cache_io_ms(values: dict[int | None, float], cache_id: int | None) -> Optional[float]:
        if cache_id in values:
            return values[cache_id]
        if None in values:
            return values[None]
        if not values:
            return None
        return statistics.median(values.values())

    @staticmethod
    def _min_cache_io_ms(values: dict[int | None, float]) -> Optional[float]:
        return min(values.values()) if values else None

    def _observe_cache_io(self, values: dict[int | None, float], cache_id: int | None, elapsed_ms: float) -> None:
        elapsed_ms = max(0.0, float(elapsed_ms))
        previous = values.get(cache_id)
        values[cache_id] = elapsed_ms if previous is None else self._ewma(previous, elapsed_ms)

    def _ewma(self, previous: float, sample: float) -> float:
        return previous * (1.0 - self.ewma_alpha) + sample * self.ewma_alpha


class MbltWorker(WorkerBase):
    MAX_FINISHED_CACHE_SNAPSHOTS = 16
    # Set for an EAGLE-3 model, which runs whole speculative rounds through transformers-mblt.
    eagle3_runner: Optional[Eagle3Runner] = None

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
        self.cache_backend: Optional[Any] = None

        self.req_states: Dict[str, RequestState] = {}
        self._warned_last_logit_prompt_logprobs = False
        self._live_cache_prefix_incomplete_count = 0
        self._vlm_image_positions_by_session: dict[str, tuple[int, int, Optional[tuple[bool, ...]]]] = {}

        self.max_batch_size = resolve_model_max_batch_size(self.vllm_config) or 1
        self.max_seq_len = self.vllm_config.model_config.max_model_len
        self.runtime_cache = MbltRuntimeCacheManager(
            max_batch_size=self.max_batch_size,
            block_size=self._kv_block_size(),
            max_finished_snapshots=self.MAX_FINISHED_CACHE_SNAPSHOTS,
            enable_prefix_caching=self._prefix_caching_enabled(),
        )
        self.request_caches: Optional[RequestCacheSlots] = None
        # Per slot (None on a non-batch model): the request whose cache the next infer on that slot binds.
        self._slot_caches: dict[Optional[int], tuple[str, Any]] = {}
        self.prefix_cache_cost_model = PrefixCacheCostModel.from_config(
            self.vllm_config,
            block_size=self._kv_block_size(),
        )
        self.sampler = Sampler(logprobs_mode="raw_logits")
        self.empty_logits_processors = LogitsProcessors(None)
        self.empty_prompt_token_ids = torch.empty((0, 0), dtype=torch.int64)
        self.enable_chunked_prefill = self.vllm_config.scheduler_config.enable_chunked_prefill
        self.max_num_batched_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # Disabled by default to avoid per-token stdout spam in production runs.
        self.print_debug = os.getenv("VLLM_MBLT_DEBUG", "0") in {"1", "true", "TRUE", "True"}
        self.enable_sampling_penalties = _resolve_enable_sampling_penalties(
            os.getenv("VLLM_MBLT_ENABLE_SAMPLING_PENALTIES")
        )
        self._warned_penalties_disabled = False
        # Built on the first profile() call, not here: resolving the trace
        # directory eagerly would make every worker create it even for runs
        # that never trace.
        self._tracer: Optional[MbltTracer] = None

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
        if isinstance(self.cache_model, MbltMoECacheModel):
            self.request_caches = RequestCacheSlots([], moe=self.cache_model)
        else:
            cache_cls, cache_kwargs = MobilintCache, {}
            deepstack_cache_kwargs = getattr(self._language_model(), "get_mobilint_cache_kwargs", None)
            if self._supports_deepstack_input() and callable(deepstack_cache_kwargs):
                # Qwen3-VL decodes on a deepstack cache, sized by its text decoder.
                cache_cls, cache_kwargs = MobilintDeepStackCache, deepstack_cache_kwargs()
            if self._is_batch_model():
                models = self._get_cache_models()
                k_per_model = max(1, int(getattr(self.cache_backend, "k_per_model", self.max_batch_size) or 1))
            else:
                models, k_per_model = [self._get_cache_model()], 1
            self.request_caches = RequestCacheSlots(models, k_per_model, cache_cls=cache_cls, cache_kwargs=cache_kwargs)
        self._slot_caches = {}
        self.runtime_cache.enable_prefix_caching = self._prefix_caching_enabled()
        self.runtime_cache.reset_slots(
            max_batch_size=self.max_batch_size,
            # New requests are balanced over the cards their slots run on.
            slot_card=self.request_caches.slot_model_index if self._is_batch_model() else None,
        )
        self.runtime_cache.max_finished_snapshots = self._finished_snapshot_cap()

    def _finished_snapshot_cap(self) -> int:
        """How many finished-request prefix snapshots fit the host memory budget, up to the count cap.

        A saved snapshot holds one slot's KV in host memory (256 MiB for a 4K-token Llama-3 slot, 48 x 4 MiB for
        Qwen3-30B-A3B), so the budget divided by that size bounds how many can be kept. Without host offload a
        batch model's finished snapshots only stay on their free slots and take no host memory, so up to one per
        slot is kept.
        """
        cap = self.MAX_FINISHED_CACHE_SNAPSHOTS
        budget_mib = _resolve_prefix_cache_max_host_mib(self.vllm_config)
        per_snapshot = self.request_caches.kv_nbytes_per_cache()
        if per_snapshot:
            cap = min(cap, budget_mib * 2**20 // per_snapshot)
        host_offload = self._ensure_prefix_cache_cost_model().host_offload
        if self._is_batch_model() and not host_offload:
            cap = max(cap, self.max_batch_size)
        logger.info(
            "MBLT keeps up to %d finished prefix snapshots (%s MiB each, budget %d MiB, host offload %s).",
            cap,
            "unknown" if not per_snapshot else f"{per_snapshot / 2**20:.0f}",
            budget_mib,
            "on" if host_offload else "off",
        )
        return cap

    def _prefix_caching_enabled(self) -> bool:
        # MbltPlatform.check_and_update_config turns an unset (None) value
        # into True, so treat None the same way here.
        cache_config = getattr(self.vllm_config, "cache_config", None)
        enabled = getattr(cache_config, "enable_prefix_caching", None)
        return True if enabled is None else bool(enabled)

    def _is_batch_model(self) -> bool:
        return self.max_batch_size > 1

    def _kv_block_size(self) -> int:
        configured = getattr(getattr(self.vllm_config, "cache_config", None), "block_size", None)
        if configured is None:
            runtime_cache = getattr(self, "runtime_cache", None)
            if runtime_cache is not None:
                return int(getattr(runtime_cache, "block_size", 128))
            return 128
        return int(configured)

    def _ensure_prefix_cache_cost_model(self) -> PrefixCacheCostModel:
        cost_model = getattr(self, "prefix_cache_cost_model", None)
        if cost_model is None:
            cost_model = PrefixCacheCostModel.from_config(
                self.vllm_config,
                block_size=self._kv_block_size(),
            )
            self.prefix_cache_cost_model = cost_model
        return cost_model

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

    def _extract_first_seq_block_hashes(
        self,
        source: object,
        *,
        candidates: tuple[str, ...] = ("block_hashes", "kv_block_hashes", "block_hash_ids"),
    ) -> Optional[tuple[Hashable, ...]]:
        for attr in candidates:
            block_hashes = getattr(source, attr, None)
            if block_hashes is None:
                continue
            if not block_hashes:
                return ()
            if isinstance(block_hashes[0], list):
                return tuple(block_hashes[0])
            return tuple(block_hashes)
        return None

    def _extract_indexed_first_seq_block_hashes(
        self,
        source: object,
        index: int,
        *,
        candidates: tuple[str, ...],
    ) -> Optional[tuple[Hashable, ...]]:
        for attr in candidates:
            block_hashes_by_req = getattr(source, attr, None)
            if block_hashes_by_req is None:
                continue
            if not block_hashes_by_req:
                return ()
            try:
                block_hashes = block_hashes_by_req[index]
            except IndexError:
                return None
            if not block_hashes:
                return ()
            if isinstance(block_hashes[0], list):
                return tuple(block_hashes[0])
            return tuple(block_hashes)
        return None

    def _append_first_seq_block_hashes(
        self,
        current_hashes: Optional[tuple[Hashable, ...]],
        new_hashes: Optional[tuple[Hashable, ...]],
    ) -> Optional[tuple[Hashable, ...]]:
        if current_hashes is None or new_hashes is None:
            return None
        return current_hashes + new_hashes

    def _get_cache_model(self) -> Any:
        if self.cache_model is None:
            if self.model is None:
                raise RuntimeError("Model is not initialized.")
            self.cache_model = self.model.get_cache_mxq_model()
        return self.cache_model

    def _resolve_cache_backend(self) -> Optional[Any]:
        """Return the model-zoo backend that owns the text MXQ model slots."""
        if self.model is None:
            return None

        candidates = [self.model]
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)
            language_model = getattr(inner_model, "language_model", None)
            if language_model is not None:
                candidates.append(language_model)
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            candidates.append(language_model)

        cache_model = self._get_cache_model()
        for candidate in candidates:
            backend = getattr(candidate, "npu_backend", None)
            models = getattr(backend, "mxq_models", None)
            if backend is not None and models and any(model is cache_model for model in models):
                return backend
        return None

    def _get_cache_models(self) -> list[Any]:
        backend = getattr(self, "cache_backend", None)
        models = getattr(backend, "mxq_models", None)
        if models:
            return list(models)
        return [self._get_cache_model()]

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

    def _supports_dynamic_qwen3_vl_inputs(self) -> bool:
        """Return whether the loaded Qwen3-VL artifact supports dynamic vision."""
        model_config = getattr(getattr(self, "model", None), "config", None)
        return _is_qwen3_vl_hf_config(model_config) and bool(getattr(model_config, "dynamic_vision", False))

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

        if self._supports_dynamic_qwen3_vl_inputs():
            unsupported = [
                str(getattr(feature, "modality", ""))
                for feature in mm_features
                if not str(getattr(feature, "modality", "")).startswith(("image", "video"))
            ]
            if unsupported:
                raise RuntimeError(
                    f"Unsupported multimodal modalities for Mobilint Qwen3-VL dynamic vision on NPU: {unsupported}"
                )
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

    @staticmethod
    def _fingerprint_multimodal_value(value: object) -> tuple[str, tuple[int, ...], str, str] | None:
        if value is None:
            return None
        value = getattr(value, "data", value)
        try:
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                array = tensor.numpy()
            else:
                array = np.asarray(value)
        except (TypeError, ValueError, RuntimeError):
            return None

        if array.dtype == object:
            return None
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        return ("sha256", tuple(int(dim) for dim in array.shape), str(array.dtype), digest.hexdigest())

    @classmethod
    def _multimodal_feature_content_fingerprint(cls, feature: MultiModalFeatureSpec) -> Hashable | None:
        modality = str(getattr(feature, "modality", ""))
        if modality.startswith("image"):
            value_keys = ("pixel_values", "image_grid_thw")
        elif modality.startswith("video"):
            value_keys = ("pixel_values_videos", "video_grid_thw")
        else:
            return None

        fingerprints: list[tuple[str, tuple[str, tuple[int, ...], str, str] | None]] = []
        for key in value_keys:
            value = cls._extract_multimodal_value(feature, key)
            fingerprint = cls._fingerprint_multimodal_value(value)
            if fingerprint is None:
                return None
            fingerprints.append((key, fingerprint))

        return (modality, tuple(fingerprints))

    @classmethod
    def _build_vlm_multimodal_cache_identity(
        cls,
        session_id: str,
        mm_features: Optional[list[MultiModalFeatureSpec]],
    ) -> Hashable | None:
        if not mm_features:
            return None

        entries = []
        for feature in mm_features:
            position = cls._multimodal_position_signature(feature.mm_position)
            fingerprint = cls._multimodal_feature_content_fingerprint(feature)
            entries.append((str(getattr(feature, "modality", "")), position, fingerprint))
        return ("vlm", str(session_id), tuple(entries))

    def _get_cache_slot(self, req_id: str) -> int:
        return self.runtime_cache.get_slot(req_id)

    def _assign_cache_slot(self, req_id: str, req_state: Optional[RequestState] = None) -> int:
        """Give a batch request a free slot: one still holding a snapshot of its prefix first (issue #40).

        A request whose prefix matches a finished snapshot still on a free slot takes that slot and continues on
        the snapshot's KV in place, with no load or dump. Otherwise the slot comes from the card running the
        fewest requests, and on it: a slot holding no snapshot, then one whose snapshot would not be worth saving
        (:meth:`PrefixCacheCostModel.should_keep_snapshot`), then the least recently used snapshot, then the
        lowest slot.
        """
        resident = self._resident_finished_snapshots(self.runtime_cache.free_slots)
        prefix_slot = None
        if req_state is not None and resident:
            prefix_slot = self._prefix_matching_slot(req_id, req_state, resident)
        cost_model = self._ensure_prefix_cache_cost_model()

        def slot_victim_rank(slot_id: int) -> tuple[int, ...]:
            snapshot_req_id = resident.get(slot_id)
            if snapshot_req_id is None:
                return (0, 0)
            snapshot = self.runtime_cache.snapshots[snapshot_req_id]
            worth_keeping = cost_model.should_keep_snapshot(num_tokens=snapshot.num_tokens)
            return (2 if worth_keeping else 1, self.runtime_cache.finished_snapshot_age(snapshot_req_id))

        return self.runtime_cache.assign_slot(req_id, prefix_slot=prefix_slot, slot_victim_rank=slot_victim_rank)

    def _resident_finished_snapshots(self, slot_ids: Sequence[Optional[int]]) -> dict[Optional[int], str]:
        """Map each of ``slot_ids`` still holding a finished request's prefix snapshot to that request.

        Only snapshots in the index whose KV the slot holds right now count: a dropped or saved-out snapshot is
        not resident, and a live request's cache is never a candidate.
        """
        if self.request_caches is None or not self.runtime_cache.snapshots:
            return {}
        resident: dict[Optional[int], str] = {}
        for req_id, snapshot in self.runtime_cache.snapshots.items():
            if snapshot.blobs is None or req_id in self.req_states or snapshot.num_tokens <= 0:
                continue
            for slot_id in slot_ids:
                if slot_id not in resident and self.request_caches.holds_slot(snapshot.blobs, self._slot_key(slot_id)):
                    resident[slot_id] = req_id
                    break
        return resident

    def _match_resident_snapshot(
        self,
        request: RuntimeCacheRequest,
        resident: dict[Optional[int], str],
    ) -> tuple[Optional[int], RuntimeCacheSnapshotMatch]:
        """The longest prefix match among the resident snapshots, with the slot that holds it.

        Uses the same index and compatibility checks as a snapshot load (block hashes, token ids, multimodal and
        prompt-embedding identity).
        """
        slot_by_req = {req_id: slot_id for slot_id, req_id in resident.items()}
        match = self.runtime_cache.choose_prefix_snapshot(request, candidate_req_ids=slot_by_req)
        match = self.runtime_cache.verify_prompt_embed_match(request, match)
        if match.snapshot is None or match.req_id not in slot_by_req:
            return None, RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)
        return slot_by_req[match.req_id], match

    def _prefix_matching_slot(
        self,
        req_id: str,
        req_state: RequestState,
        resident: dict[Optional[int], str],
    ) -> Optional[int]:
        """The free slot holding a finished snapshot of the request's prefix, if any."""
        if not self.runtime_cache.enable_prefix_caching or req_state.num_computed_tokens <= 0:
            return None
        if self._should_recompute_prompt_logprobs_from_start(req_state):
            return None
        slot_id, match = self._match_resident_snapshot(self._runtime_cache_request(req_id, req_state, None), resident)
        if not self._ensure_prefix_cache_cost_model().should_take_over(matched_tokens=match.matched_tokens):
            return None
        return slot_id

    def _runtime_cache_request(
        self, req_id: str, req_state: RequestState, slot_id: Optional[int]
    ) -> RuntimeCacheRequest:
        """The request's prefix as the snapshot index matches it."""
        target_tokens = req_state.num_computed_tokens
        return RuntimeCacheRequest(
            req_id=req_id,
            block_ids=req_state.block_ids,
            first_seq_blocks=req_state.first_seq_blocks,
            num_computed_tokens=target_tokens,
            first_seq_block_hashes=req_state.first_seq_block_hashes,
            cache_slot_id=slot_id,
            cache_token_ids=self._cache_token_ids(req_state, target_tokens),
            multimodal_cache_identity=self._cache_multimodal_identity(req_state, target_tokens),
            prompt_embed_cache_identity=self._make_prompt_embed_cache_identity(req_state),
        )

    def _release_cache_slot(self, req_id: str) -> None:
        self.runtime_cache.release_slot(req_id)

    def _request_cache(self, req_state: RequestState, slot_id: Optional[int]) -> Any:
        """Return the request's cache, building an empty one on its slot the first time."""
        if req_state.kv_cache is None:
            req_state.kv_cache = self.request_caches.new_cache(slot_id)
        return req_state.kv_cache

    def _slot_key(self, slot_id: Optional[int]) -> Optional[int]:
        return slot_id if self._is_batch_model() else None

    def _activate_request_cache(self, req_id: str, req_state: RequestState, slot_id: Optional[int]) -> None:
        """Make the infers on ``slot_id`` in this step run on ``req_id``'s cache."""
        self._slot_caches[self._slot_key(slot_id)] = (req_id, self._request_cache(req_state, slot_id))

    def _slot_cache(self, slot_id: Optional[int]) -> Optional[Any]:
        """The cache the next infer on ``slot_id`` binds; ``None`` only before any request ran (calibration)."""
        entry = self._slot_caches.get(self._slot_key(slot_id))
        if entry is None:
            if self.req_states:
                raise RuntimeError(f"No request cache is active on cache slot {slot_id}.")
            return None
        return entry[1]

    def _prepare_slot(self, slot_id: Optional[int], cache_size: int) -> Optional[Any]:
        """Bind the slot's request cache at ``cache_size`` before an infer; return it for :meth:`_commit_slot`."""
        cache = self._slot_cache(slot_id)
        self.request_caches.prepare(cache, self._slot_key(slot_id), cache_size)
        return cache

    def _own_prefix_length(self, slot_id: Optional[int]) -> int:
        """Tokens the slot's request cache holds, before a prompt-logprob replay of the request's own prefix."""
        cache = self._slot_cache(slot_id)
        return int(cache.get_seq_length()) if cache is not None else 0

    def _restore_own_prefix_length(self, slot_id: Optional[int], length: int) -> None:
        """Undo a prompt-logprob replay's rewind of the request's cache.

        The replay rewrites positions the slot already holds with the same tokens, so the slot still holds the
        request's KV up to the length it had before the replay.
        """
        cache = self._slot_cache(slot_id)
        if cache is not None and cache.get_seq_length() < length:
            cache.set_seq_length(length)

    def _calibrate_prefix_cache_prefill_costs(self) -> None:
        cost_model = self._ensure_prefix_cache_cost_model()
        if not cost_model.auto_threshold_enabled:
            return
        if os.getenv("VLLM_MBLT_PREFIX_CACHE_CALIBRATE", "1") in {"0", "false", "FALSE", "False"}:
            return
        if self.input_embeddings is None or self.cache_model is None:
            return
        if not callable(getattr(self.input_embeddings, "__call__", None)):
            return

        block_size = self._kv_block_size()
        bucket_blocks = (1, 2, 4, 8)
        bucket_tokens = [
            tokens for tokens in (block_size * blocks for blocks in bucket_blocks) if tokens <= self.max_seq_len
        ]
        if not bucket_tokens:
            bucket_tokens = [min(block_size, self.max_seq_len)]

        for num_tokens in bucket_tokens:
            try:
                embeds = self._embed_token_ids([0] * num_tokens)
                samples_ms: list[float] = []
                for sample_index in range(3):
                    start = time.perf_counter()
                    if self._is_batch_model():
                        self._infer_logits_batch([embeds], cache_sizes=[0], cache_ids=[1])
                    else:
                        self._infer_logits(embeds, None, cache_size=0)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if sample_index > 0:
                        samples_ms.append(elapsed_ms)
                if samples_ms:
                    cost_model.observe_prefill(num_tokens, statistics.median(samples_ms))
            except Exception as exc:
                logger.debug("Skipping Mobilint prefix-cache prefill calibration: %s", exc)
                break

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

    def _validate_dynamic_qwen3_vl_grid(self, modality: str, grid_thw: Optional[torch.Tensor]) -> None:
        """Enforce the compiled vision MXQ limit on actual processor output."""
        if not self._supports_dynamic_qwen3_vl_inputs() or grid_thw is None:
            return
        token_counts = grid_thw.prod(dim=-1)
        total_tokens = int(token_counts.sum().item())
        if total_tokens > QWEN3_VL_MAX_VISION_TOKENS:
            raise RuntimeError(
                f"Qwen3-VL {modality} preprocessing produced {token_counts.tolist()} pre-merge vision tokens "
                f"({total_tokens} total for one encoder call), "
                f"exceeding the NPU encoder limit of {QWEN3_VL_MAX_VISION_TOKENS}. "
                "Keep resizing enabled and reduce the input resolution or sampled video frames."
            )

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
        get_audio_features = getattr(self.model, "get_audio_features", None)
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
                image_grid_thw = self._normalize_grid_thw(image_grid_thw)
                self._validate_dynamic_qwen3_vl_grid("image", image_grid_thw)
                image_features = get_image_features(
                    pixel_values=self._to_torch_tensor(pixel_values, dtype=torch.float32),
                    image_grid_thw=image_grid_thw,
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
                video_grid_thw = self._normalize_grid_thw(video_grid_thw)
                self._validate_dynamic_qwen3_vl_grid("video", video_grid_thw)
                video_features = get_video_features(
                    pixel_values_videos=self._to_torch_tensor(pixel_values_videos, dtype=torch.float32),
                    video_grid_thw=video_grid_thw,
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
            elif modality.startswith("audio"):
                if not callable(get_audio_features):
                    raise RuntimeError(f"Model {type(self.model).__name__} does not expose get_audio_features().")
                input_features = self._extract_multimodal_value(feature, "input_features")
                if input_features is None:
                    raise RuntimeError("Audio multimodal feature is missing input_features.")
                feature_attention_mask = self._extract_multimodal_value(feature, "feature_attention_mask")
                if feature_attention_mask is None:
                    # qwen_asr's get_audio_features() falls back to feature_attention_mask.sum(-1) when the
                    # mask is None, so a missing mask would surface as an AttributeError.
                    raise RuntimeError(
                        "Audio multimodal feature is missing feature_attention_mask; the feature "
                        "extractor must run with return_attention_mask=True."
                    )
                input_features = self._to_torch_tensor(input_features, dtype=torch.float32)
                feature_attention_mask = self._to_torch_tensor(feature_attention_mask, dtype=torch.long)
                # Features arrive per item, without the leading clip axis along which get_audio_features()
                # zips input_features with the mask's lengths. Add the axis to both or to neither, so the
                # two can never disagree about how many clips they describe.
                if input_features.ndim == 2 and feature_attention_mask.ndim == 1:
                    input_features = input_features.unsqueeze(0)
                    feature_attention_mask = feature_attention_mask.unsqueeze(0)
                if (
                    input_features.ndim != 3
                    or feature_attention_mask.ndim != 2
                    or input_features.shape[0] != feature_attention_mask.shape[0]
                ):
                    raise RuntimeError(
                        "Audio features and attention mask must describe the same clips: "
                        f"input_features={tuple(input_features.shape)}, "
                        f"feature_attention_mask={tuple(feature_attention_mask.shape)}."
                    )
                audio_features = get_audio_features(
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                )
                audio_embeds = self._normalize_multimodal_embeddings(audio_features)
                self._scatter_multimodal_embeddings(
                    merged_prompt_embeds,
                    feature.mm_position,
                    audio_embeds,
                )
            else:
                raise NotImplementedError(f"Unsupported multimodal modality for MBLT worker: {modality}")

        return merged_prompt_embeds, deepstack_prompt_embeds

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

    def _language_model(self) -> Any | None:
        """The text decoder of an image-text model (its forward takes the merged embeddings), else None."""
        get_language_model = getattr(self.model, "get_language_model", None)
        language_model = get_language_model() if callable(get_language_model) else None
        if language_model is None:
            language_model = getattr(getattr(self.model, "model", None), "language_model", None)
        return language_model

    def _build_prompt_position_ids(
        self,
        prompt_token_ids: Optional[list[int]],
        mm_features: Optional[list[MultiModalFeatureSpec]],
        prompt_len: int,
    ) -> tuple[Optional[torch.Tensor], int]:
        """The prompt's M-RoPE positions ``(3, 1, prompt_len)`` and offset, for a RoPE-input Qwen3-VL text MXQ."""
        if not self._text_mxq_uses_rope_input():
            return None, 0

        if not prompt_token_ids:
            if mm_features:
                raise RuntimeError(
                    "Qwen3-VL 3-input text MXQ requires prompt_token_ids to build multimodal MRoPE positions."
                )
            return torch.arange(prompt_len, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1), 0

        get_mrope_input_positions = getattr(self.model, "get_mrope_input_positions", None)
        if callable(get_mrope_input_positions) and mm_features:
            position_ids, mrope_delta = get_mrope_input_positions(prompt_token_ids, mm_features)
            position_ids = torch.as_tensor(position_ids, dtype=torch.long)
            if position_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(1)
            delta = int(torch.as_tensor(mrope_delta).reshape(-1)[0].item())
            return position_ids, delta

        input_ids = torch.as_tensor(prompt_token_ids, dtype=torch.long).view(1, -1)
        image_grids = []
        video_grids = []
        if mm_features:
            for feature in mm_features:
                modality = str(getattr(feature, "modality", ""))
                if modality.startswith("image"):
                    image_grid_thw = self._normalize_grid_thw(self._extract_multimodal_value(feature, "image_grid_thw"))
                    if image_grid_thw is not None:
                        image_grids.append(image_grid_thw)
                elif modality.startswith("video"):
                    video_grid_thw = self._normalize_grid_thw(self._extract_multimodal_value(feature, "video_grid_thw"))
                    if video_grid_thw is not None:
                        video_grids.append(video_grid_thw)

        get_rope_index = getattr(getattr(self.model, "model", None), "get_rope_index", None)
        if callable(get_rope_index) and (image_grids or video_grids):
            # Qwen3-VL represents video timing with timestamp tokens inserted by
            # its processor. Unlike Qwen2-VL, its get_rope_index() contract has
            # no second_per_grid_ts input; the token offsets carry that timing.
            position_ids, mrope_delta = get_rope_index(
                input_ids=input_ids,
                image_grid_thw=torch.cat(image_grids, dim=0) if image_grids else None,
                video_grid_thw=torch.cat(video_grids, dim=0) if video_grids else None,
            )
            delta = int(torch.as_tensor(mrope_delta).reshape(-1)[0].item())
            return position_ids, delta

        if mm_features:
            raise RuntimeError(
                "Qwen3-VL 3-input text MXQ requires get_mrope_input_positions() or "
                "model.get_rope_index() to build multimodal MRoPE positions."
            )

        return torch.arange(input_ids.shape[1], dtype=torch.long).view(1, 1, -1).expand(3, 1, -1), 0

    @staticmethod
    def _row_position_ids(req_state: Optional[RequestState], start: int, end: int) -> torch.Tensor:
        """M-RoPE positions ``(3, end - start)`` of a request's tokens ``[start, end)``.

        Prompt tokens take the prompt's positions; generated tokens continue at ``index + mrope_position_delta``.
        Without a request (prefill-cost calibration) the tokens are plain text at ``[start, end)``.
        """
        prompt_positions = None if req_state is None else req_state.prompt_position_ids
        if prompt_positions is None:
            return torch.arange(start, end, dtype=torch.long).expand(3, -1)
        prompt_len = int(prompt_positions.shape[-1])
        pieces = []
        if start < prompt_len:
            pieces.append(prompt_positions.reshape(3, prompt_len)[:, start : min(end, prompt_len)])
        if end > prompt_len:
            delta = int(req_state.mrope_position_delta)
            pieces.append(torch.arange(max(start, prompt_len) + delta, end + delta, dtype=torch.long).expand(3, -1))
        return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]

    def _ensure_batch_vlm_supported(self, req_state: RequestState) -> None:
        if getattr(req_state, "is_multimodal", False):
            if req_state.prompt_deepstack_embeds is not None and not self._supports_deepstack_input():
                raise RuntimeError("Deepstack batch VLM inputs are only supported for Qwen3-VL models.")
            if not self._is_multimodal_model():
                raise RuntimeError(
                    "Batch VLM execution is only supported for Mobilint Qwen2-VL and Qwen3-VL model types."
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

    def _text_mxq_uses_rope_input(self) -> bool:
        if not self._supports_deepstack_input():
            return False
        return len(self._cache_model_input_shapes(self._get_cache_model())) in (3, 5)

    @staticmethod
    def _shape_dim_matches_sequence(dim: int, input_seq_len: int) -> bool:
        return dim == -1 or dim == input_seq_len

    @staticmethod
    def _buffer_max_width(cache_model: Any, getter_name: str) -> Optional[int]:
        getter = getattr(cache_model, getter_name, None)
        if not callable(getter):
            return None
        try:
            buffer_infos = getter()
        except Exception:
            return None
        if not buffer_infos:
            return None
        try:
            max_width = int(getattr(buffer_infos[0], "max_width"))
        except (AttributeError, TypeError, ValueError):
            return None
        return max_width if max_width > 0 else None

    @staticmethod
    def _compiled_batch_rows(cache_model: Any) -> int:
        get_cache_infos = getattr(cache_model, "get_cache_infos", None)
        if not callable(get_cache_infos):
            return 0
        try:
            return max((int(getattr(info, "num_batches", 0) or 0) for info in get_cache_infos()), default=0)
        except Exception:
            return 0

    def _batch_output_logits_mode(self, cache_model: Any, output_shape: tuple[int, ...]) -> str:
        """Classify a batch MXQ's ``[1, -1, vocab]`` logits as full-sequence or last-token.

        A batch MXQ packs every scheduled row along the dynamic axis, so the
        declared output shape is ``[1, -1, vocab]`` for both kinds: a last-logit
        MXQ returns one position per row, a full-logits MXQ one per input token.
        The bound of that dynamic axis tells them apart.  A full-logits MXQ
        keeps every input position through lm_head, so its output buffer is as
        wide as its input buffer (the packed token budget); a last-logit MXQ's
        output buffer is only as wide as its row count.

        A batch MXQ whose token budget equals its row count cannot be told
        apart this way and stays ``unknown`` (1-token prompt-logprob
        microsteps), as does a static sequence axis or an output buffer wider
        than the input buffer.

        Assumptions:

        * Buffer info ``[0]`` is the text-embedding input and the logits
          output, the same index ``get_model_output_shape()[0]`` is read at.
        * The row bound is the larger of vLLM's ``max_batch_size`` and the
          MXQ's compiled batch (its KV cache's ``num_batches``), so a
          ``max_num_seqs`` override below the compiled batch cannot lower it
          and make a last-logit MXQ whose token budget equals its compiled
          batch read as full-sequence.
        """

        if int(output_shape[1]) != -1:
            return "unknown"
        input_width = self._buffer_max_width(cache_model, "get_input_buffer_info")
        output_width = self._buffer_max_width(cache_model, "get_output_buffer_info")
        if input_width is None or output_width is None:
            return "unknown"
        row_bound = max(int(self.max_batch_size), self._compiled_batch_rows(cache_model))
        # ``output_width == input_width <= row_bound`` (a token budget no larger
        # than the row count) is the undecidable case and falls through to unknown.
        if output_width == input_width and output_width > row_bound:
            return "full_sequence"
        if output_width < input_width:
            return "last_token"
        return "unknown"

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
                return self._batch_output_logits_mode(cache_model, output_shape)
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

    def _warn_live_cache_prefix_incomplete(
        self,
        *,
        req_id: str,
        slot_id: Optional[int],
        live_tokens: Optional[int],
        target_tokens: int,
    ) -> None:
        """Report a request cache that holds fewer tokens than the scheduler expects.

        A request's cache keeps every token the worker computed for it, so a
        shorter cache means the worker and the scheduler disagree about the
        request's progress. The caller continues from the tokens the cache
        holds (or a longer snapshot), which is correct, but the divergence is a
        bug signal and must be visible in the log.
        """
        self._live_cache_prefix_incomplete_count = getattr(self, "_live_cache_prefix_incomplete_count", 0) + 1
        logger.warning(
            "MBLT request cache holds fewer tokens than the scheduler expects for req_id=%s "
            "(cache_id=%s, live_tokens=%s, expected=%d). Recomputing the rest of the prefix. occurrences=%d",
            req_id,
            slot_id,
            live_tokens,
            target_tokens,
            self._live_cache_prefix_incomplete_count,
        )

    def _check_text_mxq_supported(self) -> None:
        """Refuse a text MXQ the transformers-mblt forward cannot run, at load time rather than on the first batch.

        transformers-mblt's batched deepstack forward takes only a Qwen3-VL text MXQ with a RoPE input; every
        Qwen3-VL Batch16 release has one.
        """
        if self._supports_deepstack_input() and self._is_batch_model() and not self._text_mxq_uses_rope_input():
            raise RuntimeError(
                "Batched Qwen3-VL text MXQs without a RoPE input are unsupported; "
                "use a dynamic text MXQ, which has one."
            )

    def _is_audio_text_model(self) -> bool:
        model_type = getattr(getattr(getattr(self, "model_config", None), "hf_config", None), "model_type", None)
        return model_type == "mobilint-qwen3_asr"

    def _forward_model(self) -> Any:
        """What ``_forward_rows`` calls: the text decoder of an image-text or audio-text model, which takes the merged
        embeddings, else the causal LM itself.

        Qwen3-ASR's decoder sits under its thinker; given a plain request cache rather than a beam cache, it decodes
        one sequence like a text model.
        """
        if self._is_multimodal_model():
            return self._language_model()
        if self._is_audio_text_model():
            return self.model.thinker.model
        return self.model

    def _vl_forward_inputs(
        self,
        slot_ids: Sequence[Optional[int]],
        cache_sizes: Sequence[int],
        lengths: Sequence[int],
        attention_mask: torch.Tensor,
        deepstack_batch: Optional[Sequence[Optional[np.ndarray]]],
    ) -> dict[str, Any]:
        """Qwen3-VL's extra forward inputs for left-padded rows: M-RoPE positions and deepstack features.

        Positions come from each row's request (its prompt's M-RoPE positions, then ``index + delta``). The
        deepstack features are the worker's dense ``(layers, tokens, hidden)`` rows, zero at text tokens; passing
        every real token as "visual" rebuilds exactly that dense input inside the forward.
        """
        kwargs: dict[str, Any] = {}
        rows, width = attention_mask.shape
        if self._text_mxq_uses_rope_input():
            position_ids = torch.zeros((3, rows, width), dtype=torch.long)
            for row, (slot_id, start, length) in enumerate(zip(slot_ids, cache_sizes, lengths)):
                entry = self._slot_caches.get(self._slot_key(slot_id))
                req_state = None if entry is None else self.req_states.get(entry[0])
                row_positions = self._row_position_ids(req_state, int(start), int(start) + length)
                position_ids[:, row, width - length :] = row_positions
            kwargs["position_ids"] = position_ids
        if deepstack_batch is not None and any(deepstack is not None for deepstack in deepstack_batch):
            layers, hidden = next(deepstack.shape for deepstack in deepstack_batch if deepstack is not None)[::2]
            per_row = [
                torch.zeros((layers, length, hidden), dtype=torch.float32)
                if deepstack is None
                else torch.from_numpy(np.ascontiguousarray(deepstack, dtype=np.float32))
                for deepstack, length in zip(deepstack_batch, lengths)
            ]
            kwargs["visual_pos_masks"] = attention_mask.bool()
            kwargs["deepstack_visual_embeds"] = [
                torch.cat([row[layer] for row in per_row], dim=0) for layer in range(layers)
            ]
        return kwargs

    def _forward_rows(
        self,
        slot_ids: Sequence[Optional[int]],
        cache_sizes: Sequence[int],
        input_embeds_batch: Sequence[np.ndarray],
        keep_sequence_logits: Sequence[bool],
        deepstack_batch: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> list[InferenceLogits]:
        """Run one step of several requests through the model's forward, each on its own cache.

        Row ``i`` continues ``slot_ids[i]``'s request cache from ``cache_sizes[i]`` tokens (rewinding it if needed).
        transformers-mblt binds the caches to their NPU slots, splits long inputs to the MXQ width, packs the batch
        and advances every cache. Per-position logits come back only for rows that ask for them and only from a
        full-logits MXQ; the others get their last-token logits.
        """
        caches = []
        for slot_id, cache_size in zip(slot_ids, cache_sizes):
            cache = self._slot_cache(slot_id)
            if cache is None:  # prefill-cost calibration, before any request
                cache = self.request_caches.new_cache(self._slot_key(slot_id))
            if int(cache_size) > cache.get_seq_length():
                raise RuntimeError(
                    f"Cannot infer from position {cache_size}: the request's cache holds "
                    f"{cache.get_seq_length()} tokens."
                )
            cache.set_seq_length(int(cache_size))
            caches.append(cache)

        lengths = [int(embeds.shape[0]) for embeds in input_embeds_batch]
        keep_all = any(keep_sequence_logits) and self._runtime_output_logits_mode(max(lengths)) == "full_sequence"
        rows, width = len(lengths), max(lengths)
        inputs_embeds = torch.zeros((rows, width, int(input_embeds_batch[0].shape[-1])), dtype=torch.float32)
        attention_mask = torch.zeros((rows, width), dtype=torch.long)
        for row, embeds in enumerate(input_embeds_batch):
            # Left-padded, so every row's last token sits in the last column: ``logits_to_keep=1`` is then each row's
            # own next-token logits whichever way the forward reads it.
            start = width - lengths[row]
            inputs_embeds[row, start:] = torch.from_numpy(np.ascontiguousarray(embeds, dtype=np.float32))
            attention_mask[row, start:] = 1
        extra = {}
        if self._is_multimodal_model():
            extra = self._vl_forward_inputs(slot_ids, cache_sizes, lengths, attention_mask, deepstack_batch)
        if not isinstance(self.cache_model, MbltMoECacheModel):
            # vLLM already sized the step for the MXQ (chunked prefill): feed it in one infer rather than
            # re-splitting it at the model's own prefill chunk size. The split changes the NPU's numerics as well
            # as the number of infers. A Mixture-of-Experts forward keeps its own split: its
            # shared MXQs take at most that many tokens per infer.
            extra["npu_prefill_chunk_size"] = width
        with torch.no_grad():
            output = self._forward_model()(
                inputs_embeds=inputs_embeds,
                # A single row needs no mask: a batch MXQ makes one up, a non-batch MXQ takes the single-row path.
                attention_mask=attention_mask if rows > 1 else None,
                # A single request runs on its own cache (a Mixture-of-Experts cache cannot be joined).
                past_key_values=caches[0] if rows == 1 else join_request_caches(caches),
                use_cache=True,
                logits_to_keep=0 if keep_all else 1,
                **extra,
            )
        # A causal LM returns ``logits``; an image-text model's text decoder returns them as ``last_hidden_state``.
        raw = output.logits if getattr(output, "logits", None) is not None else output.last_hidden_state
        logits = raw.reshape(rows, -1, raw.shape[-1]).to(torch.float32).numpy()
        results = []
        for row, length in enumerate(lengths):
            # With ``logits_to_keep=0`` the forward returns each row's own positions from column 0 (padding dropped by
            # the mask). Copies, so what a row keeps does not hold the whole batched output alive.
            sequence = logits[row, :length].copy() if keep_all and keep_sequence_logits[row] else None
            last_token = sequence[-1] if sequence is not None else logits[row, length - 1 if keep_all else 0].copy()
            results.append(InferenceLogits(last_token_logits=last_token, full_sequence_logits=sequence))
        return results

    def _infer_logits(
        self,
        input_embeds: np.ndarray,
        deepstack_embeds: Optional[np.ndarray],
        cache_size: int,
    ) -> np.ndarray:
        deepstack = None if deepstack_embeds is None else [deepstack_embeds]
        return self._forward_rows([None], [cache_size], [input_embeds], [False], deepstack)[0].last_token_logits

    def _infer_logits_with_sequence(
        self,
        input_embeds: np.ndarray,
        deepstack_embeds: Optional[np.ndarray],
        cache_size: int,
        keep_sequence_logits: bool = True,
    ) -> InferenceLogits:
        deepstack = None if deepstack_embeds is None else [deepstack_embeds]
        return self._forward_rows([None], [cache_size], [input_embeds], [keep_sequence_logits], deepstack)[0]

    def _infer_logits_batch(
        self,
        input_embeds_batch: list[np.ndarray],
        cache_sizes: list[int],
        cache_ids: list[int],
        deepstack_embeds_batch: Optional[list[Optional[np.ndarray]]] = None,
    ) -> list[np.ndarray]:
        return [
            output.last_token_logits
            for output in self._infer_logits_batch_with_sequence(
                input_embeds_batch,
                cache_sizes,
                cache_ids,
                deepstack_embeds_batch,
                keep_sequence_logits=[False] * len(input_embeds_batch),
            )
        ]

    def _infer_logits_batch_with_sequence(
        self,
        input_embeds_batch: list[np.ndarray],
        cache_sizes: list[int],
        cache_ids: list[int],
        deepstack_embeds_batch: Optional[list[Optional[np.ndarray]]] = None,
        keep_sequence_logits: Optional[list[bool]] = None,
    ) -> list[InferenceLogits]:
        if not input_embeds_batch:
            return []
        if not (len(input_embeds_batch) == len(cache_sizes) == len(cache_ids)):
            raise RuntimeError("Batched infer inputs, cache sizes, and cache IDs must have identical lengths.")
        keep = [True] * len(input_embeds_batch) if keep_sequence_logits is None else keep_sequence_logits
        return self._forward_rows(cache_ids, cache_sizes, input_embeds_batch, keep, deepstack_embeds_batch)

    def _infer_normal_logits_batch_chunked(
        self,
        output_indices: list[int],
        input_embeds_batch: list[np.ndarray],
        cache_sizes: list[int],
        cache_ids: list[int],
        deepstack_embeds_batch: Optional[list[Optional[np.ndarray]]] = None,
        keep_sequence_logits: Optional[list[bool]] = None,
    ) -> dict[int, InferenceLogits]:
        """Forward normal batch rows in chunks.

        ``keep_sequence_logits`` marks the rows whose per-position logits are
        read afterwards (prompt logprobs).  A full-logits MXQ returns one
        vocab row per input token, so the others drop theirs and keep only a
        copy of the last row instead of the whole chunk output.
        """
        if not output_indices:
            return {}
        if keep_sequence_logits is None:
            keep_sequence_logits = [True] * len(output_indices)
        if not (len(output_indices) == len(input_embeds_batch) == len(cache_sizes) == len(cache_ids)):
            raise RuntimeError(
                "Normal batch chunk inputs must have identical lengths: "
                f"indices={len(output_indices)}, input_embeds={len(input_embeds_batch)}, "
                f"cache_sizes={len(cache_sizes)}, cache_ids={len(cache_ids)}"
            )
        if deepstack_embeds_batch is not None and len(deepstack_embeds_batch) != len(input_embeds_batch):
            raise RuntimeError(
                "Normal batch deepstack inputs must match input embeddings: "
                f"input_embeds={len(input_embeds_batch)}, deepstack={len(deepstack_embeds_batch)}"
            )
        if len(keep_sequence_logits) != len(output_indices):
            raise RuntimeError(
                "Normal batch sequence-logit flags must match output indices: "
                f"indices={len(output_indices)}, keep_sequence_logits={len(keep_sequence_logits)}"
            )

        token_cap = self._normal_batch_chunk_token_cap()
        states = [
            NormalBatchChunkState(
                output_index=output_index,
                input_embeds=input_embeds,
                deepstack_embeds=None if deepstack_embeds_batch is None else deepstack_embeds,
                cache_size=cache_size,
                cache_id=cache_id,
                full_sequence_logits=[] if keep else None,
            )
            for output_index, input_embeds, deepstack_embeds, cache_size, cache_id, keep in zip(
                output_indices,
                input_embeds_batch,
                [None] * len(input_embeds_batch) if deepstack_embeds_batch is None else deepstack_embeds_batch,
                cache_sizes,
                cache_ids,
                keep_sequence_logits,
            )
        ]

        while True:
            active_states = [state for state in states if state.offset < int(state.input_embeds.shape[0])]
            if not active_states:
                break

            for batch_start in range(0, len(active_states), self.max_batch_size):
                batch_states = active_states[batch_start : batch_start + self.max_batch_size]
                chunk_embeds_batch: list[np.ndarray] = []
                chunk_deepstack_batch: list[Optional[np.ndarray]] = []
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
                    chunk_deepstack_batch.append(
                        None if state.deepstack_embeds is None else state.deepstack_embeds[:, chunk_start:chunk_end, :]
                    )
                    chunk_cache_sizes.append(state.cache_size)
                    chunk_cache_ids.append(state.cache_id)

                if not chunk_embeds_batch:
                    continue

                infer_kwargs: dict[str, object] = {}
                if any(deepstack_embeds is not None for deepstack_embeds in chunk_deepstack_batch):
                    infer_kwargs["deepstack_embeds_batch"] = chunk_deepstack_batch
                logits_batch = self._infer_logits_batch_with_sequence(
                    input_embeds_batch=chunk_embeds_batch,
                    cache_sizes=chunk_cache_sizes,
                    cache_ids=chunk_cache_ids,
                    keep_sequence_logits=[state.full_sequence_logits is not None for state in batch_states],
                    **infer_kwargs,
                )
                for state, chunk_embeds, inference_logits in zip(batch_states, chunk_embeds_batch, logits_batch):
                    chunk_len = int(chunk_embeds.shape[0])
                    state.offset += chunk_len
                    state.cache_size += chunk_len
                    state.last_token_logits = inference_logits.last_token_logits
                    if state.full_sequence_logits is None and inference_logits.full_sequence_logits is not None:
                        # A view would keep the whole [tokens, vocab] chunk output alive.
                        state.last_token_logits = state.last_token_logits.copy()
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
            return self._resolve_vocab_size()
        return int(num_prompt_logprobs)

    def _resolve_vocab_size(self) -> int:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        candidates = (
            (getattr(self.model, "config", None), "vocab_size", "model.config.vocab_size"),
            (getattr(self.model, "config", None), "text_config.vocab_size", "model.config.text_config.vocab_size"),
            (getattr(self.model_config, "hf_config", None), "vocab_size", "model_config.hf_config.vocab_size"),
            (
                getattr(self.model_config, "hf_config", None),
                "text_config.vocab_size",
                "model_config.hf_config.text_config.vocab_size",
            ),
            # Composite audio-text models (Qwen3-ASR) nest the language model under
            # a thinker sub-config.
            (
                getattr(self.model, "config", None),
                "thinker_config.text_config.vocab_size",
                "model.config.thinker_config.text_config.vocab_size",
            ),
            (
                getattr(self.model_config, "hf_config", None),
                "thinker_config.text_config.vocab_size",
                "model_config.hf_config.thinker_config.text_config.vocab_size",
            ),
        )
        for root, path, _source in candidates:
            if root is None:
                continue
            vocab_size = _positive_int_attr(root, path)
            if vocab_size is not None:
                return vocab_size

        searched = ", ".join(source for _root, _path, source in candidates)
        raise RuntimeError(f"Unable to resolve a positive vocab_size for MBLT sampling state. Checked: {searched}.")

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
        own_prefix = self._own_prefix_length(cache_id)
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
                logits = self._infer_logits_batch(
                    input_embeds_batch=[input_embeds],
                    cache_sizes=[fallback_cache_size],
                    cache_ids=[cache_id],
                    deepstack_embeds_batch=None if input_deepstack is None else [input_deepstack],
                )[0]
            else:
                logits = self._infer_logits(input_embeds, input_deepstack, cache_size=fallback_cache_size)
            rows.append(np.asarray(logits).reshape(-1))
            fallback_cache_size += int(input_embeds.shape[0])

        self._restore_own_prefix_length(cache_id, own_prefix)

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
        work for different requests is submitted together in one batched forward.
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

        own_prefixes = [(state.cache_id, self._own_prefix_length(state.cache_id)) for state in states]
        while True:
            active_states = [state for state in states if state.current_prompt_pos < state.prompt_end]
            if not active_states:
                break

            for batch_start in range(0, len(active_states), self.max_batch_size):
                batch_states = active_states[batch_start : batch_start + self.max_batch_size]
                input_embeds_batch: list[np.ndarray] = []
                deepstack_embeds_batch: list[Optional[np.ndarray]] = []
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
                    input_deepstack = (
                        state.req_state.prompt_deepstack_embeds[:, input_start:input_end, :]
                        if state.req_state.prompt_deepstack_embeds is not None
                        else None
                    )
                    input_embeds_batch.append(input_embeds)
                    deepstack_embeds_batch.append(input_deepstack)
                    cache_sizes.append(state.fallback_cache_size)
                    cache_ids.append(state.cache_id)

                infer_kwargs: dict[str, object] = {}
                if any(deepstack_embeds is not None for deepstack_embeds in deepstack_embeds_batch):
                    infer_kwargs["deepstack_embeds_batch"] = deepstack_embeds_batch
                logits_batch = self._infer_logits_batch(
                    input_embeds_batch=input_embeds_batch,
                    cache_sizes=cache_sizes,
                    cache_ids=cache_ids,
                    **infer_kwargs,
                )

                for state, input_embeds, logits in zip(batch_states, input_embeds_batch, logits_batch):
                    state.rows.append(np.asarray(logits).reshape(-1))
                    state.fallback_cache_size += int(input_embeds.shape[0])
                    state.current_prompt_pos += 1

        for cache_id, own_prefix in own_prefixes:
            self._restore_own_prefix_length(cache_id, own_prefix)

        return {state.output_index: np.stack(state.rows, axis=0) for state in states if state.rows}

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
                [(state.req_id, state.start_idx, state.scheduled_end, state.prompt_logits_end) for state in states],
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
                deepstack_embeds_batch = [
                    self._build_deepstack_input_embeds(state.req_state, state.cache_size, state.cache_size + 1)
                    for state in batch_states
                ]
                cache_sizes = [state.cache_size for state in batch_states]
                cache_ids = [state.cache_id for state in batch_states]

                infer_kwargs: dict[str, object] = {}
                if any(deepstack_embeds is not None for deepstack_embeds in deepstack_embeds_batch):
                    infer_kwargs["deepstack_embeds_batch"] = deepstack_embeds_batch
                logits_batch = self._infer_logits_batch(
                    input_embeds_batch=input_embeds_batch,
                    cache_sizes=cache_sizes,
                    cache_ids=cache_ids,
                    **infer_kwargs,
                )

                for state, logits in zip(batch_states, logits_batch):
                    logits_row = np.asarray(logits).reshape(-1)
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
            logits_row = np.asarray(logits).reshape(-1)

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
        """Put the request's KV prefix on its slot for this step; return how many tokens of it the slot holds.

        The request's own cache is used when it holds the prefix. Otherwise the longest matching prefix snapshot
        is copied into it when loading beats recomputing, or the prefix is recomputed.
        """
        if self._is_batch_model() and slot_id is None:
            raise RuntimeError(f"Batch execution requires a cache slot for req_id={req_id}.")
        slot_label = f" slot={slot_id}" if slot_id is not None else ""
        target_tokens = req_state.num_computed_tokens
        previous = self._slot_caches.get(self._slot_key(slot_id))
        if previous is not None and previous[0] != req_id:
            # Publish the outgoing request's prefix first, so this request can start from it.
            self._snapshot_switched_out_request(previous[0], next_req_id=req_id, print_debug=print_debug)

        if self._should_recompute_prompt_logprobs_from_start(req_state):
            if print_debug:
                print(f"[cache] req={req_id}{slot_label} bypass-prefix-cache-for-prompt-logprobs")
            matched_tokens = 0
        else:
            result = self._load_request_prefix(self._runtime_cache_request(req_id, req_state, slot_id), req_state)
            matched_tokens = result.matched_tokens
            if print_debug:
                if result.action == "skip-empty":
                    print(f"[cache] req={req_id}{slot_label} skip-load target_tokens=0")
                elif result.action == "reuse-live":
                    print(f"[cache] req={req_id}{slot_label} reuse-live-cache matched={target_tokens}/{target_tokens}")
                elif result.action in ("load-own", "load-shared", "take-over", "skip-cost"):
                    action = "skip-load-cost" if result.action == "skip-cost" else result.action
                    print(f"[cache] req={req_id}{slot_label} {action} matched={matched_tokens}/{target_tokens}")
                elif result.action == "live-prefix-incomplete":
                    print(
                        f"[cache] req={req_id}{slot_label} live-prefix-incomplete "
                        f"live={result.live_cache_tokens} expected={target_tokens} "
                        f"matched={matched_tokens}/{target_tokens}"
                    )
                else:
                    print(f"[cache] req={req_id}{slot_label} cache-miss fallback matched=0/{target_tokens}")
        return matched_tokens

    def _start_request_step(
        self,
        req_id: str,
        req_state: RequestState,
        cache_size: int,
        slot_id: Optional[int] = None,
    ) -> None:
        """Make this step's infers on the request's slot run on its cache, from ``cache_size`` tokens."""
        active = self._slot_caches.get(self._slot_key(slot_id))
        if active is None or active[1] is not req_state.kv_cache:
            # The slot changes hands: settle the finished snapshot it may still hold before the bind evicts it.
            self._settle_snapshot_on_slot(slot_id)
        self._activate_request_cache(req_id, req_state, slot_id)
        self._prepare_slot(slot_id, cache_size)

    def _load_request_prefix(self, request: RuntimeCacheRequest, req_state: RequestState) -> RuntimeCacheLoadResult:
        """Choose where the request's KV prefix comes from: its own cache, a prefix snapshot, or a recompute."""
        slot_id = request.cache_slot_id
        own_cache = self._request_cache(req_state, slot_id)
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            return RuntimeCacheLoadResult(matched_tokens=0, cache_miss=True, action="skip-empty")

        self.runtime_cache.observe_request_blocks(
            request.req_id,
            request.block_ids,
            first_seq_block_hashes=request.first_seq_block_hashes,
        )

        own_tokens = int(own_cache.get_seq_length())
        if own_tokens >= target_tokens:
            return RuntimeCacheLoadResult(
                matched_tokens=target_tokens,
                reused_live_cache=True,
                action="reuse-live",
                snapshot_req_id=request.req_id,
                live_cache_tokens=own_tokens,
            )
        if own_tokens > 0:
            self._warn_live_cache_prefix_incomplete(
                req_id=request.req_id,
                slot_id=slot_id,
                live_tokens=own_tokens,
                target_tokens=target_tokens,
            )
        # Falling short of a snapshot, the request continues from the tokens its own cache holds.
        fallback = RuntimeCacheLoadResult(
            matched_tokens=own_tokens,
            reused_live_cache=own_tokens > 0,
            cache_miss=own_tokens == 0,
            action="live-prefix-incomplete" if own_tokens > 0 else "cache-miss",
            live_prefix_incomplete=own_tokens > 0,
            live_cache_tokens=own_tokens if own_tokens > 0 else None,
        )

        cost_model = self._ensure_prefix_cache_cost_model()
        match = self.runtime_cache.choose_snapshot(request)
        # A finished snapshot still on the request's own slot costs no load or dump: the request continues on it
        # unless another snapshot covers more of the prefix and passes the load test.
        _, on_slot = self._match_resident_snapshot(request, self._resident_finished_snapshots([slot_id]))
        if not (
            on_slot.snapshot is not None
            and on_slot.matched_tokens > own_tokens
            and cost_model.should_take_over(matched_tokens=on_slot.matched_tokens)
        ):
            on_slot = None

        def take_over_or(result: RuntimeCacheLoadResult) -> RuntimeCacheLoadResult:
            if on_slot is None:
                return result
            self.runtime_cache.refresh_snapshot_recency(on_slot.req_id)
            self._restore_snapshot(req_state, slot_id, on_slot.snapshot, on_slot.matched_tokens)
            return RuntimeCacheLoadResult(
                matched_tokens=on_slot.matched_tokens,
                loaded=True,
                loaded_snapshot_req_id=on_slot.req_id,
                action="take-over",
                snapshot_req_id=on_slot.req_id,
                live_prefix_incomplete=own_tokens > 0,
            )

        if on_slot is not None and on_slot.matched_tokens >= match.matched_tokens:
            return take_over_or(fallback)
        if match.snapshot is None or match.matched_tokens <= own_tokens:
            return take_over_or(fallback)
        if not cost_model.should_load(matched_tokens=match.matched_tokens, cache_id=slot_id):
            fallback.action = "skip-cost"
            fallback.loaded_snapshot_req_id = match.req_id
            fallback.is_own_snapshot = match.is_own_snapshot
            fallback.snapshot_req_id = match.req_id
            return take_over_or(fallback)
        match = self.runtime_cache.verify_prompt_embed_match(request, match)
        if match.snapshot is None or match.matched_tokens <= own_tokens:
            return take_over_or(fallback)
        if on_slot is not None and on_slot.matched_tokens >= match.matched_tokens:
            return take_over_or(fallback)
        if not cost_model.should_load(matched_tokens=match.matched_tokens, cache_id=slot_id):
            fallback.action = "skip-cost"
            fallback.loaded_snapshot_req_id = match.req_id
            fallback.is_own_snapshot = match.is_own_snapshot
            fallback.snapshot_req_id = match.req_id
            return take_over_or(fallback)

        if match.req_id is not None:
            self.runtime_cache.refresh_snapshot_recency(match.req_id)
        self._restore_snapshot(req_state, slot_id, match.snapshot, match.matched_tokens)
        return RuntimeCacheLoadResult(
            matched_tokens=match.matched_tokens,
            loaded=True,
            loaded_snapshot_req_id=match.req_id,
            is_own_snapshot=match.is_own_snapshot,
            action="load-own" if match.is_own_snapshot else "load-shared",
            snapshot_req_id=match.req_id,
            live_prefix_incomplete=own_tokens > 0,
        )

    def _restore_snapshot(
        self,
        req_state: RequestState,
        slot_id: Optional[int],
        snapshot: RuntimeCacheSnapshot,
        num_tokens: int,
    ) -> None:
        """Give the request a cache holding the snapshot's first ``num_tokens`` tokens, on the request's slot."""
        slot_key = self._slot_key(slot_id)
        snapshot_cache = snapshot.blobs
        start = time.perf_counter()
        if self.request_caches.holds_slot(snapshot_cache, slot_key):
            # The finished request that left this snapshot ran on this slot and nothing has used it since: the
            # request takes over that cache, KV in place. The snapshot keeps a host copy only with host offload on
            # and when a later load of it could beat recomputing; otherwise it is dropped from the index, and the
            # copy's dump is skipped (issue #40).
            cost_model = self._ensure_prefix_cache_cost_model()
            snapshot_req_id = self._snapshot_req_id(snapshot)
            if cost_model.should_save_snapshot(num_tokens=snapshot.num_tokens):
                snapshot.blobs = snapshot_cache.copy()
                cost_model.observe_dump(slot_id, (time.perf_counter() - start) * 1000.0)
            else:
                if snapshot_req_id is not None:
                    self.runtime_cache.remove_snapshot(snapshot_req_id)
                snapshot.blobs = None  # the cache is the request's now
            snapshot_cache.set_seq_length(int(num_tokens))
            req_state.kv_cache = snapshot_cache
            return
        self._settle_snapshot_on_slot(slot_id)
        cache = self.request_caches.relocate(snapshot_cache, slot_key, num_tokens)
        self.request_caches.prepare(cache, slot_key, num_tokens)
        req_state.kv_cache = cache
        self._ensure_prefix_cache_cost_model().observe_load(slot_id, (time.perf_counter() - start) * 1000.0)

    def _snapshot_req_id(self, snapshot: RuntimeCacheSnapshot) -> Optional[str]:
        for req_id, stored in self.runtime_cache.snapshots.items():
            if stored is snapshot:
                return req_id
        return None

    def _settle_snapshot_on_slot(self, slot_id: Optional[int]) -> None:
        """Before another cache takes ``slot_id``, drop the prefix snapshot it still holds, or save it if opted in.

        A finished request's cache stays on its slot as its snapshot, so the next cache bound to the slot would
        dump the slot's whole KV to host memory. That dump runs on the engine loop (issue #40). By default the
        snapshot is dropped from the index and emptied, so the bind that follows overwrites the slot without
        saving anything. With host offload on, it is saved instead when a later load of it could beat
        recomputing it. Live requests are never touched: a snapshot that holds a slot is a finished request's
        cache, which no request runs on.
        """
        if self.request_caches is None:
            return
        slot_key = self._slot_key(slot_id)
        cost_model = self._ensure_prefix_cache_cost_model()
        print_debug = getattr(self, "print_debug", False)
        resident = [
            (req_id, snapshot)
            for req_id, snapshot in self.runtime_cache.snapshots.items()
            if snapshot.blobs is not None and self.request_caches.holds_slot(snapshot.blobs, slot_key)
        ]
        for req_id, snapshot in resident:
            if req_id in self.req_states:
                continue
            if cost_model.should_save_snapshot(num_tokens=snapshot.num_tokens):
                start = time.perf_counter()
                snapshot.blobs.offload()
                cost_model.observe_dump(slot_id, (time.perf_counter() - start) * 1000.0)
                if print_debug:
                    print(f"[cache] req={req_id} slot={slot_id} save-snapshot tokens={snapshot.num_tokens}")
                continue
            self.runtime_cache.remove_snapshot(req_id)
            # Emptied, the cache has nothing to save when the next bind evicts it from the slot.
            snapshot.blobs.reset()
            snapshot.blobs = None
            if print_debug:
                reason = "cost" if cost_model.host_offload else "device-only"
                print(f"[cache] req={req_id} slot={slot_id} drop-snapshot tokens={snapshot.num_tokens} reason={reason}")

    def _snapshot_switched_out_request(self, req_id: str, *, next_req_id: str, print_debug: bool = False) -> None:
        """Keep the prefix of a request that is about to lose the non-batch slot, so other requests can start from it.

        Its own KV is safe either way: the slot's owner table saves it to the request's cache when the next request
        binds the slot.
        """
        if not self.runtime_cache.enable_prefix_caching:
            return
        req_state = self.req_states.get(req_id)
        if req_state is None or req_state.kv_cache is None or req_state.kv_cache.get_seq_length() <= 0:
            return
        if not self.runtime_cache.should_dump_snapshot_after_step(req_id, req_state.num_computed_tokens):
            return
        if req_state.kv_cache.is_resident:
            # Save it from the slot now, which the next request's bind would do anyway; the copy then shares it.
            start = time.perf_counter()
            req_state.kv_cache.offload()
            self._ensure_prefix_cache_cost_model().observe_dump(None, (time.perf_counter() - start) * 1000.0)
        dumped = self._dump_snapshot(
            req_id=req_id,
            req_state=req_state,
            next_num_tokens=req_state.num_computed_tokens,
            print_debug=print_debug,
        )
        if dumped and print_debug:
            print(f"[cache] req={req_id} dump-before-switch next={next_req_id}")

    def _dump_snapshot(
        self,
        req_id: str,
        req_state: RequestState,
        next_num_tokens: int,
        slot_id: Optional[int] = None,
        print_debug: bool = False,
    ) -> bool:
        """Store a copy of the request's first ``next_num_tokens`` tokens as its prefix snapshot."""
        cache = req_state.kv_cache
        live_tokens = int(cache.get_seq_length()) if cache is not None else 0
        # A snapshot must never advertise more tokens than its cache holds. The scheduler's
        # num_computed_tokens is the caller's label of choice, but the request's cache can legitimately hold a
        # shorter prefix (a rebuilt prefix). Labeling the longer count would let a later load resume past KV
        # that is not there.
        if live_tokens < max(0, int(next_num_tokens)):
            if live_tokens <= 0:
                return False
            if not self.runtime_cache.should_dump_snapshot_after_step(req_id, live_tokens):
                return False
            if print_debug:
                print(
                    f"[cache] req={req_id} slot={slot_id} dump-clamped "
                    f"tokens={live_tokens} requested={max(0, int(next_num_tokens))}"
                )
            next_num_tokens = live_tokens

        resident = bool(cache.is_resident)
        start = time.perf_counter()
        snapshot_cache = cache.copy()  # saves the KV from the slot first when the slot still holds it
        if resident:
            self._ensure_prefix_cache_cost_model().observe_dump(slot_id, (time.perf_counter() - start) * 1000.0)
        snapshot_cache.set_seq_length(int(next_num_tokens))
        self._store_snapshot(req_id, req_state, snapshot_cache, next_num_tokens)
        if print_debug:
            num_blocks = len(req_state.first_seq_blocks)
            print(
                f"[cache] req={req_id} dump tokens={max(0, int(next_num_tokens))} "
                f"blocks={num_blocks} snapshots={self.runtime_cache.snapshot_count()}"
            )
        return True

    def _store_snapshot(self, req_id: str, req_state: RequestState, cache: Any, num_tokens: int) -> None:
        self.runtime_cache.store_snapshot(
            req_id=req_id,
            blobs=cache,
            block_ids=req_state.block_ids,
            first_seq_blocks=req_state.first_seq_blocks,
            first_seq_block_hashes=req_state.first_seq_block_hashes,
            num_tokens=num_tokens,
            cache_token_ids=self._cache_token_ids(req_state, num_tokens),
            multimodal_cache_identity=self._cache_multimodal_identity(req_state, num_tokens),
            prompt_embed_cache_identity=self._make_prompt_embed_cache_identity(req_state),
        )

    @staticmethod
    def _cache_token_ids(req_state: RequestState, num_tokens: int) -> tuple[int, ...] | None:
        token_ids = list(req_state.prompt_token_ids) + list(req_state.output_token_ids)
        num_tokens = max(0, int(num_tokens))
        if len(token_ids) < num_tokens:
            return None
        return tuple(token_ids[:num_tokens])

    @staticmethod
    def _cache_multimodal_identity(req_state: RequestState, num_tokens: int) -> Hashable | None:
        if not req_state.is_multimodal:
            return None
        if req_state.multimodal_cache_identity is None:
            return None
        if max(0, int(num_tokens)) <= 0:
            return None
        return req_state.multimodal_cache_identity

    @staticmethod
    def _fingerprint_prompt_embed_prefix(
        prompt_embeds: np.ndarray,
        num_tokens: int,
    ) -> Hashable | None:
        num_tokens = max(0, int(num_tokens))
        if num_tokens <= 0 or num_tokens > int(prompt_embeds.shape[0]):
            return None
        prefix = prompt_embeds[:num_tokens]
        if prefix.dtype == object:
            return None
        if not prefix.flags.c_contiguous:
            prefix = np.ascontiguousarray(prefix)
        digest = hashlib.sha256()
        digest.update(str(prefix.dtype).encode("utf-8"))
        digest.update(repr(tuple(int(dim) for dim in prefix.shape)).encode("utf-8"))
        digest.update(prefix.tobytes())
        return ("sha256", tuple(int(dim) for dim in prefix.shape), str(prefix.dtype), digest.hexdigest())

    def _make_prompt_embed_cache_identity(
        self,
        req_state: RequestState,
    ) -> PromptEmbedCacheIdentity | None:
        if not req_state.explicit_prompt_embeds:
            return None
        prompt_len = max(0, int(req_state.prompt_len))
        if prompt_len <= 0:
            return None
        prompt_embeds = req_state.prompt_embeds
        fingerprints: dict[int, Hashable | None] = {}

        def fingerprint_for_prefix(num_tokens: int) -> Hashable | None:
            num_tokens = max(0, int(num_tokens))
            if num_tokens not in fingerprints:
                fingerprints[num_tokens] = self._fingerprint_prompt_embed_prefix(prompt_embeds, num_tokens)
            return fingerprints[num_tokens]

        return PromptEmbedCacheIdentity(
            prompt_len=prompt_len,
            fingerprint_for_prefix=fingerprint_for_prefix,
        )

    def _finalize_finished_request(
        self,
        req_id: str,
        print_debug: bool = False,
    ) -> None:
        finished_req_state = self.req_states.pop(req_id, None)
        finished_slot_id = finished_req_state.cache_slot_id if finished_req_state is not None else None
        slot_key = self._slot_key(finished_slot_id)
        active = self._slot_caches.get(slot_key)
        if active is not None and active[0] == req_id:
            del self._slot_caches[slot_key]
        cache = finished_req_state.kv_cache if finished_req_state is not None else None
        # The finished request's cache becomes its prefix snapshot as it is: its KV stays on the slot and is
        # saved to host memory only when another cache takes the slot, or never if the snapshot is dropped
        # first. Without prefix caching nothing can load it, so it is dropped (issue #28).
        if cache is not None and self._prefix_caching_enabled():
            num_tokens = min(int(finished_req_state.num_computed_tokens), int(cache.get_seq_length()))
            if num_tokens > 0:
                cache.set_seq_length(num_tokens)
                self._store_snapshot(req_id, finished_req_state, cache, num_tokens)
                if print_debug:
                    print(f"[cache] req={req_id} slot={finished_slot_id} keep-on-finish tokens={num_tokens}")
        for evicted_req_id in self.runtime_cache.mark_snapshot_finished(req_id):
            if print_debug:
                print(f"[cache] evict-finished req={evicted_req_id} reason=lru-cap")
        if self._is_batch_model():
            self._release_cache_slot(req_id)

    def init_device(self) -> None:
        self._log_init_stage("init_device")
        apply_engine_host_thread_defaults(self.vllm_config)
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
        sync_dynamic_vision_from_model = getattr(self.model, "sync_dynamic_vision_from_model", None)
        if callable(sync_dynamic_vision_from_model):
            sync_dynamic_vision_from_model()
        self._log_init_stage(
            "load_model:after_from_pretrained",
            start,
            model_type=type(self.model).__name__,
        )

        start = time.perf_counter()
        self._log_init_stage("load_model:before_eval")
        self.model.eval()
        self._log_init_stage("load_model:after_eval", start)

        if is_eagle3_model(self.model):
            # The runner keeps one MobilintEagle3Cache per request; the worker's request caches, prefix snapshots and
            # sampler stay unused.
            self.input_embeddings = self.model.get_input_embeddings()
            self.eagle3_runner = Eagle3Runner(self.model, max_model_len=self.max_seq_len)
            self._log_init_stage("load_model:done", eagle3=True)
            return

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
        if is_mixture_of_experts_model(self.model):
            # The KV cache lives in every per-layer shared MXQ and transformers-mblt runs the whole
            # forward; the adapter only presents it as one single-slot cache model.
            if self._is_batch_model():
                raise RuntimeError(
                    "Mixture-of-Experts models support only non-batch serving: "
                    f"max_batch_size={self.max_batch_size}."
                )
            self.cache_model = MbltMoECacheModel(self.model)
            self.cache_backend = None
            if self.max_seq_len > self.cache_model.max_cache_size:
                raise RuntimeError(
                    f"max_model_len={self.max_seq_len} exceeds the KV capacity of the MoE shared MXQs "
                    f"({self.cache_model.max_cache_size} tokens). Pass --max-model-len "
                    f"{self.cache_model.max_cache_size} or lower."
                )
        else:
            self.cache_model = self.model.get_cache_mxq_model()
            self.cache_backend = self._resolve_cache_backend()
        cache_models = self._get_cache_models()
        backend_capacity = len(cache_models) * max(
            1, int(getattr(self.cache_backend, "k_per_model", self.max_batch_size) or 1)
        )
        if backend_capacity < self.max_batch_size:
            raise RuntimeError(
                "Mobilint backend capacity is smaller than the vLLM scheduler capacity: "
                f"backend_capacity={backend_capacity}, max_batch_size={self.max_batch_size}."
            )
        self._log_init_stage(
            "load_model:after_get_cache_mxq_model",
            start,
            cache_model_type=type(self.cache_model).__name__,
            cache_model_slots=len(cache_models),
            cache_slots_per_model=getattr(self.cache_backend, "k_per_model", self.max_batch_size),
        )
        self._check_text_mxq_supported()
        self._reset_cache_slots()
        self._calibrate_prefix_cache_prefill_costs()
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
                    "Sampling penalties are disabled by VLLM_MBLT_ENABLE_SAMPLING_PENALTIES. "
                    "Ignoring frequency_penalty=%s, presence_penalty=%s, repetition_penalty=%s. "
                    "NPU output will not match a GPU reference run that applies them.",
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

        vocab_size = self._resolve_vocab_size()

        return CachedSamplingState(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k if sampling_params.top_k > 0 else vocab_size),
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
            fill_value=self._resolve_vocab_size(),
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
        """Advertise transcription from the model class, not the HF model_type.

        vLLM builds the handlers for /v1/audio/transcriptions and /v1/audio/translations
        from this one flag, and OpenAISpeechToText calls ``model_cls.get_speech_to_text_config()``
        while the app starts. Deriving it from the class that actually serves those
        endpoints keeps the two in step. Follows
        GPUModelRunner.get_supported_generation_tasks(), but checks the registered
        class, since this worker never instantiates it.
        """
        from vllm.model_executor.model_loader import get_model_cls
        from vllm.model_executor.models import supports_transcription

        model_cls = get_model_cls(self.model_config)
        if supports_transcription(model_cls):
            if getattr(model_cls, "supports_transcription_only", False):
                return ("transcription",)
            return ("generate", "transcription")
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
        if self.model is None or (self.cache_model is None and self.eagle3_runner is None):
            raise RuntimeError("MBLT Accelerator/Model is not initialized.")

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    def _get_tracer(self) -> MbltTracer:
        if self._tracer is None:
            trace_dir = resolve_trace_dir()
            if trace_dir is None:
                raise RuntimeError(
                    f"NPU event tracing is not enabled. Set {TRACE_DIR_ENV_VAR} to a directory "
                    "before starting the server to enable it."
                )
            self._tracer = MbltTracer(trace_dir, rank=self.rank)
        return self._tracer

    def profile(self, is_start: bool = True) -> None:
        """Start or stop an NPU event trace.

        vLLM's `Executor.profile()` reaches every worker through
        `collective_rpc("profile", ...)`, so implementing this method is what
        makes `/start_profile`, `/stop_profile`, `LLM.start_profile()` and
        `vllm bench serve --profile` work on this platform. The v1
        `WorkerBase` declares no `profile`, so those paths raise
        `AttributeError` without it.

        A torch profiler would see nothing useful here -- the model runs on the
        NPU through qbruntime, not through torch ops -- so this hook records a
        qbruntime event trace instead, the way each out-of-tree platform points
        it at its own device profiler.

        Failures propagate: if the trace cannot be started, or its log cannot
        be written, the caller hears about it rather than being told the
        profile succeeded and finding no trace on disk. A repeated start or a
        stop with nothing running is not a failure and only logs.
        """
        tracer = self._get_tracer()
        if is_start:
            tracer.start()
        else:
            tracer.stop()

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if self.input_embeddings is None:
            raise RuntimeError("Input embeddings are not initialized.")
        if self.eagle3_runner is not None:
            return self.eagle3_runner.execute(scheduler_output)

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
            first_seq_block_hashes = self._extract_first_seq_block_hashes(new_req)
            self.runtime_cache.observe_request_blocks(
                new_req.req_id,
                normalized_block_ids,
                first_seq_block_hashes=first_seq_block_hashes,
            )
            prompt_embeds_np = self._to_cpu_float32_numpy(prompt_embeds)
            prompt_deepstack_embeds_np = (
                self._to_cpu_float32_numpy(prompt_deepstack_embeds) if prompt_deepstack_embeds is not None else None
            )
            prompt_position_ids, mrope_position_delta = self._build_prompt_position_ids(
                new_req.prompt_token_ids,
                new_req.mm_features,
                int(prompt_embeds_np.shape[0]),
            )
            new_req_state = RequestState(
                is_prefill=True,
                output_token_ids=[],
                sampling_params=sampling_params,
                cached_sampling_state=self._make_cached_sampling_state(
                    sampling_params,
                    new_req.prompt_token_ids,
                ),
                block_ids=normalized_block_ids,
                first_seq_blocks=self._first_seq_blocks(normalized_block_ids),
                first_seq_block_hashes=first_seq_block_hashes,
                num_computed_tokens=new_req.num_computed_tokens,
                num_output_tokens=0,
                prompt_embeds=prompt_embeds_np,
                prompt_deepstack_embeds=prompt_deepstack_embeds_np,
                prompt_position_ids=(
                    None if prompt_position_ids is None else prompt_position_ids.reshape(3, -1).contiguous()
                ),
                mrope_position_delta=mrope_position_delta,
                is_multimodal=bool(new_req.mm_features),
                prompt_len=int(prompt_embeds_np.shape[0]),
                prompt_token_ids=new_req.prompt_token_ids or [],
                cache_slot_id=None,
                vlm_session_id=vlm_session_id,
                multimodal_cache_identity=self._build_vlm_multimodal_cache_identity(
                    vlm_session_id,
                    new_req.mm_features,
                ),
                explicit_prompt_embeds=new_req.prompt_embeds is not None,
            )
            if self._is_batch_model():
                new_req_state.cache_slot_id = self._assign_cache_slot(new_req.req_id, new_req_state)
            self.req_states[new_req.req_id] = new_req_state

        # Continue cached requests
        for i, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
            cached_request_state = self.req_states[req_id]

            # all_token_ids = scheduler_output.scheduled_cached_reqs.all_token_ids[req_id]
            cached_request_state.num_computed_tokens = scheduler_output.scheduled_cached_reqs.num_computed_tokens[i]
            cached_request_state.num_output_tokens = scheduler_output.scheduled_cached_reqs.num_output_tokens[i]

            new_block_ids = scheduler_output.scheduled_cached_reqs.new_block_ids[i]
            if new_block_ids is not None:
                new_block_hashes = self._extract_indexed_first_seq_block_hashes(
                    scheduler_output.scheduled_cached_reqs,
                    i,
                    candidates=("new_block_hashes", "new_kv_block_hashes", "new_block_hash_ids"),
                )
                if req_id in scheduler_output.scheduled_cached_reqs.resumed_req_ids:
                    cached_request_state.block_ids = self._normalize_block_ids(new_block_ids)
                    cached_request_state.first_seq_block_hashes = new_block_hashes
                else:
                    cached_request_state.block_ids = self._append_block_ids(
                        cached_request_state.block_ids,
                        new_block_ids,
                    )
                    cached_request_state.first_seq_block_hashes = self._append_first_seq_block_hashes(
                        cached_request_state.first_seq_block_hashes,
                        new_block_hashes,
                    )
                cached_request_state.first_seq_blocks = self._first_seq_blocks(cached_request_state.block_ids)
                self.runtime_cache.observe_request_blocks(
                    req_id,
                    cached_request_state.block_ids,
                    first_seq_block_hashes=cached_request_state.first_seq_block_hashes,
                )

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
            deepstack_embeds_batch: list[Optional[np.ndarray]] = []
            cache_sizes: list[int] = []
            cache_ids: list[int] = []
            microstep_indices: list[int] = []
            normal_indices: list[int] = []

            for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
                req_state = self.req_states[req_id]
                slot_id = req_state.cache_slot_id
                if slot_id is None:
                    slot_id = self._assign_cache_slot(req_id, req_state)
                    req_state.cache_slot_id = slot_id

                self._ensure_batch_vlm_supported(req_state)

                scheduled_end = req_state.num_computed_tokens + num_scheduled_token
                cache_size = self._load_snapshot_if_needed(
                    req_id,
                    req_state,
                    slot_id=slot_id,
                    print_debug=print_debug,
                )
                self._start_request_step(req_id, req_state, cache_size, slot_id=slot_id)
                input_embeds = self._build_input_embeds(req_state, cache_size, scheduled_end)
                deepstack_embeds = self._build_deepstack_input_embeds(req_state, cache_size, scheduled_end)
                sequence_length = int(input_embeds.shape[0])
                next_cache_size = cache_size + sequence_length
                req_state.is_prefill = scheduled_end < req_state.prompt_len

                req_ids.append(req_id)
                req_id_to_index[req_id] = len(req_ids) - 1
                scheduled_end_positions.append(scheduled_end)
                next_cache_sizes.append(next_cache_size)
                sequence_lengths.append(sequence_length)
                input_embeds_batch.append(input_embeds)
                deepstack_embeds_batch.append(deepstack_embeds)
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
                    deepstack_embeds_batch=[deepstack_embeds_batch[i] for i in normal_indices],
                    keep_sequence_logits=[
                        self._num_prompt_logprobs(self.req_states[req_ids[i]].sampling_params) is not None
                        for i in normal_indices
                    ],
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
                self._start_request_step(req_id, req_state, cache_size)
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
                        keep_sequence_logits=self._num_prompt_logprobs(req_state.sampling_params) is not None,
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
                req_state.num_computed_tokens = next_cache_size
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
        # qbruntime buffers the trace log and writes it only on stop, so a
        # server torn down mid-trace would otherwise lose the whole window.
        # A trace that cannot be written must not take the rest of the
        # teardown -- the model still has to be disposed -- down with it.
        if self._tracer is not None and self._tracer.is_running:
            try:
                self._tracer.stop()
            except Exception as e:
                logger.warning("Could not write the in-flight MBLT NPU trace during shutdown: %s", e)
        if self.model:
            dispose = getattr(self.model, "dispose", None)
            if callable(dispose):
                dispose()
        self.cache_model = None
        self.cache_backend = None
        self.input_embeddings = None
        self.runtime_cache.reset()
