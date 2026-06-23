import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig

from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum

logger = init_logger(__name__)

SINGLE_CORE_BATCH_MODEL_MAX_PREFILL_CHUNK_SIZE = 128


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _coerce_config_dict(value: object) -> dict | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def _get_model_loader_extra_config(vllm_config: "VllmConfig") -> dict | None:
    load_config = getattr(vllm_config, "load_config", None)
    return _coerce_config_dict(getattr(load_config, "model_loader_extra_config", None))


def _resolve_core_mode(vllm_config: "VllmConfig") -> str | None:
    extra_config = _get_model_loader_extra_config(vllm_config)
    if extra_config is not None:
        configured_core_mode = extra_config.get("core_mode")
        if isinstance(configured_core_mode, str) and configured_core_mode:
            return configured_core_mode

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    configured_core_mode = getattr(hf_config, "core_mode", None)
    if isinstance(configured_core_mode, str) and configured_core_mode:
        return configured_core_mode

    return None


def _get_model_config_value(vllm_config: "VllmConfig", *field_names: str) -> object:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return None

    config_dict = None
    to_dict = getattr(hf_config, "to_dict", None)
    if callable(to_dict):
        try:
            config_dict = to_dict()
        except Exception:
            config_dict = None

    for field_name in field_names:
        if isinstance(hf_config, dict) and field_name in hf_config:
            return hf_config[field_name]

        value = getattr(hf_config, field_name, None)
        if value is not None:
            return value

        if isinstance(config_dict, dict) and field_name in config_dict:
            return config_dict[field_name]

    return None


def _resolve_model_config_positive_int(
    vllm_config: "VllmConfig",
    raw_value: object,
    *,
    field_name: str,
) -> int | None:
    if raw_value is None:
        return None

    direct_chunk_size = _coerce_positive_int(raw_value)
    if direct_chunk_size is not None:
        return direct_chunk_size

    if not isinstance(raw_value, dict):
        return None

    core_mode = _resolve_core_mode(vllm_config)
    if core_mode is not None:
        chunk_size = _coerce_positive_int(raw_value.get(core_mode))
        if chunk_size is not None:
            return chunk_size

    for fallback_key in ("default", "DEFAULT"):
        chunk_size = _coerce_positive_int(raw_value.get(fallback_key))
        if chunk_size is not None:
            return chunk_size

    logger.warning(
        "Could not resolve %s from model config for core_mode=%s: %r",
        field_name,
        core_mode,
        raw_value,
    )
    return None


def resolve_npu_prefill_chunk_size(vllm_config: "VllmConfig") -> int | None:
    raw_chunk_size = _get_model_config_value(vllm_config, "npu_prefill_chunk_size")
    return _resolve_model_config_positive_int(
        vllm_config,
        raw_chunk_size,
        field_name="npu_prefill_chunk_size",
    )


def resolve_effective_npu_prefill_chunk_size(vllm_config: "VllmConfig") -> int:
    resolved_chunk_size = resolve_npu_prefill_chunk_size(vllm_config)
    if resolved_chunk_size is None:
        return 128

    resolved_max_batch_size = resolve_model_max_batch_size(vllm_config)
    if (
        resolved_max_batch_size is not None
        and resolved_max_batch_size > 1
        and resolved_chunk_size > SINGLE_CORE_BATCH_MODEL_MAX_PREFILL_CHUNK_SIZE
    ):
        logger.warning(
            "Clamping model-configured chunked prefill size from %d to %d for batch execution.",
            resolved_chunk_size,
            SINGLE_CORE_BATCH_MODEL_MAX_PREFILL_CHUNK_SIZE,
        )
        return SINGLE_CORE_BATCH_MODEL_MAX_PREFILL_CHUNK_SIZE

    return resolved_chunk_size


def resolve_model_max_batch_size(vllm_config: "VllmConfig") -> int | None:
    extra_config = _get_model_loader_extra_config(vllm_config)
    if extra_config is not None:
        override = _coerce_positive_int(extra_config.get("max_batch_size"))
        if override is not None:
            return override

    raw_max_batch_size = _get_model_config_value(
        vllm_config,
        "max_batch_size",
        "npu_max_batch_size",
    )
    return _resolve_model_config_positive_int(
        vllm_config,
        raw_max_batch_size,
        field_name="max_batch_size",
    )


class MbltPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_type = "cpu"
    device_name = "cpu"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config: ParallelConfig = vllm_config.parallel_config  # type: ignore
        parallel_config.worker_cls = "vllm_mblt.mblt_worker.MbltWorker"

        cache_config: CacheConfig = vllm_config.cache_config
        # Keep user-provided value if present. Otherwise, use a smaller block
        # so prefix cache can hit on realistic system prompts.
        if cache_config.block_size is None:
            cache_config.block_size = 128  # type: ignore
        if cache_config.enable_prefix_caching is None:
            cache_config.enable_prefix_caching = True

        scheduler_config: SchedulerConfig = vllm_config.scheduler_config

        scheduler_config.chunked_prefill_enabled = True
        model_chunk_size = resolve_npu_prefill_chunk_size(vllm_config)
        if model_chunk_size is None:
            resolved_chunk_size = 128
        else:
            resolved_chunk_size = resolve_effective_npu_prefill_chunk_size(vllm_config)
            logger.info(
                "Using model-configured chunked prefill size %d for core_mode=%s.",
                resolved_chunk_size,
                _resolve_core_mode(vllm_config),
            )

        scheduler_config.max_num_batched_tokens = min(
            scheduler_config.max_num_batched_tokens,
            resolved_chunk_size,
        )

        resolved_max_batch_size = resolve_model_max_batch_size(vllm_config)
        if resolved_max_batch_size is not None:
            configured_max_num_seqs = _coerce_positive_int(getattr(scheduler_config, "max_num_seqs", None))
            if configured_max_num_seqs is None:
                scheduler_config.max_num_seqs = resolved_max_batch_size
            elif configured_max_num_seqs > resolved_max_batch_size:
                logger.warning(
                    "Clamping scheduler max_num_seqs from %d to model-configured max batch size %d.",
                    configured_max_num_seqs,
                    resolved_max_batch_size,
                )
                scheduler_config.max_num_seqs = resolved_max_batch_size
            else:
                scheduler_config.max_num_seqs = configured_max_num_seqs
            logger.info(
                "Using model-configured max batch size %d for scheduler max_num_seqs=%d.",
                resolved_max_batch_size,
                scheduler_config.max_num_seqs,
            )
        return
