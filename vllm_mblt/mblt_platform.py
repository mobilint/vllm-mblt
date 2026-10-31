import json
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum

from vllm_mblt.data_parallel import (
    QBRUNTIME_VISIBLE_DEVICES_ENV,
    collect_placement_kwargs,
    export_card_list_for_ranks,
    record_data_parallel_layout,
)

logger = init_logger(f"vllm.{__name__}")

SINGLE_CORE_BATCH_MODEL_MAX_PREFILL_CHUNK_SIZE = 128
_MULTIMODAL_HF_MODEL_TYPES = frozenset(
    {
        "mobilint-qwen2_vl",
        "mobilint-qwen3_vl",
    }
)
# MoE releases cannot be compiled as batch MXQs, so they serve one sequence at a time.
_MIXTURE_OF_EXPERTS_HF_MODEL_TYPES = frozenset({"mobilint-qwen3_moe"})
# KV capacity (max_cache_size) of the released Qwen3-30B-A3B shared MXQs; the artifact config declares 40960.
# The worker checks the loaded MXQs against max_model_len, so a release compiled smaller still fails at startup.
_MIXTURE_OF_EXPERTS_MAX_MODEL_LEN = 4096
# Qwen3-ASR audio past this token can stall the NPU; the artifact config declares 65536.
_QWEN3_ASR_MAX_MODEL_LEN = 2048
# EAGLE-3 releases run speculative rounds through transformers-mblt, one sequence per base MXQ slot.
_EAGLE3_HF_MODEL_TYPES = frozenset({"mobilint-llama-eagle3", "mobilint-qwen2-eagle3", "mobilint-qwen3-eagle3"})
_TRUE_ENV_VALUES = {"1", "true", "TRUE", "True"}
# A batch LLM compiled at a global scheme mixes core modes inside one mxq (the
# collective segments target the cluster GlobalCore, the iterative ones Core0),
# so qbruntime only accepts it under CoreMode.Auto. Such a package declares
# core_mode "auto" and therefore cannot name a key in a core-mode-indexed map
# like npu_prefill_chunk_size.
AUTO_CORE_MODE = "auto"


def _mblt_debug_enabled() -> bool:
    return os.getenv("VLLM_MBLT_DEBUG", "0") in _TRUE_ENV_VALUES


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


def _get_hf_config(vllm_config: "VllmConfig") -> object:
    model_config = getattr(vllm_config, "model_config", None)
    return getattr(model_config, "hf_config", None)


def _to_config_dict(config: object) -> dict | None:
    if isinstance(config, dict):
        return config

    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            config_dict = to_dict()
        except Exception:
            return None
        return config_dict if isinstance(config_dict, dict) else None

    return None


def _get_config_field_value(config: object, field_name: str) -> object:
    if config is None:
        return None

    if isinstance(config, dict) and field_name in config:
        return config[field_name]

    value = getattr(config, field_name, None)
    if value is not None:
        return value

    config_dict = _to_config_dict(config)
    if isinstance(config_dict, dict) and field_name in config_dict:
        return config_dict[field_name]

    return None


def _is_multimodal_hf_config(hf_config: object) -> bool:
    model_type = _get_config_field_value(hf_config, "model_type")
    return isinstance(model_type, str) and model_type in _MULTIMODAL_HF_MODEL_TYPES


def _is_mixture_of_experts_hf_config(hf_config: object) -> bool:
    return _get_config_field_value(hf_config, "model_type") in _MIXTURE_OF_EXPERTS_HF_MODEL_TYPES


def _is_eagle3_hf_config(hf_config: object) -> bool:
    return _get_config_field_value(hf_config, "model_type") in _EAGLE3_HF_MODEL_TYPES


def _unsupported_eagle3_sampling(params: object) -> list[str]:
    """The sampling options an EAGLE-3 request sets that its speculative rounds cannot honor.

    A round samples with temperature, top-k and top-p only, from the global torch generator, and returns no
    logprobs.
    """
    checks = (
        ("logprobs", getattr(params, "logprobs", None) is not None),
        ("prompt_logprobs", getattr(params, "prompt_logprobs", None) is not None),
        ("presence_penalty", getattr(params, "presence_penalty", 0.0) != 0.0),
        ("frequency_penalty", getattr(params, "frequency_penalty", 0.0) != 0.0),
        ("repetition_penalty", getattr(params, "repetition_penalty", 1.0) != 1.0),
        ("min_p", getattr(params, "min_p", 0.0) > 0.0),
        ("min_tokens", getattr(params, "min_tokens", 0) > 0),
        ("seed", getattr(params, "seed", None) is not None),
        ("logit_bias", bool(getattr(params, "logit_bias", None))),
        ("allowed_token_ids", getattr(params, "allowed_token_ids", None) is not None),
        ("bad_words", bool(getattr(params, "bad_words", None))),
        ("logits_processors", bool(getattr(params, "logits_processors", None))),
        ("structured_outputs", getattr(params, "structured_outputs", None) is not None),
    )
    return [name for name, is_set in checks if is_set]


def _is_qwen3_asr_hf_config(hf_config: object) -> bool:
    return _get_config_field_value(hf_config, "model_type") == "mobilint-qwen3_asr"


def _prefer_text_runtime(vllm_config: "VllmConfig") -> bool:
    return _is_multimodal_hf_config(_get_hf_config(vllm_config))


def _resolve_core_mode(vllm_config: "VllmConfig", *, prefer_text_core_mode: bool = False) -> str | None:
    extra_config = _get_model_loader_extra_config(vllm_config)
    if extra_config is not None:
        if prefer_text_core_mode:
            configured_text_core_mode = extra_config.get("text_core_mode")
            if isinstance(configured_text_core_mode, str) and configured_text_core_mode:
                return configured_text_core_mode

        configured_core_mode = extra_config.get("core_mode")
        if isinstance(configured_core_mode, str) and configured_core_mode:
            return configured_core_mode

    if prefer_text_core_mode:
        configured_text_core_mode = _get_text_model_config_value(
            vllm_config,
            "core_mode",
            "text_core_mode",
        )
        if isinstance(configured_text_core_mode, str) and configured_text_core_mode:
            return configured_text_core_mode

    hf_config = _get_hf_config(vllm_config)
    configured_core_mode = getattr(hf_config, "core_mode", None)
    if isinstance(configured_core_mode, str) and configured_core_mode:
        return configured_core_mode

    return None


def _get_model_config_value(vllm_config: "VllmConfig", *field_names: str) -> object:
    hf_config = _get_hf_config(vllm_config)
    if hf_config is None:
        return None

    for field_name in field_names:
        value = _get_config_field_value(hf_config, field_name)
        if value is not None:
            return value

    return None


def _get_text_model_config_value(vllm_config: "VllmConfig", *field_names: str) -> object:
    text_config = _get_config_field_value(_get_hf_config(vllm_config), "text_config")
    if text_config is None:
        return None

    for field_name in field_names:
        value = _get_config_field_value(text_config, field_name)
        if value is not None:
            return value
    return None


def _resolve_model_config_positive_int(
    vllm_config: "VllmConfig",
    raw_value: object,
    *,
    field_name: str,
    prefer_text_core_mode: bool = False,
) -> int | None:
    if raw_value is None:
        return None

    direct_chunk_size = _coerce_positive_int(raw_value)
    if direct_chunk_size is not None:
        return direct_chunk_size

    if not isinstance(raw_value, dict):
        return None

    core_mode = _resolve_core_mode(vllm_config, prefer_text_core_mode=prefer_text_core_mode)
    if core_mode is not None:
        chunk_size = _coerce_positive_int(raw_value.get(core_mode))
        if chunk_size is not None:
            return chunk_size

    for fallback_key in ("default", "DEFAULT"):
        chunk_size = _coerce_positive_int(raw_value.get(fallback_key))
        if chunk_size is not None:
            return chunk_size

    if core_mode is None or core_mode == AUTO_CORE_MODE:
        # "auto" is a real core mode for global-scheme batch artifacts, but it
        # is never a key of a core-mode-indexed map. When the map describes a
        # single compiled mode, that entry is the artifact's own value; falling
        # through to the generic default would silently shrink the prefill
        # chunk size instead.
        usable_entries = {
            str(key): value for key, value in raw_value.items() if _coerce_positive_int(value) is not None
        }
        if len(usable_entries) == 1:
            only_key, only_value = next(iter(usable_entries.items()))
            resolved = _coerce_positive_int(only_value)
            logger.info(
                "Resolved %s=%s from the only core-mode entry %r for core_mode=%s.",
                field_name,
                resolved,
                only_key,
                core_mode,
            )
            return resolved

    logger.warning(
        "Could not resolve %s from model config for core_mode=%s: %r",
        field_name,
        core_mode,
        raw_value,
    )
    return None


def resolve_npu_prefill_chunk_size(vllm_config: "VllmConfig") -> int | None:
    prefer_text_runtime = _prefer_text_runtime(vllm_config)
    if prefer_text_runtime:
        raw_text_chunk_size = _get_text_model_config_value(
            vllm_config,
            "npu_prefill_chunk_size",
        )
        resolved_text_chunk_size = _resolve_model_config_positive_int(
            vllm_config,
            raw_text_chunk_size,
            field_name="text_config.npu_prefill_chunk_size",
            prefer_text_core_mode=True,
        )
        if resolved_text_chunk_size is not None:
            return resolved_text_chunk_size

    raw_chunk_size = _get_model_config_value(vllm_config, "npu_prefill_chunk_size")
    return _resolve_model_config_positive_int(
        vllm_config,
        raw_chunk_size,
        field_name="npu_prefill_chunk_size",
        prefer_text_core_mode=prefer_text_runtime,
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


def resolve_model_max_batch_size(
    vllm_config: "VllmConfig",
    *,
    prefer_text_runtime: bool | None = None,
) -> int | None:
    if prefer_text_runtime is None:
        prefer_text_runtime = _prefer_text_runtime(vllm_config)

    extra_config = _get_model_loader_extra_config(vllm_config)
    if extra_config is not None:
        if prefer_text_runtime:
            text_override = _coerce_positive_int(extra_config.get("text_max_batch_size"))
            if text_override is not None:
                return text_override

        override = _coerce_positive_int(extra_config.get("max_batch_size"))
        if override is not None:
            return override

    if prefer_text_runtime:
        raw_text_max_batch_size = _get_text_model_config_value(
            vllm_config,
            "max_batch_size",
            "npu_max_batch_size",
        )
        resolved_text_max_batch_size = _resolve_model_config_positive_int(
            vllm_config,
            raw_text_max_batch_size,
            field_name="text_config.max_batch_size",
            prefer_text_core_mode=True,
        )
        if resolved_text_max_batch_size is not None:
            return resolved_text_max_batch_size

    # An EAGLE-3 release sizes its base, draft and FC MXQs separately; the base MXQ holds the sequences.
    raw_max_batch_size = _get_model_config_value(
        vllm_config,
        *(("base_max_batch_size",) if _is_eagle3_hf_config(_get_hf_config(vllm_config)) else ()),
        "max_batch_size",
        "npu_max_batch_size",
    )
    return _resolve_model_config_positive_int(
        vllm_config,
        raw_max_batch_size,
        field_name="max_batch_size",
        prefer_text_core_mode=prefer_text_runtime,
    )


def resolve_static_image_position(vllm_config: "VllmConfig") -> tuple[int, int]:
    """Read the artifact's compiled static image placeholder position from loader settings or model config."""
    loader = _get_model_loader_extra_config(vllm_config) or {}
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = _get_hf_config(vllm_config)
    values = []
    for name in ("static_image_token_offset", "static_image_token_length"):
        value = loader.get(name)
        if value is None:
            for root in (getattr(model_config, "model_kwargs", None), getattr(model_config, "hf_overrides", None)):
                if isinstance(root, dict) and name in root:
                    value = root[name]
                    break
        if value is None:
            value = getattr(hf_config, name, None)
        minimum = 1 if name.endswith("length") else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"Split-static Qwen3-VL batch requires {name} as an integer in model-loader-extra-config "
                "or the model config, matching the compiled prompt layout."
            )
        values.append(value)
    return values[0], values[1]


def _has_static_image_position(vllm_config: "VllmConfig") -> bool:
    """A static position explicitly declared by the artifact or caller opts into frontend validation.

    The frontend cannot inspect the NPU MXQ signature, and an absent dynamic-vision flag is not proof of a static
    decoder. The worker separately rejects split-static batch MXQs without a complete position at model load.
    """
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = _get_hf_config(vllm_config)
    roots = (
        _get_model_loader_extra_config(vllm_config),
        getattr(model_config, "model_kwargs", None),
        getattr(model_config, "hf_overrides", None),
    )
    names = ("static_image_token_offset", "static_image_token_length")
    return any(isinstance(root, dict) and any(root.get(name) is not None for name in names) for root in roots) or any(
        getattr(hf_config, name, None) is not None for name in names
    )


class MbltPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_type = "cpu"
    device_name = "cpu"
    # With --data-parallel-size N, vLLM 0.32 records each engine's entry of this variable (the parent's list, or
    # 0..N-1 when unset) in parallel_config.assigned_physical_gpu_ids; the worker narrows the variable to it.
    device_control_env_var = QBRUNTIME_VISIBLE_DEVICES_ENV
    # vLLM compiles small sampling helpers (e.g. the logprob rank count) with this backend. The sampling runs on the
    # host; inductor would compile C++ at runtime and fail the engine where the host toolchain is unusable.
    simple_compile_backend = "eager"
    # The data parallel engines all-reduce their unfinished-request state over a CPU process group.
    dist_backend = "gloo"
    # Whether the engine configured last in this process serves EAGLE-3; validate_request's fallback.
    _serving_eagle3 = False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Pinned host memory is a CUDA facility and is never available here.

        The base `Platform` implementation returns True on any non-WSL host, so
        an MBLT server on plain Linux without an NVIDIA driver used to answer
        True. vLLM then builds tensors with `pin_memory=True` and
        `Tensor.pin_memory()` raises `Found no NVIDIA driver on your system` —
        for example in the sampling-penalty path
        (`apply_all_penalties` -> `make_tensor_with_pad`). MBLT executes on the
        NPU with CPU-side host tensors, so there is nothing to pin, and
        `CpuPlatform` answers False for the same reason.
        """
        return False

    @classmethod
    def _validating_for_eagle3(cls) -> bool:
        """Whether the engine validating this request serves EAGLE-3.

        vLLM passes ``validate_request`` no engine, so a process with several engines (several ``LLM`` objects) cannot
        tell them apart from its arguments. Its caller, vLLM 0.32's ``InputProcessor`` method, holds the engine's
        ``model_config``; without it, the engine configured last decides.
        """
        caller = sys._getframe(2).f_locals.get("self")
        model_config = getattr(caller, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return cls._serving_eagle3
        return _is_eagle3_hf_config(hf_config)

    @classmethod
    def validate_request(cls, processed_inputs: object, params: object) -> None:
        """Refuse requests an EAGLE-3 model cannot serve, with the options to drop (an HTTP 400 for the client).

        vLLM 0.32 maps only its VLLMClientError family to 4xx; a plain ValueError here reaches the client as a 500.
        """
        # vLLM calls this in InputProcessor after rendering, before enqueueing the request. Raising its client error
        # here prevents a bad static image position from failing the whole engine inside execute_model().
        caller = sys._getframe(1).f_locals.get("self")
        config = getattr(caller, "vllm_config", None)
        hf_config = _get_hf_config(config)
        if (
            getattr(hf_config, "model_type", None) == "mobilint-qwen3_vl"
            and not bool(getattr(hf_config, "dynamic_vision", False))
            and _has_static_image_position(config)
            and (resolve_model_max_batch_size(config) or 1) > 1
        ):
            try:
                expected = resolve_static_image_position(config)
            except ValueError as exc:
                raise VLLMValidationError(str(exc), parameter="model") from exc
            inputs = processed_inputs if isinstance(processed_inputs, dict) else {}
            placeholders = inputs.get("mm_placeholders") or {}
            images = placeholders.get("image", [])
            other_modalities = any(items for kind, items in placeholders.items() if kind != "image")
            valid = len(images) == 1 and not other_modalities
            if valid:
                image = images[0]
                mask = getattr(image, "is_embed", None)
                valid = (int(image.offset), int(image.length)) == expected and (mask is None or bool(mask.all()))
            if not valid:
                raise VLLMValidationError(
                    "Static Qwen3-VL batch requires one image at the compiled placeholder position "
                    f"offset={expected[0]}, length={expected[1]}, with all image tokens.",
                    parameter="messages",
                )

        if not cls._validating_for_eagle3():
            return
        unsupported = _unsupported_eagle3_sampling(params)
        if unsupported:
            raise VLLMValidationError(
                "EAGLE-3 models sample with temperature, top_k and top_p only; unsupported here: "
                + ", ".join(unsupported)
                + ".",
                parameter=unsupported[0],
            )
        inputs = processed_inputs if isinstance(processed_inputs, dict) else {}
        if inputs.get("type") in ("embeds", "multimodal"):
            raise VLLMValidationError(
                "EAGLE-3 models take token prompts only, without prompt embeddings or multimodal data."
            )

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """Seed the device RNGs; the NPU has none.

        vLLM's ``set_random_seed`` has already seeded Python, NumPy and torch's CPU generator, which is where this
        platform samples.
        """
        del seed

    @classmethod
    def _prepare_data_parallel(cls, vllm_config: "VllmConfig") -> None:
        """Check the data parallel card assignment before vLLM starts one engine process per card.

        This runs once, in the front end; each engine's worker places its model on its own card at load time.
        """
        parallel_config = vllm_config.parallel_config
        data_parallel_size = _coerce_positive_int(getattr(parallel_config, "data_parallel_size", 1)) or 1
        if data_parallel_size <= 1:
            return
        local_size = _coerce_positive_int(getattr(parallel_config, "data_parallel_size_local", None))
        record_data_parallel_layout(vllm_config, size=data_parallel_size, size_local=local_size or data_parallel_size)
        cards = export_card_list_for_ranks(
            collect_placement_kwargs(_get_model_loader_extra_config(vllm_config)),
            local_data_parallel_size=local_size or data_parallel_size,
        )
        logger.info(
            "Data parallel size %d: one EngineCore per card (%s=%s).",
            data_parallel_size,
            QBRUNTIME_VISIBLE_DEVICES_ENV,
            ",".join(map(str, cards)) if cards else os.environ.get(QBRUNTIME_VISIBLE_DEVICES_ENV, "<rank number>"),
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config: ParallelConfig = vllm_config.parallel_config  # type: ignore
        if getattr(vllm_config.model_config, "runner_type", None) == "pooling":
            if not getattr(vllm_config.model_config.hf_config, "mblt_pooling", None):
                raise ValueError("Mobilint pooling requires a compiled artifact package; see docs/pooling.md")
            if getattr(parallel_config, "world_size", 1) != 1:
                raise ValueError("Mobilint pooling currently requires tensor/pipeline parallel size 1")
            hf_config = vllm_config.model_config.hf_config
            expected_pooling = {"mean": "MEAN", "last": "LAST", "scalar": "CLS", "yes_no": "LAST"}
            if hf_config.mblt_pooling not in expected_pooling:
                raise ValueError("Unknown compiled Mobilint pooling task")
            pooler = getattr(vllm_config.model_config, "pooler_config", None)
            if pooler is not None:
                if pooler.pooling_type not in (None, expected_pooling[hf_config.mblt_pooling]):
                    raise ValueError("pooling_type must match the compiled Mobilint package")
                if pooler.enable_chunked_processing or pooler.max_embed_len not in (
                    None, vllm_config.model_config.max_model_len
                ):
                    raise ValueError("Mobilint pooling does not support long-input chunked processing")
                if any(getattr(pooler, key) is not None for key in ("logit_bias", "step_tag_id", "returned_token_ids")):
                    raise ValueError("Mobilint pooling does not support logit_bias or token-selection overrides")
            if getattr(hf_config, "is_matryoshka", False) and getattr(hf_config, "matryoshka_dimensions", None) is None:
                # vLLM 0.11.2 otherwise validates only dimensions > 0. Reject
                # oversized requests before they can raise inside the engine.
                hf_config.matryoshka_dimensions = list(range(1, hf_config.hidden_size + 1))
            parallel_config.worker_cls = "vllm_mblt.pooling_worker.MbltPoolingWorker"
            if getattr(vllm_config.cache_config, "block_size", None) is None:
                vllm_config.cache_config.block_size = 128
            vllm_config.cache_config.enable_prefix_caching = False
            scheduler = vllm_config.scheduler_config
            scheduler.enable_chunked_prefill = False
            scheduler.chunked_prefill_enabled = False
            scheduler.max_num_batched_tokens = max(
                scheduler.max_num_batched_tokens, vllm_config.model_config.max_model_len
            )
            return
        parallel_config.worker_cls = "vllm_mblt.mblt_worker.MbltWorker"
        # vLLM 0.32 defaults data parallel engines to the multiprocessing executor, which runs the worker in its own
        # process and gives it OMP_NUM_THREADS = all of the host's CPUs, so several engines oversubscribe the host.
        # Each engine here has one worker (no tensor or pipeline parallelism), so run it in the engine process as
        # vLLM 0.11 did; vLLM then splits the host's CPUs among the local engines itself.
        if (
            getattr(parallel_config, "world_size", 1) == 1
            and getattr(parallel_config, "distributed_executor_backend", None) == "mp"
        ):
            logger.info("Running each engine's single worker in-process (distributed_executor_backend=uni).")
            parallel_config.distributed_executor_backend = "uni"
        cls._serving_eagle3 = _is_eagle3_hf_config(_get_hf_config(vllm_config))
        cls._prepare_data_parallel(vllm_config)

        # vLLM 0.32 turns async scheduling on by default. The engine then takes each step's result from a separate
        # sample_tokens() call, while this worker samples inside execute_model() and returns the whole output there.
        scheduler_config = vllm_config.scheduler_config
        if getattr(scheduler_config, "async_scheduling", False):
            logger.info("Disabling async scheduling: the MBLT worker samples inside execute_model().")
            scheduler_config.async_scheduling = False

        cache_config: CacheConfig = vllm_config.cache_config
        # Keep user-provided value if present. Otherwise, use a smaller block
        # so prefix cache can hit on realistic system prompts.
        if cache_config.block_size is None:
            cache_config.block_size = 128  # type: ignore
        if cache_config.enable_prefix_caching is None:
            cache_config.enable_prefix_caching = True
        if cls._serving_eagle3 and cache_config.enable_prefix_caching:
            # Each EAGLE-3 request feeds its whole prompt to its own cache; a prefix hit would only skip vLLM steps.
            logger.info("Disabling prefix caching for EAGLE-3: every request feeds its whole prompt.")
            cache_config.enable_prefix_caching = False

        # vLLM 0.11.2's front end records a multimodal item as sent to the engine before the
        # length checks run. If the request is then refused, the engine never receives the item,
        # and a later request with the same audio arrives without it: the engine's input thread
        # fails an assert and stops accepting requests. Qwen3-ASR rarely sees the same audio twice,
        # so the cache gains it little.
        multimodal_config = getattr(getattr(vllm_config, "model_config", None), "multimodal_config", None)
        if (
            _is_qwen3_asr_hf_config(_get_hf_config(vllm_config))
            and getattr(multimodal_config, "mm_processor_cache_gb", 0) > 0
        ):
            logger.info(
                "Disabling the multimodal processor cache for Qwen3-ASR (mm_processor_cache_gb=%s -> 0).",
                multimodal_config.mm_processor_cache_gb,
            )
            multimodal_config.mm_processor_cache_gb = 0

        if _is_qwen3_asr_hf_config(_get_hf_config(vllm_config)):
            model_config = vllm_config.model_config
            if model_config.max_model_len > _QWEN3_ASR_MAX_MODEL_LEN:
                logger.warning(
                    "Clamping max_model_len from %d to %d for Qwen3-ASR.",
                    model_config.max_model_len,
                    _QWEN3_ASR_MAX_MODEL_LEN,
                )
                model_config.max_model_len = _QWEN3_ASR_MAX_MODEL_LEN

        if _is_mixture_of_experts_hf_config(_get_hf_config(vllm_config)):
            model_config = vllm_config.model_config
            if model_config.max_model_len > _MIXTURE_OF_EXPERTS_MAX_MODEL_LEN:
                logger.warning(
                    "Clamping max_model_len from %d to %d for Mixture-of-Experts models.",
                    model_config.max_model_len,
                    _MIXTURE_OF_EXPERTS_MAX_MODEL_LEN,
                )
                model_config.max_model_len = _MIXTURE_OF_EXPERTS_MAX_MODEL_LEN

        scheduler_config: SchedulerConfig = vllm_config.scheduler_config

        scheduler_config.chunked_prefill_enabled = True
        print_debug = _mblt_debug_enabled()
        if print_debug:
            logger.info(
                "Initial scheduler prefill config: max_num_batched_tokens=%r, max_num_seqs=%r, "
                "max_num_partial_prefills=%r, max_long_partial_prefills=%r, "
                "long_prefill_token_threshold=%r.",
                getattr(scheduler_config, "max_num_batched_tokens", None),
                getattr(scheduler_config, "max_num_seqs", None),
                getattr(scheduler_config, "max_num_partial_prefills", None),
                getattr(scheduler_config, "max_long_partial_prefills", None),
                getattr(scheduler_config, "long_prefill_token_threshold", None),
            )
        resolved_max_batch_size = resolve_model_max_batch_size(vllm_config)
        if cls._serving_eagle3 and resolved_max_batch_size is not None and resolved_max_batch_size > 1:
            raise ValueError(
                f"EAGLE-3 models serve one sequence per base MXQ, but max_batch_size={resolved_max_batch_size} "
                "was configured."
            )
        if (
            _is_mixture_of_experts_hf_config(_get_hf_config(vllm_config))
            and resolved_max_batch_size is not None
            and resolved_max_batch_size > 1
        ):
            raise ValueError(
                "Mixture-of-Experts models support only non-batch serving, "
                f"but max_batch_size={resolved_max_batch_size} was configured."
            )
        effective_max_num_seqs = None
        if resolved_max_batch_size is not None:
            configured_max_num_seqs = _coerce_positive_int(getattr(scheduler_config, "max_num_seqs", None))
            if configured_max_num_seqs is None:
                effective_max_num_seqs = resolved_max_batch_size
            elif configured_max_num_seqs > resolved_max_batch_size:
                logger.warning(
                    "Clamping scheduler max_num_seqs from %d to model-configured max batch size %d.",
                    configured_max_num_seqs,
                    resolved_max_batch_size,
                )
                effective_max_num_seqs = resolved_max_batch_size
            else:
                effective_max_num_seqs = configured_max_num_seqs
            scheduler_config.max_num_seqs = effective_max_num_seqs
            logger.info(
                "Using model-configured max batch size %d for scheduler max_num_seqs=%d.",
                resolved_max_batch_size,
                scheduler_config.max_num_seqs,
            )

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

        if effective_max_num_seqs is not None and effective_max_num_seqs > 1:
            scheduler_config.max_num_batched_tokens = resolved_chunk_size * effective_max_num_seqs
            scheduler_config.long_prefill_token_threshold = resolved_chunk_size
            for attr_name in ("max_num_partial_prefills", "max_long_partial_prefills"):
                current_value = _coerce_positive_int(getattr(scheduler_config, attr_name, None))
                if current_value is None or current_value < effective_max_num_seqs:
                    setattr(scheduler_config, attr_name, effective_max_num_seqs)
            if print_debug:
                logger.info(
                    "Using per-sequence chunk size %d across %d scheduler sequences (max_num_batched_tokens=%d).",
                    resolved_chunk_size,
                    effective_max_num_seqs,
                    scheduler_config.max_num_batched_tokens,
                )
        else:
            scheduler_config.max_num_batched_tokens = min(
                scheduler_config.max_num_batched_tokens,
                resolved_chunk_size,
            )
        if print_debug:
            logger.info(
                "Final scheduler prefill config: chunked_prefill_enabled=%r, "
                "max_num_batched_tokens=%r, max_num_seqs=%r, max_num_partial_prefills=%r, "
                "max_long_partial_prefills=%r, long_prefill_token_threshold=%r.",
                getattr(scheduler_config, "chunked_prefill_enabled", None),
                getattr(scheduler_config, "max_num_batched_tokens", None),
                getattr(scheduler_config, "max_num_seqs", None),
                getattr(scheduler_config, "max_num_partial_prefills", None),
                getattr(scheduler_config, "max_long_partial_prefills", None),
                getattr(scheduler_config, "long_prefill_token_threshold", None),
            )
        return
