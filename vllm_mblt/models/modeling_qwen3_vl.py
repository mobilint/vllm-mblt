from typing import Any

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
    MobilintQwen3VLForConditionalGeneration as OriginalMobilintQwen3VLForConditionalGeneration,
)
from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (
    MobilintQwen3VLProcessor,
)
from vllm.model_executor.models import VllmModelForTextGeneration
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_mblt.models.modeling_vl_utils import (
    QWEN3_VL_MAX_VISION_TOKENS,
    MobilintVLCachedProcessorMixin,
)


class MobilintQwen3VLSafeProcessor(MobilintQwen3VLProcessor):
    """Keep dynamic vision inputs within the NPU encoder's sequence limit."""

    max_vision_tokens = QWEN3_VL_MAX_VISION_TOKENS
    _PIXEL_BUDGET_FIELDS = ("max_pixels", "min_pixels", "total_pixels")

    @classmethod
    def _cap_pixel_budgets(cls, target: object, limit: int) -> None:
        """Cap every known pixel-budget field on a processor or kwargs dict."""
        for field in cls._PIXEL_BUDGET_FIELDS:
            value = target.get(field) if isinstance(target, dict) else getattr(target, field, None)
            if value is not None and value > limit:
                if isinstance(target, dict):
                    target[field] = limit
                else:
                    setattr(target, field, limit)

    def _vision_pixel_limit(self, processor: object, *, video: bool) -> int:
        temporal_factor = int(getattr(processor, "temporal_patch_size", 1)) if video else 1
        return self.max_vision_tokens * int(processor.patch_size) ** 2 * temporal_factor

    def _clamp_processor_defaults(self, processor: object, *, kind: str, video: bool) -> None:
        limit = self._vision_pixel_limit(processor, video=video)
        scope = {"size": processor.size}
        self._cap_size_edges(scope, limit, kind)
        processor.size = scope["size"]
        self._cap_pixel_budgets(processor, limit)

    def _clamp_call_budgets(
        self,
        kwargs: dict,
        *,
        processor: object,
        nested_key: str,
        kind: str,
        video: bool,
    ) -> int:
        limit = self._vision_pixel_limit(processor, video=video)
        for scope in self._call_kwargs_scopes(kwargs, nested_key):
            self._reject_do_resize_false(scope, kind)
            self._cap_pixel_budgets(scope, limit)
            self._cap_size_edges(scope, limit, kind)
        return limit

    def _clamp_dynamic_image_size(self) -> None:
        self._clamp_processor_defaults(self.image_processor, kind="image", video=False)

    def _clamp_dynamic_image_call_kwargs(self, kwargs: dict) -> None:
        limit = self._clamp_call_budgets(
            kwargs,
            processor=self.image_processor,
            nested_key="images_kwargs",
            kind="image",
            video=False,
        )
        self._mirror_pixel_caps_to_image_size(kwargs, limit)

    def _clamp_dynamic_video_size(self) -> None:
        if self.video_processor is None:
            return
        self._clamp_processor_defaults(self.video_processor, kind="video", video=True)

    def _clamp_dynamic_video_call_kwargs(self, kwargs: dict) -> None:
        if self.video_processor is None:
            return
        self._clamp_call_budgets(
            kwargs,
            processor=self.video_processor,
            nested_key="videos_kwargs",
            kind="video",
            video=True,
        )


class MobilintQwen3VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> MobilintQwen3VLSafeProcessor:
        return self.ctx.get_hf_processor(
            MobilintQwen3VLSafeProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class MobilintQwen3VLMultiModalProcessor(
    MobilintVLCachedProcessorMixin,
    Qwen3VLMultiModalProcessor,
):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    MobilintQwen3VLMultiModalProcessor,
    info=MobilintQwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class MobilintQwen3VLForConditionalGeneration(
    OriginalMobilintQwen3VLForConditionalGeneration,
    SupportsMultiModal,
    VllmModelForTextGeneration,
):
    merge_by_field_config = True
    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def get_language_model(self) -> Any:
        return self.model.language_model

    def launch(self) -> None:
        self.model.visual.launch()
        self.model.language_model.launch()

    def dispose(self) -> None:
        self.model.visual.dispose()
        self.model.language_model.dispose()

    def is_text_generation_model(self) -> bool:
        return True
