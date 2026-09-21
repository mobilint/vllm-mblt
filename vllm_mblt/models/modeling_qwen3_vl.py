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

from vllm_mblt.models.modeling_vl_utils import MobilintVLCachedProcessorMixin


class MobilintQwen3VLSafeProcessor(MobilintQwen3VLProcessor):
    """Keep dynamic vision inputs within the NPU encoder's sequence limit."""

    max_vision_tokens = 4096

    @staticmethod
    def _cap_stored_pixels(processor: object, limit: int) -> None:
        for attribute in ("max_pixels", "min_pixels"):
            value = getattr(processor, attribute, None)
            if value is not None and value > limit:
                setattr(processor, attribute, limit)

    def _clamp_dynamic_image_size(self) -> None:
        image_processor = self.image_processor
        limit = self.max_vision_tokens * int(image_processor.patch_size) ** 2
        scope = {"size": image_processor.size}
        self._cap_size_edges(scope, limit, "image")
        image_processor.size = scope["size"]
        self._cap_stored_pixels(image_processor, limit)

    def _clamp_dynamic_image_call_kwargs(self, kwargs: dict) -> None:
        image_processor = self.image_processor
        limit = self.max_vision_tokens * int(image_processor.patch_size) ** 2
        for scope in self._call_kwargs_scopes(kwargs, "images_kwargs"):
            self._reject_do_resize_false(scope, "image")
            self._cap_pixel_kwargs(scope, limit, "image")
            self._cap_size_edges(scope, limit, "image")
        self._mirror_pixel_caps_to_image_size(kwargs, limit)

    def _clamp_dynamic_video_size(self) -> None:
        video_processor = self.video_processor
        if video_processor is None:
            return
        limit = self.max_vision_tokens * int(video_processor.patch_size) ** 2 * int(video_processor.temporal_patch_size)
        scope = {"size": video_processor.size}
        self._cap_size_edges(scope, limit, "video")
        video_processor.size = scope["size"]
        self._cap_stored_pixels(video_processor, limit)

    def _clamp_dynamic_video_call_kwargs(self, kwargs: dict) -> None:
        video_processor = self.video_processor
        if video_processor is None:
            return
        limit = self.max_vision_tokens * int(video_processor.patch_size) ** 2 * int(video_processor.temporal_patch_size)
        for scope in self._call_kwargs_scopes(kwargs, "videos_kwargs"):
            self._reject_do_resize_false(scope, "video")
            self._cap_size_edges(scope, limit, "video")


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
