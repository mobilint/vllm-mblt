from typing import Any

from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import (
    MobilintQwen2VLForConditionalGeneration as OriginalMobilintQwen2VLForConditionalGeneration,
)
from mblt_model_zoo.hf_transformers.models.qwen2_vl.processing_qwen2_vl import (
    MobilintQwen2VLProcessor,
)
from vllm.model_executor.models import VllmModelForTextGeneration
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_mblt.models.modeling_vl_utils import MobilintVLCachedProcessorMixin


class MobilintQwen2VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> MobilintQwen2VLProcessor:
        return self.ctx.get_hf_processor(
            MobilintQwen2VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class MobilintQwen2VLMultiModalProcessor(
    MobilintVLCachedProcessorMixin,
    Qwen2VLMultiModalProcessor,
):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    MobilintQwen2VLMultiModalProcessor,
    info=MobilintQwen2VLProcessingInfo,
    dummy_inputs=Qwen2VLDummyInputsBuilder,
)
class MobilintQwen2VLForConditionalGeneration(
    OriginalMobilintQwen2VLForConditionalGeneration,
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
