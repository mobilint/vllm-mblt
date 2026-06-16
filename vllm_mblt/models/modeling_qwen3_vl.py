from typing import Any

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
    MobilintQwen3VLForConditionalGeneration as OriginalMobilintQwen3VLForConditionalGeneration,
)
from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (
    MobilintQwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models import VllmModelForTextGeneration
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)

from vllm_mblt.models.modeling_vl_utils import MobilintVLCachedProcessorMixin


class MobilintQwen3VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> MobilintQwen3VLProcessor:
        return self.ctx.get_hf_processor(
            MobilintQwen3VLProcessor,
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
