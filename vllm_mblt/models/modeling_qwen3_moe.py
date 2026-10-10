from transformers_mblt.models.qwen3_moe.modeling_qwen3_moe import (
    MobilintQwen3MoEForCausalLM as OriginalMobilintQwen3MoEForCausalLM,
)
from vllm.model_executor.models import VllmModelForTextGeneration


class MobilintQwen3MoEForCausalLM(OriginalMobilintQwen3MoEForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True
