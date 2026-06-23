from mblt_model_zoo.hf_transformers.models.qwen3.modeling_qwen3 import (
    MobilintQwen3ForCausalLM as OriginalMobilintQwen3ForCausalLM,
)
from vllm.model_executor.models import VllmModelForTextGeneration


class MobilintQwen3ForCausalLM(OriginalMobilintQwen3ForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True
