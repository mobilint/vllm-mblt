from mblt_model_zoo.hf_transformers.models.qwen2.modeling_qwen2 import (
    MobilintQwen2ForCausalLM as OriginalMobilintQwen2ForCausalLM,
)
from vllm.model_executor.models import VllmModelForTextGeneration


class MobilintQwen2ForCausalLM(OriginalMobilintQwen2ForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True
