from mblt_model_zoo.hf_transformers.models.llama.modeling_llama import (
    MobilintLlamaForCausalLM as OriginalMobilintLlamaForCausalLM
)
from vllm.model_executor.models import VllmModelForTextGeneration

class MobilintLlamaForCausalLM(OriginalMobilintLlamaForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True