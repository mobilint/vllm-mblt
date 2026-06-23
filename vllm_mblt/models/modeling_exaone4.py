from mblt_model_zoo.hf_transformers.models.exaone4.modeling_exaone4 import (
    MobilintExaone4ForCausalLM as OriginalMobilintExaone4ForCausalLM,
)
from vllm.model_executor.models import VllmModelForTextGeneration


class MobilintExaone4ForCausalLM(OriginalMobilintExaone4ForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True
