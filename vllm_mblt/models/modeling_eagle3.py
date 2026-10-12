"""vLLM registrations for Mobilint EAGLE-3 releases.

The worker runs these models through transformers-mblt's speculative rounds (see ``vllm_mblt.eagle3_runtime``), so
these classes only name the architectures for vLLM.
"""

from transformers_mblt.models.llama_eagle3.modeling_llama_eagle3 import (
    MobilintLlamaEagle3ForCausalLM as OriginalMobilintLlamaEagle3ForCausalLM,
)
from transformers_mblt.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
    MobilintQwen2Eagle3ForCausalLM as OriginalMobilintQwen2Eagle3ForCausalLM,
)
from transformers_mblt.models.qwen3_eagle3.modeling_qwen3_eagle3 import (
    MobilintQwen3Eagle3ForCausalLM as OriginalMobilintQwen3Eagle3ForCausalLM,
)
from vllm.model_executor.models import VllmModelForTextGeneration


class MobilintLlamaEagle3ForCausalLM(OriginalMobilintLlamaEagle3ForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True


class MobilintQwen2Eagle3ForCausalLM(OriginalMobilintQwen2Eagle3ForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True


class MobilintQwen3Eagle3ForCausalLM(OriginalMobilintQwen3Eagle3ForCausalLM, VllmModelForTextGeneration):
    def is_text_generation_model(self):
        return True
