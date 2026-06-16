import torch

from mblt_model_zoo.hf_transformers.models.exaone.modeling_exaone import (
    MobilintExaoneForCausalLM as OriginalMobilintExaoneForCausalLM,
)
from vllm.model_executor.models import VllmModelForTextGeneration


class MobilintExaoneForCausalLM(
    OriginalMobilintExaoneForCausalLM, VllmModelForTextGeneration
):
    def __init__(
        self,
        config=None,
        *args,
        vllm_config=None,
        prefix: str = "",
        **kwargs,
    ):
        if vllm_config is not None and config is None:
            config = vllm_config.model_config.hf_config
        super().__init__(config, *args, **kwargs)
        self._vllm_prefix = prefix
        if hasattr(self, "model_config"):
            self.model_config.is_generative_model = True

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        output = super().forward(
            input_ids=input_ids,
            position_ids=positions,
            return_dict=True,
            **kwargs,
        )
        return output.logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states
