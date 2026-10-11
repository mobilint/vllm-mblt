"""Runtime glue for Mobilint Mixture-of-Experts releases (Qwen3-30B-A3B).

A MoE release has no single text MXQ: every decoder layer is a ``shared`` MXQ
that holds its own KV cache, plus 128 stateless expert MXQs, and the routing
between them runs on the host. transformers-mblt owns all of that
(:class:`MobilintMixtureOfExpertsModelMixin.moe_forward`: chunked prefill,
top-k routing, parallel expert dispatch across devices, ``lm_head``).

The worker runs a MoE model through its forward like any text model. This
module gives the worker the MXQ facts it reads off a text model's ``cache_model``
(logits kind, KV capacity, KV size) and builds each request's
:class:`MobilintMixtureOfExpertsCache`; the forward binds that cache to the shared
MXQs, saving whichever request's KV they held. Only the non-batch path exists:
MoE releases cannot be compiled as batch MXQs.
"""

from typing import Any


def is_mixture_of_experts_model(model: object) -> bool:
    """Return True for a transformers-mblt Mixture-of-Experts model (KV state spread over shared MXQs)."""
    return callable(getattr(model, "get_shared_mxq_models", None)) and callable(
        getattr(model, "build_mobilint_cache", None)
    )


class MbltMoECacheModel:
    """``cache_model`` handle for a transformers-mblt MoE model: its MXQ facts and request caches."""

    def __init__(self, model: Any) -> None:
        if not is_mixture_of_experts_model(model):
            raise TypeError(f"{type(model).__name__} is not a Mobilint Mixture-of-Experts model.")
        self.model = model
        self.shared_mxq_models: list[Any] = list(model.get_shared_mxq_models())
        self.vocab_size = int(model.config.vocab_size)
        self.max_cache_size = min(
            int(shared.get_input_buffer_info()[0].max_cache_size) for shared in self.shared_mxq_models
        )

    def new_cache(self) -> Any:
        """Return an empty cache for one request."""
        return self.model.build_mobilint_cache(1)

    def get_model_output_shape(self) -> list[tuple[int, ...]]:
        """Report last-token logits, so prompt logprobs take the worker's 1-token microstep path."""
        return [(1, 1, self.vocab_size)]

