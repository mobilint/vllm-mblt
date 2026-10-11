"""Runtime glue for Mobilint Mixture-of-Experts releases (Qwen3-30B-A3B).

A MoE release has no single text MXQ: every decoder layer is a ``shared`` MXQ
that holds its own KV cache, plus 128 stateless expert MXQs, and the routing
between them runs on the host. transformers-mblt owns all of that
(:class:`MobilintMixtureOfExpertsModelMixin.moe_forward`: chunked prefill,
top-k routing, parallel expert dispatch across devices, ``lm_head``).

The worker drives a text model through a ``cache_model`` handle that exposes
``infer``. This module adapts a MoE model to that handle so the worker's
scheduling, sampling, and prefix-cache code serves MoE unchanged, while every
forward is delegated to transformers-mblt. Each request keeps its own
:class:`MobilintMixtureOfExpertsCache`; the forward binds it to the shared MXQs,
saving whichever request's KV they held. Only the non-batch path exists: MoE
releases cannot be compiled as batch MXQs.
"""

import weakref
from typing import Any, Optional, Sequence

import numpy as np
import torch


def is_mixture_of_experts_model(model: object) -> bool:
    """Return True for a transformers-mblt Mixture-of-Experts model (KV state spread over shared MXQs)."""
    return callable(getattr(model, "get_shared_mxq_models", None)) and callable(
        getattr(model, "build_mobilint_cache", None)
    )


class MbltMoECacheModel:
    """``cache_model`` handle that runs a transformers-mblt MoE model through its own ``forward``.

    The worker hands over the cache of the request about to run (:meth:`use_cache`)
    and passes the KV prefix length as ``cache_size`` on every call; the adapter
    rewinds that cache's cursor to ``cache_size`` before the forward.
    """

    def __init__(self, model: Any) -> None:
        if not is_mixture_of_experts_model(model):
            raise TypeError(f"{type(model).__name__} is not a Mobilint Mixture-of-Experts model.")
        self.model = model
        self._cache_ref: Optional[weakref.ref] = None
        self.shared_mxq_models: list[Any] = list(model.get_shared_mxq_models())
        self.vocab_size = int(model.config.vocab_size)
        self.max_cache_size = min(
            int(shared.get_input_buffer_info()[0].max_cache_size) for shared in self.shared_mxq_models
        )

    def new_cache(self) -> Any:
        """Return an empty cache for one request."""
        return self.model.build_mobilint_cache(1)

    def use_cache(self, cache: Optional[Any]) -> None:
        """Run the next infers on ``cache``; ``None`` (prefill-cost calibration) runs each on a fresh scratch cache.

        Held weakly: a finished request's cache that nothing else keeps is freed, so its KV is never saved.
        """
        self._cache_ref = weakref.ref(cache) if cache is not None else None

    @property
    def cache(self) -> Any:
        cache = self._cache_ref() if self._cache_ref is not None else None
        return cache if cache is not None else self.new_cache()

    def get_model_output_shape(self) -> list[tuple[int, ...]]:
        """Report last-token logits, so prompt logprobs take the worker's 1-token microstep path."""
        return [(1, 1, self.vocab_size)]

    def infer(
        self,
        inputs: np.ndarray | Sequence[np.ndarray],
        outputs: Optional[list[np.ndarray]] = None,
        cache_size: int = 0,
    ) -> list[np.ndarray]:
        """Run ``[1, seq, hidden]`` embeddings on top of a ``cache_size``-token KV prefix; return ``[logits]``.

        ``outputs`` is accepted for signature compatibility and ignored: the logits
        come back from transformers-mblt as a fresh ``[1, 1, vocab]`` array.
        """
        del outputs
        embeds = inputs if isinstance(inputs, np.ndarray) else inputs[0]
        cache = self.cache
        cache.set_seq_length(int(cache_size))
        output = self.model(
            inputs_embeds=torch.from_numpy(np.ascontiguousarray(embeds, dtype=np.float32)),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return [output.logits.numpy()]
