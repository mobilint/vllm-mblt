"""Runtime glue for Mobilint Mixture-of-Experts releases (Qwen3-30B-A3B).

A MoE release has no single text MXQ: every decoder layer is a ``shared`` MXQ
that holds its own KV cache, plus 128 stateless expert MXQs, and the routing
between them runs on the host. transformers-mblt owns all of that
(:class:`MobilintMixtureOfExpertsModelMixin.moe_forward`: chunked prefill,
top-k routing, parallel expert dispatch across devices, ``lm_head``).

The worker drives a text model through a ``cache_model`` handle that exposes
``infer`` / ``dump_cache_memory`` / ``load_cache_memory``. This module adapts a
MoE model to that handle so the worker's scheduling, sampling, and prefix-cache
code serves MoE unchanged, while every forward is delegated to transformers-mblt.
Only the non-batch path exists: MoE releases cannot be compiled as batch MXQs.
"""

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

    The worker passes the KV prefix length as ``cache_size`` on every call, so the
    adapter keeps one :class:`MobilintMixtureOfExpertsCache` and rewinds its cursor
    to ``cache_size`` before each forward. A KV snapshot is the list of every
    shared MXQ's cache memory, in layer order.
    """

    def __init__(self, model: Any) -> None:
        if not is_mixture_of_experts_model(model):
            raise TypeError(f"{type(model).__name__} is not a Mobilint Mixture-of-Experts model.")
        self.model = model
        self.cache = model.build_mobilint_cache(1)
        self.shared_mxq_models: list[Any] = list(model.get_shared_mxq_models())
        self.vocab_size = int(model.config.vocab_size)
        self.max_cache_size = min(
            int(shared.get_input_buffer_info()[0].max_cache_size) for shared in self.shared_mxq_models
        )

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
        self.cache.set_seq_length(int(cache_size))
        output = self.model(
            inputs_embeds=torch.from_numpy(np.ascontiguousarray(embeds, dtype=np.float32)),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return [output.logits.numpy()]

    def dump_cache_memory(self, cache_id: int = 0) -> list[Any]:
        return [model.dump_cache_memory(cache_id) for model in self.shared_mxq_models]

    def load_cache_memory(self, blobs: Sequence[Any], cache_id: int = 0) -> None:
        if len(blobs) != len(self.shared_mxq_models):
            raise ValueError(
                f"MoE KV snapshot has {len(blobs)} layers, but the model has {len(self.shared_mxq_models)} shared MXQs."
            )
        for model, layer_blobs in zip(self.shared_mxq_models, blobs):
            model.load_cache_memory(layer_blobs, cache_id)
