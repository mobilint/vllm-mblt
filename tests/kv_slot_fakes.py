"""Fake NPU KV slots for worker tests: real transformers-mblt caches and owner table, fake MXQ dump/load."""

from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import torch

from vllm_mblt.request_cache import RequestCacheSlots


class FakeKVSlotModel:
    """An MXQ whose KV slots hold a label of what was written there, recording every dump and load."""

    def __init__(self, name: str = "mxq", kv_nbytes: int = 0) -> None:
        self.name = name
        self.kv_nbytes = kv_nbytes
        self.slots: dict[int, Any] = {}
        self.dumps: list[int] = []
        self.loads: list[tuple[int, Any]] = []

    def get_cache_infos(self) -> list[Any]:
        """Two KV buffers splitting ``kv_nbytes``, as qbruntime reports one slot's compiled KV buffers."""
        if not self.kv_nbytes:
            raise AttributeError("get_cache_infos")
        return [SimpleNamespace(size=self.kv_nbytes // 2), SimpleNamespace(size=self.kv_nbytes - self.kv_nbytes // 2)]

    def dump_cache_memory(self, cache_id: int = 0) -> list[Any]:
        self.dumps.append(int(cache_id))
        return [("kv", self.name, int(cache_id), self.slots.get(int(cache_id)))]

    def load_cache_memory(self, blobs: list[Any], cache_id: int = 0) -> None:
        self.loads.append((int(cache_id), blobs))
        self.slots[int(cache_id)] = blobs[0][3]


def attach_request_caches(
    worker: Any,
    *,
    k_per_model: int = 1,
    models: Optional[list[FakeKVSlotModel]] = None,
) -> list[FakeKVSlotModel]:
    """Give a worker built with ``MbltWorker.__new__`` per-request caches on fake slots; return the fake MXQs."""
    models = models if models is not None else [FakeKVSlotModel()]
    worker.request_caches = RequestCacheSlots(models, k_per_model)
    worker._slot_caches = {}
    return models


def make_snapshot_cache(worker: Any, num_tokens: int, slot_id: Optional[int] = None) -> Any:
    """Return a cache holding ``num_tokens`` tokens in host memory, as a stored prefix snapshot does.

    Ran on the slot and saved from it like a real one; the fake MXQs' dump/load records are cleared afterwards.
    """
    cache = worker.request_caches.new_cache(slot_id)
    worker.request_caches.prepare(cache, slot_id, 0)
    cache.update_seen_tokens(num_tokens)
    cache.offload()
    for model in worker.request_caches.models:
        model.dumps.clear()
        model.loads.clear()
    return cache


class FakeForwardModel:
    """A transformers-mblt forward double that advances the request caches it runs on.

    Row ``i`` of a call runs ``step(embeds, cache_size, slot)`` on its real (unmasked) tokens, where ``slot`` is the
    row's cache id on the MXQ; ``step`` returns that row's logits, one per token or only the last token's. Each call
    is recorded in ``calls`` as one ``(tokens, cache_size, slot)`` tuple per row, and its extra forward arguments in
    ``kwargs``. ``logits_to_keep=1`` returns each row's last logits; ``logits_to_keep=0`` each row's positions from
    column 0, as the transformers-mblt batched forward does.
    """

    def __init__(self, step: Callable[[np.ndarray, int, int], np.ndarray], config: Any = None) -> None:
        self.step = step
        self.config = config
        self.calls: list[tuple[tuple[int, int, int], ...]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, *, inputs_embeds, attention_mask=None, past_key_values, logits_to_keep, **kwargs):
        rows, width = inputs_embeds.shape[:2]
        mask = torch.ones(rows, width, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        keep_all = logits_to_keep == 0
        call, outputs = [], []
        for row in range(rows):
            embeds = inputs_embeds[row][mask[row]].numpy()
            start = past_key_values.get_seq_length(row)
            slot = past_key_values.slot_of(row)[1]
            logits = np.asarray(self.step(embeds, start, slot), dtype=np.float32)
            logits = logits.reshape(-1, logits.shape[-1])
            past_key_values.update_seen_tokens(len(embeds), index=row)
            call.append((len(embeds), start, slot))
            if keep_all:
                assert logits.shape[0] == len(embeds), "a full-logits forward needs one row per token"
                logits = np.concatenate([logits, np.zeros((width - len(embeds), logits.shape[1]), np.float32)])
            outputs.append(logits if keep_all else logits[-1:])
        self.calls.append(tuple(call))
        self.kwargs.append(kwargs)
        return SimpleNamespace(logits=torch.from_numpy(np.stack(outputs)))

