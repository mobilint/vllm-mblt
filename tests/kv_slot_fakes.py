"""Fake NPU KV slots for worker tests: real transformers-mblt caches and owner table, fake MXQ dump/load."""

from types import SimpleNamespace
from typing import Any, Optional

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
    worker.request_caches.commit(cache, num_tokens)
    cache.offload()
    for model in worker.request_caches.models:
        model.dumps.clear()
        model.loads.clear()
    return cache
