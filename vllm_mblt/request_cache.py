"""Per-request KV caches on Mobilint NPU slots.

Every request keeps one transformers-mblt cache, placed on the NPU slot the worker assigned to it. Before an infer
writes a slot, the worker binds the cache of the request that infer belongs to: transformers-mblt's owner table
saves whichever cache held the slot to host memory and loads this cache's KV if it was moved out. A request that
loses its slot to another request therefore keeps its KV, and the worker never dumps or loads KV itself.

Prefix snapshots are cache copies. A finished request's cache becomes its snapshot as it is, so its KV is dumped
only when another cache needs the slot, and not at all if the snapshot is dropped first.
"""

from types import SimpleNamespace
from typing import Any, Optional, Sequence

from transformers_mblt.utils.cache_residency import bind_cacheless
from transformers_mblt.utils.cache_utils import MobilintCache


class RequestCacheSlots:
    """Builds request caches on the worker's NPU slots and binds them around each infer.

    Slot ``s`` is row ``divmod(s, k_per_model)`` of the cache MXQs, as in the worker's batch ``cache_id``; a
    non-batch worker has the one slot ``None`` on ``(models[0], 0)``. A Mixture-of-Experts model keeps its KV in
    every shared MXQ and binds its cache inside its own forward: ``moe`` is the worker's
    :class:`~vllm_mblt.moe_runtime.MbltMoECacheModel`, and preparing a cache hands it to that adapter.
    """

    def __init__(
        self,
        models: Sequence[Any],
        k_per_model: int = 1,
        *,
        moe: Optional[Any] = None,
    ) -> None:
        self.models = list(models)
        self.k_per_model = max(1, int(k_per_model))
        self.moe = moe
        # What MobilintCache.bind_rows reads to place rows: the worker's own topology, not a model-zoo backend.
        self._topology = SimpleNamespace(mxq_models=self.models, k_per_model=self.k_per_model)

    def _row_slot(self, slot_id: Optional[int]) -> tuple[int, int]:
        if slot_id is None:
            return (0, 0)
        model_idx, cache_id = divmod(int(slot_id), self.k_per_model)
        if model_idx >= len(self.models):
            raise RuntimeError(
                "Mobilint cache slot exceeds the loaded multi-card capacity: "
                f"cache_id={slot_id}, models={len(self.models)}, k_per_model={self.k_per_model}."
            )
        return model_idx, cache_id

    def new_cache(self, slot_id: Optional[int]) -> Any:
        """Return an empty cache for the request on ``slot_id``."""
        if self.moe is not None:
            return self.moe.new_cache()
        return MobilintCache(self.models, per_model_batch=self.k_per_model, slots=[self._row_slot(slot_id)])

    def kv_nbytes_per_cache(self) -> Optional[int]:
        """Host bytes one request's KV takes once saved: the compiled KV buffers of one slot.

        A Mixture-of-Experts cache spans every shared MXQ. ``None`` when the MXQs do not report their buffers.
        """
        models = self.moe.shared_mxq_models if self.moe is not None else self.models[:1]
        try:
            total = sum(int(info.size) for model in models for info in model.get_cache_infos())
        except (AttributeError, TypeError, ValueError):
            return None
        return total or None

    def holds_slot(self, cache: Any, slot_id: Optional[int]) -> bool:
        """Whether ``cache`` sits on ``slot_id`` and the slot holds its KV right now."""
        if self.moe is not None:
            return bool(cache.is_resident)
        return cache.slot_of(0) == self._row_slot(slot_id) and bool(cache.is_resident)

    def relocate(self, snapshot: Any, slot_id: Optional[int], num_tokens: int) -> Any:
        """Return an independent copy of ``snapshot``'s first ``num_tokens`` tokens on ``slot_id``.

        The copy saves a resident snapshot's KV to host memory; the slot loads it on the copy's first bind.
        """
        copied = snapshot.copy()
        if self.moe is None and copied.slot_of(0) != self._row_slot(slot_id):
            moved = self.new_cache(slot_id)
            # The first bind moves the copied row's KV to this slot (transformers-mblt checks that the MXQs match).
            moved.layers[0] = copied.layers[0]
            moved.rope_deltas = copied.rope_deltas
            copied = moved
        copied.set_seq_length(int(num_tokens))
        return copied

    def prepare(self, cache: Optional[Any], slot_id: Optional[int], cache_size: int) -> None:
        """Make ``slot_id`` hold ``cache``'s first ``cache_size`` tokens before an infer writes from there.

        Without a cache (prefill-cost calibration) the slot's owner is saved and the slot is left unowned.
        """
        if self.moe is not None:
            self.moe.use_cache(cache)
        elif cache is None:
            bind_cacheless([(self.models[self._row_slot(slot_id)[0]], self._row_slot(slot_id)[1])])
        if cache is None:
            return
        if cache_size > cache.get_seq_length():
            raise RuntimeError(
                f"Cannot infer from position {cache_size}: the request's cache holds {cache.get_seq_length()} tokens."
            )
        cache.set_seq_length(int(cache_size))
        if self.moe is None:
            cache.bind_rows(self._topology, [0])

    def commit(self, cache: Optional[Any], num_tokens: int) -> None:
        """Record that the infer just written ``num_tokens`` tokens of KV after the prepared position."""
        if cache is None or self.moe is not None:
            return  # the MoE forward advances its own cache
        cache.update_seen_tokens(int(num_tokens))
