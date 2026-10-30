"""Host cache of vision encoder outputs, keyed by vLLM's multimodal item hash.

vLLM's GPU model runner keeps each image's encoder output in an encoder cache keyed by the item's content hash and
reuses it while it is cached. This worker rebuilds a new request's whole prompt embedding, including every image it
carries, so a repeated image (a follow-up chat turn that resends it, or the same picture in several requests) would
run through the NPU vision encoder again. The cache below keeps the outputs of recent items in host memory instead.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Optional

import torch


@dataclass(frozen=True)
class EncodedVisionItem:
    """One image or video encoded: its merged embeddings and, for Qwen3-VL, one tensor per DeepStack layer."""

    embeds: torch.Tensor
    deepstack: Optional[tuple[torch.Tensor, ...]] = None

    @property
    def nbytes(self) -> int:
        tensors = (self.embeds, *(self.deepstack or ()))
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class EncoderOutputCache:
    """Least-recently-used encoder outputs within a host-memory budget; a budget of 0 turns the cache off."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[Hashable, EncodedVisionItem] = OrderedDict()
        self._nbytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def clear(self) -> None:
        self._items.clear()
        self._nbytes = 0

    def get(self, key: Hashable) -> Optional[EncodedVisionItem]:
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: Hashable, item: EncodedVisionItem) -> None:
        size = item.nbytes
        if size > self.max_bytes:
            return
        previous = self._items.pop(key, None)
        if previous is not None:
            self._nbytes -= previous.nbytes
        while self._items and self._nbytes + size > self.max_bytes:
            _, evicted = self._items.popitem(last=False)
            self._nbytes -= evicted.nbytes
        self._items[key] = item
        self._nbytes += size
