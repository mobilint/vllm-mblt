import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


KVBlockIds = tuple[list[int], ...]
RuntimeCacheDumpFn = Callable[[int | None], list[Any] | None]
RuntimeCacheLoadFn = Callable[[list[Any], int | None], bool]


@dataclass
class RuntimeCacheSnapshot:
    blobs: list[Any]
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_tokens: int


@dataclass
class RuntimeCacheRequest:
    req_id: str
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_computed_tokens: int
    cache_slot_id: int | None = None


@dataclass
class RuntimeCacheSnapshotIndexNode:
    children: dict[int, "RuntimeCacheSnapshotIndexNode"] = field(default_factory=dict)
    best_req_id: str | None = None
    best_num_tokens: int = 0


@dataclass
class RuntimeCacheSnapshotMatch:
    snapshot: RuntimeCacheSnapshot | None
    matched_tokens: int
    req_id: str | None = None
    is_own_snapshot: bool = False

    def __iter__(self):
        yield self.snapshot
        yield self.matched_tokens


@dataclass
class RuntimeCacheLoadResult:
    matched_tokens: int
    action: str
    snapshot_req_id: str | None = None


def normalize_block_ids(block_ids: KVBlockIds) -> KVBlockIds:
    return tuple(list(seq) for seq in block_ids)


def append_block_ids(
    current_block_ids: KVBlockIds,
    new_block_ids: KVBlockIds,
) -> KVBlockIds:
    if not current_block_ids:
        return normalize_block_ids(new_block_ids)
    if len(current_block_ids) != len(new_block_ids):
        raise RuntimeError(
            f"KV block_ids layout mismatch: current={len(current_block_ids)} seqs, new={len(new_block_ids)} seqs"
        )
    merged_block_ids: list[list[int]] = []
    for current_seq_blocks, new_seq_blocks in zip(current_block_ids, new_block_ids):
        merged_block_ids.append(list(current_seq_blocks) + list(new_seq_blocks))
    return tuple(merged_block_ids)


def first_seq_blocks(block_ids: KVBlockIds) -> tuple[int, ...]:
    if len(block_ids) == 0:
        return ()
    return tuple(block_ids[0])


def required_blocks(num_tokens: int, block_size: int) -> int:
    if num_tokens <= 0:
        return 0
    return math.ceil(num_tokens / block_size)


def prefix_compatible_tokens(
    target_blocks: tuple[int, ...],
    target_tokens: int,
    snapshot_blocks: tuple[int, ...],
    snapshot_tokens: int,
    block_size: int,
) -> int:
    if target_tokens <= 0 or snapshot_tokens <= 0:
        return 0

    needed_target_blocks = required_blocks(target_tokens, block_size)
    common_blocks = 0
    for i in range(min(needed_target_blocks, len(target_blocks), len(snapshot_blocks))):
        if target_blocks[i] != snapshot_blocks[i]:
            break
        common_blocks += 1

    if common_blocks == 0:
        return 0

    if common_blocks >= needed_target_blocks:
        return min(target_tokens, snapshot_tokens)

    return min(snapshot_tokens, common_blocks * block_size, target_tokens)


class MbltRuntimeCacheManager:
    """Owns Mobilint runtime cache snapshots and accelerator slot state.

    The manager stores opaque runtime blobs and calls injected dump/load adapters
    for qbruntime/cache-model IO. It intentionally does not import or own vLLM
    worker/request objects or qbruntime implementation types.
    """

    DEFAULT_MAX_FINISHED_CACHE_SNAPSHOTS = 16

    def __init__(
        self,
        block_size: int,
        max_finished_cache_snapshots: int | None = None,
        *,
        max_batch_size: int = 1,
        max_finished_snapshots: int | None = None,
        dump_runtime_cache: RuntimeCacheDumpFn | None = None,
        load_runtime_cache: RuntimeCacheLoadFn | None = None,
    ) -> None:
        if max_finished_snapshots is None:
            max_finished_snapshots = max_finished_cache_snapshots
        if max_finished_snapshots is None:
            max_finished_snapshots = self.DEFAULT_MAX_FINISHED_CACHE_SNAPSHOTS

        self.max_batch_size = max(0, int(max_batch_size))
        self.block_size = int(block_size)
        self.max_finished_cache_snapshots = int(max_finished_snapshots)
        self._dump_runtime_cache_fn = dump_runtime_cache
        self._load_runtime_cache_fn = load_runtime_cache

        self.snapshots: dict[str, RuntimeCacheSnapshot] = {}
        self.snapshot_index_root = RuntimeCacheSnapshotIndexNode()
        self.finished_snapshot_lru: OrderedDict[str, None] = OrderedDict()
        self.loaded_req_id: str | None = None
        self.req_to_slot: dict[str, int] = {}
        self.slot_to_req: dict[int, str] = {}
        self.free_slots: list[int] = []
        self.reset_slots(max_batch_size=self.max_batch_size)

    @property
    def cache_snapshots(self) -> dict[str, RuntimeCacheSnapshot]:
        return self.snapshots

    @cache_snapshots.setter
    def cache_snapshots(self, value: dict[str, RuntimeCacheSnapshot]) -> None:
        self.snapshots = value

    @property
    def loaded_cache_req_id(self) -> str | None:
        return self.loaded_req_id

    @loaded_cache_req_id.setter
    def loaded_cache_req_id(self, value: str | None) -> None:
        self.loaded_req_id = value

    @property
    def req_to_cache_slot(self) -> dict[str, int]:
        return self.req_to_slot

    @property
    def cache_slot_to_req(self) -> dict[int, str]:
        return self.slot_to_req

    @property
    def free_cache_slots(self) -> list[int]:
        return self.free_slots

    @property
    def max_finished_snapshots(self) -> int:
        return self.max_finished_cache_snapshots

    @max_finished_snapshots.setter
    def max_finished_snapshots(self, value: int) -> None:
        self.max_finished_cache_snapshots = int(value)

    def set_io_adapters(
        self,
        *,
        dump_runtime_cache: RuntimeCacheDumpFn | None = None,
        load_runtime_cache: RuntimeCacheLoadFn | None = None,
    ) -> None:
        if dump_runtime_cache is not None:
            self._dump_runtime_cache_fn = dump_runtime_cache
        if load_runtime_cache is not None:
            self._load_runtime_cache_fn = load_runtime_cache

    def dump_runtime_cache(self, dump_cache: Any, slot_id: int | None = None) -> list[Any] | None:
        if slot_id is None:
            return dump_cache()
        return dump_cache(cache_id=slot_id)

    def load_runtime_cache(self, load_cache: Any, blobs: list[Any], slot_id: int | None = None) -> bool:
        if slot_id is None:
            load_cache(blobs)
        else:
            load_cache(blobs, cache_id=slot_id)
        return True

    def reset_slots(self, *, max_batch_size: int | None = None) -> None:
        if max_batch_size is not None:
            self.max_batch_size = max(0, int(max_batch_size))
        self.req_to_slot = {}
        self.slot_to_req = {}
        self.free_slots = list(range(self.max_batch_size))

    def get_slot(self, req_id: str) -> int:
        slot_id = self.req_to_slot.get(req_id)
        if slot_id is None:
            raise RuntimeError(f"No accelerator cache slot is assigned for req_id={req_id}.")
        return slot_id

    def assign_slot(self, req_id: str) -> int:
        slot_id = self.req_to_slot.get(req_id)
        if slot_id is not None:
            return slot_id
        if not self.free_slots:
            raise RuntimeError(
                "No free accelerator cache slots remain for batch execution. "
                f"req_id={req_id}, max_batch_size={self.max_batch_size}"
            )
        slot_id = self.free_slots.pop(0)
        self.req_to_slot[req_id] = slot_id
        self.slot_to_req[slot_id] = req_id
        return slot_id

    def release_slot(self, req_id: str) -> None:
        slot_id = self.req_to_slot.pop(req_id, None)
        if slot_id is None:
            return
        if self.slot_to_req.get(slot_id) == req_id:
            self.slot_to_req.pop(slot_id, None)
        if slot_id not in self.free_slots:
            self.free_slots.append(slot_id)
            self.free_slots.sort()

    def live_slot_owner(self, slot_id: int) -> str | None:
        return self.slot_to_req.get(slot_id)

    def mark_slot_owner(self, slot_id: int, req_id: str) -> None:
        self.slot_to_req[slot_id] = req_id

    def clear_loaded_request(self, req_id: str | None = None) -> None:
        if req_id is None or self.loaded_req_id == req_id:
            self.loaded_req_id = None

    def mark_loaded_request(self, req_id: str) -> None:
        self.loaded_req_id = req_id

    def get_snapshot(self, req_id: str) -> RuntimeCacheSnapshot | None:
        return self.snapshots.get(req_id)

    def has_snapshot(self, req_id: str) -> bool:
        return req_id in self.snapshots

    def snapshot_count(self) -> int:
        return len(self.snapshots)

    def put_snapshot(
        self,
        req_id: str,
        blobs: list[Any],
        block_ids: KVBlockIds,
        num_tokens: int,
        first_seq_block_ids: tuple[int, ...] | None = None,
    ) -> RuntimeCacheSnapshot:
        normalized_block_ids = normalize_block_ids(block_ids)
        snapshot = RuntimeCacheSnapshot(
            blobs=blobs,
            block_ids=normalized_block_ids,
            first_seq_blocks=first_seq_block_ids if first_seq_block_ids is not None else first_seq_blocks(normalized_block_ids),
            num_tokens=max(0, int(num_tokens)),
        )
        self.finished_snapshot_lru.pop(req_id, None)
        self.snapshots[req_id] = snapshot
        self.rebuild_snapshot_index()
        return snapshot

    def store_snapshot(
        self,
        *,
        req_id: str,
        blobs: list[Any],
        block_ids: KVBlockIds,
        first_seq_blocks: tuple[int, ...],
        num_tokens: int,
    ) -> RuntimeCacheSnapshot:
        return self.put_snapshot(
            req_id=req_id,
            blobs=blobs,
            block_ids=block_ids,
            first_seq_block_ids=first_seq_blocks,
            num_tokens=num_tokens,
        )

    def dump_and_store_snapshot(
        self,
        *,
        req_id: str,
        block_ids: KVBlockIds,
        first_seq_blocks: tuple[int, ...],
        num_tokens: int,
        slot_id: int | None = None,
    ) -> RuntimeCacheSnapshot | None:
        if self._dump_runtime_cache_fn is None:
            raise RuntimeError("Runtime cache dump adapter is not configured.")
        blobs = self._dump_runtime_cache_fn(slot_id)
        if blobs is None:
            return None
        return self.store_snapshot(
            req_id=req_id,
            blobs=blobs,
            block_ids=block_ids,
            first_seq_blocks=first_seq_blocks,
            num_tokens=num_tokens,
        )

    def remove_snapshot(self, req_id: str) -> RuntimeCacheSnapshot | None:
        snapshot = self.snapshots.pop(req_id, None)
        self.finished_snapshot_lru.pop(req_id, None)
        if self.loaded_req_id == req_id:
            self.loaded_req_id = None
        if snapshot is not None:
            self.rebuild_snapshot_index()
        return snapshot

    @staticmethod
    def _update_snapshot_index_node(
        node: RuntimeCacheSnapshotIndexNode,
        req_id: str,
        num_tokens: int,
    ) -> None:
        if num_tokens >= node.best_num_tokens:
            node.best_req_id = req_id
            node.best_num_tokens = num_tokens

    def rebuild_snapshot_index(self) -> None:
        root = RuntimeCacheSnapshotIndexNode()
        for req_id, snapshot in self.snapshots.items():
            node = root
            self._update_snapshot_index_node(node, req_id, snapshot.num_tokens)
            for block_id in snapshot.first_seq_blocks:
                node = node.children.setdefault(block_id, RuntimeCacheSnapshotIndexNode())
                self._update_snapshot_index_node(node, req_id, snapshot.num_tokens)
        self.snapshot_index_root = root

    def mark_snapshot_finished(self, req_id: str) -> list[str]:
        if req_id in self.snapshots:
            self.touch_finished_snapshot(req_id)
        return self.evict_old_finished_snapshots()

    def touch_finished_snapshot(self, req_id: str) -> None:
        if req_id in self.finished_snapshot_lru:
            self.finished_snapshot_lru.move_to_end(req_id)
        else:
            self.finished_snapshot_lru[req_id] = None

    def evict_old_finished_snapshots(self) -> list[str]:
        evicted_req_ids: list[str] = []
        while len(self.finished_snapshot_lru) > self.max_finished_cache_snapshots:
            evicted_req_id, _ = self.finished_snapshot_lru.popitem(last=False)
            self.snapshots.pop(evicted_req_id, None)
            if self.loaded_req_id == evicted_req_id:
                self.loaded_req_id = None
            evicted_req_ids.append(evicted_req_id)
        if evicted_req_ids:
            self.rebuild_snapshot_index()
        return evicted_req_ids

    def required_blocks(self, num_tokens: int) -> int:
        return required_blocks(num_tokens, self.block_size)

    def prefix_compatible_tokens(
        self,
        target_blocks: tuple[int, ...],
        target_tokens: int,
        snapshot_blocks: tuple[int, ...],
        snapshot_tokens: int,
    ) -> int:
        return prefix_compatible_tokens(
            target_blocks,
            target_tokens,
            snapshot_blocks,
            snapshot_tokens,
            self.block_size,
        )

    def compatible_tokens(
        self,
        *,
        target_blocks: tuple[int, ...],
        target_tokens: int,
        snapshot_blocks: tuple[int, ...],
        snapshot_tokens: int,
    ) -> int:
        return self.prefix_compatible_tokens(target_blocks, target_tokens, snapshot_blocks, snapshot_tokens)

    def should_dump_snapshot_after_step(self, req_id: str, next_num_tokens: int) -> bool:
        snapshot = self.snapshots.get(req_id)
        if snapshot is None:
            return True
        if next_num_tokens <= snapshot.num_tokens:
            return False
        return self.required_blocks(next_num_tokens) > self.required_blocks(snapshot.num_tokens)

    def choose_prefix_snapshot(
        self,
        target_blocks: RuntimeCacheRequest | tuple[int, ...],
        target_tokens: int | None = None,
    ) -> RuntimeCacheSnapshotMatch:
        if isinstance(target_blocks, RuntimeCacheRequest):
            request = target_blocks
            target_blocks = request.first_seq_blocks
            target_tokens = request.num_computed_tokens
        if target_tokens is None:
            raise TypeError("target_tokens is required when target_blocks is not a RuntimeCacheRequest")
        if target_tokens <= 0 or not target_blocks:
            return RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)

        best_snapshot: RuntimeCacheSnapshot | None = None
        best_req_id: str | None = None
        best_tokens = 0
        node = self.snapshot_index_root
        for depth, block_id in enumerate(target_blocks, start=1):
            node = node.children.get(block_id)
            if node is None or node.best_req_id is None:
                break

            snapshot = self.snapshots.get(node.best_req_id)
            if snapshot is None:
                continue

            matched_tokens = min(snapshot.num_tokens, depth * self.block_size, target_tokens)
            if matched_tokens > best_tokens:
                best_tokens = matched_tokens
                best_snapshot = snapshot
                best_req_id = node.best_req_id

        return RuntimeCacheSnapshotMatch(snapshot=best_snapshot, matched_tokens=best_tokens, req_id=best_req_id)

    def choose_snapshot(self, request: RuntimeCacheRequest) -> RuntimeCacheSnapshotMatch:
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            return RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)

        own_snapshot = self.snapshots.get(request.req_id)
        if own_snapshot is not None:
            matched_tokens = self.prefix_compatible_tokens(
                target_blocks=request.first_seq_blocks,
                target_tokens=target_tokens,
                snapshot_blocks=own_snapshot.first_seq_blocks,
                snapshot_tokens=own_snapshot.num_tokens,
            )
            if matched_tokens >= target_tokens:
                return RuntimeCacheSnapshotMatch(
                    snapshot=own_snapshot,
                    matched_tokens=target_tokens,
                    req_id=request.req_id,
                    is_own_snapshot=True,
                )

        return self.choose_prefix_snapshot(request.first_seq_blocks, target_tokens)

    def load_for_request(self, request: RuntimeCacheRequest) -> RuntimeCacheLoadResult:
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            self.clear_loaded_request()
            return RuntimeCacheLoadResult(matched_tokens=0, action="skip-empty")

        if self.loaded_req_id == request.req_id:
            return RuntimeCacheLoadResult(matched_tokens=target_tokens, action="reuse-live", snapshot_req_id=request.req_id)

        match = self.choose_snapshot(request)
        if match.snapshot is None or match.matched_tokens <= 0:
            self.clear_loaded_request()
            return RuntimeCacheLoadResult(matched_tokens=0, action="miss")

        if self._load_runtime_cache_fn is None:
            raise RuntimeError("Runtime cache load adapter is not configured.")
        if not self._load_runtime_cache_fn(match.snapshot.blobs, None):
            self.clear_loaded_request()
            return RuntimeCacheLoadResult(matched_tokens=0, action="load-failed", snapshot_req_id=match.req_id)

        self.mark_loaded_request(request.req_id)
        action = "load-own" if match.is_own_snapshot else "load-shared"
        return RuntimeCacheLoadResult(matched_tokens=match.matched_tokens, action=action, snapshot_req_id=match.req_id)

    def load_for_slot(self, request: RuntimeCacheRequest, slot_id: int) -> RuntimeCacheLoadResult:
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            self.mark_slot_owner(slot_id, request.req_id)
            return RuntimeCacheLoadResult(matched_tokens=0, action="skip-empty", snapshot_req_id=request.req_id)

        if self.live_slot_owner(slot_id) == request.req_id:
            return RuntimeCacheLoadResult(matched_tokens=target_tokens, action="reuse-live", snapshot_req_id=request.req_id)

        match = self.choose_snapshot(request)
        if match.snapshot is not None and match.matched_tokens > 0:
            if self._load_runtime_cache_fn is None:
                raise RuntimeError("Runtime cache load adapter is not configured.")
            if self._load_runtime_cache_fn(match.snapshot.blobs, slot_id):
                self.mark_slot_owner(slot_id, request.req_id)
                action = "load-own" if match.is_own_snapshot else "load-shared"
                return RuntimeCacheLoadResult(
                    matched_tokens=match.matched_tokens,
                    action=action,
                    snapshot_req_id=match.req_id,
                )

        self.mark_slot_owner(slot_id, request.req_id)
        return RuntimeCacheLoadResult(matched_tokens=0, action="miss")

    def reset(self) -> None:
        self.snapshots.clear()
        self.finished_snapshot_lru.clear()
        self.snapshot_index_root = RuntimeCacheSnapshotIndexNode()
        self.loaded_req_id = None
        self.reset_slots()