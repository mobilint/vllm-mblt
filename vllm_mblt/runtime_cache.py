import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

KVBlockIds = tuple[list[int], ...]
RuntimeCacheDumpFn = Callable[[int | None], list[Any] | None]
RuntimeCacheLoadFn = Callable[[list[Any], int | None], bool]
DumpRuntimeCache = RuntimeCacheDumpFn
LoadRuntimeCache = RuntimeCacheLoadFn


@dataclass
class RuntimeCacheSnapshot:
    blobs: list[Any]
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_tokens: int
    cache_token_ids: tuple[int, ...] | None = None


@dataclass
class RuntimeCacheRequest:
    req_id: str
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_computed_tokens: int
    cache_slot_id: int | None = None
    cache_token_ids: tuple[int, ...] | None = None


@dataclass
class RuntimeCacheSnapshotIndexNode:
    children: dict[int, "RuntimeCacheSnapshotIndexNode"] = field(default_factory=dict)
    best_req_id: str | None = None
    best_num_tokens: int = 0
    req_ids: list[str] = field(default_factory=list)


@dataclass
class RuntimeCacheSnapshotMatch:
    snapshot: RuntimeCacheSnapshot | None
    matched_tokens: int
    req_id: str | None = None
    is_own_snapshot: bool = False

    def __iter__(self):
        # Preserve the worker-facing legacy shape: snapshot, matched_tokens.
        yield self.snapshot
        yield self.matched_tokens


@dataclass
class RuntimeCacheLoadResult:
    matched_tokens: int
    loaded: bool = False
    reused_live_cache: bool = False
    cache_miss: bool = False
    loaded_snapshot_req_id: str | None = None
    is_own_snapshot: bool = False
    action: str = "cache-miss"
    snapshot_req_id: str | None = None


class MbltRuntimeCacheManager:
    """Owns Mobilint runtime cache snapshots and accelerator ownership state.

    The manager intentionally stores opaque cache blobs only. It does not import
    qbruntime cache models or vLLM runtime objects. Actual cache dump/load IO is
    performed only through callables injected by the worker/integration layer.
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

        self.block_size = int(block_size)
        self.max_batch_size = max(0, int(max_batch_size))
        self.max_finished_cache_snapshots = int(max_finished_snapshots)
        self._dump_runtime_cache_fn = dump_runtime_cache
        self._load_runtime_cache_fn = load_runtime_cache

        self.snapshots: dict[str, RuntimeCacheSnapshot] = {}
        self.snapshot_index_root = RuntimeCacheSnapshotIndexNode()
        self.finished_snapshot_lru: OrderedDict[str, None] = OrderedDict()

        # Single-cache runtime owner.
        self.loaded_req_id: str | None = None

        # Batch cache slot assignment and live slot owners.
        self.req_to_slot: dict[str, int] = {}
        self.slot_to_req: dict[int, str] = {}
        self.slot_live_req: dict[int, str] = {}
        self.free_slots: list[int] = []
        self.reset_slots(max_batch_size=self.max_batch_size)

    @property
    def cache_snapshots(self) -> dict[str, RuntimeCacheSnapshot]:
        return self.snapshots

    @cache_snapshots.setter
    def cache_snapshots(self, snapshots: dict[str, RuntimeCacheSnapshot]) -> None:
        self.snapshots = snapshots
        self.rebuild_snapshot_index()

    @property
    def loaded_cache_req_id(self) -> str | None:
        return self.loaded_req_id

    @loaded_cache_req_id.setter
    def loaded_cache_req_id(self, req_id: str | None) -> None:
        self.loaded_req_id = req_id

    @property
    def req_to_cache_slot(self) -> dict[str, int]:
        return self.req_to_slot

    @property
    def cache_slot_to_req(self) -> dict[int, str]:
        return self.slot_to_req

    @property
    def free_cache_slots(self) -> list[int]:
        return self.free_slots

    def reset_slots(self, *, max_batch_size: int | None = None) -> None:
        if max_batch_size is not None:
            self.max_batch_size = max(0, int(max_batch_size))
        self.req_to_slot = {}
        self.slot_to_req = {}
        self.slot_live_req = {}
        self.free_slots = list(range(self.max_batch_size))

    def get_slot(self, req_id: str) -> int:
        slot_id = self.req_to_slot.get(req_id)
        if slot_id is None:
            raise RuntimeError(f"No accelerator cache slot is assigned for req_id={req_id}.")
        return slot_id

    def assign_slot(self, req_id: str) -> int:
        existing_slot = self.req_to_slot.get(req_id)
        if existing_slot is not None:
            return existing_slot
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
        if self.slot_live_req.get(slot_id) == req_id:
            self.slot_live_req.pop(slot_id, None)
        if slot_id not in self.free_slots:
            self.free_slots.append(slot_id)
            self.free_slots.sort()

    def live_slot_owner(self, slot_id: int) -> str | None:
        return self.slot_live_req.get(slot_id)

    def mark_slot_owner(self, slot_id: int, req_id: str) -> None:
        self.slot_live_req[slot_id] = req_id

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
        cache_token_ids: tuple[int, ...] | list[int] | None = None,
    ) -> RuntimeCacheSnapshot:
        return self.store_snapshot(
            req_id=req_id,
            blobs=blobs,
            block_ids=block_ids,
            first_seq_blocks=first_seq_block_ids if first_seq_block_ids is not None else first_seq_blocks(block_ids),
            num_tokens=num_tokens,
            cache_token_ids=cache_token_ids,
        )

    def store_snapshot(
        self,
        *,
        req_id: str,
        blobs: list[Any],
        block_ids: KVBlockIds,
        first_seq_blocks: tuple[int, ...],
        num_tokens: int,
        cache_token_ids: tuple[int, ...] | list[int] | None = None,
    ) -> RuntimeCacheSnapshot:
        self.finished_snapshot_lru.pop(req_id, None)
        snapshot = RuntimeCacheSnapshot(
            blobs=blobs,
            block_ids=normalize_block_ids(block_ids),
            first_seq_blocks=tuple(first_seq_blocks),
            num_tokens=max(0, int(num_tokens)),
            cache_token_ids=tuple(cache_token_ids) if cache_token_ids is not None else None,
        )
        self.snapshots[req_id] = snapshot
        self.rebuild_snapshot_index()
        return snapshot

    def remove_snapshot(self, req_id: str) -> RuntimeCacheSnapshot | None:
        snapshot = self.snapshots.pop(req_id, None)
        self.finished_snapshot_lru.pop(req_id, None)
        self.clear_loaded_request(req_id)
        if snapshot is not None:
            self.rebuild_snapshot_index()
        return snapshot

    @staticmethod
    def _update_snapshot_index_node(
        node: RuntimeCacheSnapshotIndexNode,
        req_id: str,
        num_tokens: int,
    ) -> None:
        node.req_ids.append(req_id)
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

    def dump_and_store_snapshot(
        self,
        *,
        req_id: str,
        block_ids: KVBlockIds,
        first_seq_blocks: tuple[int, ...],
        num_tokens: int,
        slot_id: int | None = None,
        cache_token_ids: tuple[int, ...] | list[int] | None = None,
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
            cache_token_ids=cache_token_ids,
        )

    def choose_prefix_snapshot(
        self,
        target_blocks: RuntimeCacheRequest | tuple[int, ...],
        target_tokens: int | None = None,
    ) -> RuntimeCacheSnapshotMatch:
        request = target_blocks if isinstance(target_blocks, RuntimeCacheRequest) else None
        if request is not None:
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
            if node is None or not node.req_ids:
                break

            for req_id in node.req_ids:
                snapshot = self.snapshots.get(req_id)
                if snapshot is None:
                    continue
                matched_tokens = min(snapshot.num_tokens, depth * self.block_size, target_tokens)
                matched_tokens = token_compatible_tokens(
                    matched_tokens,
                    request.cache_token_ids if request is not None else None,
                    snapshot.cache_token_ids,
                )
                if matched_tokens > best_tokens:
                    best_tokens = matched_tokens
                    best_snapshot = snapshot
                    best_req_id = req_id

        return RuntimeCacheSnapshotMatch(snapshot=best_snapshot, matched_tokens=best_tokens, req_id=best_req_id)

    def choose_snapshot(self, request: RuntimeCacheRequest) -> RuntimeCacheSnapshotMatch:
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            return RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)

        own_snapshot = self.snapshots.get(request.req_id)
        if own_snapshot is not None:
            matched_tokens = self.compatible_tokens(
                target_blocks=request.first_seq_blocks,
                target_tokens=target_tokens,
                snapshot_blocks=own_snapshot.first_seq_blocks,
                snapshot_tokens=own_snapshot.num_tokens,
                target_token_ids=request.cache_token_ids,
                snapshot_token_ids=own_snapshot.cache_token_ids,
            )
            if matched_tokens >= target_tokens:
                return RuntimeCacheSnapshotMatch(
                    snapshot=own_snapshot,
                    matched_tokens=target_tokens,
                    req_id=request.req_id,
                    is_own_snapshot=True,
                )

        return self.choose_prefix_snapshot(request)

    def compatible_tokens(
        self,
        *,
        target_blocks: tuple[int, ...],
        target_tokens: int,
        snapshot_blocks: tuple[int, ...],
        snapshot_tokens: int,
        target_token_ids: tuple[int, ...] | None = None,
        snapshot_token_ids: tuple[int, ...] | None = None,
    ) -> int:
        matched_tokens = prefix_compatible_tokens(
            target_blocks,
            target_tokens,
            snapshot_blocks,
            snapshot_tokens,
            self.block_size,
        )
        return token_compatible_tokens(matched_tokens, target_token_ids, snapshot_token_ids)

    def required_blocks(self, num_tokens: int) -> int:
        return required_blocks(num_tokens, self.block_size)

    def should_dump_snapshot_after_step(self, req_id: str, next_num_tokens: int) -> bool:
        snapshot = self.snapshots.get(req_id)
        if snapshot is None:
            return True
        if next_num_tokens <= snapshot.num_tokens:
            return False
        return self.required_blocks(next_num_tokens) > self.required_blocks(snapshot.num_tokens)

    def dump_snapshot_if_needed(
        self,
        request: RuntimeCacheRequest,
        dump_runtime_cache: DumpRuntimeCache | None = None,
    ) -> bool:
        if not self.should_dump_snapshot_after_step(request.req_id, request.num_computed_tokens):
            return False
        dump_runtime_cache = dump_runtime_cache or self._dump_runtime_cache_fn
        if dump_runtime_cache is None:
            raise RuntimeError("Runtime cache dump adapter is not configured.")
        blobs = dump_runtime_cache(request.cache_slot_id)
        if blobs is None:
            return False
        self.store_snapshot(
            req_id=request.req_id,
            blobs=blobs,
            block_ids=request.block_ids,
            first_seq_blocks=request.first_seq_blocks,
            num_tokens=request.num_computed_tokens,
            cache_token_ids=request.cache_token_ids,
        )
        return True

    def dump_live_request_before_switch(
        self,
        *,
        next_req_id: str,
        live_request: RuntimeCacheRequest | None,
        dump_runtime_cache: DumpRuntimeCache,
    ) -> bool:
        if self.loaded_req_id is None or self.loaded_req_id == next_req_id:
            return False
        if live_request is None or live_request.req_id != self.loaded_req_id:
            return False
        return self.dump_snapshot_if_needed(live_request, dump_runtime_cache)

    def load_snapshot_for_request(
        self,
        request: RuntimeCacheRequest,
        load_runtime_cache: LoadRuntimeCache | None = None,
    ) -> RuntimeCacheLoadResult:
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            self.clear_loaded_request()
            return RuntimeCacheLoadResult(matched_tokens=0, cache_miss=True, action="skip-empty")

        if self.loaded_req_id == request.req_id:
            return RuntimeCacheLoadResult(
                matched_tokens=target_tokens,
                reused_live_cache=True,
                action="reuse-live",
                snapshot_req_id=request.req_id,
            )

        match = self.choose_snapshot(request)
        if match.snapshot is None or match.matched_tokens <= 0:
            self.clear_loaded_request()
            return RuntimeCacheLoadResult(matched_tokens=0, cache_miss=True)

        load_runtime_cache = load_runtime_cache or self._load_runtime_cache_fn
        if load_runtime_cache is None:
            raise RuntimeError("Runtime cache load adapter is not configured.")
        if load_runtime_cache(match.snapshot.blobs, None):
            self.mark_loaded_request(request.req_id)
            return RuntimeCacheLoadResult(
                matched_tokens=match.matched_tokens,
                loaded=True,
                loaded_snapshot_req_id=match.req_id,
                is_own_snapshot=match.is_own_snapshot,
                action="load-own" if match.is_own_snapshot else "load-shared",
                snapshot_req_id=match.req_id,
            )

        self.clear_loaded_request()
        return RuntimeCacheLoadResult(matched_tokens=0, cache_miss=True)

    def load_for_request(self, request: RuntimeCacheRequest) -> RuntimeCacheLoadResult:
        return self.load_snapshot_for_request(request)

    def load_snapshot_for_slot(
        self,
        request: RuntimeCacheRequest,
        load_runtime_cache: LoadRuntimeCache | None = None,
    ) -> RuntimeCacheLoadResult:
        slot_id = request.cache_slot_id
        if slot_id is None:
            raise RuntimeError(f"Batch execution requires a cache slot for req_id={request.req_id}.")

        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            return RuntimeCacheLoadResult(matched_tokens=0, cache_miss=True, action="skip-empty")

        if self.live_slot_owner(slot_id) == request.req_id:
            return RuntimeCacheLoadResult(
                matched_tokens=target_tokens,
                reused_live_cache=True,
                action="reuse-live",
                snapshot_req_id=request.req_id,
            )

        match = self.choose_snapshot(request)
        if match.snapshot is not None and match.matched_tokens > 0:
            load_runtime_cache = load_runtime_cache or self._load_runtime_cache_fn
            if load_runtime_cache is None:
                raise RuntimeError("Runtime cache load adapter is not configured.")
            if load_runtime_cache(match.snapshot.blobs, slot_id):
                self.mark_slot_owner(slot_id, request.req_id)
                return RuntimeCacheLoadResult(
                    matched_tokens=match.matched_tokens,
                    loaded=True,
                    loaded_snapshot_req_id=match.req_id,
                    is_own_snapshot=match.is_own_snapshot,
                    action="load-own" if match.is_own_snapshot else "load-shared",
                    snapshot_req_id=match.req_id,
                )

        self.mark_slot_owner(slot_id, request.req_id)
        return RuntimeCacheLoadResult(matched_tokens=0, cache_miss=True)

    def load_for_slot(self, request: RuntimeCacheRequest, slot_id: int | None = None) -> RuntimeCacheLoadResult:
        if slot_id is not None and request.cache_slot_id != slot_id:
            request = RuntimeCacheRequest(
                req_id=request.req_id,
                block_ids=request.block_ids,
                first_seq_blocks=request.first_seq_blocks,
                num_computed_tokens=request.num_computed_tokens,
                cache_slot_id=slot_id,
            )
        return self.load_snapshot_for_slot(request)

    def touch_finished_snapshot(self, req_id: str) -> None:
        if req_id not in self.snapshots:
            return
        self.finished_snapshot_lru.pop(req_id, None)
        self.finished_snapshot_lru[req_id] = None

    def mark_snapshot_finished(self, req_id: str) -> list[str]:
        self.touch_finished_snapshot(req_id)
        return self.evict_old_finished_snapshots()

    def evict_old_finished_snapshots(self) -> list[str]:
        evicted: list[str] = []
        while len(self.finished_snapshot_lru) > self.max_finished_cache_snapshots:
            evicted_req_id, _ = self.finished_snapshot_lru.popitem(last=False)
            self.snapshots.pop(evicted_req_id, None)
            self.clear_loaded_request(evicted_req_id)
            evicted.append(evicted_req_id)
        if evicted:
            self.rebuild_snapshot_index()
        return evicted

    def reset(self) -> None:
        self.snapshots.clear()
        self.finished_snapshot_lru.clear()
        self.snapshot_index_root = RuntimeCacheSnapshotIndexNode()
        self.loaded_req_id = None
        self.reset_slots()


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


def token_compatible_tokens(
    matched_tokens: int,
    target_token_ids: tuple[int, ...] | None,
    snapshot_token_ids: tuple[int, ...] | None,
) -> int:
    if matched_tokens <= 0:
        return 0
    if target_token_ids is None or snapshot_token_ids is None:
        return matched_tokens

    common_tokens = 0
    for target_token_id, snapshot_token_id in zip(target_token_ids, snapshot_token_ids):
        if target_token_id != snapshot_token_id:
            break
        common_tokens += 1
        if common_tokens >= matched_tokens:
            break
    return min(matched_tokens, common_tokens)
