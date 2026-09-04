import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Hashable

KVBlockIds = tuple[list[int], ...]
RuntimeCacheDumpFn = Callable[[int | None], list[Any] | None]
RuntimeCacheLoadFn = Callable[[list[Any], int | None], bool]
DumpRuntimeCache = RuntimeCacheDumpFn
LoadRuntimeCache = RuntimeCacheLoadFn


@dataclass
class PromptEmbedCacheIdentity:
    prompt_len: int
    fingerprint_for_prefix: Callable[[int], Hashable | None]

    def fingerprint(self, num_tokens: int) -> Hashable | None:
        num_tokens = max(0, int(num_tokens))
        if num_tokens <= 0:
            return None
        if num_tokens > self.prompt_len:
            return None
        return self.fingerprint_for_prefix(num_tokens)


@dataclass
class RuntimeCacheSnapshot:
    blobs: list[Any]
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_tokens: int
    cache_token_ids: tuple[int, ...] | None = None
    multimodal_cache_identity: Hashable | None = None
    first_seq_block_hashes: tuple[Hashable, ...] | None = None
    prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None


@dataclass
class RuntimeCacheRequest:
    req_id: str
    block_ids: KVBlockIds
    first_seq_blocks: tuple[int, ...]
    num_computed_tokens: int
    cache_slot_id: int | None = None
    cache_token_ids: tuple[int, ...] | None = None
    multimodal_cache_identity: Hashable | None = None
    first_seq_block_hashes: tuple[Hashable, ...] | None = None
    prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None


@dataclass
class RuntimeCacheSnapshotIndexNode:
    children: dict[Hashable, "RuntimeCacheSnapshotIndexNode"] = field(default_factory=dict)
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
    live_prefix_incomplete: bool = False
    live_cache_tokens: int | None = None


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
        self.physical_block_owner: dict[int, str] = {}

        # Single-cache runtime owner.
        self.loaded_req_id: str | None = None
        # Token count the live single runtime cache is believed to hold for
        # loaded_req_id. None means the count is not tracked.
        self.loaded_req_tokens: int | None = None

        # Batch cache slot assignment and live slot owners.
        self.req_to_slot: dict[str, int] = {}
        self.slot_to_req: dict[int, str] = {}
        self.slot_live_req: dict[int, str] = {}
        self.slot_live_tokens: dict[int, int] = {}
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
        self.loaded_req_tokens = None

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
        self.slot_live_tokens = {}
        self.free_slots = list(range(self.max_batch_size))

    def _request_index_blocks(self, request: RuntimeCacheRequest) -> tuple[Hashable, ...]:
        if request.first_seq_block_hashes is not None:
            return tuple(("hash", block_hash) for block_hash in request.first_seq_block_hashes)
        return tuple(("physical", block_id) for block_id in request.first_seq_blocks)

    def _snapshot_index_blocks(self, snapshot: RuntimeCacheSnapshot) -> tuple[Hashable, ...]:
        if snapshot.first_seq_block_hashes is not None:
            return tuple(("hash", block_hash) for block_hash in snapshot.first_seq_block_hashes)
        return tuple(("physical", block_id) for block_id in snapshot.first_seq_blocks)

    def observe_request_blocks(
        self,
        req_id: str,
        block_ids: KVBlockIds,
        *,
        first_seq_block_hashes: tuple[Hashable, ...] | None = None,
    ) -> list[str]:
        """Record physical KV block ownership observed from the scheduler.

        vLLM 0.11.2 exposes physical block IDs but not scheduler block hashes.
        A changed physical owner is therefore only a collision signal, not proof
        that a finished snapshot is stale: prefix caching can intentionally share
        physical blocks across requests with identical token content. Snapshot
        removal is deferred to choose/load, where token and multimodal identity
        are available to prove incompatibility.
        """
        for block_id in first_seq_blocks(block_ids):
            self.physical_block_owner[block_id] = req_id
        return []

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
            self.slot_live_tokens.pop(slot_id, None)
        if slot_id not in self.free_slots:
            self.free_slots.append(slot_id)
            self.free_slots.sort()

    def live_slot_owner(self, slot_id: int) -> str | None:
        return self.slot_live_req.get(slot_id)

    def live_slot_tokens(self, slot_id: int) -> int | None:
        return self.slot_live_tokens.get(slot_id)

    def mark_slot_owner(self, slot_id: int, req_id: str, num_tokens: int | None = None) -> None:
        self.slot_live_req[slot_id] = req_id
        if num_tokens is None:
            self.slot_live_tokens.pop(slot_id, None)
        else:
            self.slot_live_tokens[slot_id] = max(0, int(num_tokens))

    def clear_slot_owner(self, slot_id: int) -> None:
        self.slot_live_req.pop(slot_id, None)
        self.slot_live_tokens.pop(slot_id, None)

    def live_slot_prefix_incomplete(self, slot_id: int, target_tokens: int) -> bool:
        """Report a live batch slot that does not hold the whole request prefix.

        Continuing from a live slot at ``cache_size=target_tokens`` is only
        correct when the accelerator cache really holds those tokens. Holding
        more is fine: the owner's token sequence is append-only, so the extra
        tail is simply overwritten (this is the preempt-and-resume case).
        Holding fewer means positions the request is about to build on were
        never computed in this slot, and decoding anyway is what produces
        fluent-but-wrong output. An untracked slot (None) keeps the previous
        trust-the-owner behaviour.
        """
        live_tokens = self.slot_live_tokens.get(slot_id)
        return live_tokens is not None and live_tokens < max(0, int(target_tokens))

    def live_request_prefix_incomplete(self, target_tokens: int) -> bool:
        """Single-cache counterpart of live_slot_prefix_incomplete."""
        live_tokens = self.loaded_req_tokens
        return live_tokens is not None and live_tokens < max(0, int(target_tokens))

    def clear_loaded_request(self, req_id: str | None = None) -> None:
        if req_id is None or self.loaded_req_id == req_id:
            self.loaded_req_id = None
            self.loaded_req_tokens = None

    def mark_loaded_request(self, req_id: str, num_tokens: int | None = None) -> None:
        self.loaded_req_id = req_id
        self.loaded_req_tokens = None if num_tokens is None else max(0, int(num_tokens))

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
        multimodal_cache_identity: Hashable | None = None,
        prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None,
    ) -> RuntimeCacheSnapshot:
        return self.store_snapshot(
            req_id=req_id,
            blobs=blobs,
            block_ids=block_ids,
            first_seq_blocks=first_seq_block_ids if first_seq_block_ids is not None else first_seq_blocks(block_ids),
            num_tokens=num_tokens,
            cache_token_ids=cache_token_ids,
            multimodal_cache_identity=multimodal_cache_identity,
            prompt_embed_cache_identity=prompt_embed_cache_identity,
        )

    def store_snapshot(
        self,
        *,
        req_id: str,
        blobs: list[Any],
        block_ids: KVBlockIds,
        first_seq_blocks: tuple[int, ...],
        first_seq_block_hashes: tuple[Hashable, ...] | None = None,
        num_tokens: int,
        cache_token_ids: tuple[int, ...] | list[int] | None = None,
        multimodal_cache_identity: Hashable | None = None,
        prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None,
    ) -> RuntimeCacheSnapshot:
        self.finished_snapshot_lru.pop(req_id, None)
        snapshot = RuntimeCacheSnapshot(
            blobs=blobs,
            block_ids=normalize_block_ids(block_ids),
            first_seq_blocks=tuple(first_seq_blocks),
            first_seq_block_hashes=tuple(first_seq_block_hashes) if first_seq_block_hashes is not None else None,
            num_tokens=max(0, int(num_tokens)),
            cache_token_ids=tuple(cache_token_ids) if cache_token_ids is not None else None,
            multimodal_cache_identity=multimodal_cache_identity,
            prompt_embed_cache_identity=prompt_embed_cache_identity,
        )
        self.snapshots[req_id] = snapshot
        if snapshot.first_seq_block_hashes is None:
            for block_id in snapshot.first_seq_blocks:
                self.physical_block_owner[block_id] = req_id
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
            for block_key in self._snapshot_index_blocks(snapshot):
                node = node.children.setdefault(block_key, RuntimeCacheSnapshotIndexNode())
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
        first_seq_block_hashes: tuple[Hashable, ...] | None = None,
        num_tokens: int,
        slot_id: int | None = None,
        cache_token_ids: tuple[int, ...] | list[int] | None = None,
        multimodal_cache_identity: Hashable | None = None,
        prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None,
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
            first_seq_block_hashes=first_seq_block_hashes,
            num_tokens=num_tokens,
            cache_token_ids=cache_token_ids,
            multimodal_cache_identity=multimodal_cache_identity,
            prompt_embed_cache_identity=prompt_embed_cache_identity,
        )

    def choose_prefix_snapshot(
        self,
        target_blocks: RuntimeCacheRequest | tuple[int, ...],
        target_tokens: int | None = None,
    ) -> RuntimeCacheSnapshotMatch:
        request = target_blocks if isinstance(target_blocks, RuntimeCacheRequest) else None
        if request is not None:
            target_blocks = self._request_index_blocks(request)
            target_tokens = request.num_computed_tokens
        else:
            target_blocks = tuple(("physical", block_id) for block_id in target_blocks)
        if target_tokens is None:
            raise TypeError("target_tokens is required when target_blocks is not a RuntimeCacheRequest")
        if target_tokens <= 0 or not target_blocks:
            return RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)

        best_snapshot: RuntimeCacheSnapshot | None = None
        best_req_id: str | None = None
        best_tokens = 0
        stale_req_ids: set[str] = set()
        node = self.snapshot_index_root
        for depth, block_id in enumerate(target_blocks, start=1):
            node = node.children.get(block_id)
            if node is None or not node.req_ids:
                break

            for req_id in tuple(node.req_ids):
                snapshot = self.snapshots.get(req_id)
                if snapshot is None:
                    continue
                matched_tokens = min(snapshot.num_tokens, depth * self.block_size, target_tokens)
                has_physical_collision = (
                    request is not None
                    and snapshot.first_seq_block_hashes is None
                    and request.first_seq_block_hashes is None
                    and req_id != request.req_id
                    and physical_prefix_overlaps(request.first_seq_blocks, snapshot.first_seq_blocks)
                    and any(
                        self.physical_block_owner.get(block_id) == request.req_id
                        for block_id in snapshot.first_seq_blocks
                    )
                )
                if has_physical_collision and (
                    request.cache_token_ids is None or snapshot.cache_token_ids is None
                ):
                    stale_req_ids.add(req_id)
                    continue
                token_matched_tokens = token_compatible_tokens_for_prompt_embeds(
                    matched_tokens,
                    request.cache_token_ids if request is not None else None,
                    snapshot.cache_token_ids,
                    request.prompt_embed_cache_identity if request is not None else None,
                    snapshot.prompt_embed_cache_identity,
                )
                matched_tokens = multimodal_compatible_tokens(
                    token_matched_tokens,
                    request.multimodal_cache_identity if request is not None else None,
                    snapshot.multimodal_cache_identity,
                )
                if (
                    request is not None
                    and matched_tokens <= 0
                    and token_matched_tokens <= 0
                    and has_physical_collision
                ):
                    stale_req_ids.add(req_id)
                if (
                    matched_tokens <= 0
                    and has_physical_collision
                    and snapshot.multimodal_cache_identity is not None
                    and snapshot.multimodal_cache_identity != request.multimodal_cache_identity
                ):
                    stale_req_ids.add(req_id)
                if matched_tokens > best_tokens:
                    best_tokens = matched_tokens
                    best_snapshot = snapshot
                    best_req_id = req_id

        for stale_req_id in sorted(stale_req_ids):
            self.remove_snapshot(stale_req_id)

        return RuntimeCacheSnapshotMatch(snapshot=best_snapshot, matched_tokens=best_tokens, req_id=best_req_id)

    def choose_snapshot(self, request: RuntimeCacheRequest) -> RuntimeCacheSnapshotMatch:
        target_tokens = request.num_computed_tokens
        if target_tokens <= 0:
            return RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)

        own_snapshot = self.snapshots.get(request.req_id)
        if own_snapshot is not None:
            matched_tokens = self.compatible_tokens(
                target_blocks=self._request_index_blocks(request),
                target_tokens=target_tokens,
                snapshot_blocks=self._snapshot_index_blocks(own_snapshot),
                snapshot_tokens=own_snapshot.num_tokens,
                target_token_ids=request.cache_token_ids,
                snapshot_token_ids=own_snapshot.cache_token_ids,
                target_multimodal_identity=request.multimodal_cache_identity,
                snapshot_multimodal_identity=own_snapshot.multimodal_cache_identity,
                target_prompt_embed_identity=request.prompt_embed_cache_identity,
                snapshot_prompt_embed_identity=own_snapshot.prompt_embed_cache_identity,
            )
            if matched_tokens >= target_tokens:
                return RuntimeCacheSnapshotMatch(
                    snapshot=own_snapshot,
                    matched_tokens=target_tokens,
                    req_id=request.req_id,
                    is_own_snapshot=True,
                )

        return self.choose_prefix_snapshot(request)

    def verify_prompt_embed_match(
        self,
        request: RuntimeCacheRequest,
        match: RuntimeCacheSnapshotMatch,
    ) -> RuntimeCacheSnapshotMatch:
        if match.snapshot is None or match.matched_tokens <= 0:
            return match

        matched_tokens = prompt_embed_compatible_tokens(
            match.matched_tokens,
            request.prompt_embed_cache_identity,
            match.snapshot.prompt_embed_cache_identity,
            request.cache_token_ids,
            match.snapshot.cache_token_ids,
        )
        if matched_tokens <= 0:
            return RuntimeCacheSnapshotMatch(snapshot=None, matched_tokens=0)
        if matched_tokens == match.matched_tokens:
            return match
        return RuntimeCacheSnapshotMatch(
            snapshot=match.snapshot,
            matched_tokens=matched_tokens,
            req_id=match.req_id,
            is_own_snapshot=match.is_own_snapshot,
        )

    def compatible_tokens(
        self,
        *,
        target_blocks: tuple[Hashable, ...],
        target_tokens: int,
        snapshot_blocks: tuple[Hashable, ...],
        snapshot_tokens: int,
        target_token_ids: tuple[int, ...] | None = None,
        snapshot_token_ids: tuple[int, ...] | None = None,
        target_multimodal_identity: Hashable | None = None,
        snapshot_multimodal_identity: Hashable | None = None,
        target_prompt_embed_identity: PromptEmbedCacheIdentity | None = None,
        snapshot_prompt_embed_identity: PromptEmbedCacheIdentity | None = None,
    ) -> int:
        matched_tokens = prefix_compatible_tokens(
            target_blocks,
            target_tokens,
            snapshot_blocks,
            snapshot_tokens,
            self.block_size,
        )
        matched_tokens = token_compatible_tokens_for_prompt_embeds(
            matched_tokens,
            target_token_ids,
            snapshot_token_ids,
            target_prompt_embed_identity,
            snapshot_prompt_embed_identity,
        )
        return multimodal_compatible_tokens(matched_tokens, target_multimodal_identity, snapshot_multimodal_identity)

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
            first_seq_block_hashes=request.first_seq_block_hashes,
            num_tokens=request.num_computed_tokens,
            cache_token_ids=request.cache_token_ids,
            multimodal_cache_identity=request.multimodal_cache_identity,
            prompt_embed_cache_identity=request.prompt_embed_cache_identity,
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

        self.observe_request_blocks(
            request.req_id,
            request.block_ids,
            first_seq_block_hashes=request.first_seq_block_hashes,
        )

        live_prefix_incomplete = False
        live_cache_tokens: int | None = None
        if self.loaded_req_id == request.req_id:
            if not self.live_request_prefix_incomplete(target_tokens):
                return RuntimeCacheLoadResult(
                    matched_tokens=target_tokens,
                    reused_live_cache=True,
                    action="reuse-live",
                    snapshot_req_id=request.req_id,
                    live_cache_tokens=self.loaded_req_tokens,
                )
            # The live cache no longer holds this request's prefix. Drop the
            # ownership claim so the prefix is rebuilt instead of decoded
            # against stale KV.
            live_prefix_incomplete = True
            live_cache_tokens = self.loaded_req_tokens
            self.clear_loaded_request(request.req_id)

        match = self.choose_snapshot(request)
        match = self.verify_prompt_embed_match(request, match)
        if match.snapshot is None or match.matched_tokens <= 0:
            self.clear_loaded_request()
            return RuntimeCacheLoadResult(
                matched_tokens=0,
                cache_miss=True,
                action="live-prefix-incomplete" if live_prefix_incomplete else "cache-miss",
                live_prefix_incomplete=live_prefix_incomplete,
                live_cache_tokens=live_cache_tokens,
            )

        load_runtime_cache = load_runtime_cache or self._load_runtime_cache_fn
        if load_runtime_cache is None:
            raise RuntimeError("Runtime cache load adapter is not configured.")
        if load_runtime_cache(match.snapshot.blobs, None):
            self.mark_loaded_request(request.req_id, match.matched_tokens)
            return RuntimeCacheLoadResult(
                matched_tokens=match.matched_tokens,
                loaded=True,
                loaded_snapshot_req_id=match.req_id,
                is_own_snapshot=match.is_own_snapshot,
                action="load-own" if match.is_own_snapshot else "load-shared",
                snapshot_req_id=match.req_id,
                live_prefix_incomplete=live_prefix_incomplete,
                live_cache_tokens=live_cache_tokens,
            )

        self.clear_loaded_request()
        return RuntimeCacheLoadResult(
            matched_tokens=0,
            cache_miss=True,
            live_prefix_incomplete=live_prefix_incomplete,
            live_cache_tokens=live_cache_tokens,
        )

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

        self.observe_request_blocks(
            request.req_id,
            request.block_ids,
            first_seq_block_hashes=request.first_seq_block_hashes,
        )

        live_prefix_incomplete = False
        live_cache_tokens: int | None = None
        if self.live_slot_owner(slot_id) == request.req_id:
            if not self.live_slot_prefix_incomplete(slot_id, target_tokens):
                return RuntimeCacheLoadResult(
                    matched_tokens=target_tokens,
                    reused_live_cache=True,
                    action="reuse-live",
                    snapshot_req_id=request.req_id,
                    live_cache_tokens=self.live_slot_tokens(slot_id),
                )
            # The slot no longer holds this request's prefix. Drop the
            # ownership claim so the prefix is rebuilt instead of decoded
            # against stale KV.
            live_prefix_incomplete = True
            live_cache_tokens = self.live_slot_tokens(slot_id)
            self.clear_slot_owner(slot_id)

        match = self.choose_snapshot(request)
        match = self.verify_prompt_embed_match(request, match)
        if match.snapshot is not None and match.matched_tokens > 0:
            load_runtime_cache = load_runtime_cache or self._load_runtime_cache_fn
            if load_runtime_cache is None:
                raise RuntimeError("Runtime cache load adapter is not configured.")
            if load_runtime_cache(match.snapshot.blobs, slot_id):
                self.mark_slot_owner(slot_id, request.req_id, match.matched_tokens)
                return RuntimeCacheLoadResult(
                    matched_tokens=match.matched_tokens,
                    loaded=True,
                    loaded_snapshot_req_id=match.req_id,
                    is_own_snapshot=match.is_own_snapshot,
                    action="load-own" if match.is_own_snapshot else "load-shared",
                    snapshot_req_id=match.req_id,
                    live_prefix_incomplete=live_prefix_incomplete,
                    live_cache_tokens=live_cache_tokens,
                )

        self.mark_slot_owner(slot_id, request.req_id, 0)
        return RuntimeCacheLoadResult(
            matched_tokens=0,
            cache_miss=True,
            action="live-prefix-incomplete" if live_prefix_incomplete else "cache-miss",
            live_prefix_incomplete=live_prefix_incomplete,
            live_cache_tokens=live_cache_tokens,
        )

    def load_for_slot(self, request: RuntimeCacheRequest, slot_id: int | None = None) -> RuntimeCacheLoadResult:
        if slot_id is not None and request.cache_slot_id != slot_id:
            request = RuntimeCacheRequest(
                req_id=request.req_id,
                block_ids=request.block_ids,
                first_seq_blocks=request.first_seq_blocks,
                num_computed_tokens=request.num_computed_tokens,
                first_seq_block_hashes=request.first_seq_block_hashes,
                cache_token_ids=request.cache_token_ids,
                multimodal_cache_identity=request.multimodal_cache_identity,
                cache_slot_id=slot_id,
                prompt_embed_cache_identity=request.prompt_embed_cache_identity,
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
        self.physical_block_owner.clear()
        self.loaded_req_id = None
        self.loaded_req_tokens = None
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


def token_compatible_tokens_for_prompt_embeds(
    matched_tokens: int,
    target_token_ids: tuple[int, ...] | None,
    snapshot_token_ids: tuple[int, ...] | None,
    target_identity: PromptEmbedCacheIdentity | None,
    snapshot_identity: PromptEmbedCacheIdentity | None,
) -> int:
    if target_identity is None and snapshot_identity is None:
        return token_compatible_tokens(matched_tokens, target_token_ids, snapshot_token_ids)
    if matched_tokens <= 0:
        return 0
    if target_identity is None or snapshot_identity is None:
        return 0

    target_prompt_len = max(0, int(target_identity.prompt_len))
    snapshot_prompt_len = max(0, int(snapshot_identity.prompt_len))
    embed_limit = min(target_prompt_len, snapshot_prompt_len)
    if embed_limit <= 0:
        return 0
    if matched_tokens <= embed_limit:
        return matched_tokens
    if target_prompt_len != snapshot_prompt_len:
        return embed_limit
    if target_token_ids is None or snapshot_token_ids is None:
        return embed_limit

    common_tokens = target_prompt_len
    for idx in range(target_prompt_len, min(matched_tokens, len(target_token_ids), len(snapshot_token_ids))):
        if target_token_ids[idx] != snapshot_token_ids[idx]:
            break
        common_tokens += 1
        if common_tokens >= matched_tokens:
            break
    return min(matched_tokens, common_tokens)


def prompt_embed_compatible_tokens(
    matched_tokens: int,
    target_identity: PromptEmbedCacheIdentity | None,
    snapshot_identity: PromptEmbedCacheIdentity | None,
    target_token_ids: tuple[int, ...] | None,
    snapshot_token_ids: tuple[int, ...] | None,
) -> int:
    if matched_tokens <= 0:
        return 0
    if target_identity is None and snapshot_identity is None:
        return token_compatible_tokens(matched_tokens, target_token_ids, snapshot_token_ids)

    # Explicit prompt embeddings replace token embeddings from position 0, so
    # any cross-request reuse touching that prefix must prove byte-identical
    # embedding content. If only one side has explicit embeddings, exact
    # equivalence to token-derived embeddings is not available here.
    if target_identity is None or snapshot_identity is None:
        return 0

    target_prompt_len = max(0, int(target_identity.prompt_len))
    snapshot_prompt_len = max(0, int(snapshot_identity.prompt_len))
    embed_tokens = min(matched_tokens, target_prompt_len, snapshot_prompt_len)
    if embed_tokens <= 0:
        return 0

    target_fingerprint = target_identity.fingerprint(embed_tokens)
    snapshot_fingerprint = snapshot_identity.fingerprint(embed_tokens)
    if target_fingerprint is None or snapshot_fingerprint is None:
        return 0
    if target_fingerprint != snapshot_fingerprint:
        return 0

    embed_limit = min(target_prompt_len, snapshot_prompt_len)
    if matched_tokens <= embed_limit:
        return matched_tokens

    if target_prompt_len != snapshot_prompt_len:
        return embed_limit

    if target_token_ids is None or snapshot_token_ids is None:
        return embed_limit

    common_tokens = target_prompt_len
    for idx in range(target_prompt_len, min(matched_tokens, len(target_token_ids), len(snapshot_token_ids))):
        if target_token_ids[idx] != snapshot_token_ids[idx]:
            break
        common_tokens += 1
        if common_tokens >= matched_tokens:
            break
    return min(matched_tokens, common_tokens)


def multimodal_compatible_tokens(
    matched_tokens: int,
    target_multimodal_identity: Hashable | None,
    snapshot_multimodal_identity: Hashable | None,
) -> int:
    if matched_tokens <= 0:
        return 0
    if target_multimodal_identity is None and snapshot_multimodal_identity is None:
        return matched_tokens

    if (
        target_multimodal_identity is not None
        and snapshot_multimodal_identity is not None
        and _is_resolved_multimodal_identity(target_multimodal_identity)
        and _is_resolved_multimodal_identity(snapshot_multimodal_identity)
        and target_multimodal_identity == snapshot_multimodal_identity
    ):
        return matched_tokens

    text_only_limit = _known_text_only_prefix_limit(
        target_multimodal_identity,
        snapshot_multimodal_identity,
    )
    if text_only_limit is None:
        return 0
    return min(matched_tokens, text_only_limit)


def _known_text_only_prefix_limit(
    target_multimodal_identity: Hashable | None,
    snapshot_multimodal_identity: Hashable | None,
) -> int | None:
    starts: list[int] = []
    for identity in (target_multimodal_identity, snapshot_multimodal_identity):
        spans = _multimodal_identity_spans(identity)
        if spans is None:
            if identity is not None:
                return None
            continue
        starts.extend(start for start, _end in spans)

    if not starts:
        return None
    return max(0, min(starts))


def _is_resolved_multimodal_identity(identity: Hashable) -> bool:
    if not isinstance(identity, tuple) or not identity:
        return True
    if identity[0] != "vlm":
        return True

    entries = _vlm_identity_entries(identity)
    if entries is None:
        return False
    return all(_vlm_entry_content_fingerprint(entry) is not None for entry in entries)


def _multimodal_identity_spans(identity: Hashable | None) -> tuple[tuple[int, int], ...] | None:
    if identity is None:
        return None
    if not isinstance(identity, tuple) or not identity:
        return None
    if identity[0] != "vlm":
        return None

    entries = _vlm_identity_entries(identity)
    if entries is None:
        return None

    spans: list[tuple[int, int]] = []
    for entry in entries:
        position = _vlm_entry_position_signature(entry)
        if position is None:
            return None
        offset, length, _embed_signature = position
        spans.append((offset, offset + length))
    return tuple(spans)


def _vlm_identity_entries(identity: tuple[object, ...]) -> tuple[object, ...] | None:
    if len(identity) < 3:
        return None

    payload = identity[2]
    if _is_position_signature(payload):
        # Legacy identity shape: ("vlm", session_id, position_signature). It
        # proves the text-only prefix boundary but not multimodal content.
        return (("legacy", payload, None),)

    if isinstance(payload, tuple) and all(_is_vlm_identity_entry(entry) for entry in payload):
        return payload

    return None


def _is_vlm_identity_entry(entry: object) -> bool:
    return isinstance(entry, tuple) and len(entry) >= 3 and _is_position_signature(entry[1])


def _vlm_entry_position_signature(entry: object) -> tuple[int, int, object] | None:
    if not isinstance(entry, tuple):
        return None
    if _is_position_signature(entry):
        position = entry
    elif len(entry) >= 2 and _is_position_signature(entry[1]):
        position = entry[1]
    else:
        return None
    return (int(position[0]), int(position[1]), position[2])


def _vlm_entry_content_fingerprint(entry: object) -> object | None:
    if not isinstance(entry, tuple):
        return None
    if len(entry) >= 3:
        return entry[2]
    return None


def _is_position_signature(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    offset, length, _embed_signature = value
    return isinstance(offset, int) and isinstance(length, int)


def physical_prefix_overlaps(
    target_blocks: tuple[int, ...],
    snapshot_blocks: tuple[int, ...],
) -> bool:
    return any(target_block == snapshot_block for target_block, snapshot_block in zip(target_blocks, snapshot_blocks))
