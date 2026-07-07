# KV Cache Runtime Refactor Plan

## Background

`vllm-mblt` currently keeps most Mobilint/NPU runtime cache behavior inside
`vllm_mblt/mblt_worker.py`. This includes request cache state, accelerator cache
slot ownership, runtime cache dump/load, finished-request snapshots, prefix
snapshot lookup, and cache reuse policy.

This makes `MbltWorker` responsible for too many layers at once:

- vLLM worker interface implementation
- scheduler output interpretation
- prompt embedding construction
- Mobilint runtime inference
- sampling/logprob handling
- NPU runtime cache snapshot and slot lifecycle

The goal of this refactor is to split the Mobilint runtime cache layer out of
`MbltWorker` while preserving existing runtime behavior.

## Design Direction

Do not subclass or replace upstream vLLM's scheduler-side `KVCacheManager`.

In upstream vLLM, `KVCacheManager` owns logical KV block allocation, prefix cache
lookup, block pools, and scheduler-side request/block lifecycle. In `vllm-mblt`,
the custom cache code is worker/runtime-side: it consumes scheduler-produced
block IDs and maps them to Mobilint runtime cache blobs, cache slots, and
snapshot reuse decisions.

The refactor should therefore introduce a Mobilint runtime cache abstraction
instead of another scheduler-side KV cache manager.

## Proposed Module and Names

Add a new module:

```text
vllm_mblt/runtime_cache.py
```

Primary class:

```python
class MbltRuntimeCacheManager:
    ...
```

Rationale:

- Avoids confusion with upstream `vllm.v1.core.kv_cache_manager.KVCacheManager`.
- Communicates that this manager handles Mobilint runtime cache state, not
  scheduler-side logical KV block allocation.
- Keeps room for future runtime cache state beyond strict KV cache snapshots.

Suggested supporting names:

```python
KVBlockIds = tuple[list[int], ...]

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
```

## Boundary With Upstream vLLM

The boundary between upstream scheduler-side KV management and Mobilint
runtime-side cache management is the block ID layout emitted by upstream
`KVCacheBlocks.get_block_ids()` and delivered through `SchedulerOutput`:

```python
tuple[list[int], ...]
```

`MbltRuntimeCacheManager` should treat this shape as its input contract. It
should not own vLLM `Request` objects, upstream block pools, or upstream prefix
hashing state.

## Responsibilities of `MbltRuntimeCacheManager`

The new manager should own:

1. Block ID utilities
   - normalize block ID layout
   - append newly allocated block IDs
   - extract first-sequence block IDs
   - calculate required block counts

2. Runtime cache snapshot repository
   - request ID to snapshot mapping
   - snapshot prefix index
   - best-prefix lookup
   - finished snapshot LRU cap

3. Live runtime cache ownership
   - single-cache live owner tracking
   - batch cache slot assignment
   - cache slot release and reuse

4. Runtime dump/load policy
   - decide when a snapshot should be dumped
   - load own snapshot when fully compatible
   - load shared prefix snapshot when useful
   - reuse live accelerator cache for the same request
   - handle cache miss fallback

5. Finished-request cache handling
   - dump final snapshots when appropriate
   - keep finished snapshots for cross-request prefix reuse
   - evict old finished snapshots with LRU policy

## Responsibilities That Should Stay in `MbltWorker`

`MbltWorker` should continue to own:

- vLLM `WorkerBase` interface methods
- model/runtime object lifecycle
- prompt embedding and deepstack embedding construction
- Mobilint model inference calls
- sampling metadata and logits processing
- request execution orchestration based on `SchedulerOutput`
- actual qbruntime/cache-model calls through small adapter methods

The worker should provide runtime cache IO callables to the manager:

```python
def _dump_runtime_cache(self, slot_id: int | None = None) -> list[Any] | None:
    ...

def _load_runtime_cache(self, blobs: list[Any], slot_id: int | None = None) -> bool:
    ...
```

`MbltRuntimeCacheManager` should call these injected functions rather than
importing or owning qbruntime/cache-model objects directly.

## Backward Compatibility Policy

Backward compatibility for internal APIs is intentionally not required for this
refactor.

The package is still pre-1.0, and internal APIs may continue to break before
1.0.0. After 1.0.0, compatibility should be treated more conservatively.

This means the refactor does not need to preserve imports or direct attributes
such as:

- `vllm_mblt.mblt_worker.CacheSnapshot`
- `vllm_mblt.mblt_worker.SnapshotIndexNode`
- `MbltWorker.cache_snapshots`
- `MbltWorker.snapshot_index_root`
- `MbltWorker.loaded_cache_req_id`
- `MbltWorker.req_to_cache_slot`
- `MbltWorker.cache_slot_to_req`
- `MbltWorker.free_cache_slots`

Breaking changes should be recorded in `CHANGELOG.md`.

## Proposed Implementation Steps

1. Add `vllm_mblt/runtime_cache.py`.
2. Move runtime cache dataclasses and block-ID helpers from `MbltWorker` into
   the new module.
3. Implement `MbltRuntimeCacheManager` with snapshot repository, prefix index,
   live-cache ownership, and batch slot management.
4. Update `MbltWorker` to instantiate `self.runtime_cache` and delegate cache
   decisions to it.
5. Remove cache-specific mutable state and helper methods from `MbltWorker`.
6. Add `CHANGELOG.md` with breaking changes only.
7. Replace string-based cache tests with behavior-oriented tests for
   `MbltRuntimeCacheManager`.
8. Update worker tests to use the new module and stop depending on old worker
   internals.

## Test Plan

Add focused unit tests for `vllm_mblt.runtime_cache` covering:

- block ID normalization and append behavior
- block layout mismatch errors
- first-sequence block extraction
- prefix-compatible token calculation
- snapshot index lookup behavior
- own snapshot priority
- shared prefix snapshot reuse
- live cache reuse for the same request
- cache miss fallback
- finished snapshot LRU eviction
- batch slot allocation/release
- slot-scoped dump/load behavior

Then run, where possible:

```bash
uv run pytest tests
uv run ruff check .
```

If full dependency installation is unavailable, run the dependency-light runtime
cache tests separately and document any skipped validation.

## Changelog Plan

Create `CHANGELOG.md` and record only breaking changes for this refactor.

Expected entries:

- Moved Mobilint runtime cache snapshot, live-cache ownership, and accelerator
  cache slot handling out of `MbltWorker` into
  `vllm_mblt.runtime_cache.MbltRuntimeCacheManager`.
- Removed compatibility guarantees for old cache-related `MbltWorker` internals.
- Removed direct imports of cache implementation details from
  `vllm_mblt.mblt_worker`.
- Documented that pre-1.0 releases may introduce breaking internal API changes;
  backward compatibility is planned after 1.0.0.