import pytest

from vllm_mblt.runtime_cache import (
    MbltRuntimeCacheManager,
    RuntimeCacheRequest,
    append_block_ids,
    first_seq_blocks,
    normalize_block_ids,
    prefix_compatible_tokens,
)


def _request(req_id: str, blocks: tuple[int, ...], tokens: int) -> RuntimeCacheRequest:
    return RuntimeCacheRequest(
        req_id=req_id,
        block_ids=(list(blocks),),
        first_seq_blocks=blocks,
        num_computed_tokens=tokens,
    )


def _store_snapshot(
    manager: MbltRuntimeCacheManager,
    req_id: str,
    blocks: tuple[int, ...],
    tokens: int,
    blobs: list[object] | None = None,
) -> None:
    manager.store_snapshot(
        req_id=req_id,
        blobs=blobs if blobs is not None else [f"blob:{req_id}"],
        block_ids=(list(blocks),),
        first_seq_blocks=blocks,
        num_tokens=tokens,
    )


class TestRuntimeCacheBlockHelpers:
    def test_normalize_block_ids_copies_sequences_and_append_preserves_layout(self) -> None:
        source = ([1, 2], [3])

        normalized = normalize_block_ids(source)
        appended = append_block_ids(normalized, ([4], [5, 6]))

        source[0].append(99)
        assert normalized == ([1, 2], [3])
        assert appended == ([1, 2, 4], [3, 5, 6])
        assert appended is not normalized

    def test_append_initializes_empty_block_ids_from_new_layout(self) -> None:
        new_blocks = ([10], [20, 21])

        appended = append_block_ids((), new_blocks)

        new_blocks[0].append(99)

        assert appended == ([10], [20, 21])

    def test_append_raises_for_block_layout_mismatch(self) -> None:
        with pytest.raises(RuntimeError, match="layout mismatch"):
            append_block_ids(([1], [2]), ([3],))

    def test_first_seq_blocks_extracts_first_sequence_or_empty_tuple(self) -> None:
        assert first_seq_blocks(([1, 2], [3, 4])) == (1, 2)
        assert first_seq_blocks(()) == ()

    @pytest.mark.parametrize(
        ("target_blocks", "target_tokens", "snapshot_blocks", "snapshot_tokens", "expected"),
        [
            ((1, 2), 0, (1, 2), 8, 0),
            ((1, 2), 8, (9, 2), 8, 0),
            ((1, 2, 3), 10, (1, 2, 9), 12, 8),
            ((1, 2, 3), 10, (1, 2, 3), 6, 6),
            ((1, 2, 3), 10, (1, 2, 3), 12, 10),
        ],
    )
    def test_prefix_compatible_tokens_caps_by_prefix_target_and_snapshot(
        self,
        target_blocks: tuple[int, ...],
        target_tokens: int,
        snapshot_blocks: tuple[int, ...],
        snapshot_tokens: int,
        expected: int,
    ) -> None:
        assert prefix_compatible_tokens(target_blocks, target_tokens, snapshot_blocks, snapshot_tokens, 4) == expected


class TestMbltRuntimeCacheManagerSnapshots:
    def test_snapshot_index_lookup_uses_deepest_compatible_prefix(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _store_snapshot(manager, "short", (1,), 4)
        _store_snapshot(manager, "long", (1, 2, 3), 12)

        match = manager.choose_prefix_snapshot(_request("req", (1, 2, 9), 10))

        assert match.req_id == "long"
        assert match.snapshot is manager.get_snapshot("long")
        assert match.matched_tokens == 8

    def test_own_snapshot_has_priority_when_fully_compatible(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _store_snapshot(manager, "other", (1, 2, 3), 12)
        _store_snapshot(manager, "req", (1, 2), 8)

        snapshot, matched_tokens = manager.choose_snapshot(_request("req", (1, 2), 8))

        assert snapshot is manager.get_snapshot("req")
        assert matched_tokens == 8

    def test_shared_prefix_snapshot_reuse_when_own_snapshot_is_partial(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _store_snapshot(manager, "req", (1,), 4)
        _store_snapshot(manager, "other", (1, 2, 3), 12)

        snapshot, matched_tokens = manager.choose_snapshot(_request("req", (1, 2, 3), 10))

        assert snapshot is manager.get_snapshot("other")
        assert matched_tokens == 10

    def test_cache_miss_fallback_returns_no_snapshot_or_tokens(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _store_snapshot(manager, "other", (1, 2), 8)

        snapshot, matched_tokens = manager.choose_snapshot(_request("req", (9, 2), 8))

        assert snapshot is None
        assert matched_tokens == 0

    def test_snapshot_update_and_removal_refresh_prefix_index(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _store_snapshot(manager, "req", (1, 2), 8)

        assert manager.choose_prefix_snapshot(_request("new", (1, 2), 8)).req_id == "req"
        _store_snapshot(manager, "req", (5, 6), 8)

        assert manager.choose_prefix_snapshot(_request("new", (1, 2), 8)).snapshot is None
        assert manager.choose_prefix_snapshot(_request("new", (5, 6), 8)).req_id == "req"
        assert manager.remove_snapshot("req") is not None
        assert manager.choose_prefix_snapshot(_request("new", (5, 6), 8)).snapshot is None

    def test_live_cache_reuse_for_same_request_is_tracked_without_snapshot_load(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)

        manager.mark_loaded_request("req")

        assert manager.loaded_req_id == "req"
        manager.clear_loaded_request("other")
        assert manager.loaded_req_id == "req"
        manager.clear_loaded_request("req")
        assert manager.loaded_req_id is None

    def test_finished_snapshot_lru_eviction_removes_snapshot_index_and_live_owner(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4, max_finished_snapshots=2)
        _store_snapshot(manager, "a", (1,), 4)
        _store_snapshot(manager, "b", (2,), 4)
        _store_snapshot(manager, "c", (3,), 4)

        assert manager.mark_snapshot_finished("a") == []
        assert manager.mark_snapshot_finished("b") == []
        manager.mark_loaded_request("a")
        evicted = manager.mark_snapshot_finished("c")

        assert evicted == ["a"]
        assert manager.get_snapshot("a") is None
        assert manager.loaded_req_id is None
        assert list(manager.finished_snapshot_lru) == ["b", "c"]
        assert manager.choose_prefix_snapshot(_request("new", (1,), 4)).snapshot is None


class TestMbltRuntimeCacheManagerSlots:
    def test_batch_slot_allocation_release_and_reuse(self) -> None:
        manager = MbltRuntimeCacheManager(max_batch_size=2, block_size=4)

        assert manager.assign_slot("a") == 0
        assert manager.assign_slot("b") == 1
        assert manager.assign_slot("a") == 0
        with pytest.raises(RuntimeError, match="No free accelerator cache slots"):
            manager.assign_slot("c")

        manager.release_slot("a")

        assert manager.live_slot_owner(0) is None
        assert manager.assign_slot("c") == 0

    def test_slot_scoped_dump_and_load_pass_cache_id_to_injected_callables(self) -> None:
        manager = MbltRuntimeCacheManager(max_batch_size=2, block_size=4)
        dump_calls: list[dict[str, int]] = []
        load_calls: list[tuple[list[object], dict[str, int]]] = []

        def dump_cache(**kwargs: int) -> list[object]:
            dump_calls.append(kwargs)
            return ["dumped", kwargs.get("cache_id")]

        def load_cache(blobs: list[object], **kwargs: int) -> None:
            load_calls.append((blobs, kwargs))

        assert manager.dump_runtime_cache(dump_cache) == ["dumped", None]
        assert manager.dump_runtime_cache(dump_cache, slot_id=1) == ["dumped", 1]
        assert manager.load_runtime_cache(load_cache, ["blob"])
        assert manager.load_runtime_cache(load_cache, ["slot-blob"], slot_id=1)

        assert dump_calls == [{}, {"cache_id": 1}]
        assert load_calls == [(["blob"], {}), (["slot-blob"], {"cache_id": 1})]