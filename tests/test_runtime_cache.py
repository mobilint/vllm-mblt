import pytest

from vllm_mblt.runtime_cache import (
    MbltRuntimeCacheManager,
    RuntimeCacheRequest,
    append_block_ids,
    first_seq_blocks,
    normalize_block_ids,
    prefix_compatible_tokens,
    required_blocks,
)


def _make_manager(*, block_size: int = 4, max_batch_size: int = 2, max_finished_snapshots: int = 16):
    return MbltRuntimeCacheManager(
        max_batch_size=max_batch_size,
        block_size=block_size,
        max_finished_snapshots=max_finished_snapshots,
    )


def _request(
    req_id: str,
    blocks: tuple[int, ...],
    tokens: int,
    *,
    block_hashes: tuple[object, ...] | None = None,
    cache_slot_id: int | None = None,
    cache_token_ids: tuple[int, ...] | list[int] | None = None,
) -> RuntimeCacheRequest:
    return RuntimeCacheRequest(
        req_id=req_id,
        block_ids=(list(blocks),),
        first_seq_blocks=blocks,
        num_computed_tokens=tokens,
        first_seq_block_hashes=block_hashes,
        cache_slot_id=cache_slot_id,
        cache_token_ids=tuple(cache_token_ids) if cache_token_ids is not None else None,
    )


def _store_snapshot(
    manager: MbltRuntimeCacheManager,
    req_id: str,
    blocks: tuple[int, ...] | list[int],
    tokens: int,
    *,
    blobs: list[object] | None = None,
    cache_token_ids: tuple[int, ...] | list[int] | None = None,
    block_hashes: tuple[object, ...] | None = None,
) -> None:
    block_tuple = tuple(blocks)
    block_ids = (list(block_tuple),)
    manager.store_snapshot(
        req_id=req_id,
        blobs=blobs if blobs is not None else [f"blob:{req_id}"],
        block_ids=block_ids,
        first_seq_blocks=first_seq_blocks(block_ids),
        first_seq_block_hashes=block_hashes,
        num_tokens=tokens,
        cache_token_ids=cache_token_ids,
    )


class TestRuntimeCacheBlockHelpers:
    def test_normalize_and_append_block_ids_copy_inputs(self) -> None:
        current = ([1, 2], [10])
        new = ([3], [11, 12])

        normalized = normalize_block_ids(current)
        appended = append_block_ids(normalized, new)

        current[0].append(99)
        new[0].append(99)
        assert normalized == ([1, 2], [10])
        assert normalized is not current
        assert normalized[0] is not current[0]
        assert appended == ([1, 2, 3], [10, 11, 12])
        assert first_seq_blocks(appended) == (1, 2, 3)

    def test_append_initializes_empty_block_ids_from_new_layout(self) -> None:
        new_blocks = ([10], [20, 21])

        appended = append_block_ids((), new_blocks)

        new_blocks[0].append(99)
        assert appended == ([10], [20, 21])

    def test_append_block_ids_rejects_sequence_layout_mismatch(self) -> None:
        with pytest.raises(RuntimeError, match="KV block_ids layout mismatch"):
            append_block_ids(([1],), ([2], [3]))

    def test_required_blocks_and_prefix_compatible_tokens(self) -> None:
        assert required_blocks(0, 4) == 0
        assert required_blocks(1, 4) == 1
        assert required_blocks(5, 4) == 2

        assert prefix_compatible_tokens((1, 2, 3), 10, (1, 2, 9), 12, 4) == 8
        assert prefix_compatible_tokens((1, 2), 6, (1, 2, 3), 12, 4) == 6
        assert prefix_compatible_tokens((1, 2), 6, (9, 2), 8, 4) == 0

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
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "short", (1,), 4)
        _store_snapshot(manager, "long", (1, 2, 3), 12)

        match = manager.choose_prefix_snapshot(_request("req", (1, 2, 9), 10))

        assert match.req_id == "long"
        assert match.snapshot is manager.get_snapshot("long")
        assert match.matched_tokens == 8
        assert not match.is_own_snapshot

    def test_own_snapshot_has_priority_when_fully_compatible(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "other", (1, 2, 3), 12)
        _store_snapshot(manager, "req", (1, 2), 8)

        snapshot, matched_tokens = manager.choose_snapshot(_request("req", (1, 2), 8))

        assert snapshot is manager.get_snapshot("req")
        assert matched_tokens == 8

    def test_shared_prefix_snapshot_reuse_when_own_snapshot_is_partial(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "req", (1,), 4)
        _store_snapshot(manager, "other", (1, 2, 3), 12)

        snapshot, matched_tokens = manager.choose_snapshot(_request("req", (1, 2, 3), 10))

        assert snapshot is manager.get_snapshot("other")
        assert matched_tokens == 10

    def test_cache_miss_fallback_returns_no_snapshot_or_tokens(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "other", (1, 2), 8)

        snapshot, matched_tokens = manager.choose_snapshot(_request("req", (9, 2), 8))

        assert snapshot is None
        assert matched_tokens == 0

    def test_prefix_snapshot_rejects_reused_physical_blocks_with_different_tokens(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "old-content",
            (10, 11),
            8,
            cache_token_ids=(101, 102, 103, 104, 105, 106, 107, 108),
        )

        match = manager.choose_snapshot(
            _request(
                "new-content",
                (10, 11),
                8,
                cache_token_ids=(201, 202, 203, 204, 205, 206, 207, 208),
            )
        )

        assert match.snapshot is None
        assert match.matched_tokens == 0

    def test_hashed_snapshot_does_not_match_reused_physical_block_ids(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "old", (1, 2), 8, block_hashes=("old-a", "old-b"))

        match = manager.choose_snapshot(_request("new", (1, 2), 8, block_hashes=("new-a", "new-b")))

        assert match.snapshot is None
        assert match.matched_tokens == 0

    def test_prefix_snapshot_selection_skips_incompatible_reused_block_candidate(self) -> None:
        manager = _make_manager(block_size=4)
        shared_blocks = (10, 11)
        _store_snapshot(
            manager,
            "wrong-content",
            shared_blocks,
            8,
            cache_token_ids=(101, 102, 103, 104, 105, 106, 107, 108),
        )
        _store_snapshot(
            manager,
            "right-content",
            shared_blocks,
            8,
            cache_token_ids=(201, 202, 203, 204, 205, 206, 207, 208),
        )

        match = manager.choose_snapshot(
            _request(
                "new-content",
                shared_blocks,
                8,
                cache_token_ids=(201, 202, 203, 204, 205, 206, 207, 208),
            )
        )

        assert match.req_id == "right-content"
        assert match.snapshot is manager.get_snapshot("right-content")
        assert match.matched_tokens == 8

    def test_hashless_snapshot_invalidates_when_physical_block_owner_changes(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "old", (1, 2), 8)

        evicted = manager.observe_request_blocks("new", ([1, 3],))

        assert evicted == ["old"]
        assert manager.get_snapshot("old") is None
        assert manager.choose_snapshot(_request("new", (1, 2), 8)).snapshot is None

    def test_hashless_load_invalidates_stale_snapshot_before_restore(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "old", (1, 2), 8)
        load_calls = []

        result = manager.load_snapshot_for_request(
            _request("new", (1, 2), 8),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert result.cache_miss
        assert result.matched_tokens == 0
        assert load_calls == []
        assert manager.get_snapshot("old") is None

    def test_snapshot_update_and_removal_refresh_prefix_index(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "req", (1, 2), 8)

        assert manager.choose_prefix_snapshot(_request("new", (1, 2), 8)).req_id == "req"
        _store_snapshot(manager, "req", (5, 6), 8)

        assert manager.choose_prefix_snapshot(_request("new", (1, 2), 8)).snapshot is None
        assert manager.choose_prefix_snapshot(_request("new", (5, 6), 8)).req_id == "req"
        assert manager.remove_snapshot("req") is not None
        assert manager.choose_prefix_snapshot(_request("new", (5, 6), 8)).snapshot is None

    def test_live_cache_reuse_policy_and_cache_miss_state(self) -> None:
        manager = _make_manager(block_size=4)

        assert manager.loaded_req_id is None
        manager.mark_loaded_request("req")
        assert manager.loaded_req_id == "req"
        manager.clear_loaded_request("other")
        assert manager.loaded_req_id == "req"
        manager.clear_loaded_request("req")
        assert manager.loaded_req_id is None

    def test_finished_snapshot_lru_eviction_removes_snapshot_index_and_live_owner(self) -> None:
        manager = _make_manager(block_size=4, max_finished_snapshots=2)
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
        assert manager.choose_prefix_snapshot(_request("new", (2,), 4)).req_id == "b"

    def test_finished_snapshot_lru_touch_preserves_recent_snapshot(self) -> None:
        manager = _make_manager(block_size=4, max_finished_snapshots=2)
        _store_snapshot(manager, "a", (1,), 4)
        _store_snapshot(manager, "b", (2,), 4)
        _store_snapshot(manager, "c", (3,), 4)

        manager.mark_snapshot_finished("a")
        manager.mark_snapshot_finished("b")
        manager.mark_snapshot_finished("a")
        evicted = manager.mark_snapshot_finished("c")

        assert evicted == ["b"]
        assert list(manager.finished_snapshot_lru) == ["a", "c"]
        assert manager.get_snapshot("a") is not None
        assert manager.get_snapshot("b") is None


class TestMbltRuntimeCacheManagerRuntimeDecisions:
    def test_single_cache_reuses_live_owner_without_loading(self) -> None:
        manager = _make_manager(block_size=4)
        manager.mark_loaded_request("req")
        load_calls = []

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2), 8),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert result.matched_tokens == 8
        assert result.reused_live_cache
        assert not result.loaded
        assert load_calls == []
        assert manager.loaded_req_id == "req"

    def test_single_cache_loads_fully_compatible_own_snapshot(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "req", (1, 2), 8)
        load_calls = []

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2), 8),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert result.matched_tokens == 8
        assert result.loaded
        assert result.is_own_snapshot
        assert result.loaded_snapshot_req_id == "req"
        assert load_calls == [(["blob:req"], None)]
        assert manager.loaded_req_id == "req"

    def test_single_cache_loads_useful_shared_prefix_snapshot(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "shared", (1, 2, 3), 12, block_hashes=("a", "b", "c"))
        load_calls = []

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2, 9), 10, block_hashes=("a", "b", "z")),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert result.matched_tokens == 8
        assert result.loaded
        assert not result.is_own_snapshot
        assert result.loaded_snapshot_req_id == "shared"
        assert load_calls == [(["blob:shared"], None)]
        assert manager.loaded_req_id == "req"

    def test_single_cache_miss_clears_live_owner(self) -> None:
        manager = _make_manager(block_size=4)
        manager.mark_loaded_request("other")

        result = manager.load_snapshot_for_request(_request("req", (9,), 4), lambda blobs, slot_id: True)

        assert result.matched_tokens == 0
        assert result.cache_miss
        assert manager.loaded_req_id is None

    def test_single_cache_does_not_load_snapshot_for_reused_blocks_with_different_tokens(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "old-content",
            (10, 11),
            8,
            blobs=["stale-runtime-cache"],
            cache_token_ids=(101, 102, 103, 104, 105, 106, 107, 108),
        )
        load_calls = []

        result = manager.load_snapshot_for_request(
            _request(
                "new-content",
                (10, 11),
                8,
                cache_token_ids=(201, 202, 203, 204, 205, 206, 207, 208),
            ),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert result.cache_miss
        assert result.matched_tokens == 0
        assert load_calls == []

    def test_dump_snapshot_if_needed_uses_injected_callable_and_block_boundary_policy(self) -> None:
        manager = _make_manager(block_size=4)
        dump_calls = []

        dumped = manager.dump_snapshot_if_needed(
            _request("req", (1,), 4),
            lambda slot_id: dump_calls.append(slot_id) or ["blob:req:4"],
        )

        assert dumped
        assert dump_calls == [None]
        assert manager.get_snapshot("req").blobs == ["blob:req:4"]

        block_growth_dumped = manager.dump_snapshot_if_needed(
            _request("req", (1, 2), 5),
            lambda slot_id: dump_calls.append(slot_id) or ["blob:req:5"],
        )

        assert block_growth_dumped
        assert dump_calls == [None, None]
        assert manager.get_snapshot("req").blobs == ["blob:req:5"]

        not_dumped = manager.dump_snapshot_if_needed(
            _request("req", (1, 2), 6),
            lambda slot_id: dump_calls.append(slot_id) or ["blob:req:6"],
        )

        assert not not_dumped
        assert dump_calls == [None, None]
        assert manager.get_snapshot("req").blobs == ["blob:req:5"]

    def test_dump_snapshot_if_needed_saves_token_identity_for_reused_block_detection(self) -> None:
        manager = _make_manager(block_size=4)

        dumped = manager.dump_snapshot_if_needed(
            _request("req", (10, 11), 8, cache_token_ids=(101, 102, 103, 104, 105, 106, 107, 108)),
            lambda slot_id: ["blob:req"],
        )

        assert dumped
        assert manager.get_snapshot("req").cache_token_ids == (101, 102, 103, 104, 105, 106, 107, 108)

    def test_dump_live_request_before_switch_dumps_current_owner_when_needed(self) -> None:
        manager = _make_manager(block_size=4)
        manager.mark_loaded_request("old")

        dumped = manager.dump_live_request_before_switch(
            next_req_id="new",
            live_request=_request("old", (1, 2), 8),
            dump_runtime_cache=lambda slot_id: ["blob:old"],
        )

        assert dumped
        assert manager.get_snapshot("old").num_tokens == 8


class TestMbltRuntimeCacheManagerSlots:
    def test_batch_slot_allocation_release_and_reuse(self) -> None:
        manager = _make_manager(max_batch_size=2)

        assert manager.assign_slot("a") == 0
        assert manager.assign_slot("b") == 1
        assert manager.assign_slot("a") == 0
        assert manager.req_to_cache_slot == {"a": 0, "b": 1}
        assert manager.cache_slot_to_req == {0: "a", 1: "b"}
        assert manager.free_cache_slots == []
        with pytest.raises(RuntimeError, match="No free accelerator cache slots"):
            manager.assign_slot("c")

        manager.release_slot("a")

        assert manager.live_slot_owner(0) is None
        assert manager.req_to_cache_slot == {"b": 1}
        assert manager.cache_slot_to_req == {1: "b"}
        assert manager.free_cache_slots == [0]
        assert manager.assign_slot("c") == 0
        assert manager.get_slot("c") == 0

    def test_slot_scoped_dump_and_load_pass_cache_id_to_injected_callables(self) -> None:
        manager = _make_manager(max_batch_size=2)
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

    def test_slot_load_reuses_live_owner_and_uses_slot_scoped_loads(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "req")
        load_calls = []

        reused = manager.load_snapshot_for_slot(
            _request("req", (1,), 4, cache_slot_id=slot_id),
            lambda blobs, load_slot_id: load_calls.append((blobs, load_slot_id)) or True,
        )

        assert reused.reused_live_cache
        assert reused.matched_tokens == 4
        assert load_calls == []

        _store_snapshot(manager, "other", (7, 8), 8, block_hashes=("a", "b"))
        loaded = manager.load_snapshot_for_slot(
            _request("new", (7, 9), 8, block_hashes=("a", "z"), cache_slot_id=slot_id),
            lambda blobs, load_slot_id: load_calls.append((blobs, load_slot_id)) or True,
        )

        assert loaded.loaded
        assert loaded.matched_tokens == 4
        assert loaded.loaded_snapshot_req_id == "other"
        assert load_calls == [(["blob:other"], slot_id)]
        assert manager.live_slot_owner(slot_id) == "new"

    def test_slot_cache_miss_marks_slot_owner_for_rebuild(self) -> None:
        manager = _make_manager(max_batch_size=1, block_size=4)
        slot_id = manager.assign_slot("req")

        result = manager.load_snapshot_for_slot(
            _request("req", (99,), 4, cache_slot_id=slot_id),
            lambda blobs, load_slot_id: True,
        )

        assert result.cache_miss
        assert result.matched_tokens == 0
        assert manager.live_slot_owner(slot_id) == "req"

    def test_slot_scoped_dump_passes_cache_slot_to_injected_callable(self) -> None:
        manager = _make_manager(max_batch_size=1, block_size=4)
        slot_id = manager.assign_slot("req")
        dump_calls = []

        dumped = manager.dump_snapshot_if_needed(
            _request("req", (1,), 4, cache_slot_id=slot_id),
            lambda dump_slot_id: dump_calls.append(dump_slot_id) or ["slot-blob"],
        )

        assert dumped
        assert dump_calls == [slot_id]
        assert manager.get_snapshot("req").blobs == ["slot-blob"]
