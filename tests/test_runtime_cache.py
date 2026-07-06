from vllm_mblt.runtime_cache import MbltRuntimeCacheManager, RuntimeCacheRequest


def _put_snapshot(
    manager: MbltRuntimeCacheManager,
    req_id: str,
    blocks: list[int],
    tokens: int,
) -> None:
    manager.put_snapshot(
        req_id=req_id,
        blobs=[f"blob:{req_id}"],
        block_ids=(blocks,),
        num_tokens=tokens,
    )


class TestMbltRuntimeCacheManagerSnapshots:
    def test_best_prefix_lookup_uses_deepest_indexed_match(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _put_snapshot(manager, "short", [1], 4)
        _put_snapshot(manager, "long", [1, 2, 3], 12)

        match = manager.choose_prefix_snapshot((1, 2, 9), target_tokens=10)

        assert match.req_id == "long"
        assert match.snapshot is manager.get_snapshot("long")
        assert match.matched_tokens == 8
        assert not match.is_own_snapshot

    def test_own_snapshot_has_priority_when_fully_compatible(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _put_snapshot(manager, "other", [1, 2, 3], 12)
        _put_snapshot(manager, "req", [1, 2], 8)

        match = manager.choose_snapshot(
            RuntimeCacheRequest(
                req_id="req",
                block_ids=([1, 2],),
                first_seq_blocks=(1, 2),
                num_computed_tokens=8,
            )
        )

        assert match.req_id == "req"
        assert match.snapshot is manager.get_snapshot("req")
        assert match.matched_tokens == 8
        assert match.is_own_snapshot

    def test_partial_own_snapshot_falls_back_to_best_shared_prefix(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _put_snapshot(manager, "req", [1], 4)
        _put_snapshot(manager, "other", [1, 2, 3], 12)

        match = manager.choose_snapshot(
            RuntimeCacheRequest(
                req_id="req",
                block_ids=([1, 2, 3],),
                first_seq_blocks=(1, 2, 3),
                num_computed_tokens=10,
            )
        )

        assert match.req_id == "other"
        assert match.snapshot is manager.get_snapshot("other")
        assert match.matched_tokens == 10
        assert not match.is_own_snapshot

    def test_snapshot_update_and_removal_refresh_prefix_index(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4)
        _put_snapshot(manager, "req", [1, 2], 8)

        assert manager.choose_prefix_snapshot((1, 2), target_tokens=8).req_id == "req"

        _put_snapshot(manager, "req", [5, 6], 8)

        assert manager.choose_prefix_snapshot((1, 2), target_tokens=8).snapshot is None
        assert manager.choose_prefix_snapshot((5, 6), target_tokens=8).req_id == "req"

        removed = manager.remove_snapshot("req")

        assert removed is not None
        assert manager.choose_prefix_snapshot((5, 6), target_tokens=8).snapshot is None

    def test_finished_snapshot_lru_cap_evicts_oldest_and_rebuilds_index(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4, max_finished_cache_snapshots=2)
        _put_snapshot(manager, "a", [1], 4)
        _put_snapshot(manager, "b", [2], 4)
        _put_snapshot(manager, "c", [3], 4)

        assert manager.mark_snapshot_finished("a") == []
        assert manager.mark_snapshot_finished("b") == []
        manager.loaded_cache_req_id = "a"
        evicted = manager.mark_snapshot_finished("c")

        assert evicted == ["a"]
        assert manager.get_snapshot("a") is None
        assert manager.loaded_cache_req_id is None
        assert list(manager.finished_snapshot_lru) == ["b", "c"]
        assert manager.choose_prefix_snapshot((1,), target_tokens=4).snapshot is None
        assert manager.choose_prefix_snapshot((2,), target_tokens=4).req_id == "b"

    def test_finished_snapshot_lru_touch_preserves_recent_snapshot(self) -> None:
        manager = MbltRuntimeCacheManager(block_size=4, max_finished_cache_snapshots=2)
        _put_snapshot(manager, "a", [1], 4)
        _put_snapshot(manager, "b", [2], 4)
        _put_snapshot(manager, "c", [3], 4)

        manager.mark_snapshot_finished("a")
        manager.mark_snapshot_finished("b")
        manager.mark_snapshot_finished("a")
        evicted = manager.mark_snapshot_finished("c")

        assert evicted == ["b"]
        assert list(manager.finished_snapshot_lru) == ["a", "c"]
        assert manager.get_snapshot("a") is not None
        assert manager.get_snapshot("b") is None