import pytest

from vllm_mblt.runtime_cache import (
    MbltRuntimeCacheManager,
    PromptEmbedCacheIdentity,
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
    multimodal_cache_identity: object | None = None,
    prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None,
) -> RuntimeCacheRequest:
    return RuntimeCacheRequest(
        req_id=req_id,
        block_ids=(list(blocks),),
        first_seq_blocks=blocks,
        num_computed_tokens=tokens,
        first_seq_block_hashes=block_hashes,
        cache_slot_id=cache_slot_id,
        cache_token_ids=tuple(cache_token_ids) if cache_token_ids is not None else None,
        multimodal_cache_identity=multimodal_cache_identity,
        prompt_embed_cache_identity=prompt_embed_cache_identity,
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
    multimodal_cache_identity: object | None = None,
    prompt_embed_cache_identity: PromptEmbedCacheIdentity | None = None,
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
        multimodal_cache_identity=multimodal_cache_identity,
        prompt_embed_cache_identity=prompt_embed_cache_identity,
    )


def _prompt_embed_identity(prompt_len: int, fingerprints: dict[int, object]) -> PromptEmbedCacheIdentity:
    return PromptEmbedCacheIdentity(
        prompt_len=prompt_len,
        fingerprint_for_prefix=lambda num_tokens: fingerprints.get(int(num_tokens)),
    )


def _vlm_identity(
    session_id: str,
    *,
    offset: int = 2,
    length: int = 2,
    content: object = "image-a",
) -> tuple[object, ...]:
    return ("vlm", session_id, (("image", (offset, length, None), content),))


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

    def test_hashless_snapshot_is_not_invalidated_before_token_identity_check(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "old", (1, 2), 8, cache_token_ids=tuple(range(8)))

        evicted = manager.observe_request_blocks("new", ([1, 3],))

        assert evicted == []
        assert manager.get_snapshot("old") is not None
        match = manager.choose_snapshot(_request("new", (1, 2), 8, cache_token_ids=tuple(range(8))))
        assert match.snapshot is manager.get_snapshot("old")
        assert match.matched_tokens == 8

    def test_hashless_load_invalidates_stale_snapshot_before_restore(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "old", (1, 2), 8, cache_token_ids=tuple(range(8)))
        load_calls = []

        result = manager.load_snapshot_for_request(
            _request("new", (1, 2), 8, cache_token_ids=tuple(range(100, 108))),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert result.cache_miss
        assert result.matched_tokens == 0
        assert load_calls == []
        assert manager.get_snapshot("old") is None

    def test_real_vllm_0112_hashless_shared_prefix_restores_with_token_identity(self) -> None:
        manager = _make_manager(block_size=4)
        token_ids = (10, 11, 12, 13, 14, 15, 16, 17)
        _store_snapshot(manager, "finished-a", (1, 2), 8, cache_token_ids=token_ids)
        manager.mark_snapshot_finished("finished-a")

        new_request_data = type(
            "NewRequestData0112",
            (),
            {
                "__annotations__": {
                    "req_id": str,
                    "prompt_token_ids": list[int],
                    "mm_features": object,
                    "sampling_params": object,
                    "pooling_params": object,
                    "block_ids": object,
                    "num_computed_tokens": int,
                    "lora_request": object,
                    "prompt_embeds": object,
                }
            },
        )()
        new_request_data.req_id = "new-b"
        new_request_data.prompt_token_ids = list(token_ids)
        new_request_data.block_ids = ([1, 2],)
        new_request_data.num_computed_tokens = 8

        manager.observe_request_blocks(new_request_data.req_id, new_request_data.block_ids)
        result = manager.load_snapshot_for_request(
            _request(
                new_request_data.req_id,
                tuple(new_request_data.block_ids[0]),
                new_request_data.num_computed_tokens,
                cache_token_ids=tuple(new_request_data.prompt_token_ids),
            ),
            lambda blobs, slot_id: True,
        )

        assert result.loaded
        assert result.loaded_snapshot_req_id == "finished-a"
        assert result.matched_tokens == 8

    def test_explicit_prompt_embed_identity_rejects_same_tokens_with_different_embeds(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "old",
            (1, 2),
            8,
            cache_token_ids=tuple(range(8)),
            prompt_embed_cache_identity=_prompt_embed_identity(8, {8: ("embed", "old")}),
        )

        loaded = []
        result = manager.load_snapshot_for_request(
            _request(
                "new",
                (1, 2),
                8,
                cache_token_ids=tuple(range(8)),
                prompt_embed_cache_identity=_prompt_embed_identity(8, {8: ("embed", "new")}),
            ),
            lambda blobs, slot_id: loaded.append((blobs, slot_id)) or True,
        )

        assert result.cache_miss
        assert loaded == []

    def test_explicit_prompt_embed_identity_allows_identical_embed_prefix(self) -> None:
        manager = _make_manager(block_size=4)
        fingerprint = ("embed", "same")
        _store_snapshot(
            manager,
            "old",
            (1, 2),
            8,
            cache_token_ids=tuple(range(8)),
            prompt_embed_cache_identity=_prompt_embed_identity(8, {8: fingerprint}),
        )

        loaded = []
        result = manager.load_snapshot_for_request(
            _request(
                "new",
                (1, 2),
                8,
                cache_token_ids=tuple(range(8)),
                prompt_embed_cache_identity=_prompt_embed_identity(8, {8: fingerprint}),
            ),
            lambda blobs, slot_id: loaded.append((blobs, slot_id)) or True,
        )

        assert result.loaded
        assert result.loaded_snapshot_req_id == "old"
        assert result.matched_tokens == 8
        assert loaded == [(["blob:old"], None)]

    def test_explicit_prompt_embed_generated_suffix_uses_token_ids_after_prompt(self) -> None:
        manager = _make_manager(block_size=4)
        fingerprint = ("embed", "same-prompt")
        _store_snapshot(
            manager,
            "old",
            (1, 2),
            6,
            cache_token_ids=(101, 102, 103, 104, 201, 202),
            prompt_embed_cache_identity=_prompt_embed_identity(4, {4: fingerprint}),
        )

        loaded = []
        result = manager.load_snapshot_for_request(
            _request(
                "new",
                (1, 2),
                6,
                cache_token_ids=(301, 302, 303, 304, 201, 202),
                prompt_embed_cache_identity=_prompt_embed_identity(4, {4: fingerprint}),
            ),
            lambda blobs, slot_id: loaded.append((blobs, slot_id)) or True,
        )

        assert result.loaded
        assert result.matched_tokens == 6
        assert loaded == [(["blob:old"], None)]

    def test_explicit_prompt_embed_generated_suffix_mismatch_caps_to_prompt(self) -> None:
        manager = _make_manager(block_size=4)
        fingerprint = ("embed", "same-prompt")
        _store_snapshot(
            manager,
            "old",
            (1, 2),
            6,
            cache_token_ids=(101, 102, 103, 104, 201, 202),
            prompt_embed_cache_identity=_prompt_embed_identity(4, {4: fingerprint}),
        )

        loaded = []
        result = manager.load_snapshot_for_request(
            _request(
                "new",
                (1, 2),
                6,
                cache_token_ids=(301, 302, 303, 304, 201, 999),
                prompt_embed_cache_identity=_prompt_embed_identity(4, {4: fingerprint}),
            ),
            lambda blobs, slot_id: loaded.append((blobs, slot_id)) or True,
        )

        assert result.loaded
        assert result.matched_tokens == 5
        assert loaded == [(["blob:old"], None)]

    def test_real_vllm_0112_cached_request_reused_physical_block_with_different_tokens_is_removed(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "old-content", (7,), 4, cache_token_ids=(1, 2, 3, 4))
        cached_request_data = type(
            "CachedRequestData0112",
            (),
            {
                "__annotations__": {
                    "req_ids": list[str],
                    "resumed_req_ids": set[str],
                    "new_token_ids": list[list[int]],
                    "all_token_ids": dict[str, list[int]],
                    "new_block_ids": object,
                    "num_computed_tokens": list[int],
                    "num_output_tokens": list[int],
                }
            },
        )()
        cached_request_data.req_ids = ["new-content"]
        cached_request_data.all_token_ids = {"new-content": [9, 2, 3, 4]}
        cached_request_data.new_block_ids = [([7],)]
        cached_request_data.num_computed_tokens = [4]

        manager.observe_request_blocks(cached_request_data.req_ids[0], cached_request_data.new_block_ids[0])
        result = manager.load_snapshot_for_request(
            _request(
                cached_request_data.req_ids[0],
                tuple(cached_request_data.new_block_ids[0][0]),
                cached_request_data.num_computed_tokens[0],
                cache_token_ids=tuple(cached_request_data.all_token_ids["new-content"]),
            ),
            lambda blobs, slot_id: True,
        )

        assert result.cache_miss
        assert manager.get_snapshot("old-content") is None

    def test_hashless_multimodal_identity_mismatch_is_rejected(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "vlm-old",
            (30,),
            4,
            cache_token_ids=(1, 2, 3, 4),
            multimodal_cache_identity=("vlm", "session-a", (0, 2, None)),
        )

        manager.observe_request_blocks("vlm-new", ([30],))
        result = manager.load_snapshot_for_request(
            _request(
                "vlm-new",
                (30,),
                4,
                cache_token_ids=(1, 2, 3, 4),
                multimodal_cache_identity=("vlm", "session-b", (0, 2, None)),
            ),
            lambda blobs, slot_id: True,
        )

        assert result.cache_miss
        assert manager.get_snapshot("vlm-old") is None

    def test_identityless_snapshot_is_rejected_when_vlm_prefix_overlaps_embedding(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "vlm-old",
            (30,),
            4,
            cache_token_ids=(1, 2, 3, 4),
            multimodal_cache_identity=None,
        )

        manager.observe_request_blocks("vlm-new", ([30],))
        result = manager.load_snapshot_for_request(
            _request(
                "vlm-new",
                (30,),
                4,
                cache_token_ids=(1, 2, 3, 4),
                multimodal_cache_identity=_vlm_identity("session-a", offset=0, length=2),
            ),
            lambda blobs, slot_id: True,
        )

        assert result.cache_miss
        assert result.matched_tokens == 0

    def test_identityless_snapshot_can_match_text_only_prefix_before_vlm_embedding(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "vlm-old",
            (30,),
            4,
            cache_token_ids=(1, 2, 3, 4),
            multimodal_cache_identity=None,
        )

        match = manager.choose_snapshot(
            _request(
                "vlm-new",
                (30,),
                4,
                cache_token_ids=(1, 2, 3, 4),
                multimodal_cache_identity=_vlm_identity("session-a", offset=4, length=2),
            )
        )

        assert match.req_id == "vlm-old"
        assert match.matched_tokens == 4

    def test_same_vlm_identity_reuses_snapshot(self) -> None:
        manager = _make_manager(block_size=4)
        identity = _vlm_identity("session-a", offset=2, length=2, content="same-image")
        _store_snapshot(
            manager,
            "vlm-old",
            (30,),
            4,
            cache_token_ids=(1, 2, 3, 4),
            multimodal_cache_identity=identity,
        )

        result = manager.load_snapshot_for_request(
            _request(
                "vlm-new",
                (30,),
                4,
                cache_token_ids=(1, 2, 3, 4),
                multimodal_cache_identity=identity,
            ),
            lambda blobs, slot_id: True,
        )

        assert result.loaded
        assert result.loaded_snapshot_req_id == "vlm-old"
        assert result.matched_tokens == 4

    def test_same_session_position_with_different_vlm_content_rejects_snapshot(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(
            manager,
            "vlm-old",
            (30,),
            4,
            cache_token_ids=(1, 2, 3, 4),
            multimodal_cache_identity=_vlm_identity("session-a", offset=0, length=2, content="image-a"),
        )

        manager.observe_request_blocks("vlm-new", ([30],))
        result = manager.load_snapshot_for_request(
            _request(
                "vlm-new",
                (30,),
                4,
                cache_token_ids=(1, 2, 3, 4),
                multimodal_cache_identity=_vlm_identity("session-a", offset=0, length=2, content="image-b"),
            ),
            lambda blobs, slot_id: True,
        )

        assert result.cache_miss
        assert result.matched_tokens == 0
        assert manager.get_snapshot("vlm-old") is None

    def test_unresolved_vlm_identity_trims_reuse_before_embedding_overlap(self) -> None:
        manager = _make_manager(block_size=4)
        unresolved_identity = _vlm_identity("session-a", offset=2, length=2, content=None)
        _store_snapshot(
            manager,
            "vlm-old",
            (30,),
            4,
            cache_token_ids=(1, 2, 3, 4),
            multimodal_cache_identity=unresolved_identity,
        )

        result = manager.load_snapshot_for_request(
            _request(
                "vlm-new",
                (30,),
                4,
                cache_token_ids=(1, 2, 3, 4),
                multimodal_cache_identity=unresolved_identity,
            ),
            lambda blobs, slot_id: True,
        )

        assert result.loaded
        assert result.loaded_snapshot_req_id == "vlm-old"
        assert result.matched_tokens == 2

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


class TestMbltRuntimeCacheLiveTokenTracking:
    def test_live_slot_reuse_requires_matching_token_count(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "req", 8)
        load_calls: list[object] = []

        result = manager.load_snapshot_for_slot(
            _request("req", (1, 2), 8, cache_slot_id=slot_id),
            lambda blobs, load_slot_id: load_calls.append((blobs, load_slot_id)) or True,
        )

        assert result.reused_live_cache
        assert result.matched_tokens == 8
        assert result.live_cache_tokens == 8
        assert not result.live_prefix_incomplete
        assert load_calls == []

    def test_live_slot_reuse_refused_when_token_count_diverges(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        # The slot only holds a 4-token replay while the scheduler believes the
        # request has 8 computed tokens: continuing from it would decode
        # against KV that does not belong to this prefix.
        manager.mark_slot_owner(slot_id, "req", 4)
        load_calls: list[object] = []

        result = manager.load_snapshot_for_slot(
            _request("req", (1, 2), 8, cache_slot_id=slot_id),
            lambda blobs, load_slot_id: load_calls.append((blobs, load_slot_id)) or True,
        )

        assert not result.reused_live_cache
        assert result.cache_miss
        assert result.matched_tokens == 0
        assert result.live_prefix_incomplete
        assert result.live_cache_tokens == 4
        assert result.action == "live-prefix-incomplete"
        assert load_calls == []
        # The rebuild starts from an empty prefix, so the tracked count must say so.
        assert manager.live_slot_owner(slot_id) == "req"
        assert manager.live_slot_tokens(slot_id) == 0

    def test_live_slot_mismatch_can_still_load_a_compatible_snapshot(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "req", 4)
        _store_snapshot(manager, "req", (1, 2), 8)
        load_calls: list[object] = []

        result = manager.load_snapshot_for_slot(
            _request("req", (1, 2), 8, cache_slot_id=slot_id),
            lambda blobs, load_slot_id: load_calls.append((blobs, load_slot_id)) or True,
        )

        assert result.loaded
        assert result.matched_tokens == 8
        assert result.live_prefix_incomplete
        assert load_calls == [(["blob:req"], slot_id)]
        assert manager.live_slot_tokens(slot_id) == 8

    def test_live_slot_with_extra_tokens_is_still_reused(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        # After a preemption the scheduler can report fewer computed tokens than
        # the slot still holds. The owner's token sequence is append-only, so
        # positions 0..7 in the slot are this request's and continuing at 8
        # simply overwrites the stale tail.
        manager.mark_slot_owner(slot_id, "req", 12)
        load_calls: list[object] = []

        result = manager.load_snapshot_for_slot(
            _request("req", (1, 2), 8, cache_slot_id=slot_id),
            lambda blobs, load_slot_id: load_calls.append((blobs, load_slot_id)) or True,
        )

        assert result.reused_live_cache
        assert result.matched_tokens == 8
        assert not result.live_prefix_incomplete
        assert load_calls == []

    def test_live_single_cache_with_extra_tokens_is_still_reused(self) -> None:
        manager = _make_manager(block_size=4)
        manager.mark_loaded_request("req", 12)

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2), 8),
            lambda blobs, slot_id: True,
        )

        assert result.reused_live_cache
        assert result.matched_tokens == 8
        assert not result.live_prefix_incomplete

    def test_released_slot_does_not_inherit_a_token_count(self) -> None:
        manager = _make_manager(max_batch_size=1, block_size=4)
        slot_id = manager.assign_slot("first")
        manager.mark_slot_owner(slot_id, "first", 8)
        manager.release_slot("first")

        assert manager.live_slot_owner(slot_id) is None
        assert manager.live_slot_tokens(slot_id) is None
        assert manager.assign_slot("second") == slot_id

    def test_single_cache_reuse_refused_when_token_count_diverges(self) -> None:
        manager = _make_manager(block_size=4)
        manager.mark_loaded_request("req", 4)
        load_calls: list[object] = []

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2), 8),
            lambda blobs, slot_id: load_calls.append((blobs, slot_id)) or True,
        )

        assert not result.reused_live_cache
        assert result.cache_miss
        assert result.live_prefix_incomplete
        assert result.live_cache_tokens == 4
        assert result.action == "live-prefix-incomplete"
        assert load_calls == []
        assert manager.loaded_req_id is None

    def test_untracked_live_owner_keeps_trusting_the_owner(self) -> None:
        manager = _make_manager(block_size=4)
        manager.mark_loaded_request("req")

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2), 8),
            lambda blobs, slot_id: True,
        )

        assert result.reused_live_cache
        assert result.matched_tokens == 8
        assert result.live_cache_tokens is None

    def test_snapshot_load_records_matched_token_count(self) -> None:
        manager = _make_manager(block_size=4)
        _store_snapshot(manager, "req", (1, 2), 8)

        result = manager.load_snapshot_for_request(
            _request("req", (1, 2), 8),
            lambda blobs, slot_id: True,
        )

        assert result.loaded
        assert manager.loaded_req_id == "req"
        assert manager.loaded_req_tokens == 8
        assert not manager.live_request_prefix_incomplete(8)
        assert manager.live_request_prefix_incomplete(9)

    def test_dump_snapshot_if_needed_clamps_to_tracked_slot_tokens(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "req", 4)
        dump_calls: list[object] = []

        dumped = manager.dump_snapshot_if_needed(
            _request("req", (1, 2, 3), 12, cache_slot_id=slot_id),
            lambda dump_slot_id: dump_calls.append(dump_slot_id) or ["short-blob"],
        )

        assert dumped
        assert dump_calls == [slot_id]
        assert manager.get_snapshot("req").num_tokens == 4

    def test_dump_snapshot_if_needed_skips_an_empty_tracked_slot(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "req", 0)
        dump_calls: list[object] = []

        dumped = manager.dump_snapshot_if_needed(
            _request("req", (1, 2, 3), 12, cache_slot_id=slot_id),
            lambda dump_slot_id: dump_calls.append(dump_slot_id) or ["blob"],
        )

        assert not dumped
        assert dump_calls == []
        assert manager.get_snapshot("req") is None

    def test_dump_snapshot_if_needed_keeps_a_longer_tracked_prefix_label(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "req", 20)

        dumped = manager.dump_snapshot_if_needed(
            _request("req", (1, 2, 3), 12, cache_slot_id=slot_id),
            lambda dump_slot_id: ["blob"],
        )

        assert dumped
        assert manager.get_snapshot("req").num_tokens == 12

    def test_owned_live_cache_tokens_ignores_another_requests_cache(self) -> None:
        manager = _make_manager(max_batch_size=2, block_size=4)
        slot_id = manager.assign_slot("req")
        manager.mark_slot_owner(slot_id, "other", 4)
        manager.mark_loaded_request("other", 4)

        assert manager.owned_live_cache_tokens("req", slot_id) is None
        assert manager.owned_live_cache_tokens("req", None) is None
        assert manager.owned_live_cache_tokens("other", slot_id) == 4
        assert manager.owned_live_cache_tokens("other", None) == 4
