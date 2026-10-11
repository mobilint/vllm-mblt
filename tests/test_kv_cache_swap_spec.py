from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors

from vllm_mblt.mblt_worker import MbltWorker, RequestState
from vllm_mblt.runtime_cache import MbltRuntimeCacheManager

from .kv_slot_fakes import FakeKVSlotModel, attach_request_caches, make_snapshot_cache


def _make_worker(
    *,
    max_batch_size: int = 1,
    block_size: int = 4,
    enable_prefix_caching: bool | None = None,
) -> MbltWorker:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_batch_size = max_batch_size
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, enable_prefix_caching=enable_prefix_caching),
        model_config=SimpleNamespace(max_model_len=128, hf_config=SimpleNamespace(model_type="qwen2")),
    )
    worker.model_config = worker.vllm_config.model_config
    worker.model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
    worker.req_states = {}
    worker.empty_logits_processors = LogitsProcessors(None)
    worker.empty_prompt_token_ids = None
    worker.runtime_cache = MbltRuntimeCacheManager(
        max_batch_size=max_batch_size,
        block_size=block_size,
        max_finished_snapshots=MbltWorker.MAX_FINISHED_CACHE_SNAPSHOTS,
        enable_prefix_caching=worker._prefix_caching_enabled(),
    )
    (worker.cache_model,) = attach_request_caches(worker, k_per_model=max_batch_size)
    return worker


def _make_request_state(
    worker: MbltWorker,
    *,
    block_ids: tuple[list[int], ...] = ([1],),
    num_computed_tokens: int = 0,
    prompt_len: int = 0,
    output_token_ids: list[int] | None = None,
    cache_slot_id: int | None = None,
) -> RequestState:
    sampling_params = SamplingParams.from_optional()
    prompt_token_ids = list(range(prompt_len))
    return RequestState(
        is_prefill=False,
        output_token_ids=output_token_ids or [],
        sampling_params=sampling_params,
        cached_sampling_state=worker._make_cached_sampling_state(sampling_params, prompt_token_ids),
        block_ids=block_ids,
        first_seq_blocks=tuple(block_ids[0]) if block_ids else (),
        first_seq_block_hashes=None,
        num_computed_tokens=num_computed_tokens,
        num_output_tokens=0,
        prompt_embeds=np.zeros((prompt_len, 2), dtype=np.float32),
        prompt_deepstack_embeds=None,
        prompt_rope_embeds=None,
        mrope_position_delta=0,
        is_multimodal=False,
        prompt_len=prompt_len,
        prompt_token_ids=prompt_token_ids,
        cache_slot_id=cache_slot_id,
        vlm_session_id=None,
    )


def _run(worker: MbltWorker, req_id: str, req_state: RequestState, num_tokens: int) -> int:
    """Run one step of ``req_id`` up to ``num_tokens`` tokens, as execute_model does; return its prefix start."""
    worker.req_states[req_id] = req_state
    cache_size = worker._load_snapshot_if_needed(req_id, req_state, slot_id=req_state.cache_slot_id)
    worker._start_request_step(req_id, req_state, cache_size, slot_id=req_state.cache_slot_id)
    worker.request_caches.commit(req_state.kv_cache, num_tokens - cache_size)
    req_state.num_computed_tokens = num_tokens
    return cache_size


class TestKvCacheSwapBehavior:
    def test_dump_snapshot_stores_a_copy_of_the_request_cache(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([7, 8],))
        _run(worker, "req", req_state, 6)

        assert worker._dump_snapshot("req", req_state, next_num_tokens=6)

        snapshot = worker.runtime_cache.get_snapshot("req")
        assert snapshot is not None
        assert snapshot.blobs is not req_state.kv_cache
        assert snapshot.blobs.get_seq_length() == 6
        assert snapshot.block_ids == ([7, 8],)
        assert snapshot.first_seq_blocks == (7, 8)
        assert snapshot.num_tokens == 6
        assert worker.cache_model.dumps == [0]

    def test_request_cache_is_reused_for_same_request_without_reload(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],))
        _run(worker, "req", req_state, 8)

        assert _run(worker, "req", req_state, 9) == 8
        assert worker.cache_model.loads == []
        assert worker.cache_model.dumps == []

    def test_switching_requests_saves_the_previous_request_and_restores_it_on_return(self) -> None:
        worker = _make_worker(block_size=4)
        live_state = _make_request_state(worker, block_ids=([1, 2],))
        other_state = _make_request_state(worker, block_ids=([9],))
        _run(worker, "live", live_state, 8)

        assert _run(worker, "other", other_state, 4) == 0
        snapshot = worker.runtime_cache.get_snapshot("live")
        assert snapshot is not None
        assert snapshot.num_tokens == 8
        assert worker.cache_model.dumps == [0]

        assert _run(worker, "live", live_state, 9) == 8
        assert worker.cache_model.dumps == [0, 0]
        assert [cache_id for cache_id, _ in worker.cache_model.loads] == [0]

    def test_next_request_starts_from_the_prefix_the_switched_out_request_just_computed(self) -> None:
        worker = _make_worker(block_size=4)
        a_state = _make_request_state(worker, block_ids=([1, 2],), prompt_len=8)
        _run(worker, "a", a_state, 8)
        # b shares a's 8-token prompt, so the scheduler reports it as computed.
        b_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8, prompt_len=8)

        assert _run(worker, "b", b_state, 9) == 8

        # a was saved once when b took the slot, and b loaded that copy.
        assert worker.cache_model.dumps == [0]
        assert [cache_id for cache_id, _ in worker.cache_model.loads] == [0]
        assert worker.runtime_cache.get_snapshot("a").num_tokens == 8

    def test_cache_miss_falls_back_to_recompute(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8)

        matched_tokens = worker._load_snapshot_if_needed("req", req_state)

        assert matched_tokens == 0
        assert worker.cache_model.loads == []

    def test_finished_request_keeps_its_cache_as_snapshot_for_cross_request_prefix_reuse(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],))
        _run(worker, "finished", req_state, 8)

        worker._finalize_finished_request("finished")

        snapshot = worker.runtime_cache.get_snapshot("finished")
        assert snapshot is not None and snapshot.blobs is req_state.kv_cache
        assert list(worker.runtime_cache.finished_snapshot_lru) == ["finished"]
        assert worker.cache_model.dumps == []
        assert "finished" not in worker.req_states

    def test_finished_snapshot_pool_is_lru_capped_through_runtime_cache(self) -> None:
        worker = _make_worker(block_size=4)
        worker.runtime_cache.max_finished_snapshots = 2
        for req_id, block_id in (("a", 1), ("b", 2), ("c", 3)):
            worker.runtime_cache.store_snapshot(
                req_id=req_id,
                blobs=make_snapshot_cache(worker, 4),
                block_ids=([block_id],),
                first_seq_blocks=(block_id,),
                num_tokens=4,
            )
            worker.runtime_cache.touch_finished_snapshot(req_id)

        worker.runtime_cache.evict_old_finished_snapshots()

        assert worker.runtime_cache.get_snapshot("a") is None
        assert worker.runtime_cache.get_snapshot("b") is not None
        assert worker.runtime_cache.get_snapshot("c") is not None
        assert list(worker.runtime_cache.finished_snapshot_lru) == ["b", "c"]

    def test_cached_request_block_ids_are_appended_when_not_resumed(self) -> None:
        worker = _make_worker(block_size=4)

        assert worker._append_block_ids(([1], [10]), ([2, 3], [11])) == ([1, 2, 3], [10, 11])
        with pytest.raises(RuntimeError, match="KV block_ids layout mismatch"):
            worker._append_block_ids(([1],), ([2], [3]))

    def test_batch_requests_on_their_own_slots_never_swap(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4)
        a_state = _make_request_state(worker, block_ids=([1, 2],), cache_slot_id=0)
        b_state = _make_request_state(worker, block_ids=([3, 4],), cache_slot_id=1)

        _run(worker, "a", a_state, 8)
        _run(worker, "b", b_state, 8)
        assert _run(worker, "a", a_state, 9) == 8
        assert _run(worker, "b", b_state, 9) == 8

        assert worker.cache_model.dumps == []
        assert worker.cache_model.loads == []
        assert a_state.kv_cache.slot_of(0) == (0, 0)
        assert b_state.kv_cache.slot_of(0) == (0, 1)

    def test_batch_shared_prefix_from_another_slot_is_loaded_into_the_request_slot(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4)
        done_state = _make_request_state(worker, block_ids=([1, 2],), prompt_len=8, cache_slot_id=0)
        _run(worker, "done", done_state, 8)
        worker._finalize_finished_request("done")
        new_state = _make_request_state(
            worker, block_ids=([1, 2],), num_computed_tokens=8, prompt_len=8, cache_slot_id=1
        )

        assert _run(worker, "new", new_state, 9) == 8

        # The snapshot was still on slot 0: copying it saved it, and slot 1 loaded the copy.
        assert worker.cache_model.dumps == [0]
        assert [cache_id for cache_id, _ in worker.cache_model.loads] == [1]
        assert new_state.kv_cache.slot_of(0) == (0, 1)

    def test_batch_shared_prefix_on_the_request_slot_is_taken_over_without_load(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4)
        done_state = _make_request_state(worker, block_ids=([1, 2],), prompt_len=8, cache_slot_id=0)
        _run(worker, "done", done_state, 8)
        done_cache = done_state.kv_cache
        worker._finalize_finished_request("done")
        new_state = _make_request_state(
            worker, block_ids=([1, 2],), num_computed_tokens=8, prompt_len=8, cache_slot_id=0
        )

        assert _run(worker, "new", new_state, 9) == 8

        # The request continues on the finished cache's KV in place; the snapshot keeps a host copy.
        assert new_state.kv_cache is done_cache
        assert worker.cache_model.dumps == [0]
        assert worker.cache_model.loads == []
        snapshot = worker.runtime_cache.get_snapshot("done")
        assert snapshot is not None and snapshot.blobs is not done_cache
        assert snapshot.blobs.get_seq_length() == 8


class TestKvCacheSwapWithoutPrefixCaching:
    """--no-enable-prefix-caching keeps each request's own KV but drops prefix reuse (issue #28)."""

    @staticmethod
    def _store_foreign_snapshot(worker: MbltWorker) -> None:
        # A finished snapshot from another request on the same physical blocks
        # with the same tokens: with prefix caching on this is a load-shared
        # hit; without it, physical block numbers prove nothing.
        worker.runtime_cache.store_snapshot(
            req_id="old",
            blobs=make_snapshot_cache(worker, 8),
            block_ids=([1, 2],),
            first_seq_blocks=(1, 2),
            num_tokens=8,
            cache_token_ids=tuple(range(8)),
        )
        worker.runtime_cache.touch_finished_snapshot("old")

    def test_unset_prefix_caching_flag_defaults_to_enabled(self) -> None:
        worker = _make_worker(enable_prefix_caching=None)

        assert worker._prefix_caching_enabled()
        assert worker.runtime_cache.enable_prefix_caching

    def test_finished_request_is_dropped_without_dumping_without_prefix_caching(self) -> None:
        worker = _make_worker(block_size=4, enable_prefix_caching=False)
        _run(worker, "finished", _make_request_state(worker, block_ids=([1, 2],)), 8)

        worker._finalize_finished_request("finished")
        _run(worker, "next", _make_request_state(worker, block_ids=([3],)), 4)

        assert worker.cache_model.dumps == []
        assert worker.runtime_cache.get_snapshot("finished") is None
        assert list(worker.runtime_cache.finished_snapshot_lru) == []
        assert "finished" not in worker.req_states

    def test_finished_request_is_kept_with_prefix_caching(self) -> None:
        worker = _make_worker(block_size=4, enable_prefix_caching=True)
        _run(worker, "finished", _make_request_state(worker, block_ids=([1, 2],)), 8)

        worker._finalize_finished_request("finished")

        assert worker.cache_model.dumps == []
        assert worker.runtime_cache.get_snapshot("finished") is not None
        assert list(worker.runtime_cache.finished_snapshot_lru) == ["finished"]

    def test_finished_batch_request_is_dropped_and_releases_slot_without_prefix_caching(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4, enable_prefix_caching=False)
        slot_id = worker._assign_cache_slot("finished")
        _run(worker, "finished", _make_request_state(worker, block_ids=([1, 2],), cache_slot_id=slot_id), 8)

        worker._finalize_finished_request("finished")

        assert worker.cache_model.dumps == []
        assert worker.runtime_cache.get_snapshot("finished") is None
        assert worker.runtime_cache.free_cache_slots == [0, 1]

    def test_finished_batch_request_is_kept_with_prefix_caching(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4, enable_prefix_caching=True)
        slot_id = worker._assign_cache_slot("finished")
        _run(worker, "finished", _make_request_state(worker, block_ids=([1, 2],), cache_slot_id=slot_id), 8)

        worker._finalize_finished_request("finished")

        assert worker.cache_model.dumps == []
        assert worker.runtime_cache.get_snapshot("finished") is not None

    def test_non_batch_swap_restores_each_request_without_snapshots_without_prefix_caching(self) -> None:
        worker = _make_worker(block_size=4, enable_prefix_caching=False)
        a_state = _make_request_state(worker, block_ids=([1, 2],), prompt_len=8)
        b_state = _make_request_state(worker, block_ids=([3, 4],), prompt_len=8)
        _run(worker, "a", a_state, 8)

        # Switching to b saves a's KV in a's cache so a can resume later.
        assert _run(worker, "b", b_state, 8) == 0
        assert worker.cache_model.dumps == [0]

        # Switching back to a saves b, then loads a's KV.
        assert _run(worker, "a", a_state, 9) == 8
        assert worker.cache_model.dumps == [0, 0]
        assert [cache_id for cache_id, _ in worker.cache_model.loads] == [0]
        assert worker.runtime_cache.snapshot_count() == 0

    def test_new_request_never_loads_another_requests_snapshot_without_prefix_caching(self) -> None:
        worker = _make_worker(block_size=4, enable_prefix_caching=False)
        self._store_foreign_snapshot(worker)
        new_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8, prompt_len=8)
        worker.req_states["new"] = new_state

        assert worker._load_snapshot_if_needed("new", new_state) == 0
        assert worker.cache_model.loads == []

    def test_new_batch_request_never_loads_another_requests_snapshot_without_prefix_caching(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4, enable_prefix_caching=False)
        self._store_foreign_snapshot(worker)
        slot_id = worker._assign_cache_slot("new")
        new_state = _make_request_state(
            worker, block_ids=([1, 2],), num_computed_tokens=8, prompt_len=8, cache_slot_id=slot_id
        )
        worker.req_states["new"] = new_state

        assert worker._load_snapshot_if_needed("new", new_state, slot_id=slot_id) == 0
        assert worker.cache_model.loads == []

    def test_same_foreign_snapshot_is_loaded_shared_with_prefix_caching(self) -> None:
        worker = _make_worker(block_size=4, enable_prefix_caching=True)
        self._store_foreign_snapshot(worker)
        new_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8, prompt_len=8)
        worker.req_states["new"] = new_state

        assert worker._load_snapshot_if_needed("new", new_state) == 8
        assert [cache_id for cache_id, _ in worker.cache_model.loads] == [0]


class TestFinishedSnapshotHostBudget:
    """Finished snapshots are capped by host memory as well as by count."""

    MIB = 2**20

    def _cap(self, monkeypatch, *, kv_mib: int, env: str | None = None, extra: dict | None = None) -> int:
        worker = _make_worker()
        attach_request_caches(worker, models=[FakeKVSlotModel(kv_nbytes=kv_mib * self.MIB)])
        worker.vllm_config.load_config = SimpleNamespace(model_loader_extra_config=extra or {})
        if env is None:
            monkeypatch.delenv("VLLM_MBLT_PREFIX_CACHE_MAX_HOST_MIB", raising=False)
        else:
            monkeypatch.setenv("VLLM_MBLT_PREFIX_CACHE_MAX_HOST_MIB", env)
        return worker._finished_snapshot_cap()

    def test_default_budget_keeps_sixteen_256_mib_snapshots(self, monkeypatch) -> None:
        assert self._cap(monkeypatch, kv_mib=256) == 16

    def test_larger_snapshots_are_capped_by_the_default_budget(self, monkeypatch) -> None:
        assert self._cap(monkeypatch, kv_mib=1024) == 4

    def test_budget_comes_from_the_environment_or_extra_config(self, monkeypatch) -> None:
        assert self._cap(monkeypatch, kv_mib=256, env="1024") == 4
        assert self._cap(monkeypatch, kv_mib=256, extra={"prefix_cache_max_host_mib": 512}) == 2

    def test_zero_budget_keeps_no_finished_snapshots(self, monkeypatch) -> None:
        assert self._cap(monkeypatch, kv_mib=256, env="0") == 0

    def test_unknown_snapshot_size_keeps_the_count_cap(self, monkeypatch) -> None:
        assert self._cap(monkeypatch, kv_mib=0, env="1") == MbltWorker.MAX_FINISHED_CACHE_SNAPSHOTS

    def test_invalid_budget_falls_back_to_the_default(self, monkeypatch) -> None:
        assert self._cap(monkeypatch, kv_mib=1024, env="lots") == 4

    def test_finished_snapshots_beyond_the_cap_are_dropped(self) -> None:
        worker = _make_worker(block_size=4)
        worker.runtime_cache.max_finished_snapshots = 1
        for req_id, block in (("a", 1), ("b", 2)):
            _run(worker, req_id, _make_request_state(worker, block_ids=([block],)), 4)
            worker._finalize_finished_request(req_id)

        assert worker.runtime_cache.get_snapshot("a") is None
        assert worker.runtime_cache.get_snapshot("b") is not None
