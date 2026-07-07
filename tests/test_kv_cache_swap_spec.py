from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors

from vllm_mblt.mblt_worker import MbltWorker, RequestState
from vllm_mblt.runtime_cache import MbltRuntimeCacheManager


class RecordingCacheModel:
    def __init__(self) -> None:
        self.dumps: list[dict[str, int | None]] = []
        self.loads: list[tuple[list[object], int | None]] = []

    def dump_cache_memory(self, cache_id: int | None = None) -> list[object]:
        self.dumps.append({"cache_id": cache_id})
        return [f"dump:{len(self.dumps)}:{cache_id}"]

    def load_cache_memory(self, blobs: list[object], cache_id: int | None = None) -> None:
        self.loads.append((blobs, cache_id))


def _make_worker(*, max_batch_size: int = 1, block_size: int = 4) -> MbltWorker:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_batch_size = max_batch_size
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=128, hf_config=SimpleNamespace(model_type="qwen2")),
    )
    worker.model_config = worker.vllm_config.model_config
    worker.model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
    worker.cache_model = RecordingCacheModel()
    worker.req_states = {}
    worker._warned_batch_cache_snapshot_unsupported = False
    worker.empty_logits_processors = LogitsProcessors(None)
    worker.empty_prompt_token_ids = None
    worker.runtime_cache = MbltRuntimeCacheManager(
        max_batch_size=max_batch_size,
        block_size=block_size,
        max_finished_snapshots=MbltWorker.MAX_FINISHED_CACHE_SNAPSHOTS,
        dump_runtime_cache=worker._dump_runtime_cache,
        load_runtime_cache=worker._load_runtime_cache,
    )
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
        num_computed_tokens=num_computed_tokens,
        num_output_tokens=0,
        prompt_embeds=np.zeros((prompt_len, 2), dtype=np.float32),
        prompt_deepstack_embeds=None,
        is_multimodal=False,
        prompt_len=prompt_len,
        prompt_token_ids=prompt_token_ids,
        cache_slot_id=cache_slot_id,
        vlm_session_id=None,
    )


class TestKvCacheSwapBehavior:
    def test_dump_snapshot_stores_runtime_cache_snapshot(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([7, 8],), num_computed_tokens=6)

        assert worker._dump_snapshot("req", req_state, next_num_tokens=6)

        snapshot = worker.runtime_cache.get_snapshot("req")
        assert snapshot is not None
        assert snapshot.blobs == ["dump:1:None"]
        assert snapshot.block_ids == ([7, 8],)
        assert snapshot.first_seq_blocks == (7, 8)
        assert snapshot.num_tokens == 6
        assert worker.cache_model.dumps == [{"cache_id": None}]

    def test_load_snapshot_if_needed_loads_own_snapshot_and_marks_live_request(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8)
        worker.runtime_cache.store_snapshot(
            req_id="req",
            blobs=["own-cache"],
            block_ids=req_state.block_ids,
            first_seq_blocks=req_state.first_seq_blocks,
            num_tokens=8,
        )

        matched_tokens = worker._load_snapshot_if_needed("req", req_state)

        assert matched_tokens == 8
        assert worker.cache_model.loads == [(["own-cache"], None)]
        assert worker.runtime_cache.loaded_req_id == "req"

    def test_live_cache_is_reused_for_same_request_without_reload(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8)
        worker.runtime_cache.mark_loaded_request("req")

        matched_tokens = worker._load_snapshot_if_needed("req", req_state)

        assert matched_tokens == 8
        assert worker.cache_model.loads == []
        assert worker.cache_model.dumps == []
        assert worker.runtime_cache.loaded_req_id == "req"

    def test_switching_requests_dumps_previous_live_request_before_zero_token_skip(self) -> None:
        worker = _make_worker(block_size=4)
        live_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8)
        zero_state = _make_request_state(worker, block_ids=([9],), num_computed_tokens=0)
        worker.req_states["live"] = live_state
        worker.runtime_cache.mark_loaded_request("live")

        matched_tokens = worker._load_snapshot_if_needed("zero", zero_state)

        assert matched_tokens == 0
        snapshot = worker.runtime_cache.get_snapshot("live")
        assert snapshot is not None
        assert snapshot.num_tokens == 8
        assert worker.runtime_cache.loaded_req_id is None
        assert worker.cache_model.dumps == [{"cache_id": None}]

    def test_cache_miss_clears_live_request_and_falls_back_to_recompute(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8)
        worker.runtime_cache.mark_loaded_request("other")

        matched_tokens = worker._load_snapshot_if_needed("req", req_state)

        assert matched_tokens == 0
        assert worker.runtime_cache.loaded_req_id is None
        assert worker.cache_model.loads == []

    def test_finished_request_keeps_snapshot_for_cross_request_prefix_reuse(self) -> None:
        worker = _make_worker(block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8)
        worker.req_states["finished"] = req_state
        worker.runtime_cache.mark_loaded_request("finished")

        worker._finalize_finished_request("finished")

        assert worker.runtime_cache.get_snapshot("finished") is not None
        assert list(worker.runtime_cache.finished_snapshot_lru) == ["finished"]
        assert worker.runtime_cache.loaded_req_id is None
        assert "finished" not in worker.req_states

    def test_finished_snapshot_pool_is_lru_capped_through_runtime_cache(self) -> None:
        worker = _make_worker(block_size=4)
        worker.runtime_cache.max_finished_snapshots = 2
        for req_id, block_id in (("a", 1), ("b", 2), ("c", 3)):
            worker.runtime_cache.store_snapshot(
                req_id=req_id,
                blobs=[req_id],
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

    def test_batch_slot_dump_and_load_use_slot_scoped_runtime_cache(self) -> None:
        worker = _make_worker(max_batch_size=2, block_size=4)
        req_state = _make_request_state(worker, block_ids=([1, 2],), num_computed_tokens=8, cache_slot_id=1)

        assert worker._dump_snapshot("req", req_state, next_num_tokens=8, slot_id=1)
        matched_tokens = worker._load_snapshot_if_needed("req", req_state, slot_id=1)

        assert matched_tokens == 8
        assert worker.cache_model.dumps == [{"cache_id": 1}]
        assert worker.cache_model.loads == [(["dump:1:1"], 1)]
        assert worker.runtime_cache.live_slot_owner(1) == "req"
