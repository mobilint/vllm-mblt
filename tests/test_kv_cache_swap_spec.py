import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "vllm_mblt" / "mblt_worker.py"
WORKER_CODE = WORKER_PATH.read_text(encoding="utf-8")


class TestKvCacheSwapSpec:
    """Spec-style tests for qbruntime cache swap integration.

    These tests are intentionally strict to provide a clear before/after signal.
    They validate that worker code contains the required hooks for:
    - session cache dump/load
    - cache_size-based prefix reuse after cache load
    - avoiding single-page shortcut mapping
    """

    def test_has_cache_dump_and_load_hooks(self) -> None:
        assert "dump_cache_memory(" in WORKER_CODE, (
            "Expected dump_cache_memory() hook for persisting per-session cache snapshots."
        )
        assert "load_cache_memory(" in WORKER_CODE, (
            "Expected load_cache_memory() hook for restoring selected cache snapshots."
        )

    def test_load_happens_before_infer_with_cache_size(self) -> None:
        pattern = re.compile("load_cache_memory\\s*\\(.*?\\).*?infer\\s*\\(.*?cache_size\\s*=", re.DOTALL)
        assert pattern.search(WORKER_CODE), (
            "Expected cache restore followed by infer(..., cache_size=...) to bind binary cache "
            "and effective KV length."
        )

    def test_avoids_single_block_id_shortcut(self) -> None:
        assert "block_ids[0][0]" not in WORKER_CODE, (
            "Single block-id shortcut breaks paged block compatibility for cache swapping."
        )

    def test_cached_request_block_ids_are_appended_when_not_resumed(self) -> None:
        assert "if new_block_ids is not None:" in WORKER_CODE, (
            "Expected cached request updates to consider incremental new_block_ids."
        )
        assert "cached_request_state.block_ids = self._append_block_ids(" in WORKER_CODE, (
            "Expected non-resumed cached requests to append new_block_ids."
        )

    def test_finished_request_does_not_drop_cache_snapshot(self) -> None:
        assert "self.cache_snapshots.pop(req_id, None)" not in WORKER_CODE, (
            "Finished requests should keep cache snapshots for cross-request prefix reuse."
        )

    def test_finished_snapshot_pool_is_lru_capped(self) -> None:
        assert "MAX_FINISHED_CACHE_SNAPSHOTS = 16" in WORKER_CODE, (
            "Finished-session cache snapshots should be capped at 16 entries."
        )
        assert "self._evict_old_finished_snapshots(" in WORKER_CODE, (
            "Expected eviction hook for finished-session snapshot LRU cap."
        )

    def test_dump_is_event_driven_not_unconditional(self) -> None:
        assert "def _should_dump_snapshot_after_step(" in WORKER_CODE, (
            "Expected explicit policy for event-driven cache dump."
        )
        assert "def _dump_loaded_request_before_switch(" in WORKER_CODE, (
            "Expected live cache owner switch to be the event that can trigger snapshots."
        )
        pattern = re.compile(
            r"def _dump_loaded_request_before_switch\(.*?"
            r"if not self\._should_dump_snapshot_after_step\(.*?"
            r"self\._dump_snapshot\(",
            re.DOTALL,
        )
        assert pattern.search(WORKER_CODE), (
            "Expected request-switch snapshot dumps to be gated by the event-driven policy."
        )

    def test_live_request_avoids_immediate_block_boundary_dump(self) -> None:
        assert "self.loaded_cache_req_id = req_id" in WORKER_CODE, (
            "Expected execute loop to track the live accelerator cache owner."
        )
        assert "req_state.num_computed_tokens = next_cache_size" in WORKER_CODE, (
            "Expected post-step token count to advance before a later request switch dump."
        )
        immediate_dump_pattern = re.compile(
            r"req_state\.num_computed_tokens = next_cache_size.*?"
            r"self\.loaded_cache_req_id = req_id.*?"
            r"self\._dump_snapshot\(",
            re.DOTALL,
        )
        assert not immediate_dump_pattern.search(WORKER_CODE), (
            "Expected same-request decode to keep the live accelerator cache instead of immediately dumping it."
        )

    def test_live_cache_is_reused_for_same_request(self) -> None:
        assert "if self.loaded_cache_req_id == req_id:" in WORKER_CODE, (
            "Expected same-request decode path to reuse live accelerator cache."
        )
        assert "return target_tokens" in WORKER_CODE, (
            "Expected live-cache reuse to preserve full computed-token cache_size."
        )
