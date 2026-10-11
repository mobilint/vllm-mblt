"""Text models run each step through the transformers-mblt forward on the requests' own caches."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_mblt.mblt_worker import MbltWorker
from vllm_mblt.moe_runtime import MbltMoECacheModel

from .kv_slot_fakes import attach_request_caches

VOCAB = 5


class _RecordingModel:
    """A transformers-mblt causal LM double: logits encode (row, position) so tests can tell them apart."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        cache = kwargs["past_key_values"]
        self.calls.append({**kwargs, "starts": [cache.get_seq_length(row) for row in range(len(cache.layers))]})
        rows, width = kwargs["inputs_embeds"].shape[:2]
        keep = width if kwargs["logits_to_keep"] == 0 else 1
        logits = torch.zeros(rows, keep, VOCAB)
        for row in range(rows):
            for position in range(keep):
                logits[row, position, 0] = row
                logits[row, position, 1] = position if keep > 1 else width - 1
        return SimpleNamespace(logits=logits)


def _make_worker(*, max_batch_size: int, full_logits: bool = False) -> MbltWorker:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_batch_size = max_batch_size
    worker.req_states = {}
    worker.model = _RecordingModel()
    worker._forward_through_model = True
    worker._runtime_output_logits_mode = lambda _length: "full_sequence" if full_logits else "last_token"
    attach_request_caches(worker, k_per_model=max(max_batch_size, 1))
    return worker


def _activate(worker: MbltWorker, req_id: str, slot_id, num_tokens: int) -> None:
    """Give ``req_id`` a cache on ``slot_id`` holding ``num_tokens`` tokens, active for this step."""
    cache = worker.request_caches.new_cache(worker._slot_key(slot_id))
    cache.fake_prefill(num_tokens)
    worker._slot_caches[worker._slot_key(slot_id)] = (req_id, cache)
    worker.req_states[req_id] = SimpleNamespace(kv_cache=cache)


def test_a_batch_step_runs_every_request_on_its_own_cache_in_one_forward() -> None:
    worker = _make_worker(max_batch_size=4)
    _activate(worker, "a", 2, 10)
    _activate(worker, "b", 0, 7)
    embeds = [np.ones((1, 3), dtype=np.float32), np.full((3, 3), 2.0, dtype=np.float32)]

    outputs = worker._infer_logits_batch_with_sequence(embeds, cache_sizes=[10, 4], cache_ids=[2, 0])

    (call,) = worker.model.calls
    assert call["starts"] == [10, 4]  # b was rewound to 4 tokens before the forward
    assert call["past_key_values"].layers == [worker._slot_cache(2).layers[0], worker._slot_cache(0).layers[0]]
    assert [call["past_key_values"].slot_of(row) for row in range(2)] == [(0, 2), (0, 0)]
    assert call["attention_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert call["inputs_embeds"][1].tolist() == [[2.0] * 3] * 3
    assert call["logits_to_keep"] == 1
    assert [output.last_token_logits[0] for output in outputs] == [0.0, 1.0]
    assert all(output.full_sequence_logits is None for output in outputs)


def test_a_single_request_on_a_non_batch_model_goes_without_a_mask() -> None:
    worker = _make_worker(max_batch_size=1)
    _activate(worker, "only", None, 5)

    logits = worker._infer_logits(np.ones((2, 3), dtype=np.float32), None, cache_size=5)

    (call,) = worker.model.calls
    assert call["attention_mask"] is None
    assert call["starts"] == [5]
    assert logits.shape == (VOCAB,)


def test_per_position_logits_come_back_only_for_rows_that_ask_on_a_full_logits_mxq() -> None:
    worker = _make_worker(max_batch_size=4, full_logits=True)
    _activate(worker, "plain", 0, 0)
    _activate(worker, "echo", 1, 0)
    embeds = [np.ones((2, 3), dtype=np.float32), np.ones((3, 3), dtype=np.float32)]

    plain, echo = worker._infer_logits_batch_with_sequence(
        embeds, cache_sizes=[0, 0], cache_ids=[0, 1], keep_sequence_logits=[False, True]
    )

    assert worker.model.calls[0]["logits_to_keep"] == 0
    assert plain.full_sequence_logits is None
    assert plain.last_token_logits[1] == 1.0  # the plain row's own last position, not the padded width's
    assert echo.full_sequence_logits.shape == (3, VOCAB)
    assert echo.last_token_logits[1] == 2.0


def test_last_token_logits_only_on_a_last_logit_mxq_even_when_asked() -> None:
    worker = _make_worker(max_batch_size=1)
    _activate(worker, "echo", None, 0)

    output = worker._infer_logits_with_sequence(np.ones((3, 3), dtype=np.float32), None, cache_size=0)

    assert worker.model.calls[0]["logits_to_keep"] == 1
    assert output.full_sequence_logits is None  # the prompt-logprob microsteps take over


def test_a_cache_that_holds_fewer_tokens_than_the_step_starts_from_is_refused() -> None:
    worker = _make_worker(max_batch_size=1)
    _activate(worker, "short", None, 3)

    with pytest.raises(RuntimeError, match="holds 3 tokens"):
        worker._infer_logits(np.ones((1, 3), dtype=np.float32), None, cache_size=4)


def test_calibration_before_any_request_runs_on_a_scratch_cache() -> None:
    worker = _make_worker(max_batch_size=4)

    worker._infer_logits_batch([np.ones((4, 3), dtype=np.float32)], cache_sizes=[0], cache_ids=[1])

    (call,) = worker.model.calls
    assert call["past_key_values"].slot_of(0) == (0, 1)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, True),
        ({"env": "1"}, False),
        ({"multimodal": True}, False),
        ({"model_type": "mobilint-qwen3_asr"}, False),
        ({"inputs": 2}, False),
        ({"moe": True}, False),
    ],
    ids=["text", "env-direct", "image-text", "audio-text", "extra-mxq-inputs", "moe"],
)
def test_which_models_forward_through_transformers_mblt(monkeypatch, change, expected) -> None:
    worker = MbltWorker.__new__(MbltWorker)
    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=change.get("model_type", "llama")))
    worker.cache_model = MbltMoECacheModel.__new__(MbltMoECacheModel) if change.get("moe") else object()
    worker._is_multimodal_model = lambda: change.get("multimodal", False)
    worker._get_cache_model = lambda: worker.cache_model
    worker._cache_model_input_shapes = lambda _model: [(1, -1, 8)] * change.get("inputs", 1)
    monkeypatch.setenv("VLLM_MBLT_DIRECT_INFER", change.get("env", "0"))

    assert worker._can_forward_through_model() is expected
