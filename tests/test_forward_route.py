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
    """A transformers-mblt causal LM double: logits encode (row, position) so tests can tell them apart.

    ``logits_to_keep=1`` returns the batch's last column, as upstream transformers does, so a row whose last token is
    not in that column would get a padding position. ``logits_to_keep=0`` returns each row's own positions from
    column 0, as the transformers-mblt batched forward does (padding dropped by ``attention_mask``).
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        cache = kwargs["past_key_values"]
        self.calls.append({**kwargs, "starts": [cache.get_seq_length(row) for row in range(len(cache.layers))]})
        rows, width = kwargs["inputs_embeds"].shape[:2]
        mask = kwargs["attention_mask"]
        mask = torch.ones(rows, width, dtype=torch.long) if mask is None else mask
        keep_all = kwargs["logits_to_keep"] == 0
        logits = torch.full((rows, width if keep_all else 1, VOCAB), float("-inf"))
        for row in range(rows):
            valid = [column for column in range(width) if mask[row, column]]
            columns = valid if keep_all else [width - 1]
            for out, column in enumerate(columns):
                logits[row, out, 0] = row
                # The row-local position of that column; -1 marks a padding column.
                logits[row, out, 1] = valid.index(column) if column in valid else -1
        return SimpleNamespace(logits=logits)


def _make_worker(*, max_batch_size: int, full_logits: bool = False) -> MbltWorker:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_batch_size = max_batch_size
    worker.req_states = {}
    worker.model = _RecordingModel()
    worker.cache_model = object()  # the text MXQ handle; tests never call it on this route
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
    assert call["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]  # left-padded
    assert call["inputs_embeds"][1].tolist() == [[2.0] * 3] * 3
    assert call["logits_to_keep"] == 1
    assert call["npu_prefill_chunk_size"] == 3  # the step as vLLM sized it, not re-split by the model
    assert [output.last_token_logits[0] for output in outputs] == [0.0, 1.0]
    # The 1-token row samples from its own token, not from the 3-wide batch's padding.
    assert [output.last_token_logits[1] for output in outputs] == [0.0, 2.0]
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
    # Each row keeps its own copy, not a view of the whole batched output.
    assert plain.last_token_logits.base is None and echo.full_sequence_logits.base is None


def test_last_token_rows_are_views_of_the_forward_output_not_copies() -> None:
    # One row per request is exactly what the rows keep, so copying each one only costs time.
    worker = _make_worker(max_batch_size=4)
    _activate(worker, "a", 0, 0)
    _activate(worker, "b", 1, 0)

    a, b = worker._infer_logits_batch([np.ones((2, 3), dtype=np.float32)] * 2, cache_sizes=[0, 0], cache_ids=[0, 1])

    assert a.base is not None and np.shares_memory(a.base, b)
    assert not np.shares_memory(a, b)
    assert a.tolist()[:2] == [0.0, 1.0] and b.tolist()[:2] == [1.0, 1.0]


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
    ("batch", "rope_input", "input_count", "supported"),
    [(False, False, 2, True), (True, True, 3, True), (True, True, 5, True),
     (True, False, 4, True), (True, False, 2, False), (True, False, 0, False)],
    ids=["non-batch-static", "batch-bundled-dynamic", "batch-split-dynamic", "batch-split-static",
         "batch-bundled-static", "batch-unknown"],
)
def test_qwen3_vl_batch_mxq_layout_validation(batch, rope_input, input_count, supported) -> None:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_batch_size = 16 if batch else 1
    worker._supports_deepstack_input = lambda: True
    worker._text_mxq_uses_rope_input = lambda: rope_input
    worker._get_cache_model = lambda: None
    worker._cache_model_input_shapes = lambda _: [(1, -1, 4)] * input_count

    if supported:
        worker._check_text_mxq_supported()
    else:
        with pytest.raises(RuntimeError, match="without a RoPE input"):
            worker._check_text_mxq_supported()


class _RecordingTextDecoder(_RecordingModel):
    """An image-text model's text decoder: it returns the logits as ``last_hidden_state``."""

    def __call__(self, **kwargs):
        return SimpleNamespace(last_hidden_state=super().__call__(**kwargs).logits)


def _make_vl_worker(*, max_batch_size: int) -> MbltWorker:
    worker = _make_worker(max_batch_size=max_batch_size)
    decoder = _RecordingTextDecoder()
    worker.model = SimpleNamespace(get_language_model=lambda: decoder)
    worker._is_multimodal_model = lambda: True
    worker._text_mxq_uses_rope_input = lambda: True
    return worker


def test_row_positions_continue_the_prompt_m_rope_positions_then_add_the_offset() -> None:
    prompt = torch.tensor([[0, 1, 2, 2, 3], [0, 1, 2, 3, 3], [0, 1, 2, 3, 4]])  # an image squeezes positions
    state = SimpleNamespace(prompt_position_ids=prompt, mrope_position_delta=-1)

    assert MbltWorker._row_position_ids(state, 3, 7).tolist() == [[2, 3, 4, 5], [3, 3, 4, 5], [3, 4, 4, 5]]
    assert MbltWorker._row_position_ids(None, 2, 4).tolist() == [[2, 3]] * 3  # calibration: plain text


def test_an_image_text_step_feeds_each_row_its_positions_and_deepstack_to_the_text_decoder() -> None:
    worker = _make_vl_worker(max_batch_size=4)
    _activate(worker, "image", 0, 0)
    _activate(worker, "text", 1, 6)
    worker.req_states["image"].prompt_position_ids = torch.tensor([[0, 1, 1], [0, 1, 2], [0, 2, 1]])
    worker.req_states["image"].mrope_position_delta = -1
    worker.req_states["text"].prompt_position_ids = None
    deepstack = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)  # (layers, tokens, hidden)

    worker._infer_logits_batch_with_sequence(
        [np.ones((3, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)],
        cache_sizes=[0, 6],
        cache_ids=[0, 1],
        deepstack_embeds_batch=[deepstack, None],
    )

    (call,) = worker.model.get_language_model().calls
    assert call["position_ids"][:, 0].tolist() == [[0, 1, 1], [0, 1, 2], [0, 2, 1]]
    assert call["position_ids"][:, 1].tolist() == [[0, 0, 6]] * 3  # left-padded text row at its own position
    assert call["visual_pos_masks"].tolist() == [[True, True, True], [False, False, True]]
    layer0 = call["deepstack_visual_embeds"][0]
    assert layer0[:3].tolist() == deepstack[0].tolist() and layer0[3:].tolist() == [[0.0, 0.0, 0.0]]
    assert len(call["deepstack_visual_embeds"]) == 2


def test_a_text_only_step_on_an_image_text_model_sends_no_deepstack() -> None:
    worker = _make_vl_worker(max_batch_size=1)
    _activate(worker, "text", None, 4)
    worker.req_states["text"].prompt_position_ids = None

    logits = worker._infer_logits(np.ones((2, 3), dtype=np.float32), None, cache_size=4)

    (call,) = worker.model.get_language_model().calls
    assert "deepstack_visual_embeds" not in call
    assert call["position_ids"][:, 0].tolist() == [[4, 5]] * 3
    assert logits.shape == (VOCAB,)


def test_a_mixture_of_experts_step_runs_on_the_request_cache_with_the_model_s_own_prefill_split() -> None:
    worker = _make_worker(max_batch_size=1)
    worker.cache_model = MbltMoECacheModel.__new__(MbltMoECacheModel)
    _activate(worker, "moe", None, 7)

    worker._infer_logits(np.ones((3, 3), dtype=np.float32), None, cache_size=7)

    (call,) = worker.model.calls
    assert call["past_key_values"] is worker.req_states["moe"].kv_cache  # not a joined view
    assert "npu_prefill_chunk_size" not in call
    assert call["starts"] == [7]


def test_an_audio_text_step_runs_on_the_thinker_s_text_decoder() -> None:
    worker = _make_worker(max_batch_size=1)
    decoder = _RecordingTextDecoder()
    worker.model = SimpleNamespace(thinker=SimpleNamespace(model=decoder))
    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="mobilint-qwen3_asr"))
    worker._is_multimodal_model = lambda: False
    _activate(worker, "asr", None, 9)

    logits = worker._infer_logits(np.ones((4, 3), dtype=np.float32), None, cache_size=9)

    (call,) = decoder.calls
    assert call["past_key_values"] is worker.req_states["asr"].kv_cache  # a plain cache: one sequence, no beams
    assert call["starts"] == [9] and "position_ids" not in call
    assert logits.shape == (VOCAB,)
