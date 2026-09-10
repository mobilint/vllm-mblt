"""Qwen3-VL 3-input text MXQ: which trailing input is rope and which is deepstack.

The compiler does not fix that order. Shipped Batch16 artifacts declare
[inputs, rope, deepstack]; a rebuild can emit [inputs, deepstack, rope].
Getting it wrong is silent -- both tensors are float32 and the NPU happily
returns garbage logits -- so these tests pin both the classification and the
order the tensors are actually emitted in.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_mblt.mblt_worker import MbltWorker

HIDDEN = 4096
PE = 256
LAYERS = 3
TOKENS = 7

ROPE_SHAPE = (1, -1, PE)
DEEPSTACK_SHAPE = (LAYERS, -1, HIDDEN)
INPUT_SHAPE = (1, -1, HIDDEN)


def _worker(input_shapes, *, is_batch):
    """A MbltWorker with just enough wired up to exercise the input builders."""
    w = MbltWorker.__new__(MbltWorker)
    w._cache_model_input_shapes = lambda _model: list(input_shapes)
    w._get_cache_model = lambda: SimpleNamespace()
    w._is_batch_model = lambda: is_batch
    w._supports_deepstack_input = lambda: True
    return w


def _embeds(seq=TOKENS):
    return (
        np.arange(seq * HIDDEN, dtype=np.float32).reshape(seq, HIDDEN),
        np.full((LAYERS, seq, HIDDEN), 2.0, dtype=np.float32),
        np.full((1, seq, PE), 3.0, dtype=np.float32),
    )


class TestDetectTailOrder:
    def test_shipped_batch_layout_rope_first(self) -> None:
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [ROPE_SHAPE, DEEPSTACK_SHAPE], HIDDEN, ("rope", "deepstack")
        )
        assert order == ("rope", "deepstack")

    def test_rebuilt_layout_deepstack_first(self) -> None:
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [DEEPSTACK_SHAPE, ROPE_SHAPE], HIDDEN, ("rope", "deepstack")
        )
        assert order == ("deepstack", "rope")

    def test_dynamic_last_axis_falls_back_to_default(self) -> None:
        # A deepstack input with a dynamic hidden axis cannot be classified.
        # Falling back keeps whatever the positional convention already did.
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [(LAYERS, -1, -1), ROPE_SHAPE], HIDDEN, ("deepstack", "rope")
        )
        assert order == ("deepstack", "rope")

    def test_pe_size_equal_to_hidden_falls_back_to_default(self) -> None:
        # Ambiguous: both tails end in hidden_size. Guessing here would flip
        # artifacts the positional convention already handled correctly.
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [(1, -1, HIDDEN), DEEPSTACK_SHAPE], HIDDEN, ("rope", "deepstack")
        )
        assert order == ("rope", "deepstack")

    def test_two_input_layout_is_left_alone(self) -> None:
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [DEEPSTACK_SHAPE], HIDDEN, ("rope", "deepstack")
        )
        assert order == ("rope", "deepstack")


@pytest.mark.parametrize("is_batch", [True, False])
@pytest.mark.parametrize(
    "tail_shapes, expect_rope_at",
    [((ROPE_SHAPE, DEEPSTACK_SHAPE), 1), ((DEEPSTACK_SHAPE, ROPE_SHAPE), 2)],
)
class TestEmittedOrderMatchesDeclaration:
    """The detected order must drive the returned tensors, not just validation.

    Parametrized over `is_batch` as well: the two model kinds carry different
    positional defaults, and detection has to win over both.
    """

    def test_batch_path(self, tail_shapes, expect_rope_at, is_batch) -> None:
        w = _worker((INPUT_SHAPE, *tail_shapes), is_batch=is_batch)
        emb, deep, rope = _embeds()
        out = w._build_batch_infer_inputs([emb], [deep], [rope])
        assert len(out) == 3
        expect_ds_at = 2 if expect_rope_at == 1 else 1
        np.testing.assert_allclose(out[expect_rope_at], rope)
        np.testing.assert_allclose(out[expect_ds_at], deep)

    def test_single_request_path(self, tail_shapes, expect_rope_at, is_batch) -> None:
        w = _worker((INPUT_SHAPE, *tail_shapes), is_batch=is_batch)
        emb, deep, rope = _embeds()
        out = w._build_infer_inputs(emb, deep, rope)
        assert len(out) == 3
        expect_ds_at = 2 if expect_rope_at == 1 else 1
        np.testing.assert_allclose(out[expect_rope_at], rope)
        np.testing.assert_allclose(out[expect_ds_at], deep)


class TestPathsAgree:
    def test_both_builders_pick_the_same_order(self) -> None:
        # Compare full shapes, not just the last axis: text_input and deepstack
        # both end in HIDDEN, so a builder that swapped slot 0 with the deepstack
        # slot would pass a last-axis-only assertion.
        for tail in ((ROPE_SHAPE, DEEPSTACK_SHAPE), (DEEPSTACK_SHAPE, ROPE_SHAPE)):
            for is_batch in (True, False):
                w = _worker((INPUT_SHAPE, *tail), is_batch=is_batch)
                emb, deep, rope = _embeds()
                batch_out = w._build_batch_infer_inputs([emb], [deep], [rope])
                single_out = w._build_infer_inputs(emb, deep, rope)
                assert [tuple(t.shape) for t in batch_out] == [tuple(t.shape) for t in single_out], (
                    f"batch and single-request paths disagree for tail={tail}, is_batch={is_batch}"
                )
                # slot 0 is always the text input, in both builders
                np.testing.assert_allclose(batch_out[0], np.expand_dims(emb, 0))
                np.testing.assert_allclose(single_out[0], np.expand_dims(emb, 0))
