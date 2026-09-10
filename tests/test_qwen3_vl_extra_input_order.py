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
    """A MbltWorker with just enough wired up to exercise the input builders.

    Fakes the accelerator handle, the way the other builder tests do, rather
    than shadowing `_cache_model_input_shapes`: that keeps the real reader in
    the path, including the `except Exception: return []` that decides what the
    builders see when a shape query fails.
    """

    w = MbltWorker.__new__(MbltWorker)
    w.cache_model = SimpleNamespace(
        get_num_model_variants=lambda: 1,
        get_model_variant_handle=lambda _idx: SimpleNamespace(
            get_model_input_shape=lambda: [tuple(shape) for shape in input_shapes]
        ),
    )
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

    def test_dynamic_hidden_axis_resolved_by_leading_axis(self) -> None:
        # The last axis is unreadable (dynamic hidden), so the leading axis
        # decides: only deepstack may declare more than one layer. The default
        # is the *wrong* answer here, so this fails if detection falls back.
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [(LAYERS, -1, -1), ROPE_SHAPE], HIDDEN, ("rope", "deepstack")
        )
        assert order == ("deepstack", "rope")

    def test_pe_size_equal_to_hidden_resolved_by_leading_axis(self) -> None:
        # Both tails end in hidden_size, so the last axis cannot separate them.
        # The leading axis still can, and again the default disagrees.
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [(1, -1, HIDDEN), DEEPSTACK_SHAPE], HIDDEN, ("deepstack", "rope")
        )
        assert order == ("rope", "deepstack")

    @pytest.mark.parametrize("default", [("rope", "deepstack"), ("deepstack", "rope")])
    @pytest.mark.parametrize(
        "tails, expected",
        [
            ([(1, -1, HIDDEN), (LAYERS, -1, -1)], ("rope", "deepstack")),
            ([(LAYERS, -1, -1), (1, -1, HIDDEN)], ("deepstack", "rope")),
        ],
    )
    def test_leading_axis_wins_when_the_last_axis_disagrees(self, tails, expected, default) -> None:
        # rope declaring pe_size == hidden_size *and* deepstack declaring a
        # dynamic hidden axis -- the two signatures the last axis cannot read,
        # in one pair. The last axis then reads the rope input as deepstack, so
        # consulting it first inverts the pair. The leading axis is decisive
        # (LAYERS cannot be a rope batch dimension) and has to win. Both
        # defaults are tried, so neither answer can be coming from the fallback.
        assert MbltWorker._detect_qwen3_vl_tail_order(tails, HIDDEN, default) == expected

    @pytest.mark.parametrize("default", [("rope", "deepstack"), ("deepstack", "rope")])
    def test_unreadable_on_both_axes_falls_back_to_default(self, default) -> None:
        # A single-layer deepstack sharing the rope input's declared size is
        # genuinely indistinguishable: neither axis separates the two. Guessing
        # would flip artifacts the positional convention already handled, so the
        # fallback has to win -- whichever way it points.
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [(1, -1, HIDDEN), (1, -1, HIDDEN)], HIDDEN, default
        )
        assert order == default

    def test_two_input_layout_is_left_alone(self) -> None:
        order = MbltWorker._detect_qwen3_vl_tail_order(
            [DEEPSTACK_SHAPE], HIDDEN, ("rope", "deepstack")
        )
        assert order == ("rope", "deepstack")


@pytest.mark.parametrize("is_batch", [True, False])
@pytest.mark.parametrize(
    "tail_shapes, expect_rope_at",
    [
        ((ROPE_SHAPE, DEEPSTACK_SHAPE), 1),
        ((DEEPSTACK_SHAPE, ROPE_SHAPE), 2),
        # Dynamic hidden axis: classified by the leading axis, and emitted
        # end-to-end -- not just resolved by the detection helper in isolation.
        (((LAYERS, -1, -1), ROPE_SHAPE), 2),
    ],
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


class TestShapeQueryFailure:
    def test_a_failing_shape_query_degrades_to_the_text_input_alone(self) -> None:
        # The real reader swallows every exception and returns []. Both builders
        # have to take the len < 2 path then, rather than index a list they
        # believe has three entries -- the hazard a3e29cc removed by threading
        # the caller's shapes through instead of re-reading them.
        def boom():
            raise RuntimeError("accelerator handle is gone")

        emb, deep, rope = _embeds()
        for is_batch in (True, False):
            w = _worker((INPUT_SHAPE, ROPE_SHAPE, DEEPSTACK_SHAPE), is_batch=is_batch)
            w.cache_model = SimpleNamespace(get_num_model_variants=boom)

            single = w._build_infer_inputs(emb, deep, rope)
            assert isinstance(single, np.ndarray)
            np.testing.assert_allclose(single, np.expand_dims(emb, 0))

            # The batch builder rejects deepstack tensors for a model that
            # declares no deepstack input, which is what [] now looks like...
            with pytest.raises(RuntimeError, match="require a dual-input Qwen3-VL MXQ"):
                w._build_batch_infer_inputs([emb], [deep], [rope])
            # ...and emits the text input alone when none are supplied.
            batch = w._build_batch_infer_inputs([emb], [None], [rope])
            assert len(batch) == 1


class TestAmbiguousSignatureIsReported:
    """The fallback is the one path that can emit wrong slots without raising.

    Nothing downstream can catch it -- the declared shapes are identical, so
    every validation check passes either way -- which leaves the log as the only
    way to find out. It has to fire, and it has to fire only once: this runs on
    every decode step.
    """

    AMBIGUOUS = ((1, -1, HIDDEN), (1, -1, HIDDEN))

    def test_warns_once_for_an_unreadable_signature(self, caplog) -> None:
        w = _worker((INPUT_SHAPE, *self.AMBIGUOUS), is_batch=True)
        w._warned_ambiguous_extra_input_order = False
        with caplog.at_level("WARNING"):
            for _ in range(3):
                assert w._qwen3_vl_text_extra_input_order(
                    [INPUT_SHAPE, *self.AMBIGUOUS], HIDDEN
                ) == ("rope", "deepstack")
        warnings = [r for r in caplog.records if "cannot be" in r.getMessage()]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"
        assert "(1, -1, 4096)" in warnings[0].getMessage()

    @pytest.mark.parametrize("tails", [(ROPE_SHAPE, DEEPSTACK_SHAPE), (DEEPSTACK_SHAPE, ROPE_SHAPE)])
    def test_stays_quiet_when_the_shapes_decide(self, tails, caplog) -> None:
        w = _worker((INPUT_SHAPE, *tails), is_batch=True)
        w._warned_ambiguous_extra_input_order = False
        with caplog.at_level("WARNING"):
            w._qwen3_vl_text_extra_input_order([INPUT_SHAPE, *tails], HIDDEN)
        assert [r for r in caplog.records if "cannot be" in r.getMessage()] == []


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
