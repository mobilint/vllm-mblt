import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

import vllm_mblt.sampling as host_sampling
from vllm_mblt.sampling import HostTopKTopPSampler, _sample_top_k

LOGITS = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, float("-inf")])
VOCAB = LOGITS.numel()
ROWS = 40_000


def _expected(k, p):
    logits = LOGITS.unsqueeze(0).clone()
    k_t = None if k is None else torch.tensor([k])
    p_t = None if p is None else torch.tensor([p])
    return apply_top_k_top_p_pytorch(logits, k_t, p_t).softmax(dim=-1)[0]


def _frequencies(sampler, k, p):
    logits = LOGITS.repeat(ROWS, 1)
    k_t = None if k is None else torch.full((ROWS,), k)
    p_t = None if p is None else torch.full((ROWS,), p)
    sampled, processed = sampler(logits, {}, k_t, p_t)
    assert processed is None
    return torch.bincount(sampled, minlength=VOCAB).double() / ROWS


@pytest.mark.parametrize(
    ("k", "p"),
    [(None, None), (3, None), (None, 0.7), (5, 0.6), (VOCAB, 1.0), (1, None), (None, 0.05)],
)
def test_matches_vllm_distribution(k, p):
    torch.manual_seed(0)
    freq = _frequencies(HostTopKTopPSampler(), k, p)
    expected = _expected(k, p).double()
    assert torch.allclose(freq, expected, atol=0.01), (freq, expected)
    assert freq[expected == 0].sum() == 0


def test_top_p_rows_beyond_the_candidates_fall_back_to_a_full_sort():
    torch.manual_seed(0)
    # The two most likely tokens hold about 65% of the mass, less than p = 0.9.
    freq = _frequencies(HostTopKTopPSampler(top_p_candidates=(2,)), None, 0.9)
    assert torch.allclose(freq, _expected(None, 0.9).double(), atol=0.01)


def test_top_p_widens_the_candidates_for_rows_the_first_size_does_not_cover():
    torch.manual_seed(0)
    # The two most likely tokens (65% of the mass) cover p = 0.5; p = 0.85 needs the four of the second size (89%).
    logits = LOGITS.repeat(ROWS, 1)
    p = torch.tensor([0.5, 0.85]).repeat(ROWS // 2)
    sampled, _ = HostTopKTopPSampler(top_p_candidates=(2, 4))(logits, {}, None, p)
    for i, p_i in enumerate((0.5, 0.85)):
        rows = sampled[i::2]
        freq = torch.bincount(rows, minlength=VOCAB).double() / rows.numel()
        assert torch.allclose(freq, _expected(None, p_i).double(), atol=0.015), (p_i, freq)


def test_mixed_rows_use_their_own_filters():
    torch.manual_seed(0)
    settings = [(VOCAB, 1.0), (2, 1.0), (VOCAB, 0.5), (4, 0.8)]
    logits = LOGITS.repeat(ROWS, 1)
    k = torch.tensor([s[0] for s in settings]).repeat(ROWS // len(settings))
    p = torch.tensor([s[1] for s in settings]).repeat(ROWS // len(settings))
    sampled, _ = HostTopKTopPSampler()(logits, {}, k, p)
    for i, (k_i, p_i) in enumerate(settings):
        rows = sampled[i :: len(settings)]
        freq = torch.bincount(rows, minlength=VOCAB).double() / rows.numel()
        expected = _expected(None if k_i == VOCAB else k_i, None if p_i == 1.0 else p_i).double()
        assert torch.allclose(freq, expected, atol=0.02), (k_i, p_i, freq, expected)


def test_seeded_rows_repeat_and_ignore_the_global_generator():
    sampler = HostTopKTopPSampler()
    logits = LOGITS.repeat(64, 1)

    def run(global_seed):
        torch.manual_seed(global_seed)
        generators = {i: torch.Generator().manual_seed(1000 + i) for i in range(0, 64, 2)}
        sampled, _ = sampler(logits.clone(), generators, None, torch.full((64,), 0.9))
        return sampled

    first, second = run(1), run(2)
    assert torch.equal(first[0::2], second[0::2])
    assert not torch.equal(first[1::2], second[1::2])


@pytest.mark.parametrize("p_value", [None, 1.0, 0.0, 0.25, 0.5, 0.9])
@pytest.mark.parametrize("kind", ["quantized", "equal", "broad", "few_finite"])
def test_top_k_ties_match_reference_distribution(kind, p_value):
    generator = torch.Generator().manual_seed(17)
    if kind == "quantized":
        logits = torch.randn(4, 128, generator=generator).mul_(2).round_()
    elif kind == "equal":
        logits = torch.zeros(4, 128)
    elif kind == "broad":
        logits = torch.zeros(4, 2048)
        logits[:, :2] = 1
    else:
        logits = torch.full((4, 128), float("-inf"))
        logits[:, :3] = torch.tensor([2.0, 1.0, 1.0])
    k = torch.tensor([1, 2, 7, 15])
    original_k = k.clone()
    p = None if p_value is None else torch.full((4,), p_value)
    expected = apply_top_k_top_p_pytorch(logits.clone(), k.clone(), p).softmax(-1)
    # A deterministic uniform grid measures the entire distribution, not just a few supported samples.
    draws = 8192
    uniforms = (torch.arange(draws) + 0.5) / draws
    for row in range(4):
        actual = _sample_top_k(
            logits[row : row + 1].expand(draws, -1),
            k[row : row + 1].expand(draws),
            None if p is None else p[row : row + 1].expand(draws),
            uniforms,
        )
        freq = torch.bincount(actual, minlength=logits.shape[1]) / draws
        assert torch.allclose(freq, expected[row], atol=2 / draws)
        assert not freq[expected[row] == 0].any()
    assert torch.equal(k, original_k)


def test_quantized_mixed_batch_preserves_each_rows_reference_support():
    generator = torch.Generator().manual_seed(71)
    logits = torch.randn(8, 256, generator=generator).mul_(4).round_()
    logits[0] = torch.arange(256)  # No ties: small candidate fast path.
    logits[1] = 0  # Broad ties: reference top-p mask.
    logits[2, 3:] = float("-inf")  # Nonfinite kth threshold.
    k = torch.tensor([2, 1, 8, 3, 5, 16, 64, 128])
    p = torch.tensor([1.0, 0.5, 1.0, 0.9, 0.25, 1.0, 0.0, 0.75])
    expected = apply_top_k_top_p_pytorch(logits.clone(), k.clone(), p).softmax(-1)
    draws = 4096
    sampled = _sample_top_k(
        logits.repeat_interleave(draws, dim=0),
        k.repeat_interleave(draws),
        p.repeat_interleave(draws),
        ((torch.arange(draws) + 0.5) / draws).repeat(8),
    ).reshape(8, draws)
    for row in range(8):
        freq = torch.bincount(sampled[row], minlength=256) / draws
        assert torch.allclose(freq, expected[row], atol=2 / draws)
        assert not freq[expected[row] == 0].any()


@pytest.mark.parametrize(
    ("p_value", "fallback"), [(0.1, False), (0.8, False), (0.9, False), (0.99, False), (0.95, True)]
)
def test_only_split_nucleus_ties_need_reference_order(monkeypatch, p_value, fallback):
    logits = torch.tensor([[4.0, 3.0, 3.0, 1.0, 1.0, 0.0, -1.0, float("-inf")]])
    p = torch.tensor([p_value])
    expected = apply_top_k_top_p_pytorch(logits.clone(), torch.tensor([7]), p).softmax(-1)[0]
    reference_rows = []

    def reference(values, *args, **kwargs):
        reference_rows.append(values.shape[0])
        return apply_top_k_top_p_pytorch(values, *args, **kwargs)

    monkeypatch.setattr(host_sampling, "apply_top_k_top_p_pytorch", reference)
    draws = 8192
    sampled = _sample_top_k(
        logits.expand(draws, -1), torch.full((draws,), 7), p.expand(draws), (torch.arange(draws) + 0.5) / draws
    )
    freq = torch.bincount(sampled, minlength=8) / draws
    assert torch.allclose(freq, expected, atol=2 / draws)
    assert not freq[expected == 0].any()
    assert bool(reference_rows) == fallback


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_nucleus_near_float32_boundary_uses_reference_mask(monkeypatch, offset):
    logits = torch.tensor([[4.0, 3.0, 3.0, 1.0, 1.0, 0.0, -1.0, float("-inf")]])
    ascending = logits.sort().values.softmax(-1).cumsum(-1)
    p = 1 - ascending[:, 4]  # Immediately before the fully kept score-3 group.
    direction = torch.full_like(p, float("inf") if offset > 0 else float("-inf"))
    if offset:
        p = torch.nextafter(p, direction)
    expected = apply_top_k_top_p_pytorch(logits.clone(), torch.tensor([7]), p).softmax(-1)[0]
    reference_rows = []

    def reference(values, *args, **kwargs):
        reference_rows.append(values.shape[0])
        return apply_top_k_top_p_pytorch(values, *args, **kwargs)

    monkeypatch.setattr(host_sampling, "apply_top_k_top_p_pytorch", reference)
    draws = 8192
    sampled = _sample_top_k(
        logits.expand(draws, -1), torch.full((draws,), 7), p.expand(draws), (torch.arange(draws) + 0.5) / draws
    )
    freq = torch.bincount(sampled, minlength=8) / draws
    assert torch.allclose(freq, expected, atol=2 / draws)
    assert not freq[expected == 0].any()
    assert reference_rows == [draws]


def test_initial_headroom_covers_narrow_kth_ties_in_one_scan_without_mutation(monkeypatch):
    logits = torch.arange(40).div(3, rounding_mode="floor").float().repeat(2, 1)
    original = logits.clone()
    k = torch.tensor([10, 7])
    calls = []
    topk = torch.Tensor.topk

    def tracked_topk(tensor, count, *args, **kwargs):
        calls.append(count)
        return topk(tensor, count, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "topk", tracked_topk)
    HostTopKTopPSampler()(logits, {}, k, torch.ones(2))
    assert calls == [21]
    assert torch.equal(logits, original)
    assert torch.equal(k, torch.tensor([10, 7]))


def test_safe_nucleus_masks_match_reference_across_quantized_rows_and_rounding_boundaries():
    generator = torch.Generator().manual_seed(210)
    logits = torch.randn(512, 257, generator=generator).mul_(8).round_().div_(4)
    k = torch.randint(1, 257, (512,), generator=generator)
    top_k_masked = apply_top_k_top_p_pytorch(logits.clone(), k.clone(), None)
    values, indices = top_k_masked.sort(descending=True)
    ascending = top_k_masked.sort().values.softmax(-1).cumsum(-1)
    positions = torch.randint(0, 256, (512, 1), generator=generator)
    p = torch.rand(512, generator=generator)
    boundaries = (1 - ascending.gather(1, positions).squeeze(1)).clamp_(0, 1)
    p[::3] = boundaries[::3]
    p[1::3] = torch.nextafter(boundaries[1::3], torch.ones_like(boundaries[1::3]))
    num_kept, uncertain = host_sampling._top_k_nucleus(values, p)
    actual = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        1, indices, (torch.arange(257) < num_kept.unsqueeze(1)) & values.isfinite()
    )
    expected = apply_top_k_top_p_pytorch(logits.clone(), k.clone(), p).isfinite()
    assert (~uncertain).any()
    assert torch.equal(actual[~uncertain], expected[~uncertain])


def test_tied_mixed_rows_seeded_generators_are_reproducible():
    logits = torch.tensor([[3.0, 2.0, 2.0, 2.0, 1.0, float("-inf")]]).repeat(4, 1)
    k = torch.tensor([1, 2, 4, 5])
    p = torch.tensor([1.0, 1.0, 0.5, 0.9])
    sampler = HostTopKTopPSampler()

    def run(seed):
        torch.manual_seed(seed)
        generators = {row: torch.Generator().manual_seed(42 + row) for row in range(4)}
        return [sampler(logits.clone(), generators, k, p)[0] for _ in range(10)]

    assert torch.equal(torch.stack(run(1)), torch.stack(run(2)))
    assert torch.equal(k, torch.tensor([1, 2, 4, 5]))


def test_processed_logprobs_modes_keep_vllms_sampler():
    sampler = HostTopKTopPSampler(logprobs_mode="processed_logprobs")
    assert sampler.forward != sampler.forward_host
    _, processed = sampler(LOGITS.repeat(4, 1), {}, None, None)
    assert processed is not None
