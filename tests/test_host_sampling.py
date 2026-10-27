import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_mblt.sampling import HostTopKTopPSampler

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


def test_processed_logprobs_modes_keep_vllms_sampler():
    sampler = HostTopKTopPSampler(logprobs_mode="processed_logprobs")
    assert sampler.forward != sampler.forward_host
    _, processed = sampler(LOGITS.repeat(4, 1), {}, None, None)
    assert processed is not None
