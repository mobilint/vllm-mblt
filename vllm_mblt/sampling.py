"""Random sampling for logits that live on the host.

vLLM's PyTorch sampler draws one exponential variate per vocabulary entry and row (Gumbel-max), and applies top-p
by sorting the whole vocabulary. Both are cheap on a GPU and slow on a CPU: for 48 rows of a 151,936-entry
vocabulary the exponential draw alone takes 100-200 ms, against about 60 ms for the NPU forward.

``HostTopKTopPSampler`` draws the same distribution with one uniform variate per row (inverse transform sampling
over the cumulative probabilities), and finds the top-k and top-p candidates with ``topk`` instead of a full sort.
"""

from typing import Optional

import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p_pytorch

# Top-p without top-k looks at this many most likely tokens, widening while the candidates hold less than p of the
# probability mass, and sorts the whole vocabulary for rows that even the last size does not cover. On 48 rows of a
# 151,936-entry vocabulary, topk of 1024 takes about 15 ms, of 16384 about 30 ms, and the sort about 140 ms.
TOP_P_CANDIDATES = (1024, 16384)


class HostTopKTopPSampler(TopKTopPSampler):
    """``TopKTopPSampler`` for host logits: same distribution, without per-token noise or a full-vocabulary sort.

    Tokens tied with the k-th largest logit are cut at exactly k; vLLM keeps every tied token. Sampled tokens are
    not bit-identical to vLLM's for a given seed, only identically distributed, as vLLM's own GPU kernels are.
    """

    def __init__(
        self, logprobs_mode: LogprobsMode = "raw_logits", top_p_candidates: tuple[int, ...] = TOP_P_CANDIDATES
    ) -> None:
        super().__init__(logprobs_mode)
        self.top_p_candidates = top_p_candidates
        # vLLM's processed-logprobs modes return the masked logits, which this sampler never builds.
        if logprobs_mode not in PROCESSED_LOGPROBS_MODES:
            self.forward = self.forward_host

    def forward_host(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, None]:
        num_rows, vocab_size = logits.shape
        uniform = _uniform(num_rows, generators)
        sampled = torch.empty(num_rows, dtype=torch.long)
        no_filter = torch.zeros(num_rows, dtype=torch.bool)
        has_k = no_filter if k is None else k < vocab_size
        has_p = no_filter if p is None else p < 1.0

        rows = has_k.nonzero().flatten()
        if rows.numel():
            sampled[rows] = _sample_top_k(logits[rows], k[rows], None if p is None else p[rows], uniform[rows])

        rows = (has_p & ~has_k).nonzero().flatten()
        if rows.numel():
            sampled[rows] = self._sample_top_p(logits[rows], p[rows], uniform[rows])

        rows = (~has_p & ~has_k).nonzero().flatten()
        if rows.numel():
            sampled[rows] = _sample_full(logits[rows], uniform[rows])
        return sampled, None

    def _sample_top_p(self, logits: torch.Tensor, p: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
        sampled = torch.empty(logits.shape[0], dtype=torch.long)
        pending = torch.arange(logits.shape[0])
        log_total = logits.logsumexp(dim=-1, keepdim=True)
        for num_candidates in self.top_p_candidates:
            if num_candidates >= logits.shape[1] or not pending.numel():
                break
            values, indices = logits[pending].topk(num_candidates, dim=-1)
            cumulative = (values - log_total[pending]).exp_().cumsum_(dim=-1)
            # Every token outside the candidates has at least their total mass ahead of it, so when that total
            # reaches p no such token survives top-p and the candidates decide the row exactly.
            covered = cumulative[:, -1] >= p[pending]
            rows = pending[covered]
            if rows.numel():
                sampled[rows] = _sample_candidates(cumulative[covered], indices[covered], p[rows], uniform[rows])
            pending = pending[~covered]
        if pending.numel():
            masked = apply_top_k_top_p_pytorch(logits[pending].clone(), None, p[pending])
            sampled[pending] = _inverse_cdf(masked.softmax(dim=-1).cumsum_(dim=-1), uniform[pending])
        return sampled


def _uniform(num_rows: int, generators: dict[int, torch.Generator]) -> torch.Tensor:
    uniform = torch.rand(num_rows)
    for row, generator in generators.items():
        uniform[row] = torch.rand(1, generator=generator)
    return uniform


def _sample_top_k(
    logits: torch.Tensor, k: torch.Tensor, p: Optional[torch.Tensor], uniform: torch.Tensor
) -> torch.Tensor:
    values, indices = logits.topk(int(k.max()), dim=-1)
    beyond_k = torch.arange(values.shape[1]) >= k.unsqueeze(1)
    cumulative = values.masked_fill_(beyond_k, float("-inf")).softmax(dim=-1).cumsum_(dim=-1)
    return _sample_candidates(cumulative, indices, p, uniform)


def _sample_candidates(
    cumulative: torch.Tensor, indices: torch.Tensor, p: Optional[torch.Tensor], uniform: torch.Tensor
) -> torch.Tensor:
    """Sample among candidates sorted by descending probability, given their cumulative probabilities."""
    num_kept = None
    if p is not None:
        # vLLM drops a token when the ascending cumulative mass up to and including it is at most 1 - p, that is
        # when the mass of the more likely tokens is at least p. That mass is the previous candidate's cumulative
        # probability, so the first candidate stays and so does every one after a cumulative probability below p.
        num_kept = torch.searchsorted(cumulative, p.unsqueeze(1)).squeeze(1).add_(1).clamp_(max=cumulative.shape[1])
    return indices.gather(1, _inverse_cdf(cumulative, uniform, num_kept).unsqueeze(1)).squeeze(1)


def _sample_full(logits: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    return _inverse_cdf(logits.softmax(dim=-1).cumsum_(dim=-1), uniform)


def _inverse_cdf(
    cumulative: torch.Tensor, uniform: torch.Tensor, num_kept: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Index of the first entry whose cumulative weight exceeds ``uniform`` times the total of the kept entries.

    Only the first ``num_kept`` entries of each row are kept (all when None). Entries of weight 0 are never picked:
    their cumulative weight equals the previous entry's.
    """
    if num_kept is None:
        num_kept = torch.full((cumulative.shape[0],), cumulative.shape[1])
    last_kept = (num_kept - 1).unsqueeze(1)
    target = uniform.unsqueeze(1) * cumulative.gather(1, last_kept)
    index = torch.searchsorted(cumulative, target, right=True)
    return torch.minimum(index, last_kept).squeeze(1)
