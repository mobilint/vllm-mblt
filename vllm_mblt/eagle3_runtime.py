"""EAGLE-3 serving through transformers-mblt's speculative rounds.

vLLM schedules a decoding request one token per step. An EAGLE-3 round settles several: the tokens the base model
accepted from the draft tree, then the next token sampled after them. The runner returns them all as the request's
sampled tokens, vLLM appends them (cutting them at a stop condition), and the next step schedules them as the
request's new tokens. The round already fed them to the request's ``MobilintEagle3Cache``, so that step runs the
next round directly.

Each request keeps its own cache, which also holds its draft tree between rounds, so requests' rounds can
interleave. The runner keeps vLLM's view of each request's tokens and feeds the prompt once the scheduler has
scheduled all of it; chunks before that only advance vLLM's count.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from transformers_mblt.utils.cache_utils import MobilintEagle3Cache
from transformers_mblt.utils.eagle3.decoding import prepare_logits_processor
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput


def is_eagle3_model(model: Any) -> bool:
    """Whether ``model`` is a transformers-mblt EAGLE-3 model (it has a base backend)."""
    return hasattr(model, "eagle3_base_model")


def stop_token_ids(sampling_params: Any) -> Optional[list[int]]:
    """The token ids that finish the request in vLLM's scheduler, for a round to stop at; ``None`` when there are none.

    vLLM stops on ``stop_token_ids`` and, unless ``ignore_eos``, on the model's EOS ids, which
    ``all_stop_token_ids`` adds.
    """
    ids = set(getattr(sampling_params, "stop_token_ids", None) or ())
    if not getattr(sampling_params, "ignore_eos", False):
        ids.update(getattr(sampling_params, "all_stop_token_ids", None) or ())
    return sorted(ids) or None


def make_logits_processor(sampling_params: Any) -> Optional[Any]:
    """The EAGLE-3 logits processor for a request: ``None`` (greedy) at temperature 0."""
    top_k = int(getattr(sampling_params, "top_k", 0) or 0)
    return prepare_logits_processor(
        temperature=float(sampling_params.temperature),
        top_p=float(sampling_params.top_p),
        top_k=max(0, top_k),
    )


@dataclass
class Eagle3Request:
    token_ids: list[int]
    prompt_len: int
    max_tokens: int
    logits_processor: Optional[Any]
    # A round ends at these ids; vLLM finishes the request on them anyway.
    stop_token_ids: Optional[list[int]] = None
    num_computed_tokens: int = 0
    # None until the prompt is fed.
    cache: Optional[MobilintEagle3Cache] = None
    # A round reached a stop token, ``max_tokens`` or the model length; vLLM finishes the request with it.
    exhausted: bool = False


class Eagle3Runner:
    def __init__(self, model: Any, *, max_model_len: int) -> None:
        if not callable(getattr(model, "eagle3_decode_step", None)):
            raise RuntimeError(
                "Serving EAGLE-3 needs transformers-mblt 0.5.0 or newer, which adds eagle3_prefill / "
                "eagle3_decode_step."
            )
        self.model = model
        self.max_model_len = int(max_model_len)
        # The draft tree size from the release's generation_config.json, as generate() resolves it.
        model.configure_eagle3_tree()
        self.requests: dict[str, Eagle3Request] = {}

    def _new_cache(self) -> MobilintEagle3Cache:
        cache = MobilintEagle3Cache(*self.model.get_cache_mxq_models())
        cache.reset()
        return cache

    def _add_request(self, new_req: Any) -> None:
        if new_req.prompt_embeds is not None or new_req.mm_features:
            raise ValueError("EAGLE-3 models take token prompts only, without prompt embeddings or multimodal data.")
        prompt = list(new_req.prompt_token_ids or [])
        sampling_params = new_req.sampling_params
        max_tokens = sampling_params.max_tokens
        self.requests[new_req.req_id] = Eagle3Request(
            token_ids=prompt,
            prompt_len=len(prompt),
            max_tokens=self.max_model_len - len(prompt) if max_tokens is None else int(max_tokens),
            logits_processor=make_logits_processor(sampling_params),
            stop_token_ids=stop_token_ids(sampling_params),
            num_computed_tokens=int(new_req.num_computed_tokens),
        )

    def _remaining_tokens(self, req: Eagle3Request) -> int:
        """How many more tokens the request may receive: within ``max_tokens`` and the model length."""
        num_output = len(req.token_ids) - req.prompt_len
        return min(req.max_tokens - num_output, self.max_model_len - len(req.token_ids))

    def _run(self, req_id: str, req: Eagle3Request) -> list[int]:
        """Feed the prompt or run the next round; return the tokens it settles."""
        if req.exhausted:
            raise RuntimeError(f"EAGLE-3 request {req_id} was scheduled after it reached its token limit.")
        remaining = self._remaining_tokens(req)
        if req.cache is None:
            req.cache = self._new_cache()
            # Like vLLM's own runner, prefill samples one token even without room for it (a prompt that fills the
            # model length); the scheduler then caps the request at the limit.
            root = self.model.eagle3_prefill(
                torch.tensor([req.token_ids], dtype=torch.long),
                req.cache,
                logits_processor=req.logits_processor,
                max_new_tokens=max(1, remaining),
            )
            return [int(root)]
        # The round's first token is the root the previous step already returned; the cap counts it.
        step = self.model.eagle3_decode_step(
            req.cache,
            logits_processor=req.logits_processor,
            max_new_tokens=remaining + 1,
            eos_token_id=req.stop_token_ids,
        )
        tokens = step.tokens[0, 1:].tolist()
        if step.next_token is None:
            req.exhausted = True
        else:
            tokens.append(int(step.next_token))
        return tokens

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        for new_req in scheduler_output.scheduled_new_reqs:
            self._add_request(new_req)
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            req = self.requests[req_id]
            req.num_computed_tokens = int(cached.num_computed_tokens[i])
            if req_id in cached.resumed_req_ids:
                # Preempted: its cache was dropped with its blocks. Start over from the tokens vLLM kept.
                req.token_ids = list(cached.all_token_ids[req_id])
                req.cache = None
                req.exhausted = False

        req_ids: list[str] = []
        sampled_token_ids: list[np.ndarray] = []
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            req = self.requests[req_id]
            scheduled_end = req.num_computed_tokens + int(num_scheduled)
            tokens: list[int] = []
            # Until vLLM has scheduled every token it knows of (a prompt in chunks, or a round's tokens split across
            # steps), there is nothing to run: the cache gets them all at once.
            if scheduled_end >= len(req.token_ids):
                tokens = self._run(req_id, req)
                req.token_ids.extend(tokens)
            req.num_computed_tokens = scheduled_end
            req_ids.append(req_id)
            sampled_token_ids.append(np.asarray(tokens, dtype=np.int64))

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
