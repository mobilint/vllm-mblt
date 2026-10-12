"""EAGLE-3 serving: the runner's step contract with vLLM's scheduler, the platform's limits, and registration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import ModelRegistry

from tests.test_mblt_platform_prefill import _make_vllm_config
from vllm_mblt import register_model
from vllm_mblt.eagle3_runtime import Eagle3Runner, is_eagle3_model, make_logits_processor
from vllm_mblt.mblt_platform import MbltPlatform, resolve_model_max_batch_size
from vllm_mblt.mblt_worker import MbltWorker


class _FakeCache:
    def __init__(self, index: int) -> None:
        self.index = index


class _FakeEagle3Model:
    """Settles ``rounds[i]`` tokens per round: the accepted drafts, then the next root (``root + 1``)."""

    def __init__(self, accepted_per_round: tuple[int, ...] = (2,)) -> None:
        self.eagle3_base_model = object()
        self.accepted_per_round = list(accepted_per_round)
        self.configured = 0
        self.prefills: list[dict] = []
        self.steps: list[dict] = []
        self.roots: dict[int, int] = {}

    def configure_eagle3_tree(self) -> None:
        self.configured += 1

    def eagle3_prefill(self, input_ids, cache, *, logits_processor=None, max_new_tokens=None):
        self.prefills.append(
            {"ids": input_ids.tolist()[0], "cache": cache, "lp": logits_processor, "max": max_new_tokens}
        )
        self.roots[cache.index] = 100 + cache.index * 1000
        return torch.tensor([[self.roots[cache.index]]])

    def eagle3_decode_step(self, cache, *, logits_processor=None, max_new_tokens=None, eos_token_id=None):
        self.steps.append({"cache": cache, "max": max_new_tokens, "eos": eos_token_id})
        root = self.roots[cache.index]
        accepted = self.accepted_per_round[min(len(self.steps) - 1, len(self.accepted_per_round) - 1)]
        tokens = [root] + [root + 1 + i for i in range(accepted)]
        stopped = len(tokens) >= max_new_tokens
        tokens = tokens[:max_new_tokens]
        next_token = None if stopped else torch.tensor([[tokens[-1] + 1]])
        if next_token is not None:
            self.roots[cache.index] = tokens[-1] + 1
        return SimpleNamespace(tokens=torch.tensor([tokens]), next_token=next_token)


def _runner(model: _FakeEagle3Model, max_model_len: int = 4096) -> Eagle3Runner:
    runner = Eagle3Runner(model, max_model_len=max_model_len)
    counter = iter(range(100))
    runner._new_cache = lambda: _FakeCache(next(counter))
    return runner


def _new_req(req_id, prompt, *, computed=0, temperature=0.0, max_tokens=16, **extra):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=extra.get("mm_features", []),
        prompt_embeds=extra.get("prompt_embeds"),
        sampling_params=SimpleNamespace(
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            max_tokens=max_tokens,
            stop_token_ids=extra.get("stop_token_ids", []),
            ignore_eos=extra.get("ignore_eos", False),
            all_stop_token_ids=extra.get("all_stop_token_ids", set()),
        ),
        num_computed_tokens=computed,
    )


def _step(scheduled, *, new=(), cached=(), finished=(), resumed=(), all_token_ids=None):
    """``cached`` lists (req_id, num_computed_tokens)."""
    return SimpleNamespace(
        finished_req_ids=set(finished),
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[req_id for req_id, _ in cached],
            num_computed_tokens=[computed for _, computed in cached],
            resumed_req_ids=set(resumed),
            all_token_ids=all_token_ids or {},
        ),
        num_scheduled_tokens=dict(scheduled),
    )


def _sampled(output, req_id):
    return output.sampled_token_ids[output.req_id_to_index[req_id]].tolist()


def test_a_prompt_is_fed_once_vllm_has_scheduled_all_of_it_and_returns_the_first_token() -> None:
    model = _FakeEagle3Model()
    runner = _runner(model)
    prompt = list(range(10))

    first = runner.execute(_step({"a": 6}, new=[_new_req("a", prompt)]))
    second = runner.execute(_step({"a": 4}, cached=[("a", 6)]))

    assert _sampled(first, "a") == []
    assert _sampled(second, "a") == [100]
    assert model.prefills == [{"ids": prompt, "cache": runner.requests["a"].cache, "lp": None, "max": 16}]
    assert model.configured == 1


def test_each_step_returns_the_accepted_drafts_and_the_next_token() -> None:
    model = _FakeEagle3Model(accepted_per_round=(2, 0))
    runner = _runner(model)
    runner.execute(_step({"a": 3}, new=[_new_req("a", [1, 2, 3], max_tokens=10)]))

    # vLLM schedules the token it got last step; the round already holds it.
    round1 = runner.execute(_step({"a": 1}, cached=[("a", 3)]))
    round2 = runner.execute(_step({"a": 3}, cached=[("a", 4)]))

    assert _sampled(round1, "a") == [101, 102, 103]
    assert _sampled(round2, "a") == [104]
    assert runner.requests["a"].token_ids == [1, 2, 3, 100, 101, 102, 103, 104]
    # The cap counts the round's root, which the previous step returned: max_tokens - outputs + 1.
    assert [step["max"] for step in model.steps] == [10 - 1 + 1, 10 - 4 + 1]
    assert all(step["eos"] is None for step in model.steps)


def test_rounds_stop_at_the_tokens_that_finish_the_request_in_vllm() -> None:
    model = _FakeEagle3Model()
    runner = _runner(model)
    runner.execute(
        _step(
            {"a": 1, "b": 1},
            new=[
                _new_req("a", [1], stop_token_ids=[7], all_stop_token_ids={7, 2}),
                _new_req("b", [1], ignore_eos=True, stop_token_ids=[7], all_stop_token_ids={7, 2}),
            ],
        )
    )

    runner.execute(_step({"a": 1, "b": 1}, cached=[("a", 1), ("b", 1)]))

    # ignore_eos keeps the explicit stop tokens but drops the model's EOS.
    assert [step["eos"] for step in model.steps] == [[2, 7], [7]]


def test_a_round_split_across_steps_runs_once_all_its_tokens_are_scheduled() -> None:
    model = _FakeEagle3Model(accepted_per_round=(2,))
    runner = _runner(model)
    runner.execute(_step({"a": 3}, new=[_new_req("a", [1, 2, 3])]))
    runner.execute(_step({"a": 1}, cached=[("a", 3)]))  # returns 3 tokens

    partial = runner.execute(_step({"a": 2}, cached=[("a", 4)]))
    rest = runner.execute(_step({"a": 1}, cached=[("a", 6)]))

    assert _sampled(partial, "a") == []
    assert _sampled(rest, "a") == [104, 105, 106]
    assert len(model.steps) == 2


def test_the_token_limit_caps_the_round_and_ends_the_request() -> None:
    model = _FakeEagle3Model(accepted_per_round=(5,))
    runner = _runner(model)
    runner.execute(_step({"a": 2}, new=[_new_req("a", [1, 2], max_tokens=3)]))

    last = runner.execute(_step({"a": 1}, cached=[("a", 2)]))

    assert _sampled(last, "a") == [101, 102]
    assert runner.requests["a"].exhausted
    with pytest.raises(RuntimeError, match="after it reached its token limit"):
        runner.execute(_step({"a": 2}, cached=[("a", 3)]))


def test_the_model_length_caps_the_round() -> None:
    model = _FakeEagle3Model(accepted_per_round=(5,))
    runner = _runner(model, max_model_len=6)
    runner.execute(_step({"a": 3}, new=[_new_req("a", [1, 2, 3], max_tokens=100)]))

    last = runner.execute(_step({"a": 1}, cached=[("a", 3)]))

    assert model.prefills[0]["max"] == 3
    assert model.steps[0]["max"] == 6 - 4 + 1
    assert _sampled(last, "a") == [101, 102]


def test_requests_keep_their_own_caches_and_finished_ones_are_dropped() -> None:
    model = _FakeEagle3Model()
    runner = _runner(model)
    runner.execute(_step({"a": 2, "b": 2}, new=[_new_req("a", [1, 2]), _new_req("b", [7, 8])]))

    out = runner.execute(_step({"a": 1, "b": 1}, cached=[("a", 2), ("b", 2)]))

    assert _sampled(out, "a") == [101, 102, 103]
    assert _sampled(out, "b") == [1101, 1102, 1103]
    assert [step["cache"].index for step in model.steps] == [0, 1]
    runner.execute(_step({}, finished=["a"]))
    assert list(runner.requests) == ["b"]


def test_a_preempted_request_starts_over_from_the_tokens_vllm_kept() -> None:
    model = _FakeEagle3Model()
    runner = _runner(model)
    runner.execute(_step({"a": 2}, new=[_new_req("a", [1, 2])]))
    runner.execute(_step({"a": 1}, cached=[("a", 2)]))

    resumed = runner.execute(
        _step({"a": 5}, cached=[("a", 0)], resumed=["a"], all_token_ids={"a": [1, 2, 100, 101, 102]})
    )

    assert model.prefills[-1]["ids"] == [1, 2, 100, 101, 102]
    assert model.prefills[-1]["cache"].index == 1
    assert _sampled(resumed, "a") == [1100]


def test_prompt_embeddings_and_multimodal_inputs_are_refused() -> None:
    runner = _runner(_FakeEagle3Model())

    with pytest.raises(ValueError, match="token prompts only"):
        runner.execute(_step({"a": 1}, new=[_new_req("a", [1], prompt_embeds=torch.zeros(1, 4))]))
    with pytest.raises(ValueError, match="token prompts only"):
        runner.execute(_step({"a": 1}, new=[_new_req("a", [1], mm_features=[object()])]))


def test_runner_needs_the_round_api() -> None:
    with pytest.raises(RuntimeError, match="eagle3_decode_step"):
        Eagle3Runner(SimpleNamespace(eagle3_base_model=object()), max_model_len=16)


def test_sampling_params_map_to_the_eagle3_logits_processor(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "vllm_mblt.eagle3_runtime.prepare_logits_processor", lambda **kwargs: calls.append(kwargs) or "lp"
    )

    assert make_logits_processor(SimpleNamespace(temperature=0.6, top_p=0.9, top_k=20)) == "lp"
    make_logits_processor(SimpleNamespace(temperature=1.0, top_p=1.0, top_k=-1))

    # vLLM disables top-k with 0 or -1; transformers-mblt with 0.
    assert calls == [
        {"temperature": 0.6, "top_p": 0.9, "top_k": 20},
        {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
    ]


def test_execute_model_hands_eagle3_steps_to_the_runner() -> None:
    worker = MbltWorker.__new__(MbltWorker)
    worker.model = object()
    worker.input_embeddings = object()
    sentinel = object()
    worker.eagle3_runner = SimpleNamespace(execute=lambda scheduler_output: (sentinel, scheduler_output))

    assert worker.execute_model("step") == (sentinel, "step")


def test_load_model_serves_an_eagle3_release_through_the_runner(monkeypatch) -> None:
    worker = MbltWorker.__new__(MbltWorker)
    worker.max_seq_len = 2048
    worker.rank = 0
    worker.local_rank = 0
    worker.model = None
    worker.cache_model = None
    worker.eagle3_runner = None
    worker.load_config = SimpleNamespace(model_loader_extra_config={})
    worker.model_config = SimpleNamespace(
        model="mobilint/EAGLE3-JPharmatron-7B",
        hf_config=SimpleNamespace(model_type="mobilint-qwen2-eagle3"),
        model_kwargs={},
        hf_overrides={},
    )
    worker.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(model_loader_extra_config={}), model_config=worker.model_config
    )
    model = _FakeEagle3Model()
    model.eval = lambda: None
    model.get_input_embeddings = lambda: "embeddings"
    monkeypatch.setattr("vllm_mblt.mblt_worker.AutoModelForCausalLM.from_pretrained", lambda *a, **k: model)

    worker.load_model()

    assert is_eagle3_model(model)
    assert worker.eagle3_runner.model is model
    assert worker.eagle3_runner.max_model_len == 2048
    assert worker.input_embeddings == "embeddings"
    worker.check_health()


def _eagle3_vllm_config(base_max_batch_size=1, **kwargs):
    hf_config = SimpleNamespace(
        model_type="mobilint-qwen2-eagle3",
        npu_prefill_chunk_size=None,
        base_max_batch_size=base_max_batch_size,
        text_config=None,
        core_mode=None,
    )
    return _make_vllm_config(None, hf_config=hf_config, **kwargs)


def test_platform_serves_eagle3_one_sequence_at_a_time_without_prefix_caching() -> None:
    config = _eagle3_vllm_config(scheduler_max_num_seqs=8)

    MbltPlatform.check_and_update_config(config)

    assert resolve_model_max_batch_size(config) == 1
    assert config.scheduler_config.max_num_seqs == 1
    assert config.cache_config.enable_prefix_caching is False
    with pytest.raises(ValueError, match="one sequence per base MXQ"):
        MbltPlatform.check_and_update_config(_eagle3_vllm_config(base_max_batch_size=4))


def test_platform_refuses_sampling_options_eagle3_rounds_cannot_honor() -> None:
    from vllm import SamplingParams

    MbltPlatform.check_and_update_config(_eagle3_vllm_config())
    try:
        MbltPlatform.validate_request("p", SamplingParams(temperature=0.6, top_p=0.9, top_k=20), {"type": "token"})
        # vLLM fans n > 1 out into child requests with their own ids, so each candidate gets its own cache.
        MbltPlatform.validate_request("p", SamplingParams(n=3, temperature=0.8), {"type": "token"})
        with pytest.raises(ValueError, match="unsupported here: logprobs, repetition_penalty, seed"):
            MbltPlatform.validate_request(
                "p", SamplingParams(logprobs=1, repetition_penalty=1.1, seed=3), {"type": "token"}
            )
        with pytest.raises(ValueError, match="token prompts only"):
            MbltPlatform.validate_request("p", SamplingParams(), {"type": "embeds"})
    finally:
        MbltPlatform.check_and_update_config(_make_vllm_config(128))
    # Other models keep every option.
    MbltPlatform.validate_request("p", SamplingParams(logprobs=1, seed=3), {"type": "token"})


class _Processor:
    """Stands in for vLLM's Processor, which calls validate_request with its engine's model_config at hand."""

    def __init__(self, model_type: str) -> None:
        self.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))

    def process_inputs(self, params) -> None:
        MbltPlatform.validate_request(prompt="p", params=params, processed_inputs={"type": "token"})


def test_validation_follows_the_engine_that_processes_the_request() -> None:
    from vllm import SamplingParams

    eagle3, dense = _Processor("mobilint-qwen2-eagle3"), _Processor("mobilint-qwen2")
    penalized = SamplingParams(repetition_penalty=1.1, logprobs=1)
    for configured_last in (_eagle3_vllm_config(), _make_vllm_config(128)):
        MbltPlatform.check_and_update_config(configured_last)
        dense.process_inputs(penalized)
        with pytest.raises(ValueError, match="repetition_penalty"):
            eagle3.process_inputs(penalized)
    MbltPlatform.check_and_update_config(_make_vllm_config(128))


def test_register_model_registers_the_eagle3_architectures_by_import_path() -> None:
    register_model()

    for architecture in (
        "MobilintLlamaEagle3ForCausalLM",
        "MobilintQwen2Eagle3ForCausalLM",
        "MobilintQwen3Eagle3ForCausalLM",
    ):
        registered = ModelRegistry.models[architecture]
        assert registered.module_name == "vllm_mblt.models.modeling_eagle3"
        assert registered.class_name == architecture


def test_sampled_token_arrays_are_int64() -> None:
    model = _FakeEagle3Model()
    runner = _runner(model)
    out = runner.execute(_step({"a": 1}, new=[_new_req("a", [5])]))
    assert out.sampled_token_ids[0].dtype == np.int64
