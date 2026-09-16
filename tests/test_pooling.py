from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.model_executor.models.interfaces import supports_cross_encoding, supports_score_template
from vllm.model_executor.models.interfaces_base import is_pooling_model

from vllm_mblt.mblt_platform import MbltPlatform
from vllm_mblt.models.modeling_pooling import (
    QWEN_RERANK_SUFFIX,
    MobilintEmbeddingModel,
    MobilintForSequenceClassification,
    MobilintLastTokenEmbeddingModel,
    MobilintQwen3ForSequenceClassification,
)
from vllm_mblt.pooling import artifact_file, pool_output
from vllm_mblt.pooling_worker import MbltPoolingWorker


def params(task="embed", **kwargs):
    return SimpleNamespace(task=task, dimensions=None, normalize=True, use_activation=True, **kwargs)


def test_model_registry_interfaces():
    for model in (
        MobilintEmbeddingModel,
        MobilintLastTokenEmbeddingModel,
        MobilintForSequenceClassification,
        MobilintQwen3ForSequenceClassification,
    ):
        assert is_pooling_model(model)
    assert not supports_cross_encoding(MobilintEmbeddingModel)
    assert supports_cross_encoding(MobilintForSequenceClassification)
    assert supports_score_template(MobilintQwen3ForSequenceClassification)


def test_mean_and_last_pooling_differ_and_normalize():
    x = np.array([[[3.0, 0.0], [0.0, 4.0]]])
    assert torch.allclose(pool_output(x, "mean", params(), hidden_size=2), torch.tensor([0.6, 0.8]))
    assert torch.equal(pool_output(x, "last", params(), hidden_size=2), torch.tensor([0.0, 1.0]))


def test_matryoshka_truncates_before_normalizing():
    p = params()
    p.dimensions = 2
    result = pool_output([3, 4, 12], "last", p, hidden_size=3, matryoshka=True)
    assert torch.allclose(result, torch.tensor([0.6, 0.8]))
    with pytest.raises(ValueError):
        pool_output([3, 4, 12], "last", p, hidden_size=3)


def test_no_yes_softmax_and_activation_override():
    p = params("score")
    value = pool_output([-1000, -998], "yes_no", p, hidden_size=1024)
    assert value.item() == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    p.use_activation = False
    assert pool_output([-1000, -998], "yes_no", p, hidden_size=1024).item() == 2
    assert pool_output([-2], "scalar", p, hidden_size=1024).item() == -2


def test_reject_invalid_outputs():
    with pytest.raises(RuntimeError, match="non-finite"):
        pool_output([float("nan")], "scalar", params("score"), hidden_size=1)
    with pytest.raises(RuntimeError, match="no, yes"):
        pool_output([1, 2, 3], "yes_no", params("score"), hidden_size=1)
    with pytest.raises(ValueError):
        pool_output([1], "mean", params("score"), hidden_size=1)


def test_rerank_template_keeps_query_document_and_suffix():
    prompt = MobilintQwen3ForSequenceClassification.get_score_template("a query", "a document")
    assert "<Query>: a query\n<Document>: a document" in prompt
    assert prompt.endswith(QWEN_RERANK_SUFFIX)


def test_artifact_cannot_escape_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (tmp_path / "outside").write_text("data")
    with pytest.raises(ValueError):
        artifact_file(package, "../outside")


def test_pooling_scheduler_disables_partial_prompts_and_prefix_cache():
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="pooling", max_model_len=512, hf_config=SimpleNamespace(mblt_pooling="mean")
        ),
        parallel_config=SimpleNamespace(world_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True, chunked_prefill_enabled=True, max_num_batched_tokens=128
        ),
    )
    MbltPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls.endswith("MbltPoolingWorker")
    assert not cfg.cache_config.enable_prefix_caching
    assert cfg.cache_config.block_size == 128
    assert not cfg.scheduler_config.enable_chunked_prefill
    assert cfg.scheduler_config.max_num_batched_tokens == 512
    cfg.model_config.hf_config.is_matryoshka = True
    cfg.model_config.hf_config.hidden_size = 1024
    MbltPlatform.check_and_update_config(cfg)
    assert 256 in cfg.model_config.hf_config.matryoshka_dimensions
    assert 0 not in cfg.model_config.hf_config.matryoshka_dimensions
    assert 1025 not in cfg.model_config.hf_config.matryoshka_dimensions


def test_worker_variable_length_batch_keeps_request_order():
    worker = object.__new__(MbltPoolingWorker)
    worker.requests = {}
    worker.model = SimpleNamespace(encode=lambda ids, p: torch.tensor([sum(ids)]))

    def request(rid, ids):
        return SimpleNamespace(
            req_id=rid,
            prompt_token_ids=ids,
            pooling_params=params(),
            prompt_embeds=None,
            mm_features=[],
            lora_request=None,
            num_computed_tokens=0,
        )

    schedule = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[request("a", [1]), request("b", [2, 3])],
        num_scheduled_tokens={"b": 2, "a": 1},
    )
    result = worker.execute_model(schedule)
    assert result.req_ids == ["b", "a"]
    assert result.sampled_token_ids == []
    assert [v.item() for v in result.pooler_output] == [5, 1]
    assert not worker.requests
    schedule.scheduled_new_reqs = [request("a", [1, 2])]
    schedule.num_scheduled_tokens = {"a": 1}
    with pytest.raises(ValueError, match="full-sequence"):
        worker.execute_model(schedule)
