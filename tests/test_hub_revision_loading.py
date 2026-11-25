"""MbltWorker.load_model against a local Hub cache whose branches ship an MXQ under the same name.

transformers, transformers-mblt and the NPU backend's MXQ path resolution run unchanged, offline, against the cache;
only the backend's qbruntime accelerator and model creation is replaced, since that needs an NPU.
"""

import json
from types import SimpleNamespace

import huggingface_hub.constants as hub_constants
import pytest
import torch
from mblt_npu.npu_backend import MobilintNPUBackend
from safetensors.torch import save_file
from transformers_mblt.models.qwen2.modeling_qwen2 import MobilintQwen2ForCausalLM  # registers the Auto classes

from vllm_mblt.mblt_worker import MbltWorker
from vllm_mblt.runtime_cache import MbltRuntimeCacheManager

REPO_ID = "mobilint/Example-Qwen2"
MXQ_NAME = "Example-Qwen2.mxq"
COMMITS = {"main": "a" * 40, "bs16": "b" * 40}
MAX_BATCH_SIZES = {"main": 1, "bs16": 16}


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """A Hub cache holding ``main`` and ``bs16``, each with its own config, weights and same-name MXQ."""
    repo_dir = tmp_path / f"models--{REPO_ID.replace('/', '--')}"
    (repo_dir / "refs").mkdir(parents=True)
    for branch, commit in COMMITS.items():
        (repo_dir / "refs" / branch).write_text(commit)
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents=True)
        config = {
            "model_type": "mobilint-qwen2",
            "architectures": ["MobilintQwen2ForCausalLM"],
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "mxq_path": MXQ_NAME,
            "max_batch_size": MAX_BATCH_SIZES[branch],
        }
        (snapshot / "config.json").write_text(json.dumps(config))
        # Each branch's host embedding is filled with its batch size, so the test can tell which branch it came from.
        weights = {"embed_tokens.weight": torch.full((8, 4), float(MAX_BATCH_SIZES[branch]))}
        save_file(weights, str(snapshot / "model.safetensors"))
        (snapshot / MXQ_NAME).write_text(branch)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    return tmp_path


@pytest.fixture
def resolved_mxq_paths(monkeypatch):
    """The MXQ paths the backend resolves when it creates its models, in place of loading them on an NPU."""
    paths = []
    monkeypatch.setattr(MobilintNPUBackend, "create", lambda self: paths.append(self.check_model_path(self.mxq_path)))
    monkeypatch.setattr(MobilintNPUBackend, "launch", lambda self: None)
    monkeypatch.setattr(MobilintQwen2ForCausalLM, "get_cache_mxq_model", lambda self: SimpleNamespace())
    return paths


def _make_worker(revision):
    worker = MbltWorker.__new__(MbltWorker)
    worker.rank = 0
    worker.local_rank = 0
    worker.model = None
    worker.cache_model = None
    worker.max_seq_len = 2048
    worker.max_batch_size = 1
    worker.runtime_cache = MbltRuntimeCacheManager(max_batch_size=1, block_size=128)
    worker.load_config = SimpleNamespace(model_loader_extra_config={})
    worker.model_config = SimpleNamespace(
        model=REPO_ID,
        revision=revision,
        code_revision=None,
        hf_config=SimpleNamespace(model_type="mobilint-qwen2"),
        model_kwargs={},
        hf_overrides={},
    )
    worker.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(model_loader_extra_config={}), model_config=worker.model_config
    )
    return worker


@pytest.mark.parametrize(
    ("revision", "branch"),
    [(None, "main"), ("main", "main"), ("bs16", "bs16"), (COMMITS["bs16"], "bs16")],
)
def test_load_model_resolves_config_weights_and_mxq_from_the_requested_revision(
    hub_cache, resolved_mxq_paths, revision, branch
) -> None:
    worker = _make_worker(revision)

    worker.load_model()

    snapshot = hub_cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / COMMITS[branch]
    (mxq_path,) = resolved_mxq_paths
    assert mxq_path == str(snapshot / MXQ_NAME)
    assert open(mxq_path).read() == branch
    assert worker.model.config._commit_hash == COMMITS[branch]
    assert worker.model.config.npu_backend.max_batch_size == MAX_BATCH_SIZES[branch]
    assert torch.all(worker.model.get_input_embeddings().weight == MAX_BATCH_SIZES[branch])
