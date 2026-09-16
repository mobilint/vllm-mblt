"""CPU preprocessing and postprocessing around a resident pooling MXQ."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn


def artifact_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-package pooling artifact: {name}")
    return path


def pool_output(output, kind, params, *, hidden_size, matryoshka=False):
    values = torch.as_tensor(np.asarray(output).copy(), dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise RuntimeError("Pooling MXQ returned non-finite values")
    task = params.task
    if kind in ("mean", "last"):
        if task != "embed":
            raise ValueError(f"Embedding artifact does not support {task!r}")
        tokens = values.reshape(-1, hidden_size)
        vector = tokens.mean(0) if kind == "mean" else tokens[-1]
        dimensions = params.dimensions
        if dimensions is not None:
            if not matryoshka or not 0 < dimensions <= hidden_size:
                raise ValueError("Unsupported embedding dimensions")
            vector = vector[:dimensions]
        return torch.nn.functional.normalize(vector, dim=0) if params.normalize is not False else vector
    if task not in ("score", "classify"):
        raise ValueError(f"Reranking artifact does not support {task!r}")
    values = values.reshape(-1)
    if kind == "yes_no":
        if values.numel() != 2:
            raise RuntimeError("Qwen reranker MXQ must return [no, yes] logits")
        logit = (values[1] - values[0]).reshape(1)
    elif kind == "scalar" and values.numel() == 1:
        logit = values
    else:
        raise RuntimeError(f"Unexpected reranker output shape: {tuple(values.shape)}")
    return logit.sigmoid() if params.use_activation is not False else logit


class PoolingRuntime(nn.Module):
    def __init__(self, model_path, *, revision=None, dev_no=0, core=0):
        super().__init__()
        import qbruntime

        root = Path(model_path)
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(model_path, revision=revision))
        self.manifest = json.loads(artifact_file(root, "pooling.json").read_text())
        manifest = self.manifest
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported pooling artifact format")
        self.kind = manifest["pooling"]
        self.max_length = manifest["max_length"]
        self.hidden_size = manifest["hidden_size"]
        source = json.loads(artifact_file(root, "source_config.json").read_text())
        state = load_file(str(artifact_file(root, manifest["embeddings"])))
        if manifest["input_kind"] in ("bert", "xlm-roberta"):
            if manifest["input_kind"] == "bert":
                from transformers import BertConfig
                from transformers.models.bert.modeling_bert import BertEmbeddings

                self.embeddings = BertEmbeddings(BertConfig(**source))
            else:
                from transformers import XLMRobertaConfig
                from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings

                self.embeddings = XLMRobertaEmbeddings(XLMRobertaConfig(**source))
            self.embeddings.load_state_dict(state, strict=True)
        else:
            self.embeddings = nn.Embedding.from_pretrained(state["weight"], freeze=True)
        self.embeddings.eval()
        if not isinstance(core, int) or isinstance(core, bool) or not 0 <= core < 8:
            raise ValueError("core must be an integer in [0, 7]")
        mc = qbruntime.ModelConfig()
        core_id = qbruntime.CoreId(
            getattr(qbruntime.Cluster, f"Cluster{core // 4}"), getattr(qbruntime.Core, f"Core{core % 4}")
        )
        mc.set_single_core_mode(None, [core_id])
        self.acc = qbruntime.Accelerator(dev_no)
        self.mxq = qbruntime.Model(str(artifact_file(root, manifest["mxq"])), mc)
        self.mxq.launch(self.acc)

    @torch.inference_mode()
    def encode(self, token_ids, params):
        if not token_ids or len(token_ids) > self.max_length:
            raise ValueError(f"Expected 1..{self.max_length} input tokens; received {len(token_ids)}")
        ids = torch.tensor([token_ids], dtype=torch.long)
        inputs = self.embeddings(ids).float()
        inputs = inputs.contiguous().numpy()
        output = self.mxq.infer([inputs])[0]
        if self.kind == "mean" and np.asarray(output).size != len(token_ids) * self.hidden_size:
            raise RuntimeError("Mean pooling requires every input token's hidden state")
        return pool_output(
            output, self.kind, params, hidden_size=self.hidden_size, matryoshka=self.manifest.get("matryoshka", False)
        )

    def close(self):
        if getattr(self, "mxq", None) is not None:
            self.mxq.dispose()
            self.mxq = None
