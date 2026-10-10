"""Unpadded, bidirectional Ministral3 encoder for the Nemotron embedding checkpoint.

Uses the Transformers Llama RMSNorm/MLP and YaRN frequency implementation.
There is deliberately no decoder cache or causal mask. Sequence lengths above
the compiled limit must be rejected by the caller.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers.modeling_rope_utils import _compute_yarn_parameters
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.groups = self.heads // self.kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dim, config.hidden_size, bias=False)

    def rotate(self, x, cos, sin):
        rotated = torch.cat((-x[..., self.dim // 2 :], x[..., : self.dim // 2]), dim=-1)
        return x * cos + rotated * sin

    def forward(self, x, cos, sin, scale):
        length = x.shape[1]
        q = self.q_proj(x).reshape(1, length, self.heads, self.dim).transpose(1, 2)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.dim).transpose(1, 2)
        q = self.rotate(q, cos, sin) * scale
        k = self.rotate(k, cos, sin)
        k = (
            k[:, :, None]
            .expand(1, self.kv_heads, self.groups, length, self.dim)
            .reshape(1, self.heads, length, self.dim)
        )
        v = (
            v[:, :, None]
            .expand(1, self.kv_heads, self.groups, length, self.dim)
            .reshape(1, self.heads, length, self.dim)
        )
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.dim**-0.5, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(1, length, self.heads * self.dim)
        return self.o_proj(attended)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, scale):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, scale)
        return x + self.mlp(self.post_attention_layernorm(x))


class NemotronEncoder(nn.Module):
    def __init__(self, config, max_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.head.weight.data.copy_(torch.eye(config.hidden_size))
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        frequencies, factor = _compute_yarn_parameters(config, torch.device("cpu"))
        phases = torch.arange(max_length).float()[:, None] * frequencies[None, :]
        phases = torch.cat((phases, phases), dim=-1)
        rp = config.rope_scaling
        scale = 1 + rp["llama_4_scaling_beta"] * torch.log(
            1 + torch.floor(torch.arange(max_length).float() / rp["original_max_position_embeddings"])
        )
        self.register_buffer("cos", (phases.cos() * factor)[None, None], persistent=False)
        self.register_buffer("sin", (phases.sin() * factor)[None, None], persistent=False)
        self.register_buffer("scale", scale[None, None, :, None], persistent=False)

    def forward(self, inputs_embeds):
        length = inputs_embeds.shape[1]
        cos, sin, scale = self.cos[:, :, :length], self.sin[:, :, :length], self.scale[:, :, :length]
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, scale)
        return self.head(self.norm(x))


def load_nemotron(model_id, revision, max_length):
    root = Path(snapshot_download(model_id, revision=revision, allow_patterns=["*.safetensors", "config.json"]))
    raw = json.loads((root / "config.json").read_text())
    if raw.get("is_causal") is not False or raw["model_type"] != "ministral3":
        raise ValueError("Expected the bidirectional Ministral3 embedding checkpoint")
    if raw.get("sliding_window") is not None or raw["rope_parameters"]["rope_type"] != "yarn":
        raise ValueError("Unsupported Nemotron attention/rotary configuration")
    config = SimpleNamespace(**raw)
    config.rope_scaling = raw["rope_parameters"]
    config.mlp_bias = False
    model = NemotronEncoder(config, max_length)
    state = {}
    for file in sorted(root.glob("*.safetensors")):
        state.update(load_file(str(file)))
    embeddings = nn.Embedding.from_pretrained(state.pop("embed_tokens.weight").float(), freeze=True)
    state["head.weight"] = model.head.weight.detach()
    model.load_state_dict(state, strict=True)
    return model.float().eval(), embeddings, raw
