#!/usr/bin/env python3
"""Build a self-contained pooling MXQ package (run in a qbcompiler environment)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from qbcompiler import CalibrationConfig, mxq_compile
from safetensors.torch import save_file
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "intfloat/e5-small": ("bert", "mean", "MobilintEmbeddingModel"),
    "BAAI/bge-reranker-v2-m3": ("xlm-roberta", "scalar", "MobilintForSequenceClassification"),
    "Qwen/Qwen3-Embedding-0.6B": ("qwen3", "last", "MobilintLastTokenEmbeddingModel"),
    "Qwen/Qwen3-Reranker-0.6B": ("qwen3", "yes_no", "MobilintQwen3ForSequenceClassification"),
    "nvidia/Nemotron-3-Embed-1B-BF16": ("ministral3", "mean", "MobilintEmbeddingModel"),
}


class EncoderOutput(torch.nn.Module):
    def __init__(self, model, classifier=None):
        super().__init__()
        self.encoder = model.encoder
        self.classifier = classifier

    def forward(self, inputs_embeds):
        hidden = self.encoder(inputs_embeds, return_dict=False)[0]
        return self.classifier(hidden) if self.classifier is not None else hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True, help="JSONL: text, or query/document pairs")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--target-device", choices=["aries-rb", "regulus-rb"], default="aries-rb")
    parser.add_argument("--inference-scheme", choices=["single", "all"], default="single")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory for each build")
    if args.samples < 1 or args.max_length < 2:
        raise ValueError("samples and max-length must be positive")
    recipe_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("nemotron_encoder.py"))
    }
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    family, pooling, architecture = MODELS[args.model]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.calibration = args.calibration.resolve()
    os.chdir(out)
    if family == "ministral3":
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(args.model, "tokenizer.json", revision=revision),
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
    else:
        tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if family == "ministral3":
        from nemotron_encoder import load_nemotron

        model, embeddings, source_config = load_nemotron(args.model, revision, args.max_length)
    else:
        cls = (
            AutoModelForCausalLM
            if family == "qwen3"
            else (AutoModelForSequenceClassification if pooling == "scalar" else AutoModel)
        )
        model = cls.from_pretrained(
            args.model,
            revision=revision,
            torch_dtype=torch.float32,
            attn_implementation="sdpa" if family == "qwen3" else "eager",
        ).eval()
        source_config = model.config.to_dict()
        embeddings = (
            model.get_input_embeddings()
            if family == "qwen3"
            else (getattr(model, model.base_model_prefix, model).embeddings)
        )
    hidden = source_config["hidden_size"]
    if family == "bert" and args.max_length > source_config["max_position_embeddings"]:
        raise ValueError("max-length exceeds BERT position embeddings")
    if family == "xlm-roberta" and args.max_length > source_config["max_position_embeddings"] - 2:
        raise ValueError("max-length exceeds XLM-R position embeddings")
    if family == "qwen3":
        head = torch.nn.Linear(hidden, 2 if pooling == "yes_no" else hidden, bias=False)
        if pooling == "yes_no":
            indices = [tok.convert_tokens_to_ids(word) for word in ("no", "yes")]
            head.weight.data.copy_(model.lm_head.weight.detach()[indices])
        else:
            head.weight.data.copy_(torch.eye(hidden))
        model.lm_head = head
    elif family in ("bert", "xlm-roberta"):
        model = EncoderOutput(getattr(model, model.base_model_prefix, model), getattr(model, "classifier", None)).eval()

    rows = [json.loads(line) for line in args.calibration.read_text().splitlines() if line.strip()][: args.samples]
    if len(rows) < args.samples:
        raise ValueError(f"Expected {args.samples} calibration rows, found {len(rows)}")
    calib = out / "calibration"
    calib.mkdir()
    lengths = []
    for i, row in enumerate(rows):
        if pooling == "scalar":
            encoded = tok(
                row["query"], row["document"], truncation=True, max_length=args.max_length, return_tensors="pt"
            )
        elif pooling == "yes_no":
            # Reserve the score suffix before truncating the pair's body.
            from vllm_mblt.models.modeling_pooling import (
                QWEN_RERANK_INSTRUCTION,
                QWEN_RERANK_PREFIX,
                QWEN_RERANK_SUFFIX,
            )

            prefix = tok.encode(QWEN_RERANK_PREFIX, add_special_tokens=False)
            suffix = tok.encode(QWEN_RERANK_SUFFIX, add_special_tokens=False)
            room = args.max_length - len(prefix) - len(suffix)
            if room <= 0:
                raise ValueError("max-length does not fit the reranker prompt")
            body = f"<Instruct>: {QWEN_RERANK_INSTRUCTION}\n<Query>: {row['query']}\n<Document>: {row['document']}"
            ids = tok.encode(body, add_special_tokens=False, truncation=True, max_length=room)
            encoded = {"input_ids": torch.tensor([prefix + ids + suffix])}
        else:
            encoded = tok(row["text"], truncation=True, max_length=args.max_length, return_tensors="pt")
        with torch.no_grad():
            sample = embeddings(encoded["input_ids"]).float()
            if family == "ministral3":
                # Trace and calibrate the full rotary table; MXQ retains a dynamic token axis.
                sample = sample.repeat(1, (args.max_length + sample.shape[1] - 1) // sample.shape[1], 1)
                sample = sample[:, : args.max_length]
            sample = sample.numpy()
        np.save(calib / f"{i:04d}.npy", sample)
        lengths.append(sample.shape[1])
    kwargs = dict(
        model=model,
        target_device=args.target_device,
        save_path=str(out / "model.mxq"),
        calib_data_path=str(calib),
        backend="torch",
        device="gpu",
        inference_scheme=args.inference_scheme,
        calibration_config=CalibrationConfig(mode=0),
    )
    if family == "qwen3":
        from qbcompiler.configs import LlmConfig, ResourceManagementConfig

        attr = LlmConfig.Attributes
        kwargs.update(
            hf_config={"library": "transformers", "tokenizer": "AutoTokenizer"},
            cpu_offload=False,
            optimize_option=2,
            llm_config=LlmConfig(
                apply=True,
                attributes=attr(
                    max_data_length=args.max_length,
                    max_sequence_length=args.max_length,
                    max_cache_length=args.max_length,
                    max_core_data_length=128,
                    calibration=attr.Calibration(use_full_seq_length=True),
                    runtime=attr.Runtime(batch_size=1, npu_core_ids=[0]),
                ),
            ),
            resource_management_config=ResourceManagementConfig(weight_dtype="bfloat16"),
        )
    else:
        sample = torch.from_numpy(np.load(calib / "0000.npy"))
        kwargs.update(feed_dict={"inputs_embeds": sample}, dynamic_axes={"inputs_embeds": [1]})
    if family in ("qwen3", "ministral3"):
        from qbcompiler.configs import BitConfig, EquivalentTransformationConfig

        kwargs["bit_config"] = BitConfig(
            transformer=BitConfig.Transformer(
                weight=BitConfig.Transformer.Weight(query=8, key=8, value=8, output=8, ffn=8, head=8),
                activation=BitConfig.Transformer.Activation(query=16, key=16, value=16, output=16, head=16),
            )
        )
        et = EquivalentTransformationConfig
        kwargs["equivalent_transformation_config"] = et(
            seed=0,
            apply_hadamard_rotation_matrix=True,
            spin_r1=et.SpinR1(apply=True),
            spin_r2=et.SpinR2(apply=True),
            qk=et.Qk(apply=False),
            qk_rotation=et.QkRotation(apply=False),
            ud=et.Ud(apply=False, learn=False),
            norm_conv=et.NormConv(apply=False),
            flatten_quant=et.FlattenQuant(apply=False),
            vo=et.Vo(apply=False),
            optimize_ffn=et.OptimizeFfn(apply=True, ch_per_ffn=-1),
        )
    mxq_compile(**kwargs)
    if family in ("qwen3", "ministral3"):
        rotations = list(out.glob("spinWeight/**/R1/global_rotation.pth"))
        if len(rotations) != 1:
            raise RuntimeError("Expected exactly one generated SpinR1 rotation matrix")
        rotation = torch.jit.load(str(rotations[0]), map_location="cpu").state_dict()["0"]
        embeddings.weight.data = (embeddings.weight.detach().cpu().double() @ rotation.double()).float()
    save_file(
        {k: v.detach().cpu().float().contiguous() for k, v in embeddings.state_dict().items()},
        str(out / "embeddings.safetensors"),
    )
    tok.save_pretrained(out)
    source_files = HfApi().list_repo_files(args.model, revision=revision)
    for name in ("README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE"):
        if name in source_files:
            shutil.copyfile(hf_hub_download(args.model, name, revision=revision), out / ("source_" + name))
    (out / "source_config.json").write_text(json.dumps(source_config, indent=2) + "\n")
    config = dict(
        model_type="bert",
        architectures=[architecture],
        hidden_size=hidden,
        num_hidden_layers=source_config["num_hidden_layers"],
        num_attention_heads=source_config["num_attention_heads"],
        vocab_size=source_config["vocab_size"],
        max_position_embeddings=args.max_length,
        torch_dtype="float32",
        mblt_pooling=pooling,
        pad_token_id=tok.pad_token_id,
        num_labels=1,
        is_matryoshka=family in ("qwen3", "ministral3") and pooling != "yes_no",
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = dict(
        format_version=1,
        recipe_sha256=recipe_hashes,
        source_model=args.model,
        revision=revision,
        input_kind=family,
        pooling=pooling,
        hidden_size=hidden,
        max_length=args.max_length,
        mxq="model.mxq",
        embeddings="embeddings.safetensors",
        target_device=args.target_device,
        inference_scheme=args.inference_scheme,
        matryoshka=config["is_matryoshka"],
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        calibration_samples=len(rows),
        calibration_lengths=lengths,
        versions={p: importlib.metadata.version(p) for p in ("qbcompiler", "torch", "transformers")},
    )
    manifest["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out / "pooling.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed pooling package: {out}", flush=True)


if __name__ == "__main__":
    main()
