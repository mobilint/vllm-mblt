#!/usr/bin/env python3
"""Save FP32 source-model references for independent NPU validation.

Use Transformers >=5.5 for Nemotron, and the serving environment for other models.
The output is a JSON fixture containing token IDs and reference embeddings/scores.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from vllm_mblt.models.modeling_pooling import MobilintQwen3ForSequenceClassification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = json.loads((args.package / "pooling.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.package)
    kind = manifest["pooling"]
    cls = (
        AutoModelForCausalLM
        if kind == "yes_no"
        else (AutoModelForSequenceClassification if kind == "scalar" else AutoModel)
    )
    model = (
        cls.from_pretrained(
            manifest["source_model"],
            revision=manifest["revision"],
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )
    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Whales are marine mammals."),
        ("What does a compiler do?", "A compiler translates source code into executable instructions."),
        ("What does a compiler do?", "The apple tree blossoms in spring."),
        ("What is the capital of France?", "Paris is the capital of France. " * 100),
        ("컴파일러는 무엇을 하나요?", "컴파일러는 소스 코드를 실행 가능한 프로그램으로 변환합니다."),
    ]
    texts = [
        "Hello.",
        "query: What is the capital of France?",
        "passage: Paris is the capital of France.",
        "passage: Whales are marine mammals.",
        "한국어 문서도 검색할 수 있습니다.",
        "Machine learning uses data to learn patterns. " * 120,
    ]
    cases = []
    for item in pairs if kind in ("yes_no", "scalar") else texts:
        if kind == "yes_no":
            ids = tok.encode(MobilintQwen3ForSequenceClassification.get_score_template(*item))
            if len(ids) > manifest["max_length"]:
                from vllm_mblt.models.modeling_pooling import QWEN_RERANK_SUFFIX

                suffix = tok.encode(QWEN_RERANK_SUFFIX, add_special_tokens=False)
                ids = ids[: manifest["max_length"] - len(suffix)] + suffix
            encoded = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        elif kind == "scalar":
            encoded = tok(*item, truncation=True, max_length=manifest["max_length"], return_tensors="pt")
        else:
            encoded = tok(item, truncation=True, max_length=manifest["max_length"], return_tensors="pt")
        with torch.inference_mode():
            result = model(**{k: v.to(args.device) for k, v in encoded.items()})
            if kind == "yes_no":
                logits = result.logits[0, -1, [tok.convert_tokens_to_ids(w) for w in ("no", "yes")]].float()
                expected = logits.softmax(-1)[1:2]
            elif kind == "scalar":
                expected = result.logits.float().reshape(-1).sigmoid()
            else:
                hidden = result.last_hidden_state[0].float()
                expected = hidden.mean(0) if kind == "mean" else hidden[-1]
                expected = torch.nn.functional.normalize(expected, dim=0)
        cases.append({"token_ids": encoded["input_ids"][0].tolist(), "expected": expected.cpu().tolist()})
    args.output.write_text(
        json.dumps(
            {"source_model": manifest["source_model"], "revision": manifest["revision"], "cases": cases}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
