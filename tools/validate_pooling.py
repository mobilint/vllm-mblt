#!/usr/bin/env python3
"""Compare an MXQ on NPU to separately generated source-model reference outputs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_mblt.pooling import PoolingRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument("--max-score-error", type=float, default=0.05)
    parser.add_argument("--core", type=int, default=0)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    runtime = PoolingRuntime(args.package, core=args.core)
    try:
        for key in ("source_model", "revision"):
            if reference[key] != runtime.manifest[key]:
                raise ValueError(f"Reference provenance mismatch: {key}")
        embed = runtime.kind in ("mean", "last")
        params = SimpleNamespace(
            task="embed" if embed else "score", dimensions=None, normalize=True, use_activation=True
        )
        rows = []
        for case in reference["cases"]:
            actual = runtime.encode(case["token_ids"], params)
            expected = torch.tensor(case["expected"])
            error = (actual - expected).abs().max().item()
            cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item() if embed else None
            rows.append(
                {
                    "tokens": len(case["token_ids"]),
                    "cosine": cosine,
                    "max_abs_error": error,
                    "passed": cosine >= args.min_cosine if embed else error <= args.max_score_error,
                }
            )
        report = {
            "source_model": reference["source_model"],
            "revision": reference["revision"],
            "cases": rows,
            "passed": all(row["passed"] for row in rows),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
