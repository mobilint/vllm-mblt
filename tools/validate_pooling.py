#!/usr/bin/env python3
"""Compare an MXQ on NPU to separately generated source-model reference outputs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_mblt.pooling import PoolingRuntime
from vllm_mblt.pooling_artifacts import package_identity, sha256


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
    if not reference.get("cases"):
        raise ValueError("Reference must contain at least one validation case")
    if not -1 <= args.min_cosine <= 1 or not 0 <= args.max_score_error < float("inf"):
        raise ValueError("Invalid validation thresholds")
    identity = package_identity(args.package)
    reference_digest = sha256(args.reference)
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
            if expected.shape != actual.shape or not torch.isfinite(expected).all():
                raise ValueError("Reference output must be finite and match the model output shape")
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
        if package_identity(args.package) != identity or sha256(args.reference) != reference_digest:
            raise ValueError("Artifacts or reference changed during validation")
        report = {
            **identity,
            "reference_sha256": reference_digest,
            "thresholds": {"min_cosine": args.min_cosine, "max_score_error": args.max_score_error},
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
