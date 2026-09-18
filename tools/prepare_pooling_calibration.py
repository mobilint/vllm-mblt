#!/usr/bin/env python3
"""Create reproducible calibration rows from the public SQuAD training split."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    revision = HfApi().dataset_info("rajpurkar/squad", revision=args.revision).sha
    dataset = load_dataset("rajpurkar/squad", revision=revision, split="train")
    rows = [
        {"query": row["question"], "document": row["context"], "text": "passage: " + row["context"]}
        for row in dataset.select(range(0, args.samples * 10, 10))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "dataset": "rajpurkar/squad",
                "revision": revision,
                "split": "train",
                "indices": list(range(0, args.samples * 10, 10)),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
