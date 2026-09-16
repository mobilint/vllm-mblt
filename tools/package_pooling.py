#!/usr/bin/env python3
"""Archive a validated pooling package, excluding calibration and build intermediates."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    manifest = json.loads((root / "pooling.json").read_text())
    names = set(manifest["sha256"]) | {"pooling.json", "reference.json", "validation.json", "api-validation.json"}
    for name in names:
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError(f"Missing or invalid package member: {name}")
        expected = manifest["sha256"].get(name)
        if expected and sha256(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    for name in ("validation.json", "api-validation.json"):
        report = json.loads((root / name).read_text())
        if report.get("passed") is not True or report.get("source_model") != manifest["source_model"]:
            raise ValueError(f"A successful matching validation report is required: {name}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz", compresslevel=1) as archive:
        for name in sorted(names):
            archive.add(root / name, arcname=f"{root.name}/{name}", recursive=False)
    checksum = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(f"{checksum}  {args.output}")


if __name__ == "__main__":
    main()
