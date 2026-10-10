#!/usr/bin/env python3
"""Archive a validated pooling package, excluding calibration and build intermediates."""

import argparse
import json
import tarfile
from pathlib import Path

from vllm_mblt.pooling_artifacts import artifact_file, package_identity, require_matching_report, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    manifest = json.loads((root / "pooling.json").read_text())
    identity = package_identity(root)
    names = set(manifest["sha256"]) | {"pooling.json", "reference.json", "validation.json", "api-validation.json"}
    for name in names:
        artifact_file(root, name)
    for name in ("validation.json", "api-validation.json"):
        report = json.loads((root / name).read_text())
        require_matching_report(report, identity, name)
        if name == "api-validation.json" and report.get("artifact_binding") != "managed-server":
            raise ValueError("API report must be produced with --launch-server")
        if name == "validation.json":
            if report.get("reference_sha256") != sha256(root / "reference.json"):
                raise ValueError("Reference does not match the numerical validation report")
            if not report.get("cases") or not all(row.get("passed") is True for row in report["cases"]):
                raise ValueError("Numerical report must contain passing validation cases")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz", compresslevel=1, dereference=True) as archive:
        for name in sorted(names):
            archive.add(root / name, arcname=f"{root.name}/{name}", recursive=False)
    checksum = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(f"{checksum}  {args.output}")


if __name__ == "__main__":
    main()
