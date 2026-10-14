"""Integrity checks shared by pooling validation and packaging tools."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_file(root: Path, name: str, *, cache_blobs: Path | None = None) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid pooling artifact name: {name}")
    path = (root / name).resolve()
    inside = path.is_relative_to(root.resolve())
    cached = cache_blobs is not None and path.is_relative_to(cache_blobs.resolve())
    if not (inside or cached) or not path.is_file():
        raise ValueError(f"Missing or out-of-package pooling artifact: {name}")
    return path


def package_identity(root):
    """Verify actual files, then bind reports to the complete manifest bytes."""
    root = Path(root)
    manifest_path = artifact_file(root, "pooling.json")
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.get("sha256", {})
    required = {
        manifest["mxq"],
        manifest["embeddings"],
        "config.json",
        "source_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    if not required.issubset(hashes) or "pooling.json" in hashes:
        raise ValueError("Manifest must hash every required artifact and must not hash itself")
    for name, expected in hashes.items():
        if sha256(artifact_file(root, name)) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return {
        "source_model": manifest["source_model"],
        "revision": manifest["revision"],
        "manifest_sha256": sha256(manifest_path),
    }


def require_matching_report(report, identity, name):
    if report.get("passed") is not True or any(report.get(k) != v for k, v in identity.items()):
        raise ValueError(f"A successful report for these exact artifacts is required: {name}; rerun validation")
