import json
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools import package_pooling, validate_pooling
from vllm_mblt.pooling_artifacts import artifact_file, package_identity, sha256
from vllm_mblt.pooling_worker import MbltPoolingWorker


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    files = [
        "model.mxq",
        "embeddings.safetensors",
        "config.json",
        "source_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    for name in files:
        (root / name).write_text(name)
    manifest = dict(
        format_version=1,
        source_model="source/model",
        revision="revision-one",
        mxq="model.mxq",
        embeddings="embeddings.safetensors",
        max_length=512,
        sha256={name: sha256(root / name) for name in files},
    )
    (root / "pooling.json").write_text(json.dumps(manifest))
    (root / "reference.json").write_text("{}")
    identity = package_identity(root)
    for name in ["validation.json", "api-validation.json"]:
        report = dict(
            identity,
            passed=True,
            cases=[{"passed": True}],
            artifact_binding="managed-server",
            reference_sha256=sha256(root / "reference.json"),
        )
        (root / name).write_text(json.dumps(report))
    return root


def run_packaging(monkeypatch, root, output):
    monkeypatch.setattr(sys, "argv", ["package_pooling.py", str(root), "--output", str(output)])
    package_pooling.main()


def test_package_accepts_matching_reports(package, tmp_path, monkeypatch):
    output = tmp_path / "package.tar.gz"
    run_packaging(monkeypatch, package, output)
    with tarfile.open(output) as archive:
        assert archive.extractfile("package/model.mxq").read() == b"model.mxq"
    assert output.with_suffix(".gz.sha256").read_text().startswith(sha256(output))


@pytest.mark.parametrize("change", ["weights", "revision", "max_length", "missing_digest", "api_digest", "reference"])
def test_package_rejects_stale_reports(package, tmp_path, monkeypatch, change):
    manifest = json.loads((package / "pooling.json").read_text())
    if change == "weights":
        (package / "model.mxq").write_text("different build of same source")
        manifest["sha256"]["model.mxq"] = sha256(package / "model.mxq")
    elif change in ["revision", "max_length"]:
        manifest[change] = "different-revision" if change == "revision" else 256
    elif change in ["missing_digest", "api_digest"]:
        name = "validation.json" if change == "missing_digest" else "api-validation.json"
        report = json.loads((package / name).read_text())
        report.pop("manifest_sha256")
        (package / name).write_text(json.dumps(report))
    else:
        (package / "reference.json").write_text('{"changed": true}')
    (package / "pooling.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run_packaging(monkeypatch, package, tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()


def test_identity_rejects_unrecorded_file_change(package):
    (package / "embeddings.safetensors").write_text("replaced table")
    with pytest.raises(ValueError, match="checksum mismatch"):
        package_identity(package)


def test_hub_snapshot_symlinks_resolve_only_to_download_cache(tmp_path):
    snapshot = tmp_path / "snapshots" / "commit"
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "hash").write_text("{}")
    (snapshot / "pooling.json").symlink_to(blobs / "hash")
    assert artifact_file(snapshot, "pooling.json", cache_blobs=blobs) == blobs / "hash"
    with pytest.raises(ValueError):
        artifact_file(snapshot, "pooling.json")
    outside = tmp_path / "outside"
    outside.write_text("private")
    (snapshot / "escape").symlink_to(outside)
    with pytest.raises(ValueError):
        artifact_file(snapshot, "escape", cache_blobs=blobs)
    with pytest.raises(ValueError):
        artifact_file(snapshot, "../blobs/hash", cache_blobs=blobs)


def test_empty_reference_cannot_pass_validation(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text('{"cases": []}')
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["validate_pooling.py", str(tmp_path), "--reference", str(reference), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="at least one"):
        validate_pooling.main()
    assert not output.exists()


def test_worker_rejects_mismatched_metadata_and_releases_model(monkeypatch):
    closed = []
    runtime = SimpleNamespace(
        kind="mean", hidden_size=384, max_length=512, manifest={}, close=lambda: closed.append(True)
    )
    monkeypatch.setattr("vllm_mblt.pooling_worker.PoolingRuntime", lambda *a, **kw: runtime)
    worker = SimpleNamespace(
        load_config=SimpleNamespace(model_loader_extra_config={}),
        model_config=SimpleNamespace(
            model="package", revision=None, hf_config=SimpleNamespace(mblt_pooling="scalar", hidden_size=384)
        ),
    )
    with pytest.raises(ValueError, match="do not match"):
        MbltPoolingWorker.load_model(worker)
    assert closed == [True]
    assert worker.model is None


@pytest.mark.parametrize("invalid", ["external_api", "empty_cases", "failed_case"])
def test_package_rejects_unbound_or_incomplete_evidence(package, tmp_path, monkeypatch, invalid):
    name = "api-validation.json" if invalid == "external_api" else "validation.json"
    path = package / name
    report = json.loads(path.read_text())
    if invalid == "external_api":
        report["artifact_binding"] = "unverified-external-server"
    else:
        report["cases"] = [] if invalid == "empty_cases" else [{"passed": False}]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        run_packaging(monkeypatch, package, tmp_path / "bad.tar.gz")
