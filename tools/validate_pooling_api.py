#!/usr/bin/env python3
"""Exercise a running, API-key-protected pooling server on loopback."""

import argparse
import concurrent.futures
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from vllm_mblt.pooling_artifacts import package_identity


@contextmanager
def managed_server(package, manifest, *, core=0, timeout=180):
    """Start the exact local package; do not attest an independently running server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, VLLM_API_KEY=key)
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(package.resolve()),
        "--runner",
        "pooling",
        "--served-model-name",
        manifest["source_model"],
        "--max-model-len",
        str(manifest["max_length"]),
        "--max-num-seqs",
        "4",
        "--model-loader-extra-config",
        json.dumps({"core": core}),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while True:
                output = os.pread(log.fileno(), os.fstat(log.fileno()).st_size, 0).decode(errors="replace")
                if process.poll() is not None:
                    raise RuntimeError("Validation server exited during startup:\n" + output[-8000:])
                if "Application startup complete." in output:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Validation server startup timed out:\n" + output[-8000:])
                time.sleep(0.5)
            yield f"http://127.0.0.1:{port}", key
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url", help="Check an existing server; cannot produce a package-bound report")
    mode.add_argument(
        "--launch-server", action="store_true", help="Launch the package locally for a package-bound report"
    )
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.package / "pooling.json").read_text())
    if args.launch_server:
        identity = package_identity(args.package)
        with managed_server(args.package, manifest, core=args.core) as (url, key):
            report = check_api(manifest, url, key)
        if package_identity(args.package) != identity:
            raise ValueError("Artifacts changed during API validation")
        report.update(identity, artifact_binding="managed-server")
    else:
        report = check_api(manifest, args.url, os.environ["VLLM_API_KEY"])
        report["artifact_binding"] = "unverified-external-server"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def check_api(manifest, url, key):
    headers = {"Authorization": "Bearer " + key}
    embed = manifest["pooling"] in ("mean", "last")
    endpoint = "/v1/embeddings" if embed else "/v1/rerank"
    payload = {"model": manifest["source_model"]}
    if embed:
        payload["input"] = ["Hello.", "query: What is the capital of France?", "한국어 문서 검색"]
    else:
        payload.update(
            query="What is the capital of France?",
            documents=["Whales are marine mammals.", "Paris is the capital of France."],
        )

    def require(condition):
        if not condition:
            raise ValueError("Pooling API validation failed")

    def run(_):
        response = requests.post(url + endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        body = response.json()
        if embed:
            require(len(body["data"]) == 3)
            require([row["index"] for row in body["data"]] == [0, 1, 2])
            for row in body["data"]:
                require(len(row["embedding"]) == manifest["hidden_size"])
                require(abs(sum(x * x for x in row["embedding"]) - 1) < 1e-4)
        else:
            require([row["index"] for row in body["results"]] == [1, 0])
            require(body["results"][0]["relevance_score"] > body["results"][1]["relevance_score"])
        return response.status_code

    unauth = requests.post(url + endpoint, json=payload, timeout=30).status_code
    require(unauth == 401)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(run, range(4)))
    long_payload = dict(payload)
    if embed:
        long_payload["input"] = "word " * (manifest["max_length"] * 2)
    else:
        long_payload["documents"] = ["word " * (manifest["max_length"] * 2)]
    too_long = requests.post(url + endpoint, json=long_payload, headers=headers, timeout=30).status_code
    require(too_long == 400)
    if manifest["pooling"] == "yes_no":
        truncated = dict(long_payload, truncate_prompt_tokens=manifest["max_length"])
        response = requests.post(url + endpoint, json=truncated, headers=headers, timeout=30)
        require(response.status_code == 400)
        run(None)
    if embed and manifest.get("matryoshka"):
        for dimensions in (0, manifest["hidden_size"] + 1):
            invalid = dict(payload, dimensions=dimensions)
            response = requests.post(url + endpoint, json=invalid, headers=headers, timeout=30)
            require(response.status_code == 400)
            # A 400 alone is insufficient: an engine-side failure can also
            # surface as a 400 before the API process shuts down.
            run(None)
    report = dict(
        source_model=manifest["source_model"],
        endpoint=endpoint,
        concurrent_statuses=statuses,
        unauthenticated_status=unauth,
        overlength_status=too_long,
        passed=True,
    )
    return report


if __name__ == "__main__":
    main()
