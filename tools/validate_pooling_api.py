#!/usr/bin/env python3
"""Exercise a running, API-key-protected pooling server on loopback."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.package / "pooling.json").read_text())
    headers = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
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

    def run(_):
        response = requests.post(args.url + endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        body = response.json()
        if embed:
            assert len(body["data"]) == 3
            assert [row["index"] for row in body["data"]] == [0, 1, 2]
            for row in body["data"]:
                assert len(row["embedding"]) == manifest["hidden_size"]
                assert abs(sum(x * x for x in row["embedding"]) - 1) < 1e-4
        else:
            assert [row["index"] for row in body["results"]] == [1, 0]
            assert body["results"][0]["relevance_score"] > body["results"][1]["relevance_score"]
        return response.status_code

    unauth = requests.post(args.url + endpoint, json=payload, timeout=30).status_code
    assert unauth == 401
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(run, range(4)))
    long_payload = dict(payload)
    if embed:
        long_payload["input"] = "word " * (manifest["max_length"] * 2)
    else:
        long_payload["documents"] = ["word " * (manifest["max_length"] * 2)]
    too_long = requests.post(args.url + endpoint, json=long_payload, headers=headers, timeout=30).status_code
    assert too_long == 400
    if embed and manifest.get("matryoshka"):
        for dimensions in (0, manifest["hidden_size"] + 1):
            invalid = dict(payload, dimensions=dimensions)
            response = requests.post(args.url + endpoint, json=invalid, headers=headers, timeout=30)
            assert response.status_code == 400
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
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
