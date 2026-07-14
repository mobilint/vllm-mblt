#!/usr/bin/env python3
"""Reproduce OpenAI completions echo+logprobs concurrency behavior.

Run against a live server, for example:
  vllm serve mobilint/Llama-3.2-1B-Instruct-Batch16 --trust-remote-code
  uv run python tests/repro_openai_echo_logprobs_concurrency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Response:
    status: int
    body: str


def post_completion(base_url: str, model: str, prompt: str, *, echo: bool = True, logprobs: int | None = 1) -> Response:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1,
        "echo": echo,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return Response(status=response.status, body=response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        return Response(status=exc.code, body=exc.read().decode("utf-8", errors="replace"))


def run_batch(base_url: str, model: str, prompts: list[str], *, name: str) -> None:
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(post_completion, base_url, model, prompt) for prompt in prompts]
        responses = [future.result() for future in as_completed(futures)]

    failures = [response for response in responses if response.status >= 500]
    if failures:
        sample = failures[0].body[:500]
        raise RuntimeError(f"{name}: {len(failures)} server errors; first body={sample!r}")

    bad_statuses = [response for response in responses if response.status not in {200, 400, 501}]
    if bad_statuses:
        sample = bad_statuses[0]
        raise RuntimeError(f"{name}: unexpected HTTP {sample.status}; body={sample.body[:500]!r}")

    print(f"{name}: {len(responses)} responses, statuses={sorted({response.status for response in responses})}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mobilint/Llama-3.2-1B-Instruct-Batch16")
    args = parser.parse_args()

    short_prompts = [f"Short echo logprobs request {i}." for i in range(16)]
    eval_prompts = [
        (
            f"Sample {i:02d}. This calibration sample is designed to exercise echo log probability scoring "
            "with a moderately sized prompt, similar to perplexity evaluation records. It contains enough "
            f"ordinary words to exceed sixteen tokenizer tokens. The final sentence index is {i}."
        )
        for i in range(32)
    ]

    run_batch(args.base_url, args.model, [short_prompts[0]], name="single-short-echo-logprobs")
    run_batch(args.base_url, args.model, short_prompts, name="sixteen-short-echo-logprobs")
    run_batch(args.base_url, args.model, eval_prompts, name="thirty-two-eval-echo-logprobs")

    normal = post_completion(args.base_url, args.model, "Server should still answer normal completions.", echo=False, logprobs=None)
    if normal.status != 200:
        raise RuntimeError(f"normal completion after repro failed: HTTP {normal.status}; body={normal.body[:500]!r}")
    print("post-repro-normal-completion: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
