#!/usr/bin/env python3
"""Reproduce OpenAI completions echo+logprobs concurrency behavior.

Run against a live server, for example:
  vllm serve mobilint/Llama-3.2-1B-Instruct-Batch16 --trust-remote-code
  uv run python tests/repro_openai_echo_logprobs_concurrency.py
"""

from __future__ import annotations

import argparse
import json
import math
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


def validate_echo_logprobs_response(response: Response, prompt: str, *, name: str) -> None:
    if response.status != 200:
        raise RuntimeError(f"{name}: expected HTTP 200, got {response.status}; body={response.body[:500]!r}")

    try:
        payload = json.loads(response.body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name}: response body is not JSON: {exc}; body={response.body[:500]!r}") from exc

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{name}: response missing choices")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"{name}: choices[0] is not an object")

    text = choice.get("text")
    if not isinstance(text, str) or not text.startswith(prompt):
        raise RuntimeError(f"{name}: echoed text does not start with prompt; text={text!r}")

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(f"{name}: choices[0].logprobs is missing or not an object")

    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    if not isinstance(tokens, list) or not isinstance(token_logprobs, list) or not isinstance(top_logprobs, list):
        raise RuntimeError(f"{name}: logprobs must include tokens, token_logprobs, and top_logprobs lists")

    token_count = len(tokens)
    if token_count == 0:
        raise RuntimeError(f"{name}: logprobs.tokens is empty")
    for field_name, values in (
        ("token_logprobs", token_logprobs),
        ("top_logprobs", top_logprobs),
        ("text_offset", logprobs.get("text_offset")),
    ):
        if values is not None and (not isinstance(values, list) or len(values) != token_count):
            raise RuntimeError(f"{name}: logprobs.{field_name} length does not match tokens")

    if not all(isinstance(token, str) for token in tokens):
        raise RuntimeError(f"{name}: logprobs.tokens must contain strings")

    echoed_prompt_token_count = None
    echoed = ""
    for index, token in enumerate(tokens, start=1):
        echoed += token
        if echoed == prompt:
            echoed_prompt_token_count = index
            break
        if not prompt.startswith(echoed):
            raise RuntimeError(f"{name}: token text diverges from prompt at token {index}; echoed={echoed!r}")
    if echoed_prompt_token_count is None:
        raise RuntimeError(f"{name}: logprobs.tokens do not contain the echoed prompt prefix")

    for index, value in enumerate(token_logprobs):
        if value is None and index == 0:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise RuntimeError(f"{name}: token_logprobs[{index}] is not finite: {value!r}")

    for index, value in enumerate(top_logprobs):
        if value is None and index == 0:
            continue
        if not isinstance(value, dict):
            raise RuntimeError(f"{name}: top_logprobs[{index}] is not an object: {value!r}")
        for token, logprob in value.items():
            if not isinstance(token, str):
                raise RuntimeError(f"{name}: top_logprobs[{index}] contains non-string token: {token!r}")
            if not isinstance(logprob, (int, float)) or isinstance(logprob, bool) or not math.isfinite(logprob):
                raise RuntimeError(f"{name}: top_logprobs[{index}][{token!r}] is not finite: {logprob!r}")


def run_batch(base_url: str, model: str, prompts: list[str], *, name: str) -> None:
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {
            executor.submit(post_completion, base_url, model, prompt): prompt
            for prompt in prompts
        }
        responses = []
        for future in as_completed(futures):
            prompt = futures[future]
            response = future.result()
            validate_echo_logprobs_response(response, prompt, name=name)
            responses.append(response)

    print(f"{name}: {len(responses)} validated HTTP 200 echo+logprobs responses")


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

    normal = post_completion(
        args.base_url,
        args.model,
        "Server should still answer normal completions.",
        echo=False,
        logprobs=None,
    )
    if normal.status != 200:
        raise RuntimeError(f"normal completion after repro failed: HTTP {normal.status}; body={normal.body[:500]!r}")
    print("post-repro-normal-completion: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
