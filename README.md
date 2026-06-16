# vLLM MBLT

<div align="center">

<a href="https://github.com/vllm-project/vllm" target="_blank">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mobilint/vllm-mblt/refs/heads/main/assets/header-dark.png">
    <img src="https://raw.githubusercontent.com/mobilint/vllm-mblt/refs/heads/main/assets/header-light.png" alt="vLLM × Mobilint" width="720">
  </picture>
</a>

[![PyPI - Version](https://img.shields.io/pypi/v/vllm-mblt?logo=pypi)](https://pypi.org/project/vllm-mblt/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vllm-mblt?logo=python)](https://pypi.org/project/vllm-mblt/)
[![vLLM](https://img.shields.io/badge/vLLM-0.11.2-blue)](https://github.com/vllm-project/vllm)
[![Mobilint](https://img.shields.io/badge/Mobilint-NPU-green)](https://www.mobilint.com/)

</div>

**vllm-mblt** is an out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin that integrates
[Mobilint](https://www.mobilint.com/) NPU runtime support into the vLLM serving and benchmarking stack.

It provides a custom vLLM platform, worker, and model registry hooks so Mobilint-optimized LLM/VLM artifacts
can be served through familiar vLLM commands and OpenAI-compatible APIs.

## Highlights

- **Out-of-tree vLLM plugin**: registers the `mblt` platform without patching vLLM itself.
- **Mobilint NPU worker**: dispatches text-generation and multimodal execution to Mobilint runtime models.
- **Model registry integration**: supports Mobilint wrappers for Llama, HyperCLOVAX, EXAONE/EXAONE4, Qwen2/3,
  and Qwen2/3-VL families.
- **Runtime-aware scheduling**: reads model-configured `npu_prefill_chunk_size` and `max_batch_size` values to
  tune chunked prefill and scheduler concurrency automatically.
- **vLLM benchmark compatibility**: works with `vllm serve`, `vllm bench serve`, and `vllm bench throughput`.

## Requirements

- Python 3.10+
- `vllm==0.11.2`
- `mblt-model-zoo[transformers] >= 1.5.1`
- A Mobilint NPU environment. If you are not yet a Mobilint customer, please contact
  [tech-support@mobilint.com](mailto:tech-support@mobilint.com).

The package pins vLLM for compatibility:

```text
vllm>=0.11.2,<=0.11.2
```

## Installation

Install from PyPI:

```bash
pip install vllm-mblt
```

Or install the latest source checkout:

```bash
git clone https://github.com/mobilint/vllm-mblt.git
cd vllm-mblt
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quick Start

### 1. Verify Plugin Registration

After installation, run:

```bash
vllm --help
```

You should see plugin logs indicating that the Mobilint `mblt` platform plugin has been discovered and activated.

### 2. Serve a Text Model

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct --trust-remote-code
```

Then query the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/models
```

### 3. Serve a VLM Model

Qwen2-VL and Qwen3-VL Mobilint models can be loaded through the same vLLM server path:

```bash
vllm serve mobilint/Qwen2-VL-2B-Instruct --trust-remote-code
```

```bash
vllm serve mobilint/Qwen3-VL-2B-Instruct --trust-remote-code
```

Current Mobilint Qwen2/3-VL notes:

- The worker loads VLMs through `AutoModelForImageTextToText`.
- Image inputs are processed through vLLM's multimodal pipeline and merged into Mobilint language-model prompt
  embeddings inside the custom worker.
- The NPU path currently supports exactly one image in the initial multimodal request.
- Subsequent turns in the same session must be text-only or reuse the same image-token position.
- Video inputs are not supported by the current Mobilint Qwen2/3-VL NPU path.

## Runtime Tuning

### Runtime Layout Overrides

By default, `vllm-mblt` follows the runtime layout encoded in the Mobilint model artifact/config. Use
`--model-loader-extra-config` only when you intentionally want to override runtime placement or testing knobs.

Runtime settings such as `dev_no`, `target_cores`, `target_clusters`, `core_mode`, and `max_batch_size` are
forwarded to `from_pretrained(...)` through `--model-loader-extra-config`.
For detailed `core_mode` and multicore runtime layout guidance, see the
[Mobilint multicore documentation](https://docs.mobilint.com/v1.2/en/multicore.html).

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct \
  --trust-remote-code \
  --model-loader-extra-config '{"dev_no": 0, "target_cores": ["1:0"]}'
```

### Chunked Prefill Auto-Tuning

If a model config includes `npu_prefill_chunk_size`, `vllm-mblt` uses it to tune vLLM chunked prefill.

- Integer values are used directly.
- Dict values are selected by `core_mode`.
- `core_mode` is resolved from `--model-loader-extra-config` first, then from the model config default.
- The selected value is applied to vLLM's `max_num_batched_tokens` for chunked prefill.
- If no matching value is found, `vllm-mblt` falls back to `128`.
- For batch-compiled models with `max_batch_size > 1`, the effective chunked prefill limit is clamped to `128`
  to match the qbruntime batch execution limit used by the worker.

Example model config:

```json
{
  "npu_prefill_chunk_size": {
    "single": 64,
    "global4": 256,
    "global8": 512
  }
}
```

With this command, `vllm-mblt` selects `256` for `global4`:

```bash
vllm serve mobilint/YourModel \
  --trust-remote-code \
  --model-loader-extra-config '{"dev_no": 0, "core_mode": "global4", "target_clusters": [0]}'
```

If you also pass `--max-num-batched-tokens`, the effective value becomes the smaller of the user-provided value
and the model-configured `npu_prefill_chunk_size`.

Use `--block-size` only when you intentionally want to override the model-configured/default block size:

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct \
  --trust-remote-code \
  --block-size 64
```

### Model-Configured Batch Capacity

If a model config includes `max_batch_size`, `vllm-mblt` uses that value to support batch-compiled Mobilint models.

- The worker uses `max_batch_size` for KV cache memory sizing.
- The platform applies it to vLLM `max_num_seqs` automatically.
- You do not need to pass `--max-num-seqs` unless you intentionally want a smaller scheduler cap.
- `max_batch_size` also supports the same `core_mode` keyed dict form as `npu_prefill_chunk_size`.
- For local testing, `--model-loader-extra-config '{"max_batch_size": 32}'` overrides the model config value.

Example:

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct-Batch32 --trust-remote-code
```

For batch-compiled MXQs such as `mobilint/Llama-3.2-1B-Instruct-Batch32`, the plugin also caps the effective
chunked prefill limit to `128`, even when the model config advertises a larger `npu_prefill_chunk_size`.

## Benchmarking

This repository includes `sonnet.txt`, which can be used with vLLM benchmark commands.

### Serve Benchmark

Terminal 1:

```bash
vllm serve --model mobilint/Llama-3.2-1B-Instruct --trust-remote-code
```

Terminal 2:

```bash
vllm bench serve --model mobilint/Llama-3.2-1B-Instruct \
  --trust-remote-code \
  --port 8000 \
  --num-warmups 1 \
  --dataset-name sonnet \
  --dataset-path sonnet.txt \
  --num-prompts 10
```

### Throughput Benchmark

```bash
vllm bench throughput --model mobilint/Llama-3.2-1B-Instruct \
  --trust-remote-code \
  --dataset-name sonnet \
  --dataset-path sonnet.txt \
  --num-prompts 10
```

Notes:

- `vllm bench serve` uses a separate server process; `vllm bench throughput` runs the engine directly.
- `vllm bench serve --max-concurrency` is a benchmark client load setting, not the server-side scheduler limit.
- Reported latency and throughput are environment-dependent. Capture results from your target board for documentation
  or performance comparisons.

## Supported Model Families

`vllm-mblt` registers Mobilint model wrappers for:

| Family | Registry class |
| --- | --- |
| Llama / HyperCLOVAX-compatible text models | `MobilintLlamaForCausalLM` |
| EXAONE | `MobilintExaoneForCausalLM` |
| EXAONE4 | `MobilintExaone4ForCausalLM` |
| Qwen2 | `MobilintQwen2ForCausalLM` |
| Qwen3 | `MobilintQwen3ForCausalLM` |
| Qwen2-VL | `MobilintQwen2VLForConditionalGeneration` |
| Qwen3-VL | `MobilintQwen3VLForConditionalGeneration` |

Model artifacts are available through Mobilint model repositories such as the
[Mobilint Hugging Face Hub](https://huggingface.co/mobilint).

## Cache Behavior

`MbltWorker` uses snapshot-based KV cache reuse with these policies:

- Event-driven dump, not every step.
- Reuse live cache for same-request continuous decode.
- Keep finished-session snapshots for prefix reuse.
- Evict finished snapshots with an LRU cap of 16 sessions.

Implementation file: [`vllm_mblt/mblt_worker.py`](vllm_mblt/mblt_worker.py)

## Tests

```bash
python -m pytest tests
```

## Project Structure

```text
vllm_mblt/
├── __init__.py                 # vLLM plugin and model registration entry points
├── mblt_platform.py            # platform config overrides and runtime-aware defaults
├── mblt_worker.py              # custom worker, prefill/decode flow, KV snapshot logic
└── models/                     # Mobilint model wrappers for LLM/VLM families

tests/
├── test_kv_cache_swap_spec.py
├── test_mblt_platform_prefill.py
└── test_mblt_worker_optimizations.py
```
