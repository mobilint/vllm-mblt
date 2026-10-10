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
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-mblt?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://clickpy.clickhouse.com/dashboard/vllm-mblt)

</div>

**vllm-mblt** is an out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin that integrates
[Mobilint](https://www.mobilint.com/) NPU runtime support into the vLLM serving and benchmarking stack.

It provides a custom vLLM platform, worker, and model registry hooks so Mobilint-optimized LLM, VLM and ASR
artifacts can be served through familiar vLLM commands and OpenAI-compatible APIs.

## Highlights

- **Out-of-tree vLLM plugin**: registers the `mblt` platform without patching vLLM itself.
- **Mobilint NPU worker**: dispatches text-generation and multimodal execution to Mobilint runtime models.
- **Model registry integration**: supports Mobilint wrappers for Llama, HyperCLOVAX, EXAONE/EXAONE4, Qwen2/3,
  Qwen2/3-VL, Qwen3-ASR, and Qwen3-MoE families.
- **Runtime-aware scheduling**: reads model-configured `npu_prefill_chunk_size` and `max_batch_size` values to
  tune chunked prefill and scheduler concurrency automatically.
- **vLLM benchmark compatibility**: works with `vllm serve`, `vllm bench serve`, and `vllm bench throughput`.

## Requirements

- Python 3.10+
- `vllm==0.11.2`
- `mblt-model-zoo[transformers] >= 2.7.0`
- For Qwen3-ASR, the `qwen-asr` extra: `pip install "vllm-mblt[qwen-asr]"`
- For Qwen3-MoE (`mobilint/Qwen3-30B-A3B`), the `qwen3-moe` extra: `pip install "vllm-mblt[qwen3-moe]"`
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
- Qwen3-VL dynamic-vision artifacts support multiple images and video inputs. Legacy static-vision Qwen3-VL
  artifacts support exactly one image and reject video inputs.
- Qwen2-VL supports exactly one image in the initial multimodal request and rejects video inputs.
- For static-vision Qwen3-VL and Qwen2-VL, subsequent turns in the same session must be text-only or reuse the
  same image-token position.
- Dynamic Qwen3-VL image and video preprocessing is capped at 4096 pre-merge vision tokens per encoder invocation,
  matching the NPU vision MXQ input limit. Oversized resolution overrides are clamped and `do_resize=False` is
  rejected because it can bypass this safety limit.

### 4. Serve a Speech-to-Text Model

Qwen3-ASR needs the optional `qwen-asr` extra:

```bash
pip install "vllm-mblt[qwen-asr]"
vllm serve mobilint/Qwen3-ASR-1.7B --trust-remote-code --max-num-seqs 1
```

It serves vLLM's OpenAI-compatible transcription endpoint:

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.flac -F model=mobilint/Qwen3-ASR-1.7B -F language=en
```

Current Mobilint Qwen3-ASR notes:

- Pass `language` to get plain text. Without it the model detects the language itself and its native preface
  (for example `language English<asr_text>`) is returned as part of `text`, once per chunk.
- vllm-mblt clamps `max_model_len` to 2048 for this model, so longer prompts get a 400: the artifact config
  declares 65536, but a request whose audio runs past token 2048 can stall the NPU.
- `--max-num-seqs 1`: the artifact config declares no `max_batch_size`, so without the flag vLLM interleaves
  requests on the batch-1 compiled decoder, and four concurrent requests take roughly 3-3.5x as long as sending
  them one by one.
- The optional `prompt` field reaches the model as context, such as names or terms the audio contains.
- On `/v1/audio/transcriptions`, audio longer than the artifact's 30 s window is split by vLLM and transcribed
  chunk by chunk. The chat endpoint takes each clip whole, so send long recordings to the transcription endpoint.
- On the chat endpoint, the artifact's chat template keeps only the system message and the audio, so put any
  context in the system message. Replies start with the model's preface, such as `language English<asr_text>`.
- One audio clip per request. `/v1/audio/translations` returns 400: the model transcribes but does not
  translate.
- vllm-mblt turns off vLLM's multimodal processor cache for this model. In vLLM 0.11.2, a request refused
  after preprocessing, such as one over the length limit, leaves that cache out of step with the engine, and
  sending the same audio again would stop the server from accepting requests.
- vLLM 0.11.2 can hang the whole API server while splitting some malformed long uploads (fixed in vLLM 0.25.0
  by [vllm-project/vllm#46463](https://github.com/vllm-project/vllm/pull/46463), after the version this plugin
  pins). Validate or re-encode uploads in front of a public deployment.

### 5. Serve a Mixture-of-Experts Model

Qwen3-30B-A3B needs the optional `qwen3-moe` extra, which installs `transformers-mblt>=0.2.0`:

```bash
pip install "vllm-mblt[qwen3-moe]"
vllm serve mobilint/Qwen3-30B-A3B --trust-remote-code
```

Current Mobilint Qwen3-MoE notes:

- transformers-mblt runs the whole forward pass: the per-layer shared MXQs, host-side top-8 routing, the expert
  MXQs in parallel across devices, and `lm_head`. vllm-mblt only feeds it embeddings at the scheduled KV offset
  and samples from the logits it returns.
- Non-batch serving only. MoE releases cannot be compiled as batch MXQs, so the config pins `max_batch_size` to 1,
  the scheduler runs one sequence at a time, and `--model-loader-extra-config '{"max_batch_size": N}'` with
  `N > 1` is refused.
- vllm-mblt clamps `max_model_len` to 4096 for this model: the artifact config declares 40960, but the shared
  MXQs hold 4096 tokens of KV. The worker also checks the loaded MXQs and refuses to start when `max_model_len`
  exceeds their capacity.
- The KV cache lives in all 48 shared MXQs, so a prefix-cache snapshot holds every layer's cache memory
  (192 MiB for Qwen3-30B-A3B).
- Placement follows the release config: shared MXQs round-robin over devices 0-2, experts over all 24 cores of
  those devices, and `lm_head` on `2:0:0`. Override it with the `shared_`, `expert_`, and `lm_head_` prefixed
  layout keys (for example `expert_target_cores`) and `num_expert_workers`.
- Prompt logprobs use the 1-token microstep path: the last layer only emits the last token's hidden state.
- Greedy output can differ from run to run where two tokens are nearly tied: transformers-mblt adds the expert
  outputs in the order they finish.

## Runtime Tuning

### Runtime Layout Overrides

By default, `vllm-mblt` follows the runtime layout encoded in the Mobilint model artifact/config. Use
`--model-loader-extra-config` only when you intentionally want to override runtime placement or testing knobs.

Runtime settings such as `dev_no`, `target_cores`, `target_clusters`, `core_mode`, and `max_batch_size` can be
provided through `--model-loader-extra-config`.
For detailed `core_mode` and multicore runtime layout guidance, see the
[Mobilint multicore documentation](https://docs.mobilint.com/v1.2/en/multicore.html).

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct \
  --trust-remote-code \
  --model-loader-extra-config '{"dev_no": 0, "target_cores": ["1:0"]}'
```

For VLMs such as `mobilint/Qwen3-VL-2B-Instruct`, shared runtime layout keys are applied to both Mobilint
submodules by forwarding them as model-zoo VLM subconfig keys (`vision_*` and `text_*`). Use explicit prefixed
keys when the vision encoder and text model need different placement:

```bash
vllm serve mobilint/Qwen3-VL-2B-Instruct \
  --trust-remote-code \
  --model-loader-extra-config '{"dev_no": 0, "core_mode": "global4", "vision_core_mode": "single"}'
```

For Mixture-of-Experts models such as `mobilint/Qwen3-30B-A3B`, unprefixed keys are not used. Each MXQ role
takes its own prefix (`shared_`, `expert_`, `lm_head_`), as in transformers-mblt:

```bash
vllm serve mobilint/Qwen3-30B-A3B \
  --trust-remote-code \
  --model-loader-extra-config '{"lm_head_target_cores": ["2:0:1"], "num_expert_workers": 16}'
```

For VLM-specific MXQ path overrides, use `vision_mxq_path` and/or `text_mxq_path`; a single top-level `mxq_path`
is only meaningful for single-module text models.

### Chunked Prefill Auto-Tuning

If a model config includes `npu_prefill_chunk_size`, `vllm-mblt` uses it to tune vLLM chunked prefill.

- Integer values are used directly.
- Dict values are selected by `core_mode`.
- `core_mode` is resolved from `--model-loader-extra-config` first, then from the model config default.
- The selected value is applied to vLLM's `max_num_batched_tokens` for chunked prefill.
- A `default` (or `DEFAULT`) key is used when the resolved `core_mode` has no entry.
- `core_mode: "auto"` is never a dict key. When the resolved `core_mode` is `"auto"` (or is absent) and the
  dict holds exactly one usable entry, that entry is used — a global-scheme batch mxq ships only the mode it
  was compiled for. Multiple entries under `"auto"` are not guessed.
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

### Multi-Card Software Batching

`dev_no` may be a list when the installed `mblt-model-zoo` / `mblt-npu-python` backend supports multi-slot
inference. The backend creates enough MXQ model slots for the requested aggregate `max_batch_size`, distributes
the slots round-robin across the selected cards, and `vllm-mblt` dispatches each scheduled batch to the owning
model slots concurrently.

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct \
  --trust-remote-code \
  --model-loader-extra-config '{"dev_no": [0, 1], "max_batch_size": 2}'
```

If one MXQ model instance has compiled batch capacity `K`, the backend loads
`ceil(max_batch_size / K)` model instances. `max_batch_size` is therefore the aggregate serving capacity across
all loaded instances, not a per-card value. KV-cache rows, including prefix-cache snapshots, remain bound to the
model instance and local cache ID that own them.

For explicit core or cluster placement across multiple cards, use canonical device-qualified target strings such
as `"0:0:0"` / `"1:0:0"` for `target_cores` or `"0:0"` / `"1:0"` for `target_clusters`.

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

## NPU Event Tracing

The worker can record NPU activity as a Chrome Tracing log through qbruntime's
event tracer, so you can see where time actually goes on the accelerator. It is
wired into vLLM's standard worker profiler hook, which means the usual controls
apply -- there is no MBLT-specific flag or endpoint.

Set `VLLM_TORCH_PROFILER_DIR` to a directory before starting the server. This is
the switch: without it the OpenAI server does not register the profile routes,
and the worker reports that tracing is not enabled.

```bash
export VLLM_TORCH_PROFILER_DIR=/tmp/mblt_traces
vllm serve mobilint/Llama-3.2-1B-Instruct --trust-remote-code
```

Then bracket the work you care about:

```bash
curl -X POST http://localhost:8000/start_profile
# send the requests you want to trace
curl -X POST http://localhost:8000/stop_profile
```

For an offline run, `LLM.start_profile()` / `LLM.stop_profile()` do the same, and
`vllm bench serve --profile` brackets the benchmark for you.

Each window writes `{hostname}_{pid}.mblt_npu_rank{rank}.{time_ns}.json` into
that directory. Open it at <https://ui.perfetto.dev/>. The name follows the
convention `torch.profiler.tensorboard_trace_handler` uses for its own traces,
which vLLM and vllm-ascend both build on: the nanosecond timestamp is what
keeps successive windows from clashing, the pid separates concurrent processes
on a host, and the hostname separates containers that share a mounted trace
directory but not a pid namespace. vLLM's front-end trace lands beside it as
`{hostname}_{pid}.async_llm.{time_ns}.pt.trace.json.gz`. The events are the runtime's own device-level
spans -- `infer`, `run npu`, `copy to npu`, `lock core`, `read device` and the
like -- so a window shows what each inference step spent on the accelerator.

vLLM writes its own front-end CPU trace (`*.async_llm.*.pt.trace.json.gz`) into
the same directory, because `VLLM_TORCH_PROFILER_DIR` also enables the API
server's `AsyncLLM` profiler. The two are complementary: that file covers
CPU-side scheduling, the MBLT file covers NPU execution.

Notes:

- Trace a short window. qbruntime buffers the whole log in the process and
  writes it only when tracing stops, so leaving a trace on for the life of a
  server grows memory and produces a file too large to be useful. If the worker
  shuts down while a trace is running it is stopped first so the window is not
  lost.
- Only one qbruntime trace can record per process. A start while one is
  already recording is refused with a warning rather than cutting the first
  window short. qbruntime has no way to report whether a trace is running or
  who owns it, so this covers traces started through this plugin and, when its
  module is already imported, through `mblt_model_zoo`'s benchmark helpers. A
  trace started by any other client cannot be detected.
- If a trace cannot be started, or its log cannot be written, `/start_profile`
  and `/stop_profile` fail rather than reporting success for a trace that will
  not be on disk. A repeated start or a stop with nothing running is not a
  failure and only logs. During shutdown a trace that cannot be written is
  logged and skipped so the rest of the teardown still runs.
- A `/stop_profile` with no trace running answers `500`. That comes from vLLM's
  front-end `AsyncLLM` profiler, which raises when stopped before it was
  started; the worker-side trace is unaffected and only logs that there was
  nothing to stop.
- A torch profiler on the worker would show nothing useful here: the model runs
  on the NPU through qbruntime rather than through torch ops, which is why this
  hook records a qbruntime trace instead.

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
| Qwen3-ASR | `MobilintQwen3ASRForConditionalGeneration` |
| Qwen3-MoE | `MobilintQwen3MoEForCausalLM` |

Model artifacts are available through Mobilint model repositories such as the
[Mobilint Hugging Face Hub](https://huggingface.co/mobilint).

## Cache Behavior

`MbltWorker` uses snapshot-based KV cache reuse with these policies:

- Event-driven dump, not every step.
- Reuse live cache for same-request continuous decode.
- Keep finished-session snapshots for prefix reuse.
- Evict finished snapshots with an LRU cap of 16 sessions.
- Load matched snapshots only when the worker-side cost model expects the
  one-cache-id load to beat recomputing the matched prefix.

The prefix-cache load threshold is enabled by default. During model warmup the
worker tries to measure optimistic one-cache-id prefill costs for 1, 2, 4, and
8 KV blocks; for batch MXQs this still submits exactly one active `cache_id`.
Real snapshot loads and dumps update per-`cache_id` EWMA timings. A snapshot is
loaded only when `load_ms < prefill_ms * 0.9`; otherwise the snapshot remains
stored and the prompt is recomputed. The policy can be adjusted with environment
variables or equivalent model loader extra config keys:

- `VLLM_MBLT_PREFIX_CACHE_AUTO_THRESHOLD` / `prefix_cache_auto_threshold`
  defaults to enabled. Set `0` to disable measured thresholding.
- `VLLM_MBLT_PREFIX_CACHE_MIN_HIT_TOKENS` / `prefix_cache_min_hit_tokens`
  sets a manual minimum matched-token count before any snapshot load.
- `VLLM_MBLT_PREFIX_CACHE_LOAD_MARGIN` / `prefix_cache_load_margin` defaults
  to `0.9`.
- `VLLM_MBLT_PREFIX_CACHE_CALIBRATE` defaults to enabled. Set `0` to skip
  startup prefill calibration.

The worker also tracks how many tokens each live runtime cache (or batch
`cache_id` slot) actually holds. A request may continue from a live cache
without reloading only when that count matches the scheduler's
`num_computed_tokens`. On a mismatch the worker logs a warning naming the
request and `cache_id`, drops the ownership claim, and rebuilds the prefix
instead of decoding against another sequence's KV. Seeing
`MBLT runtime cache token count holds fewer tokens than the scheduler expects` in the log
means measurements taken from that server should be re-checked.

### Sampling Penalties

`frequency_penalty`, `presence_penalty`, and `repetition_penalty` are applied
by default on the CPU-hosted MBLT sampler. Set
`VLLM_MBLT_ENABLE_SAMPLING_PENALTIES=0` to ignore them instead; the worker then
logs once which penalties it dropped. Released MBLT packages ship
`repetition_penalty` in `generation_config.json`, so ignoring them makes NPU
output generated under different sampling than a GPU reference run.

VLM prefix caching currently covers the language-model KV cache only. The
worker may load a compatible LM KV prefix snapshot and run only the uncached
text/embedding suffix. Image/video feature extraction is not cached by this
layer: image requests still rebuild vision features through the model's
multimodal feature hooks before the LM prefill/decode step.

Batch-compiled VLM text backends (`max_batch_size > 1`) are supported for
Mobilint Qwen2-VL and Qwen3-VL model types. With `mblt-model-zoo>=2.7.0`,
Qwen3-VL dynamic-vision Batch16 artifacts forward packed text embeddings plus
the matching packed RoPE and DeepStack tensors. Both the bundled 3-input text
layout and the per-layer split 5-input layout are supported; batched split-static
artifacts remain unsupported.
Qwen3-VL dynamic image and video processor resolution overrides are forwarded up to
the NPU encoder's 4096 pre-merge vision-token limit. Larger overrides are clamped and
`do_resize=False` is rejected so preprocessing cannot produce an unsupported MXQ input shape.
Unsupported multimodal model types fail before runtime inference with a clear
error.

Implementation file: [`vllm_mblt/mblt_worker.py`](vllm_mblt/mblt_worker.py)

## Known Issues

### Offline `LLM` scripts can hang or abort at exit

An offline `vllm.LLM` script can print all its results and then never exit, or abort at exit with
`terminate called without an active exception`. Both come from vLLM 0.11.2's synchronous engine client, not from
this plugin, and they happen with any model. `vllm serve` is not affected.

- **Hang.** The engine-core process is stopped only by a `weakref.finalize` callback. When the script imports
  `transformers` (which imports `filelock`) before vLLM, the `weakref` exit hook is registered before
  `multiprocessing`'s, so at exit `multiprocessing` waits for the engine-core process, which nothing has told to
  stop.
- **Abort.** When the last outputs carry logprobs tensors, vLLM's output thread can free them while the
  interpreter is shutting down, and torch's destructor aborts the process.

Workaround: shut the engine down yourself before the script ends.

```python
llm = LLM(model=..., trust_remote_code=True)
...
llm.llm_engine.engine_core.shutdown()
```

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
├── moe_runtime.py              # Mixture-of-Experts cache-model adapter over transformers-mblt
├── tracing.py                  # qbruntime NPU event tracing behind vLLM's profiler hook
└── models/                     # Mobilint model wrappers for LLM/VLM families

tests/
├── test_kv_cache_swap_spec.py
├── test_mblt_platform_prefill.py
├── test_mblt_tracing.py
├── test_mblt_worker_optimizations.py
└── test_qwen3_moe.py
```
