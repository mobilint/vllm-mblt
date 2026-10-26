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
  Qwen2/3-VL, Qwen3-ASR, Qwen3-MoE, and EAGLE-3 (Llama, Qwen2, Qwen3) families.
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

Qwen3-30B-A3B needs the optional `qwen3-moe` extra, which installs `transformers-mblt>=0.5.0`:

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

### 6. Serve an EAGLE-3 Model

EAGLE-3 releases pair a base model with a one-block draft model. vllm-mblt serves them with speculative decoding:

```bash
vllm serve mobilint/EAGLE3-JPharmatron-7B --trust-remote-code
```

Current Mobilint EAGLE-3 notes:

- transformers-mblt runs each speculative round: the base MXQ verifies the draft tree, the accepted tokens are kept,
  the next token is sampled after them, and the draft MXQ builds the next tree. A step returns all of the round's
  tokens for its request, so a streamed response arrives a few tokens per chunk. The tree size comes from the
  release's `generation_config.json` (`num_assistant_tokens`, `eagle3_tree_depth`, `eagle3_tree_top_k`).
- Requires transformers-mblt 0.5.0 or newer, which adds the round API (`eagle3_prefill` / `eagle3_decode_step`).
- One sequence at a time: the base MXQ holds one sequence (`base_max_batch_size` is 1), so the scheduler runs one
  request per step, and a release configured with more is refused. Prefix caching is turned off: every request
  feeds its whole prompt to its own cache.
- Requests sample with `temperature`, `top_k` and `top_p` only. Requests that set logprobs, prompt logprobs,
  penalties, `min_p`, `min_tokens`, `seed`, `logit_bias`, `allowed_token_ids`, `bad_words`, logits processors or
  structured output are refused with HTTP 400. Token prompts only: no prompt embeddings or multimodal data.
- Greedy output matches transformers-mblt `generate()` in the same environment. EAGLE3-JPharmatron-7B decodes at
  about 11.9 tokens/s on one Aries.

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
model instance and local cache ID that own them. A new request takes a free slot on the model instance running
the fewest requests, so concurrent requests spread over every loaded instance instead of filling the first
instance's `K` rows before the next one gets any. The one exception is a free slot still holding a prefix snapshot
of the request, which it takes on any instance (see [Cache Behavior](#cache-behavior)).

For explicit core or cluster placement across multiple cards, use canonical device-qualified target strings such
as `"0:0:0"` / `"1:0:0"` for `target_cores` or `"0:0"` / `"1:0"` for `target_clusters`.

A `dev_no` list still runs every card from one engine: each step waits for the slowest card, and the host work for
all rows runs between NPU calls. To scale throughput with the number of cards, run one engine per card instead,
with [Data Parallel](#data-parallel) or one `vllm serve` process per card (see
[Host CPU Threads](#host-cpu-threads)).

### Data Parallel

`--data-parallel-size N` serves one model from one `vllm serve` with N EngineCores, one per NPU card. vLLM's
built-in load balancer sends each request to the least busy engine, so the cards run independently instead of in
one lockstep step:

```bash
vllm serve mobilint/Qwen3-0.6B \
  --trust-remote-code \
  --data-parallel-size 3
```

- Engine `r` (local data parallel rank) runs on card `r`. To pick the cards, set `QBRUNTIME_VISIBLE_DEVICES`
  (for example `QBRUNTIME_VISIBLE_DEVICES=1,2,3`) or pass a `dev_no` list in `--model-loader-extra-config`
  (`'{"dev_no": [1, 2, 3]}'`); engine `r` takes the `r`-th card. Set only one of them. A single `dev_no` card
  is refused, as it would put every engine on one card, and so is a list shorter than the number of engines.
- vLLM records each engine's card in its config (`assigned_physical_gpu_ids`), and the worker narrows
  `QBRUNTIME_VISIBLE_DEVICES` to that card before it opens the NPU. qbruntime renumbers the listed cards from 0,
  as `CUDA_VISIBLE_DEVICES` does, and cannot open the others (verified on qbruntime 1.5.0 with 3 cards), so every
  engine opens its card as `dev_no` 0.
- Every NPU role of the model runs on the engine's card: the vision and text MXQs of a VLM, the base, draft and
  FC MXQs of an EAGLE-3 release, the encoder and decoder of Qwen3-ASR, and the shared, expert and `lm_head` MXQs
  of a Mixture-of-Experts model. Each role keeps the cores or clusters the release (or a prefixed `*_target_cores`
  / `*_target_clusters` key) gives it, on that card. Prefixed `dev_no` lists (`text_dev_no`,
  `draft_dev_no`, ...) are accepted when they all name the same cards. A role whose targets span several cards,
  or an unprefixed `target_cores` / `target_clusters` on a multi-role model, is refused.
- `max_batch_size` and `max_num_seqs` apply to each engine: three engines of a Batch16 MXQ serve up to 48
  sequences. Multi-card software batching (a `dev_no` list in a single engine) does not combine with data
  parallel: each engine owns one card.
- The prefix cache belongs to each engine. A prompt prefix cached on one card is not reused on another; the
  load balancer does not route by prefix.
- While any engine has requests, the engines step in waves and all-reduce their unfinished-request state over a
  gloo group every 32 steps. An idle engine waits about one of its recent model steps per wave step instead of
  running the model, so it neither spins a CPU core nor blocks early at that all-reduce.
- `--data-parallel-size 1` (the default) serves exactly as before.
- vLLM's data parallel front end exchanges requests through shared memory (`/dev/shm`). A three-engine serve
  allocates 160 MiB there, more than a container's default 64 MiB, and engine start-up then fails with
  "Insufficient space in /dev/shm". Give the container more, e.g. `docker run --shm-size 1g` or `--ipc=host`.
  vLLM 0.32 also starts one API server process per engine by default (`--api-server-count`).

### Host CPU Threads

Each engine sizes its torch CPU thread pool for its share of the host when it starts: the CPUs the process may use
(its affinity mask, lowered to a cgroup CPU quota) divided by the number of local data parallel engines, clamped
to 2–8 threads and never above PyTorch's default. The engine logs the count under `[mblt-init] host threads`.
`OMP_NUM_THREADS` or `MKL_NUM_THREADS`, when set, is kept as is.

Engines started as separate `vllm serve` processes, for example one per card pinned with `dev_no` and listening
on its own port, do not know about each other: each one counts itself as the only engine and takes up to 8
threads. Set `OMP_NUM_THREADS` for each of them, and keep the total thread budget of all engines and clients on
the host within the cores they are allowed to use (cpuset, container CPU limit). Start each process in its own
terminal or under a process manager; `vllm serve` stays in the foreground.

Reference measurements (Qwen3-0.6B, MXQ `global8`, runtime batch 16; three MLA100 cards with one ARIES each on
one host with a 16-core reservation; random prompts of 128 tokens with 64 output tokens and `--ignore-eos`;
vLLM 0.11.2, mobilint-qb-runtime 1.5.0). They were taken on vllm-mblt 0.8.0 and the data parallel branch, before
the prefix cache became device-only and before engines sized their own threads, so treat them as relative:

| Setup | Throughput |
|---|---|
| One engine, `dev_no: [0, 1, 2]`, concurrency 48 | 392–411 tok/s |
| `--data-parallel-size 3`, `OMP_NUM_THREADS=2`, prefix caching off, concurrency 48 | 508 tok/s |
| One `vllm serve` per card, `OMP_NUM_THREADS=2` | 557–580 tok/s total |
| One `vllm serve` per card, default threads, with three clients | 194–216 tok/s total (reservation saturated at 12.5–13.7 cores) |

On one card at concurrency 16, a default-thread engine used 2.8–2.9 cores and 45 threads; with
`OMP_NUM_THREADS=2` it used 0.9 cores at the same throughput (180–182 vs 181 tok/s). The cost is CPU-side
sampling: a 48-row sampling step took 9.5 ms with 2 threads instead of 1.6 ms, so give an engine more threads
when it has a large host to itself. These numbers depend on the host and model; measure on your own target
before choosing a value.

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

Start the server with vLLM's profiler config pointing at a directory. This is
the switch: without `--profiler-config.profiler` the OpenAI server does not
register the profile routes, and without `torch_profiler_dir` the worker reports
that tracing is not enabled. vLLM accepts the directory only with
`profiler=torch`; that also turns on its own front-end CPU trace (below).

```bash
vllm serve mobilint/Llama-3.2-1B-Instruct --trust-remote-code \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/tmp/mblt_traces
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
the same directory, because `--profiler-config.profiler=torch` also enables the
API server's `AsyncLLM` profiler. The two are complementary: that file covers
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
| EAGLE-3 (Llama, Qwen2, Qwen3) | `MobilintLlamaEagle3ForCausalLM`, `MobilintQwen2Eagle3ForCausalLM`, `MobilintQwen3Eagle3ForCausalLM` |

Model artifacts are available through Mobilint model repositories such as the
[Mobilint Hugging Face Hub](https://huggingface.co/mobilint).

## Cache Behavior

Text models, Mixture-of-Experts models, Qwen2-VL, Qwen3-VL and Qwen3-ASR run
each scheduled step through the transformers-mblt forward. On a batch MXQ the
step's request caches are joined with `MobilintCache.from_rows` and run in one
batched forward; a single request runs on its own cache. Image-text models call
their text decoder with the merged embeddings, M-RoPE `position_ids` and
deepstack features, and Qwen3-ASR calls its thinker's text decoder with the
merged audio-text embeddings. Mixture-of-Experts models serve one request at a
time, so each step runs on that request's `MobilintMixtureOfExpertsCache`. The
worker never calls an MXQ itself. A Qwen3-VL batch MXQ without a RoPE input is
refused at load, as the batched deepstack forward cannot run it; every Qwen3-VL
Batch16 release has one.

Every request keeps its own transformers-mblt cache on the NPU slot it runs on,
and the worker binds it before each infer. transformers-mblt records which
cache's KV each slot holds and saves it to host memory when another cache takes
the slot, so a request that loses its slot resumes from its own KV. On top of
that, `MbltWorker` reuses KV prefixes across requests with these policies:

- A finished request's cache becomes its prefix snapshot, without a dump, and
  its KV stays on its slot. By default the prefix cache lives on the NPU only,
  as vLLM's GPU prefix cache keeps evicted blocks nowhere: a snapshot is
  reusable while its slot still holds it, and when another cache takes the
  slot the snapshot is dropped and the slot is overwritten without a dump.
- A new request on a batch model takes a free slot that still holds a snapshot
  of its prefix, if there is one, and continues on that KV in place: no load,
  no dump, and no cost test (only `prefix_cache_min_hit_tokens` applies).
  The match uses the same block hashes, token ids and multimodal and
  prompt-embedding checks as a snapshot load. Otherwise the request takes a
  free slot on the card running the fewest requests and, on that card, a slot
  holding no snapshot, then one whose snapshot would not be worth saving under
  the cost test below, then the least recently finished or hit snapshot; the
  lowest slot breaks the remaining ties. A prefix hit marks its snapshot as
  recently used.
- A non-batch model has one slot, so by default a finished prefix is reused
  only by the next request that runs; any other request in between drops it.
- `VLLM_MBLT_PREFIX_CACHE_HOST_OFFLOAD=1` / `prefix_cache_host_offload: true`
  (off by default) also saves a finished snapshot to host memory when its slot
  changes hands, but only if a later load of it could beat recomputing it
  under the cost test below. Saving dumps the slot's whole KV synchronously on
  the engine loop (about 380 ms for a 224 MiB Qwen3-0.6B slot), so turn it on
  only for workloads whose prefixes repeat after their slot is reused.
- On a non-batch model, a request that loses the slot to another one is also
  kept as a snapshot, so the next request can start from its prefix.
- Finished snapshots are evicted LRU-first beyond 16, or beyond what fits the
  host memory budget, whichever is fewer. Each snapshot holds one slot's
  compiled KV buffers in host memory once saved (256 MiB for a 4K-token
  Llama-3 slot); the budget is `VLLM_MBLT_PREFIX_CACHE_MAX_HOST_MIB` /
  `prefix_cache_max_host_mib`, 4096 MiB by default. Without host offload a
  batch model's finished snapshots stay on their slots and take no host
  memory, so it keeps up to one per slot instead. The startup log reports
  the resulting cap.
- Matched snapshots on another slot or in host memory are loaded only when the
  worker-side cost model expects the one-cache-id load to beat recomputing the
  matched prefix.

The prefix-cache load threshold is enabled by default. During model warmup the
worker tries to measure optimistic one-cache-id prefill costs for 1, 2, 4, and
8 KV blocks; for batch MXQs this still submits exactly one active `cache_id`.
Real snapshot loads and dumps update per-`cache_id` EWMA timings. A snapshot is
loaded only when `load_ms < prefill_ms * 0.9`; otherwise the snapshot remains
stored and the prompt is recomputed. With host offload on, the same test
decides whether a finished snapshot is saved when its slot is reused: it is
saved only when a load of some prefix of it could pass. Until a load has been
measured, the measured dump time stands in for `load_ms`, since both move the
slot's whole KV buffers; with no measurement at all, the snapshot is saved.
Saving a snapshot dumps the slot's whole KV synchronously, so with host offload
on a workload whose prefixes are long enough to pass the test pays that dump on
every slot reuse, hit or not. The policy can be adjusted with environment
variables or equivalent model loader extra config keys:

- `VLLM_MBLT_PREFIX_CACHE_AUTO_THRESHOLD` / `prefix_cache_auto_threshold`
  defaults to enabled. Set `0` to disable measured thresholding.
- `VLLM_MBLT_PREFIX_CACHE_MIN_HIT_TOKENS` / `prefix_cache_min_hit_tokens`
  sets a manual minimum matched-token count before any snapshot load.
- `VLLM_MBLT_PREFIX_CACHE_LOAD_MARGIN` / `prefix_cache_load_margin` defaults
  to `0.9`.
- `VLLM_MBLT_PREFIX_CACHE_HOST_OFFLOAD` / `prefix_cache_host_offload` defaults
  to off. Set `1` to save finished snapshots to host memory as described above.
- `VLLM_MBLT_PREFIX_CACHE_CALIBRATE` defaults to enabled. Set `0` to skip
  startup prefill calibration.

A request's cache holds exactly the tokens the worker computed for it. If it
holds fewer than the scheduler's `num_computed_tokens`, the worker logs
`MBLT request cache holds fewer tokens than the scheduler expects`, naming the
request and `cache_id`, and recomputes the rest of the prefix. Seeing it in the
log means measurements taken from that server should be re-checked.

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
├── data_parallel.py            # data parallel card assignment and per-role placement on the engine's card
├── moe_runtime.py              # Mixture-of-Experts cache-model adapter over transformers-mblt
├── eagle3_runtime.py           # EAGLE-3 speculative rounds through transformers-mblt, one cache per request
├── tracing.py                  # qbruntime NPU event tracing behind vLLM's profiler hook
└── models/                     # Mobilint model wrappers for LLM/VLM families

tests/
├── test_data_parallel.py
├── test_eagle3.py
├── test_kv_cache_swap_spec.py
├── test_mblt_platform_prefill.py
├── test_mblt_tracing.py
├── test_mblt_worker_optimizations.py
└── test_qwen3_moe.py
```
