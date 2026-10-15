# Changelog

## Unreleased

### Changed

- Each engine sizes its torch CPU thread pool for its share of the host instead of keeping PyTorch's host-sized
  default: the worker calls transformers-mblt's `apply_host_thread_defaults` with the number of local engines
  (`data_parallel_size_local`, 1 without data parallelism), which divides the CPUs the process may use among them,
  clamped to 2–8 threads and never above torch's default. `OMP_NUM_THREADS` or `MKL_NUM_THREADS` still wins when set.
  vLLM's own `OMP_NUM_THREADS=1` default applies only to its multiprocessing executor, not to the `uni` executor
  vllm-mblt runs on. Several default-thread engines on one 16-core reservation reached 194–216 tok/s in total, against
  557–580 tok/s with two threads each (#44). Requires transformers-mblt 0.6.1.
- The prefix cache is device-only by default, as vLLM's GPU prefix cache is (#40). A finished request's snapshot is
  reused while its NPU slot still holds it; when another cache takes the slot, the snapshot is dropped without a
  dump, whether or not a load of it could pay. `VLLM_MBLT_PREFIX_CACHE_HOST_OFFLOAD=1` /
  `prefix_cache_host_offload` (off by default) saves it to host memory instead, under the cost test that decided it
  before. On a non-batch model, a finished prefix is therefore reused only by the next request unless host offload is
  on. Without host offload a batch model keeps up to one finished snapshot per slot, as they take no host memory.
- A new request takes a free slot that still holds a snapshot of its prefix and continues on it in place, with no
  load or dump and no cost test (only `prefix_cache_min_hit_tokens` applies). Otherwise it takes a slot on the
  least-loaded card and, on it, an empty slot first, then one whose snapshot would not be worth saving, then the
  least recently used snapshot. A prefix hit now marks its snapshot as recently used; before, only finishing did.

### Fixed

- With `dev_no` listing several cards, concurrent requests now spread over every card instead of filling card 0's
  batch first. Worker slot `s` addresses row `(s % N, s // N)` of the `N` cache MXQs (round-robin, the layout
  mobilint/transformers-mblt#31 gives its default cache rows) instead of `divmod(s, K)`, and a new request takes a
  free slot on the card running the fewest requests. With one card nothing changes.
- With prefix caching on, reusing a slot no longer dumps the finished request's KV that is still on it when no
  load of that snapshot could pay. The dump moved the slot's whole KV to host memory on the engine loop (about
  380 ms for a 224 MiB Qwen3-0.6B slot) for every new request, even with no prefix hits, and raised TTFT
  (#40). The worker now applies the snapshot-load cost test before the slot changes hands: the snapshot is saved
  only when a load of some prefix of it could beat recomputing that prefix, and dropped otherwise. Until a load is
  measured, the measured dump time stands in for the load time; with no measurement, the snapshot is saved as
  before. A request whose prefix snapshot is still on its own slot takes the cache over in place and, under the
  same test, no longer dumps a host copy for the snapshot.

## 0.8.0

### Added

- EAGLE-3 releases (`MobilintLlamaEagle3ForCausalLM`, `MobilintQwen2Eagle3ForCausalLM`,
  `MobilintQwen3Eagle3ForCausalLM`) serve with speculative decoding. Each step runs one transformers-mblt round per
  scheduled request (`eagle3_prefill`, then `eagle3_decode_step`) on the request's own `MobilintEagle3Cache` and
  returns all the tokens the round settles. Greedy output matches transformers-mblt `generate()`; on
  EAGLE3-JPharmatron-7B it decodes at about 11.9 tokens/s. The base MXQ holds one sequence, so the scheduler runs one
  request at a time, and prefix caching is off. Requests sample with temperature, top-k and top-p only: the platform
  refuses logprobs, penalties, `min_p`, `min_tokens`, seeds, logit bias, allowed or bad words, logits processors and
  structured output with HTTP 400. Needs transformers-mblt 0.5.0, which adds the round API.

### Changed

- The worker no longer calls the MXQ itself: every model runs through the transformers-mblt forward, and the direct
  MXQ path is removed with `VLLM_MBLT_DIRECT_INFER`. Building MXQ inputs (RoPE tables, deepstack layouts and their
  order), `BatchParam`s, output-buffer reuse and the multi-card fan-out all live in transformers-mblt now. A Qwen3-VL
  batch text MXQ without a RoPE input is refused at load; every Qwen3-VL Batch16 release has one.
- Qwen3-ASR now runs through the transformers-mblt forward too: each step calls the thinker's text decoder with the
  merged audio-text embeddings on the request's plain cache, which it decodes as one sequence. Transcriptions and
  their speed are unchanged.
- Mixture-of-Experts models (Qwen3-30B-A3B) now run through the same `_forward_rows` route as the other models:
  the request's own `MobilintMixtureOfExpertsCache` goes to the model forward directly (it cannot be joined), and the
  forward keeps its own prefill split, which the shared MXQs' width requires. Outputs stay within the model's
  run-to-run variation, and throughput is unchanged.
- Qwen2-VL and Qwen3-VL now run through the transformers-mblt forward as well, calling the text decoder with the
  merged embeddings. Each request keeps its prompt's M-RoPE positions and passes them as `position_ids`
  (transformers-mblt builds the RoPE input), and Qwen3-VL passes its deepstack features with them; Qwen3-VL request
  caches are `MobilintDeepStackCache`. Covered: Qwen3-VL non-batch MXQs, Qwen3-VL batch MXQs with a RoPE input, and
  Qwen2-VL with a plain text MXQ. Every forward takes the step as vLLM sized it
  (`npu_prefill_chunk_size` = the step's width), as the direct path did, instead of re-splitting it at the model's
  prefill chunk size, which also changes the NPU's numerics.
- Text-only models now run every infer through the transformers-mblt forward instead of calling the MXQ directly:
  the step's requests are joined with `MobilintCache.from_rows` and run in one batched forward, each on its own
  cache from its own length. transformers-mblt binds the slots, splits inputs to the MXQ width, packs the batch,
  spreads it across cards and reads the logits layout. Outputs are unchanged; offline throughput is the same on
  Llama-3.2-1B-Batch16 (16 requests) and 3% higher on Llama-3.2-1B (4 requests).
- Every request now keeps its own transformers-mblt cache on the NPU slot it runs on, and the worker binds it
  before each infer. transformers-mblt records which cache's KV each slot holds and saves or restores it when
  another cache takes the slot, so the worker no longer calls `dump_cache_memory` / `load_cache_memory` itself
  and no longer tracks which request the live slot belongs to.
  - Switching requests on a non-batch model saves the outgoing request's KV exactly, not rounded to a KV block, so
    the request resumes where it stopped instead of recomputing the tail.
  - A finished request's cache becomes its prefix snapshot without a dump. Its KV is saved only when another
    cache takes the slot, and not at all if the snapshot is dropped first.
  - A new request whose prefix snapshot is still on its own slot takes that cache over, KV in place, instead of
    loading a copy.
  - Prefix snapshots are cache copies instead of raw dump blobs, for dense and Mixture-of-Experts models alike.
- Requires `transformers-mblt>=0.5.0` (the round API for EAGLE-3; `from_rows` came in 0.4.0) and
  `mblt-model-zoo[transformers]>=2.11.0`.
- Finished-request prefix snapshots are also capped by host memory: at most
  `VLLM_MBLT_PREFIX_CACHE_MAX_HOST_MIB` / `prefix_cache_max_host_mib` MiB (4096 by default), each counted as one
  slot's compiled KV buffers (256 MiB for a 4K-token Llama-3 slot, 192 MiB for Qwen3-30B-A3B). The count cap of 16
  still applies; the startup log reports the resulting cap.

### Added

- Qwen3-MoE (`mobilint/Qwen3-30B-A3B`) can now be served. The base requirements cover it
  (`transformers-mblt>=0.5.0`); the `qwen3-moe` optional extra is kept so existing install commands keep working.
  The worker hands each scheduled chunk to the transformers-mblt MoE forward (shared MXQs, top-k routing, parallel
  expert MXQs, `lm_head`) on the request's own `MobilintMixtureOfExpertsCache`, which spans every per-layer shared
  MXQ.
- Only non-batch serving is supported: a `max_batch_size` above 1 is refused at startup, because MoE releases
  cannot be compiled as batch MXQs.
- `max_model_len` is clamped to 4096 for MoE models, the KV capacity of the released shared MXQs, and the worker
  refuses to start when the loaded MXQs hold fewer tokens than `max_model_len`.
- `--model-loader-extra-config` forwards the MoE role layout keys (`shared_*`, `expert_*`, `lm_head_*`) and
  `num_expert_workers` to the model.

### Fixed

- With `--no-enable-prefix-caching`, a finished request's KV cache is no longer dumped. The synchronous dump ran
  on the engine loop and delayed the next request's prefill, though nothing could reuse it: without prefix caching
  a snapshot can only be matched by physical block number (#28).
- With prefix caching off, a request loads only its own snapshot, never another request's, and a finished
  request's snapshot is dropped instead of kept in the finished-snapshot LRU. Non-batch request swapping still
  saves the outgoing request's KV and restores it when the request resumes.

### Documentation

- README lists the vLLM 0.11.2 offline `LLM` exit hang and abort under Known Issues, with a workaround.

## 0.7.0

### Added

- Qwen3-ASR (`mobilint/Qwen3-ASR-1.7B`) can now be served for speech-to-text on `/v1/audio/transcriptions`,
  through the new `qwen-asr` optional extra.

## 0.6.1

### Changed

- Batch MXQs built with full logits now serve prompt logprobs (`echo=true`) from the prefill's per-position
  logits instead of the 1-token microstep path. Batch MXQs declare `[1, -1, vocab]` either way; the worker tells
  them apart by the output buffer's max width (equal to the input buffer's for full logits, the row count for
  last-token logits). Last-logit batch MXQs, and any whose widths cannot tell, keep the microstep path.
- Batched infer outputs are split from their last two axes, so a full-logits batch MXQ's `[1, 1, tokens, vocab]`
  output is sliced per request instead of being reshaped into rows.
- Normal batch rows that do not request prompt logprobs keep only their last-token logits, not a full-logits
  MXQ's per-position rows.
- A full-logits batch MXQ returns `tokens x vocab x 4` bytes per packed prefill, so serving one needs up to
  `max_num_batched_tokens x vocab x 4` bytes of host memory per step (about 1.2 GiB for Qwen3-0.6B at a 2048-token
  budget), plus a per-row copy while it is split. Size the serving container's memory for it: on aries a 24 GiB
  container failed the first such step with `Driver_FailedToAllocateHostMemory`, and 64 GiB served it.

## 0.6.0

### Added

- Qwen3-VL dynamic-vision artifacts now accept multi-image and video requests through the vLLM serving path.
  Static-vision Qwen3-VL artifacts retain their single-image/no-video constraints, and Qwen2-VL behavior is
  unchanged.
- Dynamic Qwen3-VL preprocessing now caps each NPU vision encoder invocation at 4096 pre-merge tokens and rejects
  `do_resize=False`, preventing oversized images or video frames from terminating the engine with an NPU shape error.

## 0.5.0

### Added

- Multi-card software batching now uses Model Zoo's multi-slot NPU backend
  when `dev_no` is a list. The worker maps each global vLLM cache row to its
  owning MXQ model and local cache ID, dispatches independent model-slot
  batches concurrently, and restores outputs to scheduler order.
- KV-cache dump/load and prefix-cache snapshots follow the same multi-slot
  mapping, keeping request state bound to the correct model instance and local
  cache ID across cards.

### Changed

- Model loading now validates that the aggregate backend capacity (`N * K`)
  covers the scheduler's resolved `max_batch_size` and reports multi-slot
  topology in initialization logs.

## 0.4.0

### Added

- Qwen3-VL text dispatch now supports Model Zoo 2.7's per-layer split
  DeepStack layouts: 4-input static artifacts for non-batch execution and
  5-input dynamic artifacts for both non-batch and Batch16 execution. The
  worker validates every declared layer shape, emits one tensor per DeepStack
  layer, appends external RoPE for the dynamic layout, and rejects unsupported
  batched split-static artifacts before runtime inference.
- External RoPE generation now reads Model Zoo 2.7's preserved
  `_mobilint_rotary_emb` runtime helper, while retaining the legacy
  `rotary_emb` fallback. Without this lookup, dynamic Batch16 artifacts loaded
  successfully but the first request fatally stopped the engine.

### Changed

- Raised the minimum `mblt-model-zoo[transformers]` version from 2.3.0 to
  2.7.0. This establishes the Model Zoo contract needed for Qwen3-VL Batch16
  split-dynamic text MXQs and removes the former processor-side 2048-token
  dynamic-vision cap.
- Qwen3-VL processor integration now has regression coverage ensuring image and
  video resolution overrides reach Model Zoo unchanged and the removed
  2048-token cap is not reintroduced at the vllm-mblt boundary.

## 0.3.0

### Added

- `MbltWorker.profile()` records NPU activity as a qbruntime event trace, so
  users can see where time goes on the accelerator. vLLM reaches every worker
  through `collective_rpc("profile", ...)` and the v1 `WorkerBase` declares no
  `profile`, so `/start_profile`, `/stop_profile`, `LLM.start_profile()` and
  `vllm bench serve --profile` previously raised `AttributeError` on this
  platform; implementing the hook is what makes all of them work. A torch
  profiler would show nothing useful here because the model runs on the NPU
  through qbruntime rather than through torch ops, so the hook drives
  `qbruntime.start_tracing_events` / `stop_tracing_events` instead, the way
  each out-of-tree platform points the same hook at its own device profiler.
  `VLLM_TORCH_PROFILER_DIR` stays the switch and the output directory -- no
  MBLT-specific flag or endpoint is added -- and each window writes
  `{hostname}_{pid}.mblt_npu_rank{rank}.{time_ns}.json` for
  <https://ui.perfetto.dev/>, the convention
  `torch.profiler.tensorboard_trace_handler` uses for its own traces and that
  vLLM and vllm-ascend build on: the nanosecond timestamp keeps successive
  windows from clashing, the pid separates concurrent processes on a host, and
  the hostname separates containers that share a mounted trace directory but
  not a pid namespace. Because qbruntime buffers the log
  and writes it only on stop, `shutdown()` stops a running trace so a server
  torn down mid-window does not lose it, and a second start while one is
  recording is refused with a warning rather than taking over the first owner's
  window. A trace that cannot be started, or whose log cannot be written, fails
  the profile request instead of reporting success for a trace that will not be
  on disk; a repeated start or a stop with nothing running only logs. Verified
  on an Aries board: a start/stop window around one completion request produced
  a 1330-event trace, and a `SIGTERM` mid-trace wrote the window before
  `Model disposed.`

## 0.2.3

### Fixed

- The 3-input Qwen3-VL text MXQ no longer assumes a fixed order for its two
  trailing inputs. `_build_infer_inputs` and `_build_batch_infer_inputs` read
  `input_shapes[1]`/`[2]` positionally, which holds for the MXQs shipped so far
  but is not something the compiler guarantees: a rebuild of the same model can
  declare `[inputs, deepstack, rope]`, and serving one failed with
  `RuntimeError: 3-input Qwen3-VL batch text MXQ rope batch dimension must be
  1, got (3, -1, 4096).` The two inputs are now classified from their declared
  shapes -- by the last axis (deepstack's is the hidden size) and, when that is
  unreadable, by the leading axis (only deepstack may declare more than one
  layer) -- and the same answer drives both validation and the order the
  tensors are emitted in. Classifying one way while emitting the other would
  have put RoPE in the deepstack slot and returned wrong logits with no error.
  A signature neither discriminator can read -- a single-layer deepstack whose
  declared size the rope input shares -- still falls back to the positional
  order for the model kind, so shipped artifacts are unchanged. That fallback
  remains the one path that can emit the two tensors in the wrong slots without
  raising, because the declared shapes are then identical and every validation
  check passes either way; it now logs a warning once per worker naming the
  shapes it could not tell apart.

## 0.2.2

### Fixed

- `MbltPlatform` now reports `is_pin_memory_available() == False`. The base
  vLLM `Platform` answers `True` on any non-WSL host, so an MBLT server on
  plain Linux without an NVIDIA driver made vLLM allocate pinned tensors and
  `Tensor.pin_memory()` raised `Found no NVIDIA driver on your system`. The
  sampling-penalty path hit this first (`apply_all_penalties` ->
  `make_tensor_with_pad`), which is why the fixes below could not ship as
  0.2.1. MBLT runs on the NPU with CPU-side host tensors, so there is nothing
  to pin, and `CpuPlatform` answers `False` for the same reason.

## 0.2.1

Tagged on GitHub but never published to PyPI: the release pipeline's test gate
caught the pinned memory failure above, so no 0.2.1 artifact was ever built.
Install 0.2.2 to get the fixes below.

### Fixed

- Sampling penalties (`frequency_penalty`, `presence_penalty`,
  `repetition_penalty`) are now applied by default instead of only when CUDA is
  available. vLLM applies them through a pure-torch fallback when the fused
  CUDA kernel is missing, so the CPU-hosted MBLT sampler can honour them. Set
  `VLLM_MBLT_ENABLE_SAMPLING_PENALTIES=0` for the previous ignore-and-warn
  behaviour.
- The worker now tracks the token count held by each live runtime cache and
  batch `cache_id` slot, and refuses the no-reload live-cache fast path when
  that count disagrees with the scheduler's `num_computed_tokens`. Such a
  divergence used to be served silently as fluent-but-wrong output; it now logs
  a warning and rebuilds the prefix.
- A finished or aborted batch request is no longer snapshotted when it does not
  own its live cache slot, so an abort before the first step can no longer
  publish another request's KV as a prefix snapshot.
- Prefix snapshots are never labeled with more tokens than the live cache holds.
  Dump-before-switch and dump-on-finish used the scheduler's
  `num_computed_tokens`, so a cache holding a shorter prefix produced a snapshot
  advertising KV that was not in the blobs, and a later load resumed past the
  missing entries. The dump is now clamped to the tracked count, or skipped when
  that count is zero.
- `npu_prefill_chunk_size` / `max_batch_size` dicts now resolve for
  `core_mode: "auto"` artifacts when the dict holds exactly one usable entry,
  instead of silently falling back to the `128` default. Note that accepting
  `"auto"` at model load still requires `normalize_core_mode` in
  `mblt_model_zoo.utils.core_mode` to allow it.

## 0.2.0

### Breaking Changes

- Moved Mobilint runtime cache snapshot, live-cache ownership, and accelerator
  cache slot handling out of `MbltWorker` into
  `vllm_mblt.runtime_cache.MbltRuntimeCacheManager`.
- Removed compatibility guarantees for old cache-related `MbltWorker` internals.
- Removed direct imports of cache implementation details from
  `vllm_mblt.mblt_worker`.
- Documented that pre-1.0 releases may introduce breaking internal API changes;
  backward compatibility is planned after 1.0.0.
