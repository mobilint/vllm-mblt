# Changelog

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
  `mblt_trace_{rank}_{pid}_{window}.json` for <https://ui.perfetto.dev/>. The
  pid is in the name because rank and window alone are not unique across
  processes: a restarted server counts windows from zero again, and two engines
  sharing one trace directory are both rank 0, so either could otherwise
  overwrite an earlier experiment's trace. Because qbruntime buffers the log
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