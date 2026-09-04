# Changelog

## 0.2.1

### Fixed

- `MbltPlatform` now reports `is_pin_memory_available() == False`. The base
  vLLM `Platform` answers `True` on any non-WSL host, so an MBLT server on
  plain Linux without an NVIDIA driver made vLLM allocate pinned tensors and
  `Tensor.pin_memory()` raised `Found no NVIDIA driver on your system`. The
  sampling-penalty path hit this first (`apply_all_penalties` ->
  `make_tensor_with_pad`). MBLT runs on the NPU with CPU-side host tensors, so
  there is nothing to pin, and `CpuPlatform` answers `False` for the same
  reason.
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