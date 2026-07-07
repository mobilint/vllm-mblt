# Changelog

## Unreleased

### Breaking Changes

- Moved Mobilint runtime cache snapshot, live-cache ownership, and accelerator
  cache slot handling out of `MbltWorker` into
  `vllm_mblt.runtime_cache.MbltRuntimeCacheManager`.
- Removed compatibility guarantees for old cache-related `MbltWorker` internals.
- Removed direct imports of cache implementation details from
  `vllm_mblt.mblt_worker`.
- Documented that pre-1.0 releases may introduce breaking internal API changes;
  backward compatibility is planned after 1.0.0.