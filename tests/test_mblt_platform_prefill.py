import json
import logging
from types import SimpleNamespace
from vllm_mblt.mblt_platform import MbltPlatform, resolve_effective_npu_prefill_chunk_size, resolve_model_max_batch_size, resolve_npu_prefill_chunk_size

def _make_vllm_config(npu_prefill_chunk_size, *, loader_core_mode=None, loader_extra_config=None, hf_core_mode=None, max_batch_size=None, scheduler_max_num_batched_tokens=2048, scheduler_max_num_seqs=None):
    if loader_extra_config is None:
        loader_extra_config = {'core_mode': loader_core_mode} if loader_core_mode is not None else {}
    return SimpleNamespace(parallel_config=SimpleNamespace(worker_cls=None), cache_config=SimpleNamespace(block_size=None, enable_prefix_caching=None), scheduler_config=SimpleNamespace(chunked_prefill_enabled=False, max_num_batched_tokens=scheduler_max_num_batched_tokens, max_num_seqs=scheduler_max_num_seqs), model_config=SimpleNamespace(hf_config=SimpleNamespace(npu_prefill_chunk_size=npu_prefill_chunk_size, max_batch_size=max_batch_size, core_mode=hf_core_mode)), load_config=SimpleNamespace(model_loader_extra_config=loader_extra_config))

class TestMbltPlatformPrefill:

    def test_resolves_direct_integer_chunk_size(self) -> None:
        config = _make_vllm_config(192, loader_core_mode='global4')
        assert resolve_npu_prefill_chunk_size(config) == 192

    def test_prefers_loader_core_mode_over_hf_core_mode(self) -> None:
        config = _make_vllm_config({'single': 64, 'global4': 256}, loader_core_mode='global4', hf_core_mode='single')
        assert resolve_npu_prefill_chunk_size(config) == 256

    def test_prefers_json_string_loader_core_mode_over_hf_core_mode(self) -> None:
        config = _make_vllm_config({'single': 64, 'global4': 256}, loader_extra_config=json.dumps({'core_mode': 'global4'}), hf_core_mode='single')
        assert resolve_npu_prefill_chunk_size(config) == 256

    def test_resolves_json_string_loader_max_batch_size_override(self) -> None:
        config = _make_vllm_config({'global4': 256}, loader_extra_config=json.dumps({'core_mode': 'global4', 'max_batch_size': 16}), max_batch_size=32)
        assert resolve_model_max_batch_size(config) == 16

    def test_falls_back_to_hf_core_mode(self) -> None:
        config = _make_vllm_config({'single': 64, 'global8': 512}, hf_core_mode='global8')
        assert resolve_npu_prefill_chunk_size(config) == 512

    def test_supports_default_fallback_key(self) -> None:
        config = _make_vllm_config({'default': 160}, loader_core_mode='multi')
        assert resolve_npu_prefill_chunk_size(config) == 160

    def test_returns_none_for_missing_core_mode_mapping(self) -> None:
        config = _make_vllm_config({'single': 64}, loader_core_mode='global4')
        assert resolve_npu_prefill_chunk_size(config) is None

    def test_caps_batch_model_prefill_to_128(self) -> None:
        config = _make_vllm_config({'single': 256}, hf_core_mode='single', max_batch_size=32)
        assert resolve_effective_npu_prefill_chunk_size(config) == 128

    def test_keeps_non_batch_model_prefill_value(self) -> None:
        config = _make_vllm_config({'single': 256}, hf_core_mode='single')
        assert resolve_effective_npu_prefill_chunk_size(config) == 256

    def test_caps_non_single_batch_model_prefill_value(self) -> None:
        config = _make_vllm_config({'global4': 256}, loader_core_mode='global4', max_batch_size=32)
        assert resolve_effective_npu_prefill_chunk_size(config) == 128

    def test_platform_caps_scheduler_prefill_for_batch_model(self) -> None:
        config = _make_vllm_config({'single': 256}, hf_core_mode='single', max_batch_size=32)
        MbltPlatform.check_and_update_config(config)
        assert config.scheduler_config.chunked_prefill_enabled
        assert config.scheduler_config.max_num_batched_tokens == 128
        assert config.scheduler_config.max_num_seqs == 32

    def test_platform_warns_when_max_num_seqs_exceeds_model_batch_size(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="vllm_mblt.mblt_platform")
        config = _make_vllm_config(
            {'single': 256},
            hf_core_mode='single',
            max_batch_size=32,
            scheduler_max_num_seqs=64,
        )

        MbltPlatform.check_and_update_config(config)

        assert config.scheduler_config.max_num_seqs == 32
        assert "Clamping scheduler max_num_seqs from 64 to model-configured max batch size 32." in caplog.text

    def test_platform_keeps_smaller_user_max_num_seqs_without_warning(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="vllm_mblt.mblt_platform")
        config = _make_vllm_config(
            {'single': 256},
            hf_core_mode='single',
            max_batch_size=32,
            scheduler_max_num_seqs=8,
        )

        MbltPlatform.check_and_update_config(config)

        assert config.scheduler_config.max_num_seqs == 8
        assert "Clamping scheduler max_num_seqs" not in caplog.text

    def test_platform_keeps_equal_user_max_num_seqs_without_warning(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="vllm_mblt.mblt_platform")
        config = _make_vllm_config(
            {'single': 256},
            hf_core_mode='single',
            max_batch_size=32,
            scheduler_max_num_seqs=32,
        )

        MbltPlatform.check_and_update_config(config)

        assert config.scheduler_config.max_num_seqs == 32
        assert "Clamping scheduler max_num_seqs" not in caplog.text
