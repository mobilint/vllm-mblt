from vllm import ModelRegistry

from vllm_mblt import register_model


def test_register_model_registers_qwen3_asr_by_import_path() -> None:
    register_model()

    registered = ModelRegistry.models["MobilintQwen3ASRForConditionalGeneration"]

    # A lazy entry: the wrapper, and the optional qwen-asr package it imports, load only when a
    # Qwen3-ASR model is served, so installs without the extra are unaffected.
    assert registered.module_name == "vllm_mblt.models.modeling_qwen3_asr"
    assert registered.class_name == "MobilintQwen3ASRForConditionalGeneration"
