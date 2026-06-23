__version__ = "0.1.0"


def register():
    return "vllm_mblt.mblt_platform.MbltPlatform"


def register_model():
    from vllm import ModelRegistry

    ModelRegistry.register_model("MobilintLlamaForCausalLM", "vllm_mblt.models.modeling_llama:MobilintLlamaForCausalLM")

    ModelRegistry.register_model(
        "MobilintExaoneForCausalLM", "vllm_mblt.models.modeling_exaone:MobilintExaoneForCausalLM"
    )

    ModelRegistry.register_model(
        "MobilintExaone4ForCausalLM", "vllm_mblt.models.modeling_exaone4:MobilintExaone4ForCausalLM"
    )

    ModelRegistry.register_model("MobilintQwen2ForCausalLM", "vllm_mblt.models.modeling_qwen2:MobilintQwen2ForCausalLM")

    ModelRegistry.register_model("MobilintQwen3ForCausalLM", "vllm_mblt.models.modeling_qwen3:MobilintQwen3ForCausalLM")

    ModelRegistry.register_model(
        "MobilintQwen2VLForConditionalGeneration",
        "vllm_mblt.models.modeling_qwen2_vl:MobilintQwen2VLForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "MobilintQwen3VLForConditionalGeneration",
        "vllm_mblt.models.modeling_qwen3_vl:MobilintQwen3VLForConditionalGeneration",
    )
