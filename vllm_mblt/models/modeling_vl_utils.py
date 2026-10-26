import vllm.model_executor.models.interfaces as _vllm_interfaces
from vllm.multimodal.registry import _ProcessorFactories

QWEN3_VL_MAX_VISION_TOKENS = 4096

# transformers 5 evaluates every class annotation with get_type_hints when a PreTrainedModel subclass is created.
# Our multimodal models also inherit vLLM's SupportsMultiModal, whose `_processor_factory` annotation names
# `_ProcessorFactories`, a type vLLM imports only under TYPE_CHECKING. Bind it in that module so the annotation
# resolves; import this module before defining a model class that mixes in SupportsMultiModal.
if not hasattr(_vllm_interfaces, "_ProcessorFactories"):
    _vllm_interfaces._ProcessorFactories = _ProcessorFactories
