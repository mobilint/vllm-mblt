from copy import deepcopy
from types import SimpleNamespace

from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (
    MobilintQwen3VLProcessor,
)

from vllm_mblt.models.modeling_qwen3_vl import MobilintQwen3VLProcessingInfo


def test_processing_info_forwards_resolution_overrides_to_model_zoo() -> None:
    calls = []
    info = object.__new__(MobilintQwen3VLProcessingInfo)
    info.ctx = SimpleNamespace(
        get_hf_processor=lambda processor_cls, **kwargs: calls.append((processor_cls, kwargs)) or object()
    )

    info.get_hf_processor(
        use_fast=False,
        max_pixels=16_777_216,
        min_pixels=65_536,
        size={"longest_edge": 16_777_216},
    )

    assert calls == [
        (
            MobilintQwen3VLProcessor,
            {
                "use_fast": False,
                "max_pixels": 16_777_216,
                "min_pixels": 65_536,
                "size": {"longest_edge": 16_777_216},
            },
        )
    ]


def test_model_zoo_dynamic_processor_does_not_reintroduce_2048_token_cap() -> None:
    processor = object.__new__(MobilintQwen3VLProcessor)
    image_kwargs = {
        "max_pixels": 16_777_216,
        "do_resize": False,
        "images_kwargs": {"size": {"longest_edge": 16_777_216}},
    }
    video_kwargs = {
        "max_pixels": 16_777_216,
        "do_resize": False,
        "videos_kwargs": {"size": {"longest_edge": 16_777_216}},
    }
    expected_image_kwargs = deepcopy(image_kwargs)
    expected_video_kwargs = deepcopy(video_kwargs)

    processor._clamp_dynamic_image_call_kwargs(image_kwargs)
    processor._clamp_dynamic_video_call_kwargs(video_kwargs)

    assert not hasattr(MobilintQwen3VLProcessor, "max_vision_tokens")
    assert image_kwargs == expected_image_kwargs
    assert video_kwargs == expected_video_kwargs
