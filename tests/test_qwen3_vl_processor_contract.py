from types import SimpleNamespace

import pytest

from vllm_mblt.models.modeling_qwen3_vl import (
    MobilintQwen3VLProcessingInfo,
    MobilintQwen3VLSafeProcessor,
)


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
            MobilintQwen3VLSafeProcessor,
            {
                "use_fast": False,
                "max_pixels": 16_777_216,
                "min_pixels": 65_536,
                "size": {"longest_edge": 16_777_216},
            },
        )
    ]


def _make_safe_processor() -> MobilintQwen3VLSafeProcessor:
    processor = object.__new__(MobilintQwen3VLSafeProcessor)
    processor.image_processor = SimpleNamespace(
        patch_size=16,
        merge_size=2,
        size={"longest_edge": 16_777_216, "shortest_edge": 65_536},
        max_pixels=16_777_216,
        min_pixels=65_536,
    )
    processor.video_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        size={"longest_edge": 16_777_216, "shortest_edge": 65_536},
        max_pixels=16_777_216,
        min_pixels=65_536,
    )
    return processor


def test_dynamic_processor_caps_defaults_at_4096_vision_tokens() -> None:
    processor = _make_safe_processor()

    processor._clamp_dynamic_image_size()
    processor._clamp_dynamic_video_size()

    image_limit = 4096 * 16**2
    video_limit = 4096 * 16**2 * 2
    assert processor.image_processor.size["longest_edge"] == image_limit
    assert processor.image_processor.max_pixels == image_limit
    assert processor.video_processor.size["longest_edge"] == video_limit
    assert processor.video_processor.max_pixels == video_limit


def test_dynamic_processor_caps_call_overrides_at_4096_vision_tokens() -> None:
    processor = _make_safe_processor()
    image_kwargs = {
        "max_pixels": 16_777_216,
        "images_kwargs": {"size": {"longest_edge": 16_777_216}},
    }
    video_kwargs = {
        "videos_kwargs": {"size": {"longest_edge": 16_777_216}},
    }

    processor._clamp_dynamic_image_call_kwargs(image_kwargs)
    processor._clamp_dynamic_video_call_kwargs(video_kwargs)

    assert image_kwargs["max_pixels"] == 4096 * 16**2
    assert image_kwargs["images_kwargs"]["size"]["longest_edge"] == 4096 * 16**2
    assert video_kwargs["videos_kwargs"]["size"]["longest_edge"] == 4096 * 16**2 * 2


@pytest.mark.parametrize(
    ("method", "nested_key"),
    [
        ("_clamp_dynamic_image_call_kwargs", "images_kwargs"),
        ("_clamp_dynamic_video_call_kwargs", "videos_kwargs"),
    ],
)
def test_dynamic_processor_rejects_resize_bypass(method: str, nested_key: str) -> None:
    processor = _make_safe_processor()

    with pytest.raises(ValueError, match="4096-token NPU vision-token ceiling"):
        getattr(processor, method)({nested_key: {"do_resize": False}})
