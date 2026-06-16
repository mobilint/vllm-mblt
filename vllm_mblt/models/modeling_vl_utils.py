from collections.abc import Mapping
from typing import Any, cast

from vllm.multimodal.inputs import MultiModalKwargsItems, MultiModalUUIDDict
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import MultiModalProcessingInfo


class MobilintVLCachedProcessorMixin:
    """Shared Mobilint VL workaround for full multimodal cache hits.

    vLLM's Qwen2-VL/Qwen3-VL cached processor path processes only
    cache-missing multimodal items. When every image/video item is already
    cached, the missing-item set is empty and the generated dummy text can be
    empty. Mobilint VL processors expect a valid prompt, so on a full cache hit
    we tokenize the real prompt only and merge cached multimodal kwargs/prompt
    updates without invoking the HF processor on empty multimodal data.
    """

    def _cached_apply_hf_processor(
        self,
        prompt: str | list[int],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        processor = cast(Any, self)
        cache = processor.cache

        _, passthrough_data = processor._get_hf_mm_data(mm_data_items)
        if cache is None or passthrough_data:
            return processor._apply_hf_processor(
                prompt=prompt,
                mm_data_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        mm_hashes = processor._hash_mm_items(
            mm_data_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        mm_missing_data_items = processor._get_cache_missing_items(
            cache=cache,
            mm_data_items=mm_data_items,
            mm_hashes=mm_hashes,
        )

        if not any(mm_missing_data_items.get_all_counts().values()):
            if isinstance(prompt, str):
                prompt_ids = processor._apply_hf_processor_text_only(
                    prompt,
                    tokenization_kwargs,
                )
            else:
                prompt_ids = processor._apply_hf_processor_tokens_only(prompt)

            mm_kwargs, mm_prompt_updates = processor._merge_mm_kwargs(
                cache,
                mm_hashes=mm_hashes,
                mm_missing_kwargs=MultiModalKwargsItems({}),
                mm_missing_prompt_updates={},
            )
            mm_info = MultiModalProcessingInfo(
                kwargs=mm_kwargs,
                hashes=mm_hashes,
                prompt_updates=mm_prompt_updates,
            )

            return prompt_ids, mm_info, False

        return cast(Any, super())._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
