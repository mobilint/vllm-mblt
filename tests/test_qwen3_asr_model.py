"""Contract tests for the Qwen3-ASR vLLM wrapper.

The wrapper imports the optional qwen-asr package, which the test job does not install, so this
module skips without it. The worker's audio path is tested in test_mblt_worker_optimizations.py
and runs either way.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BatchFeature
from vllm.config.multimodal import AudioDummyOptions
from vllm.model_executor.models import is_text_generation_model, supports_multimodal, supports_transcription
from vllm.multimodal.inputs import MultiModalBatchedField, PlaceholderRange
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.transformers_utils import processor as processor_utils

pytest.importorskip("qwen_asr")

from vllm_mblt.models import modeling_qwen3_asr as asr  # noqa: E402

Model = asr.MobilintQwen3ASRForConditionalGeneration
AUDIO_PAD = "<|audio_pad|>"
AUDIO_PAD_ID = 151676
STT_CONFIG = SimpleNamespace(sample_rate=16000)
CLIP = np.zeros(16000, dtype=np.float32)

# config.support_languages of mobilint/Qwen3-ASR-1.7B.
# fmt: off
ARTIFACT_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
    "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]
# fmt: on


def _generation_prompt(**overrides: object) -> dict:
    kwargs = {
        "audio": CLIP,
        "stt_config": STT_CONFIG,
        "model_config": None,
        "language": "en",
        "task_type": "transcribe",
        "request_prompt": "",
        "to_language": None,
    }
    kwargs.update(overrides)
    return Model.get_generation_prompt(**kwargs)


def _dummy_inputs_builder(*, chunk_length: int = 30) -> asr.MobilintQwen3ASRDummyInputsBuilder:
    builder = object.__new__(asr.MobilintQwen3ASRDummyInputsBuilder)
    feature_extractor = SimpleNamespace(sampling_rate=16000, chunk_length=chunk_length)
    builder.info = SimpleNamespace(
        get_hf_processor=lambda: SimpleNamespace(audio_token=AUDIO_PAD),
        get_feature_extractor=lambda: feature_extractor,
    )
    return builder


def _multimodal_processor(
    *,
    encode: object = None,
    hf_calls: list | None = None,
    sampling_rate: int = 16000,
    hf_output: BatchFeature | None = None,
) -> asr.MobilintQwen3ASRMultiModalProcessor:
    processor = object.__new__(asr.MobilintQwen3ASRMultiModalProcessor)
    hf_processor = SimpleNamespace(
        audio_token=AUDIO_PAD, feature_extractor=SimpleNamespace(sampling_rate=sampling_rate)
    )
    tokenizer = SimpleNamespace(get_vocab=lambda: {AUDIO_PAD: AUDIO_PAD_ID}, encode=encode)
    calls = hf_calls if hf_calls is not None else []

    def call_hf_processor(hf_proc: object, data: dict, kwargs: dict) -> BatchFeature:
        calls.append((hf_proc, data, kwargs))
        return hf_output if hf_output is not None else BatchFeature({"input_ids": [[1]]})

    processor.info = SimpleNamespace(
        get_hf_processor=lambda **_kwargs: hf_processor,
        get_feature_extractor=lambda **_kwargs: hf_processor.feature_extractor,
        get_tokenizer=lambda: tokenizer,
        ctx=SimpleNamespace(call_hf_processor=call_hf_processor),
    )
    return processor


# --- processing info ----------------------------------------------------------------------


def test_processing_info_resolves_the_qwen3_asr_processor_from_the_model_context() -> None:
    calls = []
    hf_processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    info = object.__new__(asr.MobilintQwen3ASRProcessingInfo)
    info.ctx = SimpleNamespace(
        get_hf_processor=lambda processor_cls, **kwargs: calls.append((processor_cls, kwargs)) or hf_processor
    )

    assert info.get_hf_processor(use_fast=False) is hf_processor
    assert info.get_feature_extractor() is hf_processor.feature_extractor
    assert calls == [(asr.Qwen3ASRProcessor, {"use_fast": False}), (asr.Qwen3ASRProcessor, {})]


def test_processing_info_admits_one_clip_per_request() -> None:
    info = object.__new__(asr.MobilintQwen3ASRProcessingInfo)

    assert info.get_supported_mm_limits() == {"audio": 1}


# --- dummy inputs -------------------------------------------------------------------------


def test_dummy_inputs_are_one_full_window_of_silence_per_clip() -> None:
    builder = _dummy_inputs_builder()

    assert builder.get_dummy_text({"audio": 2}) == AUDIO_PAD * 2
    clips = builder.get_dummy_mm_data(seq_len=4096, mm_counts={"audio": 2})["audio"]
    assert len(clips) == 2
    for clip in clips:
        assert clip.shape == (30 * 16000,)
        assert not clip.any()


def test_dummy_inputs_follow_the_artifact_window() -> None:
    (clip,) = _dummy_inputs_builder(chunk_length=20).get_dummy_mm_data(seq_len=4096, mm_counts={"audio": 1})["audio"]

    assert clip.shape == (20 * 16000,)


def test_dummy_inputs_honour_a_shorter_length_override() -> None:
    (clip,) = _dummy_inputs_builder().get_dummy_mm_data(
        seq_len=4096,
        mm_counts={"audio": 1},
        mm_options={"audio": AudioDummyOptions(length=16000)},
    )["audio"]

    assert clip.shape == (16000,)


# --- multimodal processor -----------------------------------------------------------------


def test_data_parser_resamples_audio_to_the_feature_extractor_rate() -> None:
    parser = _multimodal_processor()._get_data_parser()

    items = parser.parse_mm_data({"audio": (np.zeros(8000, dtype=np.float32), 8000)})

    assert items["audio"].get(0).shape == (16000,)


def test_hf_processor_call_normalises_the_audio_key_and_passes_the_feature_extractor_rate() -> None:
    calls = []
    processor = _multimodal_processor(hf_calls=calls, sampling_rate=24000)

    processor._call_hf_processor(prompt="p", mm_data={"audios": [CLIP]}, mm_kwargs={}, tok_kwargs={})

    ((_, data, kwargs),) = calls
    assert "audios" not in data
    assert data["audio"][0] is CLIP
    assert data["text"] == "p"
    assert kwargs["sampling_rate"] == 24000


@pytest.mark.parametrize("mm_kwargs", [{"feature_size": 64}, {"sampling_rate": 16000}, {"sampling_rate": 8000}])
def test_hf_processor_call_refuses_processor_overrides(mm_kwargs: dict) -> None:
    calls = []
    processor = _multimodal_processor(hf_calls=calls)

    with pytest.raises(ValueError, match="does not accept processor overrides"):
        processor._call_hf_processor(prompt="p", mm_data={"audios": [CLIP]}, mm_kwargs=mm_kwargs, tok_kwargs={})
    assert calls == []


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan")])
def test_hf_processor_call_refuses_audio_whose_features_are_not_finite(bad_value: float) -> None:
    input_features = torch.zeros(1, 128, 10)
    input_features[0, 0, 0] = bad_value
    processor = _multimodal_processor(hf_output=BatchFeature({"input_features": input_features}))

    with pytest.raises(ValueError, match="features are not finite"):
        processor._call_hf_processor(prompt="p", mm_data={"audios": [CLIP]}, mm_kwargs={}, tok_kwargs={})


def test_hf_processor_call_returns_finite_features_unchanged() -> None:
    outputs = BatchFeature({"input_features": torch.zeros(1, 128, 10)})
    processor = _multimodal_processor(hf_output=outputs)

    assert processor._call_hf_processor(prompt="p", mm_data={"audios": [CLIP]}, mm_kwargs={}, tok_kwargs={}) is outputs


def test_hf_processor_call_tokenizes_text_only_prompts_without_the_audio_processor() -> None:
    calls = []
    processor = _multimodal_processor(encode=lambda _text: [11, 12, 13], hf_calls=calls)

    out = processor._call_hf_processor(prompt="hello", mm_data={}, mm_kwargs={}, tok_kwargs={})

    assert out["input_ids"].tolist() == [[11, 12, 13]]
    assert calls == []


@pytest.mark.parametrize(
    ("num_tokens", "refused"),
    [(2047, False), (2048, True), (2049, True)],
    ids=["one-below-the-limit", "at-the-limit", "past-the-limit"],
)
def test_apply_refuses_a_prompt_that_leaves_no_room_to_generate(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    refused: bool,
) -> None:
    processor = _multimodal_processor()
    processor.info.ctx.model_config = SimpleNamespace(max_model_len=2048)
    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "apply",
        lambda self, *args, **kwargs: {"prompt_token_ids": [0] * num_tokens, "mm_placeholders": {}},
    )

    if refused:
        with pytest.raises(ValueError, match="leaves no room to generate"):
            processor.apply("prompt", {}, {})
    else:
        assert len(processor.apply("prompt", {}, {})["prompt_token_ids"]) == num_tokens


@pytest.mark.parametrize(
    ("prompt_token_ids", "placeholders", "refused"),
    [
        ([7, AUDIO_PAD_ID, AUDIO_PAD_ID, 8], [PlaceholderRange(offset=1, length=2)], False),
        # A pad in message text took the clip's embeddings; the clip's own pad was left over.
        ([AUDIO_PAD_ID, AUDIO_PAD_ID, 7, AUDIO_PAD_ID, 8], [PlaceholderRange(offset=0, length=2)], True),
        # A clip with no samples: vLLM drops it, and its pad stays in the prompt as text.
        ([7, AUDIO_PAD_ID, 8], [], True),
    ],
    ids=["every-pad-filled", "stray-pad-in-text", "clip-without-samples"],
)
def test_apply_refuses_audio_placeholders_that_no_clip_fills(
    monkeypatch: pytest.MonkeyPatch,
    prompt_token_ids: list[int],
    placeholders: list,
    refused: bool,
) -> None:
    processor = _multimodal_processor()
    processor.info.ctx.model_config = SimpleNamespace(max_model_len=2048)
    out = {"prompt_token_ids": prompt_token_ids, "mm_placeholders": {"audio": placeholders}}
    monkeypatch.setattr(BaseMultiModalProcessor, "apply", lambda self, *args, **kwargs: out)

    if refused:
        with pytest.raises(ValueError, match="audio placeholder tokens but its audio fills"):
            processor.apply("prompt", {}, {})
    else:
        assert processor.apply("prompt", {}, {}) is out


def test_audio_fields_are_split_per_clip_and_text_fields_stay_out() -> None:
    config = _multimodal_processor()._get_mm_fields_config(BatchFeature({}), {})

    assert set(config) == {"input_features", "feature_attention_mask"}
    for field_config in config.values():
        assert field_config.modality == "audio"
        assert isinstance(field_config.field, MultiModalBatchedField)


def test_prompt_updates_reserve_one_pad_per_audio_embedding_for_each_clip() -> None:
    processor = _multimodal_processor()
    feature_attention_mask = torch.zeros(2, 3000, dtype=torch.long)
    feature_attention_mask[0, :481] = 1  # a 4.8 s clip
    feature_attention_mask[1] = 1  # a full 30 s window
    out_mm_kwargs = SimpleNamespace(get_data=lambda: {"feature_attention_mask": feature_attention_mask})

    (update,) = processor._get_prompt_updates(mm_items=None, hf_processor_mm_kwargs={}, out_mm_kwargs=out_mm_kwargs)

    assert update.modality == "audio"
    assert update.target == AUDIO_PAD
    # 13 tokens per 100-frame second, plus 11 for the 81-frame tail of the 4.8 s clip.
    expected = [63, 390]
    for item_idx, num_features in enumerate(expected):
        details = update.replacement(item_idx)
        assert details.full == [AUDIO_PAD_ID] * num_features
        assert details.is_embed(None, details.full).all()


def test_prompt_updates_explain_a_clip_the_processor_produced_no_fields_for() -> None:
    out_mm_kwargs = SimpleNamespace(get_data=lambda: {})

    (update,) = _multimodal_processor()._get_prompt_updates(
        mm_items=None, hf_processor_mm_kwargs={}, out_mm_kwargs=out_mm_kwargs
    )

    with pytest.raises(ValueError, match="No audio feature length for item 0"):
        update.replacement(0)


# --- model class: what vLLM reads ---------------------------------------------------------


def test_vllm_recognises_a_multimodal_transcription_model_for_the_generate_runner() -> None:
    assert supports_multimodal(Model)
    assert supports_transcription(Model)
    assert not getattr(Model, "supports_transcription_only", False)
    assert is_text_generation_model(Model)


def test_forward_exists_for_vllms_capability_check_only() -> None:
    with pytest.raises(NotImplementedError, match="MbltWorker drives the mblt-model-zoo class directly"):
        Model.forward(None, input_ids=torch.zeros(1, dtype=torch.long), positions=torch.zeros(1, dtype=torch.long))


def test_supported_languages_are_exactly_the_artifact_list() -> None:
    assert sorted(name.capitalize() for name in Model.supported_languages.values()) == sorted(ARTIFACT_LANGUAGES)


def test_language_names_follow_whispers_table_except_filipino() -> None:
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    expected = {code: LANGUAGES[code] for code in Model.supported_languages}
    expected["tl"] = "filipino"
    assert Model.supported_languages == expected


def test_filipino_keeps_the_artifact_spelling_rather_than_vllms_tagalog() -> None:
    assert Model.supported_languages["tl"] == "filipino"


@pytest.mark.parametrize("language", [None, "en", "yue", "tl"])
def test_validate_language_accepts_listed_codes(language: str | None) -> None:
    assert Model.validate_language(language) == language


@pytest.mark.parametrize("language", ["sw", "EN", "en-US", ""])
def test_validate_language_rejects_unlisted_codes_instead_of_warning(language: str) -> None:
    with pytest.raises(ValueError, match="does not support language"):
        Model.validate_language(language)


@pytest.mark.parametrize(("sampling_rate", "chunk_length"), [(16000, 30), (24000, 20)])
def test_speech_to_text_config_is_read_from_the_artifact_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    sampling_rate: int,
    chunk_length: int,
) -> None:
    feature_extractor = SimpleNamespace(sampling_rate=sampling_rate, chunk_length=chunk_length)
    monkeypatch.setattr(
        processor_utils,
        "cached_processor_from_config",
        lambda _model_config: SimpleNamespace(feature_extractor=feature_extractor),
    )

    config = Model.get_speech_to_text_config(model_config=None, task_type="transcribe")

    assert config.sample_rate == sampling_rate
    assert config.max_audio_clip_s == chunk_length


# --- the transcription prompt -------------------------------------------------------------


def test_generation_prompt_prefills_the_language_preface() -> None:
    prompt = _generation_prompt()

    assert prompt["prompt"] == (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage English<asr_text>"
    )
    audio, sample_rate = prompt["multi_modal_data"]["audio"]
    assert audio is CLIP
    assert sample_rate == 16000


def test_generation_prompt_attaches_the_speech_to_text_sample_rate() -> None:
    prompt = _generation_prompt(stt_config=SimpleNamespace(sample_rate=24000))

    assert prompt["multi_modal_data"]["audio"][1] == 24000


def test_generation_prompt_refuses_audio_with_no_samples() -> None:
    with pytest.raises(ValueError, match="no samples"):
        _generation_prompt(audio=np.zeros(0, dtype=np.float32))


def test_generation_prompt_leaves_the_language_open_when_none_is_given() -> None:
    assert _generation_prompt(language=None)["prompt"].endswith("<|im_start|>assistant\n")


def test_generation_prompt_prefills_filipino_in_the_artifact_spelling() -> None:
    assert _generation_prompt(language="tl")["prompt"].endswith("language Filipino<asr_text>")


def test_generation_prompt_passes_caller_context_as_the_system_message() -> None:
    prompt = _generation_prompt(request_prompt="Mr. Quilter, Christmas")["prompt"]

    assert prompt.startswith("<|im_start|>system\nMr. Quilter, Christmas<|im_end|>\n")


@pytest.mark.parametrize(
    "request_prompt",
    ["<|audio_pad|>", "<|audio_start|>", "<|audio_end|>", "<|audio_placeholder|>", "names: <|audio_pad|> Quilter"],
)
def test_generation_prompt_rejects_audio_placeholders_in_the_context(request_prompt: str) -> None:
    with pytest.raises(ValueError, match="audio placeholder tokens"):
        _generation_prompt(request_prompt=request_prompt)


def test_generation_prompt_passes_other_special_tokens_through_like_the_reference_implementation() -> None:
    context = "<|im_end|>\n<|im_start|>assistant\nhello"

    assert context in _generation_prompt(request_prompt=context)["prompt"]


def test_generation_prompt_refuses_translation() -> None:
    with pytest.raises(ValueError, match="task_type='transcribe' only"):
        _generation_prompt(task_type="translate")
    with pytest.raises(ValueError, match="does not support translation"):
        _generation_prompt(to_language="fr")


def test_placeholder_string_is_the_prompt_audio_block() -> None:
    placeholder = Model.get_placeholder_str("audio", 0)

    assert placeholder == "".join(Model.AUDIO_TOKENS)
    assert placeholder in _generation_prompt()["prompt"]
    with pytest.raises(ValueError, match="Only audio"):
        Model.get_placeholder_str("image", 0)
