"""vLLM wrapper for Mobilint's NPU Qwen3-ASR.

The model zoo already implements Qwen3-ASR on the NPU: an audio encoder MXQ
produces embeddings which are spliced into a decoder-only causal LM at
``<|audio_pad|>`` positions. That is structurally the same arrangement as
Qwen3-VL, so this file supplies the vLLM side: multimodal processing and the
transcription interface.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (
    MobilintQwen3ASRForConditionalGeneration as OriginalMobilintQwen3ASRForConditionalGeneration,
)
from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
    Qwen3ASRProcessor,
    _get_feat_extract_output_lengths,
)
from transformers import BatchFeature
from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.model_executor.models import VllmModelForTextGeneration
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsTranscription
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, BaseDummyOptions

# One clip per request: the transcription endpoint sends one per chunk, and chat requests with
# more are refused with a 400. This is a serving choice; the worker encodes each clip on its own.
_MAX_AUDIOS_PER_REQUEST = 1


class MobilintQwen3ASRProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3ASRProcessor:
        return self.ctx.get_hf_processor(Qwen3ASRProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": _MAX_AUDIOS_PER_REQUEST}


class MobilintQwen3ASRDummyInputsBuilder(BaseDummyInputsBuilder[MobilintQwen3ASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return self.info.get_hf_processor().audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios, overrides=audio_overrides)}


def _qwen3_asr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
    )


class MobilintQwen3ASRMultiModalProcessor(BaseMultiModalProcessor[MobilintQwen3ASRProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_kwargs:
            # Overrides such as feature_size change the features the compiled encoder receives.
            # Refuse them here, so the caller gets a 400.
            raise ValueError(f"Qwen3-ASR does not accept processor overrides; got {sorted(mm_kwargs)}.")
        # vLLM passes audio as "audios"; the Qwen3-ASR processor takes "audio".
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios:
            mm_data["audio"] = audios
        # vLLM's text-only pass keeps the audio placeholder but sends no clip, which the
        # processor cannot expand; tokenize directly.
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_kwargs = {"sampling_rate": self.info.get_feature_extractor().sampling_rate}
        outputs = super()._call_hf_processor(prompt=prompt, mm_data=mm_data, mm_kwargs=mm_kwargs, tok_kwargs=tok_kwargs)
        # Samples far outside [-1, 1] overflow the log-mel features; refuse them rather than
        # hand inf or NaN to the NPU encoder.
        input_features = outputs.get("input_features")
        if input_features is not None and not torch.isfinite(torch.as_tensor(input_features)).all():
            raise ValueError("The audio's samples are out of range: its features are not finite.")
        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen3_asr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        vocab = self.info.get_tokenizer().get_vocab()
        audio_token_id = vocab[processor.audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            audio_output_lengths = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1)).tolist()

        def get_replacement(item_idx: int):
            if item_idx >= len(audio_output_lengths):
                # Reached when the processor produced no audio fields, which
                # otherwise surfaces as an opaque IndexError.
                raise ValueError(
                    f"No audio feature length for item {item_idx}; "
                    f"got {len(audio_output_lengths)} length(s) from the processor."
                )
            num_features = audio_output_lengths[item_idx]
            # The chat template already supplies <|audio_start|>/<|audio_end|>;
            # the HF processor expands only the pad token, so mirror that here.
            return PromptUpdateDetails.select_token_id(
                [audio_token_id] * num_features,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=processor.audio_token,
                replacement=get_replacement,
            )
        ]

    def apply(self, *args, **kwargs):
        out = super().apply(*args, **kwargs)
        # vLLM refuses a prompt that fills max_model_len only for text-only models; a multimodal one
        # is admitted with no room to generate and never finishes.
        max_model_len = self.info.ctx.model_config.max_model_len
        if len(out["prompt_token_ids"]) >= max_model_len:
            raise ValueError(
                f"The prompt is {len(out['prompt_token_ids'])} tokens, which leaves no room to generate within "
                f"max_model_len={max_model_len}. Shorten the prompt or the audio."
            )
        # Every <|audio_pad|> must belong to a clip. vLLM expands the first pad it finds, so one in
        # message text takes the clip's embeddings, and a clip with no samples leaves its pad as text.
        # Refusing after super().apply() is safe because the platform keeps the processor cache off
        # for this model.
        audio_token_id = self.info.get_tokenizer().get_vocab()[self.info.get_hf_processor().audio_token]
        num_pads = out["prompt_token_ids"].count(audio_token_id)
        num_slots = sum(p.get_num_embeds() for p in out["mm_placeholders"].get("audio", []))
        if num_pads != num_slots:
            raise ValueError(
                f"The prompt has {num_pads} audio placeholder tokens but its audio fills {num_slots}. "
                "Keep <|audio_pad|> out of message text, and send audio that has samples."
            )
        return out


@MULTIMODAL_REGISTRY.register_processor(
    MobilintQwen3ASRMultiModalProcessor,
    info=MobilintQwen3ASRProcessingInfo,
    dummy_inputs=MobilintQwen3ASRDummyInputsBuilder,
)
class MobilintQwen3ASRForConditionalGeneration(
    OriginalMobilintQwen3ASRForConditionalGeneration,
    SupportsMultiModal,
    SupportsTranscription,
    VllmModelForTextGeneration,
):
    merge_by_field_config = True

    # The artifact separates its language preface from the transcription with
    # this marker. The zoo carries the matching token id as
    # ``MobilintQwen3ASRThinkerConfig.asr_text_token_id`` (a constructor default, not a
    # field of the published config.json).
    ASR_TEXT_MARKER = "<asr_text>"

    # The chat template frames each clip with these; the prompt below and
    # get_placeholder_str spell them the same way.
    AUDIO_TOKENS = ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>")

    # The Qwen3-ASR processor expands each <|audio_pad|> into copies of this marker and then
    # turns every copy in the text back into <|audio_pad|>, a caller's copies included.
    AUDIO_PLACEHOLDER_MARKER = "<|audio_placeholder|>"

    # The artifact's own ``config.support_languages``, all 30 of them, keyed by the code vLLM
    # validates against. The values are pre-filled into the prompt, capitalised, so they keep the
    # artifact's spelling: ``tl`` -> "filipino", where vLLM's table says "tagalog". vLLM only
    # checks the keys.
    # fmt: off
    supported_languages = {
        "zh": "chinese", "en": "english", "yue": "cantonese", "ar": "arabic",
        "de": "german", "fr": "french", "es": "spanish", "pt": "portuguese",
        "id": "indonesian", "it": "italian", "ko": "korean", "ru": "russian",
        "th": "thai", "vi": "vietnamese", "ja": "japanese", "tr": "turkish",
        "hi": "hindi", "ms": "malay", "nl": "dutch", "sv": "swedish",
        "da": "danish", "fi": "finnish", "pl": "polish", "cs": "czech",
        "fa": "persian", "el": "greek", "ro": "romanian", "hu": "hungarian",
        "mk": "macedonian", "tl": "filipino",
    }
    # fmt: on

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        """Refuse a language the artifact does not list.

        For a known language outside ``supported_languages`` the base implementation
        logs a warning and proceeds. For this model that means auto-detect, whose
        "language <Lang><asr_text>" preface then lands in the OpenAI ``text`` field, so a
        400 is the better answer.
        """
        if language is None or language in cls.supported_languages:
            return language
        raise ValueError(
            f"Qwen3-ASR does not support language {language!r}. Must be one of {sorted(cls.supported_languages)}."
        )

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: Literal["transcribe", "translate"],
    ) -> SpeechToTextConfig:
        """Describe the audio front end to vLLM's speech-to-text server.

        Sample rate and window length come from the artifact's feature extractor.
        On /v1/audio/transcriptions vLLM splits longer files at this window (chunking
        is on by default); the NPU encoder itself runs on 1 s slices.
        """
        from vllm.transformers_utils.processor import cached_processor_from_config

        feature_extractor = cached_processor_from_config(model_config).feature_extractor
        return SpeechToTextConfig(
            sample_rate=feature_extractor.sampling_rate,
            max_audio_clip_s=feature_extractor.chunk_length,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config,
        model_config,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ):
        """Build the chat prompt the artifact expects for a transcription.

        The model answers in its native ``language <Lang><asr_text><text>``
        form. When the caller names a language the model supports, we pre-fill
        that preface so the model continues with the transcription alone: the
        API contract is plain text and vLLM 0.11.2 offers no hook to strip a
        prefix afterwards.

        Without a language the model auto-detects, and the preface is returned
        as part of the text, once per chunk. A language outside
        ``supported_languages`` cannot reach here: ``validate_language`` has
        already rejected it.

        A caller-supplied ``prompt`` becomes the system message, the slot the
        model reads context from; empty by default.
        """
        if task_type != "transcribe":
            # vLLM serves /v1/audio/translations through a handler built from the same
            # capability flag as transcription ("transcription" in supported_tasks), but
            # Qwen3-ASR cannot translate. Refuse here so the endpoint returns a client
            # error rather than a source-language transcript labelled as a translation.
            raise ValueError(f"Qwen3-ASR supports task_type='transcribe' only, got {task_type!r}.")
        if to_language:
            raise ValueError("Qwen3-ASR does not support translation; `to_language` must be unset.")

        if audio.size == 0:
            raise ValueError("The audio has no samples.")

        refused_tokens = (*cls.AUDIO_TOKENS, cls.AUDIO_PLACEHOLDER_MARKER)
        if any(token in request_prompt for token in refused_tokens):
            # The processor turns every <|audio_pad|> it finds, and every copy of its
            # marker, into audio slots for the one clip. In the caller's text they would
            # move the audio into the system message or leave the processor short of
            # clips, so refuse them up front.
            raise ValueError(f"`prompt` must not contain the audio placeholder tokens {refused_tokens}.")

        prefix = ""
        name = cls.supported_languages.get(language) if language else None
        if name:
            prefix = f"language {name.capitalize()}{cls.ASR_TEXT_MARKER}"
        # The OpenAI `prompt` field carries optional context -- names, jargon, the
        # topic. Qwen3-ASR takes it as the system message, exactly as the model's own
        # qwen_asr.inference.qwen3_asr._build_messages does, and vLLM passes the same
        # string for every chunk of a long file, so each piece is decoded with it.
        return {
            "prompt": (
                f"<|im_start|>system\n{request_prompt}<|im_end|>\n"
                "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                "<|im_start|>assistant\n" + prefix
            ),
            "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
        }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError("Only audio modality is supported")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: object = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        """Declare a vLLM-shaped forward so the model is recognised as generative.

        ``interfaces_base._check_vllm_model_forward`` requires ``input_ids`` and
        ``positions`` on ``forward``; the zoo class inherits only
        ``nn.Module._forward_unimplemented``, so without this signature vLLM
        refuses the ``generate`` runner. The body is never reached: MbltWorker
        replaces the model runner and drives the zoo object directly, so this
        class is only ever inspected, never instantiated.
        """
        raise NotImplementedError(
            "MbltWorker drives the mblt-model-zoo class directly; this signature exists for vLLM's capability check."
        )
