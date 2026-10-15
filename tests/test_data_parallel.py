"""Data parallel serving: one EngineCore per card (CPU-only; no NPU is opened)."""

import os
from types import SimpleNamespace

import pytest

from vllm_mblt import data_parallel
from vllm_mblt.data_parallel import (
    QBRUNTIME_VISIBLE_DEVICES_ENV,
    IdleStepPacer,
    collect_placement_kwargs,
    export_card_list_for_ranks,
    npu_roles,
    place_model_kwargs_on_card,
    probe_runtime_device_number,
    resolve_rank_card,
)
from vllm_mblt.mblt_platform import MbltPlatform
from vllm_mblt.mblt_worker import MbltWorker, _normalize_model_kwargs_for_hf_config

# Configs as vLLM loads them for released models: each NPU role exposes its canonical targets.
TEXT_CONFIG = SimpleNamespace(model_type="mobilint-qwen3", target_cores=[], target_clusters=["0:0", "0:1"])
VL_CONFIG = SimpleNamespace(
    model_type="mobilint-qwen3_vl",
    max_batch_size=1,
    text_target_cores=[],
    text_target_clusters=["0:0", "0:1"],
    vision_target_cores=[],
    vision_target_clusters=["0:0", "0:1"],
)
EAGLE3_CONFIG = SimpleNamespace(
    model_type="mobilint-qwen2-eagle3",
    base_target_cores=["0:0:0"],
    draft_target_cores=["0:0:0"],
    fc_target_cores=["0:0:0"],
)
MOE_CONFIG = SimpleNamespace(
    model_type="mobilint-qwen3_moe",
    shared_target_cores=["0:0:0"],
    expert_target_cores=[f"0:{cluster}:{core}" for cluster in range(2) for core in range(4)],
    lm_head_target_cores=["0:0:0"],
)
ASR_CONFIG = SimpleNamespace(
    model_type="mobilint-qwen3_asr",
    encoder_target_cores=["0:0:0"],
    decoder_target_clusters=["0:0", "0:1"],
)


class TestPlatform:
    def test_names_qbruntime_visible_devices_for_vllm_per_rank_split(self) -> None:
        assert MbltPlatform.device_control_env_var == "QBRUNTIME_VISIBLE_DEVICES"
        assert MbltPlatform.dist_backend == "gloo"

    def test_vllm_gives_each_local_rank_its_entry(self, monkeypatch) -> None:
        monkeypatch.setenv(QBRUNTIME_VISIBLE_DEVICES_ENV, "3,4,5")
        assert [MbltPlatform.device_id_to_physical_device_id(rank) for rank in range(3)] == [3, 4, 5]
        monkeypatch.delenv(QBRUNTIME_VISIBLE_DEVICES_ENV)
        assert [MbltPlatform.device_id_to_physical_device_id(rank) for rank in range(3)] == [0, 1, 2]

    @staticmethod
    def _vllm_config(extra_config, *, data_parallel_size, data_parallel_size_local=None):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=data_parallel_size,
                data_parallel_size_local=data_parallel_size_local or data_parallel_size,
            ),
            load_config=SimpleNamespace(model_loader_extra_config=extra_config),
        )

    def test_dev_no_list_becomes_the_rank_card_list(self, monkeypatch) -> None:
        monkeypatch.delenv(QBRUNTIME_VISIBLE_DEVICES_ENV, raising=False)
        MbltPlatform._prepare_data_parallel(self._vllm_config({"dev_no": [2, 0, 1]}, data_parallel_size=3))
        assert os.environ[QBRUNTIME_VISIBLE_DEVICES_ENV] == "2,0,1"

    def test_single_engine_leaves_the_environment_alone(self, monkeypatch) -> None:
        monkeypatch.delenv(QBRUNTIME_VISIBLE_DEVICES_ENV, raising=False)
        MbltPlatform._prepare_data_parallel(self._vllm_config({"dev_no": [0, 1]}, data_parallel_size=1))
        assert QBRUNTIME_VISIBLE_DEVICES_ENV not in os.environ

    def test_single_card_dev_no_is_refused_under_data_parallel(self, monkeypatch) -> None:
        monkeypatch.delenv(QBRUNTIME_VISIBLE_DEVICES_ENV, raising=False)
        with pytest.raises(ValueError, match="same card"):
            MbltPlatform._prepare_data_parallel(self._vllm_config({"dev_no": 0}, data_parallel_size=3))


class TestCardList:
    def test_too_few_cards_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fewer than the 3"):
            export_card_list_for_ranks({"dev_no": [0, 1]}, local_data_parallel_size=3, environ={})

    def test_role_lists_must_agree(self) -> None:
        with pytest.raises(ValueError, match="must be the same"):
            export_card_list_for_ranks(
                {"text_dev_no": [0, 1], "vision_dev_no": [1, 0]}, local_data_parallel_size=2, environ={}
            )

    def test_role_lists_that_agree_are_one_card_list(self) -> None:
        environ: dict[str, str] = {}
        cards = export_card_list_for_ranks(
            {"base_dev_no": [1, 2], "draft_dev_no": [1, 2], "fc_dev_no": [1, 2]},
            local_data_parallel_size=2,
            environ=environ,
        )
        assert cards == [1, 2]
        assert environ == {QBRUNTIME_VISIBLE_DEVICES_ENV: "1,2"}

    def test_both_env_and_dev_no_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="set only one"):
            export_card_list_for_ranks(
                {"dev_no": [0, 1]}, local_data_parallel_size=2, environ={QBRUNTIME_VISIBLE_DEVICES_ENV: "0,1"}
            )

    def test_short_env_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fewer cards"):
            export_card_list_for_ranks({}, local_data_parallel_size=3, environ={QBRUNTIME_VISIBLE_DEVICES_ENV: "0,1"})

    def test_no_placement_keeps_vllm_rank_numbering(self) -> None:
        environ: dict[str, str] = {}
        assert export_card_list_for_ranks({}, local_data_parallel_size=3, environ=environ) is None
        assert environ == {}

    def test_collects_placement_kwargs_of_every_role(self) -> None:
        collected = collect_placement_kwargs(
            {"draft_dev_no": [0, 1], "core_mode": "single", "lm_head_target_cores": ["0:0:1"]}, None
        )
        assert collected == {"draft_dev_no": [0, 1], "lm_head_target_cores": ["0:0:1"]}


class TestRankCard:
    def test_reads_the_card_vllm_set_for_this_rank(self) -> None:
        assert resolve_rank_card({}, local_rank=2, environ={QBRUNTIME_VISIBLE_DEVICES_ENV: "5"}) == 5

    def test_indexes_a_full_list_by_local_rank(self) -> None:
        assert resolve_rank_card({}, local_rank=1, environ={QBRUNTIME_VISIBLE_DEVICES_ENV: "3,4,5"}) == 4

    def test_defaults_to_the_local_rank(self) -> None:
        assert resolve_rank_card({}, local_rank=2, environ={}) == 2

    def test_dev_no_list_names_the_card(self) -> None:
        kwargs = {"text_dev_no": [3, 4, 5], "vision_dev_no": [3, 4, 5]}
        assert resolve_rank_card(kwargs, local_rank=1, environ={QBRUNTIME_VISIBLE_DEVICES_ENV: "4"}) == 4

    def test_dev_no_list_disagreeing_with_vllm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names card 4"):
            resolve_rank_card({"dev_no": [3, 4]}, local_rank=1, environ={QBRUNTIME_VISIBLE_DEVICES_ENV: "1"})

    def test_rank_past_the_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has no card"):
            resolve_rank_card({"dev_no": [0, 1]}, local_rank=2, environ={})


class TestRuntimeDeviceNumber:
    def test_card_zero_needs_no_probe(self) -> None:
        def fail(dev_no):
            raise AssertionError("probed")

        assert probe_runtime_device_number(0, fail) == 0

    def test_physical_numbering_keeps_the_card(self) -> None:
        opened = []
        assert probe_runtime_device_number(2, lambda dev_no: opened.append(dev_no) or object()) == 2
        assert opened == [2]

    def test_renumbered_visible_cards_start_at_zero(self) -> None:
        def renumbered(dev_no):
            if dev_no != 0:
                raise RuntimeError("Acc_DeviceNotFound occurred.")
            return object()

        assert probe_runtime_device_number(2, renumbered) == 0

    def test_card_error_surfaces_when_no_card_opens(self) -> None:
        def broken(dev_no):
            raise RuntimeError(f"driver error on {dev_no}")

        with pytest.raises(RuntimeError, match="driver error on 2"):
            probe_runtime_device_number(2, broken)


class TestPlacement:
    def test_single_backend_text_model(self) -> None:
        assert npu_roles({}, TEXT_CONFIG) == [""]
        placed = place_model_kwargs_on_card({"max_batch_size": 16}, TEXT_CONFIG, device_number=2)
        assert placed == {"max_batch_size": 16, "dev_no": 2, "target_clusters": ["2:0", "2:1"]}

    def test_vlm_places_vision_and_text_roles(self) -> None:
        kwargs = _normalize_model_kwargs_for_hf_config({"dev_no": [3, 4, 5], "core_mode": "global8"}, VL_CONFIG)
        placed = place_model_kwargs_on_card(kwargs, VL_CONFIG, device_number=4)
        assert placed == {
            "vision_dev_no": 4,
            "vision_core_mode": "global8",
            "vision_target_clusters": ["4:0", "4:1"],
            "text_dev_no": 4,
            "text_core_mode": "global8",
            "text_target_clusters": ["4:0", "4:1"],
        }

    def test_eagle3_draft_keeps_its_single_core(self) -> None:
        # A bare draft_dev_no would expand the draft MXQ to every core of the card.
        placed = place_model_kwargs_on_card({"dev_no": [0, 1]}, EAGLE3_CONFIG, device_number=1)
        assert placed == {
            "base_dev_no": 1,
            "base_target_cores": ["1:0:0"],
            "draft_dev_no": 1,
            "draft_target_cores": ["1:0:0"],
            "fc_dev_no": 1,
            "fc_target_cores": ["1:0:0"],
        }

    def test_mixture_of_experts_roles_keep_their_cores(self) -> None:
        placed = place_model_kwargs_on_card({"num_expert_workers": 16}, MOE_CONFIG, device_number=2)
        assert placed["shared_target_cores"] == ["2:0:0"]
        assert placed["lm_head_target_cores"] == ["2:0:0"]
        assert placed["expert_target_cores"] == [f"2:{cluster}:{core}" for cluster in range(2) for core in range(4)]
        assert placed["num_expert_workers"] == 16
        assert {placed[f"{role}_dev_no"] for role in ("shared", "expert", "lm_head")} == {2}

    def test_speech_encoder_and_decoder_roles(self) -> None:
        placed = place_model_kwargs_on_card({}, ASR_CONFIG, device_number=1)
        assert placed == {
            "encoder_dev_no": 1,
            "encoder_target_cores": ["1:0:0"],
            "decoder_dev_no": 1,
            "decoder_target_clusters": ["1:0", "1:1"],
        }

    def test_caller_targets_win_and_legacy_items_follow_dev_no(self) -> None:
        placed = place_model_kwargs_on_card(
            {"lm_head_target_cores": ["0:0:1"], "draft_target_cores": ["1:0"]},
            SimpleNamespace(**vars(MOE_CONFIG), draft_target_cores=["0:0:0"]),
            device_number=3,
        )
        assert placed["lm_head_target_cores"] == ["3:0:1"]
        assert placed["draft_target_cores"] == ["1:0"]
        assert placed["draft_dev_no"] == 3

    def test_role_spanning_cards_is_refused(self) -> None:
        config = SimpleNamespace(target_cores=[], target_clusters=["0:0", "1:0"])
        with pytest.raises(ValueError, match="span cards"):
            place_model_kwargs_on_card({}, config, device_number=0)

    def test_unprefixed_targets_on_a_multi_role_model_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must name its NPU role"):
            place_model_kwargs_on_card({"target_cores": ["0:0:0"]}, EAGLE3_CONFIG, device_number=1)


class TestTransformersMbltConfigs:
    """The placed kwargs build the release layout on the rank's card through the transformers-mblt configs."""

    @staticmethod
    def _shifted(values, device_number):
        return [f"{device_number}:{value.split(':', 1)[1]}" for value in values]

    def test_eagle3_roles(self) -> None:
        from transformers_mblt.models.qwen3_eagle3.configuration_qwen3_eagle3 import MobilintQwen3Eagle3Config

        release = MobilintQwen3Eagle3Config()
        placed = MobilintQwen3Eagle3Config(**place_model_kwargs_on_card({}, release, device_number=2))
        for role in ("base_", "draft_", "fc_"):
            for grain in ("target_cores", "target_clusters"):
                assert getattr(placed, role + grain) == self._shifted(getattr(release, role + grain), 2)
            assert getattr(placed, role + "dev_no") == 2

    def test_mixture_of_experts_roles(self) -> None:
        from transformers_mblt.models.qwen3_moe.configuration_qwen3_moe import MobilintQwen3MoEConfig

        release = MobilintQwen3MoEConfig()
        placed = MobilintQwen3MoEConfig(**place_model_kwargs_on_card({}, release, device_number=1))
        for role in ("shared_", "expert_", "lm_head_"):
            assert getattr(placed, role + "target_cores") == self._shifted(getattr(release, role + "target_cores"), 1)

    def test_vlm_roles(self) -> None:
        from transformers_mblt.models.qwen3_vl.configuration_qwen3_vl import MobilintQwen3VLConfig

        release = MobilintQwen3VLConfig()
        placed = MobilintQwen3VLConfig(**place_model_kwargs_on_card({}, release, device_number=2))
        for role in ("vision_", "text_"):
            for grain in ("target_cores", "target_clusters"):
                assert getattr(placed, role + grain) == self._shifted(getattr(release, role + grain), 2)


class TestWorker:
    @staticmethod
    def _worker(*, data_parallel_size, local_rank=0):
        worker = object.__new__(MbltWorker)
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=data_parallel_size,
                data_parallel_rank=local_rank,
                data_parallel_rank_local=local_rank,
            )
        )
        worker.rank = 0
        worker.local_rank = 0
        return worker

    def test_single_engine_keeps_kwargs(self) -> None:
        kwargs = {"dev_no": [0, 1], "max_batch_size": 32}
        assert self._worker(data_parallel_size=1)._place_on_data_parallel_card(kwargs, TEXT_CONFIG) is kwargs

    def test_rank_places_vlm_on_its_card(self, monkeypatch) -> None:
        import vllm_mblt.mblt_worker as worker_module

        monkeypatch.setenv(QBRUNTIME_VISIBLE_DEVICES_ENV, "5")
        monkeypatch.setattr(worker_module, "probe_runtime_device_number", lambda card: card)
        kwargs = _normalize_model_kwargs_for_hf_config({"max_batch_size": 16}, VL_CONFIG)
        placed = self._worker(data_parallel_size=3, local_rank=2)._place_on_data_parallel_card(kwargs, VL_CONFIG)
        assert placed["text_dev_no"] == placed["vision_dev_no"] == 5
        assert placed["text_target_clusters"] == ["5:0", "5:1"]
        assert placed["text_max_batch_size"] == placed["vision_max_batch_size"] == 16

    def test_rank_places_eagle3_draft_with_renumbered_cards(self, monkeypatch) -> None:
        import vllm_mblt.mblt_worker as worker_module

        monkeypatch.setenv(QBRUNTIME_VISIBLE_DEVICES_ENV, "2")
        monkeypatch.setattr(worker_module, "probe_runtime_device_number", lambda card: 0)
        placed = self._worker(data_parallel_size=3, local_rank=1)._place_on_data_parallel_card(
            {"draft_dev_no": [1, 2, 0]}, EAGLE3_CONFIG
        )
        assert placed["draft_dev_no"] == placed["base_dev_no"] == placed["fc_dev_no"] == 0
        assert placed["draft_target_cores"] == ["0:0:0"]

    def test_dummy_batch_runs_nothing_and_waits_about_one_step(self, monkeypatch) -> None:
        import vllm_mblt.mblt_worker as worker_module

        slept = []
        monkeypatch.setattr(worker_module.time, "sleep", slept.append)
        worker = self._worker(data_parallel_size=2)
        assert worker.execute_dummy_batch() is None
        assert slept == [data_parallel.DEFAULT_IDLE_STEP_SECONDS]

        worker._idle_step_pacer = IdleStepPacer(default_seconds=0.08)
        worker.execute_dummy_batch()
        assert slept[-1] == pytest.approx(0.08)

    def test_pacer_follows_model_steps_within_bounds(self) -> None:
        pacer = IdleStepPacer(default_seconds=0.05)
        for _ in range(50):
            pacer.observe_step(0.09)
        assert pacer.idle_seconds() == pytest.approx(0.09, rel=1e-3)
        for _ in range(50):
            pacer.observe_step(5.0)
        assert pacer.idle_seconds() == data_parallel.MAX_IDLE_STEP_SECONDS
        for _ in range(200):
            pacer.observe_step(0.0)
        assert pacer.idle_seconds() == data_parallel.MIN_IDLE_STEP_SECONDS

    def test_model_steps_feed_the_pacer(self, monkeypatch) -> None:
        worker = self._worker(data_parallel_size=2)
        monkeypatch.setattr(MbltWorker, "_execute_scheduled_step", lambda self, output: "ran")
        assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=0)) == "ran"
        assert worker._idle_step_pacer is None
        assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4)) == "ran"
        assert worker._idle_step_pacer is not None
