"""Data parallel serving: one EngineCore per NPU card.

``vllm serve --data-parallel-size N`` starts N engine processes behind vLLM's own load balancer. On a non-CUDA
platform vLLM gives each engine process its device through the platform's ``device_control_env_var``: the parent's
value (or ``0..N-1`` when unset) is split, and local rank ``r`` gets the ``r``-th entry. The MBLT platform names
qbruntime's own ``QBRUNTIME_VISIBLE_DEVICES`` there, so each engine process starts with exactly one card visible.

The worker still builds the model through transformers-mblt ``from_pretrained``; this module only decides which card
an engine owns and rewrites the placement kwargs of every NPU role (``text_``/``vision_``, ``base_``/``draft_``/
``fc_``, ``encoder_``/``decoder_``, ``shared_``/``expert_``/``lm_head_``) onto that card, keeping each role's
cores and clusters as the release lays them out. qbruntime renumbers the visible cards from 0, so the engine's card
is ``dev_no`` 0 in its own process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

QBRUNTIME_VISIBLE_DEVICES_ENV = "QBRUNTIME_VISIBLE_DEVICES"
# Set by the process that checked the card assignment (the front end) and inherited by the engine processes it
# starts. vLLM sets QBRUNTIME_VISIBLE_DEVICES to one card in each engine process, and the platform config hook runs
# there again; the marker tells that run the assignment was already checked and exported.
DATA_PARALLEL_CARDS_CHECKED_ENV = "VLLM_MBLT_DATA_PARALLEL_CARDS_CHECKED"

# The NPU role prefixes a Mobilint release can use; "" is the single-backend model.
NPU_ROLE_PREFIXES = (
    "",
    "text_",
    "vision_",
    "encoder_",
    "decoder_",
    "base_",
    "draft_",
    "fc_",
    "shared_",
    "expert_",
    "lm_head_",
)
_PLACEMENT_FIELDS = ("dev_no", "target_cores", "target_clusters")

# An idle engine in a running wave repeats the dummy step until the wave ends, and all engines all-reduce their
# unfinished state every 32 steps. Pacing the dummy step like a real one keeps an idle engine from reaching that
# all-reduce long before its busy peers (where it would block, unable to take a new request) and from spinning a
# CPU core.
DEFAULT_IDLE_STEP_SECONDS = 0.05
MIN_IDLE_STEP_SECONDS = 0.001
MAX_IDLE_STEP_SECONDS = 0.25


def _coerce_dev_no_list(key: str, value: object) -> list[int] | None:
    """Return ``value`` as a card list, ``None`` for a single card; reject anything else."""
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a card number or a list of card numbers, got {value!r}.")
    if isinstance(value, (list, tuple)):
        cards = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{key} must list non-negative card numbers, got {value!r}.")
            cards.append(item)
        if not cards:
            raise ValueError(f"{key} is an empty list.")
        if len(set(cards)) != len(cards):
            raise ValueError(f"{key} lists a card more than once: {value!r}.")
        return cards
    return None


def _visible_card_list(visible: str) -> list[int]:
    """``QBRUNTIME_VISIBLE_DEVICES`` as distinct non-negative card numbers; reject anything else."""
    cards = []
    for entry in visible.split(","):
        entry = entry.strip()
        if not entry.isdigit():
            raise ValueError(f"{QBRUNTIME_VISIBLE_DEVICES_ENV} must list non-negative card numbers, got {visible!r}.")
        cards.append(int(entry))
    if len(set(cards)) != len(cards):
        raise ValueError(f"{QBRUNTIME_VISIBLE_DEVICES_ENV} lists a card more than once: {visible!r}.")
    return cards


def data_parallel_card_list(placement_kwargs: Mapping[str, object]) -> list[int] | None:
    """The cards the data parallel ranks use, from the ``dev_no`` placement kwargs.

    Every ``<role>dev_no`` must be a list, and every list the same: under data parallel an engine owns one card,
    and all of its roles run there. Local rank ``r`` takes the ``r``-th card. ``None`` when no ``dev_no`` is given.
    """
    lists: dict[str, list[int]] = {}
    for prefix in NPU_ROLE_PREFIXES:
        key = f"{prefix}dev_no"
        if key not in placement_kwargs or placement_kwargs[key] is None:
            continue
        cards = _coerce_dev_no_list(key, placement_kwargs[key])
        if cards is None:
            raise ValueError(
                f"{key}={placement_kwargs[key]!r} would put every data parallel engine on the same card. "
                f"Pass a list with one card per engine (for example {key}: [0, 1, 2]) or omit it."
            )
        lists[key] = cards
    if not lists:
        return None
    first_key, first_cards = next(iter(lists.items()))
    for key, cards in lists.items():
        if cards != first_cards:
            raise ValueError(
                "With data parallel each engine runs all of its NPU roles on its own card, so every dev_no list "
                f"must be the same; got {first_key}={first_cards} and {key}={cards}."
            )
    return first_cards


def export_card_list_for_ranks(
    placement_kwargs: Mapping[str, object],
    *,
    local_data_parallel_size: int,
    environ: MutableMapping[str, str] | None = None,
) -> list[int] | None:
    """Check the ``dev_no`` lists before the engines start, and hand them to vLLM's per-rank device split.

    vLLM gives local rank ``r`` the ``r``-th entry of ``QBRUNTIME_VISIBLE_DEVICES``, so a ``dev_no`` list is written
    there when the variable is unset. Setting both is refused: it is ambiguous which one names the cards. Only the
    first process checks; the engine processes it starts inherit its marker and keep their per-rank variable.
    """
    environ = os.environ if environ is None else environ
    cards = data_parallel_card_list(placement_kwargs)
    if DATA_PARALLEL_CARDS_CHECKED_ENV in environ:
        # An engine process: its QBRUNTIME_VISIBLE_DEVICES is the one card vLLM gave this rank.
        return cards
    visible = environ.get(QBRUNTIME_VISIBLE_DEVICES_ENV, "")
    if cards is None:
        if visible and len(_visible_card_list(visible)) < local_data_parallel_size:
            raise ValueError(
                f"{QBRUNTIME_VISIBLE_DEVICES_ENV}={visible!r} lists fewer cards than the "
                f"{local_data_parallel_size} local data parallel engines."
            )
    elif len(cards) < local_data_parallel_size:
        raise ValueError(
            f"dev_no lists {len(cards)} card(s) {cards}, fewer than the {local_data_parallel_size} local data "
            "parallel engines; each engine needs its own card."
        )
    elif visible:
        raise ValueError(
            f"Both {QBRUNTIME_VISIBLE_DEVICES_ENV}={visible!r} and a dev_no list {cards} name the data parallel "
            "cards; set only one."
        )
    else:
        environ[QBRUNTIME_VISIBLE_DEVICES_ENV] = ",".join(str(card) for card in cards)
    environ[DATA_PARALLEL_CARDS_CHECKED_ENV] = "1"
    return cards


def narrow_visible_devices_to_rank_card(
    parallel_config: object,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> int | None:
    """Show this data parallel engine only its own card, before qbruntime opens a device.

    vLLM 0.32 no longer narrows the device control variable for each engine process on platforms other than
    CUDA and XPU. It records the rank's physical cards in ``parallel_config.assigned_physical_gpu_ids`` instead,
    computed from ``QBRUNTIME_VISIBLE_DEVICES`` (or ``0..N-1``). Writing that card back to the variable restores
    the 0.11 layout the rest of this module expects: one visible card, opened as ``dev_no`` 0. Returns the card,
    or ``None`` for a single engine or when vLLM assigned none.
    """
    environ = os.environ if environ is None else environ
    if (getattr(parallel_config, "data_parallel_size", 1) or 1) <= 1:
        return None
    assigned = getattr(parallel_config, "assigned_physical_gpu_ids", None)
    if not assigned:
        return None
    if len(assigned) != 1:
        raise ValueError(f"A data parallel engine runs on one card, but vLLM assigned cards {list(assigned)}.")
    card = int(assigned[0])
    environ[QBRUNTIME_VISIBLE_DEVICES_ENV] = str(card)
    return card


def resolve_rank_card(
    placement_kwargs: Mapping[str, object],
    *,
    local_rank: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """The physical card this data parallel engine owns.

    A ``dev_no`` list names it by local rank. Otherwise vLLM has set ``QBRUNTIME_VISIBLE_DEVICES`` to this engine's
    card (one entry); a launcher that leaves the full list is indexed by local rank, and no variable at all means
    card ``local_rank``.
    """
    environ = os.environ if environ is None else environ
    visible = [entry.strip() for entry in environ.get(QBRUNTIME_VISIBLE_DEVICES_ENV, "").split(",") if entry.strip()]
    cards = data_parallel_card_list(placement_kwargs)
    if cards is not None:
        if local_rank >= len(cards):
            raise ValueError(
                f"Data parallel local rank {local_rank} has no card: dev_no lists only {len(cards)} card(s) {cards}."
            )
        card = cards[local_rank]
        if len(visible) == 1 and int(visible[0]) != card:
            raise ValueError(
                f"Data parallel local rank {local_rank} was given card {visible[0]} through "
                f"{QBRUNTIME_VISIBLE_DEVICES_ENV}, but dev_no names card {card}. Pass the dev_no list in "
                "--model-loader-extra-config so both agree."
            )
        return card
    if len(visible) == 1:
        return int(visible[0])
    if len(visible) > 1:
        if local_rank >= len(visible):
            raise ValueError(
                f"Data parallel local rank {local_rank} has no card: "
                f"{QBRUNTIME_VISIBLE_DEVICES_ENV}={','.join(visible)!r}."
            )
        return int(visible[local_rank])
    return local_rank


def runtime_device_number(card: int, *, environ: Mapping[str, str] | None = None) -> int:
    """The ``dev_no`` that opens physical ``card`` in this process.

    qbruntime renumbers the cards ``QBRUNTIME_VISIBLE_DEVICES`` lists from 0, as ``CUDA_VISIBLE_DEVICES`` does, and
    a hidden card fails to open (verified on qbruntime 1.5.0 with 3 cards). vLLM starts each data parallel engine
    with only its own card listed, so every role of the engine opens ``dev_no`` 0. Without the variable the
    physical number is used.
    """
    environ = os.environ if environ is None else environ
    visible = [entry.strip() for entry in environ.get(QBRUNTIME_VISIBLE_DEVICES_ENV, "").split(",") if entry.strip()]
    if not visible:
        return card
    if str(card) not in visible:
        raise ValueError(f"Card {card} is not in {QBRUNTIME_VISIBLE_DEVICES_ENV}={','.join(visible)!r}.")
    return visible.index(str(card))


def _retarget(items: object, device_number: int, *, grain: str) -> list[object]:
    """``items`` moved onto ``device_number``; legacy items without a device are kept for ``dev_no`` to qualify."""
    retargeted: list[object] = []
    for item in list(items or []):
        if isinstance(item, str):
            parts = item.split(":")
            # Canonical forms carry the device first: "d:c:k" for cores, "d:c" for clusters.
            if (grain == "cores" and len(parts) == 3) or (grain == "clusters" and len(parts) == 2):
                retargeted.append(":".join([str(device_number), *parts[1:]]))
                continue
        retargeted.append(item)
    return retargeted


def _canonical_devices(items: object, *, grain: str) -> set[str]:
    devices = set()
    for item in list(items or []):
        if isinstance(item, str):
            parts = item.split(":")
            if (grain == "cores" and len(parts) == 3) or (grain == "clusters" and len(parts) == 2):
                devices.add(parts[0])
    return devices


def _config_value(hf_config: object, name: str) -> object:
    try:
        return getattr(hf_config, name, None)
    except Exception:
        return None


def npu_roles(model_kwargs: Mapping[str, object], hf_config: object) -> list[str]:
    """The role prefixes whose NPU placement this model takes, from its config and the given kwargs.

    A role counts when the config exposes its targets or the kwargs place it. Prefixed roles replace the unprefixed
    one: a multi-backend release (VLM, EAGLE-3, ASR, MoE) is placed per role.
    """
    roles = []
    for prefix in NPU_ROLE_PREFIXES:
        configured = any(_config_value(hf_config, f"{prefix}{grain}") for grain in ("target_cores", "target_clusters"))
        given = any(f"{prefix}{field}" in model_kwargs for field in _PLACEMENT_FIELDS)
        if configured or given:
            roles.append(prefix)
    prefixed = [prefix for prefix in roles if prefix]
    return prefixed or [""]


def place_model_kwargs_on_card(
    model_kwargs: Mapping[str, object],
    hf_config: object,
    *,
    device_number: int,
) -> dict[str, object]:
    """Return ``model_kwargs`` with every NPU role placed on ``device_number``.

    Each role keeps its cores or clusters: the release's (or the caller's) targets are moved to the device, and
    ``<role>dev_no`` names it, so transformers-mblt builds the same layout on this engine's card. A role whose
    targets span several cards cannot be served by one engine.
    """
    roles = npu_roles(model_kwargs, hf_config)
    placed = dict(model_kwargs)
    if roles != [""]:
        for field in ("target_cores", "target_clusters"):
            if field in placed:
                raise ValueError(
                    f"With data parallel, {field} must name its NPU role ({', '.join(p + field for p in roles)})."
                )
        placed.pop("dev_no", None)
    for prefix in roles:
        cores_key, clusters_key = f"{prefix}target_cores", f"{prefix}target_clusters"
        if cores_key in model_kwargs or clusters_key in model_kwargs:
            cores, clusters = model_kwargs.get(cores_key), model_kwargs.get(clusters_key)
        else:
            cores, clusters = _config_value(hf_config, cores_key), _config_value(hf_config, clusters_key)
        devices = _canonical_devices(cores, grain="cores") | _canonical_devices(clusters, grain="clusters")
        if len(devices) > 1:
            raise ValueError(
                f"The {prefix or 'model'} NPU targets span cards {sorted(devices)}; with data parallel each engine "
                "runs on one card."
            )
        placed[f"{prefix}dev_no"] = device_number
        placed.pop(cores_key, None)
        placed.pop(clusters_key, None)
        if cores:
            placed[cores_key] = _retarget(cores, device_number, grain="cores")
        if clusters:
            placed[clusters_key] = _retarget(clusters, device_number, grain="clusters")
    return placed


class IdleStepPacer:
    """Paces the dummy step of an idle data parallel engine like this engine's recent real steps."""

    def __init__(self, default_seconds: float = DEFAULT_IDLE_STEP_SECONDS) -> None:
        self.step_seconds = default_seconds

    def observe_step(self, elapsed_seconds: float) -> None:
        self.step_seconds = 0.8 * self.step_seconds + 0.2 * max(0.0, elapsed_seconds)

    def idle_seconds(self) -> float:
        return min(MAX_IDLE_STEP_SECONDS, max(MIN_IDLE_STEP_SECONDS, self.step_seconds))


def collect_placement_kwargs(*sources: object) -> dict[str, Any]:
    """The ``<role>dev_no`` / target kwargs given in any of ``sources`` (dicts), later sources winning."""
    collected: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for prefix in NPU_ROLE_PREFIXES:
            for field in _PLACEMENT_FIELDS:
                key = f"{prefix}{field}"
                if key in source:
                    collected[key] = source[key]
    return collected
