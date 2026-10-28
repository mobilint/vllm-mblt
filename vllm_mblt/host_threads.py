"""Per-engine torch CPU thread defaults for the Mobilint worker.

vLLM lowers ``OMP_NUM_THREADS`` to 1 only for workers started by its multiprocessing executor. vllm-mblt runs on
the ``uni`` executor, so each engine process keeps PyTorch's host-sized thread pool. Several such engines on one host
(one per card, or one per data-parallel rank) then compete for the same cores. The thread count comes from
transformers-mblt, which owns the policy and its measurements; this module only tells it how many engines share
the host.
"""

from typing import Any, Optional

from transformers_mblt.utils.host_threads import apply_host_thread_defaults
from vllm.logger import init_logger

from vllm_mblt.data_parallel import data_parallel_layout

logger = init_logger(f"vllm.{__name__}")


def local_engine_count(vllm_config: Any) -> int:
    """Return the number of engine processes on this host, as the data-parallel config describes it.

    Each local data-parallel rank is its own engine process, so this is the local data-parallel size. Without a
    data-parallel config it is one engine. Engines launched as separate ``vllm serve`` processes are not counted;
    set ``OMP_NUM_THREADS`` for those.

    Args:
        vllm_config: The engine's ``VllmConfig``.
    """
    return data_parallel_layout(vllm_config)[1]


def apply_engine_host_thread_defaults(vllm_config: Any) -> Optional[int]:
    """Size this engine's torch CPU thread pool for its share of the host.

    Leaves torch alone when ``OMP_NUM_THREADS`` or ``MKL_NUM_THREADS`` is set.

    Args:
        vllm_config: The engine's ``VllmConfig``.

    Returns:
        The thread count set, or ``None`` when an explicit environment setting was kept.
    """
    n_engines = local_engine_count(vllm_config)
    threads = apply_host_thread_defaults(n_engines_on_host=n_engines)
    if threads is None:
        logger.info("[mblt-init] host threads: keeping OMP_NUM_THREADS / MKL_NUM_THREADS from the environment")
    else:
        logger.info("[mblt-init] host threads: torch.set_num_threads(%d) for %d local engine(s)", threads, n_engines)
    return threads
