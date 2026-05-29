import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from ltx_pipelines.utils.helpers import cleanup_memory

_M = TypeVar("_M", bound=torch.nn.Module)

logger = logging.getLogger(__name__)


def _keep_models() -> bool:
    """Check if persistent pipeline model caching is enabled."""
    return os.getenv("LTX_KEEP_PIPELINE_MODELS", "").lower() in {"1", "true", "yes", "on"}


@contextmanager
def gpu_model(model: _M) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.

    When ``LTX_KEEP_PIPELINE_MODELS`` is enabled, the cleanup is **skipped**
    so that models stay resident in GPU memory between pipeline stages.

    When the env var is not set (or set to a falsy value), moves all
    parameters and buffers to ``meta`` device on exit, which immediately
    releases the underlying storage on **both** GPU and CPU, then runs
    ``cleanup_memory()`` to reclaim fragmented CUDA memory.

    Usage::

        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder — typed as the concrete class
        # GPU + CPU memory freed automatically (unless persistence is on)
    """
    try:
        yield model
    finally:
        if _keep_models():
            logger.info(
                "[gpu_model] LTX_KEEP_PIPELINE_MODELS is ON — keeping %s in VRAM (skipping cleanup)",
                type(model).__name__,
            )
        else:
            torch.cuda.synchronize()
            # .to("meta") releases storage for all parameters/buffers regardless
            # of their original device (CUDA or CPU).
            model.to("meta")
            cleanup_memory()
