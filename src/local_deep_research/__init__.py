"""
Local Deep Research - A tool for conducting deep research using AI.
"""

import os
import platform


def _configure_apple_silicon_native_runtime() -> None:
    """Avoid a reproducible FAISS + PyTorch native-thread crash on macOS ARM.

    Loading/searching an ``IndexIDMap2`` and running sentence-transformers in
    the same Apple Silicon process can segfault inside PyTorch's layer norm
    when the default OpenMP worker pool is enabled.  ``OMP_NUM_THREADS=1`` is
    sufficient to make the native LibraryRAG path stable; VECLIB's thread cap
    alone is not.  Respect an explicit operator value and allow opting out for
    future runtimes where the upstream binary issue is fixed.
    """

    enabled = os.environ.get(
        "LDR_MACOS_FAISS_TORCH_SAFE_MODE", "true"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if (
        enabled
        and platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    ):
        os.environ.setdefault("OMP_NUM_THREADS", "1")


_configure_apple_silicon_native_runtime()

__author__ = "LearningCircuit"
__description__ = "A tool for conducting deep research using AI"

from loguru import logger

from .__version__ import __version__

# Disable logging by default to not interfere with user setup.
logger.disable("local_deep_research")

__all__ = [
    "__version__",
]
