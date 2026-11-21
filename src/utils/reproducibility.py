from __future__ import annotations

import os
import random

import numpy as np

# This module provides utilities for reproducible execution of Python code, 
# particularly in the context of random number generation. 
# It includes functions to set global random seeds and create isolated random number generators.


def _validate_seed(seed: int) -> int:
    """Validate that `seed` is a strict non-negative integer (not bool)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int (not bool), got {type(seed).__name__}.")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    return seed


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy global random seeds for reproducible execution.

    Important Python runtime limitation regarding `PYTHONHASHSEED`:
    Setting `os.environ["PYTHONHASHSEED"]` at runtime does NOT change the hash
    randomization seed of the already-running Python interpreter process. It
    only propagates to child processes spawned after this call. For full
    process-level hash determinism (e.g., set/dict iteration order across runs),
    `PYTHONHASHSEED` must be exported in the shell environment before launching
    Python (for example: `PYTHONHASHSEED=42 uv run ...`).

    Note:
    This module intentionally contains no TensorFlow/Keras imports or APIs.
    """
    validated_seed = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(validated_seed)
    random.seed(validated_seed)
    np.random.seed(validated_seed)


def create_rng(seed: int) -> np.random.Generator:
    """Create an isolated, deterministic NumPy random Generator."""
    validated_seed = _validate_seed(seed)
    return np.random.default_rng(validated_seed)
