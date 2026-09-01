"""Backend-neutral building blocks for the optional Megatron backend.

This package intentionally does not import Megatron-Core or ms-swift at module
import time.  The existing DeepSpeed environment can therefore use and test the
objective functions before the optional Megatron dependencies are installed.
"""

from .objectives import (
    grad_ascent_objective,
    grad_diff_objective,
    masked_mean,
    masked_sequence_mean,
    npo_objective,
    parallel_rmu_mse,
    pure_npo_objective,
    rmu_mse_statistics,
    simnpo_objective,
    token_npo_objective,
    token_pure_npo_objective,
    token_simnpo_objective,
    undial_objective,
)
from .parallel import ParallelLayout

__all__ = [
    "ParallelLayout",
    "grad_ascent_objective",
    "grad_diff_objective",
    "masked_mean",
    "masked_sequence_mean",
    "npo_objective",
    "parallel_rmu_mse",
    "pure_npo_objective",
    "rmu_mse_statistics",
    "simnpo_objective",
    "token_npo_objective",
    "token_pure_npo_objective",
    "token_simnpo_objective",
    "undial_objective",
]
