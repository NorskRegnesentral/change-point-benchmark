"""change_bench – utilities for benchmarking change-point detection algorithms."""

from change_bench.datasets.null_case import NullDatasetConfig
from change_bench.problems.base import BenchmarkProblem, make_null_problems

__all__ = [
    "BenchmarkProblem",
    "NullDatasetConfig",
    "make_null_problems",
]
