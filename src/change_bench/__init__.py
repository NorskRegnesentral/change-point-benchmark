"""change_bench – utilities for benchmarking change-point detection algorithms."""

from change_bench.datasets.null_case import NullDatasetConfig
from change_bench.problems.base import BenchmarkProblem, make_null_problems
from change_bench.runner import BenchmarkResult, run_benchmark

__all__ = [
    "BenchmarkProblem",
    "BenchmarkResult",
    "NullDatasetConfig",
    "make_null_problems",
    "run_benchmark",
]
