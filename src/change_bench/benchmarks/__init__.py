"""Benchmark case definitions.

Each module in this subpackage exposes:
- ``BENCHMARK_PAIRS``: mapping of pair name → factory callable
- ``collect_cases()``: convenience function to gather cases
"""

from change_bench.benchmarks.registry import (
    BENCHMARK_PAIRS,
    PAIR_CATEGORIES,
    Pair,
    collect_cases,
)
from change_bench.benchmarks.comparison_pairs import BenchmarkCase

__all__ = [
    "BENCHMARK_PAIRS",
    "PAIR_CATEGORIES",
    "BenchmarkCase",
    "Pair",
    "collect_cases",
]
