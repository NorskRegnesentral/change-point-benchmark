"""Benchmark case definitions.

Each module in this subpackage exposes:
- ``BENCHMARK_PAIRS``: mapping of pair name → factory callable
- ``collect_cases()``: convenience function to gather cases
"""

from change_bench.benchmarks.null_case import (
    BENCHMARK_PAIRS,
    PAIR_CATEGORIES,
    BenchmarkCase,
    collect_cases,
)

__all__ = [
    "BENCHMARK_PAIRS",
    "PAIR_CATEGORIES",
    "BenchmarkCase",
    "collect_cases",
]
