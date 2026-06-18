"""Benchmark case definitions.

Each module in this subpackage exposes:
- ``BENCHMARK_GROUPS``: mapping of group name → factory callable
- ``collect_cases()``: convenience function to gather cases
"""

from change_bench.benchmarks.null_case import (
    BENCHMARK_GROUPS,
    BenchmarkCase,
    collect_cases,
)

__all__ = [
    "BENCHMARK_GROUPS",
    "BenchmarkCase",
    "collect_cases",
]
