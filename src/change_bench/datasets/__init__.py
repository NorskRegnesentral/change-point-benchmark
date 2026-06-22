"""Dataset generation utilities for change-point benchmarks."""

from change_bench.datasets.change_case import ChangeDatasetConfig, SegmentParams
from change_bench.datasets.null_case import NAMED_DISTRIBUTIONS, NullDatasetConfig

__all__ = [
    "ChangeDatasetConfig",
    "NAMED_DISTRIBUTIONS",
    "NullDatasetConfig",
    "SegmentParams",
]
