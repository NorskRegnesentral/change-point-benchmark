"""Comparison pair factories for benchmarking skchange vs ruptures.

Each submodule defines a single ``pair_*()`` factory function that returns
a list of :class:`~change_bench.benchmarks.comparison_pairs._common.BenchmarkCase`
instances (one for skchange, one for ruptures) for every problem it receives.
"""

from change_bench.benchmarks.comparison_pairs._common import BenchmarkCase
from change_bench.benchmarks.comparison_pairs.binseg import pair_binseg_l2_cusum
from change_bench.benchmarks.comparison_pairs.binseg_mv_gaussian import (
    pair_binseg_mv_gaussian,
)
from change_bench.benchmarks.comparison_pairs.moving_window_l1 import (
    pair_moving_window_l1,
)
from change_bench.benchmarks.comparison_pairs.moving_window_l2 import (
    pair_moving_window_l2,
)
from change_bench.benchmarks.comparison_pairs.moving_window_mv_gaussian import (
    pair_moving_window_mv_gaussian,
)
from change_bench.benchmarks.comparison_pairs.moving_window_rank import (
    pair_moving_window_rank,
)
from change_bench.benchmarks.comparison_pairs.pelt_gaussian import (
    pair_pelt_1d_gaussian,
)
from change_bench.benchmarks.comparison_pairs.pelt_l2 import pair_pelt_l2
from change_bench.benchmarks.comparison_pairs.pelt_linear_trend import (
    pair_pelt_linear_trend,
)
from change_bench.benchmarks.comparison_pairs.pelt_mv_gaussian import (
    pair_pelt_mv_gaussian,
)
from change_bench.benchmarks.comparison_pairs.pelt_poisson import (
    CostPoisson,
    pair_pelt_poisson,
)
from change_bench.benchmarks.comparison_pairs.pelt_rank import pair_pelt_rank

__all__ = [
    "BenchmarkCase",
    "CostPoisson",
    "pair_binseg_l2_cusum",
    "pair_binseg_mv_gaussian",
    "pair_moving_window_l1",
    "pair_moving_window_l2",
    "pair_moving_window_mv_gaussian",
    "pair_moving_window_rank",
    "pair_pelt_1d_gaussian",
    "pair_pelt_l2",
    "pair_pelt_linear_trend",
    "pair_pelt_mv_gaussian",
    "pair_pelt_poisson",
    "pair_pelt_rank",
]
