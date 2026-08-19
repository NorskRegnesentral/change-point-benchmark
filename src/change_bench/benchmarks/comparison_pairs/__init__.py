"""Comparison pair factories for benchmarking skchange vs ruptures.

Each submodule defines a single ``pair_*()`` factory function that returns
a list of :class:`~change_bench.benchmarks.comparison_pairs._common.BenchmarkCase`
instances for every supported package and problem it receives.
"""

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.benchmarks.comparison_pairs.binseg import pair_binseg_l2_cusum
from change_bench.benchmarks.comparison_pairs.binseg_continuous_linear_trend import (
    pair_binseg_continuous_linear_trend,
)
from change_bench.benchmarks.comparison_pairs.binseg_esac import pair_binseg_esac
from change_bench.benchmarks.comparison_pairs.binseg_l1 import pair_binseg_l1
from change_bench.benchmarks.comparison_pairs.binseg_linreg import pair_binseg_linreg
from change_bench.benchmarks.comparison_pairs.binseg_mv_gaussian import (
    pair_binseg_mv_gaussian,
)
from change_bench.benchmarks.comparison_pairs.binseg_poisson import (
    pair_binseg_poisson,
)
from change_bench.benchmarks.comparison_pairs.binseg_rank import pair_binseg_rank
from change_bench.benchmarks.comparison_pairs.fpop_l2 import pair_fpop_l2
from change_bench.benchmarks.comparison_pairs.moving_window_continuous_linear_trend import (  # noqa: E501
    pair_moving_window_continuous_linear_trend,
)
from change_bench.benchmarks.comparison_pairs.moving_window_esac import (
    pair_moving_window_esac,
)
from change_bench.benchmarks.comparison_pairs.moving_window_l1 import (
    pair_moving_window_l1,
)
from change_bench.benchmarks.comparison_pairs.moving_window_l2 import (
    pair_moving_window_l2,
)
from change_bench.benchmarks.comparison_pairs.moving_window_linreg import (
    pair_moving_window_linreg,
)
from change_bench.benchmarks.comparison_pairs.moving_window_mv_gaussian import (
    pair_moving_window_mv_gaussian,
)
from change_bench.benchmarks.comparison_pairs.moving_window_poisson import (
    pair_moving_window_poisson,
)
from change_bench.benchmarks.comparison_pairs.moving_window_rank import (
    pair_moving_window_rank,
)
from change_bench.benchmarks.comparison_pairs.pelt_gaussian import (
    pair_pelt_1d_gaussian,
)
from change_bench.benchmarks.comparison_pairs.pelt_l1 import pair_pelt_l1
from change_bench.benchmarks.comparison_pairs.pelt_l2 import pair_pelt_l2
from change_bench.benchmarks.comparison_pairs.pelt_linear_trend import (
    pair_pelt_linear_trend,
)
from change_bench.benchmarks.comparison_pairs.pelt_linreg import pair_pelt_linreg
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
    "PairConfig",
    "build_pair_cases",
    "pair_binseg_continuous_linear_trend",
    "pair_binseg_esac",
    "pair_binseg_l1",
    "pair_binseg_linreg",
    "pair_binseg_l2_cusum",
    "pair_binseg_mv_gaussian",
    "pair_binseg_poisson",
    "pair_binseg_rank",
    "pair_fpop_l2",
    "pair_moving_window_esac",
    "pair_moving_window_continuous_linear_trend",
    "pair_moving_window_l1",
    "pair_moving_window_linreg",
    "pair_moving_window_l2",
    "pair_moving_window_mv_gaussian",
    "pair_moving_window_poisson",
    "pair_moving_window_rank",
    "pair_pelt_1d_gaussian",
    "pair_pelt_l1",
    "pair_pelt_l2",
    "pair_pelt_linreg",
    "pair_pelt_linear_trend",
    "pair_pelt_mv_gaussian",
    "pair_pelt_poisson",
    "pair_pelt_rank",
]
