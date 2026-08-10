"""PELT + linear regression cost comparison pair.

Column 0 is the response and all remaining columns are predictors, matching
ruptures' ``CostLinear`` convention explicitly in skchange.
"""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import PELT as SkchangePELT
from skchange.new_api.interval_scorers import LinearRegressionCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="pelt_linreg",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_linreg",
    rpt_name_prefix="ruptures_pelt_linear",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=LinearRegressionCost(response_col=0),
        penalty=PELT_PENALTY,
        min_segment_length=msl,
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(
        custom_cost=rpt.costs.CostLinear(), min_size=msl, jump=1
    ),
    effective_msl=lambda msl, n_cols: max(msl, 2),
)


def pair_pelt_linreg(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 2,
) -> list[BenchmarkCase]:
    """PELT with linear regression cost, comparing skchange with ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )