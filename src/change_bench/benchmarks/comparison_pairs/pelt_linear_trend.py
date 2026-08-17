"""PELT + Linear Trend cost comparison pair.

skchange: PELT(cost=LinearTrendCost())
ruptures: Pelt(custom_cost=CostCLinear(), min_size=1, jump=1)

NOTE: Not really a fair comparison — skchange solves for the *best* linear
trend in the segment, whilst ruptures draws a line from the start value to
the end value.
"""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import PELT as SkchangePELT
from skchange.interval_scorers import LinearTrendCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="pelt_linear_trend",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_linear_trend",
    rpt_name_prefix="ruptures_pelt_clinear",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=LinearTrendCost(), penalty=PELT_PENALTY, min_segment_length=msl
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(
        custom_cost=rpt.costs.CostCLinear(), min_size=msl, jump=1
    ),
)


def pair_pelt_linear_trend(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with linear-trend cost — skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
