"""Moving window + linear regression cost comparison pair."""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import MovingWindow
from skchange.interval_scorers import CostChangeScore, LinearRegressionCost

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_MW_LINREG_PENALTY = 10.0


def _make_sk_detector(msl: int):
    return MovingWindow(
        change_score=CostChangeScore(LinearRegressionCost(response_col=0)),
        penalty=JOINT_MW_LINREG_PENALTY,
        bandwidth=max(MW_BANDWIDTH, msl),
    )


_CONFIG = PairConfig(
    pair_name="moving_window_linreg",
    penalty=JOINT_MW_LINREG_PENALTY,
    sk_name_prefix="skchange_moving_window_linreg",
    rpt_name_prefix="ruptures_window_linear",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Window(
        custom_cost=rpt.costs.CostLinear(),
        width=2 * max(MW_BANDWIDTH, msl),
        min_size=msl,
        jump=1,
    ),
    effective_msl=lambda msl, n_cols: max(msl, 2),
)


def pair_moving_window_linreg(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 2,
) -> list[BenchmarkCase]:
    """Moving window with linear regression cost for response column 0."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )