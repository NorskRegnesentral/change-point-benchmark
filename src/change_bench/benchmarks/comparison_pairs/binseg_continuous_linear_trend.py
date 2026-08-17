"""SeededBinarySegmentation continuous linear trend comparison pair."""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import SeededBinarySegmentation
from skchange.interval_scorers import ContinuousLinearTrendScore

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_BINSEG_CONTINUOUS_LINEAR_TREND_PENALTY = 100.0

_CONFIG = PairConfig(
    pair_name="binseg_continuous_linear_trend",
    penalty=JOINT_BINSEG_CONTINUOUS_LINEAR_TREND_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_continuous_linear_trend",
    rpt_name_prefix="ruptures_binseg_clinear",
    make_sk_detector=lambda msl: SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(),
        penalty=JOINT_BINSEG_CONTINUOUS_LINEAR_TREND_PENALTY,
    ),
    make_rpt_algo=lambda msl: rpt.Binseg(
        custom_cost=rpt.costs.CostCLinear(), min_size=msl, jump=1
    ),
    effective_msl=lambda msl, n_cols: max(msl, 3),
)


def pair_binseg_continuous_linear_trend(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 3,
) -> list[BenchmarkCase]:
    """Build SBS trend-score cases for skchange and ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )