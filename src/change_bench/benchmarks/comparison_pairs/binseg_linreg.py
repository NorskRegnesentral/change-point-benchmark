"""Seeded binary segmentation + linear regression cost comparison pair."""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.interval_scorers import CostChangeScore, LinearRegressionCost

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_BINSEG_LINREG_PENALTY = 10.0


def _make_sk_detector(msl: int):
    return SeededBinarySegmentation(
        change_score=CostChangeScore(LinearRegressionCost(response_col=0)),
        penalty=JOINT_BINSEG_LINREG_PENALTY,
        min_subinterval_length=msl,
        max_interval_length=max(200, 2 * msl),
    )


_CONFIG = PairConfig(
    pair_name="binseg_linreg",
    penalty=JOINT_BINSEG_LINREG_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_linreg",
    rpt_name_prefix="ruptures_binseg_linear",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Binseg(
        custom_cost=rpt.costs.CostLinear(), min_size=msl, jump=1
    ),
    effective_msl=lambda msl, n_cols: max(msl, 2),
)


def pair_binseg_linreg(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 2,
) -> list[BenchmarkCase]:
    """Seeded binary segmentation with linear regression cost for column 0."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
