"""MovingWindow + L2 cost comparison pair (explicit bandwidth).

skchange: MovingWindow(change_score=PenalisedScore(CUSUM(), penalty=P), bandwidth=BW)
ruptures: Window(model="l2", width=2*BW, min_size=1, jump=1)
"""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import MovingWindow
from skchange.new_api.interval_scorers import CUSUM, PenalisedScore

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_MW_L2_PENALTY = 4.0


def _make_sk_detector(msl: int):
    fixed_penalty_score = PenalisedScore(CUSUM(), penalty=JOINT_MW_L2_PENALTY)
    return MovingWindow(change_score=fixed_penalty_score, bandwidth=MW_BANDWIDTH)


_CONFIG = PairConfig(
    pair_name="moving_window_l2",
    penalty=JOINT_MW_L2_PENALTY,
    sk_name_prefix="skchange_moving_window_l2",
    rpt_name_prefix="ruptures_window_l2",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Window(
        model="l2", width=2 * MW_BANDWIDTH, min_size=msl, jump=1
    ),
)


def pair_moving_window_l2(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with L2 cost (fixed bandwidth) — skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
