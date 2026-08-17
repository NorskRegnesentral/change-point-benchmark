"""MovingWindow + Rank score comparison pair (multivariate only).

skchange: MovingWindow(change_score=RankScore(),
                    penalty=P, bandwidth=BW)
ruptures: Window(model="rank", width=2*BW, min_size=1, jump=1)
"""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import MovingWindow
from skchange.interval_scorers import RankScore, RankCost, CostChangeScore

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_MW_RANK_PENALTY = 20.0


def _make_sk_detector(msl: int):
    return MovingWindow(
        # change_score=RankScore(),
        change_score=CostChangeScore(RankCost(), deduplicate=False),
        penalty=JOINT_MW_RANK_PENALTY,
        bandwidth=MW_BANDWIDTH,
    )


_CONFIG = PairConfig(
    pair_name="moving_window_rank",
    penalty=JOINT_MW_RANK_PENALTY,
    sk_name_prefix="skchange_moving_window_rank",
    rpt_name_prefix="ruptures_window_rank",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Window(
        model="rank", width=2 * MW_BANDWIDTH, min_size=msl, jump=1
    ),
)


def pair_moving_window_rank(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with Rank score (fixed bandwidth) — skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
