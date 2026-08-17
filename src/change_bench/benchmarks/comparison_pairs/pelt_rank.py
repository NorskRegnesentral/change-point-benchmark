"""PELT + Rank cost comparison pair (multivariate).

skchange: PELT(cost=RankCost(), penalty=P, min_segment_length=2)
ruptures: Pelt(model="rank", min_size=2, jump=1)
"""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import PELT as SkchangePELT
from skchange.interval_scorers import RankCost

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_PELT_RANK_PENALTY = 20.0

_CONFIG = PairConfig(
    pair_name="pelt_rank",
    penalty=JOINT_PELT_RANK_PENALTY,
    sk_name_prefix="skchange_pelt_rank",
    rpt_name_prefix="ruptures_pelt_rank",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=RankCost(), penalty=JOINT_PELT_RANK_PENALTY, min_segment_length=msl
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(model="rank", min_size=msl, jump=1),
    effective_msl=lambda msl, n_cols: max(msl, 2),
)


def pair_pelt_rank(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 2,
) -> list[BenchmarkCase]:
    """PELT with Rank cost — skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
