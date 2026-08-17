"""SeededBinarySegmentation + rank score comparison pair."""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import SeededBinarySegmentation
from skchange.interval_scorers import CostChangeScore, RankCost

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_BINSEG_RANK_PENALTY = 100.0

_CONFIG = PairConfig(
    pair_name="binseg_rank",
    penalty=JOINT_BINSEG_RANK_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_rank",
    rpt_name_prefix="ruptures_binseg_rank",
    make_sk_detector=lambda msl: SeededBinarySegmentation(
        change_score=CostChangeScore(RankCost()),
        penalty=JOINT_BINSEG_RANK_PENALTY,
    ),
    make_rpt_algo=lambda msl: rpt.Binseg(model="rank", min_size=msl, jump=1),
    effective_msl=lambda msl, n_cols: max(msl, 2),
)


def pair_binseg_rank(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 2,
) -> list[BenchmarkCase]:
    """Seeded binary segmentation with rank score, skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
