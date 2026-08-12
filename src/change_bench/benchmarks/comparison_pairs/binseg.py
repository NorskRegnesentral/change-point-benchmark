"""Binary Segmentation + CUSUM / L2 comparison pair.

skchange: SeededBinarySegmentation(
    change_score=CostChangeScore(L2Cost()), penalty=JOINT_PENALTY
)
ruptures: Binseg(model="l2", min_size=1, jump=1)
"""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.interval_scorers import CUSUM

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_PENALTY = 10.0


def _make_sk_detector(msl: int):
    return SeededBinarySegmentation(change_score=CUSUM(), penalty=JOINT_PENALTY)


_CONFIG = PairConfig(
    pair_name="binseg_l2_cusum",
    penalty=JOINT_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_cusum",
    rpt_name_prefix="ruptures_binseg_l2",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Binseg(model="l2", min_size=msl, jump=1),
)


def pair_binseg_l2_cusum(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Binary segmentation with CUSUM/L2 — skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
