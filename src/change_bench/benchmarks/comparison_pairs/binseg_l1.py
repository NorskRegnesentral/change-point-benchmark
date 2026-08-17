"""Binary segmentation + L1 cost comparison pair."""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import SeededBinarySegmentation
from skchange.interval_scorers import CostChangeScore, L1Cost

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_PENALTY = 100.0


def _make_sk_detector(msl: int):
    return SeededBinarySegmentation(
        change_score=CostChangeScore(L1Cost()), penalty=JOINT_PENALTY
    )


_CONFIG = PairConfig(
    pair_name="binseg_l1",
    penalty=JOINT_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_l1",
    rpt_name_prefix="ruptures_binseg_l1",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Binseg(model="l1", min_size=msl, jump=1),
)


def pair_binseg_l1(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Binary segmentation with L1 cost, comparing skchange with ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )