"""PELT + L1 cost comparison pair."""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import PELT as SkchangePELT
from skchange.new_api.interval_scorers import L1Cost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="pelt_l1",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_l1",
    rpt_name_prefix="ruptures_pelt_l1",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=L1Cost(), penalty=PELT_PENALTY, min_segment_length=msl
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(model="l1", min_size=msl, jump=1),
)


def pair_pelt_l1(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with L1 cost, comparing skchange with ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )