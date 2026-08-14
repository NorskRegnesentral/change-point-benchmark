"""PELT + Gaussian cost comparison pair.

skchange: PELT(cost=GaussianCost())
ruptures: Pelt(model="normal", min_size=1, jump=1)
"""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import PELT as SkchangePELT
from skchange.interval_scorers import GaussianCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="pelt_1d_gaussian",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_1d_gaussian",
    rpt_name_prefix="ruptures_pelt_1d_normal",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=GaussianCost(), penalty=PELT_PENALTY, min_segment_length=msl
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(model="normal", min_size=msl, jump=1),
)


def pair_pelt_1d_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with Gaussian/normal cost — skchange vs ruptures."""
    return build_pair_cases(
        problems, _CONFIG, include_fit=include_fit, min_segment_length=min_segment_length
    )
