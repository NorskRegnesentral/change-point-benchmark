"""Binary segmentation + Poisson cost comparison pair.

skchange: SeededBinarySegmentation(change_score=CostChangeScore(PoissonCost()))
ruptures: Binseg(custom_cost=CostPoisson(), min_size=1, jump=1)

Ruptures has no built-in Poisson model, so the custom ``CostPoisson`` from
the PELT pair is reused.
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from skchange.detectors import SeededBinarySegmentation
from skchange.interval_scorers import CostChangeScore, PoissonCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.benchmarks.comparison_pairs.pelt_poisson import CostPoisson
from change_bench.problems.base import BenchmarkProblem


def _make_sk_detector(msl: int):
    return SeededBinarySegmentation(
        change_score=CostChangeScore(PoissonCost()), penalty=PELT_PENALTY
    )


_CONFIG = PairConfig(
    pair_name="binseg_poisson",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_poisson",
    rpt_name_prefix="ruptures_binseg_poisson",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Binseg(
        custom_cost=CostPoisson(), min_size=msl, jump=1
    ),
    prepare_transform=lambda x: np.abs(x) + 0.01,
)


def pair_binseg_poisson(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Binary segmentation with Poisson cost — skchange vs ruptures (custom cost)."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
