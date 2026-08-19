"""SeededBinarySegmentation + MultivariateGaussianCost comparison pair (multivariate).

skchange: SeededBinarySegmentation(
              change_score=MultivariateGaussianScore(
                  apply_bartlett_correction=False), penalty=P)
ruptures: Binseg(model="normal", min_size=max(msl, n_columns+1), jump=1)

MultivariateGaussianCost requires min_size >= data_dimension + 1, so
the effective minimum segment length is automatically adjusted per problem.
"""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import SeededBinarySegmentation
from skchange.interval_scorers import MultivariateGaussianScore

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_BINSEG_MV_GAUSSIAN_PENALTY = 1.0e4


def _make_sk_detector(msl: int):
    return SeededBinarySegmentation(
        change_score=MultivariateGaussianScore(
            apply_bartlett_correction=False, store_cov=True
        ),
        penalty=JOINT_BINSEG_MV_GAUSSIAN_PENALTY,
        max_interval_length=max(200, 2 * msl),
    )


_CONFIG = PairConfig(
    pair_name="binseg_mv_gaussian",
    penalty=JOINT_BINSEG_MV_GAUSSIAN_PENALTY,
    sk_name_prefix="skchange_seeded_binseg_mv_gaussian",
    rpt_name_prefix="ruptures_binseg_normal",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Binseg(model="normal", min_size=msl, jump=1),
    effective_msl=lambda msl, n_cols: max(msl, n_cols + 1),
)


def pair_binseg_mv_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Seeded binary segmentation with MultivariateGaussianScore/normal cost."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
