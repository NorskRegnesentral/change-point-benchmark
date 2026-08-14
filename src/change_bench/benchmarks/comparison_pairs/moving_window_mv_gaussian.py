"""MovingWindow + MultivariateGaussianScore comparison pair (multivariate).

skchange: MovingWindow(change_score=MultivariateGaussianScore(
              apply_bartlett_correction=False), penalty=P,
              bandwidth=BW)
ruptures: Window(model="normal", width=2*BW, min_size=max(msl, n_columns+1), jump=1)

MultivariateGaussianScore requires min_size >= data_dimension + 1, so
the effective minimum segment length is automatically adjusted per problem.
"""

from __future__ import annotations

import ruptures as rpt
from skchange.detectors import MovingWindow
from skchange.interval_scorers import MultivariateGaussianScore

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_MW_MV_GAUSSIAN_PENALTY = 4.0


def _make_sk_detector(msl: int):
    return MovingWindow(
        change_score=MultivariateGaussianScore(
            apply_bartlett_correction=False, store_cov=True
        ),
        penalty=JOINT_MW_MV_GAUSSIAN_PENALTY,
        bandwidth=max(MW_BANDWIDTH, msl),
    )


_CONFIG = PairConfig(
    pair_name="moving_window_mv_gaussian",
    penalty=JOINT_MW_MV_GAUSSIAN_PENALTY,
    sk_name_prefix="skchange_moving_window_mv_gaussian",
    rpt_name_prefix="ruptures_window_normal",
    make_sk_detector=_make_sk_detector,
    make_rpt_algo=lambda msl: rpt.Window(
        model="normal", width=2 * max(MW_BANDWIDTH, msl), min_size=msl, jump=1
    ),
    effective_msl=lambda msl, n_cols: max(msl, n_cols + 1),
)


def pair_moving_window_mv_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with MultivariateGaussianScore/normal cost."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
