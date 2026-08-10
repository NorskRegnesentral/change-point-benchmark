"""PELT + MultivariateGaussianCost comparison pair (multivariate).

skchange: PELT(cost=MultivariateGaussianCost())
ruptures: Pelt(model="normal", min_size=max(msl, n_columns+1), jump=1)

MultivariateGaussianCost requires min_size >= data_dimension + 1, so
the effective minimum segment length is automatically adjusted per problem.
"""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import PELT as SkchangePELT
from skchange.new_api.interval_scorers import MultivariateGaussianCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="pelt_mv_gaussian",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_mv_gaussian",
    rpt_name_prefix="ruptures_pelt_normal",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=MultivariateGaussianCost(store_cov=True),
        penalty=PELT_PENALTY,
        min_segment_length=msl,
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(model="normal", min_size=msl, jump=1),
    effective_msl=lambda msl, n_cols: max(msl, n_cols + 1),
)


def pair_pelt_mv_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 2,
) -> list[BenchmarkCase]:
    """PELT with MultivariateGaussianCost/normal — skchange vs ruptures."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
