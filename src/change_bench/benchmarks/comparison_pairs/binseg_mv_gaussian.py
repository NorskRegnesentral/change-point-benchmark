"""SeededBinarySegmentation + MultivariateGaussianCost comparison pair (multivariate).

skchange: SeededBinarySegmentation(change_score=PenalisedScore(
              CostChangeScore(MultivariateGaussianCost()), penalty=P))
ruptures: Binseg(model="normal", min_size=max(msl, n_columns+1), jump=1)

MultivariateGaussianCost requires min_size >= data_dimension + 1, so
the effective minimum segment length is automatically adjusted per problem.
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.interval_scorers import (
    CostChangeScore,
    MultivariateGaussianCost,
    PenalisedScore,
)

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    make_prepare,
    skchange_predict_only,
    skchange_run,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_BINSEG_MV_GAUSSIAN_PENALTY = 10.0


def pair_binseg_mv_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Seeded binary segmentation with MultivariateGaussianCost/normal — skchange vs ruptures."""
    pair_name = "binseg_mv_gaussian"
    cases: list[BenchmarkCase] = []
    sk_func = skchange_run if include_fit else skchange_predict_only

    for problem in problems:
        cfg = problem.dataset_config
        prepare = make_prepare(problem)
        effective_msl = max(min_segment_length, cfg.n_columns + 1)

        def make_sk_setup(fit=include_fit):
            def setup(data: np.ndarray):
                fixed_penalty_score = PenalisedScore(
                    CostChangeScore(MultivariateGaussianCost()),
                    penalty=JOINT_BINSEG_MV_GAUSSIAN_PENALTY,
                )
                det = SeededBinarySegmentation(change_score=fixed_penalty_score)
                if not fit:
                    det.fit(data)
                return (det, data), {}

            return setup

        def make_rpt_setup(fit=include_fit, msl=effective_msl):
            def setup(data: np.ndarray):
                algo = rpt.Binseg(model="normal", min_size=msl, jump=1)
                if not fit:
                    algo.fit(data)
                return (algo, data), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=JOINT_BINSEG_MV_GAUSSIAN_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_seeded_binseg_mv_gaussian/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=effective_msl,
                prepare=prepare,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_binseg_normal/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=effective_msl,
                prepare=prepare,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases
