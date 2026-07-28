"""PELT + Gaussian cost comparison pair.

skchange: PELT(cost=GaussianCost())
ruptures: Pelt(model="normal", min_size=1, jump=1)
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import PELT as SkchangePELT
from skchange.new_api.interval_scorers import GaussianCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    make_prepare,
    skchange_fit_predict,
    skchange_predict_only,
)
from change_bench.problems.base import BenchmarkProblem


def pair_pelt_1d_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with Gaussian/normal cost — skchange vs ruptures."""
    pair_name = "pelt_1d_gaussian"
    cases: list[BenchmarkCase] = []
    sk_func = skchange_fit_predict if include_fit else skchange_predict_only

    for problem in problems:
        cfg = problem.dataset_config
        prepare = make_prepare(problem)

        def make_sk_setup(fit=include_fit, msl=min_segment_length):
            def setup(data: np.ndarray):
                det = SkchangePELT(
                    cost=GaussianCost(),
                    penalty=PELT_PENALTY,
                    min_segment_length=msl,
                )
                if not fit:
                    det.fit(data)
                return (det, data), {}

            return setup

        def make_rpt_setup(fit=include_fit, msl=min_segment_length):
            def setup(data: np.ndarray):
                algo = rpt.Pelt(model="normal", min_size=msl, jump=1)
                if not fit:
                    algo.fit(data)
                return (algo, data), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=PELT_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_pelt_1d_gaussian/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                prepare=prepare,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_pelt_1d_normal/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                prepare=prepare,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases
