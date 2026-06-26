"""Binary Segmentation + CUSUM / L2 comparison pair.

skchange: SeededBinarySegmentation(change_score=CUSUM())
ruptures: Binseg(model="l2", min_size=1, jump=1)
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import SeededBinarySegmentation
from skchange.new_api.interval_scorers import (
    CostChangeScore,
    L2Cost,
    PenalisedScore,
)

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    make_prepare,
    skchange_predict_only,
    skchange_run,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_PENALTY = 10.0


def pair_binseg_l2_cusum(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Binary segmentation with CUSUM/L2 — skchange vs ruptures."""
    pair_name = "binseg_l2_cusum"
    cases: list[BenchmarkCase] = []
    sk_func = skchange_run if include_fit else skchange_predict_only

    for problem in problems:
        cfg = problem.dataset_config
        prepare = make_prepare(problem)

        def make_sk_setup(fit=include_fit):
            def setup(data: np.ndarray):
                fixed_penalty_score = PenalisedScore(
                    CostChangeScore(L2Cost()), penalty=JOINT_PENALTY
                )
                det = SeededBinarySegmentation(change_score=fixed_penalty_score)
                if not fit:
                    det.fit(data)
                return (det, data), {}

            return setup

        def make_rpt_setup(fit=include_fit, msl=min_segment_length):
            def setup(data: np.ndarray):
                algo = rpt.Binseg(model="l2", min_size=msl, jump=1)
                if not fit:
                    algo.fit(data)
                return (algo, data), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=JOINT_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_seeded_binseg_cusum/{problem.name}",
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
                name=f"ruptures_binseg_l2/{problem.name}",
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
