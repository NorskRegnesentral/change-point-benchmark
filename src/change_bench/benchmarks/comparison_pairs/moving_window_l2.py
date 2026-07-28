"""MovingWindow + L2 cost comparison pair (explicit bandwidth).

skchange: MovingWindow(change_score=CostChangeScore(L2Cost()), bandwidth=BW)
ruptures: Window(model="l2", width=2*BW, min_size=1, jump=1)
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import MovingWindow
from skchange.new_api.interval_scorers import CUSUM, PenalisedScore

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    BenchmarkCase,
    make_prepare,
    skchange_fit_predict,
    skchange_predict_only,
)
from change_bench.problems.base import BenchmarkProblem

JOINT_MW_L2_PENALTY = 4.0


def pair_moving_window_l2(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with L2 cost (fixed bandwidth) — skchange vs ruptures."""
    pair_name = "moving_window_l2"
    bw = MW_BANDWIDTH
    cases: list[BenchmarkCase] = []
    sk_func = skchange_fit_predict if include_fit else skchange_predict_only

    for problem in problems:
        cfg = problem.dataset_config
        prepare = make_prepare(problem)

        def make_sk_setup(bandwidth=bw, fit=include_fit):
            def setup(data: np.ndarray):
                fixed_penalty_score = PenalisedScore(
                    CUSUM(), penalty=JOINT_MW_L2_PENALTY
                )
                det = MovingWindow(
                    change_score=fixed_penalty_score, bandwidth=bandwidth
                )
                if not fit:
                    det.fit(data)
                return (det, data), {}

            return setup

        def make_rpt_setup(width=2 * bw, fit=include_fit, msl=min_segment_length):
            def setup(data: np.ndarray):
                algo = rpt.Window(model="l2", width=width, min_size=msl, jump=1)
                if not fit:
                    algo.fit(data)
                return (algo, data), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=JOINT_MW_L2_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_moving_window_l2/{problem.name}",
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
                name=f"ruptures_window_l2/{problem.name}",
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
