"""PELT + Poisson cost comparison pair.

skchange: PELT(cost=PoissonCost())
ruptures: Pelt(custom_cost=CostPoisson(), min_size=1, jump=1)

Ruptures has no built-in Poisson model, so we provide a custom ``BaseCost``
subclass (``CostPoisson``) that computes the same twice-negative-log-likelihood
as skchange's ``PoissonCost``.
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import PELT as SkchangePELT
from skchange.new_api.interval_scorers import PoissonCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    make_prepare,
    skchange_predict_only,
    skchange_fit_predict,
)
from change_bench.problems.base import BenchmarkProblem


# ---------------------------------------------------------------------------
# Custom Poisson cost for ruptures (no built-in Poisson model).
# Computes twice negative log-likelihood matching skchange's PoissonCost.
# ---------------------------------------------------------------------------
class CostPoisson(rpt.base.BaseCost):
    """Poisson cost for ruptures: 2 * negative log-likelihood per segment."""

    model = "poisson_custom"
    min_size = 1

    def fit(self, signal):
        """Store signal and precompute cumulative sums."""
        self.signal = signal
        self._cumsum = np.concatenate(
            [np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)]
        )
        from scipy.special import gammaln

        self._cumsum_logfact = np.concatenate(
            [np.zeros((1, signal.shape[1])), np.cumsum(gammaln(signal + 1), axis=0)]
        )
        return self

    def error(self, start, end):
        """Return the Poisson cost on segment [start:end]."""
        n = end - start
        seg_sum = self._cumsum[end] - self._cumsum[start]
        seg_logfact = self._cumsum_logfact[end] - self._cumsum_logfact[start]
        lam = seg_sum / n  # MLE rate per column
        # 2 * NLL = 2 * (n*lam - sum(x)*log(lam) + sum(log(x!)))
        # Avoid log(0) for zero-rate segments
        log_lam = np.log(lam, where=lam > 0, out=np.zeros_like(lam))
        cost = 2 * (n * lam - seg_sum * log_lam + seg_logfact)
        return cost.sum()


def pair_pelt_poisson(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with Poisson cost — skchange vs ruptures (custom cost)."""
    pair_name = "pelt_poisson"
    cases: list[BenchmarkCase] = []
    sk_func = skchange_fit_predict if include_fit else skchange_predict_only

    for problem in problems:
        cfg = problem.dataset_config
        _base_prepare = make_prepare(problem)

        def make_poisson_prepare(base=_base_prepare):
            """Wrap base prepare to ensure non-negative data for Poisson cost."""

            def prepare() -> np.ndarray:
                return np.abs(base()) + 0.01

            return prepare

        prepare = make_poisson_prepare()

        def make_sk_setup(fit=include_fit, msl=min_segment_length):
            def setup(data: np.ndarray):
                det = SkchangePELT(
                    cost=PoissonCost(),
                    penalty=PELT_PENALTY,
                    min_segment_length=msl,
                )
                if not fit:
                    det.fit(data)
                return (det, data), {}

            return setup

        def make_rpt_setup(fit=include_fit, msl=min_segment_length):
            def setup(data: np.ndarray):
                algo = rpt.Pelt(custom_cost=CostPoisson(), min_size=msl, jump=1)
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
                name=f"skchange_pelt_poisson/{problem.name}",
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
                name=f"ruptures_pelt_poisson/{problem.name}",
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
