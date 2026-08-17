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
from skchange.detectors import PELT as SkchangePELT
from skchange.interval_scorers import PoissonCost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
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


_CONFIG = PairConfig(
    pair_name="pelt_poisson",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_poisson",
    rpt_name_prefix="ruptures_pelt_poisson",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=PoissonCost(), penalty=PELT_PENALTY, min_segment_length=msl
    ),
    make_rpt_algo=lambda msl: rpt.Pelt(custom_cost=CostPoisson(), min_size=msl, jump=1),
    prepare_transform=lambda x: np.abs(x) + 0.01,
)


def pair_pelt_poisson(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with Poisson cost — skchange vs ruptures (custom cost)."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
