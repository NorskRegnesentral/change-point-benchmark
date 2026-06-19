"""Tests that _CostPoisson (ruptures) agrees with PoissonCost (skchange)."""

from __future__ import annotations

import numpy as np
import pytest
from skchange.new_api.interval_scorers import PoissonCost

from change_bench.benchmarks.null_case import _CostPoisson


class TestCostPoissonAgreement:
    """Verify _CostPoisson matches skchange PoissonCost on various segments."""

    def _skchange_segment_cost(self, X: np.ndarray, start: int, end: int) -> float:
        """Compute total segment cost using skchange PoissonCost."""
        cost = PoissonCost()
        cost.fit(X)
        cache = cost.precompute(X)
        intervals = np.array([[start, end]])
        # evaluate returns shape (1, n_columns); sum over columns for total
        return cost.evaluate(cache, intervals).sum()

    def _ruptures_segment_cost(self, X: np.ndarray, start: int, end: int) -> float:
        """Compute total segment cost using _CostPoisson."""
        cost = _CostPoisson()
        cost.fit(X)
        return cost.error(start, end)

    @pytest.mark.parametrize("n_cols", [1, 3])
    def test_nonzero_rate_poisson_data(self, n_cols: int) -> None:
        """Both costs agree on Poisson-distributed data (non-zero rate)."""
        rng = np.random.default_rng(42)
        X = rng.poisson(lam=5.0, size=(100, n_cols)).astype(float)

        for start, end in [(0, 50), (20, 80), (0, 100), (50, 100)]:
            sk = self._skchange_segment_cost(X, start, end)
            rpt = self._ruptures_segment_cost(X, start, end)
            assert sk == pytest.approx(rpt, rel=1e-10), (
                f"Mismatch on [{start}:{end}] n_cols={n_cols}: "
                f"skchange={sk}, ruptures={rpt}"
            )

    @pytest.mark.parametrize("n_cols", [1, 3])
    def test_nonzero_rate_continuous_data(self, n_cols: int) -> None:
        """Both costs agree on continuous non-negative data (gamma-distributed)."""
        rng = np.random.default_rng(7)
        X = rng.gamma(2.0, 3.0, size=(80, n_cols))

        for start, end in [(0, 40), (10, 70), (0, 80)]:
            sk = self._skchange_segment_cost(X, start, end)
            rpt = self._ruptures_segment_cost(X, start, end)
            assert sk == pytest.approx(rpt, rel=1e-10), (
                f"Mismatch on [{start}:{end}] n_cols={n_cols}: "
                f"skchange={sk}, ruptures={rpt}"
            )

    @pytest.mark.parametrize("n_cols", [1, 2])
    def test_zero_rate_segment(self, n_cols: int) -> None:
        """Both costs return 0 for a segment where all observations are zero."""
        X = np.zeros((20, n_cols))

        sk = self._skchange_segment_cost(X, 0, 20)
        rpt = self._ruptures_segment_cost(X, 0, 20)
        assert sk == pytest.approx(0.0, abs=1e-15)
        assert rpt == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("n_cols", [1, 2])
    def test_mixed_zero_and_nonzero_segments(self, n_cols: int) -> None:
        """Costs agree when data has a zero-rate segment adjacent to non-zero."""
        rng = np.random.default_rng(99)
        # First 30 rows: zeros; last 30 rows: Poisson(3)
        zeros = np.zeros((30, n_cols))
        nonzero = rng.poisson(lam=3.0, size=(30, n_cols)).astype(float)
        X = np.vstack([zeros, nonzero])

        # Segment entirely within zeros
        sk = self._skchange_segment_cost(X, 0, 30)
        rpt = self._ruptures_segment_cost(X, 0, 30)
        assert sk == pytest.approx(0.0, abs=1e-15)
        assert rpt == pytest.approx(0.0, abs=1e-15)

        # Segment entirely within non-zero part
        sk = self._skchange_segment_cost(X, 30, 60)
        rpt = self._ruptures_segment_cost(X, 30, 60)
        assert sk == pytest.approx(rpt, rel=1e-10)

        # Segment spanning both (rate > 0 overall due to non-zero part)
        sk = self._skchange_segment_cost(X, 0, 60)
        rpt = self._ruptures_segment_cost(X, 0, 60)
        assert sk == pytest.approx(rpt, rel=1e-10)

    def test_single_observation_segment(self) -> None:
        """Both costs agree on a segment of length 1 (min_size=1)."""
        X = np.array([[7.0], [0.0], [3.0]])

        for i in range(3):
            sk = self._skchange_segment_cost(X, i, i + 1)
            rpt = self._ruptures_segment_cost(X, i, i + 1)
            assert sk == pytest.approx(rpt, rel=1e-10), (
                f"Mismatch on single obs [{i}:{i+1}]: "
                f"skchange={sk}, ruptures={rpt}"
            )
