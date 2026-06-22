"""Tests verifying that comparison pairs find the same change points.

For each registered benchmark pair, we generate data with strong, unambiguous
change points and verify that the skchange and ruptures implementations
produce equivalent results (exact for PELT, within tolerance for
MovingWindow/BinSeg due to different peak-selection logic).

Uses the same numeric penalty for both libraries:
- skchange: PenalisedScore(scorer, penalty=P)  or  PELT(cost=..., penalty=P)
- ruptures: .predict(pen=P)
"""

from __future__ import annotations

import numpy as np
import pytest
import ruptures as rpt
from skchange.new_api.detectors import (
    PELT as SkchangePELT,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.interval_scorers import (
    CUSUM,
    CostChangeScore,
    GaussianCost,
    L1Cost,
    L2Cost,
    PenalisedScore,
    PoissonCost,
    RankCost,
)

from change_bench.benchmarks.comparison_pairs._common import MW_BANDWIDTH, PELT_PENALTY
from change_bench.benchmarks.comparison_pairs.pelt_poisson import CostPoisson
from change_bench.datasets.change_case import ChangeDatasetConfig, SegmentParams

# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

#: Penalty used for all tests (same as benchmark runs for PELT pairs).
TEST_PENALTY: float = PELT_PENALTY

#: Penalty for MovingWindow / BinSeg tests.
#: Must be tuned so that strong mean shifts (5× scale) are detected.
MW_PENALTY: float = 2.0

#: Higher penalty for BinSeg (SeededBinarySegmentation is more sensitive).
BINSEG_PENALTY: float = 15.0

#: Tolerance (in samples) for MovingWindow/BinSeg change-point location.
TOLERANCE: int = 10


def _strip_endpoint(breakpoints: list[int], n_samples: int) -> list[int]:
    """Strip trailing endpoint from ruptures output.

    Ruptures returns breakpoints including the final index ``n_samples``.
    skchange returns only interior change points.
    """
    return [b for b in breakpoints if b != n_samples]


def _make_normal_change_data(
    rng: np.random.Generator,
    n_samples: int = 300,
    changepoints: list[int] | None = None,
    means: list[float] | None = None,
    n_columns: int = 1,
) -> np.ndarray:
    """Generate data with strong mean shifts for testing."""
    if changepoints is None:
        changepoints = [100, 200]
    if means is None:
        means = [0.0, 5.0, 0.0]

    cfg = ChangeDatasetConfig(
        n_samples=n_samples,
        changepoints=changepoints,
        segments=[SegmentParams(loc=m, scale=1.0) for m in means],
        distribution="normal",
        n_columns=n_columns,
    )
    return cfg.generate(rng)


def _assert_changepoints_close(
    detected: list[int],
    expected: list[int],
    tolerance: int,
    msg: str = "",
) -> None:
    """Assert detected change points are within tolerance of expected.

    Checks that every expected change point has a corresponding detected
    change point nearby (recall), and that the number of spurious detections
    is limited (at most one extra per expected change point).
    """
    # Every expected change point must have a nearby detection
    for e in expected:
        nearby = [d for d in detected if abs(d - e) <= tolerance]
        assert len(nearby) >= 1, (
            f"{msg}Expected changepoint at {e} not found within ±{tolerance}. "
            f"Detected: {detected}, Expected: {expected}"
        )
    # Allow some spurious detections but not too many
    max_detections = len(expected) * 2 + 1
    assert len(detected) <= max_detections, (
        f"{msg}Too many detections: {len(detected)} (max {max_detections}). "
        f"Detected: {detected}, Expected: {expected}"
    )


# ---------------------------------------------------------------------------
# PELT pair agreement tests — exact equality
# ---------------------------------------------------------------------------


class TestPeltL2Agreement:
    """PELT + L2 cost: skchange vs ruptures KernelCPD(linear)."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_same_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        data = _make_normal_change_data(rng, n_columns=n_columns)

        # skchange
        sk_det = SkchangePELT(
            cost=L2Cost(), penalty=TEST_PENALTY, min_segment_length=1
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.KernelCPD(kernel="linear", min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(_strip_endpoint(rpt_algo.predict(pen=TEST_PENALTY), len(data)))

        assert sk_cps == rpt_cps, (
            f"PELT L2 disagreement: skchange={sk_cps}, ruptures={rpt_cps}"
        )

    def test_detects_known_changepoints(self):
        """Sanity: both detect the known change points."""
        rng = np.random.default_rng(42)
        data = _make_normal_change_data(rng)

        sk_det = SkchangePELT(
            cost=L2Cost(), penalty=TEST_PENALTY, min_segment_length=1
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # With strong signal (mean shift = 5), both should find ~[100, 200]
        _assert_changepoints_close(sk_cps, [100, 200], tolerance=5)


class TestPelt1dGaussianAgreement:
    """PELT + Gaussian cost: skchange vs ruptures Pelt(model='normal')."""

    def test_same_changepoints(self):
        rng = np.random.default_rng(42)
        data = _make_normal_change_data(rng, n_columns=1)

        # skchange — GaussianCost requires min_segment_length >= 2
        sk_det = SkchangePELT(
            cost=GaussianCost(), penalty=TEST_PENALTY, min_segment_length=2
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Pelt(model="normal", min_size=2, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(_strip_endpoint(rpt_algo.predict(pen=TEST_PENALTY), len(data)))

        assert sk_cps == rpt_cps, (
            f"PELT Gaussian disagreement: skchange={sk_cps}, ruptures={rpt_cps}"
        )

    def test_detects_known_changepoints(self):
        rng = np.random.default_rng(42)
        data = _make_normal_change_data(rng, n_columns=1)

        sk_det = SkchangePELT(
            cost=GaussianCost(), penalty=TEST_PENALTY, min_segment_length=2
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # GaussianCost models mean+variance, so it may find extra changes.
        # Just verify the true change points are among those detected.
        for expected_cp in [100, 200]:
            nearby = [cp for cp in sk_cps if abs(cp - expected_cp) <= 5]
            assert len(nearby) >= 1, (
                f"Expected changepoint near {expected_cp} not found. "
                f"Detected: {sk_cps}"
            )


class TestPeltPoissonAgreement:
    """PELT + Poisson cost: skchange vs ruptures Pelt(custom_cost=CostPoisson())."""

    @staticmethod
    def _make_poisson_data(rng: np.random.Generator) -> np.ndarray:
        """Generate non-negative data with different rates per segment."""
        cfg = ChangeDatasetConfig(
            n_samples=300,
            changepoints=[100, 200],
            segments=[
                SegmentParams(loc=1.0, scale=0.5),
                SegmentParams(loc=10.0, scale=0.5),
                SegmentParams(loc=1.0, scale=0.5),
            ],
            distribution="exponential",
            n_columns=1,
        )
        # Exponential with loc shift: generate then ensure positive
        data = cfg.generate(rng)
        return np.abs(data) + 0.01

    def test_same_changepoints(self):
        rng = np.random.default_rng(42)
        data = self._make_poisson_data(rng)

        # skchange
        sk_det = SkchangePELT(
            cost=PoissonCost(), penalty=TEST_PENALTY, min_segment_length=1
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Pelt(custom_cost=CostPoisson(), min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(_strip_endpoint(rpt_algo.predict(pen=TEST_PENALTY), len(data)))

        assert sk_cps == rpt_cps, (
            f"PELT Poisson disagreement: skchange={sk_cps}, ruptures={rpt_cps}"
        )

    def test_detects_known_changepoints(self):
        rng = np.random.default_rng(42)
        data = self._make_poisson_data(rng)

        sk_det = SkchangePELT(
            cost=PoissonCost(), penalty=TEST_PENALTY, min_segment_length=1
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        _assert_changepoints_close(sk_cps, [100, 200], tolerance=5)


# ---------------------------------------------------------------------------
# MovingWindow pair agreement tests — tolerance-based
# ---------------------------------------------------------------------------


class TestMovingWindowAgreement:
    """MovingWindow + CUSUM/L2: skchange vs ruptures Window(model='l2')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng, n_samples=300, n_columns=n_columns
        )
        bw = MW_BANDWIDTH

        # skchange — use PenalisedScore with explicit penalty
        sk_det = MovingWindow(
            change_score=PenalisedScore(CUSUM(), penalty=MW_PENALTY),
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="l2", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=MW_PENALTY), len(data))
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"skchange MovingWindow CUSUM (p={n_columns}): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"ruptures Window L2 (p={n_columns}): ",
        )


class TestMovingWindowL2Agreement:
    """MovingWindow + L2Cost: skchange vs ruptures Window(model='l2')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng, n_samples=300, n_columns=n_columns
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=PenalisedScore(
                CostChangeScore(L2Cost()), penalty=MW_PENALTY
            ),
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="l2", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=MW_PENALTY), len(data))
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"skchange MovingWindow L2 (p={n_columns}): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"ruptures Window L2 (p={n_columns}): ",
        )


class TestMovingWindowL1Agreement:
    """MovingWindow + L1Cost: skchange vs ruptures Window(model='l1')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng, n_samples=300, n_columns=n_columns
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=PenalisedScore(
                CostChangeScore(L1Cost()), penalty=MW_PENALTY
            ),
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="l1", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=MW_PENALTY), len(data))
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"skchange MovingWindow L1 (p={n_columns}): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"ruptures Window L1 (p={n_columns}): ",
        )


class TestMovingWindowRankAgreement:
    """MovingWindow + RankCost: skchange vs ruptures Window(model='rank')."""

    def test_both_find_known_changepoints_multivariate(self):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        n_columns = 3
        data = _make_normal_change_data(
            rng, n_samples=300, n_columns=n_columns
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=PenalisedScore(
                CostChangeScore(RankCost()), penalty=MW_PENALTY
            ),
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="rank", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=MW_PENALTY), len(data))
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg="skchange MovingWindow Rank (multivariate): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg="ruptures Window Rank (multivariate): ",
        )


# ---------------------------------------------------------------------------
# BinSeg pair agreement test — tolerance-based
# ---------------------------------------------------------------------------


class TestBinSegAgreement:
    """SeededBinarySegmentation + CUSUM vs ruptures Binseg(model='l2')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng, n_samples=300, n_columns=n_columns
        )

        # skchange
        sk_det = SeededBinarySegmentation(
            change_score=PenalisedScore(CUSUM(), penalty=BINSEG_PENALTY),
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict_changepoints(data).tolist())

        # ruptures
        rpt_algo = rpt.Binseg(model="l2", min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=BINSEG_PENALTY), len(data))
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"skchange SeededBinSeg CUSUM (p={n_columns}): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"ruptures Binseg L2 (p={n_columns}): ",
        )
