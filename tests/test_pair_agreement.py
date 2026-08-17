"""Tests verifying that comparison pairs find the same change points.

For each registered benchmark pair, we generate data with strong, unambiguous
change points and verify that the skchange and ruptures implementations
produce equivalent results (exact for PELT, within tolerance for
MovingWindow/BinSeg due to different peak-selection logic).

Uses the same numeric penalty for both libraries:
- skchange: Detector(change_score=scorer, penalty=P) or PELT(cost=..., penalty=P)
- ruptures: .predict(pen=P)
"""

from __future__ import annotations

import numpy as np
import pytest
import ruptures as rpt
from ruptures.costs import CostLinear
from skchange.detectors import (
    PELT as SkchangePELT,
)
from skchange.detectors import (
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.interval_scorers import (
    ContinuousLinearTrendScore,
    CostChangeScore,
    ESACScore,
    GaussianCost,
    L1Cost,
    L2Cost,
    LinearRegressionCost,
    MultivariateGaussianCost,
    PoissonCost,
    RankCost,
    RankScore,
)

from change_bench.benchmarks.comparison_pairs._common import MW_BANDWIDTH, PELT_PENALTY
from change_bench.benchmarks.comparison_pairs.binseg_mv_gaussian import (
    pair_binseg_mv_gaussian,
)
from change_bench.benchmarks.comparison_pairs.binseg_continuous_linear_trend import (
    JOINT_BINSEG_CONTINUOUS_LINEAR_TREND_PENALTY,
)
from change_bench.benchmarks.comparison_pairs.binseg_rank import (
    JOINT_BINSEG_RANK_PENALTY,
)
from change_bench.benchmarks.comparison_pairs.moving_window_continuous_linear_trend import (
    JOINT_MW_CONTINUOUS_LINEAR_TREND_PENALTY,
)
from change_bench.benchmarks.comparison_pairs.moving_window_rank import (
    JOINT_MW_RANK_PENALTY,
)
from change_bench.benchmarks.comparison_pairs.pelt_poisson import CostPoisson
from change_bench.benchmarks.comparison_pairs.pelt_rank import (
    JOINT_PELT_RANK_PENALTY,
)
from change_bench.datasets.change_case import ChangeDatasetConfig, SegmentParams
from change_bench.problems.base import make_null_problems

# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

#: Penalty used for all tests (same as benchmark runs for PELT pairs).
TEST_PENALTY: float = PELT_PENALTY

#: Penalty for MovingWindow / BinSeg tests.
#: Must be tuned so that strong mean shifts (5× scale) are detected.
SKCHANGE_MW_PENALTY: float = 20.0
RUPTURES_MW_PENALTY: float = SKCHANGE_MW_PENALTY

#: Higher penalty for BinSeg (SeededBinarySegmentation is more sensitive).
SKCHANGE_BINSEG_PENALTY: float = 20.0
RUPTURES_BINSEG_PENALTY: float = SKCHANGE_BINSEG_PENALTY

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
    *,
    n_samples: int,
    changepoints: list[int],
    means: list[float],
    n_columns: int = 1,
) -> np.ndarray:
    """Generate data with strong mean shifts for testing."""
    cfg = ChangeDatasetConfig(
        n_samples=n_samples,
        changepoints=changepoints,
        segments=[SegmentParams(loc=m, scale=1.0) for m in means],
        distribution="normal",
        n_columns=n_columns,
    )
    return cfg.generate(rng)


def _make_linear_regression_change_data(rng: np.random.Generator) -> np.ndarray:
    predictor = rng.normal(size=300)
    slopes = np.repeat([1.0, 4.0, -2.0], 100)
    response = slopes * predictor + rng.normal(scale=0.2, size=300)
    return np.column_stack([response, predictor])


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

    # Do not allow spurious detections:
    max_detections = len(expected)
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
        changepoints = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        # skchange
        sk_det = SkchangePELT(cost=L2Cost(), penalty=TEST_PENALTY, min_segment_length=1)
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

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
        changepoints = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
        )

        sk_det = SkchangePELT(cost=L2Cost(), penalty=TEST_PENALTY, min_segment_length=1)
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # With strong signal (mean shift = 5), both should find ~[100, 200]
        _assert_changepoints_close(sk_cps, changepoints, tolerance=5)


class TestPeltL1Agreement:
    """PELT + L1 cost: skchange vs ruptures Pelt(model='l1')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_same_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        sk_detector = SkchangePELT(
            cost=L1Cost(), penalty=TEST_PENALTY, min_segment_length=1
        ).fit(data)
        sk_changepoints = sorted(sk_detector.predict(data).tolist())

        rpt_algorithm = rpt.Pelt(model="l1", min_size=1, jump=1).fit(data)
        rpt_changepoints = sorted(
            _strip_endpoint(
                rpt_algorithm.predict(pen=TEST_PENALTY), len(data)
            )
        )

        assert sk_changepoints == rpt_changepoints


class TestPelt1dGaussianAgreement:
    """PELT + Gaussian cost: skchange vs ruptures Pelt(model='normal')."""

    def test_same_changepoints(self):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=1,
        )

        # skchange — GaussianCost requires min_segment_length >= 2
        sk_det = SkchangePELT(
            cost=GaussianCost(), penalty=TEST_PENALTY, min_segment_length=2
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Pelt(model="normal", min_size=2, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(_strip_endpoint(rpt_algo.predict(pen=TEST_PENALTY), len(data)))

        assert sk_cps == rpt_cps, (
            f"PELT Gaussian disagreement: skchange={sk_cps}, ruptures={rpt_cps}"
        )

    def test_detects_known_changepoints(self):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=1,
        )

        sk_det = SkchangePELT(
            cost=GaussianCost(), penalty=TEST_PENALTY, min_segment_length=2
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # GaussianCost models mean+variance, so it may find extra changes.
        # Just verify the true change points are among those detected.
        for expected_cp in changepoints:
            nearby = [cp for cp in sk_cps if abs(cp - expected_cp) <= 5]
            assert len(nearby) >= 1, (
                f"Expected changepoint near {expected_cp} not found. Detected: {sk_cps}"
            )


class TestPeltPoissonAgreement:
    """PELT + Poisson cost: skchange vs ruptures Pelt(custom_cost=CostPoisson())."""

    #: Change points used by _make_poisson_data.
    CHANGEPOINTS = [100, 200]

    @staticmethod
    def _make_poisson_data(rng: np.random.Generator) -> np.ndarray:
        """Generate non-negative data with different rates per segment."""
        cfg = ChangeDatasetConfig(
            n_samples=300,
            changepoints=TestPeltPoissonAgreement.CHANGEPOINTS,
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
        sk_cps = sorted(sk_det.predict(data).tolist())

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
        sk_cps = sorted(sk_det.predict(data).tolist())

        _assert_changepoints_close(sk_cps, self.CHANGEPOINTS, tolerance=5)


class TestPeltRankAgreement:
    """PELT + Rank cost: skchange vs ruptures Pelt(model='rank')."""

    def test_same_changepoints(self):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        n_columns = 3
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        # skchange — RankCost requires min_segment_length >= 2
        sk_det = SkchangePELT(
            cost=RankCost(),
            penalty=JOINT_PELT_RANK_PENALTY,
            min_segment_length=2,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Pelt(model="rank", min_size=2, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(
                rpt_algo.predict(pen=JOINT_PELT_RANK_PENALTY), len(data)
            )
        )

        assert sk_cps == rpt_cps, (
            f"PELT Rank disagreement: skchange={sk_cps}, ruptures={rpt_cps}"
        )

    def test_detects_known_changepoints(self):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        n_columns = 3
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        sk_det = SkchangePELT(
            cost=RankCost(),
            penalty=JOINT_PELT_RANK_PENALTY,
            min_segment_length=2,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        _assert_changepoints_close(sk_cps, changepoints, tolerance=5)


class TestPeltLinRegAgreement:
    """PELT linear regression costs use column 0 as the response."""

    def test_same_changepoints(self):
        data = _make_linear_regression_change_data(np.random.default_rng(42))

        sk_detector = SkchangePELT(
            cost=LinearRegressionCost(response_col=0),
            penalty=TEST_PENALTY,
            min_segment_length=2,
        ).fit(data)
        sk_changepoints = sorted(sk_detector.predict(data).tolist())

        rpt_algorithm = rpt.Pelt(
            custom_cost=CostLinear(), min_size=2, jump=1
        ).fit(data)
        rpt_changepoints = sorted(
            _strip_endpoint(rpt_algorithm.predict(pen=TEST_PENALTY), len(data))
        )

        assert sk_changepoints == rpt_changepoints == [100, 200]


class TestMovingWindowL2Agreement:
    """MovingWindow + L2 Change Score: skchange vs ruptures Window(model='l2')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_cps,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=CostChangeScore(L2Cost()),
            penalty=SKCHANGE_MW_PENALTY,
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="l2", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=RUPTURES_MW_PENALTY), len(data))
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
    """MovingWindow + L1 Change Score: skchange vs ruptures Window(model='l1')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_cps,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=CostChangeScore(L1Cost()),
            penalty=SKCHANGE_MW_PENALTY,
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="l1", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=RUPTURES_MW_PENALTY), len(data))
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
    """MovingWindow + Rank Change Score: skchange vs ruptures Window(model='rank')."""

    def test_both_find_known_changepoints_multivariate(self):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        n_columns = 3
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_cps,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=RankScore(),
            penalty=JOINT_MW_RANK_PENALTY,
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="rank", width=2 * bw, min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=JOINT_MW_RANK_PENALTY), len(data))
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


class TestMovingWindowLinRegAgreement:
    """Moving-window linear regression scores agree on slope changes."""

    def test_both_find_known_changepoints(self):
        data = _make_linear_regression_change_data(np.random.default_rng(42))

        sk_detector = MovingWindow(
            change_score=CostChangeScore(LinearRegressionCost(response_col=0)),
            penalty=TEST_PENALTY,
            bandwidth=MW_BANDWIDTH,
        ).fit(data)
        sk_changepoints = sorted(sk_detector.predict(data).tolist())

        rpt_algorithm = rpt.Window(
            custom_cost=CostLinear(),
            width=2 * MW_BANDWIDTH,
            min_size=2,
            jump=1,
        ).fit(data)
        rpt_changepoints = sorted(
            _strip_endpoint(rpt_algorithm.predict(pen=TEST_PENALTY), len(data))
        )

        assert sk_changepoints == rpt_changepoints == [100, 199]


# ---------------------------------------------------------------------------
# BinSeg pair agreement test — tolerance-based
# ---------------------------------------------------------------------------


class TestBinSegAgreement:
    """SeededBinarySegmentation + L2 Change Score vs ruptures Binseg(model='l2')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_cps,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        # skchange
        sk_det = SeededBinarySegmentation(
            change_score=CostChangeScore(L2Cost()),
            penalty=SKCHANGE_BINSEG_PENALTY,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Binseg(model="l2", min_size=1, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(rpt_algo.predict(pen=RUPTURES_BINSEG_PENALTY), len(data))
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"skchange SeededBinSeg L2 Change Score (p={n_columns}): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg=f"ruptures Binseg L2 Change Score (p={n_columns}): ",
        )


class TestBinSegL1Agreement:
    """Seeded binary segmentation + L1 vs ruptures Binseg(model='l1')."""

    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_both_find_known_changepoints(self, n_columns: int):
        rng = np.random.default_rng(42)
        expected_changepoints = [100, 200]
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        sk_detector = SeededBinarySegmentation(
            change_score=CostChangeScore(L1Cost()),
            penalty=SKCHANGE_BINSEG_PENALTY,
        ).fit(data)
        sk_changepoints = sorted(sk_detector.predict(data).tolist())

        rpt_algorithm = rpt.Binseg(model="l1", min_size=1, jump=1).fit(data)
        rpt_changepoints = sorted(
            _strip_endpoint(
                rpt_algorithm.predict(pen=RUPTURES_BINSEG_PENALTY), len(data)
            )
        )

        _assert_changepoints_close(
            sk_changepoints,
            expected_changepoints,
            tolerance=TOLERANCE,
            msg=f"skchange SeededBinSeg L1 (p={n_columns}): ",
        )
        _assert_changepoints_close(
            rpt_changepoints,
            expected_changepoints,
            tolerance=TOLERANCE,
            msg=f"ruptures Binseg L1 (p={n_columns}): ",
        )


class TestBinSegLinRegAgreement:
    """Binary-segmentation linear regression scores agree on slope changes."""

    def test_both_find_known_changepoints(self):
        data = _make_linear_regression_change_data(np.random.default_rng(42))

        sk_detector = SeededBinarySegmentation(
            change_score=CostChangeScore(LinearRegressionCost(response_col=0)),
            penalty=TEST_PENALTY,
            min_subinterval_length=2,
            max_interval_length=200,
        ).fit(data)
        sk_changepoints = sorted(sk_detector.predict(data).tolist())

        rpt_algorithm = rpt.Binseg(
            custom_cost=CostLinear(), min_size=2, jump=1
        ).fit(data)
        rpt_changepoints = sorted(
            _strip_endpoint(rpt_algorithm.predict(pen=TEST_PENALTY), len(data))
        )

        assert sk_changepoints == rpt_changepoints == [100, 200]


class TestEsacDetection:
    """ESAC works with both supported skchange search algorithms."""

    @pytest.mark.parametrize("search", ["moving_window", "seeded_binseg"])
    def test_detects_known_multivariate_changepoint(self, search: str):
        rng = np.random.default_rng(42)
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=[150],
            means=[0.0, 5.0],
            n_columns=5,
        )
        if search == "moving_window":
            detector = MovingWindow(change_score=ESACScore(), bandwidth=MW_BANDWIDTH)
        else:
            detector = SeededBinarySegmentation(change_score=ESACScore())

        detected = detector.fit_predict(data).tolist()

        _assert_changepoints_close(detected, [150], tolerance=5)


class TestBinSegRankAgreement:
    """Seeded binary segmentation and ruptures both recover a rank change."""

    def test_both_find_known_changepoint(self):
        rng = np.random.default_rng(42)
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=[150],
            means=[0.0, 5.0],
            n_columns=5,
        )

        sk_detector = SeededBinarySegmentation(
            change_score=RankScore(),
            penalty=JOINT_BINSEG_RANK_PENALTY,
        )
        sk_changepoints = sk_detector.fit_predict(data).tolist()

        rpt_algorithm = rpt.Binseg(model="rank", min_size=2, jump=1).fit(data)
        rpt_changepoints = _strip_endpoint(
            rpt_algorithm.predict(pen=JOINT_BINSEG_RANK_PENALTY), len(data)
        )

        assert any(abs(changepoint - 150) <= 5 for changepoint in sk_changepoints)
        assert any(abs(changepoint - 150) <= 5 for changepoint in rpt_changepoints)


class TestContinuousLinearTrendAgreement:
    """Score-based searches and CostCLinear recover a continuous kink."""

    @pytest.mark.parametrize("search", ["moving_window", "seeded_binseg"])
    def test_both_find_known_kink(self, search: str):
        rng = np.random.default_rng(42)
        n_samples = 300
        expected_changepoint = 150
        time = np.arange(n_samples)
        data = (
            0.03 * time
            + 0.20 * np.maximum(time - expected_changepoint, 0)
            + rng.normal(scale=0.2, size=n_samples)
        )[:, None]

        if search == "moving_window":
            penalty = JOINT_MW_CONTINUOUS_LINEAR_TREND_PENALTY
            sk_detector = MovingWindow(
                change_score=ContinuousLinearTrendScore(),
                penalty=penalty,
                bandwidth=MW_BANDWIDTH,
            )
            rpt_algorithm = rpt.Window(
                custom_cost=rpt.costs.CostCLinear(),
                width=2 * MW_BANDWIDTH,
                min_size=3,
                jump=1,
            )
        else:
            penalty = JOINT_BINSEG_CONTINUOUS_LINEAR_TREND_PENALTY
            sk_detector = SeededBinarySegmentation(
                change_score=ContinuousLinearTrendScore(),
                penalty=penalty,
            )
            rpt_algorithm = rpt.Binseg(
                custom_cost=rpt.costs.CostCLinear(), min_size=3, jump=1
            )

        sk_changepoints = sk_detector.fit_predict(data).tolist()
        rpt_changepoints = _strip_endpoint(
            rpt_algorithm.fit(data).predict(pen=penalty), n_samples
        )

        _assert_changepoints_close(
            sk_changepoints,
            [expected_changepoint],
            tolerance=5,
            msg=f"skchange {search} continuous trend: ",
        )
        _assert_changepoints_close(
            rpt_changepoints,
            [expected_changepoint],
            tolerance=5,
            msg=f"ruptures {search} CostCLinear: ",
        )


# ---------------------------------------------------------------------------
# MultivariateGaussianCost pair agreement tests (multivariate only)
# ---------------------------------------------------------------------------

#: Penalty for MW / BinSeg mv_gaussian tests (match benchmark pair constants).
SKCHANGE_MW_MV_GAUSSIAN_PENALTY: float = 20.0
RUPTURES_MW_MV_GAUSSIAN_PENALTY: float = SKCHANGE_MW_MV_GAUSSIAN_PENALTY
SKCHANGE_BINSEG_MV_GAUSSIAN_PENALTY: float = 60.0
RUPTURES_BINSEG_MV_GAUSSIAN_PENALTY: float = SKCHANGE_BINSEG_MV_GAUSSIAN_PENALTY


class TestPeltMvGaussianAgreement:
    """PELT + MultivariateGaussianCost: skchange vs ruptures Pelt(model='normal')."""

    mv_normal_pelt_penalty = 60.0

    def test_skchange_better_segmentation_cost(self):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        n_columns = 3
        min_seg = n_columns + 1  # MultivariateGaussianCost constraint

        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        # skchange
        sk_det = SkchangePELT(
            cost=MultivariateGaussianCost(),
            penalty=self.mv_normal_pelt_penalty,
            min_segment_length=min_seg,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Pelt(model="normal", min_size=min_seg, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(
                rpt_algo.predict(pen=self.mv_normal_pelt_penalty), len(data)
            )
        )

        sk_changepoint_interval_specs = np.array(
            [[0, sk_cps[0]]]
            + [[sk_cps[i], sk_cps[i + 1]] for i in range(len(sk_cps) - 1)]
            + [[sk_cps[-1], len(data)]]
        )
        rpt_changepoint_interval_specs = np.array(
            [[0, rpt_cps[0]]]
            + [[rpt_cps[i], rpt_cps[i + 1]] for i in range(len(rpt_cps) - 1)]
            + [[rpt_cps[-1], len(data)]]
        )
        sk_mv_gaussian_cost = MultivariateGaussianCost().fit(data)
        data_cache = sk_mv_gaussian_cost.precompute(data)
        sk_sk_cps_segmentation_cost = sk_mv_gaussian_cost.evaluate(
            data_cache,
            sk_changepoint_interval_specs,
        ).sum()
        sk_rpt_cps_segmentation_cost = sk_mv_gaussian_cost.evaluate(
            data_cache,
            rpt_changepoint_interval_specs,
        ).sum()

        rpt_rpt_cps_cost = rpt_algo.cost.sum_of_costs(rpt_cps + [len(data)])
        rpt_sk_cps_cost = rpt_algo.cost.sum_of_costs(sk_cps + [len(data)])

        assert sk_sk_cps_segmentation_cost <= sk_rpt_cps_segmentation_cost, (
            f"Skchange PELT MvGaussian found worse segmentation than ruptures: "
            f"skchange={sk_cps}, ruptures={rpt_cps}"
        )
        assert rpt_sk_cps_cost <= rpt_rpt_cps_cost, (
            f"Skchange PELT MvGaussian found worse segmentation than ruptures: "
            f"skchange={sk_cps}, ruptures={rpt_cps}"
        )

    def test_detects_known_changepoints(self):
        rng = np.random.default_rng(42)
        changepoints = [100, 200]
        n_columns = 3
        min_seg = n_columns + 1
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=changepoints,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        sk_det = SkchangePELT(
            cost=MultivariateGaussianCost(),
            penalty=self.mv_normal_pelt_penalty,
            min_segment_length=min_seg,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        _assert_changepoints_close(sk_cps, changepoints, tolerance=5)


class TestMovingWindowMvGaussianAgreement:
    """Compare MovingWindow with multivariate Gaussian costs."""

    def test_both_find_known_changepoints_multivariate(self):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        n_columns = 3
        min_seg = n_columns + 1
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_cps,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )
        bw = MW_BANDWIDTH

        # skchange
        sk_det = MovingWindow(
            change_score=CostChangeScore(MultivariateGaussianCost()),
            penalty=SKCHANGE_MW_MV_GAUSSIAN_PENALTY,
            bandwidth=bw,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Window(model="normal", width=2 * bw, min_size=min_seg, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(
                rpt_algo.predict(pen=RUPTURES_MW_MV_GAUSSIAN_PENALTY), len(data)
            )
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg="skchange MovingWindow MvGaussian (multivariate): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg="ruptures Window normal (multivariate): ",
        )


class TestBinSegMvGaussianAgreement:
    """Compare binary segmentation with multivariate Gaussian costs."""

    def test_both_find_known_changepoints_multivariate(self):
        rng = np.random.default_rng(42)
        expected_cps = [100, 200]
        n_columns = 3
        min_seg = n_columns + 1
        data = _make_normal_change_data(
            rng,
            n_samples=300,
            changepoints=expected_cps,
            means=[0.0, 5.0, 0.0],
            n_columns=n_columns,
        )

        # skchange
        sk_det = SeededBinarySegmentation(
            change_score=CostChangeScore(MultivariateGaussianCost()),
            penalty=SKCHANGE_BINSEG_MV_GAUSSIAN_PENALTY,
        )
        sk_det.fit(data)
        sk_cps = sorted(sk_det.predict(data).tolist())

        # ruptures
        rpt_algo = rpt.Binseg(model="normal", min_size=min_seg, jump=1)
        rpt_algo.fit(data)
        rpt_cps = sorted(
            _strip_endpoint(
                rpt_algo.predict(pen=RUPTURES_BINSEG_MV_GAUSSIAN_PENALTY), len(data)
            )
        )

        _assert_changepoints_close(
            sk_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg="skchange SeededBinSeg MvGaussian (multivariate): ",
        )
        _assert_changepoints_close(
            rpt_cps,
            expected_cps,
            tolerance=TOLERANCE,
            msg="ruptures Binseg normal (multivariate): ",
        )

    def test_high_dimension_uses_valid_max_interval_length(self):
        n_columns = 101
        problems = make_null_problems(
            n_samples_list=[300],
            distributions=["normal"],
            scale=1.0,
            n_columns_list=[n_columns],
        )
        sk_case = pair_binseg_mv_gaussian(problems)[0]
        detector, data = sk_case.setup(sk_case.prepare())[0]

        detector.fit(data)

        assert detector.min_subinterval_length_ == n_columns + 1
        assert detector.max_interval_length_ == 2 * (n_columns + 1)
