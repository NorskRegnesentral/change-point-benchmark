"""Tests for ChangeDatasetConfig and its integration with BenchmarkProblem."""

from __future__ import annotations

import numpy as np
import pytest

from change_bench.benchmarks.comparison_pairs._common import make_prepare
from change_bench.datasets.change_case import ChangeDatasetConfig, SegmentParams
from change_bench.problems.base import BenchmarkProblem

# ---------------------------------------------------------------------------
# ChangeDatasetConfig generation tests
# ---------------------------------------------------------------------------


class TestChangeDatasetConfigShape:
    """Output shape matches configuration."""

    def test_basic_shape(self):
        cfg = ChangeDatasetConfig(
            n_samples=300,
            changepoints=[100, 200],
            segments=[
                SegmentParams(loc=0.0),
                SegmentParams(loc=5.0),
                SegmentParams(loc=0.0),
            ],
        )
        data = cfg.generate(np.random.default_rng(42))
        assert data.shape == (300, 1)

    @pytest.mark.parametrize("n_columns", [1, 3, 5])
    def test_multivariate_shape(self, n_columns: int):
        cfg = ChangeDatasetConfig(
            n_samples=200,
            changepoints=[100],
            segments=[SegmentParams(loc=0.0), SegmentParams(loc=3.0)],
            n_columns=n_columns,
        )
        data = cfg.generate(np.random.default_rng(42))
        assert data.shape == (200, n_columns)

    def test_single_segment_no_changepoints(self):
        cfg = ChangeDatasetConfig(
            n_samples=100,
            changepoints=[],
            segments=[SegmentParams(loc=2.0, scale=0.5)],
        )
        data = cfg.generate(np.random.default_rng(42))
        assert data.shape == (100, 1)


class TestChangeDatasetConfigSegmentMeans:
    """Segments have approximately correct mean values."""

    def test_two_segments_mean_shift(self):
        cfg = ChangeDatasetConfig(
            n_samples=2000,
            changepoints=[1000],
            segments=[SegmentParams(loc=0.0), SegmentParams(loc=10.0)],
        )
        data = cfg.generate(np.random.default_rng(42))
        seg1_mean = data[:1000].mean()
        seg2_mean = data[1000:].mean()
        assert abs(seg1_mean - 0.0) < 0.2
        assert abs(seg2_mean - 10.0) < 0.2

    def test_three_segments(self):
        cfg = ChangeDatasetConfig(
            n_samples=3000,
            changepoints=[1000, 2000],
            segments=[
                SegmentParams(loc=0.0),
                SegmentParams(loc=5.0),
                SegmentParams(loc=-3.0),
            ],
        )
        data = cfg.generate(np.random.default_rng(42))
        assert abs(data[:1000].mean() - 0.0) < 0.2
        assert abs(data[1000:2000].mean() - 5.0) < 0.2
        assert abs(data[2000:].mean() - (-3.0)) < 0.2

    def test_scale_affects_spread(self):
        cfg = ChangeDatasetConfig(
            n_samples=2000,
            changepoints=[1000],
            segments=[
                SegmentParams(loc=0.0, scale=1.0),
                SegmentParams(loc=0.0, scale=5.0),
            ],
        )
        data = cfg.generate(np.random.default_rng(42))
        std1 = data[:1000].std()
        std2 = data[1000:].std()
        assert std2 > std1 * 3  # scale=5 should be much wider than scale=1


class TestChangeDatasetConfigValidation:
    """Validation in __post_init__ catches invalid configs."""

    def test_wrong_number_of_segments(self):
        with pytest.raises(ValueError, match="Expected 3 segments"):
            ChangeDatasetConfig(
                n_samples=300,
                changepoints=[100, 200],
                segments=[SegmentParams(), SegmentParams()],  # need 3
            )

    def test_changepoint_at_zero(self):
        with pytest.raises(ValueError, match="must be in"):
            ChangeDatasetConfig(
                n_samples=100,
                changepoints=[0],
                segments=[SegmentParams(), SegmentParams()],
            )

    def test_changepoint_at_n_samples(self):
        with pytest.raises(ValueError, match="must be in"):
            ChangeDatasetConfig(
                n_samples=100,
                changepoints=[100],
                segments=[SegmentParams(), SegmentParams()],
            )

    def test_unsorted_changepoints(self):
        with pytest.raises(ValueError, match="sorted"):
            ChangeDatasetConfig(
                n_samples=300,
                changepoints=[200, 100],
                segments=[SegmentParams(), SegmentParams(), SegmentParams()],
            )


class TestChangeDatasetConfigDistributions:
    """Works with different distribution families."""

    @pytest.mark.parametrize("dist", ["normal", "t", "laplace"])
    def test_symmetric_distributions(self, dist: str):
        cfg = ChangeDatasetConfig(
            n_samples=2000,
            changepoints=[1000],
            segments=[SegmentParams(loc=0.0), SegmentParams(loc=5.0)],
            distribution=dist,
        )
        data = cfg.generate(np.random.default_rng(42))
        assert data.shape == (2000, 1)
        # Both segments should be roughly centered on their loc
        assert abs(data[:1000].mean() - 0.0) < 0.5
        assert abs(data[1000:].mean() - 5.0) < 0.5


# ---------------------------------------------------------------------------
# Integration with BenchmarkProblem
# ---------------------------------------------------------------------------


class TestBenchmarkProblemWithChangeConfig:
    """ChangeDatasetConfig works within BenchmarkProblem infrastructure."""

    def test_generate(self):
        cfg = ChangeDatasetConfig(
            n_samples=300,
            changepoints=[100, 200],
            segments=[
                SegmentParams(loc=0.0),
                SegmentParams(loc=5.0),
                SegmentParams(loc=0.0),
            ],
        )
        problem = BenchmarkProblem(
            name="test_change",
            dataset_config=cfg,
            true_changepoints=[100, 200],
        )
        data = problem.generate(np.random.default_rng(42))
        assert data.shape == (300, 1)

    def test_make_prepare_works(self):
        """The benchmark machinery's make_prepare closure works."""
        cfg = ChangeDatasetConfig(
            n_samples=300,
            changepoints=[100, 200],
            segments=[
                SegmentParams(loc=0.0),
                SegmentParams(loc=5.0),
                SegmentParams(loc=0.0),
            ],
        )
        problem = BenchmarkProblem(
            name="test_change",
            dataset_config=cfg,
            true_changepoints=[100, 200],
        )
        prepare = make_prepare(problem)
        data = prepare()
        assert data.shape == (300, 1)
        # Same seed should give same data
        data2 = prepare()
        np.testing.assert_array_equal(data, data2)
