"""Unit tests for null-case dataset generation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from change_bench.datasets.null_case import NAMED_DISTRIBUTIONS, NullDatasetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def fresh_rng() -> np.random.Generator:
    """Return a fresh deterministic generator for each test."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNullDatasetConfigShape:
    """Generated arrays have the expected shape."""

    @pytest.mark.parametrize("n_samples", [10, 100, 500])
    def test_n_samples(self, n_samples: int) -> None:
        cfg = NullDatasetConfig(n_samples=n_samples, distribution="normal")
        data = cfg.generate(fresh_rng())
        assert data.shape[0] == n_samples

    @pytest.mark.parametrize("n_columns", [1, 3, 10])
    def test_n_columns(self, n_columns: int) -> None:
        cfg = NullDatasetConfig(n_samples=50, distribution="normal", n_columns=n_columns)
        data = cfg.generate(fresh_rng())
        assert data.shape == (50, n_columns)


class TestNullDatasetConfigDistributions:
    """All named distributions can be generated without error."""

    @pytest.mark.parametrize("dist_name", list(NAMED_DISTRIBUTIONS))
    def test_named_distribution(self, dist_name: str) -> None:
        cfg = NullDatasetConfig(n_samples=200, distribution=dist_name)
        data = cfg.generate(fresh_rng())
        assert data.shape == (200, 1)
        assert np.all(np.isfinite(data)), "Generated data contains non-finite values."

    def test_unknown_distribution_raises(self) -> None:
        cfg = NullDatasetConfig(n_samples=10, distribution="not_a_dist")
        with pytest.raises(ValueError, match="Unknown distribution"):
            cfg.generate(fresh_rng())

    def test_frozen_scipy_distribution(self) -> None:
        frozen = sp_stats.norm(loc=5.0, scale=2.0)
        cfg = NullDatasetConfig(n_samples=100, distribution=frozen)
        data = cfg.generate(fresh_rng())
        assert data.shape == (100, 1)
        # Mean should be near 5 (not zero) because we passed a frozen dist.
        assert abs(data.mean() - 5.0) < 1.0

    def test_unfrozen_scipy_distribution(self) -> None:
        cfg = NullDatasetConfig(
            n_samples=100, distribution=sp_stats.norm, scale=3.0
        )
        data = cfg.generate(fresh_rng())
        assert data.shape == (100, 1)

    def test_invalid_type_raises(self) -> None:
        cfg = NullDatasetConfig(n_samples=10, distribution=42)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="distribution must be"):
            cfg.generate(fresh_rng())


class TestNullDatasetConfigScale:
    """The scale parameter controls the spread of the data."""

    @pytest.mark.parametrize("scale", [0.1, 1.0, 5.0])
    def test_scale_affects_std(self, scale: float) -> None:
        cfg = NullDatasetConfig(n_samples=5000, distribution="normal", scale=scale)
        data = cfg.generate(fresh_rng())
        assert abs(data.std() - scale) < 0.1 * scale  # within 10 %


class TestNullDatasetConfigMean:
    """Named distributions are centred at zero (null mean)."""

    @pytest.mark.parametrize(
        "dist_name",
        ["normal", "t", "laplace"],  # symmetric → exact zero mean
    )
    def test_zero_mean_symmetric(self, dist_name: str) -> None:
        cfg = NullDatasetConfig(n_samples=10_000, distribution=dist_name, scale=1.0)
        data = cfg.generate(fresh_rng())
        assert abs(data.mean()) < 0.1, f"Mean too far from zero: {data.mean()}"

    @pytest.mark.parametrize(
        "dist_name",
        ["gamma", "exponential", "lognormal"],  # shifted to zero
    )
    def test_zero_mean_shifted(self, dist_name: str) -> None:
        cfg = NullDatasetConfig(n_samples=10_000, distribution=dist_name, scale=1.0)
        data = cfg.generate(fresh_rng())
        assert abs(data.mean()) < 0.2, f"Mean too far from zero: {data.mean()}"


class TestReproducibility:
    """Same RNG seed produces identical datasets."""

    def test_same_seed_same_data(self) -> None:
        cfg = NullDatasetConfig(n_samples=100, distribution="normal")
        data1 = cfg.generate(np.random.default_rng(7))
        data2 = cfg.generate(np.random.default_rng(7))
        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_different_data(self) -> None:
        cfg = NullDatasetConfig(n_samples=100, distribution="normal")
        data1 = cfg.generate(np.random.default_rng(1))
        data2 = cfg.generate(np.random.default_rng(2))
        assert not np.array_equal(data1, data2)
