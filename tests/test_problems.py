"""Unit tests for benchmark problem definitions."""

from __future__ import annotations

import numpy as np

from change_bench.datasets.null_case import NullDatasetConfig
from change_bench.problems.base import BenchmarkProblem, make_null_problems


class TestBenchmarkProblem:
    """Basic BenchmarkProblem behaviour."""

    def test_generate_returns_array(self) -> None:
        prob = BenchmarkProblem(
            name="test",
            dataset_config=NullDatasetConfig(n_samples=50, distribution="normal"),
        )
        data = prob.generate(np.random.default_rng(0))
        assert isinstance(data, np.ndarray)
        assert data.shape == (50, 1)

    def test_default_no_changepoints(self) -> None:
        prob = BenchmarkProblem(
            name="null",
            dataset_config=NullDatasetConfig(n_samples=50, distribution="normal"),
        )
        assert prob.true_changepoints == []


class TestMakeNullProblems:
    """make_null_problems factory."""

    def test_returns_list_of_problems(self) -> None:
        problems = make_null_problems(
            n_samples_list=[100],
            distributions=["normal"],
        )
        assert len(problems) == 1
        assert isinstance(problems[0], BenchmarkProblem)

    def test_count(self) -> None:
        problems = make_null_problems(
            n_samples_list=[100, 500],
            distributions=["normal", "t", "gamma"],
        )
        assert len(problems) == 6  # 2 × 3

    def test_count_with_dimensions(self) -> None:
        problems = make_null_problems(
            n_samples_list=[100, 500],
            distributions=["normal", "t"],
            n_columns_list=[1, 5],
        )
        assert len(problems) == 8  # 2 n_samples × 2 dists × 2 dims

    def test_names_are_unique_per_n_samples_and_dimension(self) -> None:
        problems = make_null_problems(n_columns_list=[1, 5])
        keys = [
            (p.name, p.dataset_config.n_samples, p.dataset_config.n_columns)
            for p in problems
        ]
        assert len(keys) == len(set(keys))

    def test_all_null_changepoints(self) -> None:
        problems = make_null_problems(n_samples_list=[200], distributions=["normal"])
        assert all(p.true_changepoints == [] for p in problems)

    def test_generate_produces_correct_shape(self) -> None:
        rng = np.random.default_rng(99)
        problems = make_null_problems(
            n_samples_list=[300],
            distributions=["normal"],
            n_columns_list=[2],
        )
        data = problems[0].generate(rng)
        assert data.shape == (300, 2)

    def test_multivariate_shapes(self) -> None:
        rng = np.random.default_rng(42)
        problems = make_null_problems(
            n_samples_list=[100],
            distributions=["normal"],
            n_columns_list=[1, 5, 10],
        )
        assert len(problems) == 3
        for prob, expected_p in zip(problems, [1, 5, 10]):
            data = prob.generate(rng)
            assert data.shape == (100, expected_p)
