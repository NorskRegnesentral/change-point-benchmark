"""Benchmark problem definitions.

A :class:`BenchmarkProblem` couples a dataset configuration with the ground-truth
change-point locations so that detectors can be evaluated for accuracy as well
as speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from change_bench.datasets.null_case import DistributionLike, NullDatasetConfig


@dataclass
class BenchmarkProblem:
    """A fully-specified benchmark problem.

    Parameters
    ----------
    name:
        Human-readable identifier used in benchmark output and reports.
    dataset_config:
        Configuration object that knows how to generate the raw time series.
    true_changepoints:
        Ground-truth change-point indices (empty list for null-case problems).
    """

    name: str
    dataset_config: NullDatasetConfig
    true_changepoints: list[int] = field(default_factory=list)

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        """Generate the dataset array for this problem.

        Parameters
        ----------
        rng:
            Explicit NumPy random generator for reproducibility.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_columns)``.
        """
        return self.dataset_config.generate(rng)


def make_null_problems(
    n_samples_list: Sequence[int] = (500, 1000, 5000),
    distributions: Sequence[DistributionLike] = (
        "normal",
        "t",
        "gamma",
        "laplace",
        "exponential",
    ),
    scale: float = 1.0,
    n_columns: int = 1,
) -> list[BenchmarkProblem]:
    """Create a standard battery of null-case benchmark problems.

    Every combination of ``n_samples_list × distributions`` produces one
    :class:`BenchmarkProblem` with no change points (``true_changepoints=[]``).

    Parameters
    ----------
    n_samples_list:
        Sequence of time-series lengths to benchmark.
    distributions:
        Distribution names (strings from
        :data:`~change_bench.datasets.null_case.NAMED_DISTRIBUTIONS`)
        or frozen scipy distributions to include.
    scale:
        Spread parameter forwarded to
        :class:`~change_bench.datasets.null_case.NullDatasetConfig`.
    n_columns:
        Number of independent channels per time series.

    Returns
    -------
    list[BenchmarkProblem]
        One problem per ``(n_samples, distribution)`` combination.
    """
    problems: list[BenchmarkProblem] = []
    for n in n_samples_list:
        for dist in distributions:
            dist_label = dist if isinstance(dist, str) else type(dist).__name__
            problems.append(
                BenchmarkProblem(
                    name=f"null_{dist_label}",
                    dataset_config=NullDatasetConfig(
                        n_samples=n,
                        distribution=dist,
                        scale=scale,
                        n_columns=n_columns,
                    ),
                    true_changepoints=[],
                )
            )
    return problems
