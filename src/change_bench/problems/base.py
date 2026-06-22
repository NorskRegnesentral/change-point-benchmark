"""Benchmark problem definitions.

A :class:`BenchmarkProblem` couples a dataset configuration with the ground-truth
change-point locations so that detectors can be evaluated for accuracy as well
as speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from change_bench.datasets.change_case import ChangeDatasetConfig
from change_bench.datasets.null_case import DistributionLike, NullDatasetConfig

#: Type alias for any dataset config that provides a ``generate(rng)`` method.
DatasetConfig = NullDatasetConfig | ChangeDatasetConfig


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
    dataset_config: DatasetConfig
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
    n_columns_list: Sequence[int] = (1,),
) -> list[BenchmarkProblem]:
    """Create a standard battery of null-case benchmark problems.

    Every combination of ``n_samples_list × distributions × n_columns_list``
    produces one :class:`BenchmarkProblem` with no change points.

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
    n_columns_list:
        Sequence of dimensionalities (number of columns) to benchmark.

    Returns
    -------
    list[BenchmarkProblem]
        One problem per ``(n_samples, distribution, n_columns)`` combination.
    """
    problems: list[BenchmarkProblem] = []
    for n in n_samples_list:
        for dist in distributions:
            for n_cols in n_columns_list:
                dist_label = dist if isinstance(dist, str) else type(dist).__name__
                problems.append(
                    BenchmarkProblem(
                        name=f"null_{dist_label}",
                        dataset_config=NullDatasetConfig(
                            n_samples=n,
                            distribution=dist,
                            scale=scale,
                            n_columns=n_cols,
                        ),
                        true_changepoints=[],
                    )
                )
    return problems
