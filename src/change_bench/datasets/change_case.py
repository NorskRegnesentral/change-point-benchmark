"""Change-case dataset generation (piecewise-stationary with known change points).

Generates time series with distinct segments, each drawn from the same
distribution family but with different location/scale parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from change_bench.datasets.null_case import (
    DistributionLike,
    NullDatasetConfig,
)


@dataclass
class SegmentParams:
    """Per-segment distribution parameters.

    Parameters
    ----------
    loc:
        Location (mean) parameter for this segment.
    scale:
        Scale (spread) parameter for this segment.
    """

    loc: float = 0.0
    scale: float = 1.0


@dataclass
class ChangeDatasetConfig:
    """Configuration for a dataset with known change points.

    Each segment is drawn from the same distribution family but with
    segment-specific location and scale parameters.

    Parameters
    ----------
    n_samples:
        Total number of observations (rows in the output array).
    changepoints:
        Sorted list of interior change-point indices.  Each value must be
        in ``(0, n_samples)``.  The number of segments is
        ``len(changepoints) + 1``.
    segments:
        Per-segment parameters.  Must have length ``len(changepoints) + 1``.
    distribution:
        Distribution family for all segments.  Either a string key from
        :data:`~change_bench.datasets.null_case.NAMED_DISTRIBUTIONS` or a
        scipy distribution object.
    n_columns:
        Number of independent channels / dimensions.
    df:
        Degrees-of-freedom for the Student-t distribution.
    shape:
        Shape parameter for gamma / lognormal distributions.
    """

    n_samples: int
    changepoints: list[int]
    segments: list[SegmentParams]
    distribution: DistributionLike = "normal"
    n_columns: int = 1
    df: float = 5.0
    shape: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        n_segments = len(self.changepoints) + 1
        if len(self.segments) != n_segments:
            raise ValueError(
                f"Expected {n_segments} segments for {len(self.changepoints)} "
                f"changepoints, got {len(self.segments)}."
            )
        for i, cp in enumerate(self.changepoints):
            if cp <= 0 or cp >= self.n_samples:
                raise ValueError(
                    f"Changepoint {i} has value {cp}, must be in "
                    f"(0, {self.n_samples})."
                )
        if self.changepoints != sorted(self.changepoints):
            raise ValueError("Changepoints must be sorted in ascending order.")

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a piecewise-stationary dataset.

        Parameters
        ----------
        rng:
            Explicit NumPy random generator for reproducibility.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_columns)`` with segment-specific
            distribution parameters.
        """
        # Build segment boundaries: [0, cp1, cp2, ..., n_samples]
        boundaries = [0] + self.changepoints + [self.n_samples]

        parts: list[np.ndarray] = []
        for seg_idx, seg_params in enumerate(self.segments):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            n = end - start

            # Build a NullDatasetConfig for this segment to reuse the frozen
            # distribution logic.
            seg_config = NullDatasetConfig(
                n_samples=n,
                distribution=self.distribution,
                scale=seg_params.scale,
                df=self.df,
                shape=self.shape,
                n_columns=self.n_columns,
            )
            # Generate zero-mean data and shift by segment loc
            segment_data = seg_config.generate(rng) + seg_params.loc
            parts.append(segment_data)

        return np.concatenate(parts, axis=0)
