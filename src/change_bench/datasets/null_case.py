"""Null-case dataset generation (zero change points in mean).

All generators take an explicit :class:`numpy.random.Generator` so benchmarks
are fully reproducible without relying on global seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Named-distribution catalogue
# ---------------------------------------------------------------------------

#: Mapping from friendly distribution names to their ``scipy.stats`` classes.
NAMED_DISTRIBUTIONS: dict[str, sp_stats.rv_continuous] = {
    "normal": sp_stats.norm,
    "t": sp_stats.t,
    "gamma": sp_stats.gamma,
    "laplace": sp_stats.laplace,
    "uniform": sp_stats.uniform,
    "exponential": sp_stats.expon,
    "lognormal": sp_stats.lognorm,
}

# Type alias for what a "distribution" argument can be.
DistributionLike = Union[
    str,
    sp_stats.rv_continuous,
    "sp_stats.rv_frozen",  # already-frozen distribution
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class NullDatasetConfig:
    """Configuration for a null-case dataset (no change points in mean).

    Parameters
    ----------
    n_samples:
        Number of time steps / observations (rows in the output array).
    distribution:
        Which distribution to draw samples from.  Either a string key from
        :data:`NAMED_DISTRIBUTIONS` or any scipy frozen distribution
        (e.g. ``scipy.stats.norm(loc=0, scale=2)``).  When a string is
        provided the distribution is parameterised with ``loc=0`` and the
        given *scale* (see below).
    scale:
        Controls the **width** (spread) of the distribution.  For ``"normal"``
        this equals the standard deviation; for ``"t"`` and ``"laplace"`` it
        is the scale parameter; for ``"gamma"`` it is the per-sample scale.
        Ignored when *distribution* is already a frozen scipy object.
    df:
        Degrees-of-freedom for the Student-t (``"t"``) distribution.
        Ignored for all other named distributions.
    shape:
        Shape parameter ``a`` for the ``"gamma"`` distribution, and ``s``
        (log-scale std) for ``"lognormal"``.
        Ignored for all other named distributions.
    n_columns:
        Number of independent channels / dimensions.  The output array will
        have shape ``(n_samples, n_columns)``.
    """

    n_samples: int
    distribution: DistributionLike
    scale: float = 1.0
    df: float = 5.0
    shape: float = 2.0
    n_columns: int = 1
    # internal: cached frozen distribution (not part of the public config)
    _frozen: sp_stats.rv_frozen | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_frozen(self) -> sp_stats.rv_frozen | sp_stats.rv_continuous:
        """Return a frozen scipy distribution based on the current config."""
        dist = self.distribution

        # Already a frozen distribution – use as-is.
        if hasattr(dist, "rvs") and hasattr(dist, "args"):
            return dist  # type: ignore[return-value]

        # Unfrozen scipy rv_continuous subclass passed directly.
        if isinstance(dist, sp_stats.rv_continuous):
            return dist(loc=0.0, scale=self.scale)

        # String name.
        if not isinstance(dist, str):
            raise TypeError(
                f"distribution must be a string or a scipy distribution, "
                f"got {type(dist)!r}."
            )
        name = dist.lower()
        if name not in NAMED_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {name!r}. "
                f"Choose from: {sorted(NAMED_DISTRIBUTIONS)}."
            )
        base = NAMED_DISTRIBUTIONS[name]

        if name == "t":
            return base(df=self.df, loc=0.0, scale=self.scale)

        if name == "gamma":
            # Shift so that the mean is zero: mean = shape * scale.
            mean = self.shape * self.scale
            return base(a=self.shape, loc=-mean, scale=self.scale)

        if name == "lognormal":
            # Shift by the distribution mean so E[X] ≈ 0.
            # For LN(0, s²): mean = exp(s²/2).
            mean = np.exp(self.shape**2 / 2.0)
            return base(s=self.shape, loc=-mean, scale=1.0)

        if name == "uniform":
            # Symmetric uniform on [-scale/2, scale/2].
            return base(loc=-self.scale / 2.0, scale=self.scale)

        if name == "exponential":
            # Shift by the mean (= scale) so that E[X] = 0.
            return base(loc=-self.scale, scale=self.scale)

        # Default: loc=0, scale=scale (works for "normal", "laplace", …).
        return base(loc=0.0, scale=self.scale)

    @property
    def frozen(self) -> sp_stats.rv_frozen:
        """Lazily-built frozen scipy distribution."""
        if self._frozen is None:
            object.__setattr__(self, "_frozen", self._build_frozen())
        return self._frozen  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a null dataset with shape ``(n_samples, n_columns)``.

        Parameters
        ----------
        rng:
            Explicit NumPy random generator.  Pass ``numpy.random.default_rng(seed)``
            for reproducible results.

        Returns
        -------
        numpy.ndarray
            Float array of shape ``(n_samples, n_columns)`` drawn from the
            configured distribution with a constant mean (no change points).
        """
        return self.frozen.rvs(
            size=(self.n_samples, self.n_columns),
            random_state=rng,
        )
