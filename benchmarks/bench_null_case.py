"""Null-case benchmarks: zero change-points, various distributions.

Run with::

    uv run pytest benchmarks/bench_null_case.py --benchmark-only -v

The ``benchmark.pedantic`` API is used throughout so that *only* the detector
fit/predict call is timed.  Dataset generation and detector instantiation happen
in the ``setup`` callable and are excluded from timing.

skchange benchmarks use the ``skchange.new_api`` submodule:

* Input is a plain ``numpy.ndarray`` of shape ``(n_samples, n_features)`` —
  no ``pandas.DataFrame`` wrapper needed.
* ``detector.fit(X)`` fits the detector; ``detector.predict_changepoints(X)``
  returns a numpy array of change-point indices.
* Costs and change-scores are explicit constructor arguments, e.g.
  ``PELT(cost=L2Cost())``.
"""

from __future__ import annotations

import numpy as np
import pytest
import ruptures as rpt
from skchange.new_api.detectors import (
    CROPS,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.detectors import (
    PELT as SkchangePELT,
)
from skchange.new_api.interval_scorers import (
    CUSUM,
    GaussianCost,
    L2Cost,
)

# ---------------------------------------------------------------------------
# ruptures benchmarks
# ---------------------------------------------------------------------------


class TestRupturesNull:
    """Benchmark ruptures change-point detectors on null-case data."""

    def test_pelt_rbf(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """ruptures PELT with RBF cost."""
        data = null_dataset

        def setup():
            algo = rpt.Pelt(model="rbf")
            algo.fit(data)
            return (algo,), {}

        def run(algo):
            return algo.predict(pen=10)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_pelt_l2(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """ruptures PELT with L2 cost."""
        data = null_dataset

        def setup():
            algo = rpt.Pelt(model="l2")
            algo.fit(data)
            return (algo,), {}

        def run(algo):
            return algo.predict(pen=10)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_binseg_rbf(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """ruptures Binary Segmentation with RBF cost."""
        data = null_dataset

        def setup():
            algo = rpt.Binseg(model="rbf")
            algo.fit(data)
            return (algo,), {}

        def run(algo):
            return algo.predict(n_bkps=0)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_window_rbf(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """ruptures Window sliding with RBF cost."""
        data = null_dataset

        def setup():
            algo = rpt.Window(model="rbf")
            algo.fit(data)
            return (algo,), {}

        def run(algo):
            return algo.predict(n_bkps=0)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)


# ---------------------------------------------------------------------------
# skchange new_api benchmarks
# ---------------------------------------------------------------------------


class TestSkchangeNull:
    """Benchmark skchange (new_api) change-point detectors on null-case data.

    All detectors receive a plain ``numpy.ndarray`` of shape
    ``(n_samples, n_features)`` as required by the new single-series API.
    The timed operation is ``fit(X)`` + ``predict_changepoints(X)`` together,
    which mirrors realistic usage where the detector is re-fitted per series.
    """

    def test_pelt_l2(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange new_api PELT with L2Cost."""

        def setup():
            det = SkchangePELT(cost=L2Cost())
            return (det, null_dataset), {}

        def run(det, X):
            det.fit(X)
            return det.predict_changepoints(X)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_pelt_gaussian(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange new_api PELT with GaussianCost."""

        def setup():
            det = SkchangePELT(cost=GaussianCost())
            return (det, null_dataset), {}

        def run(det, X):
            det.fit(X)
            return det.predict_changepoints(X)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_moving_window_cusum(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange new_api MovingWindow with CUSUM change-score."""

        def setup():
            det = MovingWindow(change_score=CUSUM())
            return (det, null_dataset), {}

        def run(det, X):
            det.fit(X)
            return det.predict_changepoints(X)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_seeded_binseg_cusum(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange new_api SeededBinarySegmentation with CUSUM change-score."""

        def setup():
            det = SeededBinarySegmentation(change_score=CUSUM())
            return (det, null_dataset), {}

        def run(det, X):
            det.fit(X)
            return det.predict_changepoints(X)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_crops_l2(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange new_api CROPS with L2Cost."""

        def setup():
            det = CROPS(cost=L2Cost())
            return (det, null_dataset), {}

        def run(det, X):
            det.fit(X)
            return det.predict_changepoints(X)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

