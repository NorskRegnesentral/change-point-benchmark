"""Null-case benchmarks: zero change-points, various distributions.

Run with::

    uv run pytest benchmarks/bench_null_case.py --benchmark-only -v

The ``benchmark.pedantic`` API is used throughout so that *only* the detector
fit/predict call is timed.  Dataset generation and detector instantiation happen
in the ``setup`` callable and are excluded from timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import ruptures as rpt
from skchange.change_detectors import PELT as SkchangePELT
from skchange.change_detectors import MovingWindow, SeededBinarySegmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dataframe(array: np.ndarray) -> pd.DataFrame:
    """Convert a numpy array to a pandas DataFrame expected by skchange."""
    return pd.DataFrame(array)


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
# skchange benchmarks
# ---------------------------------------------------------------------------


class TestSkchangeNull:
    """Benchmark skchange change-point detectors on null-case data."""

    def test_pelt(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange PELT."""
        df = _to_dataframe(null_dataset)

        def setup():
            det = SkchangePELT()
            return (det, df), {}

        def run(det, data):
            return det.fit_predict(data)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_moving_window(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange MovingWindow."""
        df = _to_dataframe(null_dataset)

        def setup():
            det = MovingWindow()
            return (det, df), {}

        def run(det, data):
            return det.fit_predict(data)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)

    def test_seeded_binseg(
        self,
        benchmark: pytest.fixture,
        null_dataset: np.ndarray,
    ) -> None:
        """skchange Seeded Binary Segmentation."""
        df = _to_dataframe(null_dataset)

        def setup():
            det = SeededBinarySegmentation()
            return (det, df), {}

        def run(det, data):
            return det.fit_predict(data)

        benchmark.pedantic(run, setup=setup, iterations=1, rounds=5)
