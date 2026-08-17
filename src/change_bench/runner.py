"""Benchmark runner using timeit for timing, with statistics collection.

This module replaces the pytest-benchmark infrastructure with a standalone
runner that collects mean, std, median, and min timing results and stores
them in a Parquet file.
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Timing results for a single benchmark case."""

    package: str
    cpd_algorithm: str
    name: str
    n_samples: int
    n_changepoints: int
    data_dimension: int
    include_fit: bool
    min_segment_length: int
    n_runs: int
    penalty: float | None
    n_detected_changepoints: int = 0
    times: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def min(self) -> float:
        return min(self.times)

    @property
    def ski_jump_mean(self) -> float:
        """Mean after removing fastest and slowest run."""
        if len(self.times) <= 2:
            return self.mean
        trimmed = sorted(self.times)[1:-1]
        return statistics.mean(trimmed)

    @property
    def ski_jump_std(self) -> float:
        """Std after removing fastest and slowest run."""
        if len(self.times) <= 2:
            return self.std
        trimmed = sorted(self.times)[1:-1]
        return statistics.stdev(trimmed) if len(trimmed) > 1 else 0.0

    def as_dict(self) -> dict:
        return {
            "package": self.package,
            "cpd_algorithm": self.cpd_algorithm,
            "name": self.name,
            "n_samples": self.n_samples,
            "n_changepoints": self.n_changepoints,
            "data_dimension": self.data_dimension,
            "include_fit": self.include_fit,
            "min_segment_length": self.min_segment_length,
            "n_runs": self.n_runs,
            "penalty": self.penalty,
            "n_detected_changepoints": self.n_detected_changepoints,
            "mean_s": self.mean,
            "std_s": self.std,
            "median_s": self.median,
            "min_s": self.min,
            "ski_jump_mean_s": self.ski_jump_mean,
            "ski_jump_std_s": self.ski_jump_std,
        }


def run_benchmark(
    *,
    package: str,
    cpd_algorithm: str,
    name: str,
    n_samples: int,
    n_changepoints: int,
    data_dimension: int,
    include_fit: bool,
    min_segment_length: int,
    prepare: Callable,
    setup: Callable,
    func: Callable,
    n_runs: int,
    penalty: float | None,
) -> BenchmarkResult:
    """Run a single benchmark case *n_runs* times and return statistics.

    Uses a two-phase protocol for memory efficiency:
    1. ``prepare()`` generates the data array (called once).
    2. ``setup(data)`` creates a fresh detector per run.
    3. ``func(*args, **kwargs)`` is timed.

    After timing, one fresh detector is run once without timing to record the
    number of detected changepoints. The terminal endpoint returned by ruptures
    is not counted as a changepoint.

    Parameters
    ----------
    package:
        Library package (e.g. ``"ruptures"`` or ``"skchange"``).
    cpd_algorithm:
        Name of the comparison algorithm pair.
    name:
        Human-readable benchmark name (e.g. distribution label).
    n_samples:
        Number of samples in the dataset.
    n_changepoints:
        Number of true change points in the dataset.
    data_dimension:
        Dimensionality of the time series (number of columns).
    include_fit:
        Whether the timed operation includes fitting.
    min_segment_length:
        Minimum segment length used by the detector.
    prepare:
        Callable that generates the data array.  Called once before the
        timing loop.
    setup:
        Callable taking the data array and returning ``(args, kwargs)`` to
        pass to *func*.  Called once per run for a fresh detector.
    func:
        The function to time.  Called as ``func(*args, **kwargs)``.
    n_runs:
        Number of timed repetitions.
    penalty:
        Penalty configured for the benchmark comparison.
    """
    result = BenchmarkResult(
        package=package,
        cpd_algorithm=cpd_algorithm,
        name=name,
        n_samples=n_samples,
        n_changepoints=n_changepoints,
        data_dimension=data_dimension,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
        n_runs=n_runs,
        penalty=penalty,
    )

    data = prepare()

    for _ in range(n_runs):
        args, kwargs = setup(data)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        t0 = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if gc_was_enabled:
            gc.enable()
        result.times.append(elapsed)

    args, kwargs = setup(data)
    prediction = func(*args, **kwargs)
    changepoints = list(prediction)
    if package == "ruptures":
        changepoints = [
            changepoint for changepoint in changepoints if changepoint != n_samples
        ]
    result.n_detected_changepoints = len(changepoints)

    return result


def print_penalty_summary(frame) -> None:
    """Print penalty and detected-changepoint counts per benchmark case.

    Benchmarks are run on null data with a penalty high enough that no
    spurious change points should be detected; this summary lets users
    confirm that ``n_detected_changepoints`` is zero everywhere.
    """
    import polars as pl

    summary = frame.select(
        "cpd_algorithm",
        "package",
        "n_samples",
        "data_dimension",
        "penalty",
        "n_detected_changepoints",
    ).sort("cpd_algorithm", "package", "data_dimension", "n_samples")
    n_spurious = summary.filter(pl.col("n_detected_changepoints") > 0).height
    print("\nPenalty summary (expected n_detected_changepoints = 0):")
    with pl.Config(tbl_rows=-1):
        print(summary)
    if n_spurious:
        print(
            f"WARNING: {n_spurious} case(s) detected spurious change points; "
            "consider raising the penalty."
        )
    else:
        print("All cases detected 0 change points at the configured penalties.")
