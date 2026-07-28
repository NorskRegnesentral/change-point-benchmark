"""Shared constants, helpers, and the BenchmarkCase dataclass.

All comparison-pair modules import from here to avoid circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from change_bench.problems.base import BenchmarkProblem

# ---------------------------------------------------------------------------
# Benchmark case definition
# ---------------------------------------------------------------------------

BENCHMARK_SEED: int = 42


@dataclass
class BenchmarkCase:
    """A single benchmark case ready to be run.

    Uses a two-phase setup for memory efficiency:
    - ``prepare()`` generates data just-in-time (called once before timing loop)
    - ``setup(data)`` creates a fresh detector per run
    - ``func`` is the timed operation
    """

    package: str
    cpd_algorithm: str
    name: str
    n_samples: int
    n_changepoints: int
    data_dimension: int
    include_fit: bool
    min_segment_length: int
    prepare: Callable[[], np.ndarray]
    setup: Callable[[np.ndarray], tuple[tuple, dict]]
    func: Callable


#: Penalty used for all PELT-based pairs (same for skchange and ruptures).
PELT_PENALTY: float = 10.0

#: Bandwidth for moving-window pairs.
MW_BANDWIDTH: int = 25


def _default_effective_msl(msl: int, n_columns: int) -> int:
    return msl


@dataclass
class PairConfig:
    """Configuration for a comparison pair (skchange vs ruptures).

    Parameters
    ----------
    pair_name : str
        Identifier for the pair, e.g. "pelt_l2".
    penalty : float
        Penalty passed to ruptures' .predict(pen=...).
    sk_name_prefix : str
        Prefix for skchange benchmark case names.
    rpt_name_prefix : str
        Prefix for ruptures benchmark case names.
    make_sk_detector : Callable[[int], Any]
        Factory: (effective_msl) -> configured skchange detector instance.
    make_rpt_algo : Callable[[int], Any]
        Factory: (effective_msl) -> configured ruptures algo instance.
    effective_msl : Callable[[int, int], int]
        (min_segment_length, n_columns) -> actual msl to use.
        Override for multivariate costs that require msl >= n_columns + 1.
    prepare_transform : Callable[[np.ndarray], np.ndarray] | None
        Optional post-processing of generated data (e.g. Poisson abs+offset).
    """

    pair_name: str
    penalty: float
    sk_name_prefix: str
    rpt_name_prefix: str
    make_sk_detector: Callable[[int], Any]
    make_rpt_algo: Callable[[int], Any]
    effective_msl: Callable[[int, int], int] = field(default=_default_effective_msl)
    prepare_transform: Callable[[np.ndarray], np.ndarray] | None = None


def make_prepare(problem: BenchmarkProblem, seed: int = BENCHMARK_SEED):
    """Create a prepare closure that generates data just-in-time."""

    def prepare() -> np.ndarray:
        return problem.generate(np.random.default_rng(seed))

    return prepare


def build_pair_cases(
    problems: list[BenchmarkProblem],
    config: PairConfig,
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Build skchange + ruptures BenchmarkCase pairs from a PairConfig.

    Parameters
    ----------
    problems : list[BenchmarkProblem]
        Problems to benchmark on.
    config : PairConfig
        Configuration describing the comparison pair.
    include_fit : bool
        Whether to include .fit() in the timed operation.
    min_segment_length : int
        Minimum segment length (may be overridden by config.effective_msl).
    """
    cases: list[BenchmarkCase] = []

    for problem in problems:
        cfg = problem.dataset_config
        eff_msl = config.effective_msl(min_segment_length, cfg.n_columns)

        # --- prepare ---
        base_prepare = make_prepare(problem)
        transform = config.prepare_transform

        def prepare(base=base_prepare, t=transform) -> np.ndarray:
            data = base()
            return t(data) if t is not None else data

        # --- skchange setup & func ---
        def make_sk_setup(msl=eff_msl, fit=include_fit):
            def setup(data: np.ndarray):
                det = config.make_sk_detector(msl)
                if not fit:
                    det.fit(data)
                return (det, data), {}

            return setup

        def sk_func(det, X, _fit=include_fit):
            if _fit:
                det.fit(X)
            return det.predict_changepoints(X)

        # --- ruptures setup & func ---
        def make_rpt_setup(msl=eff_msl, fit=include_fit):
            def setup(data: np.ndarray):
                algo = config.make_rpt_algo(msl)
                if not fit:
                    algo.fit(data)
                return (algo, data), {}

            return setup

        def rpt_func(algo, X, _fit=include_fit, _pen=config.penalty):
            if _fit:
                algo.fit(X)
            return algo.predict(pen=_pen)

        # --- cases ---
        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=config.pair_name,
                name=f"{config.sk_name_prefix}/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=eff_msl,
                prepare=prepare,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=config.pair_name,
                name=f"{config.rpt_name_prefix}/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=eff_msl,
                prepare=prepare,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases
