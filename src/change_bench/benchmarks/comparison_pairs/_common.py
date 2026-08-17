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
    problem_name: str
    n_samples: int
    n_changepoints: int
    data_dimension: int
    include_fit: bool
    min_segment_length: int
    penalty: float | None
    prepare: Callable[[], np.ndarray]
    setup: Callable[[np.ndarray], tuple[tuple, dict]]
    func: Callable


#: Penalty used for all PELT-based pairs (same for skchange and ruptures).
PELT_PENALTY: float = 100.0

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
    penalty : float | None
        Penalty passed to ruptures' .predict(pen=...).
    sk_name_prefix : str | None
        Prefix for skchange benchmark case names.
    rpt_name_prefix : str | None
        Prefix for ruptures benchmark case names.
    make_sk_detector : Callable[[int], Any] | None
        Factory: (effective_msl) -> configured skchange detector instance.
    make_rpt_algo : Callable[[int], Any] | None
        Factory: (effective_msl) -> configured ruptures algo instance.
    effective_msl : Callable[[int, int], int]
        (min_segment_length, n_columns) -> actual msl to use.
        Override for multivariate costs that require msl >= n_columns + 1.
    prepare_transform : Callable[[np.ndarray], np.ndarray] | None
        Optional post-processing of generated data (e.g. Poisson abs+offset).
    """

    pair_name: str
    penalty: float | None = None
    sk_name_prefix: str | None = None
    rpt_name_prefix: str | None = None
    make_sk_detector: Callable[[int], Any] | None = None
    make_rpt_algo: Callable[[int], Any] | None = None
    effective_msl: Callable[[int, int], int] = field(default=_default_effective_msl)
    prepare_transform: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.make_sk_detector is None and self.make_rpt_algo is None:
            raise ValueError("PairConfig must define at least one benchmark side")
        if self.make_sk_detector is not None and self.sk_name_prefix is None:
            raise ValueError("sk_name_prefix is required for the skchange side")
        if self.make_rpt_algo is not None:
            if self.rpt_name_prefix is None:
                raise ValueError("rpt_name_prefix is required for the ruptures side")
            if self.penalty is None:
                raise ValueError("penalty is required for the ruptures side")


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

        if config.make_sk_detector is not None:
            make_sk_detector = config.make_sk_detector

            def sk_setup(data: np.ndarray, msl=eff_msl, fit=include_fit):
                detector = make_sk_detector(msl)
                if not fit:
                    detector.fit(data)
                return (detector, data), {}

            def sk_func(detector, X, fit=include_fit):
                if fit:
                    detector.fit(X)
                return detector.predict(X)

            cases.append(
                BenchmarkCase(
                    package="skchange",
                    cpd_algorithm=config.pair_name,
                    name=f"{config.sk_name_prefix}/{problem.name}",
                    problem_name=problem.name,
                    n_samples=cfg.n_samples,
                    n_changepoints=len(problem.true_changepoints),
                    data_dimension=cfg.n_columns,
                    include_fit=include_fit,
                    min_segment_length=eff_msl,
                    penalty=config.penalty,
                    prepare=prepare,
                    setup=sk_setup,
                    func=sk_func,
                )
            )

        if config.make_rpt_algo is not None:
            make_rpt_algo = config.make_rpt_algo
            penalty = config.penalty

            def rpt_setup(data: np.ndarray, msl=eff_msl, fit=include_fit):
                algorithm = make_rpt_algo(msl)
                if not fit:
                    algorithm.fit(data)
                return (algorithm, data), {}

            def rpt_func(algorithm, X, fit=include_fit, pen=penalty):
                if fit:
                    algorithm.fit(X)
                return algorithm.predict(pen=pen)

            cases.append(
                BenchmarkCase(
                    package="ruptures",
                    cpd_algorithm=config.pair_name,
                    name=f"{config.rpt_name_prefix}/{problem.name}",
                    problem_name=problem.name,
                    n_samples=cfg.n_samples,
                    n_changepoints=len(problem.true_changepoints),
                    data_dimension=cfg.n_columns,
                    include_fit=include_fit,
                    min_segment_length=eff_msl,
                    penalty=config.penalty,
                    prepare=prepare,
                    setup=rpt_setup,
                    func=rpt_func,
                )
            )

    return cases
