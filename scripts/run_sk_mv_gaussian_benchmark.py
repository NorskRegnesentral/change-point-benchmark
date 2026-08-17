#!/usr/bin/env python
"""Run the skchange-only multivariate Gaussian cost benchmark.

Benchmarks skchange's ``MultivariateGaussianCost`` across the search
algorithms that support it (PELT, moving window, seeded binary segmentation)
for increasing sample sizes and dimensions, with and without the cumulative
covariance cache (``store_cov``). No ruptures comparison; intended for
skchange development. Run with::

    uv run python scripts/run_sk_mv_gaussian_benchmark.py

Results are written to
``results/skchange/sk-mv-gaussian-benchmark.parquet`` (kept out of the
top-level ``results/`` glob used by the ruptures-comparison analyses).
The ``store_cov`` setting is encoded in the ``name`` column as
``[store_cov=True|False]``. Existing cases are skipped unless
:data:`OVERRIDE_RESULTS` is enabled.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import polars as pl
from skchange.detectors import PELT as SkchangePELT
from skchange.detectors import MovingWindow, SeededBinarySegmentation
from skchange.interval_scorers import (
    MultivariateGaussianCost,
    MultivariateGaussianScore,
)

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import make_null_problems
from change_bench.runner import run_benchmark

# ============================================================================
# Configuration
# ============================================================================
STORE_COV_VALUES: list[bool] = [True, False]
DIMENSIONS: list[int] = [5, 10, 25]
N_SAMPLES: list[int] = [100, 250, 500, 750, 1000, 2500]
MIN_SEGMENT_LENGTH: int = 1
INCLUDE_FIT: bool = True
DISTRIBUTIONS: list[str] = ["normal"]

MW_MV_GAUSSIAN_PENALTY: float = 4.0
BINSEG_MV_GAUSSIAN_PENALTY: float = 10.0

# Each (threshold, n_runs) entry applies when n_samples <= threshold.
RUNS_REGIME: list[tuple[int, int]] = [
    (250, 25),
    (500, 20),
    (1000, 15),
    (2500, 10),
]
RUNS_DEFAULT: int = 5

OVERRIDE_RESULTS: bool = False
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "skchange"
OUTPUT_PATH = RESULTS_DIR / "sk-mv-gaussian-benchmark.parquet"

_CASE_KEY_COLS = [
    "name",
    "package",
    "n_samples",
    "data_dimension",
    "include_fit",
    "min_segment_length",
    "n_runs",
]


# ============================================================================
# Helpers
# ============================================================================
def _n_runs_for(n_samples: int) -> int:
    """Look up the number of timed repetitions for a sample size."""
    for threshold, n_runs in RUNS_REGIME:
        if n_samples <= threshold:
            return n_runs
    return RUNS_DEFAULT


def _case_key(case: BenchmarkCase, n_runs: int) -> tuple:
    """Return the columns that uniquely identify a completed case."""
    return (
        case.name,
        case.package,
        case.n_samples,
        case.data_dimension,
        case.include_fit,
        case.min_segment_length,
        n_runs,
    )


def _load_existing_keys(path: Path) -> set[tuple]:
    """Load completed case keys from an existing result file."""
    if not path.exists():
        return set()
    frame = pl.read_parquet(path, columns=_CASE_KEY_COLS)
    return set(frame.iter_rows())


def _make_configs(store_cov: bool) -> list[PairConfig]:
    """Create skchange-only pair configs for a given store_cov setting."""
    suffix = f"[store_cov={store_cov}]"
    effective_msl = lambda msl, n_cols: max(msl, n_cols + 1)  # noqa: E731
    return [
        PairConfig(
            pair_name="pelt_mv_gaussian",
            sk_name_prefix=f"skchange_pelt_mv_gaussian{suffix}",
            make_sk_detector=lambda msl, sc=store_cov: SkchangePELT(
                cost=MultivariateGaussianCost(store_cov=sc),
                penalty=PELT_PENALTY,
                min_segment_length=msl,
            ),
            effective_msl=effective_msl,
        ),
        PairConfig(
            pair_name="moving_window_mv_gaussian",
            sk_name_prefix=f"skchange_moving_window_mv_gaussian{suffix}",
            make_sk_detector=lambda msl, sc=store_cov: MovingWindow(
                change_score=MultivariateGaussianScore(
                    apply_bartlett_correction=False, store_cov=sc
                ),
                penalty=MW_MV_GAUSSIAN_PENALTY,
                bandwidth=max(MW_BANDWIDTH, msl),
            ),
            effective_msl=effective_msl,
        ),
        PairConfig(
            pair_name="binseg_mv_gaussian",
            sk_name_prefix=f"skchange_seeded_binseg_mv_gaussian{suffix}",
            make_sk_detector=lambda msl, sc=store_cov: SeededBinarySegmentation(
                change_score=MultivariateGaussianScore(
                    apply_bartlett_correction=False, store_cov=sc
                ),
                penalty=BINSEG_MV_GAUSSIAN_PENALTY,
                max_interval_length=max(200, 2 * msl),
            ),
            effective_msl=effective_msl,
        ),
    ]


def _collect_cases() -> list[BenchmarkCase]:
    """Create the configured skchange-only MV Gaussian benchmark cases."""
    problems = make_null_problems(
        n_samples_list=N_SAMPLES,
        distributions=DISTRIBUTIONS,
        scale=1.0,
        n_columns_list=DIMENSIONS,
    )
    cases: list[BenchmarkCase] = []
    for store_cov in STORE_COV_VALUES:
        for config in _make_configs(store_cov):
            cases.extend(
                build_pair_cases(
                    problems,
                    config,
                    include_fit=INCLUDE_FIT,
                    min_segment_length=MIN_SEGMENT_LENGTH,
                )
            )
    return cases


def _run_cases(cases: list[BenchmarkCase]) -> None:
    """Run incomplete cases and merge their results with existing output."""
    existing_keys: set[tuple] = set()
    existing_frame: pl.DataFrame | None = None
    if not OVERRIDE_RESULTS:
        existing_keys = _load_existing_keys(OUTPUT_PATH)
        if existing_keys:
            existing_frame = pl.read_parquet(OUTPUT_PATH)

    results: list[dict] = []
    skipped = 0
    started = time.perf_counter()

    for index, case in enumerate(cases, 1):
        n_runs = _n_runs_for(case.n_samples)
        if _case_key(case, n_runs) in existing_keys:
            skipped += 1
            continue

        fit_label = "fit+predict" if case.include_fit else "predict"
        print(
            f"  ({index}/{len(cases)}) [{case.package}] {case.name} "
            f"(n={case.n_samples}, p={case.data_dimension}, {fit_label}, "
            f"runs={n_runs}) ...",
            end=" ",
            flush=True,
        )

        result = run_benchmark(
            package=case.package,
            cpd_algorithm=case.cpd_algorithm,
            name=case.name,
            n_samples=case.n_samples,
            n_changepoints=case.n_changepoints,
            data_dimension=case.data_dimension,
            include_fit=case.include_fit,
            min_segment_length=case.min_segment_length,
            prepare=case.prepare,
            setup=case.setup,
            func=case.func,
            n_runs=n_runs,
            penalty=case.penalty,
        )
        print(
            f"ski_jump={result.ski_jump_mean:.4f}s "
            f"+- {result.ski_jump_std:.4f}s  min={result.min:.4f}s "
            f"changes={result.n_detected_changepoints}"
        )
        results.append(result.as_dict())

    elapsed = time.perf_counter() - started
    if skipped:
        print(f"  Skipped {skipped} already-completed case(s).")

    if not results and existing_frame is None:
        print("No results were produced.")
        return

    if results and existing_frame is not None:
        output_frame = pl.concat(
            [existing_frame, pl.DataFrame(results)], how="diagonal_relaxed"
        )
    elif existing_frame is not None:
        output_frame = existing_frame
    else:
        output_frame = pl.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_frame.write_parquet(OUTPUT_PATH)
    print(f"Finished in {elapsed:.1f}s ({len(results)} new result(s)).")
    print(f"Results written to {OUTPUT_PATH}")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    warnings.filterwarnings("ignore")
    cases = _collect_cases()

    print("=" * 60)
    print("skchange MV Gaussian Cost Benchmark")
    print("=" * 60)
    print("Algorithms:  ['pelt', 'moving_window', 'seeded_binseg'] (skchange)")
    print(f"store_cov:   {STORE_COV_VALUES}")
    print(f"N samples:   {N_SAMPLES}")
    print(f"Dimensions:  {DIMENSIONS}")
    print(f"Cases:       {len(cases)}")
    print(f"Output:      {OUTPUT_PATH}")
    print(f"Override:    {OVERRIDE_RESULTS}")
    print(f"Runs regime: {RUNS_REGIME} (default={RUNS_DEFAULT})")
    print()

    _run_cases(cases)


if __name__ == "__main__":
    main()
