#!/usr/bin/env python
"""Run the fixed-sample multivariate dimension benchmark.

Compares valid combinations of PELT, moving window, and seeded binary
segmentation with L2, ESAC, multivariate Gaussian, and rank scores/costs.
ESAC cases are skchange-only. Run with::

    uv run python scripts/paper_benchmarks/run_multivariate_dimension_benchmark.py

Results are written to a file named like
``results/multivariate-change-detection-benchmark_2026-08-10_skchange-0.9.0_ruptures-1.1.10.parquet``.
Existing cases are skipped unless :data:`OVERRIDE_RESULTS` is enabled.
"""

from __future__ import annotations

import time
import warnings
from datetime import date
from importlib.metadata import version
from pathlib import Path

import polars as pl

from change_bench.benchmarks.comparison_pairs._common import BenchmarkCase
from change_bench.benchmarks.registry import Pair, collect_cases
from change_bench.paths import prepare_results_path
from change_bench.runner import print_penalty_summary, run_benchmark

# ============================================================================
# Configuration
# ============================================================================
PAIRS: list[Pair] = [
    Pair.MOVING_WINDOW_L2,
    Pair.MOVING_WINDOW_ESAC,
    Pair.MOVING_WINDOW_RANK,
    Pair.MOVING_WINDOW_MV_GAUSSIAN,
    Pair.BINSEG_L2_CUSUM,
    Pair.BINSEG_ESAC,
    Pair.BINSEG_RANK,
    Pair.BINSEG_MV_GAUSSIAN,
    Pair.PELT_L2,
    Pair.PELT_RANK,
    Pair.PELT_MV_GAUSSIAN,
]
N_SAMPLES: int = 2000
DIMENSIONS: list[int] = [5, 10, 25, 50, 75, 100]
MIN_SEGMENT_LENGTH: int = 1
INCLUDE_FIT: bool = True
DISTRIBUTIONS: list[str] = ["normal"]
N_RUNS: int = 10

OVERRIDE_RESULTS: bool = False
RUN_STARTED_ON = date.today()
SKCHANGE_VERSION = version("skchange")
RUPTURES_VERSION = version("ruptures")
OUTPUT_PATH = prepare_results_path(
    f"multivariate-change-detection-benchmark_{RUN_STARTED_ON.isoformat()}_"
    f"skchange-{SKCHANGE_VERSION}_ruptures-{RUPTURES_VERSION}.parquet",
    Path(__file__),
    subdir=Path("paper"),
)

_CASE_KEY_COLS = [
    "name",
    "package",
    "n_samples",
    "data_dimension",
    "include_fit",
    "min_segment_length",
    "n_runs",
    "penalty",
]


# ============================================================================
# Helpers
# ============================================================================
def _case_key(case: BenchmarkCase) -> tuple:
    """Return the columns that uniquely identify a completed case."""
    return (
        case.name,
        case.package,
        case.n_samples,
        case.data_dimension,
        case.include_fit,
        case.min_segment_length,
        N_RUNS,
        case.penalty,
    )


def _load_existing_keys(path: Path) -> set[tuple]:
    """Load completed case keys from an existing result file."""
    if not path.exists():
        return set()
    if "penalty" not in pl.read_parquet_schema(path):
        return set()
    frame = pl.read_parquet(path, columns=_CASE_KEY_COLS)
    return set(frame.iter_rows())


def _collect_cases() -> list[BenchmarkCase]:
    """Create the configured multivariate dimension benchmark cases."""
    return collect_cases(
        pairs=PAIRS,
        n_samples_list=[N_SAMPLES],
        include_fit=INCLUDE_FIT,
        min_segment_length=MIN_SEGMENT_LENGTH,
        dimensions=DIMENSIONS,
        distributions=DISTRIBUTIONS,
    )


def _run_cases(cases: list[BenchmarkCase]) -> None:
    """Run incomplete cases and merge their results with existing output."""
    existing_keys: set[tuple] = set()
    existing_frame: pl.DataFrame | None = None
    if not OVERRIDE_RESULTS and OUTPUT_PATH.exists():
        existing_frame = pl.read_parquet(OUTPUT_PATH)
        if "penalty" in existing_frame.columns:
            penalties = list({case.penalty for case in cases})
            existing_frame = existing_frame.filter(pl.col("penalty").is_in(penalties))
            existing_keys = set(existing_frame.select(_CASE_KEY_COLS).iter_rows())
        else:
            existing_frame = None

    results: list[dict] = []
    skipped = 0
    started = time.perf_counter()

    for index, case in enumerate(cases, 1):
        if _case_key(case) in existing_keys:
            skipped += 1
            continue

        fit_label = "fit+predict" if case.include_fit else "predict"
        print(
            f"  ({index}/{len(cases)}) [{case.package}] {case.cpd_algorithm} "
            f"(n={case.n_samples}, p={case.data_dimension}, {fit_label}, "
            f"runs={N_RUNS}) ...",
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
            n_runs=N_RUNS,
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
    print_penalty_summary(output_frame)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    warnings.filterwarnings("ignore")
    cases = _collect_cases()

    print("=" * 60)
    print("Multivariate Change Detection Benchmark")
    print("=" * 60)
    print(f"Pairs:       {[pair.value for pair in PAIRS]}")
    print(f"N samples:   {N_SAMPLES}")
    print(f"Dimensions:  {DIMENSIONS}")
    print(f"Cases:       {len(cases)}")
    print(f"Runs:        {N_RUNS}")
    print(f"Output:      {OUTPUT_PATH}")
    print(f"Override:    {OVERRIDE_RESULTS}")
    print()

    _run_cases(cases)


if __name__ == "__main__":
    main()
