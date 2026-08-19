#!/usr/bin/env python
"""Run the Poisson cost benchmark.

Compares skchange and ruptures implementations of PELT, moving window, and
seeded binary segmentation with a Poisson cost. Ruptures has no built-in
Poisson model, so its side uses a custom ``BaseCost``. Run with::

    uv run python scripts/paper_benchmarks/run_poisson_benchmark.py

Results are written to a versioned file. Existing cases are
skipped unless :data:`OVERRIDE_RESULTS` is enabled.
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

PAIRS: list[Pair] = [
    Pair.PELT_POISSON,
    Pair.MOVING_WINDOW_POISSON,
    Pair.BINSEG_POISSON,
]
DIMENSIONS: list[int] = [1]
N_SAMPLES: list[int] = [
    100,
    250,
    500,
    750,
    1500,
    2500,
    # 5000,
    # int(1.0e4),
]
MIN_SEGMENT_LENGTH: int = 1
INCLUDE_FIT: bool = True
DISTRIBUTIONS: list[str] = ["normal"]

# Each (threshold, n_runs) entry applies when n_samples <= threshold.
RUNS_REGIME: list[tuple[int, int]] = [(1000, 10)]
RUNS_DEFAULT: int = 5

OVERRIDE_RESULTS: bool = False
RUN_STARTED_ON = date.today()
SKCHANGE_VERSION = version("skchange")
RUPTURES_VERSION = version("ruptures")
OUTPUT_PATH = prepare_results_path(
    f"poisson-benchmark_{RUN_STARTED_ON.isoformat()}_"
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
        case.penalty,
    )


def _load_existing_keys(path: Path) -> set[tuple]:
    """Load completed case keys from an existing result file."""
    if not path.exists():
        return set()
    if "penalty" not in pl.read_parquet_schema(path):
        return set()
    return set(pl.read_parquet(path, columns=_CASE_KEY_COLS).iter_rows())


def _collect_cases() -> list[BenchmarkCase]:
    """Create the configured Poisson benchmark cases."""
    return collect_cases(
        pairs=PAIRS,
        n_samples_list=N_SAMPLES,
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

    output_frame = existing_frame
    completed = 0
    skipped = 0
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        n_runs = _n_runs_for(case.n_samples)
        if _case_key(case, n_runs) in existing_keys:
            skipped += 1
            continue

        print(
            f"  ({index}/{len(cases)}) [{case.package}] {case.cpd_algorithm} "
            f"(n={case.n_samples}, p={case.data_dimension}, runs={n_runs}) ...",
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
            f"ski_jump={result.ski_jump_mean:.4f}s min={result.min:.4f}s "
            f"changes={result.n_detected_changepoints}"
        )
        result_frame = pl.DataFrame([result.as_dict()])
        output_frame = (
            pl.concat([output_frame, result_frame], how="diagonal_relaxed")
            if output_frame is not None
            else result_frame
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        output_frame.write_parquet(OUTPUT_PATH)
        completed += 1

    if skipped:
        print(f"  Skipped {skipped} already-completed case(s).")
    if completed == 0 and output_frame is None:
        print("No results were produced.")
        return
    elapsed = time.perf_counter() - started
    print(f"Finished in {elapsed:.1f}s ({completed} new result(s)).")
    print(f"Results written to {OUTPUT_PATH}")
    print_penalty_summary(output_frame)


def main() -> None:
    warnings.filterwarnings("ignore")
    cases = _collect_cases()
    print("=" * 60)
    print("Poisson Cost Null Benchmark")
    print("=" * 60)
    print(f"Pairs:      {[pair.value for pair in PAIRS]}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"N samples:  {N_SAMPLES}")
    print(f"Cases:      {len(cases)}")
    print(f"Runs:       {RUNS_REGIME} (default={RUNS_DEFAULT})")
    print(f"Output:     {OUTPUT_PATH}")
    print()
    _run_cases(cases)


if __name__ == "__main__":
    main()
