#!/usr/bin/env python
"""Run the change-in-mean L1 benchmark.

Compares skchange and ruptures implementations of PELT, moving window, and
binary segmentation on the same L1 change-in-mean problem. Run with::

    uv run python scripts/paper_benchmarks/run_change_in_mean_l1_benchmark.py

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
    Pair.PELT_L1,
    Pair.MOVING_WINDOW_L1,
    Pair.BINSEG_L1,
]
DIMENSIONS: list[int] = [1]
N_SAMPLES: list[int] = [
    100,
    250,
    500,
    750,
    1500,
    2500,
    5000,
    ### Take a really long time per sample. Skip.
    # int(1.0e4),
    # int(2.5e4),
    # int(5.0e4),
    # int(1.0e5),
]
MIN_SEGMENT_LENGTH: int = 1
INCLUDE_FIT: bool = True
DISTRIBUTIONS: list[str] = ["normal"]

RUNS_REGIME: list[tuple[int, int]] = [
    # More than 1000 samples: 5 runs
    (1000, 10),
]
RUNS_DEFAULT: int = 5

OVERRIDE_RESULTS: bool = False
RUN_STARTED_ON = date.today()
SKCHANGE_VERSION = version("skchange")
RUPTURES_VERSION = version("ruptures")
OUTPUT_PATH = prepare_results_path(
    f"change-in-mean-l1-benchmark_{RUN_STARTED_ON.isoformat()}_"
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
    frame = pl.read_parquet(path, columns=_CASE_KEY_COLS)
    return set(frame.iter_rows())


def _collect_cases() -> list[BenchmarkCase]:
    """Create all configured L1 change-in-mean benchmark cases."""
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
            f"  ({index}/{len(cases)}) [{case.package}] {case.cpd_algorithm} "
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

    output_frame.write_parquet(OUTPUT_PATH)
    print(f"Finished in {elapsed:.1f}s ({len(results)} new result(s)).")
    print(f"Results written to {OUTPUT_PATH}")
    print_penalty_summary(output_frame)


def main() -> None:
    warnings.filterwarnings("ignore")
    cases = _collect_cases()

    print("=" * 60)
    print("Change-in-Mean L1 Benchmark")
    print("=" * 60)
    print(f"Pairs:       {[pair.value for pair in PAIRS]}")
    print(f"Dimensions:  {DIMENSIONS}")
    print(f"N samples:   {N_SAMPLES}")
    print(f"Cases:       {len(cases)}")
    print(f"Output:      {OUTPUT_PATH}")
    print(f"Override:    {OVERRIDE_RESULTS}")
    print(f"Runs regime: {RUNS_REGIME} (default={RUNS_DEFAULT})")
    print()

    _run_cases(cases)


if __name__ == "__main__":
    main()
