#!/usr/bin/env python
"""Full benchmark orchestration script.

Configure benchmark jobs as plain Python data structures below, then run::

    uv run python scripts/run_full_benchmark.py

Each :class:`Job` specifies which comparison pairs to run, with which
dimensions, minimum segment length, and fit modes.  The :data:`RUNS_REGIME`
table controls how many timed repetitions to use depending on ``n_samples``
(more runs for smaller data, fewer for large).

Results are written to ``results/<job.output_name>.parquet`` (one file per
job), using the same format as ``uv run bench``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from change_bench.benchmarks.comparison_pairs._common import BenchmarkCase
from change_bench.benchmarks.registry import Pair, collect_cases
from change_bench.runner import run_benchmark


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Job:
    """A group of benchmark cases to run with shared settings.

    Parameters
    ----------
    pairs:
        Comparison-pair names (keys from ``BENCHMARK_PAIRS``).
    dimensions:
        Data dimensionalities to benchmark.
    output_name:
        Stem for the output parquet file (written to ``results/<name>.parquet``).
    min_segment_length:
        Minimum segment length (``min_size`` in ruptures).
    include_fit:
        Which fit modes to run. ``[True]`` = fit+predict, ``[False]`` =
        predict only, ``[True, False]`` = both.
    n_samples:
        List of sample sizes to benchmark.
    distributions:
        Null-case distributions.  ``None`` uses the default.
    """

    pairs: list[Pair]
    dimensions: list[int]
    output_name: str
    n_samples: list[int]
    min_segment_length: int = 1
    include_fit: list[bool] = field(default_factory=lambda: [True])
    distributions: list[str] | None = None


# ---------------------------------------------------------------------------
# Runs regime: vary n_runs by n_samples.
# Each (threshold, n_runs) means: if n_samples <= threshold, use n_runs.
# Evaluated in order; first match wins.  RUNS_DEFAULT is the fallback.
# ---------------------------------------------------------------------------
RUNS_REGIME: list[tuple[int, int]] = [
    (250, 25),
    (500, 20),
    (1000, 15),
    (2500, 10),
]
RUNS_DEFAULT: int = 5

# ---------------------------------------------------------------------------
# Override toggle: when False, skip cases already present in existing parquet.
# ---------------------------------------------------------------------------
OVERRIDE_RESULTS: bool = False
COMMON_N_SAMPLES: list[int] = [100, 250, 500, 750, 1000]

# ---------------------------------------------------------------------------
# Jobs to run
# ---------------------------------------------------------------------------
JOBS: list[Job] = [
    Job(
        pairs=[
            Pair.PELT_L2,
            Pair.MOVING_WINDOW_L2,
            Pair.MOVING_WINDOW_L1,
            Pair.BINSEG_L2_CUSUM,
        ],
        dimensions=[1, 2, 5],
        min_segment_length=1,
        output_name="various_mean_change",
        n_samples=COMMON_N_SAMPLES,
    ),
    Job(
        pairs=[Pair.PELT_1D_GAUSSIAN],
        dimensions=[1],
        min_segment_length=2,  # Fitting variance, so need at least 2 samples per segment
        output_name="1d_gaussian",
        n_samples=COMMON_N_SAMPLES,
    ),
    Job(
        pairs=[
            Pair.MOVING_WINDOW_RANK,
            Pair.PELT_RANK,
            Pair.BINSEG_RANK,
            Pair.BINSEG_MV_GAUSSIAN,
            Pair.PELT_MV_GAUSSIAN,
            Pair.MOVING_WINDOW_MV_GAUSSIAN,
        ],
        dimensions=[2, 5],
        min_segment_length=6,  # must be > max(dimensions) for rank costs
        output_name="multivariate_rank_and_mv_gaussian",
        n_samples=COMMON_N_SAMPLES,
    ),
]


# ============================================================================
# Helpers
# ============================================================================
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Columns that uniquely identify a benchmark case in the parquet output.
_CASE_KEY_COLS = [
    "name",
    "package",
    "n_samples",
    "data_dimension",
    "include_fit",
    "min_segment_length",
    "n_runs",
]


def _n_runs_for(n_samples: int) -> int:
    """Look up the number of runs for a given n_samples."""
    for threshold, n_runs in RUNS_REGIME:
        if n_samples <= threshold:
            return n_runs
    return RUNS_DEFAULT


def _case_key(case: BenchmarkCase, n_runs: int) -> tuple:
    """Return a hashable key that uniquely identifies a benchmark case."""
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
    """Load existing result keys from a parquet file (if it exists)."""
    if not path.exists():
        return set()
    df = pl.read_parquet(path, columns=_CASE_KEY_COLS)
    return set(df.iter_rows())


def _run_job(job: Job, job_idx: int, total_jobs: int) -> None:
    """Collect cases for a single Job and run them."""
    output_path = RESULTS_DIR / f"{job.output_name}.parquet"

    cases: list[BenchmarkCase] = []
    for include_fit in job.include_fit:
        cases.extend(
            collect_cases(
                pairs=job.pairs,
                n_samples_list=job.n_samples,
                include_fit=include_fit,
                min_segment_length=job.min_segment_length,
                dimensions=job.dimensions,
                distributions=job.distributions,
            )
        )

    if not cases:
        print(f"  No cases for job {job.output_name!r}, skipping.")
        return

    print(
        f"--- [{job_idx}/{total_jobs}] {job.output_name} "
        f"({len(cases)} cases, pairs={[p.value for p in job.pairs]}) ---"
    )

    # Load existing results when not overriding.
    existing_keys: set[tuple] = set()
    existing_df: pl.DataFrame | None = None
    if not OVERRIDE_RESULTS:
        existing_keys = _load_existing_keys(output_path)
        if existing_keys:
            existing_df = pl.read_parquet(output_path)

    results = []
    skipped = 0
    t0 = time.perf_counter()

    for i, case in enumerate(cases, 1):
        n_runs = _n_runs_for(case.n_samples)
        key = _case_key(case, n_runs)
        if key in existing_keys:
            skipped += 1
            continue
        label = f"[{case.package}] {case.name}"
        dims = f"n={case.n_samples}, p={case.data_dimension}"
        fit_label = "fit+predict" if case.include_fit else "predict"
        print(
            f"  ({i}/{len(cases)}) {label} ({dims}, {fit_label}, runs={n_runs}) ...",
            end=" ",
            flush=True,
        )

        res = run_benchmark(
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
        )

        print(
            f"ski_jump={res.ski_jump_mean:.4f}s ± {res.ski_jump_std:.4f}s  "
            f"min={res.min:.4f}s"
        )
        results.append(res.as_dict())

    elapsed = time.perf_counter() - t0

    if skipped:
        print(f"  Skipped {skipped} already-completed case(s).")

    if not results and existing_df is None:
        print(f"  No new results for job {job.output_name!r}.\n")
        return

    # Merge new results with existing ones.
    if results and existing_df is not None:
        df = pl.concat([existing_df, pl.DataFrame(results)])
    elif existing_df is not None:
        df = existing_df
    else:
        df = pl.DataFrame(results)

    print(f"  Job finished in {elapsed:.1f}s ({len(results)} new result(s)).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"  Results written to {output_path}\n")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("Full Benchmark Run")
    print("=" * 60)
    print(f"Jobs:        {len(JOBS)}")
    print(f"Output dir:  {RESULTS_DIR}")
    print(f"Override:    {OVERRIDE_RESULTS}")
    print(f"Runs regime: {RUNS_REGIME} (default={RUNS_DEFAULT})")
    print()

    t_global = time.perf_counter()

    for idx, job in enumerate(JOBS, 1):
        _run_job(job, idx, len(JOBS))

    elapsed = time.perf_counter() - t_global
    print(f"All jobs finished in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
