"""CLI entrypoint for running change-point benchmarks.

Usage::

    uv run bench --runs 10 --packages ruptures skchange --output results.parquet
    uv run bench --runs 5 --packages skchange --problem-set full
    uv run bench --list
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import polars as pl

from change_bench.benchmarks.comparison_pairs._common import BenchmarkCase
from change_bench.benchmarks.registry import (
    ALL_DISTRIBUTIONS,
    PAIR_CATEGORIES,
    Pair,
    collect_cases,
)
from change_bench.runner import run_benchmark


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench",
        description=(
            "Run change-point detection benchmarks and save results to Parquet."
        ),
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=5,
        help="Number of timed repetitions per benchmark case (default: 5).",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=None,
        help=(
            "Filter to specific library packages (ruptures, skchange). "
            "Omit to run both sides of each pair."
        ),
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            f"Comparison pairs to run. Available: {sorted(p.value for p in Pair)}. "
            "Omit to run all pairs."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help=(
            f"Filter pairs by category. Available: {sorted(PAIR_CATEGORIES)}. "
            "Omit to run all categories."
        ),
    )
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        required=True,
        help="Sample sizes to benchmark (e.g. --n-samples 100 250 500 750).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark_results.parquet"),
        help="Path for the output Parquet file (default: benchmark_results.parquet).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_cases",
        help="List available benchmark cases and exit.",
    )
    parser.add_argument(
        "--include-fit",
        choices=["yes", "no", "both"],
        default="yes",
        help=(
            "Whether to include fit() in the timed operation. "
            "'yes' (default): fit+predict. 'no': predict only. "
            "'both': run each case twice (once with fit, once without)."
        ),
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=1,
        help=(
            "Minimum segment length for the detector (default: 1). "
            "Maps to min_size in ruptures and min_segment_length in skchange."
        ),
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1],
        help=(
            "Data dimensionalities (number of columns) to benchmark "
            "(default: 1). Pairs that don't support multivariate data "
            "will only run with p=1 regardless of this setting."
        ),
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=None,
        help=(
            f"Null-case distributions to benchmark. Available: {ALL_DISTRIBUTIONS}. "
            "Omit to use the default for the chosen problem set "
            "(normal only for 'small', all for 'full')."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    args = _parse_args(argv)

    fit_modes: list[bool] = {
        "yes": [True],
        "no": [False],
        "both": [True, False],
    }[args.include_fit]

    parsed_pairs: list[Pair] | None = (
        [Pair(p) for p in args.pairs] if args.pairs else None
    )

    cases: list[BenchmarkCase] = []
    for include_fit in fit_modes:
        cases.extend(
            collect_cases(
                packages=args.packages,
                pairs=parsed_pairs,
                categories=args.categories,
                n_samples_list=args.n_samples,
                include_fit=include_fit,
                min_segment_length=args.min_segment_length,
                dimensions=args.dimensions,
                distributions=args.distributions,
            )
        )

    if args.list_cases:
        for case in cases:
            print(
                f"  [{case.package}] ({case.cpd_algorithm}) {case.name}  "
                f"n={case.n_samples} p={case.data_dimension}"
            )
        print(f"\n{len(cases)} benchmark case(s) total.")
        return

    print(
        f"Running {len(cases)} benchmark(s), {args.runs} run(s) each → {args.output}\n"
    )

    results = []
    t0 = time.perf_counter()

    for i, case in enumerate(cases, 1):
        label = f"[{case.package}] {case.name}"
        dims = f"n={case.n_samples}, p={case.data_dimension}"
        print(f"  ({i}/{len(cases)}) {label} ({dims}) ...", end=" ", flush=True)

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
            n_runs=args.runs,
            penalty=case.penalty,
        )

        print(
            f"mean={res.mean:.4f}s  std={res.std:.4f}s  "
            f"median={res.median:.4f}s  min={res.min:.4f}s  "
            f"changes={res.n_detected_changepoints}"
        )
        results.append(res.as_dict())

    elapsed = time.perf_counter() - t0
    print(f"\nAll benchmarks finished in {elapsed:.1f}s.")

    # Write Parquet
    df = pl.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
