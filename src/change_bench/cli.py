"""CLI entrypoint for running change-point benchmarks.

Usage::

    uv run bench --runs 10 --groups ruptures skchange --output results.parquet
    uv run bench --runs 5 --groups skchange --problem-set full
    uv run bench --list
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from change_bench.benchmarks.null_case import BENCHMARK_PAIRS, collect_cases
from change_bench.runner import run_benchmark


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Run change-point detection benchmarks and save results to Parquet.",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=5,
        help="Number of timed repetitions per benchmark case (default: 5).",
    )
    parser.add_argument(
        "-g",
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Filter to specific library groups (ruptures, skchange). "
            "Omit to run both sides of each pair."
        ),
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            f"Comparison pairs to run. Available: {sorted(BENCHMARK_PAIRS)}. "
            "Omit to run all pairs."
        ),
    )
    parser.add_argument(
        "-p",
        "--problem-set",
        choices=["small", "full"],
        default="small",
        help="Problem battery size (default: small).",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include fit() in the timed operation (default: True). "
            "Use --no-include-fit to time only predict."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    cases = collect_cases(
        groups=args.groups,
        pairs=args.pairs,
        problem_set=args.problem_set,
        include_fit=args.include_fit,
        min_segment_length=args.min_segment_length,
    )

    if args.list_cases:
        for case in cases:
            print(
                f"  [{case.group}] ({case.cpd_algorithm}) {case.name}  "
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
        label = f"[{case.group}] {case.name}"
        print(f"  ({i}/{len(cases)}) {label} ...", end=" ", flush=True)

        res = run_benchmark(
            group=case.group,
            cpd_algorithm=case.cpd_algorithm,
            name=case.name,
            n_samples=case.n_samples,
            n_changepoints=case.n_changepoints,
            data_dimension=case.data_dimension,
            include_fit=case.include_fit,
            min_segment_length=case.min_segment_length,
            setup=case.setup,
            func=case.func,
            n_runs=args.runs,
        )

        print(
            f"mean={res.mean:.4f}s  std={res.std:.4f}s  "
            f"median={res.median:.4f}s  min={res.min:.4f}s"
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
