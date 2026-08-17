#!/usr/bin/env python
"""Verify that paired paper benchmark cases detect no changes on null data.

Each selected benchmark case is run once with its configured penalty and data.
Both packages must return zero interior changes. The terminal endpoint returned
by ruptures is excluded from its change count.

Run all paper benchmark setups with::

    uv run python scripts/paper_benchmarks/verify_benchmarks.py

Use ``--benchmark`` and ``--max-n-samples`` for a quicker targeted check.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

from change_bench.benchmark_verification import (
    VerificationResult,
    calibrate_penalties,
    collect_benchmark_cases,
    verify_cases,
)

BENCHMARK_FILES: dict[str, str] = {
    "change_in_mean_l2": "run_change_in_mean_benchmark.py",
    "change_in_mean_l1": "run_change_in_mean_l1_benchmark.py",
    "multivariate_dimension": "run_multivariate_dimension_benchmark.py",
    "rank_score": "run_rank_score_benchmark.py",
    "continuous_linear_trend": "run_continuous_linear_trend_benchmark.py",
}


def load_benchmark_module(filename: str) -> ModuleType:
    """Load a neighboring benchmark script without executing its main function."""
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmark configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _format_result(benchmark: str, result: VerificationResult) -> str:
    key = result.key
    status = "PASS" if result.passes else "FAIL"
    return (
        f"[{status}] {benchmark}: {key.cpd_algorithm}, n={key.n_samples}, "
        f"p={key.data_dimension}, penalty={key.penalty:g}: "
        f"skchange={result.skchange_count}, ruptures={result.ruptures_count}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=sorted(BENCHMARK_FILES),
        help="Benchmark setup to verify; repeat to select several (default: all).",
    )
    parser.add_argument(
        "--max-n-samples",
        type=int,
        help="Only verify configured sample sizes at or below this value.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Find sufficient per-pair penalties over the selected configurations.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.5,
        help="Multiplier applied to the first sufficient penalty (default: 1.5).",
    )
    return parser.parse_args()


def main() -> int:
    """Run configured verifications and return a process exit code."""
    args = parse_args()
    selected = args.benchmark or list(BENCHMARK_FILES)
    selected_cases = []
    for benchmark in selected:
        module = load_benchmark_module(BENCHMARK_FILES[benchmark])
        selected_cases.extend(
            collect_benchmark_cases(module, max_n_samples=args.max_n_samples)
        )

    if args.calibrate:
        calibrations = calibrate_penalties(selected_cases, margin=args.margin)
        for calibration in calibrations:
            print(
                f"{calibration.cpd_algorithm}: initial="
                f"{calibration.initial_penalty:g}, sufficient="
                f"{calibration.sufficient_penalty:g}, selected="
                f"{calibration.selected_penalty:g}"
            )
        return 0

    total_results = 0
    total_skipped = 0
    mismatches = 0

    for benchmark in selected:
        module = load_benchmark_module(BENCHMARK_FILES[benchmark])
        cases = collect_benchmark_cases(module, max_n_samples=args.max_n_samples)
        results, skipped = verify_cases(cases)
        total_results += len(results)
        total_skipped += len(skipped)
        mismatches += sum(not result.passes for result in results)
        for result in results:
            print(_format_result(benchmark, result))
        for key in skipped:
            print(
                f"[SKIP] {benchmark}: {key.cpd_algorithm}, n={key.n_samples}, "
                f"p={key.data_dimension} has no two-sided comparison"
            )

    print(
        f"\nVerified {total_results} paired cases; "
        f"{mismatches} nonzero or mismatched results; "
        f"{total_skipped} one-sided cases skipped."
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
