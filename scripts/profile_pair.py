#!/usr/bin/env python
"""Profile a single benchmark pair and produce a flamegraph/flamechart.

This script reuses the benchmark pipeline (registry, prepare/setup/func)
but runs the timed function under a profiler instead of timing it.

Default profiler: pyinstrument (statistical/sampling profiler, no sudo needed).
Install:  uv add pyinstrument

Usage examples
--------------
# Profile skchange moving_window_l1 (fit+predict), 500 samples → HTML flamechart:
#   uv run python scripts/profile_pair.py \
#       --pair moving_window_l1 --package skchange --n-samples 500 --iterations 100 \
#       -o sk_mw_l1.html

# Profile ruptures side (fit+predict):
#   uv run python scripts/profile_pair.py \
#       --pair moving_window_l1 --package ruptures --n-samples 500 --iterations 100 \
#       -o rpt_mw_l1.html

# Profile skchange predict-only:
#   uv run python scripts/profile_pair.py \
#       --pair moving_window_l1 --package skchange --no-fit --n-samples 500 \
#       -o sk_mw_l1_predict.html

# Text output to terminal (no -o flag):
#   uv run python scripts/profile_pair.py \
#       --pair moving_window_l1 --package skchange --n-samples 500

# cProfile fallback (less visual, but can dump .prof for snakeviz/flameprof):
#   uv run python scripts/profile_pair.py \
#       --pair moving_window_l1 --package skchange --cprofile

Notes
-----
* pyinstrument is a statistical (sampling) profiler that works entirely
  in-process — no ptrace, no sudo, no external wrapper needed.
* It shows numba-jitted functions as opaque call nodes (you see WHICH numba
  function takes time, but not the native instructions within it). This is
  the best you can get without ptrace/perf access.
* --warmup runs the function once before the profiled loop to trigger JIT
  compilation (enabled by default), so the flamegraph reflects steady-state
  performance rather than compilation overhead.
* Output formats: .html (interactive flamechart), .txt (tree), .json (raw).
  Determined by the -o file extension, or defaults to text on stdout.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import warnings
from pathlib import Path

from change_bench.benchmarks.comparison_pairs._common import BenchmarkCase
from change_bench.benchmarks.registry import (
    Pair,
    collect_cases,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a benchmark pair and produce a flamechart."
    )
    parser.add_argument(
        "--pair",
        required=True,
        choices=[p.value for p in Pair],
        help="Comparison pair to profile.",
    )
    parser.add_argument(
        "--package",
        required=True,
        choices=["skchange", "ruptures"],
        help="Which side of the pair to profile.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples in the generated dataset (default: 500).",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Profile predict only (exclude fit from timed code).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations to run under the profiler (default: 50).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warmup iteration (not recommended for numba code).",
    )
    parser.add_argument(
        "--distribution",
        default="normal",
        help="Null-case distribution for the data (default: normal).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=1,
        help="Number of data columns (default: 1).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file path. Extension determines format: "
            ".html (interactive flamechart), .json (raw data), .txt (text tree). "
            "Default: print text tree to stdout."
        ),
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Use cProfile instead of pyinstrument (less visual, but no deps).",
    )
    parser.add_argument(
        "--cprofile-output",
        type=Path,
        default=None,
        help="Path to dump cProfile .prof stats (for snakeviz/flameprof).",
    )
    return parser.parse_args()


def _select_case(
    cases: list[BenchmarkCase], package: str, n_samples: int
) -> BenchmarkCase:
    """Pick the matching case from collected cases."""
    matches = [c for c in cases if c.package == package and c.n_samples == n_samples]
    if not matches:
        available = [(c.package, c.n_samples) for c in cases]
        raise SystemExit(
            f"No case found for package={package!r}, n_samples={n_samples}.\n"
            f"Available: {available}"
        )
    return matches[0]


def _run_loop(case: BenchmarkCase, iterations: int) -> None:
    """Execute the benchmark function in a loop (to be sampled by py-spy)."""
    data = case.prepare()

    for _ in range(iterations):
        args, kwargs = case.setup(data)
        case.func(*args, **kwargs)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = _parse_args()

    include_fit = not args.no_fit

    # Use the same collect_cases pipeline as the CLI benchmark runner.
    cases = collect_cases(
        packages=[args.package],
        pairs=[Pair(args.pair)],
        problem_set="small",
        include_fit=include_fit,
        dimensions=[args.dimensions],
        distributions=[args.distribution],
    )

    case = _select_case(cases, args.package, args.n_samples)

    fit_label = "fit+predict" if include_fit else "predict_only"
    print(
        f"Profiling: [{case.package}] {case.cpd_algorithm} | "
        f"{fit_label} | n={case.n_samples}, p={case.data_dimension} | "
        f"iterations={args.iterations}",
        file=sys.stderr,
    )

    if not args.no_warmup:
        print(
            "Warmup: running one iteration to trigger JIT compilation...",
            file=sys.stderr,
        )
        _run_loop(case, 1)
        print("Warmup complete.", file=sys.stderr)

    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        _run_loop(case, args.iterations)
        profiler.disable()

        if args.cprofile_output:
            profiler.dump_stats(str(args.cprofile_output))
            print(f"cProfile stats saved to {args.cprofile_output}", file=sys.stderr)
        else:
            stats = pstats.Stats(profiler, stream=sys.stdout)
            stats.sort_stats("cumulative")
            stats.print_stats(40)
    else:
        import pyinstrument
        from pyinstrument.renderers import JSONRenderer

        profiler = pyinstrument.Profiler(interval=0.001)
        profiler.start()
        _run_loop(case, args.iterations)
        profiler.stop()

        if args.output:
            ext = args.output.suffix.lower()
            if ext == ".html":
                output = profiler.output_html()
            elif ext == ".json":
                output = profiler.output(renderer=JSONRenderer())
            else:
                output = profiler.output_text(unicode=True, color=False)
            args.output.write_text(output)
            print(f"Profile saved to {args.output}", file=sys.stderr)
        else:
            print(profiler.output_text(unicode=True, color=True))

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
