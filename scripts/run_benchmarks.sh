#!/usr/bin/env bash
# Run change-point detection benchmarks and store results in results/.
#
# Usage:
#   ./scripts/run_benchmarks.sh                  # defaults: 10 runs, small, all pairs
#   ./scripts/run_benchmarks.sh --runs 20 --problem-set full
#   ./scripts/run_benchmarks.sh --pairs pelt_l2 binseg --no-include-fit
#
# All arguments are forwarded to `uv run bench`.  The output path is set
# automatically to results/benchmark_results.parquet (override with -o).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Default arguments (can be overridden by caller)
RUNS=10
OUTPUT="$RESULTS_DIR/benchmark_results.parquet"

# Check if user already specified -o / --output
has_output=false
for arg in "$@"; do
    if [[ "$arg" == "-o" || "$arg" == "--output" ]]; then
        has_output=true
        break
    fi
done

EXTRA_ARGS=()
if [[ "$has_output" == false ]]; then
    EXTRA_ARGS+=("-o" "$OUTPUT")
fi

# Check if user already specified -n / --runs
has_runs=false
for arg in "$@"; do
    if [[ "$arg" == "-n" || "$arg" == "--runs" ]]; then
        has_runs=true
        break
    fi
done

if [[ "$has_runs" == false ]]; then
    EXTRA_ARGS+=("-n" "$RUNS")
fi

echo "=== Change-Point Benchmark Runner ==="
echo "Project:  $PROJECT_DIR"
# echo "Results:  ${OUTPUT}"
echo ""

cd "$PROJECT_DIR"
uv run bench "${EXTRA_ARGS[@]}" "$@"

echo ""
echo "Done. Results saved to: ${OUTPUT}"
echo "Analyse with:  uv run scripts/analyse_results.py ${OUTPUT}"
