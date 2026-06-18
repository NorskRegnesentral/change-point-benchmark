#!/usr/bin/env bash
# Run change-point detection benchmarks and store results in results/.
#
# Runs two passes:
#   1. mean_change pairs (min-segment-length 1)
#   2. needs_min_segment_length pairs (min-segment-length configurable, default 3)
#
# Usage:
#   ./scripts/run_benchmarks.sh                       # defaults: 10 runs, small
#   ./scripts/run_benchmarks.sh --runs 20 --problem-set full
#   ./scripts/run_benchmarks.sh --min-segment-length 5
#
# Recognised flags (consumed by this script, NOT forwarded to bench):
#   --min-segment-length N   min_segment_length for needs_min_segment_length
#                            pairs (default: 3)
#
# All other arguments are forwarded to both `uv run bench` invocations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Parse script-specific flags; collect remaining args to forward
# ---------------------------------------------------------------------------
RUNS=10
MIN_SEGMENT_LENGTH=3
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-segment-length)
            MIN_SEGMENT_LENGTH="$2"
            shift 2
            ;;
        -n|--runs)
            RUNS="$2"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

MEAN_CHANGE_OUTPUT="$RESULTS_DIR/mean_change.parquet"
MSL_OUTPUT="$RESULTS_DIR/needs_min_segment_length.parquet"

echo "=== Change-Point Benchmark Runner ==="
echo "Project:                        $PROJECT_DIR"
echo "Runs per case:                  $RUNS"
echo "mean_change output:             $MEAN_CHANGE_OUTPUT"
echo "needs_min_segment_length output: $MSL_OUTPUT  (min_segment_length=$MIN_SEGMENT_LENGTH)"
echo ""

cd "$PROJECT_DIR"

# --- Pass 1: mean_change (min_segment_length=1) ---------------------------
echo "--- [1/2] Running mean_change benchmarks ---"
uv run bench \
    -n "$RUNS" \
    --categories mean_change \
    --min-segment-length 1 \
    -o "$MEAN_CHANGE_OUTPUT" \
    "${FORWARD_ARGS[@]}"

echo ""

# --- Pass 2: needs_min_segment_length (min_segment_length > 1) -------------
echo "--- [2/2] Running needs_min_segment_length benchmarks (min_segment_length=$MIN_SEGMENT_LENGTH) ---"
uv run bench \
    -n "$RUNS" \
    --categories needs_min_segment_length \
    --min-segment-length "$MIN_SEGMENT_LENGTH" \
    -o "$MSL_OUTPUT" \
    "${FORWARD_ARGS[@]}"

echo ""
echo "Done."
echo "  mean_change                → $MEAN_CHANGE_OUTPUT"
echo "  needs_min_segment_length   → $MSL_OUTPUT"
echo ""
echo "Analyse with:"
echo "  uv run scripts/analyse_results.py $MEAN_CHANGE_OUTPUT"
echo "  uv run scripts/analyse_results.py $MSL_OUTPUT"
