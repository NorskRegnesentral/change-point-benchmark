#!/usr/bin/env bash
# Run change-point detection benchmarks and store results in results/.
#
# Runs two passes:
#   1. mean_change  pairs  (min-segment-length 1)
#   2. mean_variance pairs (min-segment-length = max data dimension + 1, default 2)
#
# Usage:
#   ./scripts/run_benchmarks.sh                       # defaults: 10 runs, small
#   ./scripts/run_benchmarks.sh --runs 20 --problem-set full
#   ./scripts/run_benchmarks.sh --min-var-segment-length 5
#
# Recognised flags (consumed by this script, NOT forwarded to bench):
#   --min-var-segment-length N   min_segment_length for mean_variance pairs
#                                (default: 2)
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
MIN_VAR_SEGMENT_LENGTH=2
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-var-segment-length)
            MIN_VAR_SEGMENT_LENGTH="$2"
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
MEAN_VARIANCE_OUTPUT="$RESULTS_DIR/mean_variance.parquet"

echo "=== Change-Point Benchmark Runner ==="
echo "Project:              $PROJECT_DIR"
echo "Runs per case:        $RUNS"
echo "mean_change output:   $MEAN_CHANGE_OUTPUT"
echo "mean_variance output: $MEAN_VARIANCE_OUTPUT  (min_segment_length=$MIN_VAR_SEGMENT_LENGTH)"
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

# --- Pass 2: mean_variance (min_segment_length > max dimension) ------------
echo "--- [2/2] Running mean_variance benchmarks (min_segment_length=$MIN_VAR_SEGMENT_LENGTH) ---"
uv run bench \
    -n "$RUNS" \
    --categories mean_variance \
    --min-segment-length "$MIN_VAR_SEGMENT_LENGTH" \
    -o "$MEAN_VARIANCE_OUTPUT" \
    "${FORWARD_ARGS[@]}"

echo ""
echo "Done."
echo "  mean_change  → $MEAN_CHANGE_OUTPUT"
echo "  mean_variance → $MEAN_VARIANCE_OUTPUT"
echo ""
echo "Analyse with:"
echo "  uv run scripts/analyse_results.py $MEAN_CHANGE_OUTPUT"
echo "  uv run scripts/analyse_results.py $MEAN_VARIANCE_OUTPUT"
