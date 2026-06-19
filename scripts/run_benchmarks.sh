#!/usr/bin/env bash
# Run change-point detection benchmarks and store results in results/.
#
# Runs two passes with different min-segment-length settings, then merges
# results into a single Parquet file (min_segment_length is recorded per row).
#
# Usage:
#   ./scripts/run_benchmarks.sh                       # defaults: 10 runs, small
#   ./scripts/run_benchmarks.sh --runs 20 --problem-set full
#   ./scripts/run_benchmarks.sh --min-segment-length 5
#
# Recognised flags (consumed by this script, NOT forwarded to bench):
#   --min-segment-length N   min_segment_length for needs_min_segment_length
#                            pairs (default: 3)
#   --dimensions N [N ...]   data dimensionalities to benchmark (default: 1 2 5 10)
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
DIMENSIONS=(1 2 5)
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
        --dimensions)
            DIMENSIONS=()
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DIMENSIONS+=("$1")
                shift
            done
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

MEAN_CHANGE_OUTPUT="$RESULTS_DIR/mean_change.parquet"
MSL_OUTPUT="$RESULTS_DIR/needs_min_segment_length.parquet"
MV_OUTPUT="$RESULTS_DIR/multivariate.parquet"

echo "=== Change-Point Benchmark Runner ==="
echo "Project:                        $PROJECT_DIR"
echo "Runs per case:                  $RUNS"
echo "Dimensions:                     ${DIMENSIONS[*]}"
echo "Outputs:                        $RESULTS_DIR/*.parquet"
echo "  min_segment_length (msl pairs): $MIN_SEGMENT_LENGTH"
echo ""

cd "$PROJECT_DIR"

# --- Pass 1: mean_change (min_segment_length=1) ---------------------------
echo "--- [1/3] Running mean_change benchmarks ---"
uv run bench \
    -n "$RUNS" \
    --categories mean_change \
    --include-fit both \
    --min-segment-length 1 \
    --dimensions "${DIMENSIONS[@]}" \
    -o "$MEAN_CHANGE_OUTPUT" \
    ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}

echo ""

# --- Pass 2: needs_min_segment_length (min_segment_length > 1) -------------
echo "--- [2/3] Running needs_min_segment_length benchmarks (min_segment_length=$MIN_SEGMENT_LENGTH) ---"
uv run bench \
    -n "$RUNS" \
    --categories needs_min_segment_length \
    --include-fit both \
    --min-segment-length "$MIN_SEGMENT_LENGTH" \
    --dimensions "${DIMENSIONS[@]}" \
    -o "$MSL_OUTPUT" \
    ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}

echo ""

# --- Pass 3: multivariate (only for dimensions > 1) -----------------------
# Filter dimensions to only those > 1 for multivariate-only pairs.
MV_DIMENSIONS=()
for d in "${DIMENSIONS[@]}"; do
    if [[ "$d" -gt 1 ]]; then
        MV_DIMENSIONS+=("$d")
    fi
done

if [[ ${#MV_DIMENSIONS[@]} -gt 0 ]]; then
    echo "--- [3/3] Running multivariate benchmarks (dimensions: ${MV_DIMENSIONS[*]}) ---"
    uv run bench \
        -n "$RUNS" \
        --categories multivariate \
        --include-fit both \
        --min-segment-length 1 \
        --dimensions "${MV_DIMENSIONS[@]}" \
        -o "$MV_OUTPUT" \
        ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
    echo ""
else
    echo "--- [3/3] Skipping multivariate benchmarks (no dimensions > 1) ---"
    echo ""
fi

echo ""
echo "Done. Results written to: $RESULTS_DIR/"
