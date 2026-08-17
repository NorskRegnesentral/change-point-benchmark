#!/usr/bin/env bash
# Run the full focused benchmark suite and regenerate all paper figures.
#
# Each benchmark script is resumable: completed cases already present in its
# output Parquet file are skipped, so an interrupted run can be restarted.
#
# Usage:
#   ./scripts/run_benchmarks.sh              # run benchmarks + plots
#   ./scripts/run_benchmarks.sh --no-plots   # run benchmarks only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RUN_PLOTS=1
[[ "${1:-}" == "--no-plots" ]] && RUN_PLOTS=0

BENCHMARKS=(
    scripts/paper_benchmarks/run_change_in_mean_benchmark.py
    scripts/paper_benchmarks/run_change_in_mean_l1_benchmark.py
    scripts/paper_benchmarks/run_multivariate_dimension_benchmark.py
    scripts/paper_benchmarks/run_rank_score_benchmark.py
)

for script in "${BENCHMARKS[@]}"; do
    echo "=== Running $script ==="
    uv run python "$script"
done

# The compact score figure reads curated results from visualized_results/.
VISUALIZED_DIR="results/paper/visualized_results"
mkdir -p "$VISUALIZED_DIR"
for prefix in change-in-mean-benchmark rank-score-benchmark; do
    latest="$(ls results/paper/${prefix}_*.parquet 2>/dev/null | sort | tail -n 1 || true)"
    [[ -n "$latest" ]] && cp "$latest" "$VISUALIZED_DIR/"
done

if [[ "$RUN_PLOTS" == "1" ]]; then
    PLOTS=(
        scripts/paper_plotting/plot_change_in_mean_benchmark.py
        scripts/paper_plotting/plot_l1_change_in_mean_benchmark.py
        scripts/paper_plotting/plot_mv_dimension_benchmark.py
        scripts/paper_plotting/plot_compact_score_benchmarks.py
    )
    for script in "${PLOTS[@]}"; do
        echo "=== Plotting $script ==="
        uv run python "$script"
    done
fi

echo "Done. Results in results/paper/, figures in figures/paper/."
