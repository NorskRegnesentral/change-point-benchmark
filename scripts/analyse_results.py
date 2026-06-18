"""Interactive analysis of benchmark results.

Loads a Parquet results file and produces comparison plots of runtime vs
n_samples for each algorithm pair, with one line per library (skchange vs
ruptures).

Usage (standalone)::

    uv run scripts/analyse_results.py results/benchmark_results.parquet

Or interactively in an editor / REPL — just set ``results_path`` below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration — change this path when running interactively
# ---------------------------------------------------------------------------

results_path: Path = Path("results/benchmark_results.parquet")

# Override from CLI if provided
if len(sys.argv) > 1:
    results_path = Path(sys.argv[1])

if not results_path.exists():
    print(f"Results file not found: {results_path}")
    print("Run benchmarks first:  uv run bench -o results/benchmark_results.parquet")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pl.read_parquet(results_path)
print(f"Loaded {len(df)} rows from {results_path}")
print(f"Columns: {df.columns}")
print(f"Algorithm pairs: {df['cpd_algorithm'].unique().sort().to_list()}")
print(f"Packages: {df['package'].unique().sort().to_list()}")
print()

# ---------------------------------------------------------------------------
# Plot: mean runtime vs n_samples for each algorithm pair
# ---------------------------------------------------------------------------

pairs = df["cpd_algorithm"].unique().sort().to_list()
packages = df["package"].unique().sort().to_list()

n_pairs = len(pairs)
fig = make_subplots(
    rows=1,
    cols=n_pairs,
    subplot_titles=pairs,
    shared_yaxes=False,
)

for col_idx, pair in enumerate(pairs, start=1):
    pair_df = df.filter(pl.col("cpd_algorithm") == pair)

    for pkg in packages:
        pkg_df = (
            pair_df.filter(pl.col("package") == pkg)
            .group_by("n_samples")
            .agg(
                pl.col("mean_s").mean().alias("mean"),
                pl.col("std_s").mean().alias("std"),
            )
            .sort("n_samples")
        )

        n_samples = pkg_df["n_samples"].to_list()
        means = pkg_df["mean"].to_list()
        stds = pkg_df["std"].to_list()

        fig.add_trace(
            go.Scatter(
                x=n_samples,
                y=means,
                error_y=dict(type="data", array=stds, visible=True),
                mode="lines+markers",
                name=pkg,
                legendgroup=pkg,
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )

    fig.update_xaxes(title_text="n_samples", row=1, col=col_idx)
    fig.update_yaxes(title_text="time (s)", row=1, col=col_idx)

fig.update_layout(
    title="Runtime comparison: skchange vs ruptures",
    height=500,
    width=500 * n_pairs,
)
fig.show()

# ---------------------------------------------------------------------------
# Plot: log-scale version (useful when ruptures is orders of magnitude faster)
# ---------------------------------------------------------------------------

fig_log = make_subplots(
    rows=1,
    cols=n_pairs,
    subplot_titles=pairs,
    shared_yaxes=False,
)

for col_idx, pair in enumerate(pairs, start=1):
    pair_df = df.filter(pl.col("cpd_algorithm") == pair)

    for pkg in packages:
        pkg_df = (
            pair_df.filter(pl.col("package") == pkg)
            .group_by("n_samples")
            .agg(pl.col("mean_s").mean().alias("mean"))
            .sort("n_samples")
        )

        fig_log.add_trace(
            go.Scatter(
                x=pkg_df["n_samples"].to_list(),
                y=pkg_df["mean"].to_list(),
                mode="lines+markers",
                name=pkg,
                legendgroup=pkg,
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )

    fig_log.update_xaxes(title_text="n_samples", row=1, col=col_idx)
    fig_log.update_yaxes(title_text="time (s)", type="log", row=1, col=col_idx)

fig_log.update_layout(
    title="Runtime comparison (log scale): skchange vs ruptures",
    height=500,
    width=500 * n_pairs,
)

# ---------------------------------------------------------------------------
# Summary table: speedup ratio (skchange / ruptures) per pair × n_samples
# ---------------------------------------------------------------------------

print("=" * 60)
print("Speedup ratio (skchange mean / ruptures mean)")
print("=" * 60)

for pair in pairs:
    pair_df = df.filter(pl.col("cpd_algorithm") == pair)

    sk = (
        pair_df.filter(pl.col("package") == "skchange")
        .group_by("n_samples")
        .agg(pl.col("mean_s").mean().alias("sk_mean"))
        .sort("n_samples")
    )
    rpt = (
        pair_df.filter(pl.col("package") == "ruptures")
        .group_by("n_samples")
        .agg(pl.col("mean_s").mean().alias("rpt_mean"))
        .sort("n_samples")
    )

    joined = sk.join(rpt, on="n_samples").with_columns(
        (pl.col("sk_mean") / pl.col("rpt_mean")).alias("ratio")
    )

    print(f"\n  {pair}:")
    for row in joined.iter_rows(named=True):
        print(
            f"    n={row['n_samples']:>6}  "
            f"skchange={row['sk_mean']:.4f}s  "
            f"ruptures={row['rpt_mean']:.4f}s  "
            f"ratio={row['ratio']:.1f}x"
        )

print()

fig_log.show()
