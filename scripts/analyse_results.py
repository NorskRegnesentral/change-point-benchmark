# %%
"""Interactive analysis of benchmark results.

Loads a Parquet results file and produces comparison plots of runtime vs
n_samples for each algorithm pair, with one line per library (skchange vs
ruptures).

Usage (standalone)::

    uv run scripts/analyse_results.py results/mean_change.parquet

Or interactively in an editor / REPL — just set ``results_path`` below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Consistent color scheme for packages
# ---------------------------------------------------------------------------
PACKAGE_COLORS: dict[str, str] = {
    "skchange": "#1f77b4",  # blue
    "ruptures": "#ff7f0e",  # orange
}

# ---------------------------------------------------------------------------
# Configuration — change this path when running interactively
# ---------------------------------------------------------------------------
project_dir = Path(__file__).parent.parent
# results_path: Path = project_dir / "results/mean_change.parquet"
# results_path: Path = project_dir / "results/needs_min_segment_length.parquet"
results_path: Path = project_dir / "results/pelt_l2.parquet"

if not results_path.exists():
    print(f"Error: results file not found at {results_path}")
    print(
        "Please set the 'results_path' variable in the script "
        "to point to your Parquet file."
    )
    sys.exit(1)

# %% ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pl.read_parquet(results_path)
print(f"Loaded {len(df)} rows from {results_path}")
print(f"Columns: {df.columns}")
print(f"Algorithm pairs: {df['cpd_algorithm'].unique().sort().to_list()}")
print(f"Packages: {df['package'].unique().sort().to_list()}")
print()

# %% ---------------------------------------------------------------------------
# Plot: mean runtime vs n_samples (log scale) for each algorithm pair
# ---------------------------------------------------------------------------

pairs = df["cpd_algorithm"].unique().sort().to_list()
packages = df["package"].unique().sort().to_list()
include_fit_values = df["include_fit"].unique().sort().to_list()
fit_label = (
    "fit+predict" if all(include_fit_values) else
    "predict only" if not any(include_fit_values) else
    "mixed (fit+predict / predict only)"
)

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
                line=dict(color=PACKAGE_COLORS.get(pkg)),
                marker=dict(color=PACKAGE_COLORS.get(pkg)),
            ),
            row=1,
            col=col_idx,
        )

    fig.update_xaxes(title_text="n_samples", row=1, col=col_idx)
    fig.update_yaxes(
        title_text="time (s)",
        type="log",
        minor=dict(ticks="inside", showgrid=True),
        exponentformat="power",
        showexponent="all",
        row=1,
        col=col_idx,
    )

fig.update_layout(
    title=f"Runtime comparison (log scale): skchange vs ruptures [{fit_label}]",
    height=500,
    width=500 * n_pairs,
)
fig.show()

# ---------------------------------------------------------------------------
# Summary table: speedup ratio (skchange / ruptures) per pair × n_samples
# ---------------------------------------------------------------------------

print("=" * 60)
print("Speedup ratio (skchange mean / ruptures mean)")
print("=" * 60)

for fit_val in include_fit_values:
    fit_desc = "fit+predict" if fit_val else "predict only"
    fit_df = df.filter(pl.col("include_fit") == fit_val)
    fit_pairs = fit_df["cpd_algorithm"].unique().sort().to_list()

    if not fit_pairs:
        continue

    print(f"\n  --- {fit_desc} ---")

    for pair in fit_pairs:
        pair_df = fit_df.filter(pl.col("cpd_algorithm") == pair)

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
                f"ratio={row['ratio']:.2f}x"
            )

print()

# %%
