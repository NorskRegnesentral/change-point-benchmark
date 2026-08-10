# %%
"""Interactive analysis of benchmark results.

Loads one or more Parquet results file and produces comparison plots of runtime vs
n_samples for each algorithm pair, with one line per library
(skchange vs ruptures).
"""

from __future__ import annotations

import math
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

#: Line dash style per fit mode
FIT_DASH: dict[bool, str] = {
    True: "solid",  # fit+predict
    False: "dash",  # predict only
}

# ---------------------------------------------------------------------------
# Toggle: which fit modes to include in the analysis
# Set to [True] for fit+predict only, [False] for predict only, or both.
# ---------------------------------------------------------------------------
INCLUDE_FIT_VALUES: list[bool] = [True]

# ---------------------------------------------------------------------------
# Toggle: metric to plot and report
# "ski-jump-mean" — uses ski_jump_mean_s/ski_jump_std_s (mean ± std after
#                    removing the fastest and slowest run)
# "min"           — uses min_s (best-of-N, no error bars)
# ---------------------------------------------------------------------------
# METRIC: str = "ski-jump-mean"  # "ski-jump-mean" or "min"
METRIC: str = "min"  # "ski-jump-mean" or "min"

# ---------------------------------------------------------------------------
# Configuration — change this path when running interactively
# ---------------------------------------------------------------------------
project_dir = Path(__file__).parent.parent.parent
results_dir: Path = project_dir / "results" / "varied"
# results_dir: Path = project_dir / "old_results"

if not results_dir.exists():
    print(f"Error: results directory not found at {results_dir}")
    sys.exit(1)

parquet_files = sorted(results_dir.glob("*.parquet"))
if not parquet_files:
    print(f"Error: no .parquet files found in {results_dir}")
    sys.exit(1)

# %% ---------------------------------------------------------------------------
# Load data — concatenate all parquet files in the results directory
# ---------------------------------------------------------------------------

parts = [pl.read_parquet(p) for p in parquet_files]
df = pl.concat(parts)
print(f"Loaded {len(df)} rows from {len(parquet_files)} file(s) in {results_dir}/")
for p in parquet_files:
    print(f"  - {p.name}")
print(f"Columns: {df.columns}")
print(f"Algorithm pairs: {df['cpd_algorithm'].unique().sort().to_list()}")
print(f"Packages: {df['package'].unique().sort().to_list()}")
print()

# %% ---------------------------------------------------------------------------
# Plot: mean runtime vs n_samples (log scale) for each algorithm pair
# One figure per data dimension (p).
# ---------------------------------------------------------------------------

pairs = df["cpd_algorithm"].unique().sort().to_list()
packages = df["package"].unique().sort().to_list()
include_fit_values = INCLUDE_FIT_VALUES
dimensions = df["data_dimension"].unique().sort().to_list()

for dim in dimensions:
    dim_df = df.filter(pl.col("data_dimension") == dim)
    dim_pairs = dim_df["cpd_algorithm"].unique().sort().to_list()
    n_pairs = len(dim_pairs)

    if n_pairs == 0:
        continue

    max_cols = 2
    n_cols = min(n_pairs, max_cols)
    n_rows = math.ceil(n_pairs / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=dim_pairs,
        shared_yaxes=False,
        vertical_spacing=0.25 / n_rows if n_rows > 1 else 0.15,
    )

    shown_legends: set[str] = set()

    for subplot_idx, pair in enumerate(dim_pairs):
        row_idx = subplot_idx // n_cols + 1
        col_idx = subplot_idx % n_cols + 1
        pair_df = dim_df.filter(pl.col("cpd_algorithm") == pair)

        for fit_val in include_fit_values:
            fit_suffix = "fit+predict" if fit_val else "predict only"
            dash = FIT_DASH.get(fit_val, "solid")

            for pkg in packages:
                base_df = pair_df.filter(
                    (pl.col("package") == pkg) & (pl.col("include_fit") == fit_val)
                )

                if METRIC == "min":
                    pkg_df = (
                        base_df.group_by("n_samples")
                        .agg(pl.col("min_s").min().alias("value"))
                        .sort("n_samples")
                    )
                else:
                    pkg_df = (
                        base_df.group_by("n_samples")
                        .agg(
                            pl.col("ski_jump_mean_s").mean().alias("value"),
                            pl.col("ski_jump_std_s").mean().alias("std"),
                        )
                        .sort("n_samples")
                    )

                if pkg_df.is_empty():
                    continue

                n_samples = pkg_df["n_samples"].to_list()
                values = pkg_df["value"].to_list()

                legend_name = pkg
                show_legend = legend_name not in shown_legends
                shown_legends.add(legend_name)

                if METRIC == "min":
                    error_y = None
                    custom_hover = [
                        f"n={n}, min={v:.1e} s" for n, v in zip(n_samples, values)
                    ]
                else:
                    stds = pkg_df["std"].to_list()
                    error_y = dict(type="data", array=stds, visible=True)
                    custom_hover = [
                        f"n={n}, {v:.1e} ± {s:.1e} s"
                        for n, v, s in zip(n_samples, values, stds)
                    ]

                fig.add_trace(
                    go.Scatter(
                        x=n_samples,
                        y=values,
                        error_y=error_y,
                        mode="lines+markers",
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=show_legend,
                        line=dict(color=PACKAGE_COLORS.get(pkg), dash=dash),
                        marker=dict(color=PACKAGE_COLORS.get(pkg)),
                        text=custom_hover,
                        hoverinfo="text+name",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

        fig.update_xaxes(title_text="n_samples", row=row_idx, col=col_idx)
        fig.update_yaxes(
            title_text="time (s)",
            type="log",
            minor=dict(ticks="inside", showgrid=True),
            exponentformat="power",
            showexponent="all",
            row=row_idx,
            col=col_idx,
        )

    metric_label = "min runtime" if METRIC == "min" else "ski-jump-mean ± std"
    fig.update_layout(
        title=(
            f"Runtime comparison ({metric_label}, log scale):"
            + f" skchange vs ruptures — p={dim}"
        ),
        height=400 * n_rows,
        width=500 * n_cols,
    )
    fig.show()

# %% ---------------------------------------------------------------------------
# Summary table: speedup ratio (skchange / ruptures) per pair × n_samples
# ---------------------------------------------------------------------------

metric_col = "min_s" if METRIC == "min" else "ski_jump_mean_s"
metric_label_summary = "min" if METRIC == "min" else "ski-jump-mean"
print("=" * 60)
print(
    f"Speedup ratio (ruptures {metric_label_summary} / skchange {metric_label_summary})"
)
print("=" * 60)

for dim in dimensions:
    dim_df = df.filter(pl.col("data_dimension") == dim)
    print(f"\n{'=' * 60}")
    print(f"  Data dimension p={dim}")
    print(f"{'=' * 60}")

    for fit_val in include_fit_values:
        fit_desc = "fit+predict" if fit_val else "predict only"
        fit_df = dim_df.filter(pl.col("include_fit") == fit_val)
        fit_pairs = fit_df["cpd_algorithm"].unique().sort().to_list()

        if not fit_pairs:
            continue

        print(f"\n  --- {fit_desc} ---")

        for pair in fit_pairs:
            pair_df = fit_df.filter(pl.col("cpd_algorithm") == pair)

            agg_expr = (
                pl.col(metric_col).min()
                if METRIC == "min"
                else pl.col(metric_col).mean()
            )

            sk = (
                pair_df.filter(pl.col("package") == "skchange")
                .group_by("n_samples")
                .agg(agg_expr.alias("sk_val"))
                .sort("n_samples")
            )
            rpt = (
                pair_df.filter(pl.col("package") == "ruptures")
                .group_by("n_samples")
                .agg(agg_expr.alias("rpt_val"))
                .sort("n_samples")
            )

            joined = sk.join(rpt, on="n_samples").with_columns(
                (pl.col("rpt_val") / pl.col("sk_val")).alias("rpt_sk_ratio")
            )

            print(f"\n  {pair}:")
            for row in joined.iter_rows(named=True):
                print(
                    f"    n={row['n_samples']:>6}  "
                    f"skchange={row['sk_val']:.4f}s  "
                    f"ruptures={row['rpt_val']:.4f}s  "
                    f"rpt_over_sk_ratio={row['rpt_sk_ratio']:.3f}x"
                )

print()

# %%
