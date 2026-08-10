#!/usr/bin/env python
"""Comparison figure for the MV Gaussian change-point benchmark.

One subplot per data dimension, showing runtime vs. n_samples for both
skchange and ruptures using PELT, moving window, and binary segmentation
with the Gaussian cost. Color distinguishes package, line dash and marker
symbol distinguish the search algorithm.

A relative-speed table (ruptures / skchange min-runtime ratio) is printed
at the end.

Run with::

    uv run python scripts/plotting/plot_mv_gaussian_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = PROJECT_DIR / "results" / "mv_gaussian_benchmark.parquet"
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUT_PATH = FIGURES_DIR / "mv-gaussian-benchmark.html"
OUTPUT_PATH_PDF = OUTPUT_PATH.with_suffix(".pdf")

DIMENSIONS: list[int] = [2, 5, 10, 25]

PACKAGE_COLORS: dict[str, str] = {
    "skchange": "#1f77b4",  # blue
    "ruptures": "#ff7f0e",  # orange
}

#: search algorithm -> (line dash, marker symbol)
ALGORITHM_STYLE: dict[str, tuple[str, str]] = {
    "pelt_mv_gaussian": ("solid", "circle"),
    "moving_window_mv_gaussian": ("dash", "square-open"),
    "binseg_mv_gaussian": ("dot", "diamond"),
}

ALGORITHM_LABELS: dict[str, str] = {
    "pelt_mv_gaussian": "PELT",
    "moving_window_mv_gaussian": "Moving window",
    "binseg_mv_gaussian": "Binary segmentation",
}

METRIC_COL = "min_s"


def build_figure(df: pl.DataFrame, dimensions: list[int]) -> go.Figure:
    """Build a runtime-vs-n_samples figure with one subplot per dimension."""
    n_cols = len(dimensions)

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[f"p = {dim}" for dim in dimensions],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    shown_legends: set[str] = set()

    for col_idx, dim in enumerate(dimensions, start=1):
        dim_df = df.filter(pl.col("data_dimension") == dim)

        for algorithm, (dash, symbol) in ALGORITHM_STYLE.items():
            for package, color in PACKAGE_COLORS.items():
                line_df = (
                    dim_df.filter(
                        (pl.col("cpd_algorithm") == algorithm)
                        & (pl.col("package") == package)
                    )
                    .group_by("n_samples")
                    .agg(pl.col(METRIC_COL).min().alias("value"))
                    .sort("n_samples")
                )
                if line_df.is_empty():
                    continue

                legend_name = f"{package} — {ALGORITHM_LABELS[algorithm]}"
                show_legend = legend_name not in shown_legends
                shown_legends.add(legend_name)
                is_skchange = package == "skchange"

                n_samples = line_df["n_samples"].to_list()
                values = line_df["value"].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=n_samples,
                        y=values,
                        mode="lines+markers",
                        name="\u00a0",
                        legend="legend" if is_skchange else "legend2",
                        legendgroup=algorithm,
                        showlegend=show_legend,
                        line=dict(color=color, dash=dash, width=2),
                        marker=dict(
                            color=color,
                            symbol=symbol,
                            size=9,
                            line=dict(color=color, width=1.5),
                        ),
                        text=[
                            f"{legend_name}<br>n={n}, {v:.2e} s"
                            for n, v in zip(n_samples, values)
                        ],
                        hoverinfo="text",
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_xaxes(
            title_text="n samples",
            type="log",
            exponentformat="power",
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            type="log",
            minor=dict(ticks="inside", showgrid=True),
            exponentformat="power",
            showexponent="all",
            row=1,
            col=col_idx,
        )

    fig.update_yaxes(title_text="runtime (s)", row=1, col=1)

    # Label-only legend column: invisible traces carry the algorithm names.
    for algorithm, label in ALGORITHM_LABELS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                legend="legend3",
                legendgroup=algorithm,
                showlegend=True,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
            )
        )

    margin_l = 80
    margin_r = 460
    axes_width = max(420 * n_cols, 560)
    width = axes_width + margin_l + margin_r
    x_labels = 1 + 15 / axes_width
    x_skchange = 1 + (15 + 170) / axes_width
    x_ruptures = x_skchange + 130 / axes_width

    fig.update_layout(
        title=(
            "MV Gaussian change-point benchmark: runtime vs. sample size"
            "<br><sup>Legend column = package, line style + marker = search"
            " algorithm. Best of N runs, fit + predict.</sup>"
        ),
        height=520,
        width=width,
        legend3=dict(
            title=dict(text="\u00a0"),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=x_labels,
            itemwidth=30,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        legend=dict(
            title=dict(
                text="<b>skchange</b>",
                font=dict(color=PACKAGE_COLORS["skchange"]),
            ),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=x_skchange,
            itemwidth=30,
        ),
        legend2=dict(
            title=dict(
                text="<b>ruptures</b>",
                font=dict(color=PACKAGE_COLORS["ruptures"]),
            ),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=x_ruptures,
            itemwidth=30,
        ),
        margin=dict(l=margin_l, r=margin_r, b=80),
        template="plotly_white",
    )
    return fig


def print_relative_speed(df: pl.DataFrame, dimensions: list[int]) -> None:
    """Print ruptures / skchange runtime ratio per algorithm, dimension, and n."""
    print()
    print("=" * 70)
    print("Relative speed: ruptures / skchange (min runtime, >1 means skchange faster)")
    print("=" * 70)

    for algorithm, label in ALGORITHM_LABELS.items():
        print(f"\n  Algorithm: {label}")
        alg_df = df.filter(pl.col("cpd_algorithm") == algorithm)

        sk = (
            alg_df.filter(pl.col("package") == "skchange")
            .group_by(["data_dimension", "n_samples"])
            .agg(pl.col(METRIC_COL).min().alias("sk_s"))
        )
        rpt = (
            alg_df.filter(pl.col("package") == "ruptures")
            .group_by(["data_dimension", "n_samples"])
            .agg(pl.col(METRIC_COL).min().alias("rpt_s"))
        )
        joined = (
            sk.join(rpt, on=["data_dimension", "n_samples"], how="inner")
            .with_columns((pl.col("rpt_s") / pl.col("sk_s")).alias("ratio"))
            .sort(["data_dimension", "n_samples"])
        )

        if joined.is_empty():
            print("    (no paired data)")
            continue

        for dim in dimensions:
            dim_rows = joined.filter(pl.col("data_dimension") == dim)
            if dim_rows.is_empty():
                continue
            print(f"\n    p = {dim}")
            print(f"    {'n_samples':>10}  {'skchange (s)':>14}  {'ruptures (s)':>14}  {'ratio':>8}")
            print(f"    {'-'*10}  {'-'*14}  {'-'*14}  {'-'*8}")
            for row in dim_rows.iter_rows(named=True):
                marker = "  <-- skchange faster" if row["ratio"] > 1 else ""
                print(
                    f"    {row['n_samples']:>10}  {row['sk_s']:>14.4f}  "
                    f"{row['rpt_s']:>14.4f}  {row['ratio']:>8.2f}x{marker}"
                )


def main() -> None:
    df = pl.read_parquet(RESULTS_PATH).filter(pl.col("include_fit"))
    available_dims = sorted(df["data_dimension"].unique().to_list())
    dimensions = [d for d in DIMENSIONS if d in available_dims]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = build_figure(df, dimensions)
    fig.write_html(OUTPUT_PATH)
    fig.write_image(OUTPUT_PATH_PDF)
    print(f"Figure written to {OUTPUT_PATH} and {OUTPUT_PATH_PDF}")
    fig.show()

    print_relative_speed(df, dimensions)


if __name__ == "__main__":
    main()
