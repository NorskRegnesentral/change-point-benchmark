#!/usr/bin/env python
"""Comparison figures for the change-in-mean (L2) benchmark.

Produces two figures:

1. One subplot per data dimension, showing runtime vs. n_samples for all
   algorithm variants using the same L2 cost.
2. A single-panel figure with all variants for p = 1 only.

Color distinguishes package (skchange vs ruptures), line dash and marker
symbol distinguish the search algorithm.

Run with::

    uv run python scripts/plotting/plot_change_in_mean_benchmark.py
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
RESULTS_PATH = PROJECT_DIR / "results" / "change-in-mean-benchmark.parquet"
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUT_PATH = FIGURES_DIR / "change-in-mean-benchmark.html"
OUTPUT_PATH_P1 = FIGURES_DIR / "change-in-mean-benchmark-p1.html"

DIMENSIONS: list[int] = [1, 2, 5]

PACKAGE_COLORS: dict[str, str] = {
    "skchange": "#1f77b4",  # blue
    "ruptures": "#ff7f0e",  # orange
}

#: search algorithm -> (line dash, marker symbol)
ALGORITHM_STYLE: dict[str, tuple[str, str]] = {
    "pelt_l2": ("solid", "circle"),
    "moving_window_l2": ("dash", "square-open"),
    "binseg_l2_cusum": ("dot", "diamond"),
}

ALGORITHM_LABELS: dict[str, str] = {
    "pelt_l2": "PELT",
    "moving_window_l2": "Moving window",
    "binseg_l2_cusum": "Binary segmentation",
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
                        legendgroup=legend_name,
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
    for label in ALGORITHM_LABELS.values():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                legend="legend3",
                showlegend=True,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=(
            "Change-in-mean benchmark (L2 cost): runtime vs. sample size"
            "<br><sup>Legend column = package, line style + marker = search"
            " algorithm. Best of N runs, fit + predict.</sup>"
        ),
        height=620,
        width=max(420 * n_cols, 700),
        legend3=dict(
            title=dict(text="\u00a0"),
            yanchor="top",
            y=-0.2,
            xanchor="right",
            x=0.40,
            itemwidth=30,
        ),
        legend=dict(
            title=dict(
                text="<b>skchange</b>",
                font=dict(color=PACKAGE_COLORS["skchange"]),
            ),
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.44,
            itemwidth=30,
        ),
        legend2=dict(
            title=dict(
                text="<b>ruptures</b>",
                font=dict(color=PACKAGE_COLORS["ruptures"]),
            ),
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.56,
            itemwidth=30,
        ),
        margin=dict(b=200),
        template="plotly_white",
    )
    return fig


def main() -> None:
    df = pl.read_parquet(RESULTS_PATH).filter(pl.col("include_fit"))
    available_dims = df["data_dimension"].unique().to_list()
    dimensions = [d for d in DIMENSIONS if d in available_dims]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_all = build_figure(df, dimensions)
    fig_all.write_html(OUTPUT_PATH)
    print(f"Figure written to {OUTPUT_PATH}")
    fig_all.show()

    if 1 in available_dims:
        fig_p1 = build_figure(df, [1])
        fig_p1.write_html(OUTPUT_PATH_P1)
        print(f"Figure written to {OUTPUT_PATH_P1}")
        fig_p1.show()


if __name__ == "__main__":
    main()
