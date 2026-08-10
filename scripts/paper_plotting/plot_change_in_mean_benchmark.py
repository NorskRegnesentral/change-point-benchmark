# %%
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

from change_bench.plotting import relative_speed_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = (
    PROJECT_DIR
    / "results"
    / "change-in-mean-benchmark_2026-08-10_skchange-0.16.0_ruptures-1.1.10.parquet"
)
# RESULTS_PATH = PROJECT_DIR / "results" / "old-change-in-mean-benchmark.parquet"
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUT_PATH = FIGURES_DIR / "change-in-mean-benchmark.html"
OUTPUT_PATH_P1 = FIGURES_DIR / "change-in-mean-benchmark-p1.html"
RELATIVE_OUTPUT_PATH = FIGURES_DIR / "change-in-mean-benchmark-relative.html"
OUTPUT_PATH_PDF = OUTPUT_PATH.with_suffix(".pdf")
OUTPUT_PATH_P1_PDF = OUTPUT_PATH_P1.with_suffix(".pdf")
RELATIVE_OUTPUT_PATH_PDF = RELATIVE_OUTPUT_PATH.with_suffix(".pdf")

DIMENSIONS: list[int] = [1]
USE_SKI_JUMP_AVERAGE: bool = True

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

ALGORITHM_COLORS: dict[str, str] = {
    "pelt_l2": "#2ca02c",
    "moving_window_l2": "#9467bd",
    "binseg_l2_cusum": "#8c564b",
}

METRIC_COL = "ski_jump_mean_s" if USE_SKI_JUMP_AVERAGE else "min_s"
METRIC_LABEL = "ski-jump average" if USE_SKI_JUMP_AVERAGE else "minimum"


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
            showticklabels=True,
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

    # Legend columns live in the right margin; x positions are normalized
    # to the plot-area width, so convert the pixel offsets accordingly.
    margin_l = 80
    margin_r = 460
    axes_width = max(420 * n_cols, 560)
    width = axes_width + margin_l + margin_r
    x_labels = 1 + 15 / axes_width
    x_skchange = 1 + (15 + 170) / axes_width
    x_ruptures = x_skchange + 130 / axes_width

    fig.update_layout(
        title=(
            "Change-in-mean benchmark (L2 cost): runtime vs. sample size"
            "<br><sup>Legend column = package, line style + marker = search"
            f" algorithm. {METRIC_LABEL.capitalize()} runtime across N runs,"
            " fit+predict.</sup>"
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


def build_relative_speed_figure(df: pl.DataFrame, dimensions: list[int]) -> go.Figure:
    """Build a ruptures/skchange runtime-ratio figure."""
    ratios = relative_speed_frame(
        df, ["cpd_algorithm", "data_dimension", "n_samples"], METRIC_COL
    )
    fig = make_subplots(
        rows=1,
        cols=len(dimensions),
        subplot_titles=[f"p = {dim}" for dim in dimensions],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    for col_idx, dim in enumerate(dimensions, start=1):
        for algorithm, (dash, symbol) in ALGORITHM_STYLE.items():
            line_df = ratios.filter(
                (pl.col("data_dimension") == dim)
                & (pl.col("cpd_algorithm") == algorithm)
            )
            if line_df.is_empty():
                continue

            n_samples = line_df["n_samples"].to_list()
            speedups = line_df["relative_speed"].to_list()
            skchange_times = line_df["skchange_s"].to_list()
            ruptures_times = line_df["ruptures_s"].to_list()
            fig.add_trace(
                go.Scatter(
                    x=n_samples,
                    y=speedups,
                    mode="lines+markers",
                    name=ALGORITHM_LABELS[algorithm],
                    legendgroup=algorithm,
                    showlegend=col_idx == 1,
                    line=dict(color=ALGORITHM_COLORS[algorithm], dash=dash, width=2),
                    marker=dict(
                        color=ALGORITHM_COLORS[algorithm], symbol=symbol, size=9
                    ),
                    text=[
                        f"{ALGORITHM_LABELS[algorithm]}<br>n={n}"
                        f"<br>ruptures / skchange = {ratio:.2f}x"
                        f"<br>skchange: {sk_s:.2e} s"
                        f"<br>ruptures: {rpt_s:.2e} s"
                        for n, ratio, sk_s, rpt_s in zip(
                            n_samples, speedups, skchange_times, ruptures_times
                        )
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
        fig.update_yaxes(type="log", showticklabels=True, row=1, col=col_idx)

    fig.add_hline(y=1, line=dict(color="#666666", dash="dash", width=1))
    fig.update_yaxes(title_text="relative runtime (ruptures / skchange)", row=1, col=1)
    fig.update_layout(
        title=(
            "Change-in-mean benchmark (L2 cost): relative runtime"
            "<br><sup>Values above 1 mean skchange is faster."
            f" Ratio of {METRIC_LABEL} fit+predict times.</sup>"
        ),
        height=520,
        width=max(420 * len(dimensions), 560) + 240,
        legend=dict(title=dict(text="<b>Search algorithm</b>")),
        margin=dict(l=90, r=220, b=80),
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
    fig_all.write_image(OUTPUT_PATH_PDF)
    print(f"Figure written to {OUTPUT_PATH} and {OUTPUT_PATH_PDF}")
    fig_all.show()

    relative_fig = build_relative_speed_figure(df, dimensions)
    relative_fig.write_html(RELATIVE_OUTPUT_PATH)
    relative_fig.write_image(RELATIVE_OUTPUT_PATH_PDF)
    print(f"Figure written to {RELATIVE_OUTPUT_PATH} and {RELATIVE_OUTPUT_PATH_PDF}")
    relative_fig.show()

    if 1 in available_dims:
        fig_p1 = build_figure(df, [1])
        fig_p1.write_html(OUTPUT_PATH_P1)
        fig_p1.write_image(OUTPUT_PATH_P1_PDF)
        print(f"Figure written to {OUTPUT_PATH_P1} and {OUTPUT_PATH_P1_PDF}")
        fig_p1.show()


if __name__ == "__main__":
    main()

# %%
