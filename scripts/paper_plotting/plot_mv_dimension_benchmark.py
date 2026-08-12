# %%
"""Comparison figure for the multivariate change detection benchmark.

Fixed sample size, runtime vs. data dimension p. One subplot per cost
function (L2, ESAC, MV Gaussian, rank). Color distinguishes package
(skchange vs ruptures), line dash distinguishes the search algorithm
(PELT, moving window, seeded/binary segmentation).

Run with::

    uv run python scripts/paper_plotting/plot_mv_dimension_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from change_bench.paths import find_repo_root
from change_bench.plotting import relative_speed_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = find_repo_root(Path(__file__))
benchmark_date = "2026-08-11"
RESULTS_PATH = (
    PROJECT_DIR
    / "results"
    / "paper"
    / (
        f"multivariate-change-detection-benchmark_{benchmark_date}_"
        + "skchange-0.16.0_ruptures-1.1.10.parquet"
    )
)
FIGURES_DIR = PROJECT_DIR / "figures" / "paper"
OUTPUT_PATH = FIGURES_DIR / f"mv-dimension-benchmark-{benchmark_date}.html"
OUTPUT_PATH_PDF = OUTPUT_PATH.with_suffix(".pdf")
RELATIVE_OUTPUT_PATH = (
    FIGURES_DIR / f"mv-dimension-benchmark-relative-{benchmark_date}.html"
)
RELATIVE_OUTPUT_PATH_PDF = RELATIVE_OUTPUT_PATH.with_suffix(".pdf")
USE_SKI_JUMP_AVERAGE: bool = True

PACKAGE_COLORS: dict[str, str] = {
    "skchange": "#1f77b4",  # blue
    "ruptures": "#ff7f0e",  # orange
}

#: search algorithm -> (line dash, marker symbol)
SEARCH_STYLE: dict[str, tuple[str, str]] = {
    "pelt": ("solid", "circle"),
    "moving_window": ("dash", "square-open"),
    "binseg": ("dot", "diamond"),
}

SEARCH_LABELS: dict[str, str] = {
    "pelt": "PELT",
    "moving_window": "Moving window",
    "binseg": "Binary segmentation",
}

SEARCH_COLORS: dict[str, str] = {
    "pelt": "#2ca02c",
    "moving_window": "#9467bd",
    "binseg": "#8c564b",
}

#: cost -> {search algorithm -> cpd_algorithm value in the results file}
COSTS: dict[str, dict[str, str]] = {
    "L2": {
        "pelt": "pelt_l2",
        "moving_window": "moving_window_l2",
        "binseg": "binseg_l2_cusum",
    },
    "ESAC": {
        "moving_window": "moving_window_esac",
        "binseg": "binseg_esac",
    },
    "MV Gaussian": {
        "pelt": "pelt_mv_gaussian",
        "moving_window": "moving_window_mv_gaussian",
        "binseg": "binseg_mv_gaussian",
    },
    "Rank": {
        "pelt": "pelt_rank",
        "moving_window": "moving_window_rank",
        "binseg": "binseg_rank",
    },
}

METRIC_COL = "ski_jump_mean_s" if USE_SKI_JUMP_AVERAGE else "min_s"
METRIC_LABEL = "ski-jump average" if USE_SKI_JUMP_AVERAGE else "minimum"


def build_relative_speed_figure(df: pl.DataFrame, n_label: str) -> go.Figure:
    """Build ruptures/skchange runtime ratios across data dimensions."""
    ratios = relative_speed_frame(
        df, ["cpd_algorithm", "data_dimension", "n_samples"], METRIC_COL
    )
    comparable_costs = {
        cost: searches for cost, searches in COSTS.items() if cost != "ESAC"
    }
    fig = make_subplots(
        rows=1,
        cols=len(comparable_costs),
        subplot_titles=list(comparable_costs),
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    for col_idx, (cost, searches) in enumerate(comparable_costs.items(), start=1):
        for search, algorithm in searches.items():
            dash, symbol = SEARCH_STYLE[search]
            line_df = ratios.filter(pl.col("cpd_algorithm") == algorithm)
            if line_df.is_empty():
                continue

            dimensions = line_df["data_dimension"].to_list()
            speedups = line_df["relative_speed"].to_list()
            skchange_times = line_df["skchange_s"].to_list()
            ruptures_times = line_df["ruptures_s"].to_list()
            fig.add_trace(
                go.Scatter(
                    x=dimensions,
                    y=speedups,
                    mode="lines+markers",
                    name=SEARCH_LABELS[search],
                    legendgroup=search,
                    showlegend=col_idx == 1,
                    line=dict(color=SEARCH_COLORS[search], dash=dash, width=2),
                    marker=dict(color=SEARCH_COLORS[search], symbol=symbol, size=9),
                    text=[
                        f"{SEARCH_LABELS[search]} ({cost})<br>p={dimension}"
                        f"<br>ruptures / skchange = {ratio:.2f}x"
                        f"<br>skchange: {sk_s:.2e} s"
                        f"<br>ruptures: {rpt_s:.2e} s"
                        for dimension, ratio, sk_s, rpt_s in zip(
                            dimensions, speedups, skchange_times, ruptures_times
                        )
                    ],
                    hoverinfo="text",
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(
            title_text="data dimension p",
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
            f"Multivariate change detection benchmark (n = {n_label}):"
            " relative runtime"
            "<br><sup>Values above 1 mean skchange is faster."
            f" Ratio of {METRIC_LABEL} fit + predict times.</sup>"
        ),
        height=520,
        width=480 * len(comparable_costs) + 300,
        legend=dict(title=dict(text="<b>Search algorithm</b>")),
        margin=dict(l=90, r=220, b=80),
        template="plotly_white",
    )
    return fig


def main() -> None:
    df = pl.read_parquet(RESULTS_PATH).filter(pl.col("include_fit"))
    n_samples = df["n_samples"].unique().to_list()

    cost_names = list(COSTS)
    n_cols = 2
    n_rows = (len(cost_names) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=cost_names,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    shown_legends: set[str] = set()

    for subplot_idx, cost in enumerate(cost_names):
        row_idx = subplot_idx // n_cols + 1
        col_idx = subplot_idx % n_cols + 1

        for search, algorithm in COSTS[cost].items():
            dash, symbol = SEARCH_STYLE[search]

            for package, color in PACKAGE_COLORS.items():
                line_df = (
                    df.filter(
                        (pl.col("cpd_algorithm") == algorithm)
                        & (pl.col("package") == package)
                    )
                    .group_by("data_dimension")
                    .agg(pl.col(METRIC_COL).min().alias("value"))
                    .sort("data_dimension")
                )
                if line_df.is_empty():
                    continue

                legend_name = f"{package} — {SEARCH_LABELS[search]}"
                show_legend = legend_name not in shown_legends
                shown_legends.add(legend_name)
                is_skchange = package == "skchange"

                dims = line_df["data_dimension"].to_list()
                values = line_df["value"].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=dims,
                        y=values,
                        mode="lines+markers",
                        name="\u00a0",
                        legend="legend" if is_skchange else "legend2",
                        legendgroup=search,
                        showlegend=show_legend,
                        line=dict(color=color, dash=dash, width=2),
                        marker=dict(
                            color=color,
                            symbol=symbol,
                            size=9,
                            line=dict(color=color, width=1.5),
                        ),
                        text=[
                            f"{legend_name} ({cost})<br>p={p}, {v:.2e} s"
                            for p, v in zip(dims, values)
                        ],
                        hoverinfo="text",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

        fig.update_xaxes(
            title_text="data dimension p",
            type="log",
            exponentformat="power",
            row=row_idx,
            col=col_idx,
        )
        fig.update_yaxes(
            type="log",
            minor=dict(ticks="inside", showgrid=True),
            exponentformat="power",
            showexponent="all",
            showticklabels=True,
            row=row_idx,
            col=col_idx,
        )

    for row_idx in range(1, n_rows + 1):
        fig.update_yaxes(title_text="runtime (s)", row=row_idx, col=1)

    # Label-only legend column: invisible traces carry the algorithm names.
    for search, label in SEARCH_LABELS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                legend="legend3",
                legendgroup=search,
                showlegend=True,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
            )
        )

    n_label = ", ".join(str(n) for n in sorted(n_samples))

    # Legend columns live in the right margin; x positions are normalized
    # to the plot-area width, so convert the pixel offsets accordingly.
    margin_l = 80
    margin_r = 460
    axes_width = 480 * n_cols
    width = axes_width + margin_l + margin_r
    x_labels = 1 + 15 / axes_width
    x_skchange = 1 + (15 + 170) / axes_width
    x_ruptures = x_skchange + 130 / axes_width

    fig.update_layout(
        title=(
            f"Multivariate change detection benchmark (n = {n_label}):"
            " runtime vs. dimension"
            "<br><sup>Legend column = package, line style + marker = search"
            f" algorithm. {METRIC_LABEL.capitalize()} runtime across N runs,"
            " fit + predict. ESAC is"
            " skchange-only.</sup>"
        ),
        height=420 * n_rows,
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_PATH)
    fig.write_image(OUTPUT_PATH_PDF)
    print(f"Figure written to {OUTPUT_PATH} and {OUTPUT_PATH_PDF}")
    # fig.show()

    relative_fig = build_relative_speed_figure(df, n_label)
    relative_fig.write_html(RELATIVE_OUTPUT_PATH)
    relative_fig.write_image(RELATIVE_OUTPUT_PATH_PDF)
    print(f"Figure written to {RELATIVE_OUTPUT_PATH} and {RELATIVE_OUTPUT_PATH_PDF}")
    # relative_fig.show()


if __name__ == "__main__":
    main()

# %%
