#!/usr/bin/env python
"""Comparison figure for the multivariate change detection benchmark.

Fixed sample size, runtime vs. data dimension p. One subplot per cost
function (L2, ESAC, MV Gaussian, rank). Color distinguishes package
(skchange vs ruptures), line dash distinguishes the search algorithm
(PELT, moving window, seeded/binary segmentation).

Run with::

    uv run python scripts/plotting/plot_mv_dimension_benchmark.py
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
RESULTS_PATH = (
    PROJECT_DIR / "results" / "multivariate-change-detection-benchmark.parquet"
)
OUTPUT_PATH = PROJECT_DIR / "figures" / "mv-dimension-benchmark.html"

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

METRIC_COL = "min_s"


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

                dims = line_df["data_dimension"].to_list()
                values = line_df["value"].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=dims,
                        y=values,
                        mode="lines+markers",
                        name=legend_name,
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
            row=row_idx,
            col=col_idx,
        )

    for row_idx in range(1, n_rows + 1):
        fig.update_yaxes(title_text="runtime (s)", row=row_idx, col=1)

    n_label = ", ".join(str(n) for n in sorted(n_samples))
    fig.update_layout(
        title=(
            f"Multivariate change detection benchmark (n = {n_label}):"
            " runtime vs. dimension"
            "<br><sup>Color = package, line style + marker = search algorithm."
            " Best of N runs, fit + predict. ESAC is skchange-only.</sup>"
        ),
        height=420 * n_rows + 80,
        width=480 * n_cols,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=140),
        template="plotly_white",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_PATH)
    print(f"Figure written to {OUTPUT_PATH}")
    fig.show()


if __name__ == "__main__":
    main()
