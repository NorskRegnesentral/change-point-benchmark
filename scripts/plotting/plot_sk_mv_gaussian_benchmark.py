# %%
"""Analysis figure for the skchange-only MV Gaussian cost benchmark.

One subplot per data dimension, showing runtime vs. n_samples for
skchange's ``MultivariateGaussianCost`` under each supported search
algorithm (PELT, moving window, seeded binary segmentation), with and
without the cumulative covariance cache (``store_cov``). Color
distinguishes the cache setting, line dash and marker distinguish the
search algorithm.

Run with::

    uv run python scripts/plotting/plot_sk_mv_gaussian_benchmark.py
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
    PROJECT_DIR / "results" / "skchange" / "sk-mv-gaussian-benchmark.parquet"
)
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUT_PATH = FIGURES_DIR / "sk-mv-gaussian-benchmark.html"
OUTPUT_PATH_PDF = OUTPUT_PATH.with_suffix(".pdf")

DIMENSIONS: list[int] = [5, 10, 25]

#: store_cov -> color
STORE_COV_COLORS: dict[bool, str] = {
    True: "#1f77b4",  # blue
    False: "#d62728",  # red
}

STORE_COV_LABELS: dict[bool, str] = {
    True: "cov cache",
    False: "no cache",
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
    "binseg_mv_gaussian": "Seeded binseg",
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
            for store_cov, color in STORE_COV_COLORS.items():
                line_df = (
                    dim_df.filter(
                        (pl.col("cpd_algorithm") == algorithm)
                        & (pl.col("store_cov") == store_cov)
                    )
                    .group_by("n_samples")
                    .agg(pl.col(METRIC_COL).min().alias("value"))
                    .sort("n_samples")
                )
                if line_df.is_empty():
                    continue

                legend_name = (
                    f"{ALGORITHM_LABELS[algorithm]} — {STORE_COV_LABELS[store_cov]}"
                )
                show_legend = legend_name not in shown_legends
                shown_legends.add(legend_name)

                n_samples = line_df["n_samples"].to_list()
                values = line_df["value"].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=n_samples,
                        y=values,
                        mode="lines+markers",
                        name=legend_name,
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
    fig.update_layout(
        title=(
            "skchange MV Gaussian cost: runtime vs. sample size"
            "<br><sup>Color = store_cov cache setting,"
            " line style + marker = search algorithm."
            " Best of N runs, fit + predict.</sup>"
        ),
        height=520,
        width=420 * n_cols + 260,
        legend=dict(
            title=dict(text="<b>Algorithm — cache</b>"),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        margin=dict(l=80, r=220, b=80),
        template="plotly_white",
    )
    return fig


def print_cache_speedup(df: pl.DataFrame, dimensions: list[int]) -> None:
    """Print the runtime ratio no-cache / cov-cache per algorithm, p, and n."""
    print()
    print("=" * 60)
    print("Cache speedup: no cache / cov cache (min runtime)")
    print("=" * 60)

    for dim in dimensions:
        print(f"\n  Data dimension p={dim}")
        print(f"  {'-' * 56}")

        for algorithm, label in ALGORITHM_LABELS.items():
            base_df = df.filter(
                (pl.col("data_dimension") == dim)
                & (pl.col("cpd_algorithm") == algorithm)
            )
            cached = (
                base_df.filter(pl.col("store_cov"))
                .group_by("n_samples")
                .agg(pl.col(METRIC_COL).min().alias("cached_s"))
            )
            uncached = (
                base_df.filter(~pl.col("store_cov"))
                .group_by("n_samples")
                .agg(pl.col(METRIC_COL).min().alias("uncached_s"))
            )
            joined = (
                cached.join(uncached, on="n_samples")
                .with_columns(
                    (pl.col("uncached_s") / pl.col("cached_s")).alias("speedup")
                )
                .sort("n_samples")
            )
            if joined.is_empty():
                continue

            print(f"\n  {label}:")
            for row in joined.iter_rows(named=True):
                print(
                    f"    n={row['n_samples']:>6}  "
                    f"cov_cache={row['cached_s']:.4f}s  "
                    f"no_cache={row['uncached_s']:.4f}s  "
                    f"speedup={row['speedup']:.2f}x"
                )

    print()


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"No results at {RESULTS_PATH}. "
            "Run scripts/run_sk_mv_gaussian_benchmark.py first."
        )

    df = pl.read_parquet(RESULTS_PATH).filter(pl.col("include_fit"))
    # store_cov is encoded in the case name as "[store_cov=True|False]".
    df = df.with_columns(
        pl.col("name").str.contains("store_cov=True").alias("store_cov")
    )
    available_dims = df["data_dimension"].unique().to_list()
    dimensions = [d for d in DIMENSIONS if d in available_dims]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = build_figure(df, dimensions)
    fig.write_html(OUTPUT_PATH)
    fig.write_image(OUTPUT_PATH_PDF)
    print(f"Figure written to {OUTPUT_PATH} and {OUTPUT_PATH_PDF}")
    fig.show()

    print_cache_speedup(df, dimensions)


if __name__ == "__main__":
    main()

# %%
