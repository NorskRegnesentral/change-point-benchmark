# %%
"""Build compact null-case runtime figures for the short paper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from change_bench.paths import find_repo_root
from change_bench.plotting import relative_speed_frame

PROJECT_DIR = find_repo_root(Path(__file__))
RESULTS_DIR = PROJECT_DIR / "results" / "paper"
FIGURES_DIR = PROJECT_DIR / "figures" / "paper"
FIGURE_DATE = date.today().isoformat()
SKCHANGE_VERSION = version("skchange")
RUPTURES_VERSION = version("ruptures")
METRIC_COL = "ski_jump_mean_s"

PACKAGE_COLORS = {"skchange": "#1f77b4", "ruptures": "#ff7f0e"}
SEARCH_STYLES = {
    "pelt": ("solid", "circle", "#2ca02c", "PELT"),
    "moving_window": ("dash", "square-open", "#9467bd", "Moving window"),
    "binseg": ("dot", "diamond", "#8c564b", "Binary segmentation"),
}


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    result_prefix: str
    dimension: int
    algorithms: dict[str, str]


PANELS = [
    PanelSpec(
        key="l2-cusum",
        title="L2Cost / CUSUM (p=1)",
        result_prefix="change-in-mean-benchmark",
        dimension=1,
        algorithms={
            "pelt": "pelt_l2",
            "moving_window": "moving_window_l2",
            "binseg": "binseg_l2_cusum",
        },
    ),
    PanelSpec(
        key="rank",
        title="RankCost / RankScore (p=10)",
        result_prefix="rank-score-benchmark",
        dimension=10,
        algorithms={
            "pelt": "pelt_rank",
            "moving_window": "moving_window_rank",
            "binseg": "binseg_rank",
        },
    ),
    # PanelSpec(
    #     key="linear-trend",
    #     title="Continuous trend score (p=1)",
    #     result_prefix="continuous-linear-trend-benchmark",
    #     dimension=1,
    #     algorithms={
    #         "moving_window": "moving_window_continuous_linear_trend",
    #         "binseg": "binseg_continuous_linear_trend",
    #     },
    # ),
]


def _latest_result(prefix: str) -> Path:
    pattern = (
        f"{prefix}_*_skchange-{SKCHANGE_VERSION}_ruptures-{RUPTURES_VERSION}.parquet"
    )
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No benchmark result matches {RESULTS_DIR / pattern}")
    return matches[-1]


def load_panel_frame(panel: PanelSpec) -> pl.DataFrame:
    """Load and validate the benchmark rows required by one panel."""
    algorithms = set(panel.algorithms.values())
    frame = (
        pl.read_parquet(_latest_result(panel.result_prefix))
        .filter(
            pl.col("include_fit")
            & (pl.col("data_dimension") == panel.dimension)
            & pl.col("cpd_algorithm").is_in(algorithms)
        )
        .group_by("package", "cpd_algorithm", "data_dimension", "n_samples")
        .agg(pl.col(METRIC_COL).min())
    )
    actual_algorithms = set(frame["cpd_algorithm"].to_list())
    if actual_algorithms != algorithms:
        missing = sorted(algorithms - actual_algorithms)
        raise ValueError(f"{panel.title} is missing algorithms: {missing}")
    if set(frame["package"].to_list()) != set(PACKAGE_COLORS):
        raise ValueError(f"{panel.title} must contain skchange and ruptures rows")
    return frame


def build_absolute_figure(
    panel_frames: list[tuple[PanelSpec, pl.DataFrame]],
) -> go.Figure:
    """Build compact absolute-runtime panels."""
    figure = make_subplots(
        rows=1,
        cols=len(panel_frames),
        subplot_titles=[panel.title for panel, _ in panel_frames],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    shown: set[str] = set()
    for column, (panel, frame) in enumerate(panel_frames, start=1):
        for search, algorithm in panel.algorithms.items():
            dash, symbol, _, search_label = SEARCH_STYLES[search]
            for package, color in PACKAGE_COLORS.items():
                line = frame.filter(
                    (pl.col("cpd_algorithm") == algorithm)
                    & (pl.col("package") == package)
                ).sort("n_samples")
                legend_name = f"{package}: {search_label}"
                figure.add_trace(
                    go.Scatter(
                        x=line["n_samples"],
                        y=line[METRIC_COL],
                        mode="lines+markers",
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=legend_name not in shown,
                        line=dict(color=color, dash=dash, width=1.8),
                        marker=dict(color=color, symbol=symbol, size=6),
                        hovertemplate=(
                            f"{legend_name}<br>n=%{{x}}<br>runtime=%{{y:.3g}} s"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
                shown.add(legend_name)
        figure.update_xaxes(type="log", title_text="n", row=1, col=column)
        figure.update_yaxes(type="log", row=1, col=column)

    figure.update_yaxes(title_text="runtime (s)", row=1, col=1)
    _apply_compact_layout(figure, len(panel_frames))
    return figure


def build_relative_figure(
    panel_frames: list[tuple[PanelSpec, pl.DataFrame]],
) -> go.Figure:
    """Build compact ruptures/skchange runtime-ratio panels."""
    figure = make_subplots(
        rows=1,
        cols=len(panel_frames),
        subplot_titles=[panel.title for panel, _ in panel_frames],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    shown: set[str] = set()
    for column, (panel, frame) in enumerate(panel_frames, start=1):
        ratios = relative_speed_frame(
            frame,
            ["cpd_algorithm", "data_dimension", "n_samples"],
            METRIC_COL,
        )
        for search, algorithm in panel.algorithms.items():
            dash, symbol, color, search_label = SEARCH_STYLES[search]
            line = ratios.filter(pl.col("cpd_algorithm") == algorithm).sort("n_samples")
            figure.add_trace(
                go.Scatter(
                    x=line["n_samples"],
                    y=line["relative_speed"],
                    mode="lines+markers",
                    name=search_label,
                    legendgroup=search,
                    showlegend=search not in shown,
                    line=dict(color=color, dash=dash, width=1.8),
                    marker=dict(color=color, symbol=symbol, size=6),
                    hovertemplate=(
                        f"{search_label}<br>n=%{{x}}<br>ruptures/skchange=%{{y:.2f}}x"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            shown.add(search)
        figure.update_xaxes(type="log", title_text="n", row=1, col=column)
        figure.update_yaxes(type="log", row=1, col=column)

    figure.add_hline(y=1, line=dict(color="#666666", dash="dash", width=1))
    figure.update_yaxes(title_text="runtime ratio", row=1, col=1)
    _apply_compact_layout(figure, len(panel_frames))
    return figure


def _apply_compact_layout(figure: go.Figure, n_panels: int) -> None:
    width = 420 if n_panels == 1 else 1080
    figure.update_layout(
        width=width,
        height=340,
        template="plotly_white",
        font=dict(size=11),
        margin=dict(l=58, r=18, t=42, b=88),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
            font=dict(size=9),
        ),
    )
    figure.update_annotations(font=dict(size=11))


def _write_figure(figure: go.Figure, stem: str) -> None:
    html_path = FIGURES_DIR / f"{stem}-{FIGURE_DATE}.html"
    pdf_path = html_path.with_suffix(".pdf")
    figure.write_html(html_path, include_plotlyjs="cdn")
    figure.write_image(pdf_path)
    print(f"Figure written to {html_path} and {pdf_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel_frames = [(panel, load_panel_frame(panel)) for panel in PANELS]

    for panel, frame in panel_frames:
        _write_figure(build_absolute_figure([(panel, frame)]), f"compact-{panel.key}")
        _write_figure(
            build_relative_figure([(panel, frame)]),
            f"compact-{panel.key}-relative",
        )

    _write_figure(build_absolute_figure(panel_frames), "compact-score-benchmarks")
    _write_figure(
        build_relative_figure(panel_frames), "compact-score-benchmarks-relative"
    )


if __name__ == "__main__":
    main()

# %%
