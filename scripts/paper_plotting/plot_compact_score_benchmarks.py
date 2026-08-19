# %%
"""Build compact null-case runtime figures for the short paper."""

from __future__ import annotations

from datetime import date
from importlib.metadata import version
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from change_bench.paths import find_repo_root
from change_bench.plotting import (
    PanelSpec,
    add_absolute_traces,
    add_relative_traces,
    apply_compact_layout,
    load_panel_frame,
    write_figure,
)

PROJECT_DIR = find_repo_root(Path(__file__))
# RESULTS_DIR = PROJECT_DIR / "results" / "paper" / "visualized_results"
RESULTS_DIR = PROJECT_DIR / "results" / "paper"
FIGURES_DIR = PROJECT_DIR / "figures" / "paper" / "compact"
FIGURE_DATE = date.today().isoformat()
SKCHANGE_VERSION = version("skchange")
RUPTURES_VERSION = version("ruptures")

# Set to False to drop the "Skchange: PELT" line from the absolute figure.
SHOW_SKCHANGE_PELT: bool = True


PANELS = [
    PanelSpec(
        key="l2-cusum",
        title="L2Cost, 1 feature",
        result_prefix="change-in-mean-benchmark",
        dimension=1,
        algorithms={
            "pelt": "pelt_l2",
            "fpop": "fpop_l2",
            "moving_window": "moving_window_l2",
            "binseg": "binseg_l2_cusum",
        },
    ),
    PanelSpec(
        key="rank",
        title="RankCost, 10 features",
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


def build_absolute_figure(
    panel_frames: list[tuple[PanelSpec, pl.DataFrame]],
) -> go.Figure:
    """Build compact absolute-runtime panels."""
    figure = make_subplots(
        rows=1,
        cols=len(panel_frames),
        subplot_titles=[panel.title for panel, _ in panel_frames],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )
    shown: set[str] = set()
    skip = () if SHOW_SKCHANGE_PELT else (("skchange", "pelt"),)
    for column, (panel, frame) in enumerate(panel_frames, start=1):
        add_absolute_traces(figure, panel, frame, column, shown, skip=skip)

    figure.update_yaxes(title_text="Wall time (s)", row=1, col=1)
    apply_compact_layout(figure, len(panel_frames))
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
        horizontal_spacing=0.04,
    )
    shown: set[str] = set()
    for column, (panel, frame) in enumerate(panel_frames, start=1):
        add_relative_traces(figure, panel, frame, column, shown)

    figure.add_hline(y=1, line=dict(color="#666666", dash="dash", width=1))
    figure.update_yaxes(title_text="Wall-time ratio", row=1, col=1)
    apply_compact_layout(figure, len(panel_frames))
    return figure


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel_frames = [
        (panel, load_panel_frame(panel, _latest_result(panel.result_prefix)))
        for panel in PANELS
    ]

    write_figure(
        build_absolute_figure(panel_frames),
        FIGURES_DIR,
        "compact-score-benchmarks",
        FIGURE_DATE,
    )
    write_figure(
        build_relative_figure(panel_frames),
        FIGURES_DIR,
        "compact-score-benchmarks-relative",
        FIGURE_DATE,
    )


if __name__ == "__main__":
    main()

# %%

# Machine information:
# CPU: Intel Xeon Silver 4110 @ 2.10 GHz
# RAM: 49 GiB
# OS: Ubuntu 22.04.5 LTS
