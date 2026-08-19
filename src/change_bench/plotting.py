"""Shared data preparation and figure building for benchmark plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

METRIC_COL = "ski_jump_mean_s"

PACKAGE_COLORS = {"skchange": "#1f77b4", "ruptures": "#ff7f0e"}
PACKAGE_LABELS = {"skchange": "Skchange", "ruptures": "Ruptures"}
PACKAGE_SEARCH_LABELS = {
    ("skchange", "moving_window"): "MovingWindow",
    ("ruptures", "moving_window"): "Window",
    ("skchange", "binseg"): "Seeded Bin. Seg",
    ("ruptures", "binseg"): "Binseg",
}
#: search key -> (line dash, marker symbol, ratio-line color, label)
SEARCH_STYLES = {
    "pelt": ("solid", "circle", "#2ca02c", "PELT"),
    "fpop": ("dashdot", "triangle-up", "#d62728", "FPOP"),
    "moving_window": ("dash", "square-open", "#9467bd", "Moving window"),
    "binseg": ("dot", "diamond", "#8c564b", "Binary segmentation"),
}
# The triangle renders visually smaller than other symbols at equal size.
MARKER_SIZES = {"fpop": 9}
DEFAULT_MARKER_SIZE = 6


@dataclass(frozen=True)
class PanelSpec:
    """One benchmark slice: a cost/dimension and its algorithm variants."""

    key: str
    title: str
    result_prefix: str
    dimension: int
    #: search key (see SEARCH_STYLES) -> ``cpd_algorithm`` value in the results
    algorithms: dict[str, str]


def relative_speed_frame(
    df: pl.DataFrame,
    group_cols: list[str],
    metric_col: str = "min_s",
) -> pl.DataFrame:
    """Compute ruptures/skchange runtime ratios for unique matched cases."""
    key_cols = ["package", *group_cols]
    duplicates = df.group_by(key_cols).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        duplicate_keys = duplicates.select(key_cols).head(5).to_dicts()
        raise ValueError(
            "Relative-speed keys must select one row per package; "
            f"duplicate keys include {duplicate_keys}"
        )

    package_frames = {}
    for package in ("skchange", "ruptures"):
        package_frames[package] = df.filter(
            pl.col("package") == package
        ).select(
            *group_cols,
            pl.col(metric_col).alias(f"{package}_s"),
        )

    return (
        package_frames["skchange"]
        .join(package_frames["ruptures"], on=group_cols, how="inner")
        .with_columns(
            (pl.col("ruptures_s") / pl.col("skchange_s")).alias(
                "relative_speed"
            )
        )
        .sort(group_cols)
    )


def load_panel_frame(
    panel: PanelSpec, results_path: Path, metric_col: str = METRIC_COL
) -> pl.DataFrame:
    """Load and validate the benchmark rows required by one panel."""
    algorithms = set(panel.algorithms.values())
    frame = (
        pl.read_parquet(results_path)
        .filter(
            pl.col("include_fit")
            & (pl.col("data_dimension") == panel.dimension)
            & pl.col("cpd_algorithm").is_in(algorithms)
        )
        .group_by("package", "cpd_algorithm", "data_dimension", "n_samples")
        .agg(pl.col(metric_col).min())
    )
    actual_algorithms = set(frame["cpd_algorithm"].to_list())
    if actual_algorithms != algorithms:
        missing = sorted(algorithms - actual_algorithms)
        raise ValueError(f"{panel.title} is missing algorithms: {missing}")
    if set(frame["package"].to_list()) != set(PACKAGE_COLORS):
        raise ValueError(f"{panel.title} must contain skchange and ruptures rows")
    return frame


def add_absolute_traces(
    figure: go.Figure,
    panel: PanelSpec,
    frame: pl.DataFrame,
    column: int,
    shown: set[str],
    metric_col: str = METRIC_COL,
    skip: tuple[tuple[str, str], ...] = (),
    package_legends: dict[str, str] | None = None,
) -> None:
    """Add absolute-runtime traces for one panel to a subplot column."""
    for search, algorithm in panel.algorithms.items():
        dash, symbol, _, search_label = SEARCH_STYLES[search]
        for package, color in PACKAGE_COLORS.items():
            if (package, search) in skip:
                continue
            line = frame.filter(
                (pl.col("cpd_algorithm") == algorithm)
                & (pl.col("package") == package)
            ).sort("n_samples")
            if line.is_empty():
                # Skchange-only algorithms (e.g. FPOP) have no ruptures rows.
                continue
            package_search_label = PACKAGE_SEARCH_LABELS.get(
                (package, search), search_label
            )
            legend_name = f"{PACKAGE_LABELS[package]}: {package_search_label}"
            figure.add_trace(
                go.Scatter(
                    x=line["n_samples"],
                    y=line[metric_col],
                    mode="lines+markers",
                    name=legend_name,
                    legend=(package_legends or {}).get(package, "legend"),
                    legendgroup=legend_name,
                    showlegend=legend_name not in shown,
                    line=dict(color=color, dash=dash, width=1.8),
                    marker=dict(
                        color=color,
                        symbol=symbol,
                        size=MARKER_SIZES.get(search, DEFAULT_MARKER_SIZE),
                    ),
                    hovertemplate=(
                        f"{legend_name}<br>Number of samples=%{{x}}"
                        "<br>Wall time=%{y:.3g} s"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            shown.add(legend_name)
    figure.update_xaxes(type="log", title_text="Number of samples", row=1, col=column)
    figure.update_yaxes(type="log", showticklabels=True, row=1, col=column)


def add_relative_traces(
    figure: go.Figure,
    panel: PanelSpec,
    frame: pl.DataFrame,
    column: int,
    shown: set[str],
    metric_col: str = METRIC_COL,
    legend_ref: str = "legend",
) -> None:
    """Add ruptures/skchange runtime-ratio traces for one panel."""
    ratios = relative_speed_frame(
        frame,
        ["cpd_algorithm", "data_dimension", "n_samples"],
        metric_col,
    )
    for search, algorithm in panel.algorithms.items():
        dash, symbol, color, search_label = SEARCH_STYLES[search]
        line = ratios.filter(pl.col("cpd_algorithm") == algorithm).sort("n_samples")
        if line.is_empty():
            # Skchange-only algorithms (e.g. FPOP) have no ruptures ratio.
            continue
        figure.add_trace(
            go.Scatter(
                x=line["n_samples"],
                y=line["relative_speed"],
                mode="lines+markers",
                name=search_label,
                legend=legend_ref,
                legendgroup=search,
                showlegend=search not in shown,
                line=dict(color=color, dash=dash, width=1.8),
                marker=dict(
                    color=color,
                    symbol=symbol,
                    size=MARKER_SIZES.get(search, DEFAULT_MARKER_SIZE),
                ),
                hovertemplate=(
                    f"{search_label}<br>n=%{{x}}<br>Ruptures/Skchange=%{{y:.2f}}x"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        shown.add(search)
    figure.update_xaxes(type="log", title_text="Number of samples", row=1, col=column)
    figure.update_yaxes(type="log", showticklabels=True, row=1, col=column)


def apply_compact_layout(
    figure: go.Figure, n_panels: int, title: str | None = None
) -> None:
    """Apply the compact paper layout with a horizontal bottom legend."""
    width = 420 if n_panels == 1 else 1080
    figure.update_layout(
        width=width,
        height=340,
        template="plotly_white",
        font=dict(size=11),
        margin=dict(l=58, r=18, t=60 if title else 42, b=88),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
            yanchor="top",
            font=dict(size=10),
            itemwidth=30,
        ),
    )
    if title is not None:
        figure.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    figure.update_annotations(font=dict(size=11))


def build_cost_comparison_figure(
    panel: PanelSpec, frame: pl.DataFrame, metric_col: str = METRIC_COL
) -> go.Figure:
    """Build a per-cost figure: absolute runtime (left), runtime ratio (right)."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Wall time", "Wall-time ratio"],
        horizontal_spacing=0.07,
    )
    shown: set[str] = set()
    add_absolute_traces(
        figure,
        panel,
        frame,
        1,
        shown,
        metric_col,
        package_legends={"skchange": "legend", "ruptures": "legend2"},
    )
    add_relative_traces(figure, panel, frame, 2, shown, metric_col, legend_ref="legend3")
    figure.add_hline(
        y=1,
        line=dict(color="#666666", dash="dash", width=1),
        row=1,  # type: ignore[arg-type]  # plotly stubs mistype row/col as str
        col=2,  # type: ignore[arg-type]
    )
    figure.update_yaxes(title_text="Wall time (s)", row=1, col=1)
    figure.update_yaxes(title_text="Ruptures / Skchange", row=1, col=2)
    apply_compact_layout(figure, 2, title=panel.title)

    # Subplot domains: left [0, 0.465], right [0.535, 1].
    left_center = 0.2325
    right_center = 0.7675
    # Fixed entry widths keep the skchange/ruptures rows column-aligned.
    row_legend = dict(
        orientation="h",
        xanchor="center",
        yanchor="top",
        font=dict(size=10),
        entrywidth=150,
        itemwidth=30,
    )
    figure.update_layout(
        height=365,
        margin=dict(b=115),
        legend={**row_legend, "x": left_center, "y": -0.24},
        legend2={**row_legend, "x": left_center, "y": -0.38},
        legend3=dict(
            orientation="h",
            x=right_center,
            xanchor="center",
            y=-0.24,
            yanchor="top",
            font=dict(size=10),
            itemwidth=30,
        ),
    )
    return figure


def write_figure(
    figure: go.Figure, figures_dir: Path, stem: str, figure_date: str
) -> None:
    """Write date-stamped HTML/PDF outputs plus a stable-named PNG."""
    html_path = figures_dir / f"{stem}-{figure_date}.html"
    pdf_path = html_path.with_suffix(".pdf")
    # Stable-named PNG so README image links never go stale.
    png_path = figures_dir / f"{stem}.png"
    figure.write_html(html_path, include_plotlyjs="cdn")
    figure.write_image(pdf_path)
    figure.write_image(png_path, scale=2)
    print(f"Figure written to {html_path}, {pdf_path} and {png_path}")