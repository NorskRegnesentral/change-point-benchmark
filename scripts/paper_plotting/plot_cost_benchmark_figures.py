# %%
"""Per-cost comparison figures against ruptures.

One figure per cost, sized like the compact figures: absolute runtime on the
left, ruptures/skchange runtime ratio on the right, with a horizontal legend
below the panels.

Run with::

    uv run python scripts/paper_plotting/plot_cost_benchmark_figures.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from change_bench.paths import find_repo_root, latest_result_path
from change_bench.plotting import (
    PanelSpec,
    build_cost_comparison_figure,
    load_panel_frame,
    write_figure,
)

PROJECT_DIR = find_repo_root(Path(__file__))
FIGURES_DIR = PROJECT_DIR / "figures" / "paper" / "cost"
FIGURE_DATE = date.today().isoformat()

PANELS = [
    PanelSpec(
        key="l1-change-in-mean",
        title="L1Cost, 1 feature",
        result_prefix="change-in-mean-l1-benchmark",
        dimension=1,
        algorithms={
            "pelt": "pelt_l1",
            "moving_window": "moving_window_l1",
            "binseg": "binseg_l1",
        },
    ),
]


def main() -> None:
    for panel in PANELS:
        panel_dir = FIGURES_DIR / panel.key
        panel_dir.mkdir(parents=True, exist_ok=True)
        frame = load_panel_frame(panel, latest_result_path(panel.result_prefix))
        figure = build_cost_comparison_figure(panel, frame)
        write_figure(figure, panel_dir, f"cost-{panel.key}", FIGURE_DATE)


if __name__ == "__main__":
    main()

# %%
