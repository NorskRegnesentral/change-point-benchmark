# %%
"""Compare three skchange Binseg versions and ruptures against skchange v1.

Run with::

    uv run python scripts/plotting/compare_skchange_linreg_runs.py
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "varied"
VERSION_PATHS = {
    version: RESULTS_DIR / f"{version}-linear_regression.parquet"
    for version in ("v1", "v2", "v3")
}
OUTPUT_PATH = PROJECT_DIR / "figures" / "binseg-linreg-version-comparison.html"
RELATIVE_OUTPUT_PATH = (
    PROJECT_DIR / "figures" / "binseg-linreg-relative-runtime-comparison.html"
)

METRIC_COL = "min_s"
ALGORITHM = "binseg_linreg"
CASE_KEYS = [
    "cpd_algorithm",
    "n_samples",
    "data_dimension",
    "include_fit",
    "min_segment_length",
    "n_runs",
]

RUN_COLORS = {
    "v1": "#777777",
    "v2": "#1f77b4",
    "v3": "#2ca02c",
    "ruptures": "#ff7f0e",
}
RUN_DASHES = {"v1": "dash", "v2": "dot", "v3": "solid", "ruptures": "dashdot"}


def load_run(path: Path, package: str, value_name: str) -> pl.DataFrame:
    """Load one package's Binseg rows from a benchmark run."""
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark results not found: {path}")
    return (
        pl.read_parquet(path)
        .filter(
            (pl.col("package") == package)
            & (pl.col("cpd_algorithm") == ALGORITHM)
        )
        .select(*CASE_KEYS, pl.col(METRIC_COL).alias(value_name))
    )


def main() -> None:
    runs = {
        version: load_run(path, "skchange", f"{version}_s")
        for version, path in VERSION_PATHS.items()
    }
    ruptures = load_run(VERSION_PATHS["v1"], "ruptures", "ruptures_s")

    comparison = runs["v1"]
    for version in ("v2", "v3"):
        comparison = comparison.join(
            runs[version], on=CASE_KEYS, how="inner", validate="1:1"
        )
    comparison = comparison.join(
        ruptures, on=CASE_KEYS, how="inner", validate="1:1"
    )

    expected_rows = {len(run) for run in (*runs.values(), ruptures)}
    if len(expected_rows) != 1 or len(comparison) != expected_rows.pop():
        raise ValueError("The result files do not contain the same Binseg cases")

    comparison = comparison.with_columns(
        (pl.col("v2_s") / pl.col("v1_s")).alias("v2_v1_ratio"),
        (pl.col("v3_s") / pl.col("v1_s")).alias("v3_v1_ratio"),
        (pl.col("ruptures_s") / pl.col("v1_s")).alias("ruptures_v1_ratio"),
    )

    comparison = comparison.sort("n_samples")
    n_samples = comparison["n_samples"].to_list()
    fig = go.Figure()
    for run in ("v1", "v2", "v3", "ruptures"):
        values = comparison[f"{run}_s"].to_list()
        fig.add_trace(
            go.Scatter(
                x=n_samples,
                y=values,
                mode="lines+markers",
                name=run,
                line=dict(color=RUN_COLORS[run], dash=RUN_DASHES[run]),
                marker=dict(color=RUN_COLORS[run]),
                text=[
                    f"{run}<br>n={n}<br>{value:.4g} s"
                    for n, value in zip(n_samples, values)
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Binseg linear-regression runtime: skchange versions and ruptures",
        template="plotly_white",
        xaxis=dict(title="n samples", type="log"),
        yaxis=dict(title="min runtime (s)", type="log"),
        height=500,
        width=700,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_PATH)

    relative_fig = go.Figure()
    ratio_columns = {
        "v1": None,
        "v2": "v2_v1_ratio",
        "v3": "v3_v1_ratio",
        "ruptures": "ruptures_v1_ratio",
    }
    for run, ratio_column in ratio_columns.items():
        ratios = (
            [1.0] * len(comparison)
            if ratio_column is None
            else comparison[ratio_column].to_list()
        )
        relative_fig.add_trace(
            go.Scatter(
                x=n_samples,
                y=ratios,
                mode="lines+markers",
                name=run,
                line=dict(color=RUN_COLORS[run], dash=RUN_DASHES[run]),
                marker=dict(color=RUN_COLORS[run]),
                text=[
                    f"{run}<br>n={n}<br>runtime / v1={ratio:.3f}"
                    for n, ratio in zip(n_samples, ratios)
                ],
                hoverinfo="text",
            )
        )
    relative_fig.add_hline(y=1.0, line_color="#444444", line_width=1)
    relative_fig.update_layout(
        title="Binseg linear-regression runtime relative to skchange v1",
        template="plotly_white",
        xaxis=dict(title="n samples", type="log"),
        yaxis=dict(title="comparison runtime / skchange v1 runtime"),
        height=500,
        width=700,
    )
    relative_fig.write_html(RELATIVE_OUTPUT_PATH)

    print("Runtime ratio: comparison / skchange v1 (<1 means faster than v1)")
    print("=" * 88)
    for result in comparison.iter_rows(named=True):
        print(
            f"  n={result['n_samples']:>5}  "
            f"v1={result['v1_s']:.6f}s  "
            f"v2={result['v2_s']:.6f}s ({result['v2_v1_ratio']:.2f}x)  "
            f"v3={result['v3_s']:.6f}s ({result['v3_v1_ratio']:.2f}x)  "
            f"ruptures={result['ruptures_s']:.6f}s "
            f"({result['ruptures_v1_ratio']:.2f}x)"
        )

    print(f"\nRuntime figure written to {OUTPUT_PATH}")
    print(f"Relative runtime figure written to {RELATIVE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# %%
