"""Shared data preparation for benchmark plots."""

from __future__ import annotations

import polars as pl


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