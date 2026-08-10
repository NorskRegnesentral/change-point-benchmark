"""Tests for shared benchmark plot data preparation."""

from __future__ import annotations

import polars as pl
import pytest

from change_bench.plotting import relative_speed_frame


def test_relative_speed_joins_unique_package_rows():
    frame = pl.DataFrame(
        {
            "package": ["skchange", "ruptures"],
            "algorithm": ["pelt", "pelt"],
            "n_samples": [100, 100],
            "min_s": [2.0, 6.0],
        }
    )

    result = relative_speed_frame(frame, ["algorithm", "n_samples"])

    assert result.to_dicts() == [
        {
            "algorithm": "pelt",
            "n_samples": 100,
            "skchange_s": 2.0,
            "ruptures_s": 6.0,
            "relative_speed": 3.0,
        }
    ]


def test_relative_speed_omits_one_sided_cases():
    frame = pl.DataFrame(
        {
            "package": ["skchange", "skchange", "ruptures"],
            "algorithm": ["esac", "pelt", "pelt"],
            "min_s": [1.0, 2.0, 6.0],
        }
    )

    result = relative_speed_frame(frame, ["algorithm"])

    assert result["algorithm"].to_list() == ["pelt"]


def test_relative_speed_rejects_duplicate_package_keys():
    frame = pl.DataFrame(
        {
            "package": ["skchange", "skchange", "ruptures"],
            "algorithm": ["pelt", "pelt", "pelt"],
            "min_s": [2.0, 1.0, 6.0],
        }
    )

    with pytest.raises(ValueError, match="one row per package"):
        relative_speed_frame(frame, ["algorithm"])
