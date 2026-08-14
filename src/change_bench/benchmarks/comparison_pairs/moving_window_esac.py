"""MovingWindow + ESACScore benchmark (skchange only)."""

from __future__ import annotations

from skchange.detectors import MovingWindow
from skchange.interval_scorers import ESACScore

from change_bench.benchmarks.comparison_pairs._common import (
    MW_BANDWIDTH,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="moving_window_esac",
    sk_name_prefix="skchange_moving_window_esac",
    make_sk_detector=lambda msl: MovingWindow(
        change_score=ESACScore(),
        bandwidth=MW_BANDWIDTH,
    ),
)


def pair_moving_window_esac(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with the inherently penalised ESAC score."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )