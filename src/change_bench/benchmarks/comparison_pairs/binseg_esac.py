"""SeededBinarySegmentation + ESACScore benchmark (skchange only)."""

from __future__ import annotations

from skchange.detectors import SeededBinarySegmentation
from skchange.interval_scorers import ESACScore

from change_bench.benchmarks.comparison_pairs._common import (
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="binseg_esac",
    sk_name_prefix="skchange_seeded_binseg_esac",
    make_sk_detector=lambda msl: SeededBinarySegmentation(
        change_score=ESACScore(),
    ),
)


def pair_binseg_esac(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Seeded binary segmentation with the inherently penalised ESAC score."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
