"""FPOP + L2 cost comparison pair (skchange only).

skchange: FPOP(penalty=PELT_PENALTY)

FPOP is specialised to the L2 cost for univariate change-in-mean, so there
is no ruptures counterpart and ``min_segment_length`` is ignored (FPOP
always uses a minimum segment length of 1).
"""

from __future__ import annotations

from skchange.detectors import FPOP as SkchangeFPOP

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="fpop_l2",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_fpop_l2",
    make_sk_detector=lambda msl: SkchangeFPOP(penalty=PELT_PENALTY),
)


def pair_fpop_l2(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """FPOP with L2 cost — skchange only."""
    return build_pair_cases(
        problems,
        _CONFIG,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
    )
