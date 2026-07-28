"""PELT + L2 cost comparison pair.

skchange: PELT(cost=L2Cost())
ruptures: KernelCPD(kernel="linear", min_size=1, jump=1)
"""

from __future__ import annotations

import ruptures as rpt
from skchange.new_api.detectors import PELT as SkchangePELT
from skchange.new_api.interval_scorers import L2Cost

from change_bench.benchmarks.comparison_pairs._common import (
    PELT_PENALTY,
    BenchmarkCase,
    PairConfig,
    build_pair_cases,
)
from change_bench.problems.base import BenchmarkProblem

_CONFIG = PairConfig(
    pair_name="pelt_l2",
    penalty=PELT_PENALTY,
    sk_name_prefix="skchange_pelt_l2",
    rpt_name_prefix="ruptures_kernelcpd_linear",
    make_sk_detector=lambda msl: SkchangePELT(
        cost=L2Cost(), penalty=PELT_PENALTY, min_segment_length=msl
    ),
    make_rpt_algo=lambda msl: rpt.KernelCPD(kernel="linear", min_size=msl, jump=1),
)


def pair_pelt_l2(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with L2/linear-kernel cost — skchange vs ruptures."""
    return build_pair_cases(
        problems, _CONFIG, include_fit=include_fit, min_segment_length=min_segment_length
    )
