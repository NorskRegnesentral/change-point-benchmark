"""Utilities for verifying paired benchmark detector outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from change_bench.benchmarks.comparison_pairs._common import BenchmarkCase
from change_bench.benchmarks.registry import collect_cases


@dataclass(frozen=True)
class VerificationKey:
    """Metadata shared by the package cases in one comparison."""

    cpd_algorithm: str
    problem_name: str
    n_samples: int
    data_dimension: int
    min_segment_length: int
    penalty: float | None


@dataclass(frozen=True)
class VerificationResult:
    """Detected change counts for one two-sided comparison."""

    key: VerificationKey
    skchange_count: int
    ruptures_count: int

    @property
    def counts_match(self) -> bool:
        """Whether both packages detected the same number of changes."""
        return self.skchange_count == self.ruptures_count

    @property
    def passes(self) -> bool:
        """Whether both packages detected no changes on the null dataset."""
        return self.skchange_count == self.ruptures_count == 0


def _verification_key(case: BenchmarkCase) -> VerificationKey:
    return VerificationKey(
        cpd_algorithm=case.cpd_algorithm,
        problem_name=case.problem_name,
        n_samples=case.n_samples,
        data_dimension=case.data_dimension,
        min_segment_length=case.min_segment_length,
        penalty=case.penalty,
    )


def count_changes(prediction, *, package: str, n_samples: int) -> int:
    """Count interior changes, excluding ruptures' terminal endpoint."""
    changepoints = np.asarray(prediction).reshape(-1)
    if package == "ruptures":
        changepoints = changepoints[changepoints != n_samples]
    return len(changepoints)


def verify_cases(
    cases: Sequence[BenchmarkCase],
) -> tuple[list[VerificationResult], list[VerificationKey]]:
    """Run paired cases once and return results plus one-sided skipped keys."""
    grouped: dict[VerificationKey, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped[_verification_key(case)].append(case)

    results: list[VerificationResult] = []
    skipped: list[VerificationKey] = []
    for key, paired_cases in grouped.items():
        by_package = {case.package: case for case in paired_cases}
        if set(by_package) != {"skchange", "ruptures"}:
            skipped.append(key)
            continue

        data = paired_cases[0].prepare()
        counts: dict[str, int] = {}
        for package in ("skchange", "ruptures"):
            case = by_package[package]
            args, kwargs = case.setup(data.copy())
            prediction = case.func(*args, **kwargs)
            counts[package] = count_changes(
                prediction,
                package=package,
                n_samples=case.n_samples,
            )

        results.append(
            VerificationResult(
                key=key,
                skchange_count=counts["skchange"],
                ruptures_count=counts["ruptures"],
            )
        )
    return results, skipped


def collect_benchmark_cases(
    module: ModuleType, max_n_samples: int | None = None
) -> list[BenchmarkCase]:
    """Collect cases using a paper benchmark module's exact configuration."""
    n_samples = getattr(module, "N_SAMPLES")
    n_samples_list = [n_samples] if isinstance(n_samples, int) else n_samples
    if max_n_samples is not None:
        n_samples_list = [n for n in n_samples_list if n <= max_n_samples]

    return collect_cases(
        pairs=module.PAIRS,
        n_samples_list=n_samples_list,
        include_fit=module.INCLUDE_FIT,
        min_segment_length=module.MIN_SEGMENT_LENGTH,
        dimensions=module.DIMENSIONS,
        distributions=module.DISTRIBUTIONS,
    )