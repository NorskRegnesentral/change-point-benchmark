"""Utilities for verifying paired benchmark detector outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
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


@dataclass(frozen=True)
class PenaltyCalibration:
    """A sufficient and margin-adjusted penalty for one comparison pair."""

    cpd_algorithm: str
    initial_penalty: float
    sufficient_penalty: float
    selected_penalty: float


def _verification_key(
    case: BenchmarkCase, penalty: float | None = None
) -> VerificationKey:
    return VerificationKey(
        cpd_algorithm=case.cpd_algorithm,
        problem_name=case.problem_name,
        n_samples=case.n_samples,
        data_dimension=case.data_dimension,
        min_segment_length=case.min_segment_length,
        penalty=case.penalty if penalty is None else penalty,
    )


def count_changes(prediction, *, package: str, n_samples: int) -> int:
    """Count interior changes, excluding ruptures' terminal endpoint."""
    changepoints = np.asarray(prediction).reshape(-1)
    if package == "ruptures":
        changepoints = changepoints[changepoints != n_samples]
    return len(changepoints)


def verify_cases(
    cases: Sequence[BenchmarkCase],
    penalty_overrides: Mapping[str, float] | None = None,
) -> tuple[list[VerificationResult], list[VerificationKey]]:
    """Run paired cases once and return results plus one-sided skipped keys."""
    grouped: dict[VerificationKey, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        penalty = None
        if penalty_overrides is not None:
            penalty = penalty_overrides.get(case.cpd_algorithm)
        grouped[_verification_key(case, penalty)].append(case)

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
            if penalty_overrides is None or key.cpd_algorithm not in penalty_overrides:
                prediction = case.func(*args, **kwargs)
            else:
                detector, detector_data = args
                if package == "skchange":
                    detector.set_params(penalty=key.penalty)
                    detector.fit(detector_data)
                    prediction = detector.predict(detector_data)
                else:
                    detector.fit(detector_data)
                    prediction = detector.predict(pen=key.penalty)
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


def calibrate_penalties(
    cases: Sequence[BenchmarkCase],
    *,
    margin: float = 1.5,
    max_doublings: int = 10,
) -> list[PenaltyCalibration]:
    """Find sufficient all-zero penalties and apply a multiplicative margin.

    Starting from each pair's configured penalty, the candidate is doubled until
    every two-sided case for that pair detects zero changes in both packages.
    One-sided pairs and pairs without an external numeric penalty are omitted.
    """
    if margin < 1.0:
        raise ValueError("margin must be at least 1.0")

    by_algorithm: dict[str, list[BenchmarkCase]] = defaultdict(list)
    seen_cases: set[tuple] = set()
    for case in cases:
        case_key = (
            case.package,
            case.cpd_algorithm,
            case.problem_name,
            case.n_samples,
            case.data_dimension,
            case.min_segment_length,
            case.penalty,
        )
        if case_key in seen_cases:
            continue
        seen_cases.add(case_key)
        by_algorithm[case.cpd_algorithm].append(case)

    calibrations: list[PenaltyCalibration] = []
    for algorithm, algorithm_cases in by_algorithm.items():
        packages = {case.package for case in algorithm_cases}
        penalties = {case.penalty for case in algorithm_cases}
        if packages != {"skchange", "ruptures"} or None in penalties:
            continue
        if len(penalties) != 1:
            raise ValueError(f"{algorithm} has inconsistent penalties: {penalties}")

        initial_penalty = float(next(iter(penalties)))
        candidate = initial_penalty
        for _ in range(max_doublings + 1):
            results, _ = verify_cases(
                algorithm_cases,
                penalty_overrides={algorithm: candidate},
            )
            if results and all(result.passes for result in results):
                selected = candidate * margin
                selected_results, _ = verify_cases(
                    algorithm_cases,
                    penalty_overrides={algorithm: selected},
                )
                if not all(result.passes for result in selected_results):
                    raise RuntimeError(
                        f"{algorithm} stopped passing after increasing its penalty"
                    )
                calibrations.append(
                    PenaltyCalibration(
                        cpd_algorithm=algorithm,
                        initial_penalty=initial_penalty,
                        sufficient_penalty=candidate,
                        selected_penalty=selected,
                    )
                )
                break
            candidate *= 2.0
        else:
            raise RuntimeError(
                f"No sufficient penalty found for {algorithm} after "
                f"{max_doublings} doublings"
            )
    return calibrations


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