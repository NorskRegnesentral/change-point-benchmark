"""Tests for paper benchmark change-count verification."""

from change_bench.benchmark_verification import (
    VerificationKey,
    VerificationResult,
    calibrate_penalties,
    count_changes,
)
from change_bench.benchmarks.registry import Pair, collect_cases


def test_count_changes_removes_only_ruptures_endpoint():
    assert count_changes([25, 100], package="ruptures", n_samples=100) == 1
    assert count_changes([25], package="skchange", n_samples=100) == 1


def test_verification_requires_both_counts_to_be_zero():
    key = VerificationKey("pelt_l2", "null_normal", 100, 1, 1, 10.0)

    assert VerificationResult(key, 0, 0).passes
    assert VerificationResult(key, 1, 1).counts_match
    assert not VerificationResult(key, 1, 1).passes
    assert not VerificationResult(key, 0, 1).passes


def test_calibrate_penalties_applies_margin_to_sufficient_penalty():
    cases = collect_cases(
        pairs=[Pair.PELT_L2],
        n_samples_list=[100],
        dimensions=[1],
    )

    calibrations = calibrate_penalties(cases, margin=1.5)

    assert len(calibrations) == 1
    assert calibrations[0].cpd_algorithm == "pelt_l2"
    assert calibrations[0].selected_penalty == (
        1.5 * calibrations[0].sufficient_penalty
    )
