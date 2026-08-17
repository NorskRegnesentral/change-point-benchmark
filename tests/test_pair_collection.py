"""Tests for two-sided and one-sided benchmark case collection."""

from __future__ import annotations

import pytest

from change_bench.benchmarks.comparison_pairs._common import PairConfig
from change_bench.benchmarks.registry import Pair, collect_cases


def test_existing_pair_collects_both_packages():
    cases = collect_cases(
        pairs=[Pair.PELT_L2],
        n_samples_list=[100],
        dimensions=[5],
    )

    assert {case.package for case in cases} == {"skchange", "ruptures"}
    assert {case.penalty for case in cases} == {100.0}


@pytest.mark.parametrize(
    "pair",
    [Pair.PELT_L1, Pair.MOVING_WINDOW_L1, Pair.BINSEG_L1],
)
def test_l1_pair_collects_both_packages(pair: Pair):
    cases = collect_cases(
        pairs=[pair],
        n_samples_list=[100],
        dimensions=[1],
    )

    assert {case.package for case in cases} == {"skchange", "ruptures"}
    assert {case.cpd_algorithm for case in cases} == {pair.value}


def test_pelt_rank_enforces_cost_minimum_segment_length():
    cases = collect_cases(
        pairs=[Pair.PELT_RANK],
        n_samples_list=[100],
        dimensions=[5],
        min_segment_length=1,
    )

    assert {case.min_segment_length for case in cases} == {2}


def test_continuous_linear_trend_pairs_are_univariate_and_two_sided():
    pairs = [
        Pair.MOVING_WINDOW_CONTINUOUS_LINEAR_TREND,
        Pair.BINSEG_CONTINUOUS_LINEAR_TREND,
    ]
    cases = collect_cases(
        pairs=pairs,
        n_samples_list=[100],
        dimensions=[1, 3],
        min_segment_length=1,
    )

    assert len(cases) == 4
    assert {case.package for case in cases} == {"skchange", "ruptures"}
    assert {case.cpd_algorithm for case in cases} == {pair.value for pair in pairs}
    assert {case.data_dimension for case in cases} == {1}
    assert {case.min_segment_length for case in cases} == {3}


@pytest.mark.parametrize(
    "pair",
    [Pair.PELT_LINREG, Pair.MOVING_WINDOW_LINREG, Pair.BINSEG_LINREG],
)
def test_linreg_pairs_require_response_and_predictor_columns(pair: Pair):
    cases = collect_cases(
        pairs=[pair],
        n_samples_list=[100],
        dimensions=[1, 2],
        min_segment_length=1,
    )

    assert len(cases) == 2
    assert {case.package for case in cases} == {"skchange", "ruptures"}
    assert {case.cpd_algorithm for case in cases} == {pair.value}
    assert {case.data_dimension for case in cases} == {2}
    assert {case.min_segment_length for case in cases} == {2}


@pytest.mark.parametrize(
    "pair",
    [Pair.MOVING_WINDOW_ESAC, Pair.BINSEG_ESAC],
)
def test_esac_pair_collects_only_skchange(pair: Pair):
    cases = collect_cases(
        pairs=[pair],
        n_samples_list=[100],
        dimensions=[5],
    )

    assert len(cases) == 1
    assert cases[0].package == "skchange"


def test_package_filter_handles_one_sided_pair():
    common = {
        "pairs": [Pair.MOVING_WINDOW_ESAC],
        "n_samples_list": [100],
        "dimensions": [5],
    }

    assert len(collect_cases(packages=["skchange"], **common)) == 1
    assert collect_cases(packages=["ruptures"], **common) == []


def test_pair_config_requires_at_least_one_side():
    with pytest.raises(ValueError, match="at least one benchmark side"):
        PairConfig(pair_name="empty")


def test_multivariate_dimension_category_matrix():
    dimensions = [5, 10, 50, 100, 500]
    cases = collect_cases(
        categories=["multivariate_dimension"],
        n_samples_list=[2000],
        dimensions=dimensions,
    )

    assert len(cases) == 100
    assert {case.data_dimension for case in cases} == set(dimensions)
    esac_cases = [case for case in cases if "esac" in case.cpd_algorithm]
    assert len(esac_cases) == 10
    assert {case.package for case in esac_cases} == {"skchange"}
