"""Tests for the standalone benchmark runner."""

import pytest

from change_bench.runner import run_benchmark


def test_run_benchmark_records_untimed_ruptures_changepoint_count():
    calls = 0

    def prepare():
        return object()

    def setup(data):
        return (data,), {}

    def func(data):
        nonlocal calls
        calls += 1
        return [25, 100]

    result = run_benchmark(
        package="ruptures",
        cpd_algorithm="test",
        name="test/null",
        n_samples=100,
        n_changepoints=0,
        data_dimension=1,
        include_fit=True,
        min_segment_length=1,
        prepare=prepare,
        setup=setup,
        func=func,
        n_runs=2,
        penalty=100.0,
        in_no_change_regime=False,
    )

    assert calls == 3
    assert len(result.times) == 2
    assert result.n_detected_changepoints == 1
    assert result.as_dict()["n_detected_changepoints"] == 1
    assert result.as_dict()["penalty"] == 100.0


def test_run_benchmark_raises_on_detections_in_no_change_regime():
    with pytest.raises(RuntimeError, match="detected 1 change point"):
        run_benchmark(
            package="ruptures",
            cpd_algorithm="test",
            name="test/null",
            n_samples=100,
            n_changepoints=0,
            data_dimension=1,
            include_fit=True,
            min_segment_length=1,
            prepare=lambda: object(),
            setup=lambda data: ((data,), {}),
            func=lambda data: [25, 100],
            n_runs=2,
            penalty=100.0,
        )