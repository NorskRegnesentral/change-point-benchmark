"""Tests for the standalone benchmark runner."""

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
    )

    assert calls == 3
    assert len(result.times) == 2
    assert result.n_detected_changepoints == 1
    assert result.as_dict()["n_detected_changepoints"] == 1
    assert result.as_dict()["penalty"] == 100.0