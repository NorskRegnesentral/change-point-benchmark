#!/usr/bin/env python
"""Show duplicate cost interval requests made by SBS with LinearRegressionCost.

This uses skchange's real seeded-interval generation, SBS scoring function,
and CostChangeScore expansion. LinearRegressionCost.evaluate is instrumented
to record its inputs and return dummy costs, avoiding unnecessary OLS work.

Run with::

    uv run python scripts/explain_sbs_linreg_duplicates.py
    uv run python scripts/explain_sbs_linreg_duplicates.py --n-samples 1000
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from skchange.detectors import SeededBinarySegmentation
from skchange.detectors._seeded_binseg import _score_seeded_intervals
from skchange.interval_scorers import CostChangeScore, LinearRegressionCost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--max-interval-length", type=int, default=200)
    parser.add_argument("--growth-factor", type=float, default=1.5)
    parser.add_argument("--examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)
    predictor = rng.normal(size=args.n_samples)
    response = 2.0 * predictor + rng.normal(size=args.n_samples)
    data = np.column_stack([response, predictor])

    detector = SeededBinarySegmentation(
        change_score=CostChangeScore(LinearRegressionCost(response_col=0)),
        penalty=0.0,
        min_subinterval_length=2,
        max_interval_length=min(args.max_interval_length, args.n_samples),
        growth_factor=args.growth_factor,
    ).fit(data)

    recorded_calls: list[np.ndarray] = []

    def record_evaluate(cache: dict, interval_specs: np.ndarray) -> np.ndarray:
        specs = np.asarray(interval_specs, dtype=np.int64)
        recorded_calls.append(specs.copy())
        return np.zeros((len(specs), 1))

    detector.change_score_.cost_.evaluate = record_evaluate

    _, _, seeded_starts, _ = _score_seeded_intervals(
        change_score=detector.change_score_,
        agg_mode=detector._agg_mode,
        penalty=detector.penalty_,
        X=data,
        min_subinterval_length=detector.min_subinterval_length_,
        max_interval_length=detector.max_interval_length_,
        growth_factor=detector.growth_factor,
    )

    ### Now only do one "evaluate" call:
    # call_names = ("left  [start, split)", "right [split, end)", "full  [start, end)")
    # if len(recorded_calls) != len(call_names):
    #     raise RuntimeError(
    #         f"Expected {len(call_names)} cost.evaluate calls, got "
    #         f"{len(recorded_calls)}. CostChangeScore internals may have changed."
    #     )

    all_requests = np.concatenate(recorded_calls)
    request_counts = Counter(map(tuple, all_requests.tolist()))
    unique_count = len(request_counts)
    duplicate_count = len(all_requests) - unique_count

    print("SeededBinarySegmentation -> CostChangeScore -> LinearRegressionCost")
    print("=" * 72)
    print(f"Samples:                    {args.n_samples:>10,}")
    print(f"Seeded intervals:           {len(seeded_starts):>10,}")
    print(f"Change-score (Cost) specs:         {len(recorded_calls[0]):>10,}")
    print()
    print("CostChangeScore expands each [start, split, end) into three costs:")
    print(f"  {'cost interval':<22} {'requests':>10} {'unique':>10}")
    # for name, specs in zip(call_names, recorded_calls):
    #     unique_specs = len(np.unique(specs, axis=0))
    #     print(f"  {name:<22} {len(specs):>10,} {unique_specs:>10,}")
    print()
    print(f"Total LinReg cost requests: {len(all_requests):>10,}")
    print(f"Unique cost intervals:      {unique_count:>10,}")
    print(f"Duplicate cost requests:    {duplicate_count:>10,}")
    print(f"Potentially avoidable:      {duplicate_count / len(all_requests):>9.1%}")

    print(f"\nTop {args.examples} repeated cost interval requests:")
    print(f"  {'interval':>18}  {'times requested':>15}")
    for interval, count in request_counts.most_common(args.examples):
        print(f"  {str(interval):>18}  {count:>15,}")

    # parent_counts = Counter(map(tuple, recorded_calls[2].tolist()))
    # repeated_parent, parent_count = parent_counts.most_common(1)[0]
    # repeated_parent = tuple(map(int, repeated_parent))
    # print("\nConcrete example:")
    # print(
    #     f"  Parent interval {repeated_parent} is sent to LinearRegressionCost "
    #     f"{parent_count} times as a parent cost: once for every candidate split."
    # )
    # print(
    #     "  Its RSS is independent of the split, so all but one of those parent "
    #     "evaluations are redundant."
    # )


if __name__ == "__main__":
    main()
