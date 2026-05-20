# change-point-benchmark

Repository for benchmarking the performance of change-point detection packages,
primarily [skchange](https://github.com/NorskRegnesentral/skchange) vs.
[ruptures](https://github.com/deepcharles/ruptures).

## Project layout

```
change-point-benchmark/
├── pyproject.toml               # uv-managed project config & dependencies
├── src/
│   └── change_bench/            # installable "change_bench" package
│       ├── datasets/
│       │   └── null_case.py     # null-case dataset generators
│       └── problems/
│           └── base.py          # BenchmarkProblem dataclass & factories
├── benchmarks/
│   ├── conftest.py              # shared pytest-benchmark fixtures
│   └── bench_null_case.py       # null-case detector benchmarks
└── tests/
    ├── test_null_datasets.py    # unit tests for dataset generation
    └── test_problems.py         # unit tests for problem definitions
```

## Requirements

* Python 3.12+
* [uv](https://docs.astral.sh/uv/) package manager

## Getting started

```bash
# sync all dependencies (including dev group)
uv sync --dev

# run the unit tests
uv run pytest tests/ -v

# run benchmarks (null-case, quick subset)
uv run pytest benchmarks/ --benchmark-only -v -k "normal_n500"

# run the full benchmark suite and save results
uv run pytest benchmarks/ --benchmark-only --benchmark-json=results.json -v
```

## Key concepts

### Dataset generation

A `NullDatasetConfig` describes a dataset with **no change points in the mean**.
Data can be drawn from any of the named distributions or from an arbitrary frozen
`scipy.stats` distribution:

```python
import numpy as np
from change_bench.datasets.null_case import NullDatasetConfig

rng = np.random.default_rng(42)

# Named distribution + scale parameter
cfg = NullDatasetConfig(n_samples=1000, distribution="normal", scale=2.0)
data = cfg.generate(rng)          # shape (1000, 1)

# Student-t with custom df
cfg_t = NullDatasetConfig(n_samples=1000, distribution="t", scale=1.0, df=3.0)

# Pass a frozen scipy distribution directly
from scipy import stats as sp_stats
frozen = sp_stats.norm(loc=0, scale=3)
cfg_custom = NullDatasetConfig(n_samples=1000, distribution=frozen)
```

Supported named distributions: `normal`, `t`, `gamma`, `laplace`, `uniform`,
`exponential`, `lognormal`.

### Problem definitions

A `BenchmarkProblem` couples a `NullDatasetConfig` with ground-truth
change-point locations (empty list for null-case problems):

```python
from change_bench.problems.base import make_null_problems

# Create a standard battery of null problems
problems = make_null_problems(
    n_samples_list=[500, 1000, 5000],
    distributions=["normal", "t", "gamma"],
    scale=1.0,
)
```

### Benchmarks

Benchmarks use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) with
`benchmark.pedantic()` so that **only** the detector fit/predict step is timed —
data generation and detector construction happen in the `setup` callable and are
excluded from measurements.

```bash
uv run pytest benchmarks/bench_null_case.py --benchmark-only -v
```
