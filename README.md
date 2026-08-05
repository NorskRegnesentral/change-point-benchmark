# change-point-benchmark

Repository for benchmarking the performance of change-point detection packages,
primarily [skchange](https://github.com/NorskRegnesentral/skchange) vs.
[ruptures](https://github.com/deepcharles/ruptures).

## Project layout

```
change-point-benchmark/
├── pyproject.toml               # uv-managed project config & dependencies
├── scripts/
│   ├── run_benchmarks.sh        # run all benchmark categories & save results
│   └── analyse_results.py       # load results, plot & summarise
├── src/
│   └── change_bench/            # installable "change_bench" package
│       ├── cli.py               # `uv run bench` entry point
│       ├── runner.py            # timing harness (prepare/setup/func)
│       ├── datasets/
│       │   └── null_case.py     # null-case dataset generators
│       ├── problems/
│       │   └── base.py          # BenchmarkProblem dataclass & factories
│       └── benchmarks/
│           ├── registry.py      # pair registry, categories & collect_cases()
│           └── comparison_pairs/
│               ├── _common.py             # BenchmarkCase, shared constants & helpers
│               ├── pelt_l2.py             # PELT + L2 cost
│               ├── pelt_gaussian.py       # PELT + Gaussian cost (1-D only)
│               ├── pelt_poisson.py        # PELT + Poisson cost (custom ruptures BaseCost)
│               ├── pelt_linear_trend.py   # PELT + linear trend cost
│               ├── moving_window.py       # MovingWindow + CUSUM
│               ├── moving_window_l2.py    # MovingWindow + L2 (fixed bandwidth)
│               ├── moving_window_l1.py    # MovingWindow + L1 (fixed bandwidth)
│               ├── moving_window_rank.py  # MovingWindow + Rank (multivariate only)
│               └── binseg.py              # Binary Segmentation + CUSUM
└── tests/
    ├── test_null_datasets.py    # unit tests for dataset generation
    ├── test_problems.py         # unit tests for problem definitions
    └── test_poisson_cost.py     # verify custom Poisson cost matches skchange
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

# run a quick benchmark (single pair, 2 runs)
uv run bench --pairs pelt_l2 --runs 2 -o results/pelt_l2.parquet

# run all benchmarks via the helper script (writes results/*.parquet)
./scripts/run_benchmarks.sh --runs 10

# analyse results (loads & concatenates all parquet files in results/)
uv run scripts/analyse_results.py
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

Benchmarks are organised as **comparison pairs**: each pair contains one
skchange detector and its equivalent ruptures detector so that timing
differences are directly attributable to the implementation.

The CLI (`uv run bench`) times each case using a two-phase protocol:

1. `prepare()` — generates data just-in-time (not timed)
2. `setup(data)` — creates a fresh detector per run (not timed)
3. `func(det, data)` — the timed fit+predict (or predict-only) operation

skchange detectors are benchmarked via the [`skchange.new_api`](https://github.com/NorskRegnesentral/skchange/tree/main/skchange/new_api)
submodule, which provides an sklearn-compatible single-series API:

```python
from skchange.new_api.detectors import PELT
from skchange.new_api.interval_scorers import L2Cost

X = ...  # numpy array shape (n_samples, n_features)

det = PELT(cost=L2Cost())
det.fit(X)
changepoints = det.predict_changepoints(X)  # np.ndarray of indices
labels       = det.predict(X)               # dense segment labels (n_samples,)
```

```bash
# List all available benchmark cases
uv run bench --list

# Run specific pairs with multivariate data
uv run bench --pairs pelt_l2 moving_window_rank --dimensions 1 2 5 --runs 10
```

## Ruptures - Skchange comparison pairs

Costs/scores:
- CUSUM/L2Cost/rpt.CostL2
- L1Cost/rpt.CostL1
- MultivariateGaussianScore/MultivariateGaussianCost/rpt.CostNormal
- PoissonCost/rpt.CostPoisson

All combinations of the costs/score above inside the following detectors:
- PELT/rpt.Pelt
- MovingWindow/rpt.Window
- SeededBinarySegmentation/rpt.Binseg


| Pair name | ruptures | skchange |
|-----------|----------|----------|
| `pelt_l2` | `KernelCPD("linear")` | `PELT(L2Cost())` |
| `pelt_l1` | `Pelt("l1")` | `PELT(L1Cost())` |
| `pelt_gaussian` | `Pelt("normal")` | `PELT(MultivariateGaussianCost())` |
| `moving_window_l2` | `Window("l2")` | `MovingWindow(CUSUM())` |
| `moving_window_l1` | `Window("l1")` | `MovingWindow(CostChangeScore(L1Cost()))` |
| `moving_window_gaussian` | `Window("normal")` | `MovingWindow(MultivariateGaussianScore())` |
| `binseg_l2` | `Binseg("l2")` | `SeededBinarySegmentation(CUSUM())` |
| `binseg_l1` | `Binseg("l1")` | `SeededBinarySegmentation(CostChangeScore(L1Cost()))` |
| `binseg_gaussian` | `Binseg("normal")` | `SeededBinarySegmentation(MultivariateGaussianScore())` |

The running times of the following calls are recorded:
- skchange: detector.fit(X).predict_changepoints(X)
- ruptures: detector.fit_predict(X, pen=penalty), where penalty is set to the same value as the corresponding skchange detector's penalty parameter.

Otherwise, parameters are set to make the algorithms as comparable as possible, e.g. minimum segment length, window sizes, jump sizes etc.

# TODO: Generere figurene og legge inn i README. 
# + Hvordan generere data + figurene selv.

# TODO: 
Kovarians pre-compute: (X[rad-observasjoner, kolonne-variabler]).
# cov(i, j) = sum(n = i)^j X[i, :]^T * X[i, :]
# Profilere "MvGaussian" for "p=5", lavere n <= 1000.
 - Hvor stor andel av vår tid brukes i Numba vs. Python.

# Figur for utforskning:
- Alle change-in-mean varianter: Med samme kostnad (L2).
  - Får en felles oversikt, og kan sammenligne 
    forskjellige algoritmer på samme skala. Er vår 
    "moving window + L2" raskere enn deres "Pelt + L2". 
    (per dimensjon, [1, 2, 5])
  - Skille på farge [skchange, ruptures] og linjetype [alg.].


- Figur for "multivariate change detection":
  - N samples konstant (1000), øke p 
    gjennom p (1, 5, 10, 50, 100, 500)
  - Mv-kostnader:
    - L2
    - Esac (uten sammenligning)
    - MvGaussian
    - MvRank
  - Søkealgoritmer:
    - MovingWindow
    - SeededBinary
    - PELT


