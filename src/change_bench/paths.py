"""Repository-relative paths used by benchmark scripts."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward from ``start``."""
    search_from = (start or Path(__file__)).resolve()
    if search_from.is_file():
        search_from = search_from.parent

    for candidate in (search_from, *search_from.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "change_bench"
        ).is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find repository root above {search_from}; expected "
        "pyproject.toml and src/change_bench"
    )


def latest_result_path(prefix: str, start: Path | None = None) -> Path:
    """Return the newest ``{prefix}_*.parquet`` under ``results/paper``.

    Both ``results/paper`` and ``results/paper/visualized_results`` are
    searched; the date-stamped filenames make lexicographic order
    chronological.
    """
    results_dir = find_repo_root(start) / "results" / "paper"
    matches = sorted(
        (
            *results_dir.glob(f"{prefix}_*.parquet"),
            *(results_dir / "visualized_results").glob(f"{prefix}_*.parquet"),
        ),
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(
            f"No benchmark result matches {results_dir}/**/{prefix}_*.parquet"
        )
    return matches[-1]


def result_date(path: Path) -> str:
    """Extract the ``YYYY-MM-DD`` date token from a result filename."""
    return path.stem.split("_")[1]


def prepare_results_path(
    filename: str, start: Path | None = None, subdir: Path | None = None
) -> Path:
    """Return a repository-relative result path after validating its directory."""
    repo_root = find_repo_root(start)
    results_dir = repo_root / "results"
    if subdir is not None:
        results_dir = results_dir / subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {results_dir}")
    return results_dir / filename
