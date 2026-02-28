from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return repository root based on package file location."""
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    """Return the default results directory."""
    return repo_root() / "results"


def reports_dir() -> Path:
    """Return the default reports directory."""
    return repo_root() / "reports"
