"""Filesystem path helpers for the project."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[3]


def src_dir() -> Path:
    """Return the src/ directory."""
    return project_root() / "src"


def config_dir() -> Path:
    """Return the config/ directory."""
    return project_root() / "config"


def results_dir() -> Path:
    """Return the results/ directory (created on first access)."""
    path = project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def assets_dir() -> Path:
    """Return the assets/ directory (created on first access)."""
    path = project_root() / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path
