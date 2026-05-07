"""Unit tests for shared path helpers."""

from pathlib import Path

from rnn_lstm_sinusoid_demixing.shared.paths import (
    assets_dir,
    config_dir,
    project_root,
    results_dir,
    src_dir,
)


def test_project_root_is_directory() -> None:
    root = project_root()
    assert root.is_dir()


def test_project_root_contains_pyproject() -> None:
    assert (project_root() / "pyproject.toml").exists()


def test_src_dir_inside_root() -> None:
    assert src_dir().is_relative_to(project_root())


def test_config_dir_inside_root() -> None:
    assert config_dir() == project_root() / "config"


def test_results_dir_created_and_inside_root() -> None:
    path = results_dir()
    assert path.is_dir()
    assert path.is_relative_to(project_root())


def test_assets_dir_created_and_inside_root() -> None:
    path = assets_dir()
    assert path.is_dir()
    assert path.is_relative_to(project_root())


def test_all_paths_are_absolute() -> None:
    for fn in (project_root, src_dir, config_dir, results_dir, assets_dir):
        assert Path(fn()).is_absolute()
