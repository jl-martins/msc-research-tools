"""Tests of the config validation"""

from pathlib import Path

import pytest
from conftest import DATA_DIR, write_file

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.config import SampleConfig, load_config


def write_config_variant(tmp_path: Path, old: str, new: str) -> Path:
    """Write the test config with one piece of text replaced"""
    text = (DATA_DIR / "study.toml").read_text(encoding="utf-8")
    assert old in text
    return write_file(tmp_path / "study.toml", text.replace(old, new, 1))


def test_config_loads():
    config = load_config(DATA_DIR / "study.toml")
    assert list(config.benchmarks) == ["app", "other"]


def write_sample_config(tmp_path: Path, size: int) -> Path:
    """Write the test config with a [sample] table"""
    text = (DATA_DIR / "study.toml").read_text(encoding="utf-8")
    sample_table = f"\n[sample]\nsize = {size}\nseed = 42\n"
    return write_file(tmp_path / "study.toml", text + sample_table)


def test_sample_is_optional():
    assert load_config(DATA_DIR / "study.toml").sample is None


def test_sample_loads(tmp_path):
    sample = load_config(write_sample_config(tmp_path, 100)).sample
    assert sample == SampleConfig(size=100, seed=42)


def test_zero_sample_size_fails(tmp_path):
    with pytest.raises(HotLoopsError, match="size"):
        load_config(write_sample_config(tmp_path, 0))


def test_missing_key_fails(tmp_path):
    config_path = write_config_variant(tmp_path, 'jar_version = "23.11-loops"\n', "")
    with pytest.raises(HotLoopsError, match="jar_version"):
        load_config(config_path)


def test_unknown_key_fails(tmp_path):
    unknown_key = "java_major = 21\ncolour = 1"
    config_path = write_config_variant(tmp_path, "java_major = 21", unknown_key)
    with pytest.raises(HotLoopsError, match="colour"):
        load_config(config_path)


def test_unknown_nested_key_fails(tmp_path):
    misspelled_roots = 'root = ["other"]'
    config_path = write_config_variant(tmp_path, 'roots = ["other"]', misspelled_roots)
    with pytest.raises(HotLoopsError, match="root"):
        load_config(config_path)


def test_existing_mode_requires_dir(tmp_path):
    config_path = write_config_variant(tmp_path, 'dir = "benchmarks"\n', "")
    with pytest.raises(HotLoopsError, match="requires 'dir'"):
        load_config(config_path)


def test_worktree_mode_requires_table(tmp_path):
    worktree_mode = 'mode = "worktree"'
    config_path = write_config_variant(tmp_path, 'mode = "existing"', worktree_mode)
    with pytest.raises(HotLoopsError, match=r"requires the \[dacapo.worktree\] table"):
        load_config(config_path)


def test_home_is_expanded(tmp_path):
    config_path = write_config_variant(tmp_path, '"results"', '"~/results"')
    assert load_config(config_path).results_dir == Path.home() / "results"


def test_unknown_placeholder_fails(tmp_path):
    config_path = write_config_variant(tmp_path, "{java_major}", "{java_version}")
    with pytest.raises(HotLoopsError, match="invalid placeholder"):
        load_config(config_path)


def test_min_java_above_max_fails(tmp_path):
    valid_range = "min_java = 17\nmax_java = 23"
    min_above_max = "min_java = 24\nmax_java = 23"
    config_path = write_config_variant(tmp_path, valid_range, min_above_max)
    with pytest.raises(HotLoopsError, match="must not exceed"):
        load_config(config_path)
