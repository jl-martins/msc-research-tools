"""Shared fixtures of the dacapo-hot-loops tests"""

import shutil
import zipfile
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
JAR_VERSION = "23.11-loops"
STREAM_LINE = "items.stream().forEach(item -> {"
# Source of the loop-based implementation in study.toml, and a worktree replacement
EXISTING_SOURCE = """
[implementations.loops.source]
kind = "existing"
dir = "benchmarks"
"""
WORKTREE_SOURCE = """
[implementations.loops.source]
kind = "worktree"
branch = "chopin-loops"
path = "worktree"
"""
STREAMS_TABLES = """
[implementations.streams]
jar_name = "dacapo.jar"
jar_version = "23.11-streams"

[implementations.streams.source]
kind = "existing"
dir = "streams-benchmarks"
"""
STREAMS_ONLY_FILE = Path(
    "streams-benchmarks/app/src/main/java/demo/IntStreamExtensions.java"
)


def write_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_executable(path: Path, text: str) -> Path:
    write_file(path, text)
    path.chmod(0o755)
    return path


def write_config_variant(config_dir: Path, old: str, new: str) -> Path:
    """Write the test config with its only occurrence of a piece of text replaced"""
    text = (DATA_DIR / "study.toml").read_text(encoding="utf-8")
    # A replacement that matches nothing would leave the config silently unchanged
    assert text.count(old) == 1, f"{old!r} must occur exactly once in study.toml"
    return write_file(config_dir / "study.toml", text.replace(old, new))


def write_jar(path: Path, manifest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", manifest)


def populate_benchmarks_dir(benchmarks_dir: Path) -> None:
    """Write the jar and the sources of a loops build with two benchmarks"""
    manifest = f"Manifest-Version: 1.0\nImplementation-Version: {JAR_VERSION}\n"
    main_source = (DATA_DIR / "Main.java").read_text(encoding="utf-8")
    write_jar(benchmarks_dir / "dacapo.jar", manifest)
    write_file(benchmarks_dir / "app/src/main/java/demo/Main.java", main_source)
    write_file(benchmarks_dir / "other/src/demo/Other.java", "package demo;\n")


@pytest.fixture
def study_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Working directory with the test config and its benchmarks directory"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATA_DIR / "study.toml", tmp_path / "study.toml")
    populate_benchmarks_dir(tmp_path / "benchmarks")
    return tmp_path


@pytest.fixture
def streams_study(study_dir: Path) -> Path:
    """Study with a stream-based build that holds every streams marker"""
    shutil.copytree("benchmarks", "streams-benchmarks")
    manifest = "Implementation-Version: 23.11-streams\n"
    write_jar(Path("streams-benchmarks/dacapo.jar"), manifest)
    main_file = Path("streams-benchmarks/app/src/main/java/demo/Main.java")
    write_file(main_file, main_file.read_text() + f"{STREAM_LINE}\n")
    write_file(STREAMS_ONLY_FILE, "package demo;\n")
    config_path = study_dir / "study.toml"
    config_path.write_text(config_path.read_text() + STREAMS_TABLES)
    return study_dir
