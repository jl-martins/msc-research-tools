"""Shared fixtures of the dacapo-hot-loops tests"""

import shutil
import zipfile
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
JAR_VERSION = "23.11-loops"
STREAM_LINE = "items.stream().forEach(item -> {"


def write_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_executable(path: Path, text: str) -> Path:
    write_file(path, text)
    path.chmod(0o755)
    return path


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
