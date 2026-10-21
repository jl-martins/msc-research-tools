"""Tests of the source index and the source checks"""

import logging
import re
import shutil
from pathlib import Path

import pytest
from conftest import STREAM_LINE, STREAMS_ONLY_FILE, write_file, write_jar

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.config import BenchmarkConfig, SourcesConfig, load_config
from dacapo_hot_loops.sources import (
    AmbiguousClassesError,
    SourceIndex,
    check_sources,
    index_sources,
    read_jar_version,
)

SOURCES = SourcesConfig(
    excluded_dirs=["test", "target"],
    kept_dirs=[Path("target/generated-sources")],
)


def index_files(
    benchmarks_dir: Path,
    sources_by_path: dict[str, str],
    excludes: list[str] | None = None,
    java_major: int = 21,
) -> SourceIndex:
    """Write the sources under one root and index them"""
    for relative_path, source in sources_by_path.items():
        write_file(benchmarks_dir / "root" / relative_path, source)

    benchmark = BenchmarkConfig(roots=[Path("root")], excludes=excludes or [])
    return index_sources(benchmarks_dir, benchmark, SOURCES, java_major)


def test_test_dirs_are_excluded(tmp_path):
    index = index_files(tmp_path, {
        "src/main/java/demo/Parser.java": "package demo;",
        "src/test/java/demo/ParserTest.java": "package demo;",
    })  # fmt: skip
    assert index == {"demo.Parser": Path("root/src/main/java/demo/Parser.java")}


def test_generated_sources_are_kept(tmp_path):
    index = index_files(tmp_path, {
        "target/generated-sources/javacc/demo/Parser.java": "package demo;",
        "target/classes/demo/Stale.java": "package demo;",
    })  # fmt: skip
    assert list(index) == ["demo.Parser"]


def test_glob_excludes_apply(tmp_path):
    index = index_files(
        tmp_path,
        {
            "jetty-8/src/demo/Server.java": "package demo;",
            "jetty-9/src/demo/Server.java": "package demo;",
        },
        excludes=["jetty-8/**"],
    )
    assert index == {"demo.Server": Path("root/jetty-9/src/demo/Server.java")}


@pytest.mark.parametrize(
    ("java_major", "selected_dir"),
    [(21, "java21"), (20, "java20")],
)
def test_newest_usable_release_is_kept(tmp_path, java_major, selected_dir):
    sources_by_path = {
        f"core/src/{release_dir}/demo/Input.java": "package demo;"
        for release_dir in ("java19", "java20", "java21", "java22")
    }
    index = index_files(tmp_path, sources_by_path, java_major=java_major)
    selected_file = Path(f"root/core/src/{selected_dir}/demo/Input.java")
    assert index == {"demo.Input": selected_file}


def test_release_replaces_its_base_copy(tmp_path):
    index = index_files(tmp_path, {
        "h2/src/main/demo/Bits.java": "package demo;",
        "h2/src/java9/src/demo/Bits.java": "package demo;",
    })  # fmt: skip
    assert index == {"demo.Bits": Path("root/h2/src/java9/src/demo/Bits.java")}


def test_release_keeps_other_module_ambiguous(tmp_path):
    with pytest.raises(AmbiguousClassesError, match=r"demo\.Bits"):
        index_files(tmp_path, {
            "h2/src/main/demo/Bits.java": "package demo;",
            "h2/src/java9/src/demo/Bits.java": "package demo;",
            "derby/src/main/demo/Bits.java": "package demo;",
        })  # fmt: skip


def test_duplicate_class_is_ambiguous(tmp_path):
    with pytest.raises(AmbiguousClassesError, match=r"demo\.Parser: .*a/src.*b/src"):
        index_files(tmp_path, {
            "a/src/demo/Parser.java": "package demo;",
            "b/src/demo/Parser.java": "package demo;",
        })  # fmt: skip


def test_file_without_package_is_in_default_package(tmp_path):
    index = index_files(tmp_path, {"src/Main.java": "class Main {}"})
    assert list(index) == ["Main"]


def test_package_follows_comments(tmp_path):
    source = "/* package fake; */\n// package fake;\npackage demo;\n"
    index = index_files(tmp_path, {"src/demo/Main.java": source})
    assert list(index) == ["demo.Main"]


def test_package_info_is_skipped(tmp_path):
    index = index_files(tmp_path, {
        "src/demo/package-info.java": "package demo;",
        "src/demo/Main.java": "package demo;",
    })  # fmt: skip
    assert list(index) == ["demo.Main"]


def test_folded_manifest_line_is_joined(tmp_path):
    jar = tmp_path / "dacapo.jar"
    write_jar(jar, "Implementation-Version: 23.11-MR2-chopin-jlm-lo\r\n ops\r\n")
    assert read_jar_version(jar) == "23.11-MR2-chopin-jlm-loops"


def test_missing_jar_fails(tmp_path):
    with pytest.raises(HotLoopsError, match="Could not read the manifest"):
        read_jar_version(tmp_path / "dacapo.jar")


def test_loops_build_passes(study_dir):
    config = load_config(Path("study.toml"))
    assert check_sources(config, "loops") == {"app": 1, "other": 1}


def test_indexed_classes_are_logged_per_benchmark(study_dir, caplog):
    caplog.set_level(logging.INFO)
    config = load_config(Path("study.toml"))
    check_sources(config, "loops")
    indexed = [m for m in caplog.messages if m.startswith("Indexed")]
    assert indexed == ["Indexed 1 class(es) of app", "Indexed 1 class(es) of other"]


def test_indented_stream_line_fails(study_dir):
    main_file = Path("benchmarks/app/src/main/java/demo/Main.java")
    indented_line = f"        {STREAM_LINE}\n"
    write_file(main_file, main_file.read_text() + indented_line)
    config = load_config(Path("study.toml"))
    with pytest.raises(HotLoopsError, match="contains"):
        check_sources(config, "loops")


def test_missing_line_marker_file_fails(study_dir):
    Path("benchmarks/app/src/main/java/demo/Main.java").unlink()
    config = load_config(Path("study.toml"))
    with pytest.raises(HotLoopsError, match="Main.java' does not exist"):
        check_sources(config, "loops")


def test_streams_only_file_fails(study_dir):
    write_file(Path("benchmarks/app/src/main/java/demo/IntStreamExtensions.java"), "")
    config = load_config(Path("study.toml"))
    with pytest.raises(HotLoopsError, match="Streams-only file"):
        check_sources(config, "loops")


def test_streams_jar_fails(study_dir):
    write_jar(Path("benchmarks/dacapo.jar"), "Implementation-Version: 23.11-streams\n")
    config = load_config(Path("study.toml"))
    with pytest.raises(HotLoopsError, match="Jar version is '23.11-streams'"):
        check_sources(config, "loops")


def test_root_without_sources_fails(study_dir):
    Path("benchmarks/other/src/demo/Other.java").unlink()
    config = load_config(Path("study.toml"))
    with pytest.raises(HotLoopsError, match="contains no .java files"):
        check_sources(config, "loops")


def test_streams_build_passes(streams_study):
    config = load_config(Path("study.toml"))
    # The streams-only file adds a class to app
    assert check_sources(config, "streams") == {"app": 2, "other": 1}


def test_streams_build_without_line_marker_fails(streams_study):
    shutil.copy(
        "benchmarks/app/src/main/java/demo/Main.java",
        "streams-benchmarks/app/src/main/java/demo/Main.java",
    )
    config = load_config(Path("study.toml"))
    msg = f"Main.java' lacks {STREAM_LINE!r}"
    with pytest.raises(HotLoopsError, match=re.escape(msg)):
        check_sources(config, "streams")


def test_streams_build_without_streams_only_file_fails(streams_study):
    STREAMS_ONLY_FILE.unlink()
    config = load_config(Path("study.toml"))
    with pytest.raises(HotLoopsError, match="IntStreamExtensions.java' does not exist"):
        check_sources(config, "streams")
