"""Tests of the profile and import-profiles commands"""

import logging
import re
from pathlib import Path

import pytest
from conftest import DATA_DIR, write_executable, write_file

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.cli import main
from dacapo_hot_loops.config import load_config
from dacapo_hot_loops.profiler import (
    PROFILE_SUFFIXES,
    benchmark_jvm_flags,
    check_kernel_settings,
    conversion_commands,
    java_major_version,
    profile_command,
)

CONFIG = load_config(DATA_DIR / "study.toml")
APP_FLAG_RULES = CONFIG.benchmarks["app"].jvm_flags
MAX_KERNEL_SETTINGS = CONFIG.profiling.max_kernel_settings
IMPORT_ARGS = ["import-profiles", "--config", "study.toml", "--from", "benchmarks"]


def write_fake_java(path: Path, version: str) -> Path:
    version_line = f'openjdk version "{version}" 2026-04-21'
    return write_executable(path, f"#!/bin/sh\necho '{version_line}' >&2\n")


def write_kernel_settings(proc_sys_dir: Path, settings: dict[str, int]) -> None:
    for name, value in settings.items():
        write_file(proc_sys_dir / name.replace(".", "/"), f"{value}\n")


def write_profiles(source_dir: Path) -> dict[str, str]:
    """Write the profiles of the test benchmarks and return their contents"""
    contents = {}
    for bm in CONFIG.benchmarks:
        for suffix in PROFILE_SUFFIXES:
            name = f"{bm}/{bm}-profile{suffix}"
            contents[name] = f"{name}\n"
            write_file(source_dir / "bms" / name, contents[name])

    return contents


def read_files(root: Path) -> dict[str, str]:
    files = (path for path in root.rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): path.read_text() for path in files}


@pytest.mark.parametrize(
    ("java_major", "expected_flags"),
    [
        (16, []),
        (17, ["-Dapp.security=allow"]),
        (18, ["-Dapp.security=allow", "-Dapp.java=18"]),
        (23, ["-Dapp.security=allow", "-Dapp.java=23"]),
        (24, ["-Dapp.java=24"]),
    ],
)
def test_flags_follow_java_range(java_major, expected_flags):
    assert benchmark_jvm_flags(APP_FLAG_RULES, java_major) == expected_flags


def test_other_benchmark_gets_no_flags():
    assert benchmark_jvm_flags(CONFIG.benchmarks["other"].jvm_flags, 21) == []


def test_profile_command():
    jar, profile_dir = Path("/b/dacapo.jar"), Path("/r/profiles/app")
    assert profile_command(CONFIG, "app", 21, jar, profile_dir) == [
        "java",
        "-XX:+UnlockDiagnosticVMOptions",
        "-XX:+DebugNonSafepoints",
        "-Dasync-profiler.event=cpu",
        "-Dasync-profiler.start.opts=interval=1ms",
        "-Dasync-profiler.output.fmt=jfr",
        "-Dasync-profiler.output.dir=/r/profiles/app",
        "-Dapp.security=allow",
        "-Dapp.java=21",
        "-jar",
        "/b/dacapo.jar",
        "--callback",
        "org.dacapo.profiling.AsyncProfilerCallback",
        "-n",
        "5",
        "app",
    ]


def test_conversion_commands():
    profile_dir = "/r/profiles/app"
    jfr_file = f"{profile_dir}/app-profile.jfr"
    txt_file = f"{profile_dir}/app-profile.txt"
    html_file = f"{profile_dir}/app-profile.html"
    assert conversion_commands("jfrconv", "app", Path(profile_dir)) == [
        ["jfrconv", "--lines", "-o", "collapsed", jfr_file, txt_file],
        ["jfrconv", "--lines", "-o", "html", jfr_file, html_file],
    ]


@pytest.mark.parametrize(
    ("version", "expected_major"),
    [("1.8.0_492", 8), ("11.0.2", 11), ("21.0.11", 21), ("25", 25), ("26-ea", 26)],
)
def test_java_major_version_is_parsed(tmp_path, version, expected_major):
    java = write_fake_java(tmp_path / "java", version)
    assert java_major_version(str(java)) == expected_major


def test_wrong_java_version_fails(study_dir, caplog):
    java = write_fake_java(study_dir / "bin/java", "17.0.18")
    config_path = Path("study.toml")
    config_path.write_text(config_path.read_text().replace('"java"', f'"{java}"'))
    with pytest.raises(SystemExit):
        main(["profile", "--config", "study.toml", "--out", "out"])
    assert "Java is version 17, expected 21" in caplog.text
    assert not Path("out").exists()


def test_unknown_benchmark_fails(study_dir, caplog):
    with pytest.raises(SystemExit):
        main(["profile", "--config", "study.toml", "--benchmarks", "app nope"])
    assert "Unknown benchmarks: nope" in caplog.text


def test_allowed_kernel_settings_pass(tmp_path):
    settings = {"kernel.perf_event_paranoid": 1, "kernel.kptr_restrict": 0}
    write_kernel_settings(tmp_path, settings)
    check_kernel_settings(MAX_KERNEL_SETTINGS, tmp_path)


def test_strict_kernel_settings_print_fixes(tmp_path):
    settings = {"kernel.perf_event_paranoid": 2, "kernel.kptr_restrict": 1}
    write_kernel_settings(tmp_path, settings)
    fixes = "sudo sysctl -w kernel.perf_event_paranoid=1\nsudo sysctl -w kernel.kptr_restrict=0"
    with pytest.raises(HotLoopsError, match=re.escape(fixes)):
        check_kernel_settings(MAX_KERNEL_SETTINGS, tmp_path)


def test_import_copies_profiles(study_dir):
    contents = write_profiles(Path("benchmarks"))
    main(IMPORT_ARGS)
    assert read_files(Path("results/profiles")) == contents


def test_import_logs_each_benchmark(study_dir, caplog):
    caplog.set_level(logging.INFO)
    write_profiles(Path("benchmarks"))
    main(IMPORT_ARGS)
    copying = [m for m in caplog.messages if m.startswith("Copying")]
    assert copying == [
        "Copying the profiles of app...",
        "Copying the profiles of other...",
    ]


def test_existing_target_is_refused(study_dir, caplog):
    write_profiles(Path("benchmarks"))
    Path("results/profiles/other").mkdir(parents=True)
    with pytest.raises(SystemExit):
        main(IMPORT_ARGS)
    assert "already exist" in caplog.text
    assert read_files(Path("results/profiles")) == {}
