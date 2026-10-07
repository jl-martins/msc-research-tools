"""Tests of the analyze command"""

import filecmp
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from conftest import DATA_DIR, write_file, write_jar

from dacapo_hot_loops.analysis import ANALYSIS_DIR_NAME, PROVENANCE_NAME
from dacapo_hot_loops.cli import main

CONFIG_NAME = "shared-source-study.toml"
JAR_NAME = "dacapo.jar"
JAR_VERSION = "23.11-loops"
BENCHMARKS = ("luindex", "lusearch")
# Both benchmarks index shared/Outer.java, so this loop is hot in each
SHARED_LOOP = "shared/Outer.java:15-19"


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Study where two benchmarks index the same source file"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATA_DIR / CONFIG_NAME, tmp_path / CONFIG_NAME)
    write_jar(
        tmp_path / "benchmarks" / JAR_NAME,
        f"Manifest-Version: 1.0\nImplementation-Version: {JAR_VERSION}\n",
    )
    outer = (DATA_DIR / "Outer.java").read_text(encoding="utf-8")
    write_file(tmp_path / "benchmarks/shared/Outer.java", outer)
    for bm in BENCHMARKS:
        profile = (DATA_DIR / f"{bm}-profile.txt").read_text(encoding="utf-8")
        write_file(tmp_path / f"out/profiles/{bm}/{bm}-profile.txt", profile)

    return tmp_path


def analyze_into(out_dir: str) -> Path:
    main(["analyze", "--config", CONFIG_NAME, "--out", out_dir])
    return Path(out_dir) / ANALYSIS_DIR_NAME


def read_table(analysis_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(analysis_dir / name)


def shared_loop_rows(analysis_dir: Path) -> pl.DataFrame:
    hot = read_table(analysis_dir, "hot_loops.csv")
    return hot.filter(pl.col("loop_id") == SHARED_LOOP)


def test_summary_holds_one_row_per_benchmark(study):
    summary = read_table(analyze_into("out"), "summary.csv")
    assert summary["benchmark"].to_list() == list(BENCHMARKS)


def test_buckets_add_up_to_the_total(study):
    summary = read_table(analyze_into("out"), "summary.csv")
    buckets = (
        summary["no_app_samples"]
        + summary["no_source_samples"]
        + summary["charged_samples"]
    )
    assert buckets.to_list() == summary["total_samples"].to_list()


def test_jdk_samples_land_outside_the_application(study):
    summary = read_table(analyze_into("out"), "summary.csv")
    lusearch = summary.filter(pl.col("benchmark") == "lusearch").row(0, named=True)
    assert (lusearch["no_app_samples"], lusearch["charged_samples"]) == (10, 50)


def test_shared_loop_is_hot_in_both_benchmarks(study):
    assert shared_loop_rows(analyze_into("out"))["benchmark"].to_list() == list(
        BENCHMARKS
    )


def test_shared_loop_is_charged_per_benchmark(study):
    assert shared_loop_rows(analyze_into("out"))["samples"].to_list() == [80, 50]


def test_shares_hold_six_decimal_places(study):
    text = (analyze_into("out") / "hot_loops.csv").read_text(encoding="utf-8")
    assert "0.800000" in text


def test_every_hot_loop_reaches_the_threshold(study):
    assert read_table(analyze_into("out"), "hot_loops.csv")["share"].min() >= 0.001


def loop_ids(table: pl.DataFrame, bm: str) -> set[str]:
    return set(table.filter(pl.col("benchmark") == bm)["loop_id"])


def test_loops_below_the_threshold_are_only_chargeable(study):
    analysis_dir = analyze_into("out")
    chargeable = loop_ids(read_table(analysis_dir, "chargeable_loops.csv"), "luindex")
    hot = loop_ids(read_table(analysis_dir, "hot_loops.csv"), "luindex")
    # Neither the `do` loop nor the lambda's loop takes a sample
    assert sorted(chargeable - hot) == [
        "shared/Outer.java:20-22",
        "shared/Outer.java:24-26",
    ]


def test_anonymous_class_loop_needs_its_own_hot_method(study):
    # Outer$1.run holds this loop and takes no sample, so it is never hot
    chargeable = read_table(analyze_into("out"), "chargeable_loops.csv")
    assert "shared/Outer.java:30-32" not in chargeable["loop_id"].to_list()


def test_resolved_method_is_reported_as_resolved(study):
    hot_methods = read_table(analyze_into("out"), "hot_methods.csv")
    assert hot_methods["status"].to_list() == ["resolved", "resolved"]


def test_parse_errors_holds_only_a_header(study):  # because there are no parse errors
    parse_errors = read_table(analyze_into("out"), "parse_errors.csv")
    assert parse_errors.columns == ["benchmark", "file", "error_lines"]
    assert len(parse_errors) == 0


def test_provenance_records_the_jar_version(study):
    provenance = (analyze_into("out") / PROVENANCE_NAME).read_text(encoding="utf-8")
    assert json.loads(provenance)["jar_version"] == JAR_VERSION


def test_second_run_writes_identical_tables(study):
    first = analyze_into("out")
    shutil.copytree("out/profiles", "second/profiles")
    second = analyze_into("second")

    tables = sorted(p.name for p in first.iterdir() if p.name != PROVENANCE_NAME)
    match, mismatch, errors = filecmp.cmpfiles(first, second, tables, shallow=False)
    assert (sorted(match), mismatch, errors) == (tables, [], [])


def test_missing_profile_fails(study):
    Path("out/profiles/luindex/luindex-profile.txt").unlink()
    with pytest.raises(SystemExit):
        analyze_into("out")


def test_missing_jar_stops_analysis(study):
    Path(f"benchmarks/{JAR_NAME}").unlink()
    with pytest.raises(SystemExit):
        analyze_into("out")

    assert not Path("out/analysis").exists()
