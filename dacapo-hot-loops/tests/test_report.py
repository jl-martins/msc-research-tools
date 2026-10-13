"""Tests of the report command"""

import re
import shutil
from pathlib import Path

import polars as pl
import pytest
from conftest import DATA_DIR, write_jar

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import ANALYSIS_DIR_NAME, LOOP_SCHEMA, SUMMARY_SCHEMA
from dacapo_hot_loops.cli import main
from dacapo_hot_loops.config import SampleConfig
from dacapo_hot_loops.report import (
    SUMMARY_NAME,
    compute_benchmark_results,
    compute_overall_results,
    validate_worksheet,
)
from dacapo_hot_loops.sample import sample_hot_loops
from dacapo_hot_loops.worksheet import (
    HOT_LOOPS_NAME,
    WORKSHEET_NAME,
    Check,
    LoopStatus,
    read_worksheet,
)

CONFIG_NAME = "report-study.toml"
HOT_LOOPS = pl.read_csv(DATA_DIR / HOT_LOOPS_NAME, schema=LOOP_SCHEMA)
SUMMARY = pl.read_csv(DATA_DIR / SUMMARY_NAME, schema=SUMMARY_SCHEMA)
CLASSIFIED = read_worksheet(DATA_DIR / WORKSHEET_NAME)
CONVERTIBLE_LOOP = "app/Index.java:10-20"
CALLER_LOOP = "app/Index.java:30-32"
SHARED_LOOP = "shared/Scan.java:5-9"
STALE_LOOP = "old/Gone.java:1-3"
BENCHMARK_RESULTS = compute_benchmark_results(HOT_LOOPS, SUMMARY, CLASSIFIED)
SAMPLED_HOT_LOOPS = sample_hot_loops(HOT_LOOPS, SampleConfig(size=1, seed=1))


def with_cells(loop_id: str, **values: str | None) -> pl.DataFrame:
    """Copy the classified worksheet with new values in one loop's row"""
    is_loop = pl.col("loop_id") == loop_id
    return CLASSIFIED.with_columns(
        pl.when(is_loop)
        .then(pl.lit(value, pl.String))
        .otherwise(pl.col(column))
        .alias(column)
        for column, value in values.items()
    )


def assert_invalid(
    worksheet: pl.DataFrame,
    problem: str,
    sampled_loops: pl.DataFrame = HOT_LOOPS,
) -> None:
    with pytest.raises(HotLoopsError, match=re.escape(problem)):
        validate_worksheet(worksheet, sampled_loops)


def benchmark_row(benchmark: str) -> dict[str, object]:
    is_benchmark = pl.col("benchmark") == benchmark
    return BENCHMARK_RESULTS.filter(is_benchmark).row(0, named=True)


@pytest.fixture
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Results directory that holds the analysis and the classified worksheet"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATA_DIR / CONFIG_NAME, tmp_path / CONFIG_NAME)
    write_jar(
        tmp_path / "benchmarks" / "dacapo.jar",
        "Manifest-Version: 1.0\nImplementation-Version: 23.11-loops\n",
    )
    out_dir = tmp_path / "out"
    input_paths = (
        Path(ANALYSIS_DIR_NAME, HOT_LOOPS_NAME),
        Path(ANALYSIS_DIR_NAME, SUMMARY_NAME),
        Path(WORKSHEET_NAME),
    )
    for path in input_paths:
        target = out_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA_DIR / path.name, target)

    return out_dir


def run_report() -> None:
    main(["report", "--config", CONFIG_NAME, "--out", "out"])


def test_classified_worksheet_is_valid():
    validate_worksheet(CLASSIFIED, HOT_LOOPS)


def test_stale_row_needs_no_verdict():
    stale = CLASSIFIED.filter(pl.col("loop_id") == STALE_LOOP).row(0, named=True)
    assert (stale["status"], stale["verdict"]) == (LoopStatus.STALE, None)
    validate_worksheet(CLASSIFIED, HOT_LOOPS)


def test_blank_verdict_fails():
    worksheet = with_cells(CALLER_LOOP, verdict=None, failed_checks=None)
    assert_invalid(worksheet, f"blank or invalid verdict: {CALLER_LOOP}")


def test_invalid_verdict_fails():
    worksheet = with_cells(CALLER_LOOP, verdict="maybe")
    assert_invalid(worksheet, f"blank or invalid verdict: {CALLER_LOOP}")


def test_unknown_check_fails():
    worksheet = with_cells(CALLER_LOOP, failed_checks=f"{Check.SOURCE};speed")
    assert_invalid(worksheet, f"unknown failed_checks value: {CALLER_LOOP}")


def test_convertible_verdict_with_checks_fails():
    worksheet = with_cells(CONVERTIBLE_LOOP, failed_checks=Check.SOURCE.value)
    problem = f"'convertible' verdict with failed checks: {CONVERTIBLE_LOOP}"
    assert_invalid(worksheet, problem)


def test_not_verdict_without_checks_fails():
    worksheet = with_cells(CALLER_LOOP, failed_checks=None)
    assert_invalid(worksheet, f"'not' verdict without failed checks: {CALLER_LOOP}")


def test_missing_hot_loop_fails():
    worksheet = CLASSIFIED.filter(pl.col("loop_id") != SHARED_LOOP)
    assert_invalid(worksheet, f"missing hot loop (re-run worksheet): {SHARED_LOOP}")


def test_time_shares():
    assert BENCHMARK_RESULTS["time_share"].to_list() == [0.6, 0.0, None]


def test_count_shares():
    count_shares = BENCHMARK_RESULTS["count_share"].to_list()
    assert count_shares == [pytest.approx(1 / 3), 0.0, None]


def test_shares_are_undefined_without_hot_loops():
    zxing = benchmark_row("zxing")
    assert (zxing["time_share"], zxing["count_share"]) == (None, None)


def test_loop_counts_toward_every_failed_check():
    luindex = benchmark_row("luindex")
    failure_counts = {
        column: count
        for column, count in luindex.items()
        if column.startswith("fails_")
    }
    assert failure_counts == {
        "fails_source": 1,
        "fails_local_state": 1,
        "fails_control_flow": 1,
        "fails_exceptions": 0,
        "fails_iteration_order": 0,
    }


def test_shared_loop_counts_in_both_benchmarks():
    is_sharing = pl.col("benchmark").is_in(["luindex", "lusearch"])
    failure_counts = BENCHMARK_RESULTS.filter(is_sharing)["fails_source"].to_list()
    assert failure_counts == [1, 1]  # only the shared loop fails `source`


def test_summary_columns_are_kept():
    assert benchmark_row("lusearch")["outside_loops_share"] == 0.2


def test_overall_means_skip_benchmarks_without_hot_loops():
    overall = compute_overall_results(BENCHMARK_RESULTS).row(0)
    assert overall == (pytest.approx(0.3), pytest.approx(1 / 6), 2)


def test_overall_shares_are_undefined_without_hot_loops():
    without_hot_loops = BENCHMARK_RESULTS.filter(pl.col("hot_loops") == 0)
    overall = compute_overall_results(without_hot_loops).row(0)
    assert overall == (None, None, 0)


def test_sampled_loops_equal_hot_loops_without_sample():
    sampled = BENCHMARK_RESULTS["sampled_loops"]
    assert sampled.to_list() == BENCHMARK_RESULTS["hot_loops"].to_list()


def test_sample_holds_the_convertible_loop():
    assert SAMPLED_HOT_LOOPS["loop_id"].unique().to_list() == [CONVERTIBLE_LOOP]


def test_unclassified_hot_row_outside_sample_passes_validation():
    worksheet = with_cells(CALLER_LOOP, verdict=None, failed_checks=None)
    validate_worksheet(worksheet, SAMPLED_HOT_LOOPS)


def test_unclassified_sampled_loop_fails():
    worksheet = with_cells(CONVERTIBLE_LOOP, verdict=None)
    problem = f"blank or invalid verdict: {CONVERTIBLE_LOOP}"
    assert_invalid(worksheet, problem, SAMPLED_HOT_LOOPS)


def test_sampled_loop_with_stale_row_fails():
    worksheet = with_cells(CONVERTIBLE_LOOP, status=LoopStatus.STALE.value)
    problem = f"missing hot loop (re-run worksheet): {CONVERTIBLE_LOOP}"
    assert_invalid(worksheet, problem, SAMPLED_HOT_LOOPS)


def test_count_share_uses_sampled_loops():
    results = compute_benchmark_results(SAMPLED_HOT_LOOPS, SUMMARY, CLASSIFIED)
    luindex = results.filter(pl.col("benchmark") == "luindex")
    counts = luindex.select("hot_loops", "sampled_loops", "count_share")
    assert counts.row(0, named=True) == {
        "hot_loops": 3,
        "sampled_loops": 1,
        "count_share": 1.0,
    }


def test_benchmark_without_sampled_loops_has_undefined_share():
    results = compute_benchmark_results(SAMPLED_HOT_LOOPS, SUMMARY, CLASSIFIED)
    lusearch = results.filter(pl.col("benchmark") == "lusearch")
    counts = lusearch.select("sampled_loops", "count_share")
    assert counts.row(0, named=True) == {"sampled_loops": 0, "count_share": None}


def test_report_writes_every_file(results_dir):
    run_report()
    report_dir = results_dir / "report"
    names = {"per_benchmark.csv", "overall.csv", "report.md", "provenance.json"}
    assert {path.name for path in report_dir.iterdir()} == names


def test_csv_leaves_undefined_shares_empty(results_dir):
    run_report()
    text = (results_dir / "report/per_benchmark.csv").read_text()
    assert text.splitlines()[-1].startswith("zxing,,,0,0,")


def test_markdown_shows_undefined_shares(results_dir):
    run_report()
    markdown = (results_dir / "report/report.md").read_text()
    zxing_line = next(line for line in markdown.splitlines() if "zxing" in line)
    assert zxing_line.count("N/A") == 2


def test_invalid_worksheet_stops_report(results_dir):
    with_cells(CALLER_LOOP, verdict=None).write_csv(results_dir / WORKSHEET_NAME)
    with pytest.raises(SystemExit):
        run_report()

    assert not (results_dir / "report").exists()


def test_missing_jar_stops_report(results_dir):
    Path("benchmarks/dacapo.jar").unlink()
    with pytest.raises(SystemExit):
        run_report()

    assert not (results_dir / "report").exists()
