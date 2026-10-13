"""Tests of the worksheet command"""

import shutil
from pathlib import Path

import polars as pl
import pytest
from conftest import DATA_DIR

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import ANALYSIS_DIR_NAME, LOOP_SCHEMA
from dacapo_hot_loops.cli import main
from dacapo_hot_loops.config import SampleConfig
from dacapo_hot_loops.worksheet import (
    HOT_LOOPS_NAME,
    WORKSHEET_NAME,
    WORKSHEET_SCHEMA,
    LoopStatus,
    Verdict,
    merge_worksheet,
    read_worksheet,
    write_worksheet,
)

CONFIG_NAME = "report-study.toml"
HOT_LOOPS = pl.read_csv(DATA_DIR / HOT_LOOPS_NAME, schema=LOOP_SCHEMA)
CLASSIFIED = read_worksheet(DATA_DIR / WORKSHEET_NAME)
EMPTY = pl.DataFrame(schema=WORKSHEET_SCHEMA)
CONVERTIBLE_LOOP = "app/Index.java:10-20"
CALLER_LOOP = "app/Index.java:30-32"
SHARED_LOOP = "shared/Scan.java:5-9"
STALE_LOOP = "old/Gone.java:1-3"


def loop_row(worksheet: pl.DataFrame, loop_id: str) -> dict[str, object]:
    return worksheet.filter(pl.col("loop_id") == loop_id).row(0, named=True)


@pytest.fixture
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Results directory that holds the analysis of the report study"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATA_DIR / CONFIG_NAME, tmp_path / CONFIG_NAME)
    analysis_dir = tmp_path / "out" / ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True)
    shutil.copy(DATA_DIR / HOT_LOOPS_NAME, analysis_dir / HOT_LOOPS_NAME)
    return tmp_path / "out"


def run_worksheet() -> None:
    main(["worksheet", "--config", CONFIG_NAME, "--out", "out"])


def test_one_row_per_loop():
    worksheet = merge_worksheet(HOT_LOOPS, EMPTY)
    assert worksheet["loop_id"].to_list() == [
        CONVERTIBLE_LOOP,
        CALLER_LOOP,
        SHARED_LOOP,
    ]


def test_shared_loop_lists_both_benchmarks():
    shared = loop_row(merge_worksheet(HOT_LOOPS, EMPTY), SHARED_LOOP)
    assert shared["benchmarks"] == "luindex;lusearch"


def test_new_loops_have_blank_verdicts():
    assert merge_worksheet(HOT_LOOPS, EMPTY)["verdict"].null_count() == 3


def test_worksheet_holds_no_samples():
    columns = merge_worksheet(HOT_LOOPS, EMPTY).columns
    assert {"samples", "share"}.isdisjoint(columns)


def test_same_analysis_keeps_worksheet():
    assert merge_worksheet(HOT_LOOPS, CLASSIFIED).equals(CLASSIFIED)


def test_missing_loop_becomes_stale():
    hot_loops = HOT_LOOPS.filter(pl.col("loop_id") != CALLER_LOOP)
    caller = loop_row(merge_worksheet(hot_loops, CLASSIFIED), CALLER_LOOP)
    assert (caller["status"], caller["verdict"]) == (LoopStatus.STALE, Verdict.NOT)


def test_stale_loop_becomes_hot_again():
    is_caller = pl.col("loop_id") == CALLER_LOOP
    stale = pl.lit(LoopStatus.STALE.value)
    status = pl.when(is_caller).then(stale).otherwise(pl.col("status"))
    previous = CLASSIFIED.with_columns(status=status)
    caller = loop_row(merge_worksheet(HOT_LOOPS, previous), CALLER_LOOP)
    assert (caller["status"], caller["verdict"]) == (LoopStatus.HOT, Verdict.NOT)


def test_loop_columns_are_refreshed():
    previous = CLASSIFIED.with_columns(enclosing_method_key=pl.lit("old.Key"))
    caller = loop_row(merge_worksheet(HOT_LOOPS, previous), CALLER_LOOP)
    assert caller["enclosing_method_key"] == "app.Index.flush"


def test_conflicting_loop_columns_fail():
    shared = HOT_LOOPS.filter(pl.col("loop_id") == SHARED_LOOP).head(1)
    conflict = shared.with_columns(
        benchmark=pl.lit("zxing"),
        enclosing_method_key=pl.lit("other.Key"),
    )
    with pytest.raises(HotLoopsError, match=SHARED_LOOP):
        merge_worksheet(pl.concat([HOT_LOOPS, conflict]), EMPTY)


def test_rerun_keeps_classified_file(results_dir):
    shutil.copy(DATA_DIR / WORKSHEET_NAME, results_dir / WORKSHEET_NAME)
    run_worksheet()
    expected = (DATA_DIR / WORKSHEET_NAME).read_bytes()
    assert (results_dir / WORKSHEET_NAME).read_bytes() == expected


def test_loops_outside_sample_become_stale(results_dir):
    shutil.copy(DATA_DIR / WORKSHEET_NAME, results_dir / WORKSHEET_NAME)
    write_worksheet(results_dir, SampleConfig(size=1, seed=1))
    worksheet = read_worksheet(results_dir / WORKSHEET_NAME)
    statuses = dict(zip(worksheet["loop_id"], worksheet["status"], strict=True))
    assert statuses == {
        CONVERTIBLE_LOOP: LoopStatus.HOT,
        CALLER_LOOP: LoopStatus.STALE,
        STALE_LOOP: LoopStatus.STALE,
        SHARED_LOOP: LoopStatus.STALE,
    }


def test_reordered_columns_fail(results_dir):
    reordered = CLASSIFIED.select(reversed(CLASSIFIED.columns))
    reordered.write_csv(results_dir / WORKSHEET_NAME)
    with pytest.raises(SystemExit):
        run_worksheet()


def test_duplicate_loops_fail(results_dir):
    pl.concat([CLASSIFIED, CLASSIFIED.head(1)]).write_csv(results_dir / WORKSHEET_NAME)
    with pytest.raises(SystemExit):
        run_worksheet()
