"""Tests of the coverage command"""

import json
from pathlib import Path

import polars as pl
import pytest
from conftest import write_file

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import ANALYSIS_DIR_NAME, LOOP_SCHEMA, PROVENANCE_NAME
from dacapo_hot_loops.cli import main
from dacapo_hot_loops.coverage import (
    BENCHMARK_RESULTS_NAME,
    COVERAGE_DIR_NAME,
    Conversion,
    classify,
    diff_files,
)
from dacapo_hot_loops.javasrc import LineRange, parse_java
from dacapo_hot_loops.worksheet import HOT_LOOPS_NAME

# The loop spans lines 3-5
LOOP_SOURCE = """\
class Main {
    void run(List<String> items) {
        for (String item : items) {
            System.out.println(item);
        }
    }
}
"""
SHIFTED_LOOP_SOURCE = """\
import java.util.List;

class Main {
    void run(List<String> items) {
        for (String item : items) {
            System.out.println(item);
        }
    }
}
"""
REINDENTED_LOOP_SOURCE = """\
class Main {
  void run(List<String> items) {
    for (String item : items) {
      System.out.println(item);
    }
  }
}
"""
STREAM_SOURCE = """\
class Main {
    void run(List<String> items) {
        items.stream().forEach(item -> {
            System.out.println(item);
        });
    }
}
"""
DELETED_LOOP_SOURCE = """\
class Main {
    void run(List<String> items) {
    }
}
"""
RENAMED_VARIABLE_SOURCE = """\
class Main {
    void run(List<String> items) {
        for (String name : items) {
            System.out.println(name);
        }
    }
}
"""
# The outer loop starts at line 3 and the inner loop at line 4
NESTED_LOOP_SOURCE = """\
class Grid {
    void run(int[][] rows) {
        for (int[] row : rows) {
            for (int cell : row) {
                System.out.println(cell);
            }
        }
    }
}
"""
NESTED_STREAM_SOURCE = """\
class Grid {
    void run(int[][] rows) {
        for (int[] row : rows) {
            Arrays.stream(row).forEach(cell -> {
                System.out.println(cell);
            });
        }
    }
}
"""
MAIN_FILE = "app/src/main/java/demo/Main.java"
GONE_FILE = "app/src/main/java/demo/Gone.java"
OTHER_FILE = "other/src/demo/Other.java"


def classify_loop(
    tmp_path: Path,
    loops_source: str,
    streams_source: str,
    start_line: int,
) -> tuple[Conversion, LineRange | None]:
    loops_file = write_file(tmp_path / "loops/Main.java", loops_source)
    streams_file = write_file(tmp_path / "streams/Main.java", streams_source)
    opcodes = diff_files(loops_file, streams_file)
    return classify(opcodes, parse_java(streams_source), start_line)


def hot_loop(
    benchmark: str,
    file: str,
    line_range: LineRange,
    share: float,
) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "loop_id": f"{file}:{line_range.start}-{line_range.end}",
        "file": file,
        "start_line": line_range.start,
        "end_line": line_range.end,
        "loop_type": "enhanced_for",
        "enclosing_method_key": "demo.Main.run",
        "origin": "hot-method",
        "samples": round(share * 100),
        "share": share,
    }


def test_identical_loop_is_unchanged(tmp_path):
    conversion = classify_loop(tmp_path, LOOP_SOURCE, LOOP_SOURCE, 3)
    assert conversion == (Conversion.UNCHANGED, LineRange(3, 3))


def test_unchanged_loop_maps_to_its_new_line(tmp_path):
    conversion = classify_loop(tmp_path, LOOP_SOURCE, SHIFTED_LOOP_SOURCE, 3)
    assert conversion == (Conversion.UNCHANGED, LineRange(5, 5))


def test_reindented_loop_is_unchanged(tmp_path):
    conversion = classify_loop(tmp_path, LOOP_SOURCE, REINDENTED_LOOP_SOURCE, 3)
    assert conversion == (Conversion.UNCHANGED, LineRange(3, 3))


def test_stream_loop_is_converted(tmp_path):
    conversion = classify_loop(tmp_path, LOOP_SOURCE, STREAM_SOURCE, 3)
    assert conversion == (Conversion.CONVERTED, LineRange(3, 3))


def test_deleted_loop_is_converted_without_lines(tmp_path):
    conversion = classify_loop(tmp_path, LOOP_SOURCE, DELETED_LOOP_SOURCE, 3)
    assert conversion == (Conversion.CONVERTED, None)


def test_loop_with_new_header_is_rewritten(tmp_path):
    conversion = classify_loop(tmp_path, LOOP_SOURCE, RENAMED_VARIABLE_SOURCE, 3)
    assert conversion == (Conversion.REWRITTEN, LineRange(3, 4))


@pytest.mark.parametrize(
    ("start_line", "expected"),
    [
        (3, (Conversion.UNCHANGED, LineRange(3, 3))),
        (4, (Conversion.CONVERTED, LineRange(4, 4))),
    ],
)
def test_nested_loops_are_classified_apart(tmp_path, start_line, expected):
    conversion = classify_loop(
        tmp_path,
        NESTED_LOOP_SOURCE,
        NESTED_STREAM_SOURCE,
        start_line,
    )
    assert conversion == expected


def test_line_past_the_end_fails(tmp_path):
    with pytest.raises(HotLoopsError, match="Line 99 is past the end of the file"):
        classify_loop(tmp_path, LOOP_SOURCE, LOOP_SOURCE, 99)


@pytest.fixture
def coverage_dir(streams_study: Path) -> Path:
    """Coverage of a converted loop, a missing file and an unchanged loop"""
    write_file(Path("benchmarks", MAIN_FILE), LOOP_SOURCE)
    write_file(Path("streams-benchmarks", MAIN_FILE), STREAM_SOURCE)
    write_file(Path("benchmarks", OTHER_FILE), LOOP_SOURCE)
    write_file(Path("streams-benchmarks", OTHER_FILE), LOOP_SOURCE)
    hot_loops = pl.DataFrame(
        [
            hot_loop("app", MAIN_FILE, LineRange(3, 5), 0.6),
            hot_loop("app", GONE_FILE, LineRange(1, 3), 0.1),
            hot_loop("other", OTHER_FILE, LineRange(3, 5), 0.5),
        ],
        schema=LOOP_SCHEMA,
    )
    write_file(Path("out", ANALYSIS_DIR_NAME, HOT_LOOPS_NAME), hot_loops.write_csv())
    main(["coverage", "--config", "study.toml", "--out", "out"])
    return Path("out", COVERAGE_DIR_NAME)


def test_hot_loop_table_has_conversions(coverage_dir):
    hot_loops = pl.read_csv(coverage_dir / HOT_LOOPS_NAME)
    columns = ["conversion", "streams_start_line", "streams_end_line"]
    assert hot_loops.select(columns).rows() == [
        ("converted", 3, 3),
        ("missing", None, None),
        ("unchanged", 3, 3),
    ]


def test_benchmark_table_sums_conversions(coverage_dir):
    benchmark_results = pl.read_csv(coverage_dir / BENCHMARK_RESULTS_NAME)
    assert benchmark_results.to_dicts() == [
        {
            "benchmark": "app",
            "hot_loops": 2,
            "unchanged_loops": 0,
            "converted_loops": 1,
            "rewritten_loops": 0,
            "missing_loops": 1,
            "converted_share": 0.6,
            "rewritten_share": 0.0,
        },
        {
            "benchmark": "other",
            "hot_loops": 1,
            "unchanged_loops": 1,
            "converted_loops": 0,
            "rewritten_loops": 0,
            "missing_loops": 0,
            "converted_share": 0.0,
            "rewritten_share": 0.0,
        },
    ]


def test_provenance_records_streams_jar(coverage_dir):
    provenance = json.loads((coverage_dir / PROVENANCE_NAME).read_text())
    assert provenance["jar_version"] == "23.11-streams"
