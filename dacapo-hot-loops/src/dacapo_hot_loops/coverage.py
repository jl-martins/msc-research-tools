"""Coverage module"""

import difflib
import functools
import logging
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import polars as pl

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import (
    ANALYSIS_DIR_NAME,
    LOOP_SCHEMA,
    PROVENANCE_NAME,
    SHARE_DECIMALS,
    write_provenance,
)
from dacapo_hot_loops.build import resolve_benchmarks_dir
from dacapo_hot_loops.config import StudyConfig
from dacapo_hot_loops.javasrc import JavaFile, JavaSources, LineRange
from dacapo_hot_loops.sources import read_jar_version
from dacapo_hot_loops.worksheet import HOT_LOOPS_NAME, read_analysis_table

logger = logging.getLogger(__name__)

COVERAGE_DIR_NAME = "coverage"
BENCHMARK_RESULTS_NAME = "per_benchmark.csv"


class Opcode(NamedTuple):
    """Diff block with its 0-based, end-exclusive line ranges in both files"""

    tag: str
    loops_start: int
    loops_end: int
    streams_start: int
    streams_end: int


class Conversion(StrEnum):
    """What the stream-based implementation did to a hot loop"""

    # The header line of the loop is intact
    UNCHANGED = "unchanged"
    # The header line is gone and no loop starts in the code that replaced it
    CONVERTED = "converted"
    # The header line is gone but a loop starts in the code that replaced it
    REWRITTEN = "rewritten"
    # The stream-based implementation has no file at the loop's path
    MISSING = "missing"


COVERAGE_SCHEMA: Mapping[str, pl.DataType] = {
    **LOOP_SCHEMA,
    "conversion": pl.String(),
    "streams_start_line": pl.Int64(),
    "streams_end_line": pl.Int64(),
}


def stripped_lines(path: Path) -> list[str]:
    """Lines without whitespace, so that reindenting a line does not change it"""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise HotLoopsError(f"Could not read '{path}': {e}") from e

    return ["".join(line.split()) for line in text.splitlines()]


def diff_files(loops_file: Path, streams_file: Path) -> list[Opcode]:
    matcher = difflib.SequenceMatcher(
        None,
        stripped_lines(loops_file),
        stripped_lines(streams_file),
        autojunk=False,
    )
    return [Opcode(*opcode) for opcode in matcher.get_opcodes()]


def classify(
    opcodes: list[Opcode],
    streams_file: JavaFile,
    start_line: int,
) -> tuple[Conversion, LineRange | None]:
    """Classify a loop by the diff block that holds its header line"""
    # Loop lines are 1-based and inclusive; difflib indices are 0-based and exclusive
    # at the end, so a 0-based range [start, end) is the 1-based range [start + 1, end]
    header_index = start_line - 1
    for opcode in opcodes:
        if opcode.loops_start <= header_index < opcode.loops_end:
            if opcode.tag == "equal":
                offset = header_index - opcode.loops_start
                streams_line = opcode.streams_start + offset + 1
                return Conversion.UNCHANGED, LineRange(streams_line, streams_line)
            else:
                # A deleted block has no lines in the stream-based file
                has_lines = opcode.streams_end > opcode.streams_start
                block = (
                    LineRange(opcode.streams_start + 1, opcode.streams_end)
                    if has_lines
                    else None
                )
                starts_loop = block is not None and any(
                    block.contains_line(loop.line_range.start)
                    for loop in streams_file.loops
                )
                if starts_loop:
                    return Conversion.REWRITTEN, block
                else:
                    return Conversion.CONVERTED, block

    raise HotLoopsError(f"Line {start_line} is past the end of the file")


def compute_benchmark_results(coverage_table: pl.DataFrame) -> pl.DataFrame:
    """Count the hot loops of each conversion and sum their shares"""

    def has_conversion(conversion: Conversion) -> pl.Expr:
        return pl.col("conversion") == conversion

    return (
        coverage_table.group_by("benchmark")
        .agg(
            pl.len().alias("hot_loops"),
            *(has_conversion(c).sum().alias(f"{c}_loops") for c in Conversion),
            *(
                pl.col("share").filter(has_conversion(c)).sum().alias(f"{c}_share")
                for c in (Conversion.CONVERTED, Conversion.REWRITTEN)
            ),
        )
        .sort("benchmark")
    )


def coverage(config: StudyConfig, config_path: Path, out_dir: Path) -> None:
    """Find the hot loops that the stream-based implementation converted"""
    hot_loops = read_analysis_table(out_dir, HOT_LOOPS_NAME, LOOP_SCHEMA)
    loops_dir = resolve_benchmarks_dir(config, "loops")
    streams_dir = resolve_benchmarks_dir(config, "streams")
    # Read first so an unreadable jar fails before any table is written
    streams_jar_name = config.implementation("streams").jar_name
    jar_version = read_jar_version(streams_dir / streams_jar_name)
    streams_sources = JavaSources(streams_dir)

    # Several hot loops can share a file, so each file pair is diffed once
    @functools.cache
    def file_opcodes(file: str) -> list[Opcode]:
        return diff_files(loops_dir / file, streams_dir / file)

    rows: list[dict[str, object]] = []
    for loops in hot_loops.partition_by("benchmark", maintain_order=True):
        logger.info("Comparing the hot loops of %s...", loops.item(0, "benchmark"))
        for loop in loops.iter_rows(named=True):
            file = loop["file"]
            if (streams_dir / file).is_file():
                streams_file = streams_sources.file(Path(file))
                opcodes = file_opcodes(file)
                conversion, block = classify(opcodes, streams_file, loop["start_line"])
            else:
                conversion, block = Conversion.MISSING, None

            rows.append(
                loop
                | {
                    "conversion": conversion.value,
                    "streams_start_line": block.start if block else None,
                    "streams_end_line": block.end if block else None,
                }
            )

    coverage_table = pl.DataFrame(rows, schema=COVERAGE_SCHEMA)
    benchmark_results = compute_benchmark_results(coverage_table)
    coverage_dir = out_dir / COVERAGE_DIR_NAME
    coverage_dir.mkdir(parents=True, exist_ok=True)
    coverage_table.write_csv(
        coverage_dir / HOT_LOOPS_NAME,
        float_precision=SHARE_DECIMALS,
    )
    benchmark_results.write_csv(
        coverage_dir / BENCHMARK_RESULTS_NAME,
        float_precision=SHARE_DECIMALS,
    )

    input_path = out_dir / ANALYSIS_DIR_NAME / HOT_LOOPS_NAME
    inputs = {input_path.relative_to(out_dir).as_posix(): input_path}
    write_provenance(config_path, jar_version, inputs, coverage_dir / PROVENANCE_NAME)

    # A hot loop shared by several benchmarks has one row per benchmark
    def count(conversion: Conversion) -> int:
        return (
            coverage_table.filter(pl.col("conversion") == conversion)
            .get_column("loop_id")
            .n_unique()
        )

    logger.info(
        "Classified %d hot loops (converted: %d, rewritten: %d). Output was written to '%s'.",
        coverage_table.get_column("loop_id").n_unique(),
        count(Conversion.CONVERTED),
        count(Conversion.REWRITTEN),
        coverage_dir,
    )
