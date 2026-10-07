"""Reporting module"""

import logging
from pathlib import Path

import polars as pl
from tabulate import tabulate

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import (
    ANALYSIS_DIR_NAME,
    LOOP_SCHEMA,
    PROVENANCE_NAME,
    SHARE_DECIMALS,
    SUMMARY_SCHEMA,
    write_provenance,
)
from dacapo_hot_loops.config import StudyConfig
from dacapo_hot_loops.sources import read_jar_version
from dacapo_hot_loops.worksheet import (
    HOT_LOOPS_NAME,
    LIST_SEPARATOR,
    WORKSHEET_NAME,
    Check,
    LoopStatus,
    Verdict,
    read_analysis_table,
    read_worksheet,
)

logger = logging.getLogger(__name__)

REPORT_DIR_NAME = "report"
SUMMARY_NAME = "summary.csv"
BENCHMARK_RESULTS_NAME = "per_benchmark.csv"
OVERALL_RESULTS_NAME = "overall.csv"
MARKDOWN_REPORT_NAME = "report.md"
FAILURE_COLUMNS = {check: f"fails_{check}" for check in Check}
# Columns that open each benchmark's row, before the other summary columns
LEADING_COLUMNS = (
    "benchmark",
    "time_share",
    "count_share",
    "hot_loops",
    "convertible_loops",
)
BENCHMARK_RESULT_COLUMNS = (
    *LEADING_COLUMNS,
    *(column for column in SUMMARY_SCHEMA if column not in LEADING_COLUMNS),
    *FAILURE_COLUMNS.values(),
)
# Cell that stands for an undefined share in Markdown
UNDEFINED_CELL = "N/A"
# Plain strings because Polars casts a list of enum members to an Enum type
VERDICT_VALUES = tuple(verdict.value for verdict in Verdict)
CHECK_VALUES = tuple(check.value for check in Check)


def validate_worksheet(worksheet: pl.DataFrame, hot_loops: pl.DataFrame) -> None:
    """Validate the classification of every hot loop"""
    hot_rows = worksheet.filter(pl.col("status") == LoopStatus.HOT)
    verdict = pl.col("verdict")
    has_checks = pl.col("failed_checks").is_not_null()
    has_valid_verdict = verdict.is_in(VERDICT_VALUES).fill_null(False)
    has_known_checks = (
        pl.col("failed_checks")
        .str.split(LIST_SEPARATOR)
        .list.eval(pl.element().is_in(CHECK_VALUES))
        .list.all()
        .fill_null(True)
    )
    conditions = {
        "blank or invalid verdict": ~has_valid_verdict,
        "unknown failed_checks value": ~has_known_checks,
        "'convertible' verdict with failed checks": (verdict == Verdict.CONVERTIBLE)
        & has_checks,
        "'not' verdict without failed checks": (verdict == Verdict.NOT) & ~has_checks,
    }
    problems = {
        kind: hot_rows.filter(condition)["loop_id"].to_list()
        for kind, condition in conditions.items()
    }
    # A hot loop without a hot row means the worksheet predates the analysis
    missing = hot_loops.join(hot_rows, on="loop_id", how="anti")["loop_id"]
    problems["missing hot loop (re-run worksheet)"] = missing.unique().sort().to_list()

    lines = [
        f"  {kind}: {', '.join(ids)}" for kind, ids in problems.items() if len(ids) != 0
    ]
    if len(lines) != 0:
        details = "\n".join(lines)
        raise HotLoopsError(f"Invalid classification worksheet:\n{details}")


def compute_benchmark_results(
    hot_loops: pl.DataFrame,
    summary: pl.DataFrame,
    worksheet: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the shares, counts and summary columns of each benchmark"""
    hot_rows = worksheet.filter(pl.col("status") == LoopStatus.HOT)
    verdicts = hot_rows.select("loop_id", "verdict", "failed_checks")
    is_convertible = pl.col("verdict") == Verdict.CONVERTIBLE
    failed_checks = pl.col("failed_checks").str.split(LIST_SEPARATOR)
    failure_counts = [
        failed_checks.list.contains(check).sum().alias(column)
        for check, column in FAILURE_COLUMNS.items()
    ]
    counts = (
        hot_loops.join(verdicts, on="loop_id", how="left")
        .group_by("benchmark")
        .agg(
            is_convertible.sum().alias("convertible_loops"),
            pl.col("samples").filter(is_convertible).sum().alias("convertible_samples"),
            pl.col("samples").sum().alias("hot_samples"),
            *failure_counts,
        )
    )
    count_columns = [
        "convertible_loops",
        "convertible_samples",
        *FAILURE_COLUMNS.values(),
    ]
    # A share is undefined for a benchmark without hot loops
    has_hot_loops = pl.col("hot_loops") != 0
    time_share = pl.col("convertible_samples") / pl.col("hot_samples")
    count_share = pl.col("convertible_loops") / pl.col("hot_loops")

    return (
        summary.join(counts, on="benchmark", how="left")
        .with_columns(pl.col(count_columns).fill_null(0).cast(pl.Int64))
        .with_columns(
            time_share=pl.when(has_hot_loops).then(time_share),
            count_share=pl.when(has_hot_loops).then(count_share),
        )
        .select(BENCHMARK_RESULT_COLUMNS)
        .sort("benchmark")
    )


def compute_overall_results(benchmark_results: pl.DataFrame) -> pl.DataFrame:
    """Compute the unweighted mean shares of the benchmarks with hot loops"""
    with_hot_loops = benchmark_results.filter(pl.col("hot_loops") != 0)

    return with_hot_loops.select(
        pl.col("time_share").mean().alias("mean_time_share"),
        pl.col("count_share").mean().alias("mean_count_share"),
        pl.len().cast(pl.Int64).alias("benchmarks"),
    )


def markdown_table(table: pl.DataFrame) -> str:
    return tabulate(
        table.rows(),
        headers=table.columns,
        tablefmt="github",
        floatfmt=f".{SHARE_DECIMALS}f",
        missingval=UNDEFINED_CELL,
    )


def report(
    config: StudyConfig,
    config_path: Path,
    benchmarks_dir: Path,
    out_dir: Path,
) -> None:
    """Validate the worksheet, then write the report tables and provenance"""
    hot_loops = read_analysis_table(out_dir, HOT_LOOPS_NAME, LOOP_SCHEMA)
    summary = read_analysis_table(out_dir, SUMMARY_NAME, SUMMARY_SCHEMA)
    worksheet_path = out_dir / WORKSHEET_NAME
    worksheet = read_worksheet(worksheet_path)
    validate_worksheet(worksheet, hot_loops)
    # Read first so an unreadable jar fails before any table is written
    jar_version = read_jar_version(benchmarks_dir / config.jar_name)

    benchmark_results = compute_benchmark_results(hot_loops, summary, worksheet)
    overall_results = compute_overall_results(benchmark_results)
    report_dir = out_dir / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    benchmark_results.write_csv(
        report_dir / BENCHMARK_RESULTS_NAME,
        float_precision=SHARE_DECIMALS,
    )
    overall_results.write_csv(
        report_dir / OVERALL_RESULTS_NAME,
        float_precision=SHARE_DECIMALS,
    )
    markdown = (
        "# Hot-loop report\n\n"
        f"## Per benchmark\n\n{markdown_table(benchmark_results)}\n\n"
        f"## Overall\n\n{markdown_table(overall_results)}\n"
    )
    (report_dir / MARKDOWN_REPORT_NAME).write_text(markdown, encoding="utf-8")

    analysis_dir = out_dir / ANALYSIS_DIR_NAME
    input_paths = [
        analysis_dir / HOT_LOOPS_NAME,
        analysis_dir / SUMMARY_NAME,
        worksheet_path,
    ]
    inputs = {path.relative_to(out_dir).as_posix(): path for path in input_paths}
    write_provenance(
        config_path,
        jar_version,
        inputs,
        report_dir / PROVENANCE_NAME,
    )
    logger.info("Wrote the report tables to '%s'", report_dir)
