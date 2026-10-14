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
from dacapo_hot_loops.sample import sample_hot_loops
from dacapo_hot_loops.sources import read_jar_version
from dacapo_hot_loops.stats import hypergeom_count_ci
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
    "count_share",
    "hot_loops",
    "sampled_loops",
    "convertible_loops",
)
BENCHMARK_RESULT_COLUMNS = (
    *LEADING_COLUMNS,
    *(column for column in SUMMARY_SCHEMA if column not in LEADING_COLUMNS),
    *FAILURE_COLUMNS.values(),
)
# Confidence level of the interval around the overall share
CONFIDENCE_LEVEL = 0.95
# Cell that stands for an undefined share in Markdown
UNDEFINED_CELL = "N/A"
# Plain strings because Polars casts a list of enum members to an Enum type
VERDICT_VALUES = tuple(verdict.value for verdict in Verdict)
CHECK_VALUES = tuple(check.value for check in Check)


def validate_worksheet(worksheet: pl.DataFrame, sampled_loops: pl.DataFrame) -> None:
    """Validate the classification of every sampled loop"""
    hot_rows = worksheet.filter(pl.col("status") == LoopStatus.HOT)
    sampled_rows = hot_rows.join(sampled_loops, on="loop_id", how="semi")
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
        kind: sampled_rows.filter(condition)["loop_id"].to_list()
        for kind, condition in conditions.items()
    }
    # A sampled loop without a hot row means the worksheet predates the analysis or the sample
    missing = sampled_loops.join(hot_rows, on="loop_id", how="anti")["loop_id"]
    problems["missing hot loop (re-run worksheet)"] = missing.unique().sort().to_list()

    lines = [
        f"  {kind}: {', '.join(ids)}" for kind, ids in problems.items() if len(ids) != 0
    ]
    if len(lines) != 0:
        details = "\n".join(lines)
        raise HotLoopsError(f"Invalid classification worksheet:\n{details}")


def compute_benchmark_results(
    sampled_loops: pl.DataFrame,
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
        sampled_loops.join(verdicts, on="loop_id", how="left")
        .group_by("benchmark")
        .agg(
            pl.len().alias("sampled_loops"),
            is_convertible.sum().alias("convertible_loops"),
            *failure_counts,
        )
    )
    count_columns = [
        "sampled_loops",
        "convertible_loops",
        *FAILURE_COLUMNS.values(),
    ]
    # A share is undefined for a benchmark without sampled loops
    has_sampled_loops = pl.col("sampled_loops") != 0
    count_share = pl.col("convertible_loops") / pl.col("sampled_loops")

    return (
        summary.join(counts, on="benchmark", how="left")
        .with_columns(pl.col(count_columns).fill_null(0).cast(pl.Int64))
        .with_columns(count_share=pl.when(has_sampled_loops).then(count_share))
        .select(BENCHMARK_RESULT_COLUMNS)
        .sort("benchmark")
    )


def compute_overall_results(
    sampled_loops: pl.DataFrame,
    worksheet: pl.DataFrame,
    population_size: int,
) -> pl.DataFrame:
    """
    Compute the share of convertible loops and its confidence interval

    The share counts each distinct sampled loop once, even if several benchmarks
    share it. The interval holds the exact limits for the number of convertible loops
    among the population's distinct hot loops, divided by the population size.
    """
    verdicts = worksheet.select("loop_id", "verdict")
    counts = (
        sampled_loops.unique("loop_id")
        .join(verdicts, on="loop_id", how="left")
        .select(
            (pl.col("verdict") == Verdict.CONVERTIBLE)
            .sum()
            .cast(pl.Int64)
            .alias("convertible_loops"),
            pl.len().cast(pl.Int64).alias("sampled_loops"),
        )
    )
    convertible_count, sample_size = counts.row(0)
    # The share and its interval are undefined without sampled loops
    if sample_size != 0:
        count_share = convertible_count / sample_size
        interval = hypergeom_count_ci(
            convertible_count,
            sample_size,
            population_size,
            CONFIDENCE_LEVEL,
        )
        low = interval.low / population_size
        high = interval.high / population_size
    else:
        count_share = low = high = None

    return counts.select(
        pl.lit(count_share, pl.Float64).alias("count_share"),
        pl.lit(low, pl.Float64).alias("count_share_ci_low"),
        pl.lit(high, pl.Float64).alias("count_share_ci_high"),
        "convertible_loops",
        "sampled_loops",
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
    sampled_loops = sample_hot_loops(hot_loops, config.sample)
    summary = read_analysis_table(out_dir, SUMMARY_NAME, SUMMARY_SCHEMA)
    worksheet_path = out_dir / WORKSHEET_NAME
    worksheet = read_worksheet(worksheet_path)
    validate_worksheet(worksheet, sampled_loops)
    # Read first so an unreadable jar fails before any table is written
    jar_version = read_jar_version(benchmarks_dir / config.jar_name)

    benchmark_results = compute_benchmark_results(sampled_loops, summary, worksheet)
    population_size = hot_loops["loop_id"].n_unique()
    overall_results = compute_overall_results(sampled_loops, worksheet, population_size)
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
