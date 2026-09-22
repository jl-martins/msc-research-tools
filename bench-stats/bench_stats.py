#!/usr/bin/env -S uv run

"""
Aggregate benchmark results using non-parametric statistics.

This script recursively processes files matching:

    <root>/<scenario>/<jdk>/<implementation>/<benchmark>/measurements.csv

and produces the following outputs:

1. Measurement summaries for energy, time, temperature, and memory,
   namely:
   - A `summary.csv` in each benchmark directory containing the median,
     interquartile range (IQR), and sample count.
   - A `summary.csv` in each implementation directory containing the same
     statistics aggregated across that implementation's benchmarks.
   - A `comparison.csv` comparing each variant with the `loops` baseline.
     Comparisons include:
       - the difference in medians and its bootstrap confidence interval;
       - Cliff's delta and its confidence interval.

2. Total energy (package + DRAM) statistics, including:
   - Forest plots for each variant in the corresponding JDK directory.
   - Scenario-level heatmaps in the corresponding scenario directory.
   - Line plots of the change(s) in each benchmark across JDK versions, in
     each scenario directory.

3. Time and total energy correlation consisting of:
   - A Spearman rank-correlation table under `<root>`, quantifying the
     relationship between the percentage changes in median time and total
     energy relative to the baseline. A benchmark recurs under every JDK and
     scenario, so each benchmark counts as one cluster (see `rank_corr`), and
     the table gives the standard error and confidence interval of each
     correlation as well as its number of benchmarks.
   - Scatter plots of those changes in each scenario directory.

Bootstrap resampling is deterministically seeded from the resampled data,
making results reproducible regardless of execution order or scheduling.
"""

import argparse
import functools
import itertools
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import Executor
from pathlib import Path
from typing import NamedTuple

import polars as pl

from effect_sizes import (
    BOOTSTRAP_DEFAULT_N_RESAMPLES,
    IQR_INTERPOLATION_METHOD,
    ComparisonRow,
    build_long_time_and_energy_table,
    compare_metric,
    generate_energy_time_correlation_table,
)
from jobs import Job, JobKey, run_jobs, worker_pool
from measurements import (
    BASELINE_DIR_NAME,
    IDENTITY_COLUMNS,
    MEASUREMENTS_FILE_NAME,
    METRICS,
    SampleKey,
    collect_jdk_versions,
    common_benchmarks,
    is_non_baseline_dir,
    load_and_clean_measurements,
)
from plots import (
    TREND_PLOTS,
    ForestPlotJobKey,
    effect_size_heatmap_jobs,
    energy_time_scatter_plot_jobs,
    forest_plot_jobs,
    trend_plot_jobs,
)

logger = logging.getLogger(__name__)


# Summary Statistics
SUMMARY_STATS = ["Median", "IQR", "Count"]

SUMMARY_COLUMNS = IDENTITY_COLUMNS + [
    f"{metric}_{stat}" for metric, stat in itertools.product(METRICS, SUMMARY_STATS)
]

# Fewer runs than this make a summary statistic unreliable
MIN_ADVISED_SAMPLE_SIZE = 10

# File names
LOG_FILE_NAME = "bench_stats.log"
SUMMARY_FILE_NAME = "summary.csv"
COMPARISON_FILE_NAME = "comparison.csv"
ENERGY_TIME_CORRELATION_FILE_NAME = "energy_time_correlation.csv"


class ComparisonJobKey(NamedTuple):
    """Identifies the job that compares a benchmark with the baseline"""

    baseline_dir: Path
    variant_dir: Path
    benchmark: str


# Logging
def setup_logging(root_dir: Path, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    # Root Logger Setup
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    logging.captureWarnings(True)

    # Console Handler
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    log_file = root_dir / LOG_FILE_NAME
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


# Summarization
def generate_benchmark_summary(measurements_path: Path) -> pl.DataFrame | None:
    """Summarize raw measurements, store the result in a CSV and return it"""
    summary_path = measurements_path.parent / SUMMARY_FILE_NAME
    # Remove any stale summary, so none survives even if this one cannot be written
    summary_path.unlink(missing_ok=True)

    active_runs_df = load_and_clean_measurements(measurements_path)
    if active_runs_df is None or active_runs_df.is_empty():
        logger.warning("Skipping empty data frame from '%s'...", measurements_path)
        return None

    logger.info("Generating summary for '%s'", measurements_path)
    aggs = []
    for metric in METRICS:
        p25 = pl.col(metric).quantile(0.25, interpolation=IQR_INTERPOLATION_METHOD)
        p75 = pl.col(metric).quantile(0.75, interpolation=IQR_INTERPOLATION_METHOD)
        aggs.extend(
            [
                pl.col(metric).median().alias(f"{metric}_Median"),
                (p75 - p25).alias(f"{metric}_IQR"),
                pl.col(metric).count().alias(f"{metric}_Count"),
            ]
        )

    summary_df = active_runs_df.group_by(IDENTITY_COLUMNS).agg(aggs)
    if len(summary_df) > 1:
        conflicting_values = summary_df.select(IDENTITY_COLUMNS).to_dicts()
        formatted_conflicts = "\n  - ".join([str(row) for row in conflicting_values])
        raise ValueError(
            f"Multiple combinations of {IDENTITY_COLUMNS} found in {measurements_path}.\n"
            f"Expected 1 unique combination, but found {len(summary_df)}:\n"
            f"  - {formatted_conflicts}"
        )

    if summary_df.is_empty():
        logger.warning("Empty summary for '%s'", measurements_path)
        return None

    for metric in METRICS:
        sample_size = summary_df.get_column(f"{metric}_Count")[0]
        if sample_size < MIN_ADVISED_SAMPLE_SIZE:
            logger.warning(
                "%s: %s has only %d samples", measurements_path, metric, sample_size
            )

    summary_df = summary_df.select(SUMMARY_COLUMNS)
    summary_df.write_csv(summary_path)
    logger.info("Created summary '%s'", summary_path)
    return summary_df


def generate_combined_summary(
    implementation_dir: Path,
    summaries: list[pl.DataFrame],
) -> None:
    """Combine the benchmark summaries of an implementation directory into a single file"""
    output_path = implementation_dir / SUMMARY_FILE_NAME
    # Remove any stale summary, so none survives even if this one cannot be written
    output_path.unlink(missing_ok=True)

    if not summaries:
        logger.error("No summaries were generated in '%s'", implementation_dir)
        return

    combined_df = pl.concat(summaries, how="vertical")
    combined_df.write_csv(output_path)
    logger.info("Created combined summary '%s'", output_path)


def load_comparison_pair(
    baseline_dir: Path,
    variant_dir: Path,
    benchmark: str,
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Load measurements for a benchmark from both baseline and variant dirs"""
    baseline_measurements_path = baseline_dir / benchmark / MEASUREMENTS_FILE_NAME
    variant_measurements_path = variant_dir / benchmark / MEASUREMENTS_FILE_NAME

    baseline_df = load_and_clean_measurements(baseline_measurements_path)
    variant_df = load_and_clean_measurements(variant_measurements_path)

    if baseline_df is None or variant_df is None:
        return None
    else:
        return baseline_df, variant_df


def compare_benchmark(key: ComparisonJobKey, n_resamples: int) -> list[ComparisonRow]:
    """Compare every metric of one benchmark between a baseline and a variant"""
    comparison_pair = load_comparison_pair(
        key.baseline_dir,
        key.variant_dir,
        key.benchmark,
    )
    if comparison_pair is None:
        return []

    baseline_df, variant_df = comparison_pair
    sample_key = SampleKey.from_benchmark_directory(key.variant_dir / key.benchmark)

    comparison_rows: list[ComparisonRow] = []
    for metric in METRICS:
        x = baseline_df[metric].drop_nulls().to_numpy()
        y = variant_df[metric].drop_nulls().to_numpy()
        row = compare_metric(sample_key, metric, x, y, n_resamples)
        if row is not None:
            comparison_rows.append(row)

    return comparison_rows


def build_comparison_frame(
    baseline_dir: Path,
    variant_dir: Path,
    comparison_rows: list[ComparisonRow],
) -> pl.DataFrame | None:
    """Order the rows of one variant's comparison, or `None` if it has none"""
    if not comparison_rows:
        logger.warning(
            "Empty comparison result detected for the following directories:\n"
            "  * Baseline: '%s'\n"
            "  * Variant: '%s'\n",
            baseline_dir, variant_dir
        )  # fmt: skip
        return None

    return pl.DataFrame(comparison_rows).sort(["Program", "Metric", "Variant"])


def generate_comparisons(
    root_dir: Path,
    executor: Executor | None,
    n_resamples: int,
) -> tuple[list[pl.DataFrame], int]:
    """
    Compare each benchmark variant with the baseline and store the results in a CSV in the variant's directory.

    Return the comparisons of the variants, each tagged with its scenario and JDK,
    and the number of jobs that failed.
    """
    variant_pairs: list[tuple[Path, Path]] = []
    job_keys: list[ComparisonJobKey] = []
    for jdk_dir in sorted(filter(Path.is_dir, root_dir.iterdir())):
        baseline_dir = jdk_dir / BASELINE_DIR_NAME

        if not baseline_dir.exists():
            logger.warning(
                "Baseline directory '%s' not found in '%s'.\n"
                "Comparisons will be skipped for this JDK...",
                BASELINE_DIR_NAME, jdk_dir
            )  # fmt: skip
            continue

        for variant_dir in sorted(filter(is_non_baseline_dir, jdk_dir.iterdir())):
            benchmarks = common_benchmarks(baseline_dir, variant_dir)
            logger.info(
                "Comparing %d common benchmarks between '%s' and '%s'...",
                len(benchmarks), baseline_dir, variant_dir
            )  # fmt: skip
            variant_pairs.append((baseline_dir, variant_dir))
            job_keys += [
                ComparisonJobKey(baseline_dir, variant_dir, benchmark)
                for benchmark in benchmarks
            ]

    # Each job seeds its bootstraps from the sample it compares, so its results are
    # independent of the process that runs it. The results are returned in job order.
    job_rows, n_failures = run_jobs(
        {
            job_key: (
                f"comparison of '{job_key.variant_dir / job_key.benchmark}'",
                functools.partial(compare_benchmark, job_key, n_resamples),
            )
            for job_key in job_keys
        },
        executor,
    )
    rows_by_variant: dict[Path, list[ComparisonRow]] = defaultdict(list)
    for job_key, comparison_rows in job_rows.items():
        rows_by_variant[job_key.variant_dir] += comparison_rows

    comparisons: list[pl.DataFrame] = []
    plot_jobs: dict[ForestPlotJobKey, Job[None]] = {}
    for baseline_dir, variant_dir in variant_pairs:
        comparison_path = variant_dir / COMPARISON_FILE_NAME
        # Remove any stale comparison, so none survives even if this one cannot be written
        comparison_path.unlink(missing_ok=True)

        result_df = build_comparison_frame(
            baseline_dir,
            variant_dir,
            rows_by_variant[variant_dir],
        )
        if result_df is None:
            continue

        result_df.write_csv(comparison_path)
        logger.info("Created statistical comparison '%s'", comparison_path)
        comparisons.append(
            result_df.with_columns(
                Scenario=pl.lit(root_dir.name),
                JDK=pl.lit(variant_dir.parent.name),
            )
        )

        plot_jobs |= forest_plot_jobs(variant_dir, result_df)

    _, n_plot_failures = run_jobs(plot_jobs, executor)
    return comparisons, n_failures + n_plot_failures


# Cross-Scenario Overview Statistics
def overall_plot_jobs(
    root_dir: Path,
    scenario_dirs: list[Path],
    comparisons: list[pl.DataFrame],
) -> dict[JobKey, Job[None]]:
    """Return the jobs responsible for producing overview statistics"""
    correlation_path = root_dir / ENERGY_TIME_CORRELATION_FILE_NAME
    # Remove any stale table, so none survives even if this one cannot be written
    correlation_path.unlink(missing_ok=True)

    plot_jobs: dict[JobKey, Job[None]] = {}
    if not comparisons:
        logger.error(
            "No comparison data found across %s",
            [d.name for d in scenario_dirs],
        )
        return plot_jobs

    comparison_data = pl.concat(comparisons, how="vertical")

    jdk_versions = collect_jdk_versions(scenario_dirs)
    # Merged as items, which mypy accepts for keys of any `JobKey` type
    heatmap_jobs = effect_size_heatmap_jobs(root_dir, comparison_data, jdk_versions)
    plot_jobs |= heatmap_jobs.items()
    for metric in TREND_PLOTS:
        trend_jobs = trend_plot_jobs(root_dir, comparison_data, metric, jdk_versions)
        plot_jobs |= trend_jobs.items()

    long_df = build_long_time_and_energy_table(comparison_data)
    if long_df.is_empty():
        logger.warning("No energy-time data found; skipping energy-time correlation")
        return plot_jobs

    correlation_df = generate_energy_time_correlation_table(long_df)
    correlation_df.write_csv(correlation_path)
    logger.info("Created energy-time correlation table '%s'", correlation_path)

    scatter_jobs = energy_time_scatter_plot_jobs(root_dir, long_df, jdk_versions)
    plot_jobs |= scatter_jobs.items()
    return plot_jobs


def generate_overall_stats(
    root_dir: Path,
    scenario_dirs: list[Path],
    comparisons: list[pl.DataFrame],
    executor: Executor | None,
) -> int:
    """Generate overview statistics from the comparisons of every scenario, and return the number of jobs that failed"""
    _, n_failures = run_jobs(
        overall_plot_jobs(root_dir, scenario_dirs, comparisons),
        executor,
    )
    return n_failures


# Orchestration
def generate_stats(
    root_dir: Path,
    executor: Executor | None,
    n_resamples: int,
) -> tuple[list[pl.DataFrame], int]:
    """
    Generate benchmark statistics from raw measurements.

    Return the comparisons of the variants, each tagged with its scenario and JDK,
    and the number of jobs that failed.
    """
    if not root_dir.exists():
        logger.error("Root directory '%s' does not exist", root_dir)
        return [], 0

    logger.info("Processing root directory '%s'...", root_dir)

    # 1. Generate individual benchmark summaries
    measurement_files = sorted(root_dir.rglob(MEASUREMENTS_FILE_NAME))
    logger.info("Found %d %s files", len(measurement_files), MEASUREMENTS_FILE_NAME)

    benchmark_files: list[Path] = []
    for measurements_path in measurement_files:
        if len(measurements_path.relative_to(root_dir).parts) == 4:
            benchmark_files.append(measurements_path)
        else:
            logger.warning(
                "Skipping '%s' because it does not match the expected "
                "directory structure (<root_dir>/<jdk>/<implementation>/<benchmark>/)...",
                measurements_path,
            )

    benchmark_summaries, n_summary_failures = run_jobs(
        {
            measurements_path: (
                f"summary of '{measurements_path}'",
                functools.partial(generate_benchmark_summary, measurements_path),
            )
            for measurements_path in benchmark_files
        },
        executor,
    )
    summaries: dict[Path, list[pl.DataFrame]] = defaultdict(list)
    for measurements_path, summary in benchmark_summaries.items():
        if summary is not None:
            summaries[measurements_path.parent.parent].append(summary)

    # 2. Concatenate benchmark summaries per implementation (loops, sequential streams or parallel streams)
    implementation_dirs = [
        implementation_dir
        for jdk_dir in sorted(d for d in root_dir.iterdir() if d.is_dir())
        for implementation_dir in sorted(d for d in jdk_dir.iterdir() if d.is_dir())
    ]
    _, n_combined_failures = run_jobs(
        {
            implementation_dir: (
                f"combined summary of '{implementation_dir}'",
                functools.partial(
                    generate_combined_summary,
                    implementation_dir,
                    summaries[implementation_dir],
                ),
            )
            for implementation_dir in implementation_dirs
        },
        executor,
    )

    # 3. Generate comparative statistics
    comparisons, n_comparison_failures = generate_comparisons(
        root_dir,
        executor,
        n_resamples,
    )

    # 4. Log completion
    logger.info(
        "Processing complete. Processed %d measurement files.",
        len(measurement_files),
    )
    n_failures = n_summary_failures + n_combined_failures + n_comparison_failures
    return comparisons, n_failures


def run(
    root_dir: Path,
    n_jobs: int,
    n_resamples: int = BOOTSTRAP_DEFAULT_N_RESAMPLES,
) -> int:
    """
    Process every scenario subdirectory of `root_dir` and generate overview statistics.

    Run up to `n_jobs` jobs at once, bootstrapping every interval from `n_resamples`
    resamples, and return the number of jobs that failed.
    """
    if not root_dir.exists():
        logger.error("Root directory '%s' does not exist", root_dir)
        return 0

    scenario_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    logger.info(
        "Processing %d scenario director%s: %s over %d process%s",
        len(scenario_dirs), "y" if len(scenario_dirs) == 1 else "ies",
        [d.name for d in scenario_dirs], n_jobs, "" if n_jobs == 1 else "es",
    )  # fmt: skip
    with worker_pool(n_jobs) as executor:
        comparisons: list[pl.DataFrame] = []
        n_failures = 0
        for scenario_dir in scenario_dirs:
            scenario_comparisons, n_scenario_failures = generate_stats(
                scenario_dir,
                executor,
                n_resamples,
            )
            comparisons += scenario_comparisons
            n_failures += n_scenario_failures

        n_failures += generate_overall_stats(
            root_dir,
            scenario_dirs,
            comparisons,
            executor,
        )

    return n_failures


def positive_int(value: str) -> int:
    """Parse a command line argument that must be a positive integer"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive, was {number}")

    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark statistics for Master's Dissertation",
        color=False,
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="root directory for benchmark results (default: current working directory)",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        default=os.process_cpu_count() or 1,
        help="worker processes to spread the statistics and plots over (default: %(default)s)",
    )
    parser.add_argument(
        "--resamples",
        type=positive_int,
        default=BOOTSTRAP_DEFAULT_N_RESAMPLES,
        help="resamples of every bootstrap confidence interval (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    args = parser.parse_args()

    root_dir = args.root_dir.resolve()
    setup_logging(root_dir, args.verbose)

    logger.info("Using root directory '%s'", root_dir)
    n_failures = run(root_dir, args.jobs, args.resamples)
    if n_failures:
        logger.error("%d job(s) failed; see the errors above", n_failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
