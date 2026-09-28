#!/usr/bin/env -S uv run

"""
Aggregate benchmark results using non-parametric statistics.

This script recursively processes files matching:

    <root>/<scenario>/<jdk>/<implementation>/<benchmark>/measurements.csv

and produces the following outputs:

1. A `comparison.csv` in each variant directory, comparing the energy, time,
   power, temperature, and memory of each benchmark with the `loops` baseline.
   Comparisons include:
     - the median and interquartile range (IQR) of each implementation;
     - the difference in medians and its bootstrap confidence interval;
     - Cliff's delta and its confidence interval.

2. Total energy (package + DRAM) statistics, including:
   - Forest plots for each variant in the corresponding JDK directory.
   - Scenario-level heatmaps in the corresponding scenario directory.
   - Line plots of the change(s) in each benchmark across JDK versions, in
     each scenario directory.
   - An `energy_summary.csv` under `<root>`, giving the median of the
     benchmark medians of the change in total energy and its confidence
     interval, per variant and scenario. It also counts the comparisons whose
     median increased, decreased or changed inconclusively.
   - An `energy_summary_by_jdk.csv` under `<root>`, giving the same statistics
     per variant, scenario and JDK. It covers only the benchmarks with an
     estimate on every JDK in every scenario, so that every row covers the
     same benchmarks.

3. Time and total energy correlation consisting of:
   - A Spearman rank-correlation table under `<root>`, quantifying the
     relationship between the percentage changes in median time and total
     energy relative to the baseline. A benchmark recurs under every JDK and
     scenario, so each benchmark counts as one cluster (see `rank_corr`), and
     the table gives the standard error and confidence interval of each
     correlation as well as its number of benchmarks.
   - Scatter plots of those changes in each scenario directory.

Bootstrap resampling is deterministically seeded from the identity of each sample,
making results reproducible regardless of execution order or scheduling.
"""

import argparse
import functools
import logging
import os
import sys
import textwrap
from collections import defaultdict
from concurrent.futures import Executor
from pathlib import Path
from typing import NamedTuple

import polars as pl

from effect_sizes import (
    BOOTSTRAP_DEFAULT_N_RESAMPLES,
    ComparisonRow,
    build_long_time_and_energy_table,
    compare_metric,
    generate_energy_summary_by_jdk_table,
    generate_energy_summary_table,
    generate_energy_time_correlation_table,
)
from jobs import Job, JobKey, run_jobs, worker_pool
from measurements import (
    BASELINE_DIR_NAME,
    MEASUREMENTS_FILE_NAME,
    METRICS,
    SampleKey,
    collect_jdk_versions,
    common_benchmarks,
    is_non_baseline_dir,
    load_and_clean_measurements,
)
from plots import (
    FONT_DIR_ENV_VAR,
    FONT_FILES,
    TREND_PLOTS,
    ForestPlotJobKey,
    effect_size_heatmap_jobs,
    energy_time_scatter_plot_jobs,
    forest_plot_jobs,
    trend_plot_jobs,
)

logger = logging.getLogger(__name__)


# File names
LOG_FILE_NAME = "bench_stats.log"
COMPARISON_FILE_NAME = "comparison.csv"
ENERGY_TIME_CORRELATION_FILE_NAME = "energy_time_correlation.csv"
ENERGY_SUMMARY_FILE_NAME = "energy_summary.csv"
ENERGY_SUMMARY_BY_JDK_FILE_NAME = "energy_summary_by_jdk.csv"


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


# Comparison
def compare_benchmark(
    sample_key: SampleKey,
    baseline_df: pl.DataFrame,
    variant_df: pl.DataFrame,
    n_resamples: int,
) -> list[ComparisonRow]:
    """Compare every metric of one benchmark between a baseline and a variant"""
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
    measurements: dict[Path, pl.DataFrame],
    executor: Executor | None,
    n_resamples: int,
) -> tuple[list[pl.DataFrame], int]:
    """
    Compare each benchmark variant with the baseline and store the results in a CSV in the variant's directory.

    `measurements` holds the loaded measurements by benchmark directory. A benchmark
    whose measurements are missing on either side is left out.

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
                if baseline_dir / benchmark in measurements
                and variant_dir / benchmark in measurements
            ]

    # Each job seeds its bootstraps from the sample it compares, so its results are
    # independent of the process that runs it. The results are returned in job order.
    comparison_jobs: dict[ComparisonJobKey, Job[list[ComparisonRow]]] = {}
    for job_key in job_keys:
        baseline_benchmark_dir = job_key.baseline_dir / job_key.benchmark
        variant_benchmark_dir = job_key.variant_dir / job_key.benchmark
        comparison_jobs[job_key] = (
            f"comparison of '{variant_benchmark_dir}'",
            functools.partial(
                compare_benchmark,
                SampleKey.from_benchmark_directory(variant_benchmark_dir),
                measurements[baseline_benchmark_dir],
                measurements[variant_benchmark_dir],
                n_resamples,
            ),
        )

    job_rows, n_failures = run_jobs(comparison_jobs, executor)
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
def overall_stats_jobs(
    root_dir: Path,
    scenario_dirs: list[Path],
    comparisons: list[pl.DataFrame],
) -> dict[JobKey, Job[object]]:
    """Return the jobs that write the overview tables and draw the overview plots"""
    correlation_path = root_dir / ENERGY_TIME_CORRELATION_FILE_NAME
    energy_summary_path = root_dir / ENERGY_SUMMARY_FILE_NAME
    energy_summary_by_jdk_path = root_dir / ENERGY_SUMMARY_BY_JDK_FILE_NAME
    # Remove any stale tables, so none survives even if these cannot be written
    correlation_path.unlink(missing_ok=True)
    energy_summary_path.unlink(missing_ok=True)
    energy_summary_by_jdk_path.unlink(missing_ok=True)

    jobs: dict[JobKey, Job[object]] = {}
    if not comparisons:
        logger.error(
            "No comparison data found across %s",
            [d.name for d in scenario_dirs],
        )
        return jobs

    comparison_data = pl.concat(comparisons, how="vertical")

    jdk_versions = collect_jdk_versions(scenario_dirs)
    jobs[energy_summary_path] = (
        f"energy summary table '{energy_summary_path}'",
        functools.partial(
            generate_energy_summary_table,
            energy_summary_path,
            comparison_data,
        ),
    )
    jobs[energy_summary_by_jdk_path] = (
        f"energy summary by JDK table '{energy_summary_by_jdk_path}'",
        functools.partial(
            generate_energy_summary_by_jdk_table,
            energy_summary_by_jdk_path,
            comparison_data,
            jdk_versions,
        ),
    )

    # Merged as items, which mypy accepts for keys of any `JobKey` type
    heatmap_jobs = effect_size_heatmap_jobs(root_dir, comparison_data, jdk_versions)
    jobs |= heatmap_jobs.items()
    for metric, effect_size in TREND_PLOTS:
        trend_jobs = trend_plot_jobs(
            root_dir, comparison_data, metric, effect_size, jdk_versions
        )
        jobs |= trend_jobs.items()

    long_df = build_long_time_and_energy_table(comparison_data)
    if long_df.is_empty():
        logger.warning("No energy-time data found; skipping energy-time correlation")
        return jobs

    jobs[correlation_path] = (
        f"energy-time correlation table '{correlation_path}'",
        functools.partial(
            generate_energy_time_correlation_table,
            correlation_path,
            long_df,
        ),
    )

    scatter_jobs = energy_time_scatter_plot_jobs(root_dir, long_df, jdk_versions)
    jobs |= scatter_jobs.items()
    return jobs


def generate_overall_stats(
    root_dir: Path,
    scenario_dirs: list[Path],
    comparisons: list[pl.DataFrame],
    executor: Executor | None,
) -> int:
    """Generate overview statistics from the comparisons of every scenario, and return the number of jobs that failed"""
    _, n_failures = run_jobs(
        overall_stats_jobs(root_dir, scenario_dirs, comparisons),
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

    # 1. Find measurement files and warn about those outside a benchmark directory
    measurement_files = sorted(root_dir.rglob(MEASUREMENTS_FILE_NAME))
    logger.info("Found %d %s files", len(measurement_files), MEASUREMENTS_FILE_NAME)

    for measurements_path in measurement_files:
        if len(measurements_path.relative_to(root_dir).parts) != 4:
            logger.warning(
                "Skipping '%s' because it does not match the expected "
                "directory structure (<root_dir>/<jdk>/<implementation>/<benchmark>/)...",
                measurements_path,
            )

    # 2. Load the measurements of every benchmark directory once, so each file is checked once
    benchmark_dirs = [
        benchmark_dir
        for jdk_dir in sorted(filter(Path.is_dir, root_dir.iterdir()))
        for implementation_dir in sorted(filter(Path.is_dir, jdk_dir.iterdir()))
        for benchmark_dir in sorted(filter(Path.is_dir, implementation_dir.iterdir()))
    ]
    loaded, n_load_failures = run_jobs(
        {
            benchmark_dir: (
                f"loading of '{benchmark_dir / MEASUREMENTS_FILE_NAME}'",
                functools.partial(
                    load_and_clean_measurements,
                    benchmark_dir / MEASUREMENTS_FILE_NAME,
                ),
            )
            for benchmark_dir in benchmark_dirs
        },
        executor,
    )
    measurements = {
        benchmark_dir: df for benchmark_dir, df in loaded.items() if df is not None
    }

    # 3. Generate comparative statistics
    comparisons, n_comparison_failures = generate_comparisons(
        root_dir,
        measurements,
        executor,
        n_resamples,
    )

    # 4. Log completion
    logger.info(
        "Processing complete. Processed %d measurement files.",
        len(measurement_files),
    )
    return comparisons, n_load_failures + n_comparison_failures


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
        epilog="environment variables:\n"
        + textwrap.fill(
            f"directory containing {' and '.join(FONT_FILES)}, the fonts that the "
            "plots are laid out and exported in. If unset, or if the fonts are "
            "missing, plots are laid out in matplotlib's default font.",
            width=80,
            initial_indent=f"  {FONT_DIR_ENV_VAR}  ",
            subsequent_indent=" " * (len(FONT_DIR_ENV_VAR) + 4),
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
