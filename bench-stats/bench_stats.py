#!/usr/bin/env -S uv run

"""
Benchmark Summary Aggregator with Non-Parametric Statistics

This script recursively processes benchmark result directories and generates:
1. Per-benchmark summaries (Median and IQR for 7 metrics)
2. Per-implementation aggregated summaries
3. Statistical comparisons (Cliff's Delta and difference of medians) against baseline
"""

import argparse
import functools
import itertools
import logging
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import DegenerateDataWarning, bootstrap, iqr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Schemas
METRICS_SCHEMA = {
    "Package": pl.Float64,
    "Core": pl.Float64,
    "Uncore": pl.Float64,
    "DRAM": pl.Float64,
    "PackageAndDRAM": pl.Float64,
    "Time": pl.Float64,
    "Temperature": pl.Float64,
    "Memory": pl.Float64,
}

IDENTITY_COLUMNS_SCHEMA = {
    "Language": pl.Utf8,
    "Program": pl.Utf8,
    "Powercap": pl.Utf8,
}

MEASUREMENTS_SCHEMA = {
    **IDENTITY_COLUMNS_SCHEMA,
    **METRICS_SCHEMA,
}

RAW_MEASUREMENTS_SCHEMA = {
    k: v for k, v in MEASUREMENTS_SCHEMA.items() if k != "PackageAndDRAM"
}

# Summary Statistics
SUMMARY_STATS = ["Median", "IQR", "Count"]

# Columns
METRICS = list(METRICS_SCHEMA.keys())
IDENTITY_COLUMNS = list(IDENTITY_COLUMNS_SCHEMA.keys())
MEASUREMENTS_COLUMNS = list(MEASUREMENTS_SCHEMA.keys())

SUMMARY_COLUMNS = IDENTITY_COLUMNS + [
    f"{metric}_{stat}" for metric, stat in itertools.product(METRICS, SUMMARY_STATS)
]

# Idle runs prefix
IDLE_RUN_PREFIX = "idle_"

# File and directory names
MEASUREMENTS_FILE_NAME = "measurements.csv"
SUMMARY_FILE_NAME = "summary.csv"
COMPARISON_FILE_NAME = "comparison.csv"
BASELINE_DIR_NAME = "loops"

# Significance level for statistical hypothesis testing
ALPHA = 0.05

# Valid methods for computing bootstrap confidence intervals
type BootstrapMethod = Literal["percentile", "basic", "bca", "BCa"]

# Primary method for computing bootstrap confidence intervals
PRIMARY_BOOTSTRAP_METHOD: BootstrapMethod = "BCa"

# Fallback bootstrap method for degenerate data
FALLBACK_BOOTSTRAP_METHOD: BootstrapMethod = "percentile"

# Random number generator with a fixed seed for reproducible results
RNG = np.random.default_rng(seed=42)

# Numeric types
type IntOrFloat = np.integer | np.floating

# Effect sizes
type EffectSizeClass = Literal["negligible", "small", "medium", "large"]

# Confidence interval named tuple
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class TwoSampleStatistic(Protocol):
    def __call__(
        self,
        x: npt.NDArray[IntOrFloat],
        y: npt.NDArray[IntOrFloat],
        /,
        *,
        axis: int = 0,
    ) -> IntOrFloat | npt.NDArray[IntOrFloat]: ...


class ComparisonRow(TypedDict):
    """Data structure for the statistical comparison of a specific metric (e.g., time, energy, etc.)"""

    Program: str
    Metric: str
    Variant: str
    Loops_Median: float
    Variant_Median: float
    Loops_IQR: float
    Variant_IQR: float
    Median_Difference: float
    Median_Difference_Percent: float
    Median_Difference_Percent_CI_Low: float
    Median_Difference_Percent_CI_High: float
    Cliffs_Delta: float
    Cliffs_Delta_CI_Low: float
    Cliffs_Delta_CI_High: float
    Effect_Size_Class: EffectSizeClass
    MW_p_value: float
    Significant: bool


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
    log_file = root_dir / "bench-stats.log"
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


# I/O
def is_non_baseline_dir(path: Path) -> bool:
    """Return `True` if `path` is a directory whose name differs from `BASELINE_DIR_NAME`; `False` otherwise"""
    return path.is_dir() and path.name != BASELINE_DIR_NAME


def normalize_measurements_columns(measurements_path: Path) -> None:
    """Add the derived `PackageAndDRAM` column if missing and sort columns per `MEASUREMENTS_SCHEMA`"""
    columns = pl.scan_csv(measurements_path).collect_schema().names()
    if columns == MEASUREMENTS_COLUMNS:
        return

    schema = (
        MEASUREMENTS_SCHEMA if "PackageAndDRAM" in columns else RAW_MEASUREMENTS_SCHEMA
    )
    df = pl.read_csv(measurements_path, schema=schema)
    if "PackageAndDRAM" not in df.columns:
        df = df.with_columns(
            (pl.col("Package") + pl.col("DRAM")).alias("PackageAndDRAM")
        )

    tmp_path = measurements_path.with_name(f"{measurements_path.name}.tmp")
    df.select(MEASUREMENTS_COLUMNS).write_csv(tmp_path)
    tmp_path.replace(measurements_path)


@functools.lru_cache(maxsize=256)
def load_and_clean_measurements(measurements_path: Path) -> pl.DataFrame | None:
    """Load raw measurements, clean them and return the result as a data frame"""
    try:
        normalize_measurements_columns(measurements_path)
        df = pl.read_csv(measurements_path, schema=MEASUREMENTS_SCHEMA)
        active_runs_df = df.filter(~pl.col("Program").str.starts_with(IDLE_RUN_PREFIX))
        if active_runs_df.is_empty():
            logger.warning(
                "Skipping '%s' because it only contains idle runs...", measurements_path
            )
            return None

        # Drop rows where essential metrics are missing
        len_before_drop_nulls = len(active_runs_df)
        active_runs_df = active_runs_df.drop_nulls(subset=["Time", "Package"])
        null_count = len_before_drop_nulls - len(active_runs_df)
        if null_count:
            logger.warning(
                "Dropped %d rows with missing Time/Package from '%s'",
                null_count,
                measurements_path,
            )

        return active_runs_df

    except OSError, pl.exceptions.PolarsError:
        logger.exception("Error when reading or cleaning '%s'", measurements_path)
        return None


# Summarization
def generate_benchmark_summary(measurements_path: Path) -> Path | None:
    """Summarize raw measurements and store the result in a CSV"""
    active_runs_df = load_and_clean_measurements(measurements_path)
    if active_runs_df is None or active_runs_df.is_empty():
        logger.warning("Skipping empty data frame from '%s'...", measurements_path)
        return None

    logger.info("Generating summary for '%s'", measurements_path)
    try:
        aggs = []
        for metric in METRICS:
            p25 = pl.col(metric).quantile(0.25)
            p75 = pl.col(metric).quantile(0.75)
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
            formatted_conflicts = "\n  - ".join(
                [str(row) for row in conflicting_values]
            )
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
            if sample_size < 10:
                logger.warning(
                    "%s: %s has only %d samples", measurements_path, metric, sample_size
                )

        summary_df = summary_df.select(SUMMARY_COLUMNS)
        summary_path = measurements_path.parent / SUMMARY_FILE_NAME
        summary_df.write_csv(summary_path)
        logger.info("Created summary '%s'", summary_path)
        return summary_path

    except OSError, ValueError, pl.exceptions.PolarsError:
        logger.exception("Failed to create summary for %s", measurements_path)
        return None


def generate_combined_summary(implementation_dir: Path) -> None:
    """Combine all individual benchmark summaries within an implementation directory into a single file"""
    try:
        summary_files: list[Path] = [
            d / SUMMARY_FILE_NAME
            for d in implementation_dir.iterdir()
            if d.is_dir() and (d / SUMMARY_FILE_NAME).exists()
        ]

        if not summary_files:
            logger.error("No summary files were found in '%s'", implementation_dir)
            return

        summary_dfs = [pl.read_csv(f) for f in summary_files]
        combined_df = pl.concat(summary_dfs, how="vertical")

        output_path = implementation_dir / SUMMARY_FILE_NAME
        combined_df.write_csv(output_path)
        logger.info("Created combined summary '%s'", output_path)

    except OSError, ValueError, pl.exceptions.PolarsError:
        logger.exception(
            "Failed to create combined summary in '%s'", implementation_dir
        )


# Statistical Comparison
def classify_effect_size(cliffs_d: float) -> EffectSizeClass:
    """Classify Cliff's Delta effect size according to Romano et al., 2006"""
    abs_d = abs(cliffs_d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def median_difference_percent(
    x: npt.NDArray[IntOrFloat],
    y: npt.NDArray[IntOrFloat],
    axis: int = 0,
) -> np.floating | npt.NDArray[np.floating]:
    """Return `median(y) - median(x)`."""
    med_x = np.median(x, axis=axis)
    med_y = np.median(y, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (med_y - med_x) / med_x * 100.0


def cliffs_delta(
    x: npt.NDArray[IntOrFloat],
    y: npt.NDArray[IntOrFloat],
    u: np.floating | npt.NDArray[np.floating] | None = None,
    axis: int = 0,
) -> np.floating | npt.NDArray[np.floating]:
    """
    Return Cliff's delta from samples `x` and `y`.

    If `u`, the Mann–Whitney U statistic, is specified, it can be either:
    - a scalar (single comparison), or
    - an array of U values computed along `axis`.

    The output matches the shape of the broadcasted comparisons
    (i.e., `u` with the comparison axis removed).
    """
    if u is None:
        u, _ = mannwhitneyu(x, y, alternative="two-sided", axis=axis)

    u = np.asarray(u)
    nx = x.shape[axis]
    ny = y.shape[axis]
    return (2 * u) / (nx * ny) - 1


def two_sample_bootstrap_ci(
    x: npt.NDArray[IntOrFloat],
    y: npt.NDArray[IntOrFloat],
    statistic: TwoSampleStatistic,
    n_resamples: int = 9_999,
) -> ConfidenceInterval:
    """Return the bootstrap confidence interval for a custom two-sample statistic."""
    with (
        warnings.catch_warnings(record=False),
        # Suppress expected warnings from zero-variance inputs
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("error", DegenerateDataWarning)
        try:
            res = bootstrap(
                data=(x, y),
                statistic=statistic,
                vectorized=True,
                n_resamples=n_resamples,
                confidence_level=1 - ALPHA,
                method=PRIMARY_BOOTSTRAP_METHOD,
                random_state=RNG,
            )
        except DegenerateDataWarning:
            logger.info(
                "Degenerate data encountered. Falling back to '%s' bootstrap...",
                FALLBACK_BOOTSTRAP_METHOD,
            )
            # As of SciPy 1.17.1, `bootstrap` emits `DegenerateDataWarning` when any
            # confidence interval bound is NaN, regardless of the selected bootstrap method.
            # Since the primary BCa attempt has already detected and handled this condition,
            # suppress the warning during the fallback to avoid duplicate logs.
            with warnings.catch_warnings(
                record=False,
                action="ignore",
                category=DegenerateDataWarning,
            ):
                res = bootstrap(
                    data=(x, y),
                    statistic=statistic,
                    vectorized=True,
                    n_resamples=n_resamples,
                    confidence_level=1 - ALPHA,
                    method=FALLBACK_BOOTSTRAP_METHOD,
                    random_state=RNG,
                )

    return res.confidence_interval


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


def compare_metric(
    benchmark: str,
    metric: str,
    variant: str,
    baseline_values: npt.NDArray[IntOrFloat],
    variant_values: npt.NDArray[IntOrFloat],
) -> ComparisonRow | None:
    """Compute all comparative statistics for a single metric pair"""
    x = baseline_values
    y = variant_values

    if len(x) < 2 or len(y) < 2:
        logger.warning(
            "%s/%s: insufficient samples (baseline: %d, %s: %d)",
            benchmark, metric, len(x), variant, len(y),
        )  # fmt: skip
        return None

    med_x, med_y = np.median(x), np.median(y)
    med_diff = med_y - med_x
    med_diff_pct = (med_diff / med_x) * 100 if med_x != 0 else np.nan
    med_diff_pct_ci = two_sample_bootstrap_ci(x, y, median_difference_percent)
    iqr_x, iqr_y = iqr(x), iqr(y)

    # Pass the variant as the first sample so that the sign of Cliff's delta
    # indicates the direction of the difference between variant and baseline.
    u, p_val = mannwhitneyu(y, x, alternative="two-sided")
    cliffs_d = float(cliffs_delta(y, x, u))
    cliffs_d_ci = two_sample_bootstrap_ci(y, x, cliffs_delta)

    significant = bool(p_val <= ALPHA)
    if significant:
        logger.info(
            "%s/%s: significant difference detected (p=%.4g, Cliff's Delta=%.3f)",
            benchmark, metric, p_val, cliffs_d,
        )  # fmt: skip

    row: ComparisonRow = {
        "Program": benchmark,
        "Metric": metric,
        "Variant": variant,
        "Loops_Median": med_x,
        "Variant_Median": med_y,
        "Loops_IQR": iqr_x,
        "Variant_IQR": iqr_y,
        "Median_Difference": med_diff,
        "Median_Difference_Percent": med_diff_pct,
        "Median_Difference_Percent_CI_Low": med_diff_pct_ci.low,
        "Median_Difference_Percent_CI_High": med_diff_pct_ci.high,
        "Cliffs_Delta": cliffs_d,
        "Cliffs_Delta_CI_Low": cliffs_d_ci.low,
        "Cliffs_Delta_CI_High": cliffs_d_ci.high,
        "Effect_Size_Class": classify_effect_size(cliffs_d),
        "MW_p_value": float(p_val),
        "Significant": significant,
    }

    return row


def apply_bh_correction_per_metric(comparison_df: pl.DataFrame) -> pl.DataFrame:
    """Apply the Benjamini-Hochberg correction to each individual metric of `comparison_df`"""

    def apply_bh_correction(metric_group: pl.DataFrame) -> pl.DataFrame:
        significant_adjusted, adjusted_p_value, _, _ = multipletests(
            metric_group["MW_p_value"].to_numpy(),
            alpha=ALPHA,
            method="fdr_bh",
        )

        return metric_group.with_columns(
            Adjusted_p_value=pl.Series(adjusted_p_value),
            Significant_Adjusted=pl.Series(significant_adjusted),
        )

    return comparison_df.group_by("Metric").map_groups(apply_bh_correction)


def compare_baseline_and_variant(
    baseline_dir: Path,
    variant_dir: Path,
) -> pl.DataFrame | None:
    """Compare all common benchmarks between a baseline and variant directory"""
    baseline_bms = {d.name for d in baseline_dir.iterdir() if d.is_dir()}
    variant_bms = {d.name for d in variant_dir.iterdir() if d.is_dir()}
    common_bms = sorted(baseline_bms.intersection(variant_bms))
    missing_in_baseline = variant_bms - baseline_bms
    missing_in_variant = baseline_bms - variant_bms
    if missing_in_baseline:
        logger.warning(
            "'%s' missing benchmarks from '%s': %s",
            baseline_dir, variant_dir, sorted(missing_in_baseline)
        )  # fmt: skip

    if missing_in_variant:
        logger.warning(
            "'%s' missing benchmarks from '%s': %s",
            variant_dir, baseline_dir, sorted(missing_in_variant)
        )  # fmt: skip

    logger.info(
        "Comparing %d common benchmarks between '%s' and '%s'...",
        len(common_bms), baseline_dir, variant_dir
    )  # fmt: skip
    comparison_rows: list[ComparisonRow] = []
    for benchmark in common_bms:
        pair = load_comparison_pair(baseline_dir, variant_dir, benchmark)
        if pair is None:
            continue

        baseline_df, variant_df = pair
        variant = variant_dir.name
        for metric in METRICS:
            x = baseline_df[metric].drop_nulls().to_numpy()
            y = variant_df[metric].drop_nulls().to_numpy()
            row = compare_metric(benchmark, metric, variant, x, y)
            if row is not None:
                comparison_rows.append(row)

    if not comparison_rows:
        logger.warning(
            "Empty comparison result detected for the following directories:\n"
            "  * Baseline: '%s'\n"
            "  * Variant: '%s'\n",
            baseline_dir, variant_dir
        )  # fmt: skip
        return None

    comparison_df = apply_bh_correction_per_metric(pl.DataFrame(comparison_rows))
    return comparison_df.sort(["Program", "Metric", "Variant"])


def generate_comparisons(root_dir: Path) -> None:
    """Compare each benchmark variant with the baseline and store the results in a CSV in the variant's directory"""
    for jdk_dir in filter(Path.is_dir, root_dir.iterdir()):
        baseline_dir = jdk_dir / BASELINE_DIR_NAME

        if not baseline_dir.exists():
            logger.warning(
                "Baseline directory '%s' not found in '%s'.\n"
                "Comparisons will be skipped for this JDK...",
                BASELINE_DIR_NAME, jdk_dir
            )  # fmt: skip
            continue

        for variant_dir in filter(is_non_baseline_dir, jdk_dir.iterdir()):
            result_df = compare_baseline_and_variant(baseline_dir, variant_dir)
            if result_df is not None:
                comparison_path = variant_dir / COMPARISON_FILE_NAME
                result_df.write_csv(comparison_path)
                logger.info("Created statistical comparison '%s'", comparison_path)


# Orchestration
def generate_stats(root_dir: Path) -> None:
    """Generate benchmark statistics from raw measurements"""
    if not root_dir.exists():
        logger.error("Root directory '%s' does not exist", root_dir)
        return

    logger.info("Processing root directory '%s'...", root_dir)

    # 1. Generate individual benchmark summaries
    measurement_files = list(root_dir.rglob(MEASUREMENTS_FILE_NAME))
    logger.info("Found %d %s files", len(measurement_files), MEASUREMENTS_FILE_NAME)

    for measurements_path in measurement_files:
        try:
            relative_parts = measurements_path.relative_to(root_dir).parts
            if len(relative_parts) == 4:
                generate_benchmark_summary(measurements_path)
            else:
                logger.warning(
                    "Skipping '%s' because it does not match the expected "
                    "directory structure (<root_dir>/<jdk>/<implementation>/<benchmark>/)...",
                    measurements_path,
                )

        except (ValueError, OSError) as e:
            logger.warning("Failed path validation for '%s': %s", measurements_path, e)

    # 2. Concatenate benchmark summaries per implementation (loops, sequential streams or parallel streams)
    for jdk_dir in (d for d in root_dir.iterdir() if d.is_dir()):
        for implementation_dir in (d for d in jdk_dir.iterdir() if d.is_dir()):
            generate_combined_summary(implementation_dir)

    # 3. Generate comparative statistics
    generate_comparisons(root_dir)

    # 4. Log completion
    logger.info(
        "Processing complete. Processed %d measurement files.",
        len(measurement_files),
    )
    logger.debug(
        "Measurement cache stats: %s",
        load_and_clean_measurements.cache_info(),
    )


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
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    args = parser.parse_args()

    root_dir = args.root_dir.resolve()
    setup_logging(root_dir, args.verbose)

    logger.info("Using root directory '%s'", root_dir)
    generate_stats(root_dir)


if __name__ == "__main__":
    main()
