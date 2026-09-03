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
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from inflection import underscore
from scipy.stats import DegenerateDataWarning, bootstrap, iqr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# Metrics
class Metric(StrEnum):
    """A measured quantity, named exactly as its column in `METRICS_SCHEMA`"""

    PACKAGE = "Package"
    CORE = "Core"
    UNCORE = "Uncore"
    DRAM = "DRAM"
    PACKAGE_AND_DRAM = "PackageAndDRAM"
    TIME = "Time"
    TEMPERATURE = "Temperature"
    MEMORY = "Memory"


# Human-readable names for the metrics that get plotted.
# Omitted metric keys default to their value from the `Metric` enum.
METRIC_LABELS: dict[Metric, str] = {
    Metric.PACKAGE_AND_DRAM: "Energy",
}


# Schemas
METRICS_SCHEMA = {
    Metric.PACKAGE: pl.Float64,
    Metric.CORE: pl.Float64,
    Metric.UNCORE: pl.Float64,
    Metric.DRAM: pl.Float64,
    Metric.PACKAGE_AND_DRAM: pl.Float64,
    Metric.TIME: pl.Float64,
    Metric.TEMPERATURE: pl.Float64,
    Metric.MEMORY: pl.Float64,
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
    k: v for k, v in MEASUREMENTS_SCHEMA.items() if k != Metric.PACKAGE_AND_DRAM
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

# Effect size classes from Romano et al., 2006
type EffectSizeClass = Literal["negligible", "small", "medium", "large"]


# Supported effect size measures
class EffectSize(StrEnum):
    """An effect size, named exactly as its column in `ComparisonRow`."""

    MEDIAN_DIFFERENCE_PERCENT = "Median_Difference_Percent"
    CLIFFS_DELTA = "Cliffs_Delta"


# Confidence interval named tuple
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])

# Confidence level (%) corresponding to `ALPHA`, for display purposes
CI_LEVEL_PERCENT = round((1 - ALPHA) * 100)

# DPI for plots in PNG format
PLOT_DPI = 300


# Forest plot configurations

# Canonical column names of effect size columns so that the rendering
# logic of Forest plots is independent of the effect size measure.
FOREST_PLOT_ESTIMATE_COLUMN = "Estimate"
FOREST_PLOT_CI_LOW_COLUMN = "Estimate_CI_Low"
FOREST_PLOT_CI_HIGH_COLUMN = "Estimate_CI_High"

# Columns that must be finite for a benchmark to have a plottable position in a forest plot
FOREST_PLOT_FINITE_COLUMNS = (
    FOREST_PLOT_ESTIMATE_COLUMN,
    FOREST_PLOT_CI_LOW_COLUMN,
    FOREST_PLOT_CI_HIGH_COLUMN,
)

# Right edge of the numeric column, in Axes fractions of the right-hand panel
FOREST_PLOT_VALUE_COLUMN_EDGE = 0.98

# Y coordinate of the header row of a forest plot.
# Benchmarks are listed from y=0 to positive values of y,
# so the header row must have a negative y coordinate.
FOREST_PLOT_HEADER_Y = -1.0

# Forest plot colors and fonts
FOREST_PLOT_MARKER_COLOR = "#4C72B0"
FOREST_PLOT_HEADER_FONT: dict[str, Any] = {"weight": "bold", "size": 10}
FOREST_PLOT_VALUE_FONT: dict[str, Any] = {"family": "monospace", "size": 9}


class EffectSizeStyle(NamedTuple):
    """How to label and format an effect size measure on a forest plot"""

    xlabel: str
    column_header: str
    # Format specification for both the estimate and the CI bounds.
    value_format: str
    # Optional fixed x-limits, for effect sizes with a bounded range
    limits: tuple[float, float] | None = None


FOREST_PLOT_EFFECT_SIZE_STYLES: dict[EffectSize, EffectSizeStyle] = {
    EffectSize.MEDIAN_DIFFERENCE_PERCENT: EffectSizeStyle(
        xlabel="Median Difference (%)",
        column_header="Diff (%)",
        value_format="z6.1f",
    ),
    EffectSize.CLIFFS_DELTA: EffectSizeStyle(
        xlabel="Cliff's Delta",
        column_header="Delta",
        value_format="z6.3f",
        limits=(-1.05, 1.05),
    ),
}


# Which forest plots to render for every variant directory
FOREST_PLOTS: tuple[tuple[Metric, EffectSize], ...] = (
    (Metric.PACKAGE_AND_DRAM, EffectSize.MEDIAN_DIFFERENCE_PERCENT),
)


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
                rng=RNG,
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
            if result_df is None:
                continue

            comparison_path = variant_dir / COMPARISON_FILE_NAME
            result_df.write_csv(comparison_path)
            logger.info("Created statistical comparison '%s'", comparison_path)

            for metric, effect_size in FOREST_PLOTS:
                generate_forest_plot(jdk_dir, variant_dir, result_df, metric, effect_size)  # fmt: skip


# Forest Plots
def prepare_forest_plot_data(
    comparison_df: pl.DataFrame,
    metric: Metric,
    effect_size: EffectSize,
) -> pl.DataFrame:
    """Prepare a data frame for a forest plot"""
    metric_df = comparison_df.filter(pl.col("Metric") == metric).select(
        "Program",
        pl.col(effect_size).alias(FOREST_PLOT_ESTIMATE_COLUMN),
        pl.col(f"{effect_size}_CI_Low").alias(FOREST_PLOT_CI_LOW_COLUMN),
        pl.col(f"{effect_size}_CI_High").alias(FOREST_PLOT_CI_HIGH_COLUMN),
    )

    # Filter out rows that cannot be plotted
    plottable = pl.all_horizontal(
        pl.col(FOREST_PLOT_FINITE_COLUMNS).is_finite().fill_null(False)
    )

    dropped = metric_df.filter(~plottable).get_column("Program").to_list()
    if dropped:
        logger.warning(
            "Excluding %d benchmark(s) from the %s/%s forest plot due "
            "to non-finite estimate(s) or CI bound(s): %s",
            len(dropped), metric, effect_size, sorted(dropped),
        )  # fmt: skip

    return metric_df.filter(plottable).sort("Program")


def draw_forest_plot_graph(
    ax: plt.Axes,
    plot_df: pl.DataFrame,
    style: EffectSizeStyle,
    title: str,
) -> None:
    """Draw the graphical area of a forest plot"""
    programs = plot_df.get_column("Program").to_list()
    estimates = plot_df.get_column(FOREST_PLOT_ESTIMATE_COLUMN).to_numpy()
    ci_low = plot_df.get_column(FOREST_PLOT_CI_LOW_COLUMN).to_numpy()
    ci_high = plot_df.get_column(FOREST_PLOT_CI_HIGH_COLUMN).to_numpy()

    y = np.arange(len(programs))
    x_err = np.vstack([estimates - ci_low, ci_high - estimates])

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.errorbar(
        estimates, y,
        xerr=x_err, fmt="o",
        color=FOREST_PLOT_MARKER_COLOR,
        elinewidth=1.5, capsize=3,
        markersize=5, zorder=2,
    )  # fmt: skip

    ax.set_xlabel(style.xlabel)
    if style.limits is not None:
        ax.set_xlim(style.limits)

    ax.set_yticks(y)
    ax.set_ylabel("Benchmark")
    ax.set_yticklabels(programs)

    ax.set_title(title)


def draw_forest_plot_table(
    ax: plt.Axes,
    plot_df: pl.DataFrame,
    style: EffectSizeStyle,
) -> None:
    """Draw the estimate and its interval as one right-aligned monospaced column"""
    ax.axis("off")

    # `get_yaxis_transform` blends Axes-fraction x with data-space y, so each row
    # sits at exactly the height of its whisker on the left panel and the column
    # edge stays put regardless of how wide the panel ends up.
    transform = ax.get_yaxis_transform()

    def draw_row(y: float, row_text: str, font: dict[str, Any]) -> None:
        ax.text(
            FOREST_PLOT_VALUE_COLUMN_EDGE, y, row_text,
            transform=transform, ha="right", va="center", **font,
        )  # fmt: skip

    draw_row(
        FOREST_PLOT_HEADER_Y,
        f"{style.column_header} [{CI_LEVEL_PERCENT}% CI]",
        FOREST_PLOT_HEADER_FONT,
    )

    fmt = style.value_format
    for i, row in enumerate(plot_df.iter_rows(named=True)):
        estimate = row[FOREST_PLOT_ESTIMATE_COLUMN]
        ci_low = row[FOREST_PLOT_CI_LOW_COLUMN]
        ci_high = row[FOREST_PLOT_CI_HIGH_COLUMN]
        text = f"{estimate:{fmt}} [{ci_low:{fmt}}, {ci_high:{fmt}}]"
        draw_row(float(i), text, FOREST_PLOT_VALUE_FONT)


def render_forest_plot(
    plot_df: pl.DataFrame,
    effect_size: EffectSize,
    title: str,
) -> plt.Figure:
    """Render a forest plot of `plot_df` with boostrap CI whiskers"""
    style = FOREST_PLOT_EFFECT_SIZE_STYLES[effect_size]
    n = plot_df.height

    fig, (ax, ax_labels) = plt.subplots(
        1, 2,
        figsize=(10, max(2.0, 0.35 * (n + 1) + 1.0)),
        gridspec_kw={"width_ratios": [3, 1.3]},
        sharey=True,
    )  # fmt: skip

    # Zebra striping, spanning both subplots so a row stays traceable from its
    # benchmark label all the way to its row in the right-hand side table.
    for axes in (ax, ax_labels):
        for yi in range(0, n, 2):
            axes.axhspan(yi - 0.5, yi + 0.5, color="black", alpha=0.04, zorder=0)

    draw_forest_plot_graph(ax, plot_df, style, title)
    draw_forest_plot_table(ax_labels, plot_df, style)

    # Set last, and on one Axes only: `sharey` propagates it, and it has to
    # survive the autoscaling that the panels' own drawing would otherwise trigger
    ax.set_ylim(n - 0.5, FOREST_PLOT_HEADER_Y - 0.5)

    fig.tight_layout()
    return fig


def generate_forest_plot(
    jdk_dir: Path,
    variant_dir: Path,
    comparison_df: pl.DataFrame,
    metric: Metric,
    effect_size: EffectSize,
) -> None:
    """Generate a forest plot for the specified variant directory, metric and effect size"""
    metric_label = METRIC_LABELS.get(metric, str(metric))
    title = f"{jdk_dir.name} / {variant_dir.name} — {metric_label}"

    plot_df = prepare_forest_plot_data(comparison_df, metric, effect_size)
    if plot_df.is_empty():
        logger.warning(
            "No plottable %s comparisons found for '%s'; skipping %s forest plot",
            metric, variant_dir, effect_size,
        )  # fmt: skip
        return

    fig = render_forest_plot(plot_df, effect_size, title)
    try:
        stem = f"{underscore(metric_label)}_{underscore(effect_size)}_forest_plot"
        pgf_path = variant_dir / f"{stem}.pgf"
        png_path = variant_dir / f"{stem}.png"
        fig.savefig(pgf_path, format="pgf", backend="pgf")
        fig.savefig(png_path, dpi=PLOT_DPI)
        logger.info("Created forest plots '%s' and '%s'", pgf_path, png_path)
    finally:
        plt.close(fig)


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


def run(root_dir: Path) -> None:
    """Process every scenario subdirectory of `root_dir`"""
    if not root_dir.exists():
        logger.error("Root directory '%s' does not exist", root_dir)
        return

    scenario_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    logger.info(
        "Processing %d scenario director%s: %s",
        len(scenario_dirs), "y" if len(scenario_dirs) == 1 else "ies",
        [d.name for d in scenario_dirs],
    )  # fmt: skip
    for scenario_dir in scenario_dirs:
        generate_stats(scenario_dir)


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
    run(root_dir)


if __name__ == "__main__":
    main()
