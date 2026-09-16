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
       - Cliff's delta and its bootstrap confidence interval.

2. Total energy (package + DRAM) statistics, including:
   - Forest plots for each variant in the corresponding JDK directory.
   - Scenario-level heatmaps in the corresponding scenario directory.

3. Time and total energy correlation consisting of:
   - A Spearman rank-correlation table under `<root>`, quantifying the
     relationship between the percentage changes in median time and total
     energy relative to the baseline.
   - Scatter plots of those changes in each scenario directory.

Bootstrap resampling is deterministically seeded from the resampled data,
making results reproducible regardless of execution order or scheduling.
"""

import argparse
import functools
import hashlib
import itertools
import logging
import logging.handlers
import math
import multiprocessing
import os
import re
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from queue import Queue
from typing import Any, Literal, NamedTuple, Protocol, TypedDict

import matplotlib
from matplotlib.axes import Axes

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from adjustText import adjust_text
from inflection import underscore
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator
from scipy.stats import (
    DegenerateDataWarning,
    bootstrap,
    iqr,
    rankdata,
    spearmanr,
)

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


def get_metric_label(metric: Metric) -> str:
    """Return the human-readable label for `metric`"""
    return METRIC_LABELS.get(metric, str(metric))


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
LOG_FILE_NAME = "bench_stats.log"
MEASUREMENTS_FILE_NAME = "measurements.csv"
SUMMARY_FILE_NAME = "summary.csv"
COMPARISON_FILE_NAME = "comparison.csv"
ENERGY_TIME_CORRELATION_FILE_NAME = "energy_time_correlation.csv"
JAVA_VERSION_FILE_NAME = "java_version.txt"
BASELINE_DIR_NAME = "loops"

# One minus the confidence level of every confidence interval
ALPHA = 0.05

# Valid methods for computing bootstrap confidence intervals
type BootstrapMethod = Literal["percentile", "basic", "bca", "BCa"]

# Primary method for computing bootstrap confidence intervals
PRIMARY_BOOTSTRAP_METHOD: BootstrapMethod = "BCa"

# Fallback bootstrap method for degenerate data
FALLBACK_BOOTSTRAP_METHOD: BootstrapMethod = "percentile"

# Root seed of every bootstrap. Each bootstrap derives its generator from this seed
# and a deterministic sample key, so bootstrapped confidence intervals are deterministic.
ROOT_SEED = 42

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

# Horizontal center of the numeric column, in Axes fractions of the right-hand panel
FOREST_PLOT_VALUE_COLUMN_CENTER = 0.5

# Y coordinate of the header row of a forest plot.
# Benchmarks are listed from y=0 to positive values of y,
# so the header row must have a negative y coordinate.
FOREST_PLOT_HEADER_Y = -1.0

# Forest plot colors and fonts
FOREST_PLOT_MARKER_COLOR = "#4C72B0"
FOREST_PLOT_HEADER_FONT: dict[str, Any] = {"weight": "bold", "size": 10}
FOREST_PLOT_VALUE_FONT: dict[str, Any] = {"family": "monospace", "size": 9}


class EffectSizeStyle(NamedTuple):
    """How to format an effect size measure"""

    label: str
    # Decimal places for both the estimate and the CI bounds.
    precision: int
    # Fixed column width used only by the forest plot's monospace, right-aligned value table.
    plot_column_width: int = 6
    # Optional fixed axis limits, for effect sizes with a bounded range
    limits: tuple[float, float] | None = None


EFFECT_SIZE_STYLES: dict[EffectSize, EffectSizeStyle] = {
    EffectSize.MEDIAN_DIFFERENCE_PERCENT: EffectSizeStyle(
        label="Relative median change (%)",
        precision=1,
    ),
    EffectSize.CLIFFS_DELTA: EffectSizeStyle(
        label="Cliff's δ",
        precision=3,
        limits=(-1.05, 1.05),
    ),
}


# Which forest plots to render for every variant directory
FOREST_PLOTS: tuple[tuple[Metric, EffectSize], ...] = (
    (Metric.PACKAGE_AND_DRAM, EffectSize.MEDIAN_DIFFERENCE_PERCENT),
)


# Heatmap configurations

# ColorBrewer's RdBu, reversed so that red encodes an increase and blue a decrease.
# RdBu is a colorblind-safe diverging palette that ranges from a saturated blue,
# through a neutral light gray/white, to a deep, saturated red.
HEATMAP_COLORMAP = "RdBu_r"

# Color of cells without a finite estimate or confidence interval
HEATMAP_MISSING_COLOR = "#DDDDDD"

# Effect sizes without a bounded range are colored on a symmetric logarithmic
# scale, linear within +/- this threshold and logarithmic beyond it. Relative
# median changes span orders of magnitude, so a linear scale would flatten the
# many small effects into an indistinguishable near-white block.
HEATMAP_LINEAR_THRESHOLD = 10.0

# Size of a single heatmap cell and of the margins reserved for
# the axis labels, the title and the colorbar, all in inches.
HEATMAP_CELL_WIDTH_INCHES = 1.5
HEATMAP_CELL_HEIGHT_INCHES = 0.55
HEATMAP_HORIZONTAL_MARGIN_INCHES = 2.5
HEATMAP_VERTICAL_MARGIN_INCHES = 1.5

# Heatmap cell annotation font
HEATMAP_CELL_FONT: dict[str, Any] = {"family": "monospace", "size": 7}

# Distance from the middle of the color scale beyond which a cell is
# dark enough to need light text. The colors of light and dark text are
# defined by HEATMAP_DARK_CELL_TEXT_COLOR and HEATMAP_LIGHT_CELL_TEXT_COLOR.
HEATMAP_DARK_CELL_THRESHOLD = 0.35
HEATMAP_DARK_CELL_TEXT_COLOR = "white"
HEATMAP_LIGHT_CELL_TEXT_COLOR = "black"

# Which heatmaps to render for every scenario
HEATMAPS: tuple[tuple[Metric, EffectSize], ...] = (
    (Metric.PACKAGE_AND_DRAM, EffectSize.MEDIAN_DIFFERENCE_PERCENT),
)


# Scatter plot configuration

# Marker color for scatter points
SCATTER_PLOT_MARKER_COLOR = FOREST_PLOT_MARKER_COLOR

# Fonts for the figure title, panel titles, axis ticks, per-point benchmark labels
# and per-panel correlation annotation. Matplotlib's defaults are sized for a figure
# far larger than one of these panels, so all of them are set explicitly.
SCATTER_PLOT_SUPTITLE_FONT: dict[str, Any] = {"size": 11}
SCATTER_PLOT_TITLE_FONT: dict[str, Any] = {"size": 9}
SCATTER_PLOT_TICK_FONT: dict[str, Any] = {"size": 7}
SCATTER_PLOT_LABEL_FONT: dict[str, Any] = {"size": 6}
SCATTER_PLOT_STAT_FONT: dict[str, Any] = {"size": 7}

# Headroom left past the data at each end of an axis, as a fraction of the axis'
# length. It is measured on the scale the axis is drawn on rather than on the data,
# since a multiple of a value is a vanishing part of an axis spanning many decades.
SCATTER_PLOT_AXIS_PADDING_FRACTION = 0.06

# Both axes are symmetric logarithmic, since relative changes span orders of
# magnitude, and linear within +/- this threshold so that they can cross zero. That
# region is narrow because distances within it read as differences, not as ratios.
SCATTER_PLOT_LINEAR_THRESHOLD = 1.0

# Width of the linear region, as a multiple of a tenfold step of the logarithmic
# part. Matplotlib's own default, shared so that the axes and their padding agree.
SCATTER_PLOT_LINEAR_SCALE = 1.0

# Columns the JDK panels are laid out over. A single row would leave each panel too
# narrow to hold a labeled point per benchmark, so they wrap over several rows.
SCATTER_PLOT_COLUMNS = 2

# Size of a single panel and of the margins reserved for the axis labels and the
# figure title, all in inches. The figure grows with the number of panels. Scaling
# it to a given page is left to whatever includes it.
SCATTER_PLOT_PANEL_WIDTH_INCHES = 2.9
SCATTER_PLOT_PANEL_HEIGHT_INCHES = 2.7
SCATTER_PLOT_HORIZONTAL_MARGIN_INCHES = 0.5
SCATTER_PLOT_VERTICAL_MARGIN_INCHES = 1.0

# Reference lines under the points of every panel
SCATTER_PLOT_ZERO_LINE_STYLE: dict[str, Any] = {
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.8,
    "zorder": 1,
}
SCATTER_PLOT_IDENTITY_LINE_STYLE: dict[str, Any] = {
    "color": "gray",
    "linestyle": ":",
    "linewidth": 0.8,
    "zorder": 1,
}

# Rounds `adjust_text` spends spreading the labels of a scatter plot
SCATTER_PLOT_LABEL_ITERATIONS = 150

# Leader from a point to wherever `adjust_text` had to move its label
SCATTER_PLOT_LABEL_ARROW_STYLE: dict[str, Any] = {
    "arrowstyle": "-",
    "color": "gray",
    "lw": 0.5,
}

# Where the correlation annotation sits, in Axes fractions
SCATTER_PLOT_STAT_POSITION = (0.04, 0.96)

# Names every scenario at once, in the energy-time correlation table
ENERGY_TIME_CORRELATION_ALL_LABEL = "All"

# JDK versions

# The version string of a JDK, as reported by `java -version`, is the first
# quoted token of `JAVA_VERSION_FILE_NAME` (e.g. openjdk version "25.0.3").
JAVA_VERSION_STRING_PATTERN = re.compile(r'version\s+"([^"]+)"')

# `(major, minor, security)` of a legacy (pre-JEP 223) version number, whose
# form is "1.$MAJOR.$MINOR_$UPDATE" (e.g. "1.8.0_492"). $UPDATE maps onto
# $SECURITY, as in the version data that Adoptium publishes for the JDKs
# used here (e.g. "1.8.0_504-b01" is major 8, minor 0, security 504).
JAVA_LEGACY_VERSION_NUMBER_PATTERN = re.compile(r"^1\.(\d+)(?:\.(\d+))?(?:_(\d+))?")

# `(major, minor, security)` of `$VNUM`, the leading version number of a
# JEP 223 version string (e.g. "25.0.3" of "25.0.3+9-LTS"). `$VNUM` omits
# trailing zero elements, so it may be shorter (e.g. "21" is JDK 21.0.0),
# and any fourth or later element is reserved for downstream consumers.
JAVA_VERSION_NUMBER_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")

# A `(major, minor, security)` version number, as defined by JEP 223
type JdkVersion = tuple[int, int, int]


class SampleKey(NamedTuple):
    """Identifies the sample of runs a single measurements file holds"""

    scenario: str
    jdk: str
    implementation: str
    benchmark: str

    @classmethod
    def from_benchmark_directory(cls, benchmark_dir: Path) -> SampleKey:
        """Identify the sample held in a `<scenario>/<jdk>/<implementation>/<benchmark>` directory"""
        implementation_dir = benchmark_dir.parent
        jdk_dir = implementation_dir.parent
        scenario_dir = jdk_dir.parent
        return cls(
            scenario_dir.name,
            jdk_dir.name,
            implementation_dir.name,
            benchmark_dir.name,
        )

    def __str__(self) -> str:
        return "/".join(self)


class TwoSampleStatistic(Protocol):
    def __call__(
        self,
        x: npt.NDArray[IntOrFloat],
        y: npt.NDArray[IntOrFloat],
        /,
        *,
        axis: int = 0,
    ) -> IntOrFloat | npt.NDArray[IntOrFloat]: ...


class ComparisonTask(NamedTuple):
    """A single task for comparing a benchmark with the baseline"""

    baseline_dir: Path
    variant_dir: Path
    benchmark: str


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


# Parallelism
def setup_worker_logging(log_queue: Queue[logging.LogRecord], level: int) -> None:
    """Put a worker's log records on `log_queue` for its parent process to write out"""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logging.captureWarnings(True)


@contextmanager
def worker_pool(jobs: int) -> Iterator[Executor | None]:
    """Start a pool of `jobs` worker processes, or no pool when `jobs` is 1"""
    if jobs == 1:
        yield None
        return

    # Writing one log file from several processes is not supported, so the workers
    # send their records here and this process does all the writing.
    root_logger = logging.getLogger()
    with multiprocessing.Manager() as manager:
        listener = logging.handlers.QueueListener(
            log_queue := manager.Queue(),
            *root_logger.handlers,
            respect_handler_level=True,
        )
        listener.start()
        try:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=setup_worker_logging,
                initargs=(log_queue, root_logger.level),
            ) as executor:
                yield executor
        finally:
            listener.stop()


def run_tasks[R](
    tasks: list[Callable[[], R]],
    executor: Executor | None,
) -> list[R]:
    """Run every task, in this process or spread over `executor`, and return the results in task order"""
    if executor is None:
        return [t() for t in tasks]

    futures = [executor.submit(t) for t in tasks]
    return [future.result() for future in futures]


# JDK Versions
def parse_jdk_version(java_version_output: str) -> JdkVersion | None:
    """Parse the `(major, minor, security)` version of a `java -version` output, or `None` if absent"""
    version_string_match = JAVA_VERSION_STRING_PATTERN.search(java_version_output)
    if version_string_match is None:
        return None

    version_string = version_string_match.group(1)
    version_patterns = (JAVA_VERSION_NUMBER_PATTERN, JAVA_LEGACY_VERSION_NUMBER_PATTERN)
    version_number_match = next(
        (match for p in version_patterns if (match := p.match(version_string))),
        None,
    )
    if version_number_match is None:
        return None

    major, minor, security = version_number_match.groups()
    return int(major), int(minor or 0), int(security or 0)


def read_jdk_version(jdk_dir: Path) -> JdkVersion | None:
    """Read the version of the JDK in `jdk_dir`, or `None` if it cannot be determined"""
    java_version_path = jdk_dir / JAVA_VERSION_FILE_NAME
    try:
        version = parse_jdk_version(java_version_path.read_text(encoding="utf-8"))
    except OSError:
        logger.exception("Error when reading '%s'", java_version_path)
        return None

    if version is None:
        logger.warning("Could not parse JDK version from '%s'", java_version_path)

    return version


def collect_jdk_versions(scenario_dirs: list[Path]) -> dict[str, JdkVersion]:
    """Map the name of every JDK directory to the version of the JDK it holds"""
    versions: dict[str, JdkVersion] = {}
    for scenario_dir in scenario_dirs:
        for jdk_dir in sorted(filter(Path.is_dir, scenario_dir.iterdir())):
            if jdk_dir.name in versions:
                continue

            version = read_jdk_version(jdk_dir)
            if version is not None:
                versions[jdk_dir.name] = version

    return versions


def sort_jdks(jdks: list[str], versions: dict[str, JdkVersion]) -> list[str]:
    """Sort JDK names by version, listing JDKs of unknown version last, in name order"""
    return sorted(
        jdks, key=lambda jdk: (jdk not in versions, versions.get(jdk, ()), jdk)
    )


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


def export_plot(fig: plt.Figure, output_dir: Path, stem: str, description: str) -> None:
    """Write `fig` into `output_dir` as a PGF and a PNG file, both named `stem`"""
    pgf_path = output_dir / f"{stem}.pgf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pgf_path, format="pgf", backend="pgf")
    fig.savefig(png_path, dpi=PLOT_DPI)
    logger.info("Created %s '%s' and '%s'", description, pgf_path, png_path)


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
            for d in sorted(implementation_dir.iterdir())
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
    """Return `median(y) - median(x)`, as a percentage of `median(x)."""
    med_x = np.median(x, axis=axis)
    med_y = np.median(y, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (med_y - med_x) / med_x * 100.0


def mannwhitneyu_statistic(
    x: npt.NDArray[IntOrFloat],
    y: npt.NDArray[IntOrFloat],
    axis: int = 0,
) -> np.floating | npt.NDArray[np.floating]:
    """
    Return the Mann-Whitney U statistic for two samples.

    Unlike `scipy.stats.mannwhitneyu`, this function does not broadcast inputs
    or calculate p-values, bypassing significant overhead during repeated calls,
    e.g., inside bootstrap resamplings.
    """
    ranks = rankdata(np.concatenate([x, y], axis=axis), axis=axis)

    # A slice rather than a take, so the ranks of `x` stay a view of `ranks`
    nx = x.shape[axis]
    x_slice: list[slice] = [slice(None)] * ranks.ndim
    x_slice[axis] = slice(None, nx)

    return ranks[tuple(x_slice)].sum(axis=axis) - nx * (nx + 1) / 2


def cliffs_delta(
    x: npt.NDArray[IntOrFloat],
    y: npt.NDArray[IntOrFloat],
    axis: int = 0,
) -> np.floating | npt.NDArray[np.floating]:
    """
    Return Cliff's delta from samples `x` and `y`.

    The output matches the shape of the samples with `axis` removed.
    """
    u = mannwhitneyu_statistic(x, y, axis=axis)
    nx = x.shape[axis]
    ny = y.shape[axis]
    return (2 * u) / (nx * ny) - 1


def bootstrap_rng(*identity: str) -> np.random.Generator:
    """Return the generator that the sample named by `identity` resamples with"""
    # Use SHA-256 instead of `hash()` for stable, reproducible seeds
    digest = hashlib.sha256("/".join(identity).encode()).digest()
    seed = np.random.SeedSequence(
        ROOT_SEED,
        spawn_key=(int.from_bytes(digest, "big"),),
    )
    return np.random.default_rng(seed)


def two_sample_bootstrap_ci(
    x: npt.NDArray[IntOrFloat],
    y: npt.NDArray[IntOrFloat],
    statistic: TwoSampleStatistic,
    rng: np.random.Generator,
    n_resamples: int = 9_999,
) -> ConfidenceInterval:
    """Return the bootstrap confidence interval for a custom two-sample statistic."""
    resample = functools.partial(
        bootstrap,
        data=(x, y),
        statistic=statistic,
        vectorized=True,
        confidence_level=1 - ALPHA,
        rng=rng,
    )

    # Suppress expected warnings from zero-variance inputs
    with (
        warnings.catch_warnings(record=False),
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("ignore", DegenerateDataWarning)
        res = resample(n_resamples=n_resamples, method=PRIMARY_BOOTSTRAP_METHOD)

        ci = res.confidence_interval
        if np.isfinite([ci.low, ci.high]).all():
            return ci

        logger.info(
            "Degenerate data encountered. Falling back to '%s' bootstrap...",
            FALLBACK_BOOTSTRAP_METHOD,
        )
        # Only BCa's acceleration is undefined for these resamples.
        # The resamples themselves are sound, so the bootstrap result can be reused.
        res = resample(
            n_resamples=0,
            bootstrap_result=res,
            method=FALLBACK_BOOTSTRAP_METHOD,
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
    sample_key: SampleKey,
    metric: str,
    baseline_values: npt.NDArray[IntOrFloat],
    variant_values: npt.NDArray[IntOrFloat],
) -> ComparisonRow | None:
    """Compute all comparative statistics for a single metric pair"""
    x = baseline_values
    y = variant_values
    benchmark, variant = sample_key.benchmark, sample_key.implementation

    if len(x) < 2 or len(y) < 2:
        logger.warning(
            "%s/%s: insufficient samples (baseline: %d, %s: %d)",
            sample_key, metric, len(x), variant, len(y),
        )  # fmt: skip
        return None

    med_x, med_y = np.median(x), np.median(y)
    med_diff = med_y - med_x
    med_diff_pct = (med_diff / med_x) * 100 if med_x != 0 else np.nan
    med_diff_pct_ci = two_sample_bootstrap_ci(
        x, y, median_difference_percent,
        bootstrap_rng(*sample_key, metric, EffectSize.MEDIAN_DIFFERENCE_PERCENT),
    )  # fmt: skip
    iqr_x, iqr_y = iqr(x), iqr(y)

    # Pass the variant as the first sample so that the sign of Cliff's delta
    # indicates the direction of the difference between variant and baseline.
    cliffs_d = float(cliffs_delta(y, x))
    cliffs_d_ci = two_sample_bootstrap_ci(
        y, x, cliffs_delta,
        bootstrap_rng(*sample_key, metric, EffectSize.CLIFFS_DELTA),
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
    }

    return row


def common_benchmarks(baseline_dir: Path, variant_dir: Path) -> list[str]:
    """Return the common benchmarks from the specified directories"""
    baseline_bms = {d.name for d in baseline_dir.iterdir() if d.is_dir()}
    variant_bms = {d.name for d in variant_dir.iterdir() if d.is_dir()}
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

    return sorted(baseline_bms.intersection(variant_bms))


def compare_benchmark(task: ComparisonTask) -> list[ComparisonRow]:
    """Compare every metric of one benchmark between a baseline and a variant"""
    comparison_pair = load_comparison_pair(
        task.baseline_dir,
        task.variant_dir,
        task.benchmark,
    )
    if comparison_pair is None:
        return []

    baseline_df, variant_df = comparison_pair
    sample_key = SampleKey.from_benchmark_directory(task.variant_dir / task.benchmark)

    comparison_rows: list[ComparisonRow] = []
    for metric in METRICS:
        x = baseline_df[metric].drop_nulls().to_numpy()
        y = variant_df[metric].drop_nulls().to_numpy()
        row = compare_metric(sample_key, metric, x, y)
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


def generate_comparisons(root_dir: Path, executor: Executor | None) -> None:
    """Compare each benchmark variant with the baseline and store the results in a CSV in the variant's directory"""
    variant_pairs: list[tuple[Path, Path]] = []
    tasks: list[ComparisonTask] = []
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
            tasks += [
                ComparisonTask(baseline_dir, variant_dir, benchmark)
                for benchmark in benchmarks
            ]

    # Each task seeds its bootstraps from the sample it compares, so its results are
    # independent of the process that runs it. The results are returned in task order.
    task_rows = run_tasks(
        [functools.partial(compare_benchmark, task) for task in tasks],
        executor,
    )
    rows_by_variant: dict[Path, list[ComparisonRow]] = defaultdict(list)
    for task, comparison_rows in zip(tasks, task_rows, strict=True):
        rows_by_variant[task.variant_dir] += comparison_rows

    plot_tasks: list[Callable[[], None]] = []
    for baseline_dir, variant_dir in variant_pairs:
        result_df = build_comparison_frame(
            baseline_dir,
            variant_dir,
            rows_by_variant[variant_dir],
        )
        if result_df is None:
            continue

        comparison_path = variant_dir / COMPARISON_FILE_NAME
        result_df.write_csv(comparison_path)
        logger.info("Created statistical comparison '%s'", comparison_path)

        plot_tasks += [
            functools.partial(
                generate_forest_plot,
                variant_dir.parent,
                variant_dir,
                result_df,
                metric,
                effect_size,
            )
            for metric, effect_size in FOREST_PLOTS
        ]

    run_tasks(plot_tasks, executor)


# Cross-Scenario Overview Statistics
def collect_comparison_data(scenario_dirs: list[Path]) -> pl.DataFrame:
    """Concatenate all comparison CSVs and tag them with Scenario and JDK columns"""
    frames = []
    for scenario_dir in scenario_dirs:
        for jdk_dir in sorted(filter(Path.is_dir, scenario_dir.iterdir())):
            for variant_dir in sorted(filter(is_non_baseline_dir, jdk_dir.iterdir())):
                comparison_path = variant_dir / COMPARISON_FILE_NAME
                if not comparison_path.exists():
                    logger.warning("Missing '%s'; skipping", comparison_path)
                    continue

                df = pl.read_csv(comparison_path).with_columns(
                    Scenario=pl.lit(scenario_dir.name),
                    JDK=pl.lit(jdk_dir.name),
                )
                frames.append(df)

    if not frames:
        logger.error(
            "No comparison data found across %s",
            [d.name for d in scenario_dirs],
        )
        return pl.DataFrame()

    return pl.concat(frames, how="vertical")


def build_long_time_and_energy_table(comparison_data: pl.DataFrame) -> pl.DataFrame:
    """Build a long (Benchmark, JDK, Scenario, Variant, Time_Diff_Percent, Energy_Diff_Percent) table"""
    keys = ["Program", "JDK", "Scenario", "Variant"]
    time_df = comparison_data.filter(pl.col("Metric") == Metric.TIME).select(
        *keys, pl.col("Median_Difference_Percent").alias("Time_Diff_Percent")
    )
    energy_df = comparison_data.filter(
        pl.col("Metric") == Metric.PACKAGE_AND_DRAM
    ).select(*keys, pl.col("Median_Difference_Percent").alias("Energy_Diff_Percent"))

    long_df = time_df.join(energy_df, on=keys, how="inner")
    finite = (
        pl.col("Time_Diff_Percent").is_finite()
        & pl.col("Energy_Diff_Percent").is_finite()
    )
    return long_df.filter(finite).rename({"Program": "Benchmark"})


def spearman_rho(subset: pl.DataFrame) -> float:
    """Correlate the time and energy differences of `subset` by Spearman's rank coefficient"""
    return float(
        spearmanr(
            subset.get_column("Time_Diff_Percent").to_numpy(),
            subset.get_column("Energy_Diff_Percent").to_numpy(),
        ).statistic
    )


def spearman_row(
    variant: str,
    scenario: str,
    subset: pl.DataFrame,
) -> dict[str, Any]:
    """Compute one row of the energy-time correlation table"""
    return {
        "Variant": variant,
        "Scenario": scenario,
        "N": len(subset),
        "Spearman_Rho": spearman_rho(subset),
    }


def generate_energy_time_correlation_table(long_df: pl.DataFrame) -> pl.DataFrame:
    """Compute the Spearman correlation between Time_Diff_Percent and Energy_Diff_Percent, per variant and scenario"""
    rows = []
    for variant in sorted(long_df.get_column("Variant").unique()):
        variant_df = long_df.filter(pl.col("Variant") == variant)
        rows.append(
            spearman_row(variant, ENERGY_TIME_CORRELATION_ALL_LABEL, variant_df)
        )
        rows += [
            spearman_row(
                variant,
                scenario.capitalize(),
                variant_df.filter(pl.col("Scenario") == scenario),
            )
            for scenario in sorted(variant_df.get_column("Scenario").unique())
        ]

    return pl.DataFrame(rows)


def overall_plot_tasks(
    root_dir: Path,
    scenario_dirs: list[Path],
) -> list[Callable[[], None]]:
    """Return the list of tasks responsible for producing overview statistics"""
    plot_tasks: list[Callable[[], None]] = []
    comparison_data = collect_comparison_data(scenario_dirs)
    if comparison_data.is_empty():
        return plot_tasks

    jdk_versions = collect_jdk_versions(scenario_dirs)
    for metric, effect_size in HEATMAPS:
        # Built from every scenario, so that all the heatmaps of a
        # metric share a color scale and stay comparable with each other
        norm = build_effect_size_norm(comparison_data, metric, effect_size)
        if norm is None:
            logger.warning(
                "No %s comparisons found; skipping the %s heatmaps",
                metric, effect_size,
            )  # fmt: skip
            continue

        plot_tasks += [
            functools.partial(
                generate_effect_size_heatmaps,
                root_dir / scenario,
                scenario,
                comparison_data.filter(pl.col("Scenario") == scenario),
                metric,
                effect_size,
                norm,
                jdk_versions,
            )
            for scenario in sorted(comparison_data.get_column("Scenario").unique())
        ]

    long_df = build_long_time_and_energy_table(comparison_data)
    if long_df.is_empty():
        logger.warning("No energy-time data found; skipping energy-time correlation")
        return plot_tasks

    correlation_df = generate_energy_time_correlation_table(long_df)
    correlation_path = root_dir / ENERGY_TIME_CORRELATION_FILE_NAME
    correlation_df.write_csv(correlation_path)
    logger.info("Created energy-time correlation table '%s'", correlation_path)

    return plot_tasks + energy_time_scatter_plot_tasks(root_dir, long_df, jdk_versions)


def generate_overall_stats(
    root_dir: Path,
    scenario_dirs: list[Path],
    executor: Executor | None,
) -> None:
    """Generate overview statistics"""
    run_tasks(overall_plot_tasks(root_dir, scenario_dirs), executor)


# Effect Size Heatmaps
def format_estimate_with_ci(
    estimate: float,
    ci_low: float,
    ci_high: float,
    style: EffectSizeStyle,
) -> str:
    """Format an effect size as "E\\n[L, H]" (or "" if some value is not finite)"""
    if any(not np.isfinite(v) for v in (estimate, ci_low, ci_high)):
        return ""

    fmt = f"+z.{style.precision}f"
    return f"{estimate:{fmt}}\n[{ci_low:{fmt}}, {ci_high:{fmt}}]"


class HeatmapData(NamedTuple):
    """The benchmarks, JDKs, effect sizes and cell annotations of one heatmap"""

    benchmarks: list[str]
    jdks: list[str]
    # Effect size estimates, as a (benchmark, JDK) matrix whose
    # cells without anything to plot are left masked
    estimates: np.ma.MaskedArray
    # Cell annotations, in the same shape as `estimates`
    annotations: npt.NDArray[np.object_]


def prepare_heatmap_data(
    comparison_data: pl.DataFrame,
    metric: Metric,
    effect_size: EffectSize,
    jdk_versions: dict[str, JdkVersion],
) -> dict[str, HeatmapData]:
    """Prepare the data of an effect size heatmap of a metric, for each variant"""
    style = EFFECT_SIZE_STYLES[effect_size]
    metric_df = comparison_data.filter(pl.col("Metric") == metric)

    heatmaps: dict[str, HeatmapData] = {}
    for variant in sorted(metric_df.get_column("Variant").unique()):
        variant_df = metric_df.filter(pl.col("Variant") == variant)
        benchmarks = sorted(variant_df.get_column("Program").unique())
        jdks = sort_jdks(variant_df.get_column("JDK").unique().to_list(), jdk_versions)
        rows = {benchmark: i for i, benchmark in enumerate(benchmarks)}
        columns = {jdk: j for j, jdk in enumerate(jdks)}

        estimates = np.ma.masked_all((len(benchmarks), len(jdks)))
        annotations = np.full(estimates.shape, "", dtype=np.object_)
        for row in variant_df.iter_rows(named=True):
            # A cell whose effect size or interval could not be estimated has
            # no annotation, and is left masked so that it is drawn as missing
            annotation = format_estimate_with_ci(
                row[effect_size],
                row[f"{effect_size}_CI_Low"],
                row[f"{effect_size}_CI_High"],
                style,
            )
            if not annotation:
                continue

            i, j = rows[row["Program"]], columns[row["JDK"]]
            estimates[i, j] = row[effect_size]
            annotations[i, j] = annotation

        heatmaps[variant] = HeatmapData(benchmarks, jdks, estimates, annotations)

    return heatmaps


def build_effect_size_norm(
    comparison_data: pl.DataFrame,
    metric: Metric,
    effect_size: EffectSize,
) -> Normalize | None:
    """Build the color scale of an effect size, centered at zero, or `None` if there is nothing to scale"""
    style = EFFECT_SIZE_STYLES[effect_size]
    limits = style.limits
    if limits is not None:
        return Normalize(limits[0], limits[1])

    estimates = (
        comparison_data.filter(pl.col("Metric") == metric)
        .get_column(effect_size)
        .to_numpy()
    )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return None

    limit = float(np.abs(estimates).max())
    return SymLogNorm(HEATMAP_LINEAR_THRESHOLD, vmin=-limit, vmax=limit)


def draw_heatmap_annotations(
    ax: plt.Axes,
    data: HeatmapData,
    norm: Normalize,
) -> None:
    """Draw the estimate and its interval on every cell of a heatmap"""
    for (i, j), annotation in np.ndenumerate(data.annotations):
        if not annotation:
            continue

        # Cells near either end of the color scale are dark, so light text
        # is the readable choice; the pale cells in the middle need dark text
        shade = abs(float(norm(data.estimates[i, j])) - 0.5)
        color = (
            HEATMAP_DARK_CELL_TEXT_COLOR
            if shade > HEATMAP_DARK_CELL_THRESHOLD
            else HEATMAP_LIGHT_CELL_TEXT_COLOR
        )
        # The mesh spans the cell boundaries, so a cell is centered
        # half a cell past the coordinates of its lower left corner
        ax.text(
            j + 0.5, i + 0.5, annotation,
            color=color, ha="center", va="center", **HEATMAP_CELL_FONT,
        )  # fmt: skip


def render_effect_size_heatmap(
    data: HeatmapData,
    effect_size: EffectSize,
    norm: Normalize,
    title: str,
) -> plt.Figure:
    """Render a heatmap of the effect sizes in `data`, annotated with their bootstrap CIs"""
    style = EFFECT_SIZE_STYLES[effect_size]
    rows, columns = data.estimates.shape

    fig, ax = plt.subplots(
        figsize=(
            columns * HEATMAP_CELL_WIDTH_INCHES + HEATMAP_HORIZONTAL_MARGIN_INCHES,
            rows * HEATMAP_CELL_HEIGHT_INCHES + HEATMAP_VERTICAL_MARGIN_INCHES,
        )
    )

    # Drawn as a mesh so that the PGF output is vector graphics instead of rasterized PNGs.
    colormap = plt.get_cmap(HEATMAP_COLORMAP).with_extremes(bad=HEATMAP_MISSING_COLOR)
    mesh = ax.pcolormesh(data.estimates, cmap=colormap, norm=norm)
    draw_heatmap_annotations(ax, data, norm)

    # A mesh spans the cell boundaries, so its ticks (unlike an image's) sit
    # on them, and its rows are laid out upwards. Both need correcting for
    # the benchmarks to read alphabetically from the top of a labeled grid.
    ax.set_xticks(np.arange(columns) + 0.5, labels=data.jdks)
    ax.set_xlabel("JDK")
    ax.set_yticks(np.arange(rows) + 0.5, labels=data.benchmarks)
    ax.set_ylabel("Benchmark")
    ax.invert_yaxis()
    ax.set_title(title)

    colorbar = fig.colorbar(mesh, ax=ax)
    assert colorbar.solids is not None
    colorbar.solids.set_rasterized(False)

    label = style.label
    if isinstance(norm, SymLogNorm):
        colorbar.ax.yaxis.set_major_locator(
            SymmetricalLogLocator(linthresh=HEATMAP_LINEAR_THRESHOLD, base=10)
        )

        # Ensure ticks are read in the same terms as the cells, not in scientific notation
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda tick, _: f"{tick:g}"))  # fmt: skip

        # Make it clear that the chart uses a symlog scale
        label += f"\n(logarithmic beyond ±{HEATMAP_LINEAR_THRESHOLD:g})"

    colorbar.set_label(label)
    fig.tight_layout()
    return fig


def generate_effect_size_heatmaps(
    output_dir: Path,
    scenario: str,
    comparison_data: pl.DataFrame,
    metric: Metric,
    effect_size: EffectSize,
    norm: Normalize,
    jdk_versions: dict[str, JdkVersion],
) -> None:
    """Generate an effect size heatmap of the specified metric for each variant of a scenario"""
    metric_label = get_metric_label(metric)
    heatmaps = prepare_heatmap_data(comparison_data, metric, effect_size, jdk_versions)

    for variant, data in heatmaps.items():
        title = f"{scenario} / {variant} — {metric_label}"
        fig = render_effect_size_heatmap(data, effect_size, norm, title)
        stem = f"{variant}_{underscore(metric_label)}_{underscore(effect_size)}_heatmap"
        try:
            export_plot(fig, output_dir, stem, "heatmaps")
        finally:
            plt.close(fig)


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

    ax.set_xlabel(style.label)
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
    """Draw the estimate and its interval as one centered monospaced column"""
    ax.axis("off")

    # `get_yaxis_transform` blends Axes-fraction x with data-space y, so each row
    # sits at exactly the height of its whisker on the left panel and the column
    # center stays put regardless of how wide the panel ends up. Every data row
    # renders to the same fixed width (via `style.plot_column_width`), so the
    # numeric rows stay aligned as a block; centering (rather than right-aligning)
    # also keeps the shorter header row visually centered above them.
    transform = ax.get_yaxis_transform()

    def draw_row(y: float, row_text: str, font: dict[str, Any]) -> None:
        ax.text(
            FOREST_PLOT_VALUE_COLUMN_CENTER, y, row_text,
            transform=transform, ha="center", va="center", **font,
        )  # fmt: skip

    draw_row(
        FOREST_PLOT_HEADER_Y,
        f"Estimate [{CI_LEVEL_PERCENT}% CI]",
        FOREST_PLOT_HEADER_FONT,
    )

    fmt = f"+z{style.plot_column_width}.{style.precision}f"
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
    style = EFFECT_SIZE_STYLES[effect_size]
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
    metric_label = get_metric_label(metric)
    title = f"{jdk_dir.name} / {variant_dir.name} — {metric_label}"

    plot_df = prepare_forest_plot_data(comparison_df, metric, effect_size)
    if plot_df.is_empty():
        logger.warning(
            "No plottable %s comparisons found for '%s'; skipping %s forest plot",
            metric, variant_dir, effect_size,
        )  # fmt: skip
        return

    fig = render_forest_plot(plot_df, effect_size, title)
    stem = f"{underscore(metric_label)}_{underscore(effect_size)}_forest_plot"
    try:
        export_plot(fig, variant_dir, stem, "forest plots")
    finally:
        plt.close(fig)


# Energy-Time Scatter Plot
def compute_scatter_panel_limits(
    *values: npt.NDArray[np.floating],
    padding: float = SCATTER_PLOT_AXIS_PADDING_FRACTION,
) -> tuple[float, float]:
    """Return the `padding`-widened `(min, max)` for a panel of an scatter plot"""
    # Pad on the drawn scale rather than on the data, so that the headroom is
    # the same share of an axis however many orders of magnitude it spans.
    transform = SymmetricalLogTransform(
        base=10,
        linthresh=SCATTER_PLOT_LINEAR_THRESHOLD,
        linscale=SCATTER_PLOT_LINEAR_SCALE,
    )
    low, high = transform.transform(
        [min(float(v.min()) for v in values), max(float(v.max()) for v in values)]
    )
    margin = (high - low) * padding
    low, high = transform.inverted().transform([low - margin, high + margin])

    # Clear zero by the linear region so a zero line never lies flush against a panel edge
    return (
        min(float(low), -SCATTER_PLOT_LINEAR_THRESHOLD),
        max(float(high), SCATTER_PLOT_LINEAR_THRESHOLD),
    )


def render_energy_time_scatter_panel(
    ax: plt.Axes,
    panel_df: pl.DataFrame,
    limits: tuple[float, float],
) -> list[plt.Text]:
    """Draw one JDK panel of an energy-time scatter plot, returning its unplaced point labels"""
    # Set before placing the points: adjust_text spreads labels within the
    # Axes' current view, so it must already be the final shared view
    # rather than the tight per-panel default matplotlib would autoscale to
    scale: dict[str, Any] = {
        "linthresh": SCATTER_PLOT_LINEAR_THRESHOLD,
        "linscale": SCATTER_PLOT_LINEAR_SCALE,
    }
    ax.set_xscale("symlog", **scale)
    ax.set_yscale("symlog", **scale)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.axhline(0, **SCATTER_PLOT_ZERO_LINE_STYLE)
    ax.axvline(0, **SCATTER_PLOT_ZERO_LINE_STYLE)
    ax.plot(limits, limits, **SCATTER_PLOT_IDENTITY_LINE_STYLE)

    x = panel_df.get_column("Time_Diff_Percent").to_numpy()
    y = panel_df.get_column("Energy_Diff_Percent").to_numpy()
    ax.scatter(x, y, color=SCATTER_PLOT_MARKER_COLOR, s=15, zorder=2)

    ax.text(
        *SCATTER_PLOT_STAT_POSITION,
        f"$\\rho = {spearman_rho(panel_df):.2f}$\n$n = {len(panel_df)}$",
        transform=ax.transAxes, ha="left", va="top", **SCATTER_PLOT_STAT_FONT,
    )  # fmt: skip

    return [
        ax.text(xi, yi, benchmark, **SCATTER_PLOT_LABEL_FONT)
        for xi, yi, benchmark in zip(
            x, y, panel_df.get_column("Benchmark").to_list(), strict=True
        )
    ]


def render_energy_time_scatter_grid(
    scenario_df: pl.DataFrame,
    scenario: str,
    jdks: list[str],
    limits: tuple[float, float],
) -> plt.Figure:
    """Render one scenario's grid of energy-time scatter panels, one per JDK"""
    rows = math.ceil(len(jdks) / SCATTER_PLOT_COLUMNS)
    fig, axes = plt.subplots(
        rows, SCATTER_PLOT_COLUMNS,
        figsize=(
            SCATTER_PLOT_COLUMNS * SCATTER_PLOT_PANEL_WIDTH_INCHES
            + SCATTER_PLOT_HORIZONTAL_MARGIN_INCHES,
            rows * SCATTER_PLOT_PANEL_HEIGHT_INCHES
            + SCATTER_PLOT_VERTICAL_MARGIN_INCHES,
        ),
        sharex=True, sharey=True, squeeze=False,
    )  # fmt: skip

    flat_axes: list[Axes] = list(axes.flat)
    panel_texts = []
    for ax, jdk in zip(flat_axes, jdks, strict=False):
        panel_df = scenario_df.filter(pl.col("JDK") == jdk)
        panel_texts.append((ax, render_energy_time_scatter_panel(ax, panel_df, limits)))
        ax.set_title(jdk, **SCATTER_PLOT_TITLE_FONT)

        # Sharing the axes hides every tick label that another panel repeats
        # below or to the left. Put them back, since the panels are spread
        # down a whole page and each one has to be readable on its own.
        ax.tick_params(
            labelsize=SCATTER_PLOT_TICK_FONT["size"],
            labelbottom=True,
            labelleft=True,
        )

        # Read the ticks in the same terms as the points, not in scientific notation
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(
                SymmetricalLogLocator(linthresh=SCATTER_PLOT_LINEAR_THRESHOLD, base=10)
            )
            axis.set_major_formatter(FuncFormatter(lambda tick, _: f"{tick:g}"))

    # The panels fill whole rows, so the grid can end with empty slots
    for ax in flat_axes[len(jdks) :]:
        ax.set_axis_off()

    # Make it clear that the panels use a symlog scale
    suffix = f"\n(logarithmic beyond ±{SCATTER_PLOT_LINEAR_THRESHOLD:g})"
    fig.suptitle(scenario.capitalize(), **SCATTER_PLOT_SUPTITLE_FONT)
    fig.supxlabel(f"$\\Delta$time (%){suffix}", ma="center", **SCATTER_PLOT_TITLE_FONT)
    fig.supylabel(
        f"$\\Delta$energy (%){suffix}", ma="center", **SCATTER_PLOT_TITLE_FONT
    )
    fig.tight_layout()

    fig.canvas.draw()
    for ax, texts in panel_texts:
        adjust_text(
            texts,
            ax=ax,
            iter_lim=SCATTER_PLOT_LABEL_ITERATIONS,
            arrowprops=SCATTER_PLOT_LABEL_ARROW_STYLE,
        )

    return fig


def generate_energy_time_scatter_plot(
    output_dir: Path,
    scenario_df: pl.DataFrame,
    scenario: str,
    jdks: list[str],
    limits: tuple[float, float],
    stem: str,
) -> None:
    """Generate one scenario's grid of energy-time scatter panels"""
    fig = render_energy_time_scatter_grid(scenario_df, scenario, jdks, limits)
    try:
        export_plot(fig, output_dir, stem, "energy-time scatter plot")
    finally:
        plt.close(fig)


def energy_time_scatter_plot_tasks(
    root_dir: Path,
    long_df: pl.DataFrame,
    jdk_versions: dict[str, JdkVersion],
) -> list[Callable[[], None]]:
    """Return an energy-time scatter plot for each variant and scenario"""
    # Computed once from both metrics of every variant and scenario, so that
    # all the scatter plots share axis limits and stay comparable with each other.
    limits = compute_scatter_panel_limits(
        long_df.get_column("Time_Diff_Percent").to_numpy(),
        long_df.get_column("Energy_Diff_Percent").to_numpy(),
    )

    plot_tasks: list[Callable[[], None]] = []
    for variant in sorted(long_df.get_column("Variant").unique()):
        variant_df = long_df.filter(pl.col("Variant") == variant)
        jdks = sort_jdks(variant_df.get_column("JDK").unique().to_list(), jdk_versions)

        plot_tasks += [
            functools.partial(
                generate_energy_time_scatter_plot,
                root_dir / scenario,
                variant_df.filter(pl.col("Scenario") == scenario),
                scenario,
                jdks,
                limits,
                f"{variant}_energy_time_scatter_plot",
            )
            for scenario in sorted(variant_df.get_column("Scenario").unique())
        ]

    return plot_tasks


# Orchestration
def generate_stats(root_dir: Path, executor: Executor | None) -> None:
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
    for jdk_dir in sorted(d for d in root_dir.iterdir() if d.is_dir()):
        for implementation_dir in sorted(d for d in jdk_dir.iterdir() if d.is_dir()):
            generate_combined_summary(implementation_dir)

    # 3. Generate comparative statistics
    generate_comparisons(root_dir, executor)

    # 4. Log completion
    logger.info(
        "Processing complete. Processed %d measurement files.",
        len(measurement_files),
    )
    logger.debug(
        "Measurement cache stats: %s",
        load_and_clean_measurements.cache_info(),
    )


def run(root_dir: Path, jobs: int) -> None:
    """Process every scenario subdirectory of `root_dir` and generate overview statistics"""
    if not root_dir.exists():
        logger.error("Root directory '%s' does not exist", root_dir)
        return

    scenario_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    logger.info(
        "Processing %d scenario director%s: %s over %d process%s",
        len(scenario_dirs), "y" if len(scenario_dirs) == 1 else "ies",
        [d.name for d in scenario_dirs], jobs, "" if jobs == 1 else "es",
    )  # fmt: skip
    with worker_pool(jobs) as executor:
        for scenario_dir in scenario_dirs:
            generate_stats(scenario_dir, executor)

        generate_overall_stats(root_dir, scenario_dirs, executor)


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
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    args = parser.parse_args()

    root_dir = args.root_dir.resolve()
    setup_logging(root_dir, args.verbose)

    logger.info("Using root directory '%s'", root_dir)
    run(root_dir, args.jobs)


if __name__ == "__main__":
    main()
