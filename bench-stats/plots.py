"""
Forest plots, heatmaps, scatter plots and trend plots of the comparisons, exported as PGF.
"""

import functools
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.axes import Axes

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from adjustText import adjust_text  # type: ignore[import-untyped]
from inflection import underscore
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator

from effect_sizes import ALPHA, EffectSize, clustered_spearman
from measurements import JdkVersion, Metric, sort_jdks

logger = logging.getLogger(__name__)


# Human-readable names for the metrics that get plotted.
# Omitted metric keys default to their value from the `Metric` enum.
METRIC_LABELS: dict[Metric, str] = {
    Metric.PACKAGE_AND_DRAM: "Energy",
}


def get_metric_label(metric: Metric) -> str:
    """Return the human-readable label for `metric`"""
    return METRIC_LABELS.get(metric, str(metric))


# Confidence level (%) corresponding to `ALPHA`, for display purposes
CI_LEVEL_PERCENT = round((1 - ALPHA) * 100)


# Panel configuration, shared by the scatter plots and the trend plots

# Fonts for the panel titles and axis ticks. Matplotlib's defaults are sized for
# a figure far larger than one of these panels, so both are set explicitly.
PANEL_TITLE_FONT: dict[str, Any] = {"size": 9}
PANEL_TICK_FONT: dict[str, Any] = {"size": 7}

# Reference line at zero, drawn under the data of every panel
ZERO_LINE_STYLE: dict[str, Any] = {
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.8,
    "zorder": 1,
}


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
# the axis labels and the colorbar, all in inches.
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

# Fonts for the per-point benchmark labels and the per-panel correlation annotation
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

# Size of a single panel and of the margins reserved for the axis labels,
# all in inches. The figure grows with the number of panels. Scaling
# it to a given page is left to whatever includes it.
SCATTER_PLOT_PANEL_WIDTH_INCHES = 2.9
SCATTER_PLOT_PANEL_HEIGHT_INCHES = 2.7
SCATTER_PLOT_HORIZONTAL_MARGIN_INCHES = 0.5
SCATTER_PLOT_VERTICAL_MARGIN_INCHES = 1.0

# Identity line under the points of every panel
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


# Trend plot configuration

# Number of columns the benchmark panels are laid out over
TREND_PLOT_COLUMNS = 3

# Width of the text block of an A4 page (8.27 in) with 1 in margins.
# The figure has this width, so it fits the page without rescaling.
TREND_PLOT_WIDTH_INCHES = 6.27

# Height of a single panel and of the margin reserved for the axis labels,
# all in inches. Six rows of panels still fit the text height of an A4 page.
TREND_PLOT_PANEL_HEIGHT_INCHES = 1.4
TREND_PLOT_VERTICAL_MARGIN_INCHES = 0.9

# Color of the estimate line and of its confidence band
TREND_PLOT_LINE_COLOR = FOREST_PLOT_MARKER_COLOR
TREND_PLOT_BAND_ALPHA = 0.25

# Effect size drawn by every trend plot
TREND_PLOT_EFFECT_SIZE = EffectSize.MEDIAN_DIFFERENCE_PERCENT

# Columns that must be finite for a comparison to be drawn in a trend plot
TREND_PLOT_FINITE_COLUMNS = (
    TREND_PLOT_EFFECT_SIZE,
    f"{TREND_PLOT_EFFECT_SIZE}_CI_Low",
    f"{TREND_PLOT_EFFECT_SIZE}_CI_High",
)

# Which metrics to render a trend plot of for every variant and scenario
TREND_PLOTS: tuple[Metric, ...] = (Metric.PACKAGE_AND_DRAM,)


def export_plot(fig: plt.Figure, output_dir: Path, stem: str, description: str) -> None:
    """Write `fig` into `output_dir` as a PGF file named `stem`"""
    pgf_path = output_dir / f"{stem}.pgf"
    fig.savefig(pgf_path, format="pgf", backend="pgf")
    logger.info("Created %s '%s'", description, pgf_path)


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
        fig = render_effect_size_heatmap(data, effect_size, norm)
        stem = f"{variant}_{underscore(metric_label)}_{underscore(effect_size)}_heatmap"
        try:
            export_plot(fig, output_dir, stem, "heatmap")
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

    draw_forest_plot_graph(ax, plot_df, style)
    draw_forest_plot_table(ax_labels, plot_df, style)

    # Set last, and on one Axes only: `sharey` propagates it, and it has to
    # survive the autoscaling that the panels' own drawing would otherwise trigger
    ax.set_ylim(n - 0.5, FOREST_PLOT_HEADER_Y - 0.5)

    fig.tight_layout()
    return fig


def generate_forest_plot(
    variant_dir: Path,
    comparison_df: pl.DataFrame,
    metric: Metric,
    effect_size: EffectSize,
) -> None:
    """Generate a forest plot for the specified variant directory, metric and effect size"""
    metric_label = get_metric_label(metric)

    plot_df = prepare_forest_plot_data(comparison_df, metric, effect_size)
    if plot_df.is_empty():
        logger.warning(
            "No plottable %s comparisons found for '%s'; skipping %s forest plot",
            metric, variant_dir, effect_size,
        )  # fmt: skip
        return

    fig = render_forest_plot(plot_df, effect_size)
    stem = f"{underscore(metric_label)}_{underscore(effect_size)}_forest_plot"
    try:
        export_plot(fig, variant_dir, stem, "forest plot")
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


def format_scatter_statistics(panel_df: pl.DataFrame) -> str:
    """Format the correlation of `panel_df`"""
    correlation = clustered_spearman(panel_df)
    size = f"$n = {len(panel_df)}$"
    values = (correlation.estimate, correlation.low, correlation.high)
    if any(not np.isfinite(v) for v in values):
        return size

    return (
        f"$\\rho = {correlation.estimate:.2f}$ "
        f"[{correlation.low:.2f}, {correlation.high:.2f}]\n{size}"
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

    ax.axhline(0, **ZERO_LINE_STYLE)
    ax.axvline(0, **ZERO_LINE_STYLE)
    ax.plot(limits, limits, **SCATTER_PLOT_IDENTITY_LINE_STYLE)

    x = panel_df.get_column("Time_Diff_Percent").to_numpy()
    y = panel_df.get_column("Energy_Diff_Percent").to_numpy()
    ax.scatter(x, y, color=SCATTER_PLOT_MARKER_COLOR, s=15, zorder=2)

    ax.text(
        *SCATTER_PLOT_STAT_POSITION,
        format_scatter_statistics(panel_df),
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
        ax.set_title(jdk, **PANEL_TITLE_FONT)

        # Sharing the axes hides every tick label that another panel repeats
        # below or to the left. Put them back, since the panels are spread
        # down a whole page and each one has to be readable on its own.
        ax.tick_params(
            labelsize=PANEL_TICK_FONT["size"],
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
    fig.supxlabel(f"$\\Delta$time (%){suffix}", ma="center", **PANEL_TITLE_FONT)
    fig.supylabel(f"$\\Delta$energy (%){suffix}", ma="center", **PANEL_TITLE_FONT)
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
    jdks: list[str],
    limits: tuple[float, float],
    stem: str,
) -> None:
    """Generate one scenario's grid of energy-time scatter panels"""
    fig = render_energy_time_scatter_grid(scenario_df, jdks, limits)
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
                jdks,
                limits,
                f"{variant}_energy_time_scatter_plot",
            )
            for scenario in sorted(variant_df.get_column("Scenario").unique())
        ]

    return plot_tasks


# Trend Plot
def render_trend_panel(
    ax: plt.Axes,
    benchmark_df: pl.DataFrame,
    jdk_labels: dict[str, str],
) -> None:
    """Draw one benchmark panel of a trend plot, with its confidence band"""
    # Aligned to the sorted JDKs, so that a missing JDK becomes a gap in the line
    panel_df = pl.DataFrame({"JDK": list(jdk_labels)}).join(
        benchmark_df, on="JDK", how="left", maintain_order="left"
    )
    x = np.arange(len(jdk_labels))
    estimates, ci_low, ci_high = (
        panel_df.get_column(column).to_numpy() for column in TREND_PLOT_FINITE_COLUMNS
    )

    ax.fill_between(
        x, ci_low, ci_high,
        color=TREND_PLOT_LINE_COLOR, alpha=TREND_PLOT_BAND_ALPHA, linewidth=0,
    )  # fmt: skip

    # Markers keep a point visible when gaps surround it
    ax.plot(x, estimates, color=TREND_PLOT_LINE_COLOR, marker="o", markersize=3)

    # Limits are frozen before drawing the zero line, so that the panel fits its data.
    # The line then only shows when zero lies within that range.
    ax.set_ylim(ax.get_ylim())
    ax.axhline(0, **ZERO_LINE_STYLE)

    ax.set_xticks(x, labels=list(jdk_labels.values()))
    ax.tick_params(labelsize=PANEL_TICK_FONT["size"])


def render_trend_grid(
    scenario_df: pl.DataFrame,
    metric: Metric,
    jdk_labels: dict[str, str],
) -> plt.Figure:
    """Render one scenario's grid of trend panels of a metric, one per benchmark"""
    benchmarks = sorted(scenario_df.get_column("Program").unique())
    rows = math.ceil(len(benchmarks) / TREND_PLOT_COLUMNS)
    fig, axes = plt.subplots(
        rows, TREND_PLOT_COLUMNS,
        figsize=(
            TREND_PLOT_WIDTH_INCHES,
            rows * TREND_PLOT_PANEL_HEIGHT_INCHES + TREND_PLOT_VERTICAL_MARGIN_INCHES,
        ),
        squeeze=False,
    )  # fmt: skip

    flat_axes: list[Axes] = list(axes.flat)
    for ax, benchmark in zip(flat_axes, benchmarks, strict=False):
        benchmark_df = scenario_df.filter(pl.col("Program") == benchmark)
        render_trend_panel(ax, benchmark_df, jdk_labels)
        ax.set_title(benchmark, **PANEL_TITLE_FONT)

    # The panels fill whole rows, so the grid can end with empty slots
    for ax in flat_axes[len(benchmarks) :]:
        ax.set_axis_off()

    fig.supxlabel("Java version", **PANEL_TITLE_FONT)
    fig.supylabel(
        f"$\\Delta${get_metric_label(metric).lower()} (%)\n"
        f"(shaded: {CI_LEVEL_PERCENT}% CI)",
        ma="center", **PANEL_TITLE_FONT,
    )  # fmt: skip
    fig.tight_layout()
    return fig


def generate_trend_plot(
    output_dir: Path,
    scenario_df: pl.DataFrame,
    metric: Metric,
    jdk_labels: dict[str, str],
    stem: str,
) -> None:
    """Generate one scenario's grid of trend panels of a metric"""
    fig = render_trend_grid(scenario_df, metric, jdk_labels)
    try:
        export_plot(fig, output_dir, stem, "trend plot")
    finally:
        plt.close(fig)


def trend_plot_tasks(
    root_dir: Path,
    comparison_data: pl.DataFrame,
    metric: Metric,
    jdk_versions: dict[str, JdkVersion],
) -> list[Callable[[], None]]:
    """Return a trend plot of a metric for each variant and scenario"""
    metric_df = comparison_data.filter(
        (pl.col("Metric") == metric)
        & pl.all_horizontal(
            pl.col(TREND_PLOT_FINITE_COLUMNS).is_finite().fill_null(False)
        )
    )

    # Sorted once for every trend plot of the metric, so each JDK has
    # the same position in every panel, whichever JDKs a benchmark supports
    jdks = sort_jdks(metric_df.get_column("JDK").unique().to_list(), jdk_versions)

    # Ordered like `jdks`. Each JDK is labeled with its major version,
    # or with its name if its version is unknown.
    jdk_labels = {
        jdk: str(jdk_versions[jdk][0]) if jdk in jdk_versions else jdk for jdk in jdks
    }
    stem_suffix = f"{underscore(get_metric_label(metric))}_trend_plot"

    plot_tasks: list[Callable[[], None]] = []
    for variant in sorted(metric_df.get_column("Variant").unique()):
        variant_df = metric_df.filter(pl.col("Variant") == variant)
        plot_tasks += [
            functools.partial(
                generate_trend_plot,
                root_dir / scenario,
                variant_df.filter(pl.col("Scenario") == scenario),
                metric,
                jdk_labels,
                f"{variant}_{stem_suffix}",
            )
            for scenario in sorted(variant_df.get_column("Scenario").unique())
        ]

    return plot_tasks
