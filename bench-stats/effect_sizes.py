"""
Effect sizes of a variant against the baseline, their confidence intervals, and the energy-time correlation.
"""

import functools
import hashlib
import logging
import math
import warnings
from enum import StrEnum
from typing import Any, Final, Literal, NamedTuple, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import DegenerateDataWarning, bootstrap, iqr, norm, rankdata

from measurements import Metric, SampleKey
from rank_corr import ClusteredCorrelation, total_spearman

logger = logging.getLogger(__name__)


# One minus the confidence level of every confidence interval
ALPHA = 0.05

# How every IQR interpolates between order statistics. Summaries and comparisons
# share it, so both report the same IQR of a sample.
IQR_INTERPOLATION_METHOD: Final = "linear"

# Valid methods for computing bootstrap confidence intervals
type BootstrapMethod = Literal["percentile", "basic", "bca", "BCa"]

# Bootstrap method of the median-based difference.
#
# Cliff's delta is not bootstrapped because no resampling method keeps its width
# where the samples stop overlapping, which is why its interval is a score interval.
MEDIAN_DIFFERENCE_BOOTSTRAP_METHOD: BootstrapMethod = "percentile"

# Default number of resamples per bootstrap.
BOOTSTRAP_DEFAULT_N_RESAMPLES = 9_999

# Halvings that place a score interval limit, past the resolution of a 64-bit float
BISECTIONS = 60

# Root seed of every bootstrap. Each bootstrap derives its generator from this seed
# and a deterministic sample key, so bootstrapped confidence intervals are deterministic.
ROOT_SEED = 42

# Effect size classes from Romano et al., 2006
type EffectSizeClass = Literal["negligible", "small", "medium", "large"]


# Supported effect size measures
class EffectSize(StrEnum):
    """An effect size, named exactly as its column in `ComparisonRow`."""

    MEDIAN_DIFFERENCE_PERCENT = "Median_Difference_Percent"
    CLIFFS_DELTA = "Cliffs_Delta"


class ConfidenceInterval(NamedTuple):
    """The lower and upper limits of a confidence interval"""

    low: float
    high: float


class TwoSampleStatistic(Protocol):
    def __call__(
        self,
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        /,
        *,
        axis: int = 0,
    ) -> np.floating | npt.NDArray[np.floating]: ...


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


# Names every scenario at once, in the energy-time correlation table
ENERGY_TIME_CORRELATION_ALL_LABEL = "All"


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
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    axis: int = 0,
) -> np.floating | npt.NDArray[np.floating]:
    """Return `median(y) - median(x)`, as a percentage of `median(x)."""
    med_x = np.median(x, axis=axis)
    med_y = np.median(y, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (med_y - med_x) / med_x * 100.0


def mannwhitneyu_statistic(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
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
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
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


def newcombe_score_ci(
    variant: npt.NDArray[np.floating],
    baseline: npt.NDArray[np.floating],
    alpha: float = ALPHA,
) -> ConfidenceInterval:
    """
    Return Newcombe's method 5 interval for Cliff's delta.

    Ports method 5 of Newcombe (2006, part 2), a score interval for the Mann-Whitney
    parameter `theta`, which is Cliff's delta mapped onto [0, 1]. The variance is the
    one Hanley and McNeil derived from an exponential model, with the two sample sizes
    replaced by their average, as method 5 prescribes.

    A Wald interval fixes that variance at the estimate, which leaves no width wherever
    the samples do not overlap. This method instead solves for the values of `theta`
    whose own variance places the estimate at the limit of the interval. The limits stay
    within the bounds of the statistic, and the interval keeps its width at the bounds.

    The equation has no closed form. The estimate always lies within the interval, and
    a bound of the statistic lies outside it unless the estimate reaches that bound, so
    each limit is bisected between the two.
    """
    z = norm.ppf(1 - alpha / 2)
    m, n = len(variant), len(baseline)
    theta = (float(cliffs_delta(variant, baseline)) + 1) / 2
    size = (m + n) / 2 - 1

    def score(value: float) -> float:
        """Return what separates `value` from the estimate, at most zero within the interval"""
        variance = (
            value
            * (1 - value)
            * (1 + size * (1 - value) / (2 - value) + size * value / (1 + value))
            / (m * n)
        )
        return (value - theta) ** 2 - z**2 * variance

    def limit(bound: float) -> float:
        """
        Return the limit of the interval between `bound` and the estimate.

        An estimate that reaches `bound` leaves nothing to bisect, and the limit is the
        bound itself, which is where the interval of such an estimate ends.
        """
        outside, inside = bound, theta
        for _ in range(BISECTIONS):
            middle = (outside + inside) / 2
            if score(middle) > 0:
                outside = middle
            else:
                inside = middle

        return (outside + inside) / 2

    # Map the limits back onto the scale of Cliff's delta
    return ConfidenceInterval(2 * limit(0.0) - 1, 2 * limit(1.0) - 1)


def two_sample_bootstrap_ci(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    statistic: TwoSampleStatistic,
    rng: np.random.Generator,
    sample_key: SampleKey,
    metric: str,
    method: BootstrapMethod,
    fallback_method: BootstrapMethod | None = None,
    *,
    n_resamples: int,
) -> ConfidenceInterval:
    """
    Return the bootstrap confidence interval for a custom two-sample statistic.

    A `fallback_method` recomputes the interval from the same resamples when
    `method` yields a non-finite interval (as BCa does for degenerate data).
    """
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
        res = resample(n_resamples=n_resamples, method=method)

        ci = res.confidence_interval
        if np.isfinite([ci.low, ci.high]).all() or fallback_method is None:
            return ConfidenceInterval(ci.low, ci.high)

        logger.info(
            "%s/%s: degenerate data encountered. Falling back to '%s' bootstrap...",
            sample_key,
            metric,
            fallback_method,
        )
        # The resamples are sound and can be reused.
        res = resample(
            n_resamples=0,
            bootstrap_result=res,
            method=fallback_method,
        )

    ci = res.confidence_interval
    return ConfidenceInterval(ci.low, ci.high)


def compare_metric(
    sample_key: SampleKey,
    metric: str,
    baseline_values: npt.NDArray[np.float64],
    variant_values: npt.NDArray[np.float64],
    n_resamples: int = BOOTSTRAP_DEFAULT_N_RESAMPLES,
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
        sample_key, metric, MEDIAN_DIFFERENCE_BOOTSTRAP_METHOD,
        n_resamples=n_resamples,
    )  # fmt: skip
    iqr_x = iqr(x, interpolation=IQR_INTERPOLATION_METHOD)
    iqr_y = iqr(y, interpolation=IQR_INTERPOLATION_METHOD)

    # Pass the variant as the first sample so that the sign of Cliff's delta
    # indicates the direction of the difference between variant and baseline.
    cliffs_d = float(cliffs_delta(y, x))
    cliffs_d_ci = newcombe_score_ci(y, x)

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


def build_long_time_and_energy_table(comparison_data: pl.DataFrame) -> pl.DataFrame:
    """
    Build a long table of the time and energy differences of each benchmark

    The columns are Benchmark, JDK, Scenario, Variant, `Time_Diff_Percent` and
    `Energy_Diff_Percent`, each with its `_CI_Low` and `_CI_High` bounds.
    """
    keys = ["Program", "JDK", "Scenario", "Variant"]

    def select_metric(metric: Metric, prefix: str) -> pl.DataFrame:
        return comparison_data.filter(pl.col("Metric") == metric).select(
            *keys,
            *(
                pl.col(f"Median_Difference_Percent{suffix}").alias(
                    f"{prefix}_Diff_Percent{suffix}"
                )
                for suffix in ("", "_CI_Low", "_CI_High")
            ),
        )

    time_df = select_metric(Metric.TIME, "Time")
    energy_df = select_metric(Metric.PACKAGE_AND_DRAM, "Energy")

    long_df = time_df.join(energy_df, on=keys, how="inner")
    finite = (
        pl.col("Time_Diff_Percent").is_finite()
        & pl.col("Energy_Diff_Percent").is_finite()
    )
    return long_df.filter(finite).rename({"Program": "Benchmark"})


def clustered_spearman(subset: pl.DataFrame) -> ClusteredCorrelation:
    """
    Correlate the time and energy differences of `subset` by Spearman's rank coefficient, allowing for benchmarks

    A benchmark recurs under every JDK and scenario of `subset`, so its rows are not
    independent, and each benchmark counts as one cluster. Where each benchmark occurs
    once, as in a scatter plot panel, this is the plain coefficient with an interval.
    A subset that cannot be correlated gives NaN and a warning, rather than stopping
    the run.
    """
    benchmarks = subset.get_column("Benchmark")
    try:
        correlation = total_spearman(
            subset.get_column("Time_Diff_Percent").to_numpy(),
            subset.get_column("Energy_Diff_Percent").to_numpy(),
            benchmarks.to_numpy(),
            confidence=1 - ALPHA,
        )
    except ValueError as error:
        logger.warning(
            "Skipped the correlation of %d rows of %d benchmarks: %s",
            len(subset),
            benchmarks.n_unique(),
            error,
        )
        return ClusteredCorrelation(
            math.nan, math.nan, math.nan, math.nan, benchmarks.n_unique(), len(subset)
        )
    if not math.isfinite(correlation.estimate):
        logger.warning(
            "Skipped the correlation of %d rows of %d benchmarks: time or energy is constant",
            len(subset),
            correlation.n_clusters,
        )
    return correlation


def spearman_row(
    variant: str,
    scenario: str,
    subset: pl.DataFrame,
) -> dict[str, Any]:
    """Compute one row of the energy-time correlation table"""
    correlation = clustered_spearman(subset)
    return {
        "Variant": variant,
        "Scenario": scenario,
        "N": len(subset),
        "Spearman_Rho": correlation.estimate,
        "Program_Count": correlation.n_clusters,
        "Spearman_Rho_SE": correlation.se,
        "Spearman_Rho_CI_Low": correlation.low,
        "Spearman_Rho_CI_High": correlation.high,
    }


def generate_energy_time_correlation_table(long_df: pl.DataFrame) -> pl.DataFrame:
    """Compute the clustered Spearman correlation between Time_Diff_Percent and Energy_Diff_Percent, per variant and scenario"""
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
