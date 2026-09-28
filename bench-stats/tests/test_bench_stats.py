"""Tests of the comparison and the correlation of bench-stats against known answers"""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from effect_sizes import (
    ConfidenceInterval,
    compare_metric,
    generate_energy_summary_by_jdk_table,
    generate_energy_summary_table,
    generate_energy_time_correlation_table,
)
from measurements import SampleKey, get_jdk_labels, parse_jdk_version
from plots import format_scatter_statistics, render_energy_time_scatter_panel
from rank_corr import total_spearman

KEY = SampleKey("cold", "temurin-21", "streams", "fop")
METRIC = "PackageAndDRAM"

CORRELATION_COLUMNS = [
    "Variant",
    "Scenario",
    "N",
    "Spearman_Rho",
    "Program_Count",
    "Spearman_Rho_SE",
    "Spearman_Rho_CI_Low",
    "Spearman_Rho_CI_High",
]


def long_table(benchmark_count: int) -> pl.DataFrame:
    """Build the time and energy differences of benchmarks under two JDKs and two scenarios"""
    rng = np.random.default_rng(1)
    effects = rng.normal(size=benchmark_count)
    return pl.DataFrame([
        {
            "Benchmark": f"benchmark-{index}",
            "JDK": jdk,
            "Scenario": scenario,
            "Variant": "streams",
            "Time_Diff_Percent": 10 * effect + rng.normal(),
            "Energy_Diff_Percent": 12 * effect + rng.normal(),
        }
        for index, effect in enumerate(effects)
        for jdk in ("temurin-17", "temurin-21")
        for scenario in ("cold", "warm")
    ])  # fmt: skip


def scatter_panel(benchmark_count: int) -> pl.DataFrame:
    """Build the rows of one panel of a scatter plot: one JDK of one scenario"""
    return long_table(benchmark_count).filter(
        (pl.col("JDK") == "temurin-17") & (pl.col("Scenario") == "cold")
    )


def test_cliffs_delta_interval_keeps_its_width_for_disjoint_samples():
    # A bootstrap interval has no width where the samples do not overlap
    baseline = np.linspace(90.0, 100.0, 20)
    variant = np.linspace(110.0, 120.0, 20)

    row = compare_metric(KEY, METRIC, baseline, variant)

    assert row is not None
    assert row["Cliffs_Delta"] == 1.0
    assert row["Cliffs_Delta_CI_Low"] < 1.0
    assert row["Cliffs_Delta_CI_High"] == 1.0


def test_non_finite_sample_is_refused():
    baseline = np.linspace(90.0, 100.0, 20)
    variant = np.linspace(110.0, 120.0, 20)
    variant[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        compare_metric(KEY, METRIC, baseline, variant)


def test_median_difference_interval_matches_its_values_before_the_switch():
    # The interval values were computed by the bootstrap of the median-based difference
    # before Cliff's delta stopped being bootstrapped. Each statistic has a stream of its
    # own, so the change of one must leave the other as it was.
    baseline = np.array([
        99.5, 102.5, 96.79, 91.01, 95.06, 90.09, 100.08, 113.75, 94.71, 93.5,
        104.48, 103.1, 100.54, 90.65, 99.19, 106.65, 86.97, 95.03, 82.26, 87.45,
    ])  # fmt: skip
    variant = np.array([
        91.45, 107.39, 96.86, 112.97, 111.68, 107.91, 85.48, 104.18, 109.42, 111.2,
        94.35, 104.82, 99.7, 101.4, 122.25, 101.42, 109.59, 120.11, 103.71, 108.73,
    ])  # fmt: skip

    row = compare_metric(KEY, METRIC, baseline, variant)

    assert row is not None
    assert row["Median_Difference_Percent_CI_Low"] == pytest.approx(3.0703517587939766)
    assert row["Median_Difference_Percent_CI_High"] == pytest.approx(16.28877005347594)


def test_correlation_counts_rows_and_benchmarks(tmp_path: Path):
    path = tmp_path / "table.csv"
    table = generate_energy_time_correlation_table(path, long_table(15))
    assert table.columns == CORRELATION_COLUMNS
    assert table.get_column("Scenario").to_list() == ["All", "Cold", "Warm"]
    assert table.get_column("N").to_list() == [60, 30, 30]
    assert table.get_column("Program_Count").to_list() == [15, 15, 15]


def test_correlation_is_clustered(tmp_path: Path):
    long_df = long_table(15)
    table = generate_energy_time_correlation_table(tmp_path / "table.csv", long_df)
    row = table.row(0, named=True)
    expected = total_spearman(
        long_df.get_column("Time_Diff_Percent"),
        long_df.get_column("Energy_Diff_Percent"),
        long_df.get_column("Benchmark"),
    )
    assert row["Spearman_Rho"] == pytest.approx(expected.estimate)
    assert row["Spearman_Rho_SE"] == pytest.approx(expected.se)
    assert row["Spearman_Rho_CI_Low"] == pytest.approx(expected.low)
    assert row["Spearman_Rho_CI_High"] == pytest.approx(expected.high)
    assert (
        row["Spearman_Rho_CI_Low"] < row["Spearman_Rho"] < row["Spearman_Rho_CI_High"]
    )


def test_correlation_of_too_few_benchmarks_is_undefined(tmp_path: Path, caplog):
    path = tmp_path / "table.csv"
    with caplog.at_level(logging.WARNING):
        table = generate_energy_time_correlation_table(path, long_table(2))

    row = table.row(0, named=True)
    assert "Skipped the correlation of 8 rows of 2 benchmarks" in caplog.text
    assert math.isnan(row["Spearman_Rho"])
    assert (row["N"], row["Program_Count"]) == (8, 2)


def test_correlation_of_a_constant_difference_is_undefined(tmp_path: Path, caplog):
    long_df = long_table(15).with_columns(pl.lit(1.0).alias("Energy_Diff_Percent"))
    path = tmp_path / "table.csv"
    with caplog.at_level(logging.WARNING):
        table = generate_energy_time_correlation_table(path, long_df)

    row = table.row(0, named=True)
    assert "time or energy is constant" in caplog.text
    assert math.isnan(row["Spearman_Rho"])
    assert row["Program_Count"] == 15


def test_scatter_statistics_show_the_correlation_with_its_interval():
    panel_df = scatter_panel(15)

    text = format_scatter_statistics(panel_df)

    expected = total_spearman(
        panel_df.get_column("Time_Diff_Percent"),
        panel_df.get_column("Energy_Diff_Percent"),
        panel_df.get_column("Benchmark"),
    )
    assert text == (
        f"$\\rho = {expected.estimate:.2f}$ [{expected.low:.2f}, {expected.high:.2f}]"
    )


def test_scatter_statistics_show_an_undefined_correlation_without_interval():
    # Two benchmarks are too few to correlate. A perfect correlation has no interval
    # on the Fisher scale, so its estimate is not shown either, as in the heatmaps.
    perfect = scatter_panel(15).with_columns(
        pl.col("Time_Diff_Percent").alias("Energy_Diff_Percent")
    )

    assert format_scatter_statistics(scatter_panel(2)) == r"$\rho$ undefined"
    assert format_scatter_statistics(perfect) == r"$\rho$ undefined"


def test_nan_interval_draws_a_hollow_marker():
    # Every interval is its estimate alone, which excludes 0, but the first has a NaN bound
    panel_df = scatter_panel(15).with_columns(
        pl.col(column).alias(f"{column}{suffix}")
        for column in ("Time_Diff_Percent", "Energy_Diff_Percent")
        for suffix in ("_CI_Low", "_CI_High")
    )
    first_row = pl.int_range(pl.len()) == 0
    panel_df = panel_df.with_columns(
        pl.when(first_row)
        .then(math.nan)
        .otherwise(pl.col("Time_Diff_Percent_CI_Low"))
        .alias("Time_Diff_Percent_CI_Low")
    )

    fig, ax = plt.subplots()
    render_energy_time_scatter_panel(ax, panel_df, (-100.0, 100.0), label_points=False)
    hollow = ax.collections[-1]
    plt.close(fig)

    expected = panel_df.select("Time_Diff_Percent", "Energy_Diff_Percent").row(0)
    assert hollow.get_offsets().tolist() == [list(expected)]


JDK_VERSIONS = {
    "temurin-17": (17, 0, 18),
    "temurin-21": (21, 0, 11),
    "corretto-21": (21, 0, 7),
    "temurin-25": (25, 0, 3),
}


def test_jdks_are_labelled_by_major_version():
    labels = get_jdk_labels(["temurin-17", "temurin-21"], JDK_VERSIONS)
    assert labels == {"temurin-17": "17", "temurin-21": "21"}


@pytest.mark.parametrize(
    "jdks",
    [["temurin-21", "corretto-21"], ["temurin-17", "mystery"]],
    ids=["shared-major-version", "unknown-version"],
)
def test_jdks_a_major_version_cannot_tell_apart_are_labelled_by_name(jdks):
    labels = get_jdk_labels(jdks, JDK_VERSIONS)
    assert labels == {jdk: jdk for jdk in jdks}


@pytest.mark.parametrize(
    ("version_string", "expected"),
    [
        ("1.8.0_492", (8, 0, 492)),
        ("11.0.27", (11, 0, 27)),
        ("21", (21, 0, 0)),
        ("25.0.3+9-LTS", (25, 0, 3)),
    ],
)
def test_jdk_version_is_parsed_from_legacy_and_jep_223_version_strings(
    version_string, expected
):
    assert parse_jdk_version(f'openjdk version "{version_string}"') == expected


def energy_comparisons(
    jdk: str,
    estimates: list[float],
    intervals: list[ConfidenceInterval] | None = None,
) -> pl.DataFrame:
    """Build the energy comparisons of one JDK, a benchmark per estimate, each interval 1 either side of it by default"""
    if intervals is None:
        intervals = [ConfidenceInterval(value - 1, value + 1) for value in estimates]

    return pl.DataFrame([
        {
            "Program": f"benchmark-{index}",
            "Metric": "PackageAndDRAM",
            "Variant": "streams",
            "Scenario": "cold",
            "JDK": jdk,
            "Median_Difference_Percent": estimate,
            "Median_Difference_Percent_CI_Low": interval.low,
            "Median_Difference_Percent_CI_High": interval.high,
        }
        for index, (estimate, interval) in enumerate(zip(estimates, intervals, strict=True))
    ])  # fmt: skip


def test_summary_counts_increases_decreases_and_inconclusive(tmp_path: Path):
    # One interval per outcome, and one that could not be estimated
    comparisons = energy_comparisons(
        "temurin-17",
        [5.0, -5.0, 1.0, math.nan],
        [
            ConfidenceInterval(1.0, 9.0),
            ConfidenceInterval(-9.0, -1.0),
            ConfidenceInterval(-1.0, 3.0),
            ConfidenceInterval(math.nan, math.nan),
        ],
    )
    summary = generate_energy_summary_table(tmp_path / "summary.csv", comparisons)
    row = summary.row(0, named=True)
    assert row["Benchmarks"] == 3
    assert row["Median_Difference_Percent"] == 1.0
    assert (row["Increase"], row["Decrease"], row["Inconclusive"]) == (1, 1, 2)


def test_summary_weighs_benchmarks_equally(tmp_path: Path):
    # Pooled, the benchmark under three JDKs would place the median at 10
    comparisons = pl.concat([
        energy_comparisons("temurin-17", [10.0, 0.0, 1.0]),
        energy_comparisons("temurin-21", [10.0]),
        energy_comparisons("temurin-25", [10.0]),
    ])  # fmt: skip
    summary = generate_energy_summary_table(tmp_path / "summary.csv", comparisons)
    assert summary.row(0, named=True)["Median_Difference_Percent"] == 1.0


def test_summary_interval_spans_order_statistics(tmp_path: Path):
    # Of 15 benchmarks, the 4th and 12th smallest bound the 95% interval
    comparisons = energy_comparisons("temurin-17", [float(n) for n in range(1, 16)])
    summary = generate_energy_summary_table(tmp_path / "summary.csv", comparisons)
    row = summary.row(0, named=True)
    assert row["Median_Difference_Percent_CI_Low"] == 4.0
    assert row["Median_Difference_Percent_CI_High"] == 12.0


def test_summary_interval_of_few_benchmarks_is_undefined(tmp_path: Path):
    comparisons = energy_comparisons("temurin-17", [float(n) for n in range(1, 6)])
    summary = generate_energy_summary_table(tmp_path / "summary.csv", comparisons)
    row = summary.row(0, named=True)
    assert math.isnan(row["Median_Difference_Percent_CI_Low"])
    assert math.isnan(row["Median_Difference_Percent_CI_High"])


def test_summary_ends_with_every_scenario(tmp_path: Path):
    cold = energy_comparisons("temurin-17", [5.0, 7.0])
    warm = cold.with_columns(pl.lit("warm").alias("Scenario"))
    comparisons = pl.concat([warm, cold])
    summary = generate_energy_summary_table(tmp_path / "summary.csv", comparisons)
    expected = [("Cold", 2, 2), ("Warm", 2, 2), ("All", 2, 4)]
    assert summary.select("Scenario", "Benchmarks", "Increase").rows() == expected


def test_summary_by_jdk_keeps_only_benchmarks_of_every_jdk(tmp_path: Path):
    comparisons = pl.concat([
        energy_comparisons("temurin-17", [5.0, 7.0, 9.0]),
        energy_comparisons("temurin-21", [5.0, 7.0]),
    ])  # fmt: skip
    summary = generate_energy_summary_by_jdk_table(
        tmp_path / "summary.csv", comparisons, {}
    )
    # Two benchmarks, one row per JDK, in the cold and in the combined scenario
    assert summary.get_column("Benchmarks").to_list() == [2] * 4


def test_summary_by_jdk_leaves_out_benchmarks_without_an_estimate(tmp_path: Path):
    comparisons = pl.concat([
        energy_comparisons("temurin-17", [5.0, 7.0, 9.0]),
        energy_comparisons("temurin-21", [5.0, 7.0, math.nan]),
    ])  # fmt: skip
    summary = generate_energy_summary_by_jdk_table(
        tmp_path / "summary.csv", comparisons, {}
    )
    # Two benchmarks, one row per JDK, in the cold and in the combined scenario
    assert summary.get_column("Benchmarks").to_list() == [2] * 4


def test_summary_by_jdk_orders_jdks_by_version(tmp_path: Path):
    comparisons = pl.concat([
        energy_comparisons("temurin-11", [5.0]),
        energy_comparisons("temurin-8", [5.0]),
    ])  # fmt: skip
    jdk_versions = {"temurin-8": (8, 0, 0), "temurin-11": (11, 0, 0)}
    summary = generate_energy_summary_by_jdk_table(
        tmp_path / "summary.csv", comparisons, jdk_versions
    )
    cold = summary.filter(pl.col("Scenario") == "Cold")
    assert cold.get_column("JDK").to_list() == ["temurin-8", "temurin-11"]


def test_summary_by_jdk_ends_with_every_scenario(tmp_path: Path):
    cold = pl.concat([
        energy_comparisons("temurin-17", [5.0, 7.0]),
        energy_comparisons("temurin-21", [5.0, 7.0]),
    ])  # fmt: skip
    warm = cold.with_columns(pl.lit("warm").alias("Scenario"))
    comparisons = pl.concat([warm, cold])
    summary = generate_energy_summary_by_jdk_table(
        tmp_path / "summary.csv", comparisons, {}
    )
    expected = [
        ("Cold", "temurin-17", 2),
        ("Cold", "temurin-21", 2),
        ("Warm", "temurin-17", 2),
        ("Warm", "temurin-21", 2),
        ("All", "temurin-17", 4),
        ("All", "temurin-21", 4),
    ]
    assert summary.select("Scenario", "JDK", "Increase").rows() == expected
