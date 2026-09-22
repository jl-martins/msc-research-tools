"""Tests of the comparison and the correlation of bench-stats against known answers"""

import logging
import math
import re

import numpy as np
import polars as pl
import pytest

from bench_stats import (
    SampleKey,
    compare_metric,
    format_scatter_statistics,
    generate_energy_time_correlation_table,
    parse_jdk_version,
)
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


def test_correlation_table_counts_the_rows_and_the_benchmarks_of_each_scenario():
    table = generate_energy_time_correlation_table(long_table(15))

    assert table.columns == CORRELATION_COLUMNS
    assert table.get_column("Scenario").to_list() == ["All", "Cold", "Warm"]
    assert table.get_column("N").to_list() == [60, 30, 30]
    assert table.get_column("Program_Count").to_list() == [15, 15, 15]


def test_correlation_table_reports_the_clustered_correlation_of_its_rows():
    long_df = long_table(15)

    row = generate_energy_time_correlation_table(long_df).row(0, named=True)

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


def test_correlation_table_leaves_a_subset_of_too_few_benchmarks_undefined(caplog):
    with caplog.at_level(logging.WARNING):
        row = generate_energy_time_correlation_table(long_table(2)).row(0, named=True)

    assert "Skipped the correlation of 8 rows of 2 benchmarks" in caplog.text
    assert math.isnan(row["Spearman_Rho"])
    assert (row["N"], row["Program_Count"]) == (8, 2)


def test_correlation_table_leaves_a_constant_difference_undefined(caplog):
    long_df = long_table(15).with_columns(pl.lit(1.0).alias("Energy_Diff_Percent"))

    with caplog.at_level(logging.WARNING):
        row = generate_energy_time_correlation_table(long_df).row(0, named=True)

    assert "time or energy is constant" in caplog.text
    assert math.isnan(row["Spearman_Rho"])
    assert row["Program_Count"] == 15


def test_scatter_statistics_show_the_correlation_with_its_interval_above_the_size():
    panel_df = scatter_panel(15)

    text = format_scatter_statistics(panel_df)

    expected = total_spearman(
        panel_df.get_column("Time_Diff_Percent"),
        panel_df.get_column("Energy_Diff_Percent"),
        panel_df.get_column("Benchmark"),
    )
    assert re.fullmatch(
        r"\$\\rho = -?\d\.\d\d\$ \[-?\d\.\d\d, -?\d\.\d\d\]\n\$n = 15\$", text
    )
    assert text.startswith(
        f"$\\rho = {expected.estimate:.2f}$ [{expected.low:.2f}, {expected.high:.2f}]"
    )


def test_scatter_statistics_show_only_the_size_when_the_correlation_has_no_interval():
    # Two benchmarks are too few to correlate. A perfect correlation has no interval
    # on the Fisher scale, so its estimate is not shown either, as in the heatmaps.
    perfect = scatter_panel(15).with_columns(
        pl.col("Time_Diff_Percent").alias("Energy_Diff_Percent")
    )

    assert format_scatter_statistics(scatter_panel(2)) == "$n = 2$"
    assert format_scatter_statistics(perfect) == "$n = 15$"


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
