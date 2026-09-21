"""Tests of the comparison of bench-stats against known answers"""

import numpy as np
import pytest

from bench_stats import SampleKey, compare_metric

KEY = SampleKey("cold", "temurin-21", "streams", "fop")
METRIC = "PackageAndDRAM"


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
