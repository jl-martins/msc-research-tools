"""Tests of the clustered Spearman correlation against rankCorr and known answers"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from rank_corr import (
    MIN_CLUSTERS,
    cluster_weights,
    dense_ranks,
    midrank_step,
    spearman_influence,
    total_spearman,
)

# The `Total` outputs of `rankCorrCluster(x, y, cluster, weights, fisher, conf.int = 0.95)` of
# rankCorr 1.0.3 (R 4.6.1) for 18 datasets: the four subsets of the study, and synthetic data
# with ties, singleton clusters, unequal cluster sizes, heavy tails, and negative correlations.
GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "rankcorr_golden.json").read_text()
)

GOLDEN_SETTINGS = [
    pytest.param(
        dataset,
        reference,
        id=f"{dataset['name']}-{reference['weights']}-fisher={reference['fisher']}",
    )
    for dataset in GOLDEN
    for reference in dataset["rankCorr"]
]

# The study subsets are the first four datasets, in the order of `SHIH_FAY_STUDY_VALUES`
STUDY_SUBSETS = GOLDEN[:4]

# The estimate and Shih-Fay standard error of this module for the study subsets. These are
# its own output, kept to notice a change, and not an external reference.
SHIH_FAY_STUDY_VALUES = [
    (0.9477453887, 0.03358613),
    (0.9788822080, 0.0126593),
    (0.9113372907, 0.05949138),
    (0.9362347192, 0.03960792),
]


def clustered_data(
    seed: int,
    sizes: list[int],
    noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw x and y that share a random effect within each cluster of the given sizes"""
    rng = np.random.default_rng(seed)
    effects = rng.normal(size=len(sizes))
    cluster = np.repeat(np.arange(len(sizes)), sizes)
    x = effects[cluster] + noise * rng.normal(size=len(cluster))
    y = effects[cluster] + noise * rng.normal(size=len(cluster))
    return x, y, cluster


def step(a: float, b: float) -> float:
    """Return K(a, b) = {I[b <= a] + I[b < a]} / 2 as Shih and Fay define it, for one pair"""
    # The indicators become numbers first: NumPy adds booleans as a logical or
    return (float(b <= a) + float(b < a)) / 2


@pytest.mark.parametrize(("dataset", "reference"), GOLDEN_SETTINGS)
def test_rankcorr_variance_reproduces_rankcorr(dataset, reference):
    result = total_spearman(
        dataset["x"],
        dataset["y"],
        dataset["cluster"],
        weights=reference["weights"],
        fisher=reference["fisher"],
        variance="rankcorr",
    )

    assert result.estimate == pytest.approx(reference["Estimate"], rel=1e-9, abs=1e-12)
    assert result.se == pytest.approx(reference["SE"], rel=1e-9, abs=1e-12)
    assert result.low == pytest.approx(reference["Lower"], rel=1e-9, abs=1e-12)
    assert result.high == pytest.approx(reference["Upper"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize(
    ("dataset", "expected"),
    list(zip(STUDY_SUBSETS, SHIH_FAY_STUDY_VALUES)),
    ids=[dataset["name"] for dataset in STUDY_SUBSETS],
)
def test_shih_fay_variance_keeps_its_values_on_the_study_data(dataset, expected):
    result = total_spearman(dataset["x"], dataset["y"], dataset["cluster"])

    estimate, se = expected
    assert result.estimate == pytest.approx(estimate, rel=1e-8)
    assert result.se == pytest.approx(se, rel=1e-6)


def test_both_variances_share_the_estimate():
    dataset = STUDY_SUBSETS[0]

    shih_fay = total_spearman(dataset["x"], dataset["y"], dataset["cluster"])
    rankcorr = total_spearman(
        dataset["x"], dataset["y"], dataset["cluster"], variance="rankcorr"
    )

    assert shih_fay.estimate == rankcorr.estimate
    assert shih_fay.se != rankcorr.se  # the study clusters have unequal sizes


def test_shih_fay_variance_matches_rankcorr_for_clusters_of_equal_size():
    # Unweighted and weighted averages of the kernel integrals coincide when clusters are equal
    x, y, cluster = clustered_data(11, [5] * 9)

    shih_fay = total_spearman(x, y, cluster, fisher=False)
    rankcorr = total_spearman(x, y, cluster, fisher=False, variance="rankcorr")

    assert shih_fay.estimate == pytest.approx(rankcorr.estimate, rel=1e-12)
    assert shih_fay.se == pytest.approx(rankcorr.se, rel=1e-9)


def test_shih_fay_variance_is_defined_for_cluster_weights_only():
    x, y, cluster = clustered_data(3, [4] * 6)

    with pytest.raises(ValueError, match="weights='clusters'"):
        total_spearman(x, y, cluster, weights="obs")


def test_influence_equals_three_times_the_kernel_of_equation_10_as_printed():
    # Data with ties, unequal cluster sizes, and one cluster of one observation
    x, y, cluster = clustered_data(3, [2, 5, 3, 1, 4, 3])
    x, y = np.round(x, 1), np.round(y, 1)
    ranks_x, ranks_y = dense_ranks(x), dense_ranks(y)
    w = cluster_weights(cluster, 6, "clusters")
    n = len(x)

    def cdf(a, b):  # F(a, b), the midrank-adjusted joint distribution
        return sum(w[m] * step(a, ranks_x[m]) * step(b, ranks_y[m]) for m in range(n))

    cdf_x = [
        sum(w[m] * step(ranks_x[i], ranks_x[m]) for m in range(n)) for i in range(n)
    ]
    cdf_y = [
        sum(w[m] * step(ranks_y[i], ranks_y[m]) for m in range(n)) for i in range(n)
    ]
    sigma = sum(w[i] * cdf_x[i] * cdf_y[i] for i in range(n)) - 0.25
    # h(x, y) = {1 - F_X - F_Y + F_X F_Y + int F(x, y) dF_X(x) + int F(x, y) dF_Y(y)} / 3 - 1/4 - sigma
    kernel = [
        (
            1 - cdf_x[i] - cdf_y[i] + cdf_x[i] * cdf_y[i]
            + sum(w[m] * cdf(ranks_x[m], ranks_y[i]) for m in range(n))
            + sum(w[m] * cdf(ranks_x[i], ranks_y[m]) for m in range(n))
        ) / 3 - 0.25 - sigma
        for i in range(n)
    ]  # fmt: skip

    influence = spearman_influence(midrank_step(ranks_x), midrank_step(ranks_y), w)

    assert influence == pytest.approx(3 * np.array(kernel), abs=1e-12)
    # The mean of an influence function is 0
    assert w @ influence == pytest.approx(0, abs=1e-12)


def test_midrank_step_counts_lower_ranks_fully_and_tied_ranks_by_half():
    steps = midrank_step(np.array([1, 2, 2, 3]))

    assert steps.tolist() == [
        [0.5, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.5, 0.0],
        [1.0, 0.5, 0.5, 0.0],
        [1.0, 1.0, 1.0, 0.5],
    ]  # fmt: skip


def test_dense_ranks_tie_numbers_that_are_equal_to_15_significant_digits():
    # 0.1 + 0.2 is 0.30000000000000004, which R tells apart from 0.3 no more than this does
    values = np.array([0.3, 0.1 + 0.2, 1.0, 1.0 + 1e-12, 0.05])

    assert dense_ranks(values).tolist() == [2, 2, 3, 4, 1]


def test_correlation_is_unchanged_by_increasing_transformations():
    x, y, cluster = clustered_data(5, [3, 6, 4, 2, 5, 4, 3])

    original = total_spearman(x, y, cluster)
    transformed = total_spearman(np.exp(x), y**3, cluster)

    assert transformed.estimate == pytest.approx(original.estimate, rel=1e-12)
    assert transformed.se == pytest.approx(original.se, rel=1e-12)


def test_clusters_of_one_observation_give_the_spearman_coefficient():
    rng = np.random.default_rng(5)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)

    result = total_spearman(x, y, np.arange(40), weights="obs", variance="rankcorr")

    assert result.estimate == pytest.approx(spearmanr(x, y).statistic, rel=1e-12)


def test_fewer_than_three_clusters_are_refused():
    x, y, cluster = clustered_data(1, [5, 5])

    with pytest.raises(ValueError, match=f"{MIN_CLUSTERS} clusters"):
        total_spearman(x, y, cluster)


def test_lengths_that_differ_are_refused():
    with pytest.raises(ValueError, match="same length"):
        total_spearman([1.0, 2.0, 3.0], [1.0, 2.0], [0, 1, 2])


def test_a_constant_variable_has_no_correlation():
    _, y, cluster = clustered_data(2, [4] * 5)

    result = total_spearman(np.ones(len(y)), y, cluster)

    assert math.isnan(result.estimate)
    assert math.isnan(result.se)
    assert (result.n_clusters, result.n_obs) == (5, 20)


def test_observations_with_missing_values_are_dropped(caplog):
    x, y, cluster = clustered_data(7, [4, 3, 5, 4, 3])
    expected = total_spearman(x[3:], y[3:], cluster[3:])
    x_with_gaps, y_with_gaps = x.tolist(), y.tolist()
    labels: list[object] = cluster.tolist()
    x_with_gaps[0], y_with_gaps[1], labels[2] = math.nan, None, None

    with caplog.at_level(logging.WARNING):
        result = total_spearman(x_with_gaps, y_with_gaps, labels)

    assert "Dropped 3 of 19 observations" in caplog.text
    assert result.estimate == pytest.approx(expected.estimate, rel=1e-12)
    assert result.n_obs == 16


def test_integer_and_float_labels_group_the_same_clusters():
    # `factor` writes a label with `as.character`, which gives 1 and 1.0 one level
    x, y, cluster = clustered_data(4, [3, 5, 2, 4])

    by_integer = total_spearman(x, y, cluster)
    by_float = total_spearman(x, y, cluster.astype(float))

    assert by_float == by_integer


def test_integer_and_string_labels_group_the_same_clusters():
    x, y, cluster = clustered_data(9, [3, 4, 5, 2, 4])

    by_integer = total_spearman(x, y, cluster)
    by_string = total_spearman(x, y, [f"program-{label}" for label in cluster])

    assert by_string == by_integer
