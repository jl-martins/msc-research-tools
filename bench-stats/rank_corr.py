"""
Total Spearman rank correlation of clustered data, with a standard error and confidence interval.

Observations of the same cluster (here, the same benchmark program observed under several JDKs
and scenarios) are not independent, so the pooled Spearman coefficient overstates how much
information the data holds. This module implements the total (marginal) Spearman correlation
for clustered data:

* The estimate is the Pearson correlation of the midrank-adjusted empirical distribution
  functions of both variables, each observation weighted so that every cluster counts equally
  (Shih and Fay, 2017, Section 3.3). It equals the `Total` estimate of `rankCorrCluster` in the
  R package `rankCorr` (Tu, Li and Shepherd, 2025), which the tests check against.
* The standard error is the first-order asymptotic variance of Shih and Fay (2017, Theorem 1).
  It uses the residual kernel of their Eq. 10, averaged within clusters (Lemma 2), with the
  cluster-weighted empirical distribution plugged in. The denominator of the coefficient is
  treated as known, the variant Shih and Fay recommend for Spearman's coefficient (Section 4).
  `variance="rankcorr"` instead reproduces the standard error of `rankCorr` 1.0.3, which
  evaluates two integrals of that kernel with unweighted averages. The two coincide when the
  clusters have equal size.
* The interval is a Wald interval on the Fisher scale, or on the raw scale, with a normal
  reference, exactly as in `rankCorr`.

References:

* Shih, J. H. and Fay, M. P. (2017). Pearson's chi-square test and rank correlation inferences
  for clustered data. Biometrics 73(3), 822-834.
* Tu, S., Li, C. and Shepherd, B. E. (2025). Between- and within-cluster Spearman rank
  correlations. Statistics in Medicine.

Only the total correlation is implemented, not the between- and within-cluster correlations.
The pairwise comparisons take memory quadratic in the number of observations, which suits
thousands of observations but not millions.
"""

import logging
import math
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, rankdata

logger = logging.getLogger(__name__)

# How observations are weighted: every cluster equally, or every observation equally
type Weights = Literal["clusters", "obs"]

# Which variance formula is used for the standard error
type Variance = Literal["shih-fay", "rankcorr"]

# Fewer clusters leave no information about the variation between clusters
MIN_CLUSTERS = 3

# Significant digits after which R's `factor` still tells two numbers apart
TIE_SIGNIFICANT_DIGITS = 15


class ClusteredCorrelation(NamedTuple):
    """A clustered correlation with its standard error and confidence interval"""

    estimate: float
    se: float
    low: float
    high: float
    n_clusters: int
    n_obs: int


def dense_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Rank `values` from 1 without gaps, tying numbers equal to 15 significant digits"""
    # `rankCorr` ranks with `as.numeric(factor(x))`, which compares numbers by their text
    # form of 15 significant digits. Rounding the same way reproduces its ties exactly.
    rounded = [float(f"{value:.{TIE_SIGNIFICANT_DIGITS}g}") for value in values]
    return rankdata(rounded, method="dense").astype(np.intp)


def midrank_step(ranks: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
    """Return K[i, k] = {I[r_k <= r_i] + I[r_k < r_i]} / 2, the midrank step function of Shih and Fay"""
    above, below = ranks[:, None], ranks[None, :]  # r_i and r_k
    # Since I[r_k <= r_i] = I[r_k < r_i] + I[r_k = r_i], this is I[r_k < r_i] + I[r_k = r_i] / 2,
    # the form of `rankCorr:::emp_CDF_xy`. NumPy adds booleans as a logical 'or', so the two
    # indicators of the definition cannot be added directly.
    return (below < above) + 0.5 * (below == above)


def cluster_weights(
    codes: npt.NDArray[np.intp],
    n_clusters: int,
    weights: Weights,
) -> npt.NDArray[np.float64]:
    """Weight each observation by 1 / N, or by 1 / (clusters x size of its cluster)"""
    if weights == "obs":
        return np.full(len(codes), 1 / len(codes))
    else:
        return 1 / (n_clusters * np.bincount(codes)[codes])


def spearman_influence(
    step_x: npt.NDArray[np.float64],
    step_y: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return three times the residual kernel of Shih and Fay (2017, Eq. 10) at each observation"""
    # Expanding Eq. 10 with the empirical distribution F_C gives
    #   F_X F_Y - 3 A + E[K(X, x) F_Y(Y)] + E[F_X(X) K(Y, y)],  A = E[F_X F_Y],
    # which is the influence function of the covariance of the midrank-adjusted distributions.
    cdf_x, cdf_y = step_x @ w, step_y @ w
    return (
        cdf_x * cdf_y
        - 3 * (w @ (cdf_x * cdf_y))
        + step_x.T @ (w * cdf_y)
        + step_y.T @ (w * cdf_x)
    )


def fisher_wald_interval(
    estimate: float,
    variance: float,
    confidence: float,
    fisher: bool,
) -> tuple[float, float]:
    """Return the Wald interval of `estimate` on its own scale or, if `fisher`, the Fisher scale"""
    z = norm.isf((1 - confidence) / 2)
    sd = math.sqrt(variance)
    if not fisher:
        return estimate - z * sd, estimate + z * sd

    if abs(estimate) >= 1:
        return math.nan, math.nan

    scale = 2 / (1 - estimate**2)  # derivative of 2 atanh
    centre = 2 * math.atanh(estimate)
    return (
        math.tanh((centre - z * sd * scale) / 2),
        math.tanh((centre + z * sd * scale) / 2),
    )


def shih_fay_variance(
    step_x: npt.NDArray[np.float64],
    step_y: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    codes: npt.NDArray[np.intp],
    n_clusters: int,
    scale: float,
) -> float:
    """Return the first-order variance of the correlation of Shih and Fay (2017, Theorem 1), with the denominator treated as known"""
    cluster_totals = np.bincount(
        codes,
        weights=w * spearman_influence(step_x, step_y, w),
        minlength=n_clusters,
    )

    return float(np.sum(cluster_totals**2)) * scale


def rankcorr_variance(
    step_x: npt.NDArray[np.float64],
    step_y: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    codes: npt.NDArray[np.intp],
    n_clusters: int,
    moments: tuple[float, float, float, float],
) -> float:
    """Return the variance of the correlation as computed by `rankCorr` 1.0.3 (`cor_rt`)"""
    n = len(w)
    cdf_x, cdf_y = step_x @ w, step_y @ w
    rho, cov, var_x, var_y = moments
    # `rankCorr` averages the two integrals of Eq. 10 over all observations alike, not by weight
    mean_x, mean_y = step_x.mean(axis=0), step_y.mean(axis=0)
    integrals = step_x @ (w * mean_y) + step_y @ (w * mean_x)
    kernel = (
        (2 * cdf_x * cdf_y + 2 * integrals + 2 - 2 * cdf_x - 2 * cdf_y) / 6 - 0.25 - cov
    )
    kernel_x = (cdf_x**2 + 2 * (step_x.T @ cdf_x) / n) / 3 - 0.25 - var_x
    kernel_y = (cdf_y**2 + 2 * (step_y.T @ cdf_y) / n) / 3 - 0.25 - var_y

    sizes = np.bincount(codes, minlength=n_clusters)
    cluster_means = np.stack(
        [
            np.bincount(codes, weights=values, minlength=n_clusters) / sizes
            for values in (kernel, kernel_x, kernel_y)
        ]
    )
    covariance = 9 * cluster_means @ cluster_means.T / n_clusters**2
    gradient = np.array([rho / cov, -rho / (2 * var_x), -rho / (2 * var_y)])
    return float(gradient @ covariance @ gradient)


def total_spearman(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    cluster: npt.ArrayLike,
    *,
    weights: Weights = "clusters",
    confidence: float = 0.95,
    fisher: bool = True,
    variance: Variance = "shih-fay",
) -> ClusteredCorrelation:
    """Correlate `x` and `y` by Spearman's rank coefficient, allowing for the clusters of observations"""
    if variance == "shih-fay" and weights != "clusters":
        raise ValueError("The Shih-Fay variance is defined for weights='clusters' only")

    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    labels = np.asarray(cluster, dtype=object)
    if not len(x_values) == len(y_values) == len(labels):
        raise ValueError("x, y and cluster must have the same length")

    # Like `rankCorr`, drop the observations that miss a value
    missing = np.isnan(x_values) | np.isnan(y_values)
    missing |= np.array(
        [label is None or label != label for label in labels], dtype=bool
    )

    if missing.any():
        logger.warning(
            "Dropped %d of %d observations with missing values",
            missing.sum(),
            len(missing),
        )
        x_values, y_values, labels = (
            x_values[~missing],
            y_values[~missing],
            labels[~missing],
        )

    # Integer, float, and string labels of the same clusters must group alike
    clusters, codes = np.unique(labels.astype(str), return_inverse=True)
    n_clusters, n_obs = len(clusters), len(x_values)
    if n_clusters < MIN_CLUSTERS:
        raise ValueError(
            f"At least {MIN_CLUSTERS} clusters are needed, got {n_clusters}"
        )

    ranks_x, ranks_y = dense_ranks(x_values), dense_ranks(y_values)
    if np.ptp(ranks_x) == 0 or np.ptp(ranks_y) == 0:  # a constant has no correlation
        return ClusteredCorrelation(
            math.nan, math.nan, math.nan, math.nan, n_clusters, n_obs
        )

    w = cluster_weights(codes, n_clusters, weights)
    step_x, step_y = midrank_step(ranks_x), midrank_step(ranks_y)
    cdf_x, cdf_y = step_x @ w, step_y @ w
    cov = float(w @ ((cdf_x - w @ cdf_x) * (cdf_y - w @ cdf_y)))
    var_x = float(w @ (cdf_x - w @ cdf_x) ** 2)
    var_y = float(w @ (cdf_y - w @ cdf_y) ** 2)
    rho = cov / math.sqrt(var_x * var_y)

    if variance == "shih-fay":
        var_rho = shih_fay_variance(
            step_x, step_y, w, codes, n_clusters, 1 / (var_x * var_y)
        )
    else:
        var_rho = rankcorr_variance(
            step_x, step_y, w, codes, n_clusters, (rho, cov, var_x, var_y)
        )

    low, high = fisher_wald_interval(rho, var_rho, confidence, fisher)
    return ClusteredCorrelation(rho, math.sqrt(var_rho), low, high, n_clusters, n_obs)
