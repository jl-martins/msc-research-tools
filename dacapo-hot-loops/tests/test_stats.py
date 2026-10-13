"""Statistics tests"""

import pytest
from scipy.stats import hypergeom

from dacapo_hot_loops.stats import IntConfidenceInterval, hypergeom_count_ci


# "K or T method" intervals of Table 2 in Buonaccorsi (1987), for N = 10, n = 4, α = .10
@pytest.mark.parametrize(
    ("successes", "low", "high"),
    [(0, 0, 4), (1, 1, 6), (2, 2, 8), (3, 4, 9), (4, 6, 10)],
)
def test_buonaccorsi_example(successes, low, high):
    interval = hypergeom_count_ci(successes, 4, 10, 0.90)
    assert interval == IntConfidenceInterval(low=low, high=high)


def test_each_tail_is_at_most_half_alpha_beyond_the_limits():
    low, high = hypergeom_count_ci(30, 90, 521, 0.95)
    assert hypergeom.sf(29, 521, low - 1, 90) <= 0.025 < hypergeom.sf(29, 521, low, 90)
    assert (
        hypergeom.cdf(30, 521, high + 1, 90) <= 0.025 < hypergeom.cdf(30, 521, high, 90)
    )


def test_census_gives_the_observed_count():
    assert hypergeom_count_ci(7, 10, 10, 0.95) == IntConfidenceInterval(low=7, high=7)
