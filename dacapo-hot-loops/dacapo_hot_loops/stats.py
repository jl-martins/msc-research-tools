"""Statistics module"""

from typing import NamedTuple

import numpy as np
from scipy.stats import hypergeom


class IntConfidenceInterval(NamedTuple):
    """Inclusive integer confidence limits"""

    low: int
    high: int


def hypergeom_count_ci(
    successes: int,
    sample_size: int,
    population_size: int,
    confidence_level: float,
) -> IntConfidenceInterval:
    """
    Equal-tailed exact confidence limits for the number of successes in a population

    The limits invert two one-sided hypergeometric tests, each at level
    (1 - confidence_level) / 2, for a simple random sample drawn without replacement
    (Buonaccorsi 1987, §3, equivalent to Konijn 1973). Coverage is at least the
    confidence level for every possible number of successes in the population.
    """
    tail = (1 - confidence_level) / 2
    counts = np.arange(population_size + 1)
    # SciPy's parameters are the population size, the successes in the population
    # and the sample size
    at_least = hypergeom.sf(successes - 1, population_size, counts, sample_size)
    at_most = hypergeom.cdf(successes, population_size, counts, sample_size)

    return IntConfidenceInterval(
        low=int(counts[at_least > tail].min()),
        high=int(counts[at_most > tail].max()),
    )
