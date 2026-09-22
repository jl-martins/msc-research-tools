"""Newcombe score interval tests"""

# The symmetry tests check the properties that SingleCaseES (version 0.7.4) tests for
# NAP and Tau, in test-directional-arguments.R and test-trunc-constant.R.
# NAP(A, B) maps onto `newcombe_score_ci(B, A)`, and Tau is Cliff's delta.

import re

import numpy as np
import numpy.typing as npt
import pytest

from effect_sizes import newcombe_score_ci

# Seeded normal samples, rounded to produce ties, with shifts that range from none
# to full separation
RNG = np.random.default_rng(20260924)
SAMPLE_PAIRS = [
    pytest.param(
        RNG.normal(0, 1, m).round(1),
        RNG.normal(shift, 1, n).round(1),
        id=f"shift-{shift}-{m}x{n}",
    )
    for shift in (0, 1, 3, 10)
    for m, n in ((5, 8), (30, 20))
]

# A run of equal letters in Newcombe's notation, with an optional count
NEWCOMBE_RUN = re.compile(r"(\d*)([XYT])")


def newcombe_samples(
    notation: str,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Return the X and Y samples of a data set in Newcombe's notation, such as "3X,5Y,2X".

    Each run of letters takes a value above the previous run. A run of `T`s holds X and Y
    observations in equal numbers.
    """
    x, y = [], []
    for value, (count, letter) in enumerate(NEWCOMBE_RUN.findall(notation)):
        size = int(count or 1)
        if letter == "X":
            x += [value] * size
        elif letter == "Y":
            y += [value] * size
        else:
            x += [value] * (size // 2)
            y += [value] * (size // 2)

    return np.array(x), np.array(y)


# Method 5 limits for `theta` in Tables I and II of Newcombe (2006, part 2). Table II
# gives the sizes and estimate of data sets (l) and (m), which these samples match.
RATINGS = np.arange(1, 6)
ILLUSTRATIVE_DATA_SETS = [
    pytest.param(
        np.repeat(RATINGS, [33, 6, 6, 11, 2]),
        np.repeat(RATINGS, [3, 2, 2, 11, 33]),
        0.8117,
        0.9400,
        id="I(a)",
    ),
    pytest.param(*newcombe_samples("XXXXXYYYYY"), 0.6427, 1.0000, id="I(b)"),
    pytest.param(*newcombe_samples("XXXXYXYYYY"), 0.5921, 0.9970, id="I(c)"),
    pytest.param(*newcombe_samples("XYYXY"), 0.2126, 0.9342, id="I(d)"),
    pytest.param(*newcombe_samples("YXXYX"), 0.0658, 0.7874, id="I(e)"),
    pytest.param(*newcombe_samples("YXYYX"), 0.0658, 0.7874, id="I(f)"),
    pytest.param(*newcombe_samples("XYXXY"), 0.2126, 0.9342, id="I(g)"),
    pytest.param(*newcombe_samples("18X,100Y,2X"), 0.7913, 0.9529, id="II(a)"),
    pytest.param(*newcombe_samples("10Y,20X,90Y"), 0.7913, 0.9529, id="II(b)"),
    pytest.param(*newcombe_samples("90X,20Y,10X"), 0.7913, 0.9529, id="II(c)"),
    pytest.param(*newcombe_samples("2Y,100X,18Y"), 0.7913, 0.9529, id="II(d)"),
    pytest.param(*newcombe_samples("10X,100Y,10X"), 0.3669, 0.6331, id="II(e)"),
    pytest.param(*newcombe_samples("50Y,20X,50Y"), 0.3669, 0.6331, id="II(f)"),
    pytest.param(*newcombe_samples("50X,20Y,50X"), 0.3669, 0.6331, id="II(g)"),
    pytest.param(*newcombe_samples("10Y,100X,10Y"), 0.3669, 0.6331, id="II(h)"),
    pytest.param(*newcombe_samples("3X,5Y,2X"), 0.2718, 0.8537, id="II(i)"),
    pytest.param(*newcombe_samples("3X,20Y,2X"), 0.3315, 0.8148, id="II(j)"),
    pytest.param(*newcombe_samples("1Y,12X,2Y,2T,2Y"), 0.5149, 0.9330, id="II(k)"),
    pytest.param(*newcombe_samples("3Y,8X,2T,92Y"), 0.8249, 0.9943, id="II(l)"),
    pytest.param(*newcombe_samples("43Y,3X,1Y,15X,52Y"), 0.4009, 0.6779, id="II(m)"),
]


@pytest.mark.parametrize(("baseline", "treatment"), SAMPLE_PAIRS)
def test_negation_mirrors_interval(baseline, treatment):
    increase = newcombe_score_ci(treatment, baseline)
    decrease = newcombe_score_ci(-treatment, -baseline)
    assert decrease == pytest.approx((-increase.high, -increase.low))


@pytest.mark.parametrize(("baseline", "treatment"), SAMPLE_PAIRS)
def test_negation_and_swap_keeps_interval(baseline, treatment):
    decrease = newcombe_score_ci(-treatment, -baseline)
    swapped = newcombe_score_ci(baseline, treatment)
    assert swapped == pytest.approx(decrease)


@pytest.mark.parametrize(("x", "y", "low", "high"), ILLUSTRATIVE_DATA_SETS)
def test_matches_newcombe_illustrative_data_sets(x, y, low, high):
    ci = newcombe_score_ci(y, x)
    assert ci == pytest.approx((2 * low - 1, 2 * high - 1), abs=1e-4)
