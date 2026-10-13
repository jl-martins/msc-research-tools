"""Loop sampling tests"""

import polars as pl
import pytest
from conftest import DATA_DIR

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import LOOP_SCHEMA
from dacapo_hot_loops.config import SampleConfig
from dacapo_hot_loops.sample import allocate, sample_hot_loops
from dacapo_hot_loops.worksheet import HOT_LOOPS_NAME

HOT_LOOPS = pl.read_csv(DATA_DIR / HOT_LOOPS_NAME, schema=LOOP_SCHEMA)
SHARED_LOOP = "shared/Scan.java:5-9"
STRATUM_SIZES = {"basic_for": 5, "while": 3, "enhanced_for": 2}
POPULATION_SIZE = sum(STRATUM_SIZES.values())


def test_allocation_is_proportional():
    assert allocate(STRATUM_SIZES, 5) == {"basic_for": 3, "while": 1, "enhanced_for": 1}


@pytest.mark.parametrize("sample_size", [1, POPULATION_SIZE // 2, POPULATION_SIZE])
def test_allocation_sums_to_sample_size(sample_size):
    assert sum(allocate(STRATUM_SIZES, sample_size).values()) == sample_size


def test_full_sample_takes_every_loop():
    assert allocate(STRATUM_SIZES, POPULATION_SIZE) == STRATUM_SIZES


def test_leftover_tie_goes_to_first_name():
    assert allocate({"while": 1, "basic_for": 1}, 1) == {"while": 0, "basic_for": 1}


def test_without_sampling_every_row_is_kept():
    assert sample_hot_loops(HOT_LOOPS, None).equals(HOT_LOOPS)


def test_same_seed_draws_same_loops():
    sample_config = SampleConfig(size=2, seed=7)
    first = sample_hot_loops(HOT_LOOPS, sample_config)
    second = sample_hot_loops(HOT_LOOPS, sample_config)
    assert first.equals(second)


def test_shared_loop_keeps_every_benchmark_row():
    sampled = sample_hot_loops(HOT_LOOPS, SampleConfig(size=2, seed=7))
    shared = sampled.filter(pl.col("loop_id") == SHARED_LOOP)
    assert shared["benchmark"].to_list() == ["luindex", "lusearch"]


def test_sample_of_whole_population_keeps_every_row():
    sampled = sample_hot_loops(HOT_LOOPS, SampleConfig(size=3, seed=7))
    assert sampled.sort("benchmark", "loop_id").equals(
        HOT_LOOPS.sort("benchmark", "loop_id")
    )


def test_sample_larger_than_population_fails():
    with pytest.raises(HotLoopsError, match="exceeds"):
        sample_hot_loops(HOT_LOOPS, SampleConfig(size=4, seed=7))
