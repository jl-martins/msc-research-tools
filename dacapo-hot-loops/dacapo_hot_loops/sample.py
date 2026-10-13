"""Loop sampling module"""

import logging
import random
from collections.abc import Mapping

import polars as pl

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.config import SampleConfig

logger = logging.getLogger(__name__)


def allocate(stratum_sizes: Mapping[str, int], sample_size: int) -> dict[str, int]:
    """
    Split a sample size across strata in proportion to their sizes
    (largest-remainder method)

    Ties for the leftover loops go to the stratum that comes first by name.
    """
    population_size = sum(stratum_sizes.values())
    if population_size == 0:
        return dict.fromkeys(stratum_sizes, 0)

    quotas = {
        name: divmod(sample_size * size, population_size)
        for name, size in stratum_sizes.items()
    }
    allocation = {name: whole for name, (whole, _) in quotas.items()}

    leftover = sample_size - sum(allocation.values())
    by_remainder = sorted(quotas, key=lambda name: (-quotas[name][1], name))
    for name in by_remainder[:leftover]:
        allocation[name] += 1

    return allocation


def sample_hot_loops(
    hot_loops: pl.DataFrame,
    sample_config: SampleConfig | None,
) -> pl.DataFrame:
    """The hot-loop rows drawn into the sample, or all rows if sample_config is None"""
    if sample_config is None:
        return hot_loops

    # Types and ids are sorted so the draw depends only on the seed
    ids_by_type: dict[str, list[str]] = dict(
        hot_loops.select("loop_type", "loop_id")
        .unique()
        .group_by("loop_type")
        .agg(pl.col("loop_id").sort())
        .sort("loop_type")
        .iter_rows()
    )
    stratum_sizes = {loop_type: len(ids) for loop_type, ids in ids_by_type.items()}
    total = sum(stratum_sizes.values())

    sample_size = sample_config.size
    if sample_size > total:
        raise HotLoopsError(
            f"Sample size {sample_size} exceeds the {total} distinct hot loops"
        )

    allocation = allocate(stratum_sizes, sample_size)
    details = ", ".join(
        f"{loop_type} {allocation[loop_type]}/{size}"
        for loop_type, size in stratum_sizes.items()
    )
    logger.info("Sampling %d of %d hot loops: %s", sample_size, total, details)

    rng = random.Random(sample_config.seed)
    drawn = [
        loop_id
        for loop_type, ids in ids_by_type.items()
        for loop_id in rng.sample(ids, allocation[loop_type])
    ]

    return hot_loops.filter(pl.col("loop_id").is_in(drawn))
