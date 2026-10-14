"""Classification worksheet module"""

import logging
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

import polars as pl

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import ANALYSIS_DIR_NAME, LOOP_SCHEMA
from dacapo_hot_loops.config import SampleConfig
from dacapo_hot_loops.sample import sample_hot_loops

logger = logging.getLogger(__name__)

HOT_LOOPS_NAME = "hot_loops.csv"
WORKSHEET_NAME = "classification.csv"
# Separator for the `benchmarks` and `failed_checks` lists
LIST_SEPARATOR = ";"
# Columns that identify a loop, refreshed from `hot_loops.csv` on every run
LOOP_COLUMNS = (
    "loop_id",
    "file",
    "start_line",
    "end_line",
    "loop_type",
    "enclosing_method_key",
)
# Columns that only the researcher writes, kept unchanged on every run
CLASSIFICATION_COLUMNS = ("verdict", "failed_checks", "notes")


class LoopStatus(StrEnum):
    """Whether a worksheet loop is in the current `hot_loops.csv`"""

    HOT = "hot"
    STALE = "stale"


class Verdict(StrEnum):
    """Whether a hot loop can be rewritten as a sequential stream"""

    CONVERTIBLE = "convertible"
    NOT = "not"


class Check(StrEnum):
    """Check that a loop can fail, in the order from the classification protocol"""

    SOURCE = "source"
    EXCEPTIONS = "exceptions"
    LOCAL_STATE = "local_state"
    CONTROL_FLOW = "control_flow"
    ITERATION_ORDER = "iteration_order"


WORKSHEET_SCHEMA: Mapping[str, pl.DataType] = {
    "loop_id": pl.String(),
    "file": pl.String(),
    "start_line": pl.Int64(),
    "end_line": pl.Int64(),
    "loop_type": pl.String(),
    "enclosing_method_key": pl.String(),
    "benchmarks": pl.String(),
    "status": pl.String(),
    "verdict": pl.String(),
    "failed_checks": pl.String(),
    "notes": pl.String(),
}


def read_analysis_table(
    out_dir: Path,
    name: str,
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    path = out_dir / ANALYSIS_DIR_NAME / name
    try:
        return pl.read_csv(path, schema=schema)
    except (OSError, pl.exceptions.PolarsError) as e:
        raise HotLoopsError(f"Could not read '{path}': {e}") from e


def read_worksheet(path: Path) -> pl.DataFrame:
    """Read a worksheet and check its header against the schema"""
    try:
        # Overrides match columns by name, so a reordered header stays visible
        worksheet = pl.read_csv(path, schema_overrides=WORKSHEET_SCHEMA)
    except (OSError, pl.exceptions.PolarsError) as e:
        raise HotLoopsError(f"Could not read '{path}': {e}") from e

    if worksheet.columns != list(WORKSHEET_SCHEMA):
        columns = ", ".join(WORKSHEET_SCHEMA)
        msg = f"'{path}' must have the following columns: {columns}"
        raise HotLoopsError(msg)

    duplicated = worksheet.filter(pl.col("loop_id").is_duplicated())["loop_id"]
    if duplicated.len() == 0:
        return worksheet
    else:
        msg = f"Duplicate loops in '{path}': {', '.join(duplicated.unique().sort())}"
        raise HotLoopsError(msg)


def merge_worksheet(hot_loops: pl.DataFrame, previous: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct hot loop, plus the stale rows of `previous`"""
    current = hot_loops.group_by(*LOOP_COLUMNS).agg(
        pl.col("benchmark").sort().str.join(LIST_SEPARATOR).alias("benchmarks")
    )
    conflicting = current.filter(pl.col("loop_id").is_duplicated())["loop_id"]
    if conflicting.len() != 0:
        raise HotLoopsError(
            f"Loops with conflicting columns in {HOT_LOOPS_NAME}: "
            f"{', '.join(conflicting.unique().sort())}"
        )

    classifications = previous.select("loop_id", *CLASSIFICATION_COLUMNS)
    hot = current.with_columns(status=pl.lit(LoopStatus.HOT.value)).join(
        classifications,
        on="loop_id",
        how="left",
    )
    # A stale row keeps the columns it had when its loop was last hot
    stale = previous.join(current, on="loop_id", how="anti").with_columns(
        status=pl.lit(LoopStatus.STALE.value)
    )

    return pl.concat(
        [hot.select(*WORKSHEET_SCHEMA), stale.select(*WORKSHEET_SCHEMA)]
    ).sort("loop_id")


def write_worksheet(out_dir: Path, sample_config: SampleConfig | None) -> None:
    """Write `classification.csv`, merging it with any previous version"""
    hot_loops = read_analysis_table(out_dir, HOT_LOOPS_NAME, LOOP_SCHEMA)
    path = out_dir / WORKSHEET_NAME
    if path.exists():
        previous = read_worksheet(path)
    else:
        previous = pl.DataFrame(schema=WORKSHEET_SCHEMA)

    sampled_loops = sample_hot_loops(hot_loops, sample_config)
    worksheet = merge_worksheet(sampled_loops, previous)
    worksheet.write_csv(path)

    is_hot = pl.col("status") == LoopStatus.HOT
    new_count = worksheet.filter(is_hot).join(previous, on="loop_id", how="anti").height
    stale_count = worksheet.filter(~is_hot).height
    logger.info(
        "Wrote '%s': %d hot loops (%d new), %d stale",
        path,
        worksheet.height - stale_count,
        new_count,
        stale_count,
    )
