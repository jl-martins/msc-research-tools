"""
Layout, reading and cleaning of the benchmark measurements, and the JDK versions they were taken on.
"""

import logging
import re
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import polars as pl

logger = logging.getLogger(__name__)


# Metrics
class Metric(StrEnum):
    """A measured quantity, named exactly as its column in `METRICS_SCHEMA`"""

    PACKAGE = "Package"
    CORE = "Core"
    UNCORE = "Uncore"
    DRAM = "DRAM"
    PACKAGE_AND_DRAM = "PackageAndDRAM"
    TIME = "Time"
    TEMPERATURE = "Temperature"
    MEMORY = "Memory"


# Schemas
METRICS_SCHEMA = {
    Metric.PACKAGE: pl.Float64,
    Metric.CORE: pl.Float64,
    Metric.UNCORE: pl.Float64,
    Metric.DRAM: pl.Float64,
    Metric.PACKAGE_AND_DRAM: pl.Float64,
    Metric.TIME: pl.Float64,
    Metric.TEMPERATURE: pl.Float64,
    Metric.MEMORY: pl.Float64,
}

IDENTITY_COLUMNS_SCHEMA = {
    "Language": pl.Utf8,
    "Program": pl.Utf8,
    "Powercap": pl.Utf8,
}

MEASUREMENTS_SCHEMA = IDENTITY_COLUMNS_SCHEMA | METRICS_SCHEMA

RAW_MEASUREMENTS_SCHEMA = {
    k: v for k, v in MEASUREMENTS_SCHEMA.items() if k != Metric.PACKAGE_AND_DRAM
}


# Columns
METRICS = list(METRICS_SCHEMA.keys())
IDENTITY_COLUMNS = list(IDENTITY_COLUMNS_SCHEMA.keys())
MEASUREMENTS_COLUMNS = list(MEASUREMENTS_SCHEMA.keys())


# Idle runs prefix
IDLE_RUN_PREFIX = "idle_"


# File and directory names
MEASUREMENTS_FILE_NAME = "measurements.csv"
JAVA_VERSION_FILE_NAME = "java_version.txt"
BASELINE_DIR_NAME = "loops"


# JDK versions

# The version string of a JDK, as reported by `java -version`, is the first
# quoted token of `JAVA_VERSION_FILE_NAME` (e.g. openjdk version "25.0.3").
JAVA_VERSION_STRING_PATTERN = re.compile(r'version\s+"([^"]+)"')

# `(major, minor, security)` of a legacy (pre-JEP 223) version number, whose
# form is "1.$MAJOR.$MINOR_$UPDATE" (e.g. "1.8.0_492"). $UPDATE maps onto
# $SECURITY, as in the version data that Adoptium publishes for the JDKs
# used here (e.g. "1.8.0_504-b01" is major 8, minor 0, security 504).
JAVA_LEGACY_VERSION_NUMBER_PATTERN = re.compile(r"^1\.(\d+)(?:\.(\d+))?(?:_(\d+))?")

# `(major, minor, security)` of `$VNUM`, the leading version number of a
# JEP 223 version string (e.g. "25.0.3" of "25.0.3+9-LTS"). `$VNUM` omits
# trailing zero elements, so it may be shorter (e.g. "21" is JDK 21.0.0),
# and any fourth or later element is reserved for downstream consumers.
JAVA_VERSION_NUMBER_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")

# A `(major, minor, security)` version number, as defined by JEP 223
type JdkVersion = tuple[int, int, int]


class SampleKey(NamedTuple):
    """Identifies the sample of runs a single measurements file holds"""

    scenario: str
    jdk: str
    implementation: str
    benchmark: str

    @classmethod
    def from_benchmark_directory(cls, benchmark_dir: Path) -> SampleKey:
        """Identify the sample held in a `<scenario>/<jdk>/<implementation>/<benchmark>` directory"""
        implementation_dir = benchmark_dir.parent
        jdk_dir = implementation_dir.parent
        scenario_dir = jdk_dir.parent
        return cls(
            scenario_dir.name,
            jdk_dir.name,
            implementation_dir.name,
            benchmark_dir.name,
        )

    def __str__(self) -> str:
        return "/".join(self)


# JDK Versions
def parse_jdk_version(java_version_output: str) -> JdkVersion | None:
    """Parse the `(major, minor, security)` version of a `java -version` output, or `None` if absent"""
    version_string_match = JAVA_VERSION_STRING_PATTERN.search(java_version_output)
    if version_string_match is None:
        return None

    version_string = version_string_match.group(1)
    # The legacy pattern goes first, since the JEP 223 pattern also matches a legacy
    # version string. It would read "1.8.0_492" as major 1 instead of major 8.
    version_patterns = (JAVA_LEGACY_VERSION_NUMBER_PATTERN, JAVA_VERSION_NUMBER_PATTERN)
    version_number_match = next(
        (match for p in version_patterns if (match := p.match(version_string))),
        None,
    )
    if version_number_match is None:
        return None

    major, minor, security = version_number_match.groups()
    return int(major), int(minor or 0), int(security or 0)


def read_jdk_version(jdk_dir: Path) -> JdkVersion | None:
    """Read the version of the JDK in `jdk_dir`, or `None` if it cannot be determined"""
    java_version_path = jdk_dir / JAVA_VERSION_FILE_NAME
    try:
        version = parse_jdk_version(java_version_path.read_text(encoding="utf-8"))
    except OSError:
        logger.exception("Error when reading '%s'", java_version_path)
        return None

    if version is None:
        logger.warning("Could not parse JDK version from '%s'", java_version_path)

    return version


def collect_jdk_versions(scenario_dirs: list[Path]) -> dict[str, JdkVersion]:
    """Map the name of every JDK directory to the version of the JDK it holds"""
    versions: dict[str, JdkVersion] = {}
    for scenario_dir in scenario_dirs:
        for jdk_dir in sorted(filter(Path.is_dir, scenario_dir.iterdir())):
            if jdk_dir.name in versions:
                continue

            version = read_jdk_version(jdk_dir)
            if version is not None:
                versions[jdk_dir.name] = version

    return versions


def sort_jdks(jdks: list[str], versions: dict[str, JdkVersion]) -> list[str]:
    """Sort JDK names by version, listing JDKs of unknown version last, in name order"""
    return sorted(
        jdks, key=lambda jdk: (jdk not in versions, versions.get(jdk, ()), jdk)
    )


def get_jdk_labels(jdks: list[str], versions: dict[str, JdkVersion]) -> dict[str, str]:
    """
    Label each JDK of `jdks`, in the given order.

    Major versions are the shortest labels that a plot can carry, and they name the
    JDKs of a set whose major versions differ. Any other set is labeled by name, since
    a major version would then stand for more than one JDK, or for none.
    """
    major_version_set = {versions[jdk][0] for jdk in jdks if jdk in versions}
    if len(major_version_set) < len(jdks):
        return {jdk: jdk for jdk in jdks}

    return {jdk: str(versions[jdk][0]) for jdk in jdks}


# I/O
def is_non_baseline_dir(path: Path) -> bool:
    """Return `True` if `path` is a directory whose name differs from `BASELINE_DIR_NAME`; `False` otherwise"""
    return path.is_dir() and path.name != BASELINE_DIR_NAME


def read_measurements(measurements_path: Path) -> pl.DataFrame:
    """
    Read raw measurements, with columns sorted per `MEASUREMENTS_SCHEMA`.

    Derive the `PackageAndDRAM` column in memory if it's missing.
    """
    columns = pl.scan_csv(measurements_path).collect_schema().names()
    schema = (
        MEASUREMENTS_SCHEMA if "PackageAndDRAM" in columns else RAW_MEASUREMENTS_SCHEMA
    )

    # `schema` names the columns by position and ignores the header,
    # so a header in any other order would mislabel them silently
    expected = [str(column) for column in schema]
    if columns != expected:
        raise ValueError(
            f"Unexpected columns in '{measurements_path}': "
            f"expected {expected}, found {columns}"
        )

    df = pl.read_csv(measurements_path, schema=schema)
    if "PackageAndDRAM" not in df.columns:
        df = df.with_columns(
            (pl.col("Package") + pl.col("DRAM")).alias("PackageAndDRAM")
        )

    return df.select(MEASUREMENTS_COLUMNS)


def load_and_clean_measurements(measurements_path: Path) -> pl.DataFrame | None:
    """Load raw measurements, clean them and return the result as a data frame"""
    try:
        df = read_measurements(measurements_path)
        active_runs_df = df.filter(~pl.col("Program").str.starts_with(IDLE_RUN_PREFIX))
        if active_runs_df.is_empty():
            logger.warning(
                "Skipping '%s' because it only contains idle runs...", measurements_path
            )
            return None

        # Drop rows where essential metrics are missing
        len_before_drop_nulls = len(active_runs_df)
        active_runs_df = active_runs_df.drop_nulls(subset=["Time", "Package"])
        null_count = len_before_drop_nulls - len(active_runs_df)
        if null_count:
            logger.warning(
                "Dropped %d rows with missing Time/Package from '%s'",
                null_count,
                measurements_path,
            )

        return active_runs_df

    except OSError, ValueError, pl.exceptions.PolarsError:
        logger.exception("Error when reading or cleaning '%s'", measurements_path)
        return None


def common_benchmarks(baseline_dir: Path, variant_dir: Path) -> list[str]:
    """Return the common benchmarks from the specified directories"""
    baseline_bms = {d.name for d in baseline_dir.iterdir() if d.is_dir()}
    variant_bms = {d.name for d in variant_dir.iterdir() if d.is_dir()}
    missing_in_baseline = variant_bms - baseline_bms
    missing_in_variant = baseline_bms - variant_bms
    if missing_in_baseline:
        logger.warning(
            "'%s' missing benchmarks from '%s': %s",
            baseline_dir, variant_dir, sorted(missing_in_baseline)
        )  # fmt: skip

    if missing_in_variant:
        logger.warning(
            "'%s' missing benchmarks from '%s': %s",
            variant_dir, baseline_dir, sorted(missing_in_variant)
        )  # fmt: skip

    return sorted(baseline_bms.intersection(variant_bms))
