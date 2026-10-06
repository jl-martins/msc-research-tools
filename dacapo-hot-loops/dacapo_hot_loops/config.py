"""Configuration module"""

import os
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    model_validator,
)

from dacapo_hot_loops import HotLoopsError

# Paths and executables in the config may start with `~`
ExpandedPath = Annotated[Path, AfterValidator(Path.expanduser)]
ExpandedStr = Annotated[str, AfterValidator(os.path.expanduser)]


class StrictModel(BaseModel):
    """Model that rejects unknown keys"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Executables(StrictModel):
    """Executables, found on PATH unless given as paths"""

    java: ExpandedStr = "java"
    jfrconv: ExpandedStr = "jfrconv"
    ant: ExpandedStr = "ant"
    git: ExpandedStr = "git"


class WorktreeConfig(StrictModel):
    """Git worktree that `prepare` creates and builds"""

    repository: ExpandedPath
    branch: str
    path: ExpandedPath
    ant_java_home: ExpandedPath
    ant_dist_options: list[str] = []
    local_properties: dict[str, ExpandedStr]


class DacapoConfig(StrictModel):
    """Source of the DaCapo build"""

    mode: Literal["existing", "worktree"]
    dir: ExpandedPath | None = None
    worktree: WorktreeConfig | None = None

    @model_validator(mode="after")
    def require_mode_settings(self) -> Self:
        if self.mode == "existing" and self.dir is None:
            raise ValueError("mode 'existing' requires 'dir'")
        elif self.mode == "worktree" and self.worktree is None:
            raise ValueError("mode 'worktree' requires the [dacapo.worktree] table")
        else:
            return self


class ProfilingConfig(StrictModel):
    """Settings of one profiling run"""

    jvm_flags: list[str]
    event: str
    interval: str
    iterations: PositiveInt
    # Largest allowed value of each sysctl setting
    max_kernel_settings: dict[str, int]


def check_flag_placeholders(flag: str) -> str:
    """Fail if the flag has a placeholder other than `{java_major}`"""
    try:
        flag.format(java_major=0)
    except (AttributeError, IndexError, KeyError, ValueError) as e:
        raise ValueError(f"invalid placeholder in {flag!r}: {e!r}") from e

    return flag


# JVM flag that may contain a `{java_major}` placeholder
FlagTemplate = Annotated[str, AfterValidator(check_flag_placeholders)]


class JvmFlagRule(StrictModel):
    """JVM flags for one benchmark on a range of Java major versions"""

    flags: list[FlagTemplate] = Field(min_length=1)
    min_java: PositiveInt | None = None
    max_java: PositiveInt | None = None

    @model_validator(mode="after")
    def require_ordered_range(self) -> Self:
        match self.min_java, self.max_java:
            case int(min_java), int(max_java) if min_java > max_java:
                raise ValueError("'min_java' must not exceed 'max_java'")
            case _:
                return self


class BenchmarkConfig(StrictModel):
    """Source roots and JVM flags of one benchmark"""

    roots: list[Path] = Field(min_length=1)
    # Glob patterns, relative to each root
    excludes: list[str] = []
    jvm_flags: list[JvmFlagRule] = []


class SourcesConfig(StrictModel):
    """Exclusion rules shared by every benchmark"""

    # A path with one of these directory segments is excluded...
    excluded_dirs: list[str]
    # ...unless the segment starts one of these subpaths
    kept_dirs: list[Path]


class LineMarker(StrictModel):
    """File whose streams variant contains a stream-only line"""

    kind: Literal["line"]
    path: Path
    line: str


class FileMarker(StrictModel):
    """File that only the streams variant contains"""

    kind: Literal["file"]
    path: Path


StreamsMarker = Annotated[LineMarker | FileMarker, Field(discriminator="kind")]


class AnalysisConfig(StrictModel):
    """Prefixes and thresholds of the analysis"""

    excluded_class_prefixes: list[str]
    hot_method_share: float = Field(gt=0, le=1)
    hot_loop_share: float = Field(gt=0, le=1)


class StudyConfig(StrictModel):
    """Every study-specific setting"""

    results_dir: ExpandedPath
    jar_name: str
    jar_version: str
    java_major: PositiveInt
    executables: Executables = Executables()
    dacapo: DacapoConfig
    profiling: ProfilingConfig
    sources: SourcesConfig
    streams_markers: list[StreamsMarker]
    analysis: AnalysisConfig
    benchmarks: dict[str, BenchmarkConfig] = Field(min_length=1)


def load_config(path: Path) -> StudyConfig:
    try:
        with path.open("rb") as config_file:
            data = tomllib.load(config_file)
        return StudyConfig.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as e:
        raise HotLoopsError(f"Could not load config '{path}':\n{e}") from e
