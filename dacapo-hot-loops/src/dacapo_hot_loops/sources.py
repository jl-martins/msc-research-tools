"""Source index module"""

import itertools
import logging
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePath
from typing import NamedTuple

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.config import (
    BenchmarkConfig,
    FileMarker,
    LineMarker,
    SourcesConfig,
    StreamsMarker,
    StudyConfig,
)

logger = logging.getLogger(__name__)

# Fully-qualified top-level class name -> file relative to the benchmarks directory
type SourceIndex = dict[str, Path]

# The package declaration is the first token after whitespace and comments
PACKAGE_PATTERN = re.compile(
    r"""
    (?: \s | /\*.*?\*/ | //[^\n]* )*
    package \s+ ([\w.\s]+?) \s* ;
    """,
    re.DOTALL | re.VERBOSE,
)
MULTI_RELEASE_DIR_PATTERN = re.compile(r"java(\d+)")
MANIFEST_PATH = "META-INF/MANIFEST.MF"


class AmbiguousClassesError(HotLoopsError):
    """Error raised when a class has several source files"""


def index_sources(
    benchmarks_dir: Path,
    benchmark: BenchmarkConfig,
    sources: SourcesConfig,
    java_major: int,
) -> SourceIndex:
    files_by_class: defaultdict[str, list[Path]] = defaultdict(list)
    for root in benchmark.roots:
        root_dir = benchmarks_dir / root
        for file in sorted(root_dir.rglob("*.java")):
            relative_path = file.relative_to(root_dir)
            # package-info.java and module-info.java declare no class
            declares_class = file.is_file() and file.stem.isidentifier()
            if declares_class and not is_excluded(relative_path, benchmark, sources):
                class_name = top_level_class_name(file)
                files_by_class[class_name].append(file.relative_to(benchmarks_dir))

    index: SourceIndex = {}
    ambiguous_classes: dict[str, list[Path]] = {}
    for class_name, files in files_by_class.items():
        match select_multi_release(files, java_major):
            case [file]:
                index[class_name] = file
            case []:
                pass
            case selected_files:
                ambiguous_classes[class_name] = selected_files

    if ambiguous_classes:
        listing = "\n".join(
            f"  {class_name}: {', '.join(map(str, files))}"
            for class_name, files in sorted(ambiguous_classes.items())
        )
        raise AmbiguousClassesError(f"Ambiguous classes:\n{listing}")
    else:
        return index


def is_excluded(
    relative_path: PurePath,
    benchmark: BenchmarkConfig,
    sources: SourcesConfig,
) -> bool:
    matches_pattern = any(relative_path.full_match(p) for p in benchmark.excludes)
    return matches_pattern or has_excluded_dir(relative_path, sources)


def has_excluded_dir(relative_path: PurePath, sources: SourcesConfig) -> bool:
    dir_parts = relative_path.parent.parts
    for index, part in enumerate(dir_parts):
        if part in sources.excluded_dirs:
            subpath = PurePath(*dir_parts[index:])
            if not any(subpath.is_relative_to(kept) for kept in sources.kept_dirs):
                return True

    return False


def top_level_class_name(file: Path) -> str:
    source = file.read_text(encoding="utf-8-sig", errors="replace")
    package_match = PACKAGE_PATTERN.match(source)
    if package_match is None:
        return file.stem
    else:
        # Java allows whitespace between the parts of a qualified name
        package = "".join(package_match[1].split())
        return f"{package}.{file.stem}"


class Release(NamedTuple):
    """Source file of a class for one Java version of a multi-release jar"""

    version: int
    file: Path


def multi_release_dir(file: Path) -> tuple[Path, int] | None:
    """Return the `src` directory and NN when the file is under `src/javaNN`"""
    dir_parts = file.parent.parts
    for index, (parent, child) in enumerate(itertools.pairwise(dir_parts)):
        version_match = MULTI_RELEASE_DIR_PATTERN.fullmatch(child)
        if parent == "src" and version_match is not None:
            return Path(*dir_parts[: index + 1]), int(version_match[1])

    return None


def select_multi_release(files: list[Path], java_major: int) -> list[Path]:
    """Keep the copies of a class that multi-release jars load on the JDK"""
    base_files: list[Path] = []
    newest_releases: dict[Path, Release] = {}
    for file in files:
        match multi_release_dir(file):
            case None:
                base_files.append(file)
            case src_dir, version if version <= java_major:
                newest_release = newest_releases.get(src_dir)
                if newest_release is None or version > newest_release.version:
                    newest_releases[src_dir] = Release(version, file)
            case _:
                # Only a newer JDK loads this copy
                pass

    # The newest usable copy replaces the base copy under the same `src` directory
    kept_base_files = [
        file
        for file in base_files
        if not any(file.is_relative_to(src_dir) for src_dir in newest_releases)
    ]
    return [*kept_base_files, *(release.file for release in newest_releases.values())]


def read_jar_version(jar: Path) -> str | None:
    try:
        with zipfile.ZipFile(jar) as archive:
            manifest = archive.read(MANIFEST_PATH).decode("utf-8")
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError) as e:
        raise HotLoopsError(f"Could not read the manifest of '{jar}': {e}") from e

    # A line that starts with a space continues the previous line
    unfolded_manifest = re.sub(r"\r?\n ", "", manifest)
    for line in unfolded_manifest.splitlines():
        key, _, value = line.partition(": ")
        if key == "Implementation-Version":
            return value

    return None


def jar_problems(jar: Path, jar_version: str) -> Iterator[str]:
    try:
        actual_version = read_jar_version(jar)
    except HotLoopsError as e:
        yield str(e)
    else:
        if actual_version != jar_version:
            yield f"Jar version is {actual_version!r}, expected {jar_version!r}"


def marker_problems(
    benchmarks_dir: Path,
    markers: list[StreamsMarker],
) -> Iterator[str]:
    for marker in markers:
        file = benchmarks_dir / marker.path
        match marker:
            case LineMarker(line=stream_line):
                if not file.is_file():
                    yield f"Streams marker file '{file}' does not exist"
                else:
                    text = file.read_text(encoding="utf-8-sig", errors="replace")
                    if any(line.strip() == stream_line for line in text.splitlines()):
                        yield f"Streams marker file '{file}' contains {stream_line!r}"
            case FileMarker():
                if file.exists():
                    yield f"Streams-only file '{file}' exists"


def root_problems(
    benchmarks_dir: Path,
    benchmarks: dict[str, BenchmarkConfig],
) -> Iterator[str]:
    for name, benchmark in benchmarks.items():
        for root in benchmark.roots:
            root_dir = benchmarks_dir / root
            if not root_dir.is_dir():
                yield f"Source root '{root_dir}' of {name} does not exist"
            elif next(root_dir.rglob("*.java"), None) is None:
                yield f"Source root '{root_dir}' of {name} contains no .java files"


def check_sources(config: StudyConfig, benchmarks_dir: Path) -> dict[str, int]:
    """Check the jar and the source trees, and return each benchmark's class count"""
    jar = benchmarks_dir / config.jar_name
    sources, java_major = config.sources, config.java_major
    logger.info("Checking DaCapo JAR, streams markers and source roots...")
    problems = [
        *jar_problems(jar, config.jar_version),
        *marker_problems(benchmarks_dir, config.streams_markers),
        *root_problems(benchmarks_dir, config.benchmarks),
    ]

    class_counts: dict[str, int] = {}
    for name, benchmark in config.benchmarks.items():
        try:
            index = index_sources(benchmarks_dir, benchmark, sources, java_major)
            class_counts[name] = len(index)
            logger.info("Indexed %d class(es) of %s", len(index), name)
        except AmbiguousClassesError as e:
            problems.append(f"{name}: {e}")

    if problems:
        raise HotLoopsError("Source checks failed:\n" + "\n".join(problems))
    else:
        return class_counts
