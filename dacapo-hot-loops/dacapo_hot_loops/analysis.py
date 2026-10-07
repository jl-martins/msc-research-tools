"""Analysis module"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

import polars as pl

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.collapsed import parse_collapsed
from dacapo_hot_loops.config import StudyConfig
from dacapo_hot_loops.javasrc import JavaSources, LineRange
from dacapo_hot_loops.loops import (
    ChargeableLoop,
    LoopCharges,
    caller_loops,
    charge_loops,
    declarations_of_method,
    hot_loops,
    method_loops,
    unique_loops,
)
from dacapo_hot_loops.methods import (
    BenchmarkCharges,
    CallerCharge,
    MethodCharge,
    charge_stacks,
    hot_methods,
    main_caller,
)
from dacapo_hot_loops.sources import index_sources, read_jar_version

logger = logging.getLogger(__name__)

PROFILES_DIR_NAME = "profiles"
ANALYSIS_DIR_NAME = "analysis"
PROVENANCE_NAME = "provenance.json"
# Decimal places that every share column holds
SHARE_DECIMALS = 6


class MethodStatus(StrEnum):
    """Whether the source holds a declaration of a hot method"""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


SUMMARY_SCHEMA: Mapping[str, pl.DataType] = {
    "benchmark": pl.String(),
    "total_samples": pl.Int64(),
    "no_app_samples": pl.Int64(),
    "no_source_samples": pl.Int64(),
    "charged_samples": pl.Int64(),
    "hot_methods": pl.Int64(),
    "unresolved_hot_methods": pl.Int64(),
    "unresolved_share": pl.Float64(),
    "chargeable_loops": pl.Int64(),
    "hot_loops": pl.Int64(),
    "outside_loops_share": pl.Float64(),
    "parse_error_files": pl.Int64(),
}
HOT_METHOD_SCHEMA: Mapping[str, pl.DataType] = {
    "benchmark": pl.String(),
    "rank": pl.Int64(),
    "method_key": pl.String(),
    "file": pl.String(),
    "samples": pl.Int64(),
    "share": pl.Float64(),
    "cumulative_share": pl.Float64(),
    "status": pl.String(),
    "caller_method_key": pl.String(),
    "caller_file": pl.String(),
    "caller_line": pl.Int64(),
    "caller_samples": pl.Int64(),
}
LOOP_SCHEMA: Mapping[str, pl.DataType] = {
    "benchmark": pl.String(),
    "loop_id": pl.String(),
    "file": pl.String(),
    "start_line": pl.Int64(),
    "end_line": pl.Int64(),
    "enclosing_method_key": pl.String(),
    "origin": pl.String(),
    "samples": pl.Int64(),
    "share": pl.Float64(),
}
UNRESOLVED_CLASS_SCHEMA: Mapping[str, pl.DataType] = {
    "benchmark": pl.String(),
    "class": pl.String(),
    "samples": pl.Int64(),
}
PARSE_ERROR_SCHEMA: Mapping[str, pl.DataType] = {
    "benchmark": pl.String(),
    "file": pl.String(),
    "error_lines": pl.String(),
}

type Row = dict[str, object]
type RowBuilder = Callable[[BenchmarkAnalysis], list[Row]]


class HotMethod(NamedTuple):
    """Hot method with its rank, its main caller and its declarations"""

    rank: int
    charge: MethodCharge
    caller: CallerCharge | None
    declarations: list[LineRange]
    cumulative_count: int

    @property
    def is_resolved(self) -> bool:
        """Whether any declaration in the source matches the method"""
        return bool(self.declarations)


class BenchmarkAnalysis(NamedTuple):
    """Everything the analysis tables hold about one benchmark"""

    benchmark: str
    charges: BenchmarkCharges
    methods: list[HotMethod]
    loops: list[ChargeableLoop]
    loop_charges: LoopCharges
    hot: list[ChargeableLoop]
    parse_errors: dict[Path, tuple[int, ...]]


def share_of(part: int, whole: int) -> float:
    """A fraction of a benchmark's charged samples, which may be zero"""
    return part / whole if whole else 0.0


def analyze_benchmark(
    benchmark: str,
    config: StudyConfig,
    benchmarks_dir: Path,
    profile_text: str,
) -> BenchmarkAnalysis:
    """Charge one benchmark's stacks to its methods and then to its loops"""
    analysis_config = config.analysis
    index = index_sources(
        benchmarks_dir,
        config.benchmarks[benchmark],
        config.sources,
        config.java_major,
    )
    # One cache per benchmark attributes a shared file's parse errors to each
    sources = JavaSources(benchmarks_dir)
    prefixes = tuple(analysis_config.excluded_class_prefixes)
    benchmark_charges = charge_stacks(
        parse_collapsed(profile_text, prefixes), index, sources
    )

    methods: list[HotMethod] = []
    cumulative_count = 0
    ranked = hot_methods(benchmark_charges, analysis_config.hot_method_share)
    for rank, charge in enumerate(ranked, start=1):
        cumulative_count += charge.sample_count
        declarations = declarations_of_method(charge, sources)
        caller = main_caller(benchmark_charges, charge.method_key)
        methods.append(HotMethod(rank, charge, caller, declarations, cumulative_count))

    from_methods = [
        loop
        for method in methods
        for loop in method_loops(method.charge, method.declarations, sources)
    ]
    # An unresolved method contributes no loops, its main caller's included
    from_callers = [
        loop
        for method in methods
        if method.is_resolved and method.caller is not None
        for loop in caller_loops(method.caller, sources)
    ]
    # Method loops come first, so a loop both reach keeps the method origin
    loops = unique_loops([*from_methods, *from_callers])

    loop_charges = charge_loops(benchmark_charges, loops, sources)
    hot = hot_loops(
        loops,
        loop_charges,
        benchmark_charges.charged_sample_count,
        analysis_config.hot_loop_share,
    )
    parse_errors = {
        path: parsed.error_lines
        for path, parsed in sources.parsed_files.items()
        if parsed.error_lines
    }

    return BenchmarkAnalysis(
        benchmark=benchmark,
        charges=benchmark_charges,
        methods=methods,
        loops=loops,
        loop_charges=loop_charges,
        hot=hot,
        parse_errors=parse_errors,
    )


def summary_row(analysis: BenchmarkAnalysis) -> Row:
    benchmark_charges = analysis.charges
    charged_count = benchmark_charges.charged_sample_count
    unresolved = [m for m in analysis.methods if not m.is_resolved]
    unresolved_count = sum(m.charge.sample_count for m in unresolved)
    outside_count = analysis.loop_charges.outside_count

    return {
        "benchmark": analysis.benchmark,
        "total_samples": benchmark_charges.total_sample_count,
        "no_app_samples": benchmark_charges.no_app_sample_count,
        "no_source_samples": benchmark_charges.no_source_sample_count,
        "charged_samples": charged_count,
        "hot_methods": len(analysis.methods),
        "unresolved_hot_methods": len(unresolved),
        "unresolved_share": share_of(unresolved_count, charged_count),
        "chargeable_loops": len(analysis.loops),
        "hot_loops": len(analysis.hot),
        "outside_loops_share": share_of(outside_count, charged_count),
        "parse_error_files": len(analysis.parse_errors),
    }


def caller_columns(caller: CallerCharge | None) -> Row:
    """The caller columns of a hot method (all empty if `caller` is None)"""
    if caller is None:
        return {
            "caller_method_key": None,
            "caller_file": None,
            "caller_line": None,
            "caller_samples": None,
        }
    else:
        return {
            "caller_method_key": caller.method_key,
            "caller_file": str(caller.file),
            "caller_line": caller.line,
            "caller_samples": caller.sample_count,
        }


def hot_method_rows(analysis: BenchmarkAnalysis) -> list[Row]:
    charged_count = analysis.charges.charged_sample_count

    return [
        {
            "benchmark": analysis.benchmark,
            "rank": method.rank,
            "method_key": method.charge.method_key,
            "file": str(method.charge.file),
            "samples": method.charge.sample_count,
            "share": share_of(method.charge.sample_count, charged_count),
            "cumulative_share": share_of(method.cumulative_count, charged_count),
            "status": MethodStatus.RESOLVED
            if method.is_resolved
            else MethodStatus.UNRESOLVED,
            **caller_columns(method.caller),
        }
        for method in analysis.methods
    ]


def loop_rows(
    analysis: BenchmarkAnalysis,
    loops: Iterable[ChargeableLoop],
) -> list[Row]:
    charged_count = analysis.charges.charged_sample_count
    loop_counts = analysis.loop_charges.loop_counts
    ranked = sorted(
        loops,
        key=lambda loop: (-loop_counts.get(loop.loop_id, 0), loop.loop_id),
    )

    return [
        {
            "benchmark": analysis.benchmark,
            "loop_id": loop.loop_id,
            "file": str(loop.file),
            "start_line": loop.range.start,
            "end_line": loop.range.end,
            "enclosing_method_key": loop.enclosing_method_key,
            "origin": loop.origin,
            "samples": loop_counts.get(loop.loop_id, 0),
            "share": share_of(loop_counts.get(loop.loop_id, 0), charged_count),
        }
        for loop in ranked
    ]


def unresolved_class_rows(analysis: BenchmarkAnalysis) -> list[Row]:
    ranked = sorted(
        analysis.charges.unresolved_classes.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        {"benchmark": analysis.benchmark, "class": class_name, "samples": count}
        for class_name, count in ranked
    ]


def parse_error_rows(analysis: BenchmarkAnalysis) -> list[Row]:
    return [
        {
            "benchmark": analysis.benchmark,
            "file": str(path),
            "error_lines": ";".join(map(str, error_lines)),
        }
        for path, error_lines in sorted(analysis.parse_errors.items())
    ]


def write_table(rows: list[Row], schema: Mapping[str, pl.DataType], path: Path) -> None:
    """Write one table, with every share to a fixed number of decimal places"""
    pl.DataFrame(rows, schema=schema).write_csv(path, float_precision=SHARE_DECIMALS)


def write_tables(analyses: Sequence[BenchmarkAnalysis], analysis_dir: Path) -> None:
    """Write each table in turn, so only one table's rows are held at once"""
    tables: list[tuple[str, Mapping[str, pl.DataType], RowBuilder]] = [
        ("summary.csv", SUMMARY_SCHEMA, lambda a: [summary_row(a)]),
        ("hot_methods.csv", HOT_METHOD_SCHEMA, hot_method_rows),
        ("chargeable_loops.csv", LOOP_SCHEMA, lambda a: loop_rows(a, a.loops)),
        ("hot_loops.csv", LOOP_SCHEMA, lambda a: loop_rows(a, a.hot)),
        ("unresolved_classes.csv", UNRESOLVED_CLASS_SCHEMA, unresolved_class_rows),
        ("parse_errors.csv", PARSE_ERROR_SCHEMA, parse_error_rows),
    ]
    for name, schema, build_rows in tables:
        rows = [row for analysis in analyses for row in build_rows(analysis)]
        write_table(rows, schema, analysis_dir / name)


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def tool_version() -> str:
    try:
        return version("dacapo-hot-loops")
    except PackageNotFoundError:
        return "unknown"


def write_provenance(
    config_path: Path,
    jar_version: str | None,
    inputs: Mapping[str, Path],
    path: Path,
) -> None:
    """Record the tool, the config, the jar and the hash of every input file"""
    provenance = {
        "tool_version": tool_version(),
        "config_sha256": file_sha256(config_path),
        "jar_version": jar_version,
        "inputs": {label: file_sha256(inputs[label]) for label in sorted(inputs)},
    }
    text = json.dumps(provenance, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def analyze(
    config: StudyConfig,
    config_path: Path,
    benchmarks_dir: Path,
    out_dir: Path,
) -> None:
    """Write every analysis table for every configured benchmark"""
    profiles_dir = out_dir / PROFILES_DIR_NAME
    profiles = {
        bm: profiles_dir / bm / f"{bm}-profile.txt" for bm in sorted(config.benchmarks)
    }
    missing = [str(path) for path in profiles.values() if not path.is_file()]
    if missing:
        raise HotLoopsError(f"Missing collapsed profiles: {', '.join(missing)}")

    # Read first so an unreadable jar fails before any table is written
    jar_version = read_jar_version(benchmarks_dir / config.jar_name)
    analyses: list[BenchmarkAnalysis] = []
    for bm, profile in profiles.items():
        logger.info("Analysing %s", bm)
        try:
            # A frame name that does not decode means the input is not collapsed
            profile_text = profile.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HotLoopsError(f"Could not read '{profile}': {e}") from e

        analyses.append(analyze_benchmark(bm, config, benchmarks_dir, profile_text))

    analysis_dir = out_dir / ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_tables(analyses, analysis_dir)
    write_provenance(
        config_path,
        jar_version,
        profiles,
        analysis_dir / PROVENANCE_NAME,
    )
    logger.info("Wrote the analysis tables to '%s'", analysis_dir)
