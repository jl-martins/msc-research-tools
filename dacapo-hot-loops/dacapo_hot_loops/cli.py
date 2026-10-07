"""Command-line interface module"""

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.analysis import analyze
from dacapo_hot_loops.build import prepare, resolve_benchmarks_dir
from dacapo_hot_loops.config import StudyConfig, load_config
from dacapo_hot_loops.profiler import import_profiles, profile
from dacapo_hot_loops.report import report
from dacapo_hot_loops.sources import check_sources
from dacapo_hot_loops.worksheet import write_worksheet

logger = logging.getLogger(__name__)

type CommandHandler = Callable[[argparse.Namespace, StudyConfig, Path], None]


def print_class_counts(class_counts: dict[str, int]) -> None:
    for bm, count in class_counts.items():
        print(f"{bm}: {count} indexed classes")

    logger.info("All source checks passed, with no ambiguous classes")


def run_prepare(args: argparse.Namespace, config: StudyConfig, out_dir: Path) -> None:
    benchmarks_dir = prepare(config, out_dir)
    print_class_counts(check_sources(config, benchmarks_dir))


def run_check_sources(
    args: argparse.Namespace,
    config: StudyConfig,
    out_dir: Path,
) -> None:
    benchmarks_dir = resolve_benchmarks_dir(config.dacapo)
    print_class_counts(check_sources(config, benchmarks_dir))


def run_profile(args: argparse.Namespace, config: StudyConfig, out_dir: Path) -> None:
    benchmarks_dir = resolve_benchmarks_dir(config.dacapo)
    if args.benchmarks is not None:
        benchmarks = args.benchmarks.split()
    else:
        benchmarks = list(config.benchmarks)

    profile(config, benchmarks_dir, out_dir, benchmarks)


def run_import_profiles(
    args: argparse.Namespace,
    config: StudyConfig,
    out_dir: Path,
) -> None:
    copied_count = import_profiles(config, args.source_dir, out_dir)
    logger.info(f"Copied {copied_count} profile files to '{out_dir / 'profiles'}'")


def run_analyze(args: argparse.Namespace, config: StudyConfig, out_dir: Path) -> None:
    benchmarks_dir = resolve_benchmarks_dir(config.dacapo)
    analyze(config, args.config, benchmarks_dir, out_dir)


def run_worksheet(args: argparse.Namespace, config: StudyConfig, out_dir: Path) -> None:
    write_worksheet(out_dir)


def run_report(args: argparse.Namespace, config: StudyConfig, out_dir: Path) -> None:
    benchmarks_dir = resolve_benchmarks_dir(config.dacapo)
    report(config, args.config, benchmarks_dir, out_dir)


def non_blank_str(value: str) -> str:
    if value.strip() != "":
        return value
    else:
        raise argparse.ArgumentTypeError(f"'{value}' must not be blank.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dacapo-hot-loops",
        description="Find hot loops in DaCapo benchmark profiles",
        color=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="study config file",
    )
    common_parser.add_argument(
        "--out",
        type=Path,
        help="results directory (default: the config's results_dir)",
    )

    def add_command(
        name: str,
        handler: CommandHandler,
        help_text: str,
    ) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(
            name,
            parents=[common_parser],
            help=help_text,
            description=help_text,
            color=False,
        )
        subparser.set_defaults(handler=handler)
        return subparser

    add_command(
        "prepare",
        run_prepare,
        "build DaCapo in a new git worktree (worktree mode only), then run check-sources",
    )
    add_command(
        "check-sources",
        run_check_sources,
        "check the jar and the source trees",
    )
    profile_parser = add_command("profile", run_profile, "profile the benchmarks")
    profile_parser.add_argument(
        "--benchmarks",
        type=non_blank_str,
        help="select benchmarks as a single, space-separated string (default: all benchmarks in config)",
    )
    import_parser = add_command(
        "import-profiles",
        run_import_profiles,
        "copy existing profiles into the results directory",
    )
    import_parser.add_argument(
        "--from",
        type=Path,
        required=True,
        dest="source_dir",
        metavar="DIR",
        help="benchmarks directory that holds the profiles in bms/<bm>/",
    )
    add_command("analyze", run_analyze, "find the hot loops")
    add_command("worksheet", run_worksheet, "write the classification worksheet")
    add_command("report", run_report, "write the report tables")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        out_dir = args.out if args.out is not None else config.results_dir
        args.handler(args, config, out_dir)
    except HotLoopsError as e:
        logger.error(e)
        sys.exit(1)
