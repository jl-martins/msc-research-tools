#!/usr/bin/env -S uv run
import argparse
import csv
import importlib.metadata
import os
import platform
import re
import shlex
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import psutil
from pydantic import BaseModel, PositiveInt

try:
    __version__ = importlib.metadata.version("dacapo-min-heap")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


# --- Memory Constants ---
SAFETY_MARGIN_MB = 500
MIN_HEAP_MB = 1

# --- Exit Codes ---

# From https://github.com/openjdk/jdk11u/blob/jdk11.0.20.1/src/hotspot/share/utilities/debug.cpp#L344
OOM_EXIT_CODE = 3

# --- Default Values ---
DEFAULT_JAVA_INSTALL_ROOT = "/usr/lib/jvm"
DEFAULT_JDKS = "temurin-8 temurin-11 temurin-17 temurin-21 temurin-25"
DEFAULT_GC_OPTIONS = "-XX:+UseG1GC"
DEFAULT_INPUT_SIZE = "default"
DEFAULT_ITERATIONS = 1

# --- Environment Variables ---
JAVA_INSTALL_ROOT = Path(os.getenv("JAVA_INSTALL_ROOT", DEFAULT_JAVA_INSTALL_ROOT))

# --- Patterns ---
JAVA_VERSION_PATTERN = re.compile(r'version\s+"([^"]+)"')


# --- Data Models ---
class ParsedArgs(BaseModel):
    """Parsed command-line arguments"""

    jarfile: Path
    out_dir: Path
    max_mem_mb: PositiveInt
    jdks: list[str]
    gc_opts: list[str]
    benchmarks: list[str]
    input_size: str
    iterations: PositiveInt


class JDK(BaseModel):
    """JDK metadata"""

    name: str
    java_bin: Path
    java_major_version: int
    java_version_string: str


class System(BaseModel):
    """System information"""

    system: str
    node: str
    release: str
    version: str
    machine: str
    processor: str

    @staticmethod
    def from_uname():
        uname = platform.uname()
        return System(
            system=uname.system,
            node=uname.node,
            release=uname.release,
            version=uname.version,
            machine=uname.machine,
            processor=uname.processor,
        )


class ExecutionContext(BaseModel):
    """Execution context of this script"""

    version: str
    start_time: datetime
    system: System
    args: ParsedArgs
    jdks: list[JDK]
    environment: dict[str, str]

    @staticmethod
    def create(
        start_time: datetime,
        args: ParsedArgs,
        jdks: list[JDK],
    ) -> ExecutionContext:
        return ExecutionContext(
            version=__version__,
            start_time=start_time,
            system=System.from_uname(),
            args=args,
            jdks=jdks,
            environment={
                "JAVA_INSTALL_ROOT": str(JAVA_INSTALL_ROOT),
            },
        )


class BenchmarkContext(BaseModel):
    """Benchmark execution context."""

    benchmark: str
    input_size: str
    iterations: PositiveInt
    max_mem_mb: PositiveInt
    jdk: JDK
    gc_opts: list[str]
    jarfile: Path

    @staticmethod
    def create(args: ParsedArgs, benchmark: str, jdk: JDK) -> BenchmarkContext:
        return BenchmarkContext(
            benchmark=benchmark,
            input_size=args.input_size,
            iterations=args.iterations,
            jdk=jdk,
            gc_opts=args.gc_opts,
            max_mem_mb=args.max_mem_mb,
            jarfile=args.jarfile,
        )


class MinHeapSearchException(Exception):
    """Exception raised when a minimum heap search fails."""

    bm_ctx: BenchmarkContext
    exit_code: int

    def __init__(self, msg: str, bm_ctx: BenchmarkContext, exit_code: int):
        super().__init__(msg)
        self.bm_ctx = bm_ctx
        self.exit_code = exit_code


# --- Error Handling ---
def excepthook(
    exc_type: type[BaseException],
    exc: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return

    exc_msg = str(exc)
    if exc_msg:
        print(f"Error: {exc_msg}", file=sys.stderr)
    else:
        traceback.print_tb(exc_traceback)


sys.excepthook = excepthook


def java_major_version_parse_error(version_string: str) -> str:
    return f"Could not parse Java major version from string:\n\n{version_string}"


def unexpected_exit_code_error(ctx: BenchmarkContext, exit_code: int) -> str:
    return f"{ctx.benchmark} benchmark failed with unexpected exit code {exit_code} on {ctx.jdk.name} JDK"


def upper_bound_exceeded_error(ctx: BenchmarkContext, upper_bound: int) -> str:
    return f"{ctx.benchmark} requires more memory on {ctx.jdk.name} than the identified upper bound ({upper_bound} MiB)"


def max_mem_exceeded_error(ctx: BenchmarkContext) -> str:
    return f"{ctx.benchmark} requires more than max memory ({ctx.max_mem_mb} MiB) on {ctx.jdk.name} JDK"


# --- JDK Discovery ---
def discover_jdks(jdk_names: list[str]) -> list[JDK]:
    jdks = []
    for jdk in jdk_names:
        java_bin = JAVA_INSTALL_ROOT / jdk / "bin" / "java"
        java_version_string = get_java_version_string(java_bin)
        java_major_version = parse_java_major_version(java_version_string)
        jdks.append(
            JDK(
                name=jdk,
                java_bin=java_bin,
                java_major_version=java_major_version,
                java_version_string=java_version_string,
            )
        )

    return jdks


def get_java_version_string(java_bin: Path) -> str:
    java_version_process = subprocess.run(
        [str(java_bin), "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    # The output of `java -version` goes to stderr
    return java_version_process.stderr


def parse_java_major_version(version_string: str) -> int:
    match = re.search(JAVA_VERSION_PATTERN, version_string)
    if match is None:
        msg = java_major_version_parse_error(version_string)
        raise ValueError(msg)

    try:
        version = match.group(1)
        parts = version.split(".")
        return int(parts[1] if parts[0] == "1" else parts[0])
    except (IndexError, ValueError) as e:
        msg = java_major_version_parse_error(version_string)
        raise ValueError(msg) from e


# --- Heap Size Search ---
def build_java_options(ctx: BenchmarkContext, heap_size_mb: int) -> list[str]:
    options = [
        *ctx.gc_opts,
        f"-Xms{heap_size_mb}m",
        f"-Xmx{heap_size_mb}m",
        "-XX:+ExitOnOutOfMemoryError",
    ]

    java_major_version = ctx.jdk.java_major_version
    if ctx.benchmark == "cassandra" and java_major_version >= 17:
        options.append("-Djava.security.manager=allow")

    if ctx.benchmark == "h2o" and java_major_version > 17:
        options.append(f"-Dsys.ai.h2o.debug.allowJavaVersions={java_major_version}")

    return options


def test_heap_size(ctx: BenchmarkContext, heap_size_mb: int) -> int:
    jdk = ctx.jdk
    java_options = build_java_options(ctx, heap_size_mb)
    args = [
        str(jdk.java_bin),
        *java_options,
        "-jar",
        str(ctx.jarfile),
        "-n",
        str(ctx.iterations),
        "-s",
        ctx.input_size,
        ctx.benchmark,
    ]

    try:
        result = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as e:
        cmd = shlex.join(args)
        print(f"Error: Failed to run '{cmd}': {e}", file=sys.stderr)
        return 1

    if result.returncode in (0, OOM_EXIT_CODE):
        return result.returncode
    elif "OutOfMemoryError" in result.stdout:
        # Handles edge cases like running out of direct memory,
        # where the JVM does not exit with OOM_EXIT_CODE.
        return OOM_EXIT_CODE
    else:
        cmd = shlex.join(args)
        print(
            f"Error: '{cmd}' failed with the following output:\n\n{result.stdout}",
            end="",
            file=sys.stderr,
        )
        return result.returncode


def exp_search_min_heap_bounds(ctx: BenchmarkContext) -> tuple[int, int]:
    max_mem_mb = ctx.max_mem_mb
    current = 16
    previous = MIN_HEAP_MB

    while current <= max_mem_mb:
        exit_code = test_heap_size(ctx, current)
        if exit_code == 0:
            return previous, current
        if exit_code == OOM_EXIT_CODE:
            previous = current
            current *= 2
        else:
            msg = unexpected_exit_code_error(ctx, exit_code)
            raise MinHeapSearchException(msg, ctx, exit_code)

    # Final check at absolute ceiling (max_mem_mb)
    exit_code = test_heap_size(ctx, max_mem_mb)
    if exit_code == 0:
        return previous, max_mem_mb
    elif exit_code == OOM_EXIT_CODE:
        msg = max_mem_exceeded_error(ctx)
        raise MinHeapSearchException(msg, ctx, OOM_EXIT_CODE)
    else:
        msg = unexpected_exit_code_error(ctx, exit_code)
        raise MinHeapSearchException(msg, ctx, exit_code)


def binary_search_min_heap(ctx: BenchmarkContext, low: int, high: int) -> int:
    assert low <= high
    result = None

    while low <= high:
        mid = (low + high) // 2
        exit_code = test_heap_size(ctx, mid)

        if exit_code == 0:
            result = mid
            high = mid - 1
        elif exit_code == OOM_EXIT_CODE:
            low = mid + 1
        else:
            msg = unexpected_exit_code_error(ctx, exit_code)
            raise MinHeapSearchException(msg, ctx, exit_code)

    if result is None:
        msg = upper_bound_exceeded_error(ctx, high)
        raise MinHeapSearchException(msg, ctx, OOM_EXIT_CODE)
    else:
        return result


def compute_min_heap(ctx: BenchmarkContext) -> int:
    low, high = exp_search_min_heap_bounds(ctx)
    min_heap_mb = binary_search_min_heap(ctx, low, high)
    return min_heap_mb


# --- CLI ---
def positive_int(value: str) -> int:
    try:
        int_value = int(value)
    except ValueError:
        int_value = None

    if int_value is None or int_value <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be a positive integer.")
    else:
        return int_value


def non_blank_str(value: str) -> str:
    if not value or value.isspace():
        raise argparse.ArgumentTypeError(f"'{value}' must not be blank.")
    else:
        return value


def get_default_max_mem() -> int:
    available_mem_mb = psutil.virtual_memory().available >> 20
    if available_mem_mb <= SAFETY_MARGIN_MB:
        raise OSError(
            f"Insufficient memory: {available_mem_mb} MiB. "
            f"Need at least {SAFETY_MARGIN_MB + 1} MiB"
        )

    return available_mem_mb - SAFETY_MARGIN_MB


def get_default_out_dir(jar_name: str) -> Path:
    return Path.home() / "dacapo-min-heap" / jar_name


def list_benchmarks(jarfile: Path) -> list[str]:
    args = ["java", "-jar", str(jarfile), "-l"]
    result = subprocess.run(
        args=args,
        check=True,
        capture_output=True,
        text=True,
    )

    benchmarks = result.stdout.split()
    if benchmarks:
        return benchmarks
    else:
        cmd = shlex.join(args)
        raise LookupError(f"'{cmd}' returned no benchmarks.")


def parse_args() -> ParsedArgs:
    parser = argparse.ArgumentParser(
        description="Find the minimum heap size of DaCapo benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        color=False,
        epilog=textwrap.dedent(f"""
        environment variables:
          JAVA_INSTALL_ROOT     java installation root (default: {DEFAULT_JAVA_INSTALL_ROOT})
        """),
    )
    parser.add_argument(
        "jarfile",
        type=Path,
        help="dacapo jar file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"store the results in OUT_DIR (default: {get_default_out_dir('<jar_name>')})",
    )
    parser.add_argument(
        "-m",
        "--max-mem",
        type=positive_int,
        default=argparse.SUPPRESS,
        dest="max_mem_mb",
        help=f"limit the maximum heap size in MiB (default: available memory - {SAFETY_MARGIN_MB})",
    )
    # The following 3 arguments are represented as single strings for consistency with jbench-run
    parser.add_argument(
        "--jdks",
        type=non_blank_str,
        default=DEFAULT_JDKS,
        help=f"select the target JDKs as a single, space-separated string (default: '{DEFAULT_JDKS}')",
    )
    parser.add_argument(
        "--gc-opts",
        type=non_blank_str,
        default=DEFAULT_GC_OPTIONS,
        help=f"configure garbage collector options as a single, space-separated string (default: '{DEFAULT_GC_OPTIONS}')",
    )
    parser.add_argument(
        "--benchmarks",
        type=non_blank_str,
        default=argparse.SUPPRESS,
        help="select benchmarks as a single, space-separated string (default: all benchmarks in jar file)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=non_blank_str,
        default=DEFAULT_INPUT_SIZE,
        metavar="SIZE",
        dest="input_size",
        help=f"set the size of input data (default: '{DEFAULT_INPUT_SIZE}')",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=positive_int,
        default=DEFAULT_ITERATIONS,
        metavar="ITER",
        help=f"run each benchmark <ITER> times (default: {DEFAULT_ITERATIONS})",
    )

    args = parser.parse_args()

    out_dir = getattr(args, "out_dir", None) or get_default_out_dir(args.jarfile.stem)
    max_mem_mb = getattr(args, "max_mem_mb", None) or get_default_max_mem()
    jdks = args.jdks.split()
    gc_opts = shlex.split(args.gc_opts)
    benchmarks = getattr(args, "benchmarks", "").split() or list_benchmarks(args.jarfile)

    return ParsedArgs(
        jarfile=args.jarfile,
        out_dir=out_dir,
        max_mem_mb=max_mem_mb,
        jdks=jdks,
        gc_opts=gc_opts,
        benchmarks=benchmarks,  # ty:ignore[invalid-argument-type] (false positive)
        input_size=args.input_size,
        iterations=args.iterations,
    )


# --- Execution Context ---
def save_execution_context(ctx: ExecutionContext) -> None:
    ctx_json = ctx.model_dump_json(indent=2)
    out_file = ctx.args.out_dir / "execution_context.json"
    out_file.write_text(ctx_json)
    print(f"Saved execution context to '{out_file}'")


# --- Orchestration ---
def main() -> None:
    start_time = datetime.now(timezone.utc)
    args = parse_args()

    jdks = discover_jdks(args.jdks)
    ctx = ExecutionContext.create(start_time, args, jdks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_execution_context(ctx)

    long_csv = args.out_dir / "min_heap_long.csv"
    wide_csv = args.out_dir / "min_heap_wide.csv"
    collect_min_heaps(ctx, long_csv, wide_csv)


def collect_min_heaps(ctx: ExecutionContext, long_csv: Path, wide_csv: Path) -> None:
    with (
        long_csv.open("w", newline="", encoding="utf-8") as long_csv_fp,
        wide_csv.open("w", newline="", encoding="utf-8") as wide_csv_fp,
    ):
        args, jdks = ctx.args, ctx.jdks

        long_writer = csv.writer(long_csv_fp)
        long_writer.writerow(["benchmark", "jdk", "min_heap_mb"])

        wide_writer = csv.writer(wide_csv_fp)
        wide_writer.writerow(["benchmark", *[jdk.name for jdk in jdks]])

        for bm in args.benchmarks:
            print(f"--- Running benchmark: {bm} ---")
            jdk_min_heaps: dict[str, int | None] = {}

            for jdk in jdks:
                print(f"Computing minimum heap size on {jdk.name} JDK...")

                bm_ctx = BenchmarkContext.create(args, bm, jdk)
                min_heap_mb: int | None
                try:
                    min_heap_mb = compute_min_heap(bm_ctx)
                except MinHeapSearchException as e:
                    print(f"{e}", file=sys.stderr)
                    min_heap_mb = None

                jdk_min_heaps[jdk.name] = min_heap_mb
                long_writer.writerow([bm, jdk.name, min_heap_mb])
                long_csv_fp.flush()

            wide_writer.writerow([bm, *(jdk_min_heaps.get(jdk.name) for jdk in jdks)])
            wide_csv_fp.flush()


if __name__ == "__main__":
    main()
