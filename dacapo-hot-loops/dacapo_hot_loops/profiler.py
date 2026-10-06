"""Profiling module"""

import logging
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.build import run_command
from dacapo_hot_loops.config import JvmFlagRule, StudyConfig

logger = logging.getLogger(__name__)

# Matches the major version of both "1.8.0_492" and "21.0.11"
JAVA_VERSION_PATTERN = re.compile(r'version\s+"(?:1\.)?(\d+)')
PROFILER_CALLBACK = "org.dacapo.profiling.AsyncProfilerCallback"
PROFILE_SUFFIXES = (".jfr", ".txt", ".html")
# jfrconv output format and file suffix of each converted profile
CONVERSIONS = (("collapsed", ".txt"), ("html", ".html"))
PROC_SYS_DIR = Path("/proc/sys")


def java_major_version(java: str) -> int:
    # `java -version` writes to stderr
    version_output = run_command([java, "-version"], capture_output=True).stderr
    version_match = JAVA_VERSION_PATTERN.search(version_output)
    if version_match is None:
        raise HotLoopsError(f"Could not parse the Java version from:\n{version_output}")
    else:
        return int(version_match[1])


def check_kernel_settings(
    max_settings: dict[str, int],
    proc_sys_dir: Path = PROC_SYS_DIR,
) -> None:
    fix_commands = []
    for name, max_value in max_settings.items():
        setting_file = proc_sys_dir / name.replace(".", "/")
        try:
            value = int(setting_file.read_text())
        except (OSError, ValueError) as e:
            raise HotLoopsError(f"Could not read kernel setting {name}: {e}") from e

        if value > max_value:
            fix_commands.append(f"sudo sysctl -w {name}={max_value}")

    if fix_commands:
        fixes = "\n".join(fix_commands)
        raise HotLoopsError(f"Kernel settings do not allow profiling. Run:\n{fixes}")


def rule_applies(rule: JvmFlagRule, java_major: int) -> bool:
    above_min = rule.min_java is None or rule.min_java <= java_major
    below_max = rule.max_java is None or java_major <= rule.max_java
    return above_min and below_max


def benchmark_jvm_flags(rules: list[JvmFlagRule], java_major: int) -> list[str]:
    return [
        flag.format(java_major=java_major)
        for rule in rules
        if rule_applies(rule, java_major)
        for flag in rule.flags
    ]


def profile_command(
    config: StudyConfig,
    bm: str,
    java_major: int,
    jar: Path,
    profile_dir: Path,
) -> list[str]:
    profiling = config.profiling
    flag_rules = config.benchmarks[bm].jvm_flags
    return [
        config.executables.java,
        *profiling.jvm_flags,
        f"-Dasync-profiler.event={profiling.event}",
        f"-Dasync-profiler.start.opts=interval={profiling.interval}",
        "-Dasync-profiler.output.fmt=jfr",
        f"-Dasync-profiler.output.dir={profile_dir}",
        *benchmark_jvm_flags(flag_rules, java_major),
        "-jar",
        str(jar),
        "--callback",
        PROFILER_CALLBACK,
        "-n",
        str(profiling.iterations),
        bm,
    ]


def conversion_commands(
    jfrconv: str,
    bm: str,
    profile_dir: Path,
) -> list[list[str]]:
    jfr_file = profile_dir / f"{bm}-profile.jfr"
    return [
        [
            jfrconv,
            "--lines",
            "-o",
            output_format,
            str(jfr_file),
            str(profile_dir / f"{bm}-profile{suffix}"),
        ]
        for output_format, suffix in CONVERSIONS
    ]


def require_new_dirs(dirs: Iterable[Path]) -> None:
    existing_dirs = [str(path) for path in dirs if path.exists()]
    if existing_dirs:
        raise HotLoopsError(f"Directories already exist: {', '.join(existing_dirs)}")


def profile_benchmark(
    config: StudyConfig,
    bm: str,
    java_major: int,
    jar: Path,
    profile_dir: Path,
) -> None:
    commands = [
        profile_command(config, bm, java_major, jar, profile_dir),
        *conversion_commands(config.executables.jfrconv, bm, profile_dir),
    ]

    profile_dir.mkdir(parents=True)
    command_log_path = profile_dir / "profile-commands.log"
    # DaCapo writes a scratch directory into its working directory
    with (
        command_log_path.open("w", encoding="utf-8") as log_file,
        tempfile.TemporaryDirectory(prefix="dacapo-hot-loops-") as work_dir,
    ):
        for command in commands:
            log_file.write(f"{shlex.join(command)}\n")
            log_file.flush()
            run_command(command, cwd=work_dir)

    profile_stem = profile_dir / f"{bm}-profile"
    profile_files = [profile_stem.with_suffix(suffix) for suffix in PROFILE_SUFFIXES]
    missing_files = [str(file) for file in profile_files if not file.is_file()]
    if missing_files:
        raise HotLoopsError(f"Missing profile files: {', '.join(missing_files)}")


def profile(
    config: StudyConfig,
    benchmarks_dir: Path,
    out_dir: Path,
    benchmarks: list[str],
) -> None:
    unknown_benchmarks = [bm for bm in benchmarks if bm not in config.benchmarks]
    if unknown_benchmarks:
        raise HotLoopsError(f"Unknown benchmarks: {', '.join(unknown_benchmarks)}")

    java_major = java_major_version(config.executables.java)
    expected_major = config.java_major
    if java_major != expected_major:
        raise HotLoopsError(f"Java is version {java_major}, expected {expected_major}")

    check_kernel_settings(config.profiling.max_kernel_settings)

    # The commands run in a temporary directory, so every path is absolute
    profiles_dir = out_dir.resolve() / "profiles"
    jar = (benchmarks_dir / config.jar_name).resolve()
    require_new_dirs(profiles_dir / bm for bm in benchmarks)

    failed_benchmarks = []
    for bm in benchmarks:
        logger.info(f"Profiling {bm}")
        try:
            profile_benchmark(config, bm, java_major, jar, profiles_dir / bm)
        except HotLoopsError as e:
            logger.error(f"Profiling {bm} failed: {e}")
            failed_benchmarks.append(bm)

    if failed_benchmarks:
        raise HotLoopsError(f"Profiling failed for: {', '.join(failed_benchmarks)}")


def import_profiles(config: StudyConfig, source_dir: Path, out_dir: Path) -> int:
    """Copy every benchmark's profile files and return the number copied"""
    profiles_dir = out_dir / "profiles"
    copies = [
        (
            source_dir / "bms" / bm / f"{bm}-profile{suffix}",
            profiles_dir / bm / f"{bm}-profile{suffix}",
        )
        for bm in config.benchmarks
        for suffix in PROFILE_SUFFIXES
    ]

    missing_sources = [str(source) for source, _ in copies if not source.is_file()]
    if missing_sources:
        raise HotLoopsError(f"Missing profile files: {', '.join(missing_sources)}")

    require_new_dirs(profiles_dir / bm for bm in config.benchmarks)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return len(copies)
