"""DaCapo build module"""

import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.config import DacapoConfig, StudyConfig

logger = logging.getLogger(__name__)

# A worktree's pinned dependencies are installed only when it holds one of these
MISE_CONFIG_FILES = ("mise.toml", ".mise.toml", ".tool-versions")
UV_CONFIG_FILES = ("pyproject.toml",)


def run_command(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a command, failing on a missing executable or a non-zero exit"""
    joined_command = shlex.join(command)
    try:
        result = subprocess.run(command, check=False, text=True, **kwargs)
    except OSError as e:
        raise HotLoopsError(f"Could not run '{joined_command}': {e}") from e

    if result.returncode == 0:
        return result
    else:
        # stderr is None unless the caller captured it
        details = result.stderr or ""
        message = f"'{joined_command}' exited with status {result.returncode}"
        raise HotLoopsError(f"{message}\n{details}".rstrip())


def run_optional_command(command: list[str], **kwargs: Any) -> None:
    """Run a command, skipping it when its executable is not installed"""
    if shutil.which(command[0]) is None:
        logger.warning(f"Skipping '{shlex.join(command)}': '{command[0]}' not found")
    else:
        run_command(command, **kwargs)


def resolve_benchmarks_dir(dacapo: DacapoConfig) -> Path:
    if dacapo.mode == "existing":
        assert dacapo.dir is not None
        return dacapo.dir
    else:
        assert dacapo.worktree is not None
        return dacapo.worktree.path / "benchmarks"


def prepare(config: StudyConfig, out_dir: Path) -> Path:
    """Build DaCapo if mode is `worktree`, then return the benchmarks directory"""
    if config.dacapo.mode == "worktree":
        build_worktree(config, out_dir)

    return resolve_benchmarks_dir(config.dacapo)


def build_worktree(config: StudyConfig, out_dir: Path) -> None:
    worktree = config.dacapo.worktree
    assert worktree is not None
    executables = config.executables
    benchmarks_dir = worktree.path / "benchmarks"

    if worktree.path.exists():
        raise HotLoopsError(f"Worktree path '{worktree.path}' already exists")

    logger.info(f"Creating worktree of '{worktree.branch}' at '{worktree.path}'...")
    # `git -C` would resolve a relative worktree path against the repository
    git_command = [
        executables.git,
        "-C",
        str(worktree.repository),
        "worktree",
        "add",
        str(worktree.path.absolute()),
        worktree.branch,
    ]
    run_command(git_command, capture_output=True)

    # A new worktree lacks local.properties, because git does not track it
    properties = worktree.local_properties.items()
    properties_text = "".join(f"{key}={value}\n" for key, value in properties)
    (benchmarks_dir / "local.properties").write_text(properties_text, encoding="utf-8")

    # The build calls the toolchain and the Python modules that the worktree pins
    installs = [
        ([executables.mise, "install"], MISE_CONFIG_FILES),
        ([executables.uv, "sync"], UV_CONFIG_FILES),
    ]
    pinned_commands = [
        command
        for command, manifests in installs
        if any((worktree.path / name).is_file() for name in manifests)
    ]
    for command in pinned_commands:
        logger.info(f"Running '{shlex.join(command)}' in '{worktree.path}'...")
        run_optional_command(command, cwd=worktree.path, capture_output=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    build_log = out_dir / "build.log"
    ant_command = [executables.ant, *worktree.ant_dist_options, "dist"]
    # The build runs `python`, which needs the modules synced in the worktree
    venv_dir = worktree.path / ".venv"
    path_entries = [str(venv_dir / "bin"), os.environ.get("PATH", "")]
    ant_env = os.environ | {
        "JAVA_HOME": str(worktree.ant_java_home),
        # h2o's build reads VIRTUAL_ENV, which must not be this tool's own venv
        "VIRTUAL_ENV": str(venv_dir),
        "PATH": os.pathsep.join(path_entries),
    }
    logger.info(f"Building DaCapo in '{benchmarks_dir}', logging to '{build_log}'...")
    with build_log.open("w", encoding="utf-8") as log_file:
        run_command(
            ant_command,
            cwd=benchmarks_dir,
            env=ant_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    jar = benchmarks_dir / config.jar_name
    if not jar.is_file():
        raise HotLoopsError(f"The build did not produce '{jar}'")
