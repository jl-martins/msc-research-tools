"""Tests of the prepare command"""

import os
import subprocess
from pathlib import Path

import pytest
from conftest import write_executable, write_file

from dacapo_hot_loops.cli import main

WORKTREE_TABLE = """
[dacapo.worktree]
repository = "repo"
branch = "chopin-loops"
path = "worktree"
ant_java_home = "/opt/temurin-8"
ant_dist_options = ["-Dbuild.is-official=true"]

[dacapo.worktree.local_properties]
"build.failonerror" = "true"
"jdk.11.home" = "~/temurin-11"
"""

# Records its call, then builds by copying a prepared benchmarks directory
FAKE_ANT = """\
#!/bin/sh
echo "$JAVA_HOME $*" > ant-call.txt
echo "BUILD SUCCESSFUL"
cp -R "$FAKE_ANT_OUTPUT/." .
"""

PREPARE_ARGS = ["prepare", "--config", "study.toml", "--out", "out"]
CLASS_COUNTS = "app: 1 indexed classes\nother: 1 indexed classes\n"


def git(*args: str) -> str:
    command = ["git", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def worktree_study(study_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Study in worktree mode, with a git repository and a fake ant on PATH"""
    config_path = study_dir / "study.toml"
    existing_text = config_path.read_text()
    config_text = existing_text.replace('mode = "existing"', 'mode = "worktree"')
    config_path.write_text(config_text + WORKTREE_TABLE)

    # Isolates git from the user's and the system's config
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    write_file(study_dir / "repo/benchmarks/build.xml", "<project/>\n")
    git("init", "-q", "-b", "main", "repo")
    git("-C", "repo", "add", ".")
    git("-C", "repo", *identity, "commit", "-q", "-m", "Add build")
    git("-C", "repo", "branch", "chopin-loops")

    bin_dir = study_dir / "bin"
    write_executable(bin_dir / "ant", FAKE_ANT)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_ANT_OUTPUT", str(study_dir / "benchmarks"))
    return study_dir


def test_existing_mode_only_checks_sources(study_dir, capsys):
    main(PREPARE_ARGS)
    assert capsys.readouterr().out == CLASS_COUNTS
    assert not Path("out").exists()


def test_worktree_mode_builds(worktree_study, capsys):
    main(PREPARE_ARGS)
    ant_call = Path("worktree/benchmarks/ant-call.txt").read_text()
    assert capsys.readouterr().out == CLASS_COUNTS
    assert ant_call == "/opt/temurin-8 -Dbuild.is-official=true dist\n"
    assert Path("out/build.log").read_text() == "BUILD SUCCESSFUL\n"


def test_worktree_gets_local_properties(worktree_study):
    main(PREPARE_ARGS)
    properties = Path("worktree/benchmarks/local.properties").read_text()
    expected = f"build.failonerror=true\njdk.11.home={Path.home()}/temurin-11\n"
    assert properties == expected


def test_existing_worktree_path_fails(worktree_study, caplog):
    Path("worktree").mkdir()
    with pytest.raises(SystemExit):
        main(PREPARE_ARGS)
    assert "already exists" in caplog.text
    assert len(git("-C", "repo", "worktree", "list").splitlines()) == 1


def test_build_without_jar_fails(worktree_study, monkeypatch, caplog):
    Path("empty-build").mkdir()
    monkeypatch.setenv("FAKE_ANT_OUTPUT", str(Path("empty-build").absolute()))
    with pytest.raises(SystemExit):
        main(PREPARE_ARGS)
    assert "did not produce" in caplog.text
