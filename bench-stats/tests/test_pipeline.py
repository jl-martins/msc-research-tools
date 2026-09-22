"""End-to-end tests of bench-stats on a small tree of measurements"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bench_stats import run
from measurements import load_and_clean_measurements

HEADER = [
    "Language", "Program", "Powercap", "Package", "Core",
    "Uncore", "DRAM", "Time", "Temperature", "Memory",
]  # fmt: skip

# The header of `HEADER` with `Package` and `Time` swapped
SWAPPED_HEADER = [
    "Language", "Program", "Powercap", "Time", "Core",
    "Uncore", "DRAM", "Package", "Temperature", "Memory",
]  # fmt: skip

SCENARIO = "cold"
JDKS = {"temurin-17": "17.0.18", "temurin-21": "21.0.11"}
IMPLEMENTATIONS = ("loops", "streams")
BENCHMARKS = ("alpha", "beta", "gamma")
RUNS = 10


def measurement_rows(
    rng: np.random.Generator,
    benchmark: str,
    scale: float,
) -> list[dict[str, object]]:
    """Return the active runs of one benchmark, followed by one idle run"""
    rows: list[dict[str, object]] = []
    for program in [benchmark] * RUNS + [f"idle_{benchmark}"]:
        package = scale * rng.normal(100, 5)
        rows.append({
            "Language": "Java",
            "Program": program,
            "Powercap": "none",
            "Package": package,
            "Core": 0.6 * package,
            "Uncore": 0.1 * package,
            "DRAM": scale * rng.normal(10, 1),
            "Time": scale * rng.normal(1000, 50),
            "Temperature": rng.normal(50, 2),
            "Memory": float(rng.integers(1_000, 2_000)),
        })  # fmt: skip
    return rows


def write_measurements(
    path: Path,
    rows: list[dict[str, object]],
    header: list[str] = HEADER,
) -> None:
    """Write `rows` as a measurements file whose columns follow `header`"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines += [",".join(str(row[column]) for column in header) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def build_tree(root: Path, header: list[str] = HEADER) -> Path:
    """Build `<root>/<scenario>/<jdk>/<implementation>/<benchmark>` with seeded data"""
    rng = np.random.default_rng(0)
    for jdk, version in JDKS.items():
        jdk_dir = root / SCENARIO / jdk
        jdk_dir.mkdir(parents=True)
        (jdk_dir / "java_version.txt").write_text(f'openjdk version "{version}"\n')
        for implementation in IMPLEMENTATIONS:
            for index, benchmark in enumerate(BENCHMARKS):
                # Each benchmark shifts by its own amount, so the correlation is defined
                scale = 1.0 if implementation == "loops" else 1.1 + 0.1 * index
                write_measurements(
                    jdk_dir / implementation / benchmark / "measurements.csv",
                    measurement_rows(rng, benchmark, scale),
                    header,
                )

    return root


def csv_outputs(root: Path) -> dict[str, bytes]:
    """Return the content of every CSV that bench-stats wrote, by relative path"""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*.csv"))
        if path.name != "measurements.csv"
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return build_tree(tmp_path / "results")


def test_every_output_is_written(tree: Path):
    run(tree, jobs=1)

    scenario_dir = tree / SCENARIO
    for jdk in JDKS:
        for implementation in IMPLEMENTATIONS:
            implementation_dir = scenario_dir / jdk / implementation
            assert (implementation_dir / "summary.csv").exists()
            for benchmark in BENCHMARKS:
                assert (implementation_dir / benchmark / "summary.csv").exists()

        variant_dir = scenario_dir / jdk / "streams"
        forest_plot = "energy_median_difference_percent_forest_plot.pgf"
        assert (variant_dir / "comparison.csv").exists()
        assert (variant_dir / forest_plot).exists()

    assert (tree / "energy_time_correlation.csv").exists()
    for stem in (
        "streams_energy_median_difference_percent_heatmap",
        "streams_energy_time_scatter_plot",
        "streams_energy_trend_plot",
    ):
        assert (scenario_dir / f"{stem}.pgf").exists()


@pytest.mark.parametrize(
    ("header", "extra"),
    [(SWAPPED_HEADER, {}), (HEADER + ["Extra"], {"Extra": 0})],
    ids=["reordered", "extra-column"],
)
def test_measurements_with_unexpected_columns_are_refused(
    tmp_path: Path, caplog, header: list[str], extra: dict[str, object]
):
    path = tmp_path / "measurements.csv"
    rows = measurement_rows(np.random.default_rng(0), "alpha", 1.0)
    write_measurements(path, [row | extra for row in rows], header)

    with caplog.at_level(logging.ERROR):
        assert load_and_clean_measurements(path) is None

    assert "Unexpected columns" in caplog.text


def test_a_benchmark_with_unexpected_columns_is_left_out(tree: Path, caplog):
    jdk_dir = tree / SCENARIO / "temurin-17"
    rows = measurement_rows(np.random.default_rng(1), "alpha", 1.0)
    write_measurements(jdk_dir / "loops/alpha/measurements.csv", rows, SWAPPED_HEADER)

    with caplog.at_level(logging.ERROR):
        run(tree, jobs=1)

    comparison = pl.read_csv(jdk_dir / "streams" / "comparison.csv")
    assert "Unexpected columns" in caplog.text
    assert not (jdk_dir / "loops/alpha/summary.csv").exists()
    assert sorted(comparison.get_column("Program").unique()) == ["beta", "gamma"]


def test_a_second_run_writes_the_same_results(tree: Path):
    run(tree, jobs=1)
    first = csv_outputs(tree)

    run(tree, jobs=1)
    assert csv_outputs(tree) == first
