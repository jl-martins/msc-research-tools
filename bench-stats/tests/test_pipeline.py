"""End-to-end tests of bench-stats on a small tree of measurements"""

import logging
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import plots
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


def write_invalid_measurements(path: Path) -> None:
    """Write measurements to `path` under a header that bench-stats refuses"""
    rows = measurement_rows(np.random.default_rng(1), path.parent.name, 1.0)
    write_measurements(path, rows, SWAPPED_HEADER)


def csv_outputs(root: Path) -> dict[str, bytes]:
    """Return the content of every CSV that bench-stats wrote, by relative path"""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*.csv"))
        if path.name != "measurements.csv"
    }


# Resamples of the bootstraps of a test run. These tests check which files a run reads
# and writes, not how accurate an interval is, and every resample costs time.
TEST_RESAMPLES = 199


# Plots that a test does not read. Drawing one costs far more than the statistics
# behind it, since a PGF file lays out every piece of text it holds.
UNREAD_PLOTS = (
    "generate_forest_plot",
    "generate_effect_size_heatmaps",
    "generate_energy_time_scatter_plot",
    "generate_trend_plot",
)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch) -> Path:
    for plot in UNREAD_PLOTS:
        monkeypatch.setattr(plots, plot, lambda *args, **kwargs: None)

    return build_tree(tmp_path / "results")


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory) -> Path:
    """Return a tree that one run has processed, shared by the tests that only read it"""
    tree = build_tree(tmp_path_factory.mktemp("results"))
    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    return tree


def test_every_output_is_written(completed_run: Path):
    scenario_dir = completed_run / SCENARIO
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

    assert (completed_run / "energy_time_correlation.csv").exists()
    for stem in (
        "streams_energy_median_difference_percent_heatmap",
        "streams_energy_time_scatter_plot",
        "streams_energy_median_difference_percent_trend_plot",
    ):
        assert (scenario_dir / f"{stem}.pgf").exists()


def test_every_plot_fits_the_page(completed_run: Path):
    plot_paths = sorted(completed_run.rglob("*.pgf"))
    assert plot_paths

    for path in plot_paths:
        size = re.search(r"\\pgfqpoint\{([\d.]+)in\}\{([\d.]+)in\}", path.read_text())
        assert size is not None, path
        width, height = map(float, size.groups())
        assert width <= plots.TEXT_WIDTH_INCHES, path
        assert height <= plots.MAX_FIGURE_HEIGHT_INCHES, path


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
    write_invalid_measurements(jdk_dir / "loops/alpha/measurements.csv")

    with caplog.at_level(logging.ERROR):
        run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)

    comparison = pl.read_csv(jdk_dir / "streams" / "comparison.csv")
    assert "Unexpected columns" in caplog.text
    assert not (jdk_dir / "loops/alpha/summary.csv").exists()
    assert sorted(comparison.get_column("Program").unique()) == ["beta", "gamma"]


def test_summaries_and_comparisons_report_the_same_iqr(completed_run: Path):
    for jdk in JDKS:
        jdk_dir = completed_run / SCENARIO / jdk
        comparison = pl.read_csv(jdk_dir / "streams" / "comparison.csv")
        for row in comparison.iter_rows(named=True):
            iqr_column = f"{row['Metric']}_IQR"
            for implementation, compared in (
                ("loops", "Loops_IQR"),
                ("streams", "Variant_IQR"),
            ):
                summary_path = jdk_dir / implementation / row["Program"] / "summary.csv"
                summary = pl.read_csv(summary_path)
                assert summary.get_column(iqr_column)[0] == pytest.approx(row[compared])


def test_stale_comparison_is_removed(tree: Path):
    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    streams_dir = tree / SCENARIO / "temurin-17" / "streams"
    for benchmark in BENCHMARKS:
        write_invalid_measurements(streams_dir / benchmark / "measurements.csv")

    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)

    correlation = pl.read_csv(tree / "energy_time_correlation.csv")
    assert not (streams_dir / "comparison.csv").exists()
    assert correlation.get_column("N").to_list() == [3, 3]


def test_stale_summary_is_removed(tree: Path):
    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    loops_dir = tree / SCENARIO / "temurin-17" / "loops"
    write_invalid_measurements(loops_dir / "alpha" / "measurements.csv")

    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)

    combined = pl.read_csv(loops_dir / "summary.csv")
    assert not (loops_dir / "alpha" / "summary.csv").exists()
    assert combined.get_column("Program").to_list() == ["beta", "gamma"]


def test_stale_correlation_table_is_removed(tree: Path):
    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    for jdk in JDKS:
        for benchmark in BENCHMARKS:
            streams_dir = tree / SCENARIO / jdk / "streams"
            write_invalid_measurements(streams_dir / benchmark / "measurements.csv")

    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)

    assert not (tree / "energy_time_correlation.csv").exists()


def test_run_survives_a_failed_job(tree: Path, caplog, monkeypatch):
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no forest plot today")

    monkeypatch.setattr(plots, "generate_forest_plot", fail)

    with caplog.at_level(logging.ERROR):
        n_failures = run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)

    # One forest plot job per JDK, and the rest of the run goes on
    assert n_failures == len(JDKS)
    assert "Failed: PackageAndDRAM forest plot" in caplog.text
    assert (tree / "energy_time_correlation.csv").exists()
    for jdk in JDKS:
        assert (tree / SCENARIO / jdk / "streams" / "comparison.csv").exists()


def test_two_concurrent_jobs_write_same_results_as_one(tree: Path, tmp_path: Path):
    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    expected = csv_outputs(tree)
    other_tree = build_tree(tmp_path / "parallel")

    run(other_tree, n_jobs=2, n_resamples=TEST_RESAMPLES)

    assert csv_outputs(other_tree) == expected


def test_a_second_run_writes_the_same_results(tree: Path):
    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    first = csv_outputs(tree)

    run(tree, n_jobs=1, n_resamples=TEST_RESAMPLES)
    assert csv_outputs(tree) == first
