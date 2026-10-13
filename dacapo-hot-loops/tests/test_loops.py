"""Tests of loop charging"""

import shutil
from pathlib import Path

import pytest
from conftest import DATA_DIR

from dacapo_hot_loops.collapsed import parse_collapsed
from dacapo_hot_loops.javasrc import JavaSources, LineRange, LoopType
from dacapo_hot_loops.loops import (
    LoopOrigin,
    caller_loops,
    charge_loops,
    declarations_of_method,
    hot_loops,
    method_loops,
    unique_loops,
)
from dacapo_hot_loops.methods import charge_stacks, hot_methods, main_caller

EXCLUDED_PREFIXES = ("java.", "javax.", "jdk.", "sun.", "com.sun.", "org.dacapo.")
OUTER_FILE = Path("Outer.java")
INDEX = {"p.Outer": OUTER_FILE}
# Ranges counted by hand from Outer.java
OUTER_FOR = LineRange(15, 19)
INNER_WHILE = LineRange(16, 18)
WORK_LOOPS = (OUTER_FOR, INNER_WHILE, LineRange(20, 22), LineRange(24, 26))


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    benchmarks_dir = tmp_path_factory.mktemp("benchmarks")
    shutil.copy(DATA_DIR / "Outer.java", benchmarks_dir / OUTER_FILE)
    return JavaSources(benchmarks_dir)


def charge(text, sources):
    return charge_stacks(parse_collapsed(text, EXCLUDED_PREFIXES), INDEX, sources)


def loops_of(text, sources):
    """The chargeable loops of every hot method of a collapsed fixture"""
    benchmark_charges = charge(text, sources)
    loops = []
    for method in hot_methods(benchmark_charges, 0.9):
        declarations = declarations_of_method(method, sources)
        loops.extend(method_loops(method, declarations, sources))
        caller = main_caller(benchmark_charges, method.method_key)
        if declarations and caller is not None:
            loops.extend(caller_loops(caller, sources))

    return benchmark_charges, unique_loops(loops)


def test_method_loops_include_nested_and_lambda_loops(sources):
    _, loops = loops_of("p/Outer.work:16_[j] 10\n", sources)
    assert tuple(loop.line_range for loop in loops) == WORK_LOOPS


def test_method_loops_have_their_statement_types(sources):
    _, loops = loops_of("p/Outer.work:16_[j] 10\n", sources)
    assert [loop.type for loop in loops] == [
        LoopType.BASIC_FOR,
        LoopType.WHILE,
        LoopType.DO_WHILE,
        LoopType.ENHANCED_FOR,
    ]


def test_unresolved_method_contributes_no_loops(sources):
    _, loops = loops_of("p/Outer.absent:1_[j] 10\n", sources)
    assert loops == []


def test_loop_id_names_the_file_and_range(sources):
    _, loops = loops_of("p/Outer.work:16_[j] 10\n", sources)
    assert loops[0].loop_id == "Outer.java:15-19"


def test_innermost_loop_takes_the_samples(sources):
    # Line 17 lies in both the `for` and the `while` nested inside it
    benchmark_charges, loops = loops_of("p/Outer.work:17_[j] 10\n", sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert charged.loop_counts == {"Outer.java:16-18": 10}


def test_outer_loop_takes_a_line_outside_the_inner_one(sources):
    benchmark_charges, loops = loops_of("p/Outer.work:15_[j] 10\n", sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert charged.loop_counts == {"Outer.java:15-19": 10}


def test_topmost_frame_in_a_loop_wins(sources):
    text = "p/Outer.work:17_[j];p/Outer.work:21_[j] 10\n"
    benchmark_charges, loops = loops_of(text, sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert charged.loop_counts == {"Outer.java:20-22": 10}


def test_line_in_no_loop_falls_outside(sources):
    # Line 14 is the signature of `work`, which no loop holds
    benchmark_charges, loops = loops_of("p/Outer.work:14_[j] 10\n", sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert (charged.loop_counts, charged.outside_count) == ({}, 10)


def test_inherited_method_does_not_match_a_loop(sources):
    # `inherited` is declared in a superclass, so line 17 is another file's
    text = "p/Outer.work:21_[j] 100\np/Outer.inherited:17_[j] 50\n"
    benchmark_charges, loops = loops_of(text, sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert (charged.loop_counts, charged.outside_count) == (
        {"Outer.java:20-22": 100},
        50,
    )


def test_frame_without_a_line_falls_outside(sources):
    benchmark_charges, loops = loops_of("p/Outer.work_[j] 10\n", sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert (charged.loop_counts, charged.outside_count) == ({}, 10)


def test_caller_loop_holding_the_call_site_is_chargeable(sources):
    # `spin` is called from line 17, inside the `while` of `work`
    text = "p/Outer.work:17_[j];p/Outer$Inner.spin:44_[j] 10\n"
    _, loops = loops_of(text, sources)
    caller_origin = [loop for loop in loops if loop.origin == LoopOrigin.CALLER]
    assert [loop.line_range for loop in caller_origin] == [OUTER_FOR, INNER_WHILE]


def test_method_origin_wins_a_shared_loop(sources):
    text = "p/Outer.work:17_[j];p/Outer.work:25_[j] 10\n"
    _, loops = loops_of(text, sources)
    origins = {loop.loop_id: loop.origin for loop in loops}
    assert origins["Outer.java:16-18"] == LoopOrigin.HOT_METHOD


def test_loop_below_threshold_is_not_hot(sources):
    text = "p/Outer.work:17_[j] 1\np/Outer.work:21_[j] 1999\n"
    benchmark_charges, loops = loops_of(text, sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    hot = hot_loops(loops, charged, benchmark_charges.charged_sample_count, 0.001)
    assert [loop.loop_id for loop in hot] == ["Outer.java:20-22"]


def test_loop_on_threshold_is_hot(sources):
    text = "p/Outer.work:17_[j] 2\np/Outer.work:21_[j] 1998\n"
    benchmark_charges, loops = loops_of(text, sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    hot = hot_loops(loops, charged, benchmark_charges.charged_sample_count, 0.001)
    assert sorted(loop.loop_id for loop in hot) == [
        "Outer.java:16-18",
        "Outer.java:20-22",
    ]


def test_no_charged_samples_means_no_hot_loops(sources):
    benchmark_charges, loops = loops_of("p/Outer.work:17_[j] 10\n", sources)
    charged = charge_loops(benchmark_charges, loops, sources)
    assert hot_loops(loops, charged, 0, 0.001) == []
