"""Tests of method charging"""

import shutil
from pathlib import Path

import pytest
from conftest import DATA_DIR

from dacapo_hot_loops.collapsed import parse_collapsed, parse_frame
from dacapo_hot_loops.javasrc import JavaSources
from dacapo_hot_loops.methods import (
    charge_stacks,
    hot_methods,
    main_caller,
    method_key_of,
)

EXCLUDED_PREFIXES = ("java.", "javax.", "jdk.", "sun.", "com.sun.", "org.dacapo.")
OUTER_FILE = Path("Outer.java")
INDEX = {"p.Outer": OUTER_FILE}
# Line 25 of Outer.java lies in the lambda inside `work`
LAMBDA_LINE = 25


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    benchmarks_dir = tmp_path_factory.mktemp("benchmarks")
    shutil.copy(DATA_DIR / "Outer.java", benchmarks_dir / OUTER_FILE)
    return JavaSources(benchmarks_dir)


def charge(text, sources):
    return charge_stacks(parse_collapsed(text, EXCLUDED_PREFIXES), INDEX, sources)


def key_of(raw, sources):
    return method_key_of(parse_frame(raw, EXCLUDED_PREFIXES), OUTER_FILE, sources)


def test_overloads_share_their_key(sources):
    # A frame carries no signature, so both `work` declarations merge
    keys = {key_of(f"p/Outer.work:{line}_[j]", sources) for line in (16, 38)}
    assert keys == {"p.Outer.work"}


def test_lambda_with_a_line_takes_its_enclosing_method(sources):
    raw = f"p/Outer.lambda$work$0:{LAMBDA_LINE}_[j]"
    assert key_of(raw, sources) == "p.Outer.work"


def test_lambda_without_a_line_takes_its_owner(sources):
    assert key_of("p/Outer.lambda$work$0_[j]", sources) == "p.Outer.work"


def test_lambda_owned_by_new_takes_the_constructor(sources):
    assert key_of("p/Outer.lambda$new$0_[j]", sources) == "p.Outer.<init>"


def test_lambda_owned_by_static_takes_the_class_initialiser(sources):
    assert key_of("p/Outer.lambda$static$0_[j]", sources) == "p.Outer.<clinit>"


def test_nested_lambda_keeps_its_raw_name(sources):
    # `null` names no declaration, so such a method is always unresolved
    assert key_of("p/Outer.lambda$null$1_[j]", sources) == "p.Outer.lambda$null$1"


def test_topmost_source_frame_takes_the_samples(sources):
    text = "Harness.main;p/Outer.work:16_[j];java/util/ArrayList.add:1_[j] 9\n"
    charges = charge(text, sources)
    charged = charges.stacks[0].charged
    assert (charged.method_key, charged.line, charges.charged_sample_count) == (
        "p.Outer.work",
        16,
        9,
    )


def test_deeper_source_frame_wins(sources):
    text = "p/Outer.work:16_[j];p/Outer$Inner.spin:44_[j] 4\n"
    assert charge(text, sources).stacks[0].charged.method_key == "p.Outer$Inner.spin"


def test_stack_without_application_frames_is_excluded(sources):
    text = "Harness.main;java/util/ArrayList.add:1_[j] 5\n"
    charges = charge(text, sources)
    assert (charges.no_app_sample_count, charges.charged_sample_count) == (5, 0)


def test_stack_without_source_is_excluded(sources):
    text = "Harness.main;org/springframework/Bean.get:2_[j] 7\n"
    charges = charge(text, sources)
    assert (charges.no_source_sample_count, charges.charged_sample_count) == (7, 0)


def test_unresolved_class_takes_the_whole_stack(sources):
    text = "org/springframework/Outer.a:1_[j];org/springframework/Bean.get:2_[j] 7\n"
    assert charge(text, sources).unresolved_classes == {"org.springframework.Bean": 7}


def test_buckets_add_up_to_the_total(sources):
    text = (
        "p/Outer.work:16_[j] 9\n"
        "Harness.main;java/util/ArrayList.add:1_[j] 5\n"
        "Harness.main;org/springframework/Bean.get:2_[j] 7\n"
    )
    charges = charge(text, sources)
    buckets = (
        charges.no_app_sample_count
        + charges.no_source_sample_count
        + charges.charged_sample_count
    )
    assert (buckets, charges.total_sample_count) == (21, 21)


def test_cutoff_takes_the_smallest_set(sources):
    text = (
        "p/Outer.work:16_[j] 50\n"
        "p/Outer$Inner.spin:44_[j] 30\n"
        "p/Outer.helperA:1_[j] 15\n"
        "p/Outer.helperB:1_[j] 5\n"
    )
    hot = hot_methods(charge(text, sources), 0.9)
    assert [method.method_key for method in hot] == [
        "p.Outer.work",
        "p.Outer$Inner.spin",
        "p.Outer.helperA",
    ]


def test_cutoff_keeps_methods_tied_with_the_last(sources):
    # The cutoff falls at 90 samples, where helperA and helperB tie on 10
    text = (
        "p/Outer.work:16_[j] 50\n"
        "p/Outer$Inner.spin:44_[j] 30\n"
        "p/Outer.helperA:1_[j] 10\n"
        "p/Outer.helperB:1_[j] 10\n"
    )
    hot = hot_methods(charge(text, sources), 0.9)
    assert [method.method_key for method in hot] == [
        "p.Outer.work",
        "p.Outer$Inner.spin",
        "p.Outer.helperA",
        "p.Outer.helperB",
    ]


def test_ties_rank_by_method_key(sources):
    text = "p/Outer.b:1_[j] 10\np/Outer.a:1_[j] 10\n"
    hot = hot_methods(charge(text, sources), 0.9)
    assert [method.method_key for method in hot] == ["p.Outer.a", "p.Outer.b"]


def test_no_charged_samples_means_no_hot_methods(sources):
    text = "Harness.main;java/util/ArrayList.add:1_[j] 5\n"
    assert hot_methods(charge(text, sources), 0.9) == []


def test_main_caller_is_the_most_sampled_call_site(sources):
    text = (
        "p/Outer.callerA:7_[j];p/Outer.work:16_[j] 4\n"
        "p/Outer.callerB:5_[j];p/Outer.work:16_[j] 9\n"
    )
    caller = main_caller(charge(text, sources), "p.Outer.work")
    assert caller is not None
    assert (caller.method_key, caller.line, caller.sample_count) == (
        "p.Outer.callerB",
        5,
        9,
    )


def test_caller_tie_breaks_by_method_key(sources):
    text = (
        "p/Outer.callerB:5_[j];p/Outer.work:16_[j] 4\n"
        "p/Outer.callerA:7_[j];p/Outer.work:16_[j] 4\n"
    )
    caller = main_caller(charge(text, sources), "p.Outer.work")
    assert caller is not None and caller.method_key == "p.Outer.callerA"


def test_caller_tie_breaks_by_line(sources):
    text = (
        "p/Outer.callerA:9_[j];p/Outer.work:16_[j] 4\n"
        "p/Outer.callerA:3_[j];p/Outer.work:16_[j] 4\n"
    )
    caller = main_caller(charge(text, sources), "p.Outer.work")
    assert caller is not None and caller.line == 3


def test_method_without_a_caller_has_none(sources):
    text = "p/Outer.work:16_[j] 4\n"
    assert main_caller(charge(text, sources), "p.Outer.work") is None
