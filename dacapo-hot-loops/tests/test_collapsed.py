"""Tests of collapsed stack parsing"""

from pathlib import Path

import pytest

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.collapsed import (
    Frame,
    Stack,
    parse_collapsed,
    parse_frame,
    parse_line,
    source_file,
)

EXCLUDED_PREFIXES = ("java.", "javax.", "jdk.", "sun.", "com.sun.", "org.dacapo.")
LUCENE_FRAME = "org/apache/lucene/index/Foo$Bar.next:123_[j]"
NATIVE_FRAME = "slow_subtype_check Runtime1 stub"
INDEX = {"org.apache.lucene.index.Foo": Path("libs/lucene/build/Foo.java")}


def frame_of(raw: str) -> Frame:
    return parse_frame(raw, EXCLUDED_PREFIXES)


def test_count_follows_last_space():
    native = Frame(NATIVE_FRAME, "", NATIVE_FRAME, None, False)
    assert parse_line(f"{NATIVE_FRAME} 3", EXCLUDED_PREFIXES) == Stack((native,), 3)


def test_frames_split_on_semicolons():
    stack = parse_line(f"Harness.main;{LUCENE_FRAME} 7", EXCLUDED_PREFIXES)
    assert stack.count == 7
    assert [frame.method for frame in stack.frames] == ["main", "next"]


def test_line_without_count_fails():
    with pytest.raises(HotLoopsError, match="Malformed"):
        parse_line("Harness.main", EXCLUDED_PREFIXES)


def test_slashes_become_dots():
    expected = Frame(
        raw=LUCENE_FRAME,
        class_name="org.apache.lucene.index.Foo$Bar",
        method="next",
        line=123,
        is_application=True,
    )
    assert frame_of(LUCENE_FRAME) == expected


@pytest.mark.parametrize("suffix", ["_[j]", "_[i]", "_[k]", "_[0]", "_[1]", ""])
def test_every_suffix_is_stripped(suffix):
    raw = f"org/example/A.run:5{suffix}"
    assert frame_of(raw) == Frame(raw, "org.example.A", "run", 5, True)


def test_frame_without_line_has_none():
    raw = "org/example/A.run_[j]"
    assert frame_of(raw) == Frame(raw, "org.example.A", "run", None, True)


def test_top_level_class_drops_nested_part():
    assert frame_of(LUCENE_FRAME).top_level_class == "org.apache.lucene.index.Foo"


def test_native_frame_is_not_application():
    assert not frame_of("G1RemSet::scan_heap_roots").is_application


def test_harness_frame_is_not_application():
    assert not frame_of("Harness.main").is_application


@pytest.mark.parametrize(
    "raw",
    [
        "java/util/ArrayList.add:100_[j]",
        "javax/xml/Parser.parse:1_[j]",
        "jdk/internal/Unsafe.park:2_[j]",
        "sun/nio/ch/IOUtil.read:3_[j]",
        "com/sun/crypto/Cipher.init:4_[j]",
        "org/dacapo/harness/Callback.start:5_[j]",
    ],
)
def test_excluded_prefix_is_not_application(raw):
    assert not frame_of(raw).is_application


@pytest.mark.parametrize(
    "raw",
    [
        "org/example/A$$Lambda$17.run:1_[j]",
        "java/lang/invoke/LambdaForm$MH.invoke:2_[j]",
    ],
)
def test_synthetic_class_is_not_application(raw):
    assert not frame_of(raw).is_application


def test_benchmark_frame_is_application():
    assert frame_of(LUCENE_FRAME).is_application


def test_source_frame_resolves_to_its_file():
    lucene_file = INDEX["org.apache.lucene.index.Foo"]
    assert source_file(frame_of(LUCENE_FRAME), INDEX) == lucene_file


def test_unindexed_application_frame_has_no_source():
    assert source_file(frame_of("org/springframework/Bean.get:1_[j]"), INDEX) is None


def test_non_application_frame_has_no_source():
    assert source_file(frame_of("Harness.main"), INDEX) is None


def test_blank_lines_are_skipped():
    text = f"{LUCENE_FRAME} 2\n\n   \nHarness.main 1\n"
    assert [stack.count for stack in parse_collapsed(text, EXCLUDED_PREFIXES)] == [2, 1]
