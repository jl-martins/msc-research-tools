"""Tests of Java source parsing"""

import pytest
from conftest import DATA_DIR

from dacapo_hot_loops.javasrc import JavaFile, LineRange, parse_java

# Every range below was counted by hand from its fixture


def parse_fixture(name: str) -> JavaFile:
    return parse_java((DATA_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def outer() -> JavaFile:
    return parse_fixture("Outer.java")


@pytest.fixture(scope="module")
def broken() -> JavaFile:
    return parse_fixture("Broken.java")


def test_method_resolves_in_its_type(outer):
    assert outer.declarations_of("Outer", "work", ()) == [
        LineRange(14, 35),
        LineRange(37, 39),
    ]


def test_constructor_resolves_as_init(outer):
    assert outer.declarations_of("Outer", "<init>", ()) == [LineRange(10, 12)]


def test_static_initialiser_and_field_resolve_as_clinit(outer):
    assert outer.declarations_of("Outer", "<clinit>", ()) == [
        LineRange(4, 4),
        LineRange(6, 8),
    ]


def test_nested_type_has_its_own_path(outer):
    assert outer.declarations_of("Outer$Inner", "spin", ()) == [LineRange(42, 46)]


def test_nested_method_is_not_in_outer_path(outer):
    assert outer.declarations_of("Outer", "spin", ()) == []


def test_missing_method_is_unresolved(outer):
    assert outer.declarations_of("Outer", "absent", ()) == []


def test_every_loop_kind_is_found(outer):
    assert outer.loops == (
        LineRange(11, 11),  # braceless `for` in the constructor
        LineRange(15, 19),  # `for` with a block body
        LineRange(16, 18),  # `while` nested in the `for` above
        LineRange(20, 22),  # `do`
        LineRange(24, 26),  # `for` inside a lambda
        LineRange(30, 32),  # `for` inside an anonymous class
        LineRange(43, 45),  # `while` in the nested type
    )


def test_nested_loop_lies_inside_its_outer_loop():
    assert LineRange(15, 19).contains(LineRange(16, 18))


def test_loop_outside_a_range_is_not_contained():
    assert not LineRange(15, 19).contains(LineRange(20, 22))


def test_lambda_loop_belongs_to_its_method(outer):
    work = outer.declarations_of("Outer", "work", ())[0]
    assert work.contains(LineRange(24, 26))


def test_anonymous_class_matches_by_name(outer):
    # javac names the class `Outer$1`, so a digit segment falls back to the name
    assert outer.declarations_of("Outer$1", "run", [31]) == [LineRange(29, 33)]


def test_anonymous_class_needs_a_sampled_line(outer):
    assert outer.declarations_of("Outer$1", "run", [15]) == []


def test_innermost_declaration_wins(outer):
    spin = outer.innermost_declaration(LineRange(44, 44))
    assert spin is not None and spin.name == "spin"


def test_lambda_line_resolves_to_enclosing_method(outer):
    enclosing = outer.innermost_declaration(LineRange(25, 25))
    assert enclosing is not None and enclosing.name == "work"


def test_line_outside_any_declaration_has_none(outer):
    assert outer.innermost_declaration(LineRange(2, 2)) is None


def test_clean_file_reports_no_errors(outer):
    assert outer.error_lines == ()


def test_syntax_error_lines_are_listed(broken):
    assert broken.error_lines != ()


def test_broken_file_keeps_parsed_loops(broken):
    assert LineRange(5, 7) in broken.loops


def test_broken_file_keeps_parsed_declarations(broken):
    assert broken.declarations_of("Broken", "loop", ()) == [LineRange(4, 8)]
