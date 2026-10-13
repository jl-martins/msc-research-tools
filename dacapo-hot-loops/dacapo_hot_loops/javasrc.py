"""Java source module"""

import functools
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple, Self

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from dacapo_hot_loops import HotLoopsError

# Type declarations whose name is a segment of a binary class name
NAMED_TYPE_NODES = frozenset(
    {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "annotation_type_declaration",
    }
)
CONSTRUCTOR_NODES = frozenset(
    {"constructor_declaration", "compact_constructor_declaration"}
)
LOOP_NODES = frozenset(
    {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"}
)
# A class body of one of these is anonymous, so its binary name holds a number
ANONYMOUS_BODY_PARENTS = frozenset({"object_creation_expression", "enum_constant"})
# Inside one of these, a type declaration is local rather than a member
CODE_BODY_NODES = frozenset({"block", "constructor_body", "switch_block"})
CONSTRUCTOR_NAME = "<init>"
CLASS_INITIALISER_NAME = "<clinit>"


class LineRange(NamedTuple):
    """Inclusive range of 1-based source lines"""

    start: int
    end: int

    def contains_line(self, line: int) -> bool:
        return self.start <= line <= self.end

    def contains(self, other: Self) -> bool:
        return self.start <= other.start and other.end <= self.end


class PendingNode(NamedTuple):
    """Node left to visit, with the context that its children inherit"""

    node: Node
    type_path: str | None
    in_code: bool


class Declaration(NamedTuple):
    """Method, constructor or initialiser of a type"""

    # None when the binary class name holds an anonymous or local segment
    type_path: str | None
    name: str
    line_range: LineRange


class JavaFile(NamedTuple):
    """Declarations, loops and syntax errors of one parsed source file"""

    declarations: tuple[Declaration, ...]
    loops: tuple[LineRange, ...]
    error_lines: tuple[int, ...]

    def declarations_of(
        self,
        type_path: str,
        name: str,
        sampled_lines: Collection[int],
    ) -> list[LineRange]:
        """Ranges of the declarations of one method of one type"""
        if has_anonymous_segment(type_path):
            # javac numbers anonymous and local classes, so the name must do
            return [
                declaration.line_range
                for declaration in self.declarations
                if declaration.name == name
                and any(map(declaration.line_range.contains_line, sampled_lines))
            ]
        else:
            return [
                declaration.line_range
                for declaration in self.declarations
                if declaration.type_path == type_path and declaration.name == name
            ]

    def innermost_declaration(self, lines: LineRange) -> Declaration | None:
        """The innermost declaration whose range contains a range"""
        containing = [d for d in self.declarations if d.line_range.contains(lines)]
        if containing:
            return min(
                containing,
                key=lambda d: (-d.line_range.start, d.line_range.end),
            )
        else:
            return None


def has_anonymous_segment(type_path: str) -> bool:
    """Whether a binary class name holds an anonymous or local class segment"""
    return any(segment[:1].isdigit() for segment in type_path.split("$"))


@functools.cache
def java_parser() -> Parser:
    """The shared Java parser, built on first use"""
    return Parser(Language(tree_sitter_java.language()))


def node_range(node: Node) -> LineRange:
    return LineRange(node.start_point[0] + 1, node.end_point[0] + 1)


def field_text(node: Node, field: str) -> str | None:
    child = node.child_by_field_name(field)
    if child is None or child.text is None:
        return None
    else:
        return child.text.decode("utf-8", errors="replace")


def is_static_with_value(node: Node) -> bool:
    """Whether a field declaration is static and holds an initialiser"""
    modifiers = next((c for c in node.children if c.type == "modifiers"), None)
    is_static = modifiers is not None and any(
        c.type == "static" for c in modifiers.children
    )
    declarators = (c for c in node.children if c.type == "variable_declarator")
    has_value = any(d.child_by_field_name("value") is not None for d in declarators)

    return is_static and has_value


def declaration_name(node: Node) -> str | None:
    """The name a method key uses for a declaration, or None for other nodes"""
    match node.type:
        case "method_declaration":
            return field_text(node, "name")
        case node_type if node_type in CONSTRUCTOR_NODES:
            return CONSTRUCTOR_NAME
        case "static_initializer":
            return CLASS_INITIALISER_NAME
        case "field_declaration" if is_static_with_value(node):
            return CLASS_INITIALISER_NAME
        case _:
            return None


def parse_java(source: str) -> JavaFile:
    """Find every declaration, loop and syntax error of a source file"""
    tree = java_parser().parse(source.encode("utf-8"))
    declarations: list[Declaration] = []
    loops: list[LineRange] = []
    error_lines: set[int] = set()

    # An explicit stack keeps a deeply nested file within the recursion limit
    pending = [PendingNode(tree.root_node, "", False)]
    while pending:
        node, type_path, in_code = pending.pop()
        if node.type == "ERROR" or node.is_missing:
            error_lines.add(node.start_point[0] + 1)

        if node.type in LOOP_NODES:
            loops.append(node_range(node))

        name = declaration_name(node)
        if name is not None:
            declarations.append(Declaration(type_path, name, node_range(node)))

        child_path, child_in_code = type_path, in_code
        if node.type in NAMED_TYPE_NODES:
            type_name = field_text(node, "name")
            # A local type's binary name holds a number before its own name
            is_exact = type_path is not None and not in_code and type_name is not None
            if is_exact:
                child_path = f"{type_path}${type_name}" if type_path else type_name
            else:
                child_path = None

            child_in_code = False
        elif node.type in CODE_BODY_NODES:
            child_in_code = True

        for child in node.children:
            if child.type == "class_body" and node.type in ANONYMOUS_BODY_PARENTS:
                pending.append(PendingNode(child, None, False))
            else:
                pending.append(PendingNode(child, child_path, child_in_code))

    declarations.sort(key=lambda d: (d.line_range, d.name, d.type_path or ""))
    loops.sort()

    return JavaFile(tuple(declarations), tuple(loops), tuple(sorted(error_lines)))


class JavaSources:
    """Cache of the parsed source files under the benchmarks directory"""

    def __init__(self, benchmarks_dir: Path) -> None:
        self.benchmarks_dir = benchmarks_dir
        self.parsed_files: dict[Path, JavaFile] = {}

    def file(self, path: Path) -> JavaFile:
        """Parse a Java source file relative to the benchmarks directory"""
        parsed = self.parsed_files.get(path)
        if parsed is None:
            full_path = self.benchmarks_dir / path
            try:
                source = full_path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError as e:
                raise HotLoopsError(f"Could not read '{full_path}': {e}") from e

            parsed = parse_java(source)
            self.parsed_files[path] = parsed

        return parsed
