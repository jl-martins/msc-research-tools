"""Loop charging module"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from dacapo_hot_loops.javasrc import JavaSources, LineRange, LoopType
from dacapo_hot_loops.methods import (
    BenchmarkCharges,
    CallerCharge,
    ChargedStack,
    MethodCharge,
    ResolvedFrame,
    split_method_key,
    type_path_of,
)


class LoopOrigin(StrEnum):
    """Where a chargeable loop came from"""

    HOT_METHOD = "hot-method"
    CALLER = "caller"


class ChargeableLoop(NamedTuple):
    """Loop that stacks can be charged to"""

    loop_id: str
    file: Path
    line_range: LineRange
    type: LoopType
    enclosing_method_key: str
    origin: LoopOrigin


class LoopCharges(NamedTuple):
    """Samples of each chargeable loop and those inside no loop"""

    loop_counts: dict[str, int]
    outside_count: int


def loop_id_of(file: Path, lines: LineRange) -> str:
    return f"{file}:{lines.start}-{lines.end}"


def declarations_of_method(
    charge: MethodCharge,
    sources: JavaSources,
) -> list[LineRange]:
    """Ranges of a hot method's declarations, empty when it is unresolved"""
    class_name, method = split_method_key(charge.method_key)
    java_file = sources.file(charge.file)
    return java_file.declarations_of(type_path_of(class_name), method, charge.lines)


def method_loops(
    charge: MethodCharge,
    declarations: Iterable[LineRange],
    sources: JavaSources,
) -> list[ChargeableLoop]:
    """Every loop whose innermost declaration belongs to the hot method"""
    java_file = sources.file(charge.file)
    ranges = set(declarations)

    # No declaration is named after a lambda. Its loops are assigned to the
    # enclosing method so that they can be located in the source code.
    return [
        ChargeableLoop(
            loop_id=loop_id_of(charge.file, loop.line_range),
            file=charge.file,
            line_range=loop.line_range,
            type=loop.type,
            enclosing_method_key=charge.method_key,
            origin=LoopOrigin.HOT_METHOD,
        )
        for loop in java_file.loops
        if (innermost := java_file.innermost_declaration(loop.line_range)) is not None
        and innermost.line_range in ranges
    ]


def caller_loops(caller: CallerCharge, sources: JavaSources) -> list[ChargeableLoop]:
    """Every loop in the main caller whose range holds the call site"""
    if caller.line is None:
        return []

    java_file = sources.file(caller.file)

    return [
        ChargeableLoop(
            loop_id=loop_id_of(caller.file, loop.line_range),
            file=caller.file,
            line_range=loop.line_range,
            type=loop.type,
            enclosing_method_key=caller.method_key,
            origin=LoopOrigin.CALLER,
        )
        for loop in java_file.loops
        if loop.line_range.contains_line(caller.line)
    ]


def unique_loops(loops: Iterable[ChargeableLoop]) -> list[ChargeableLoop]:
    """One entry per loop, keeping the origin that claimed it first"""
    unique: dict[str, ChargeableLoop] = {}
    for loop in loops:
        unique.setdefault(loop.loop_id, loop)

    return list(unique.values())


def declares_method(frame: ResolvedFrame, sources: JavaSources) -> bool:
    """Whether a frame's own method is declared at its line in its file"""
    assert frame.line is not None
    lines = LineRange(frame.line, frame.line)
    declaration = sources.file(frame.file).innermost_declaration(lines)
    _, method = split_method_key(frame.method_key)
    return declaration is not None and declaration.name == method


def innermost_loop(
    stack: ChargedStack,
    loops_by_file: Mapping[Path, list[ChargeableLoop]],
    sources: JavaSources,
) -> ChargeableLoop | None:
    """The innermost chargeable loop that holds a frame of the stack. Frames are
    tried from the leaf toward the root."""
    for frame in reversed(stack.frames):
        # A frame without a line falls inside no loop. A line number only locates
        # code in the file that declares the method. An inherited or synthetic
        # method is declared elsewhere, so its line could match a stray loop.
        if frame.line is not None and declares_method(frame, sources):
            holding = [
                loop
                for loop in loops_by_file.get(frame.file, ())
                if loop.line_range.contains_line(frame.line)
            ]
            if holding:
                return max(
                    holding,
                    key=lambda loop: (loop.line_range.start, -loop.line_range.end),
                )

    return None


def charge_loops(
    benchmark_charges: BenchmarkCharges,
    loops: Iterable[ChargeableLoop],
    sources: JavaSources,
) -> LoopCharges:
    """Charge each stack to the innermost chargeable loop that holds it"""
    loops_by_file: defaultdict[Path, list[ChargeableLoop]] = defaultdict(list)
    for loop in loops:
        loops_by_file[loop.file].append(loop)

    loop_counts: Counter[str] = Counter()
    outside_count = 0
    for stack in benchmark_charges.stacks:
        loop = innermost_loop(stack, loops_by_file, sources)
        if loop is None:
            outside_count += stack.count
        else:
            loop_counts[loop.loop_id] += stack.count

    return LoopCharges(dict(loop_counts), outside_count)


def hot_loops(
    loops: Iterable[ChargeableLoop],
    loop_charges: LoopCharges,
    charged_sample_count: int,
    hot_loop_share: float,
) -> list[ChargeableLoop]:
    """The chargeable loops that take at least the hot-path share of the samples"""
    if charged_sample_count == 0:
        return []

    threshold = hot_loop_share * charged_sample_count

    return [
        loop
        for loop in loops
        if loop_charges.loop_counts.get(loop.loop_id, 0) >= threshold
    ]
