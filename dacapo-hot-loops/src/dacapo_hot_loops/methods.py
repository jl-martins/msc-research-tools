"""Method charging module"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from dacapo_hot_loops.collapsed import Frame, Stack, source_file
from dacapo_hot_loops.javasrc import (
    CLASS_INITIALISER_NAME,
    CONSTRUCTOR_NAME,
    JavaSources,
    LineRange,
)
from dacapo_hot_loops.sources import SourceIndex

# The method name that javac gives to a lambda body
LAMBDA_PATTERN = re.compile(r"^lambda\$(?P<owner>.*)\$(?P<index>\d+)$")
# What a lambda's owner means when it does not name a method
LAMBDA_OWNERS = {"new": CONSTRUCTOR_NAME, "static": CLASS_INITIALISER_NAME}
# Older javac names the owner of a nested lambda `null`
NESTED_LAMBDA_OWNER = "null"


class MethodName(NamedTuple):
    """The class and method parts of a method key"""

    class_name: str
    method: str


class ResolvedFrame(NamedTuple):
    """Source frame, with the method key it charges"""

    method_key: str
    file: Path
    line: int | None


class CallSite(NamedTuple):
    """Position in a method, as a caller frame identifies it"""

    method_key: str
    file: Path
    line: int | None


class ChargedStack(NamedTuple):
    """Stack that holds at least one source frame, resolved root first"""

    frames: tuple[ResolvedFrame, ...]
    count: int

    @property
    def charged(self) -> ResolvedFrame:
        """The topmost source frame, which takes the samples"""
        return self.frames[-1]

    @property
    def caller(self) -> ResolvedFrame | None:
        """The next source frame toward the root"""
        if len(self.frames) >= 2:
            return self.frames[-2]
        else:
            return None


class BenchmarkCharges(NamedTuple):
    """Sample counts and charged stacks of one benchmark"""

    total_sample_count: int
    no_app_sample_count: int
    no_source_sample_count: int
    charged_sample_count: int
    stacks: tuple[ChargedStack, ...]
    # Application class without source -> samples of the stacks it stands for
    unresolved_classes: dict[str, int]


class MethodCharge(NamedTuple):
    """Samples charged to one method key"""

    method_key: str
    file: Path
    sample_count: int
    # Lines charged to the method, which locate its anonymous declarations
    lines: frozenset[int]


class CallerCharge(NamedTuple):
    """Call site that charges a hot method most often"""

    method_key: str
    file: Path
    line: int | None
    sample_count: int


def split_method_key(method_key: str) -> MethodName:
    class_name, _, method = method_key.rpartition(".")
    return MethodName(class_name, method)


def type_path_of(class_name: str) -> str:
    """A class name without its package, which is its type path within its file"""
    return class_name.rpartition(".")[2]


def method_key_of(frame: Frame, file: Path, sources: JavaSources) -> str:
    """The key that merges a frame's overloads, or holds its lambda's owner"""
    lambda_match = LAMBDA_PATTERN.match(frame.method)
    if lambda_match is None:
        return f"{frame.class_name}.{frame.method}"

    if frame.line is not None:
        lines = LineRange(frame.line, frame.line)
        declaration = sources.file(file).innermost_declaration(lines)
        if declaration is not None:
            return f"{frame.class_name}.{declaration.name}"

    # No line locates a declaration, so the lambda's own name names its owner
    owner = lambda_match["owner"]
    if owner == NESTED_LAMBDA_OWNER:
        # This key matches no declaration, so such a method is always unresolved
        return f"{frame.class_name}.{frame.method}"
    else:
        return f"{frame.class_name}.{LAMBDA_OWNERS.get(owner, owner)}"


def charge_stacks(
    stacks: Iterable[Stack],
    index: SourceIndex,
    sources: JavaSources,
) -> BenchmarkCharges:
    """Charge every stack to its topmost source frame, counting the rest"""
    total_count = no_app_count = no_source_count = charged_count = 0
    charged_stacks: list[ChargedStack] = []
    unresolved_classes: Counter[str] = Counter()

    for stack in stacks:
        total_count += stack.count
        resolved = []
        for frame in stack.frames:
            file = source_file(frame, index)
            if file is not None:
                key = method_key_of(frame, file, sources)
                resolved.append(ResolvedFrame(key, file, frame.line))

        if resolved:
            charged_count += stack.count
            charged_stacks.append(ChargedStack(tuple(resolved), stack.count))
        else:
            application_frames = [f for f in stack.frames if f.is_application]
            if application_frames:
                no_source_count += stack.count
                # Charging the topmost frame, as a source frame would be charged,
                # splits these samples between classes without counting any twice
                unresolved_classes[application_frames[-1].class_name] += stack.count
            else:
                no_app_count += stack.count

    return BenchmarkCharges(
        total_sample_count=total_count,
        no_app_sample_count=no_app_count,
        no_source_sample_count=no_source_count,
        charged_sample_count=charged_count,
        stacks=tuple(charged_stacks),
        unresolved_classes=dict(unresolved_classes),
    )


def method_charges(benchmark_charges: BenchmarkCharges) -> list[MethodCharge]:
    """Every charged method, by samples descending and then by key"""
    counts: Counter[str] = Counter()
    files: dict[str, Path] = {}
    lines: defaultdict[str, set[int]] = defaultdict(set)
    for stack in benchmark_charges.stacks:
        frame = stack.charged
        counts[frame.method_key] += stack.count
        files.setdefault(frame.method_key, frame.file)
        if frame.line is not None:
            lines[frame.method_key].add(frame.line)

    ranked = [
        MethodCharge(key, files[key], count, frozenset(lines[key]))
        for key, count in counts.items()
    ]
    ranked.sort(key=lambda charge: (-charge.sample_count, charge.method_key))

    return ranked


def hot_methods(
    benchmark_charges: BenchmarkCharges,
    hot_method_share: float,
) -> list[MethodCharge]:
    """The smallest top-ranked methods whose cumulative share reaches the cutoff"""
    ranked = method_charges(benchmark_charges)
    if not ranked:
        return []

    target = hot_method_share * benchmark_charges.charged_sample_count
    cumulative = 0
    hot_count = len(ranked)
    for position, charge in enumerate(ranked, start=1):
        cumulative += charge.sample_count
        if cumulative >= target:
            hot_count = position
            break

    # A method tied in samples with the last hot method is hot as well
    last_count = ranked[hot_count - 1].sample_count
    while hot_count < len(ranked) and ranked[hot_count].sample_count == last_count:
        hot_count += 1

    return ranked[:hot_count]


def caller_sort_key(caller: CallerCharge) -> tuple[int, str, int]:
    # A caller frame without a line sorts first and yields no candidate loop
    line = -1 if caller.line is None else caller.line
    return (-caller.sample_count, caller.method_key, line)


def main_caller(
    benchmark_charges: BenchmarkCharges, method_key: str
) -> CallerCharge | None:
    """The call site that charges a method most often, or None without one"""
    counts: Counter[CallSite] = Counter()
    for stack in benchmark_charges.stacks:
        if stack.charged.method_key == method_key:
            caller = stack.caller
            if caller is not None:
                counts[CallSite(caller.method_key, caller.file, caller.line)] += (
                    stack.count
                )

    callers = [CallerCharge(*site, count) for site, count in counts.items()]
    if callers:
        return min(callers, key=caller_sort_key)
    else:
        return None
