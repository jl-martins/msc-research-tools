"""Collapsed stack module"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from dacapo_hot_loops import HotLoopsError
from dacapo_hot_loops.sources import SourceIndex

# A frame's trailing `:<line>` and `_[<tag>]` are both optional
FRAME_PATTERN = re.compile(r"^(?P<name>.*?)(?::(?P<line>\d+))?(?:_\[[0-9a-z]+\])?$")
# A synthetic class has no source of its own
SYNTHETIC_MARKERS = ("$$Lambda", "LambdaForm$")


class Frame(NamedTuple):
    """One frame of a collapsed stack"""

    raw: str
    class_name: str
    method: str
    line: int | None
    is_application: bool

    @property
    def top_level_class(self) -> str:
        """The class name up to its first `$`, which names the source file"""
        return self.class_name.partition("$")[0]


class Stack(NamedTuple):
    """One collapsed line: its frames, root first, and its sample count"""

    frames: tuple[Frame, ...]
    count: int


def is_application_class(
    raw: str,
    class_name: str,
    excluded_prefixes: tuple[str, ...],
) -> bool:
    """Whether the frame is benchmark code rather than the JDK, JVM or harness"""
    is_synthetic = any(marker in class_name for marker in SYNTHETIC_MARKERS)
    is_excluded = class_name.startswith(excluded_prefixes)
    # Only a Java frame holds a `/`, which separates the parts of its package
    return "/" in raw and not is_synthetic and not is_excluded


def parse_frame(raw: str, excluded_prefixes: tuple[str, ...]) -> Frame:
    frame_match = FRAME_PATTERN.match(raw)
    # Every optional part of the pattern may be absent, so it always matches
    assert frame_match is not None
    name = frame_match["name"].replace("/", ".")
    line = frame_match["line"]
    class_name, _, method = name.rpartition(".")
    return Frame(
        raw=raw,
        class_name=class_name,
        method=method,
        line=None if line is None else int(line),
        is_application=is_application_class(raw, class_name, excluded_prefixes),
    )


def parse_line(line: str, excluded_prefixes: tuple[str, ...]) -> Stack:
    # Some native frame names hold spaces, so the count follows the last one
    frames_text, separator, count_text = line.rpartition(" ")
    if separator == "" or not count_text.isdigit():
        raise HotLoopsError(f"Malformed collapsed line: {line!r}")

    frames = tuple(
        parse_frame(raw, excluded_prefixes) for raw in frames_text.split(";")
    )
    return Stack(frames, int(count_text))


def parse_collapsed(text: str, excluded_prefixes: tuple[str, ...]) -> Iterator[Stack]:
    return (
        parse_line(line, excluded_prefixes)
        for line in text.splitlines()
        if line.strip() != ""
    )


def source_file(frame: Frame, index: SourceIndex) -> Path | None:
    """The file of a source frame, or `None` when the frame has no source"""
    if frame.is_application:
        return index.get(frame.top_level_class)
    else:
        return None
