"""TimelineResolver: turns a project into a RenderGraph.

A transition of n frames replaces the last frames of clip A and/or the first
frames of clip B, depending on its alignment:
  before -> n frames of A,  center -> n/2 of A + n/2 of B,  after -> n frames of B.
"""

import os
from dataclasses import dataclass
from typing import Sequence

from moshflow.api import FrameSource
from moshflow.core.project import Project, Transition


class TimelineError(ValueError):
    """The project cannot be rendered as it is (e.g. a clip is too short)."""


@dataclass(frozen=True)
class ClipNode:
    clip: int
    frames: range

    @property
    def length(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class TransitionNode:
    cut: int              # index of the transition in project.transitions
    a: int                # outgoing clip index
    b: int                # incoming clip index
    a_window: range
    b_window: range
    transition: Transition

    @property
    def length(self) -> int:
        return len(self.a_window) + len(self.b_window)


@dataclass(frozen=True)
class RenderGraph:
    nodes: list[ClipNode | TransitionNode]

    @property
    def total_frames(self) -> int:
        return sum(node.length for node in self.nodes)


def split_window(frames: int, alignment: str) -> tuple[int, int]:
    """(frames taken from A, frames taken from B) for a transition."""
    if alignment == "before":
        return frames, 0
    if alignment == "after":
        return 0, frames
    return frames // 2, frames - frames // 2


def resolve(project: Project, sources: Sequence[FrameSource]) -> RenderGraph:
    project.validate()
    if not project.clips:
        raise TimelineError("the project has no clips")
    if len(sources) != len(project.clips):
        raise ValueError("need exactly one FrameSource per clip")

    fps = sources[0].fps
    splits = [split_window(int(round(t.duration * fps)), t.alignment) for t in project.transitions]

    ranges = []
    for i, source in enumerate(sources):
        start = splits[i - 1][1] if i > 0 else 0
        stop = len(source) - (splits[i][0] if i < len(splits) else 0)
        if stop < start:
            name = os.path.basename(project.clips[i].path)
            raise TimelineError(f"clip {i + 1} ({name}) has {len(source)} frames but its transitions "
                                f"need {start + len(source) - stop}; shorten the transitions")
        ranges.append(range(start, stop))

    nodes: list[ClipNode | TransitionNode] = []
    for i, frames in enumerate(ranges):
        if len(frames):
            nodes.append(ClipNode(i, frames))
        if i < len(splits):
            a_len, b_len = splits[i]
            if a_len + b_len:
                a_total = len(sources[i])
                nodes.append(TransitionNode(
                    cut=i, a=i, b=i + 1,
                    a_window=range(a_total - a_len, a_total),
                    b_window=range(0, b_len),
                    transition=project.transitions[i],
                ))
    return RenderGraph(nodes)
