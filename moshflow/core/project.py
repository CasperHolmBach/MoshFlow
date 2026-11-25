"""Project model: an ordered list of clips with one transition per cut."""

from dataclasses import asdict, dataclass, field

ALIGNMENTS = ("before", "center", "after")
DEFAULT_EFFECT = "hard_cut"


@dataclass
class Clip:
    path: str
    start: float = 0.0           # trim in, seconds
    end: float | None = None     # trim out, seconds; None = end of file


@dataclass
class Transition:
    effect: str = DEFAULT_EFFECT
    duration: float = 1.0        # seconds
    alignment: str = "center"    # "before" / "center" / "after" the cut
    params: dict[str, object] = field(default_factory=dict)
    seed: int = 0


@dataclass
class OutputSettings:
    path: str = "output.mp4"
    width: int | None = None     # None = size of the first clip
    height: int | None = None
    fps: float | None = None     # None = frame rate of the first clip


@dataclass
class Project:
    clips: list[Clip] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)   # transitions[i] sits between clips[i] and clips[i + 1]
    output: OutputSettings = field(default_factory=OutputSettings)

    def add_clip(self, clip: Clip) -> None:
        if self.clips:
            self.transitions.append(Transition())
        self.clips.append(clip)

    def remove_clip(self, index: int) -> None:
        del self.clips[index]
        if self.transitions:
            del self.transitions[min(index, len(self.transitions) - 1)]

    def move_clip(self, src: int, dst: int) -> None:
        """Reorders clips; transitions stay attached to their cut position."""
        clip = self.clips.pop(src)
        self.clips.insert(dst, clip)

    def validate(self) -> None:
        if len(self.transitions) != max(0, len(self.clips) - 1):
            raise ValueError(f"{len(self.clips)} clips need {max(0, len(self.clips) - 1)} transitions, "
                             f"found {len(self.transitions)}")
        for i, t in enumerate(self.transitions, start=1):
            if t.alignment not in ALIGNMENTS:
                raise ValueError(f"transition {i}: alignment must be one of {ALIGNMENTS}")
            if t.duration < 0:
                raise ValueError(f"transition {i}: duration must not be negative")
        for i, c in enumerate(self.clips, start=1):
            if c.start < 0 or (c.end is not None and c.end <= c.start):
                raise ValueError(f"clip {i}: invalid trim {c.start}..{c.end}")

    def to_dict(self) -> dict:
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        project = cls(
            clips=[Clip(**c) for c in data.get("clips", [])],
            transitions=[Transition(**t) for t in data.get("transitions", [])],
            output=OutputSettings(**data.get("output", {})),
        )
        project.validate()
        return project
