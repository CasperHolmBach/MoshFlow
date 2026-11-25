"""ProjectModel: the state all views share. Views edit it and listen to its signals."""

from PySide6.QtCore import QObject, Signal

from moshflow.app.media import VideoInfo, probe
from moshflow.app.store import load_project, save_project
from moshflow.core.project import Clip, Project

CLIP = "clip"
TRANSITION = "transition"


class ProjectModel(QObject):
    changed = Signal()            # the project's contents changed
    selectionChanged = Signal()
    mediaChanged = Signal()
    projectReplaced = Signal()    # New / Open

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.media: list[VideoInfo] = []
        self.path: str | None = None
        self.dirty = False
        self.selection: tuple[str, int] | None = None

    # --- media -----------------------------------------------------------
    def import_media(self, paths: list[str]) -> list[str]:
        """Adds videos to the media bin; returns error messages for bad files."""
        errors = []
        known = {info.path for info in self.media}
        for path in paths:
            if path in known:
                continue
            try:
                self.media.append(probe(path))
                known.add(path)
            except IOError as e:
                errors.append(str(e))
        self.mediaChanged.emit()
        return errors

    def info(self, path: str) -> VideoInfo | None:
        for info in self.media:
            if info.path == path:
                return info
        return None

    # --- clips -----------------------------------------------------------
    def add_clip(self, path: str, index: int | None = None) -> None:
        self.project.add_clip(Clip(path))
        last = len(self.project.clips) - 1
        if index is not None and index != last:
            self.project.move_clip(last, index)
            last = index
        self.touch()
        self.select(CLIP, last)

    def remove_clip(self, index: int) -> None:
        self.project.remove_clip(index)
        self.touch()
        self.select(None)

    def move_clip(self, src: int, dst: int) -> None:
        self.project.move_clip(src, dst)
        self.touch()
        self.select(CLIP, dst)

    def clip_duration(self, index: int) -> float:
        """Length of a clip on the timeline in seconds, after trimming."""
        clip = self.project.clips[index]
        info = self.info(clip.path)
        total = info.duration if info else 0.0
        end = total if clip.end is None else min(clip.end, total)
        return max(0.0, end - clip.start)

    # --- selection and change tracking -----------------------------------
    def select(self, kind: str | None, index: int = -1) -> None:
        self.selection = (kind, index) if kind else None
        self.selectionChanged.emit()

    def touch(self) -> None:
        self.dirty = True
        self.changed.emit()

    # --- project files ---------------------------------------------------
    def new_project(self) -> None:
        self._replace(Project(), None)

    def open_project(self, path: str) -> list[str]:
        """Loads a project file; returns error messages for clips that cannot be read."""
        project = load_project(path)
        self._replace(project, path)
        return self.import_media(list(dict.fromkeys(clip.path for clip in project.clips)))

    def save_project(self, path: str) -> None:
        save_project(self.project, path)
        self.path = path
        self.dirty = False

    def _replace(self, project: Project, path: str | None) -> None:
        self.project, self.path, self.dirty, self.selection = project, path, False, None
        self.media = []
        self.mediaChanged.emit()
        self.projectReplaced.emit()
        self.changed.emit()
        self.selectionChanged.emit()
