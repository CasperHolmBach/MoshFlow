"""ProjectStore: saves and loads projects as JSON (.moshproj)."""

import json
import os

from moshflow.core.project import Project

EXTENSION = ".moshproj"


def save_project(project: Project, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project.to_dict(), f, indent=2)


def load_project(path: str) -> Project:
    """Loads a project; relative clip and output paths are relative to the file."""
    with open(path, encoding="utf-8") as f:
        project = Project.from_dict(json.load(f))
    base = os.path.dirname(os.path.abspath(path))
    for clip in project.clips:
        clip.path = _resolve(base, clip.path)
    project.output.path = _resolve(base, project.output.path)
    return project


def _resolve(base: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))
