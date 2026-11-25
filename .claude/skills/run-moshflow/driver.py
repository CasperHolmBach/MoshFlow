"""Runs MoshFlow without a screen and saves what it shows.

Run from the repository root:
    python .claude/skills/run-moshflow/driver.py --out shots
    python .claude/skills/run-moshflow/driver.py --out shots --effect crossfade --param ease=2 --export
    python .claude/skills/run-moshflow/driver.py --out shots --clips a.mp4 b.mp4 --export

It opens the main window offscreen, imports clips (or makes moving test clips),
puts them on the timeline, sets the first transition, and saves screenshots. With
--export it renders the project in the background like the Export button and saves
frames from the middle of the first transition.

This is a development tool, not part of the program: it may use Qt's image
functions to save PNG files. Nothing under moshflow/ imports it.
"""

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

EXPORT_TIMEOUT_S = 120


def make_test_clips(folder: str) -> list[str]:
    """Two 2-second clips at 320 x 180, 15 fps, like the proposal's slides.

    A: a sunset gradient with a yellow sun moving right.
    B: a teal checkerboard with a pink ball moving down and right.
    """
    from moshflow.videoio import OpenCVWriter

    width, height, frames = 320, 180, 30
    ys, xs = np.mgrid[0:height, 0:width]

    def disc(frame, cx, cy, radius, bgr):
        frame[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = bgr

    def sunset(i):
        t = (ys / height)[..., None]
        frame = ((1 - t) * np.array([120, 40, 70]) + t * np.array([60, 120, 240])).astype(np.uint8)
        disc(frame, 60 + 4 * i, 90, 30, (90, 220, 255))
        return frame

    def checkerboard(i):
        light = ((xs // 30 + ys // 30) % 2 == 0)[..., None]
        frame = np.where(light, np.uint8([160, 140, 30]), np.uint8([90, 60, 20])).astype(np.uint8)
        disc(frame, 80 + 3 * i, 50 + 2 * i, 28, (130, 60, 255))
        return frame

    paths = []
    for name, scene in (("test_a.mp4", sunset), ("test_b.mp4", checkerboard)):
        path = os.path.join(folder, name)
        with OpenCVWriter(path, (width, height), 15.0) as writer:
            for i in range(frames):
                writer.write(np.ascontiguousarray(scene(i)))
        paths.append(path)
    return paths


def parse_params(items: list[str]) -> dict[str, object]:
    params = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            params[key] = int(raw)
        except ValueError:
            try:
                params[key] = float(raw)
            except ValueError:
                params[key] = {"true": True, "false": False}.get(raw.lower(), raw)
    return params


def screenshot(app: QApplication, window, path: str) -> None:
    app.processEvents()
    window.grab().save(path)
    print("screenshot", path)


def save_frames(video: str, indices: list[int], folder: str) -> None:
    from moshflow.videoio import OpenCVReader

    with OpenCVReader(video) as reader:
        for i in indices:
            if 0 <= i < reader.info.frame_count:
                frame = np.ascontiguousarray(reader.read(i))
                height, width = frame.shape[:2]
                image = QImage(frame.data, width, height, 3 * width, QImage.Format_BGR888)
                path = os.path.join(folder, f"frame_{i:04d}.png")
                image.save(path)
                print("frame", path)


def wait_for_export(app: QApplication, bar) -> bool:
    deadline = time.monotonic() + EXPORT_TIMEOUT_S
    while bar.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return bar.worker is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="shots", help="folder for screenshots, frames and the export")
    parser.add_argument("--clips", nargs="*", help="video files to use (default: two generated test clips)")
    parser.add_argument("--effect", default="crossfade", help="effect id for the first transition")
    parser.add_argument("--param", action="append", default=[], metavar="ID=VALUE")
    parser.add_argument("--duration", type=float, default=1.0, help="first transition, seconds")
    parser.add_argument("--undo", action="store_true", help="undo the transition change and screenshot again")
    parser.add_argument("--export", action="store_true", help="render the project and save frames")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    app = QApplication.instance() or QApplication([])

    from moshflow.core.project import Transition
    from moshflow.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1100, 680)
    window.show()
    for failure in window.catalog.failures:
        print("EFFECT FAILED TO LOAD", failure.module, failure.error, sep="\n")

    clips = [os.path.abspath(c) for c in args.clips] if args.clips else make_test_clips(out)
    errors = window.model.import_media(clips)
    for error in errors:
        print("IMPORT ERROR", error)
    for clip in clips:
        window.model.add_clip(clip)

    if len(clips) > 1:
        transition = Transition(args.effect, args.duration, params=parse_params(args.param))
        window.model.set_transition(0, transition, "effect")
        window.model.select("transition", 0)
    screenshot(app, window, os.path.join(out, "1_timeline.png"))

    if args.undo:
        window.model.undo_stack.undo()
        screenshot(app, window, os.path.join(out, "2_after_undo.png"))

    status = 0
    if args.export:
        video = os.path.join(out, "export.mp4")
        bar = window.export_bar
        bar.run_worker(video)
        if not wait_for_export(app, bar):
            print("EXPORT TIMED OUT")
            status = 1
        print("export status:", bar.status.text())
        screenshot(app, window, os.path.join(out, "3_after_export.png"))
        if os.path.exists(video):
            fps = window.model.info(clips[0]).fps
            cut = round(window.model.clip_duration(0) * fps)
            half = round(args.duration * fps / 2)
            save_frames(video, [cut - half, cut, cut + half - 1], out)
        else:
            status = 1

    project = window.model.project
    print("clips:", [os.path.basename(c.path) for c in project.clips])
    print("transitions:", [(t.effect, t.duration, dict(t.params)) for t in project.transitions])
    print("undo stack:", [window.model.undo_stack.text(i) for i in range(window.model.undo_stack.count())])
    window.model.undo_stack.setClean()
    window.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
