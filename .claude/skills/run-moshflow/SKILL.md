---
name: run-moshflow
description: Run, launch, drive, screenshot or export MoshFlow without a screen. Use to see a UI change or an effect's output, check that the app starts, render a transition and look at its frames, or reproduce a UI bug.
---

# Run MoshFlow

Paths are relative to the repository root. The agent path is `driver.py` in this
folder: it opens the main window offscreen, builds a small project, and saves
screenshots, an exported video and frames from the transition.

## Prerequisites

Python 3.12 with `requirements.txt` installed (see `CLAUDE.md`, "Commands"). On
Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

## Drive the app (agent path)

```bash
.venv/bin/python .claude/skills/run-moshflow/driver.py --out shots --effect crossfade --param ease=2 --export
```

Then read the images in `shots/`:

| File | Shows |
| --- | --- |
| `1_timeline.png` | The window with two clips and the first transition selected in the Inspector |
| `2_after_undo.png` | Only with `--undo`: the window after undoing the transition change |
| `3_after_export.png` | The window after the export finished; the status text is printed too |
| `frame_NNNN.png` | Exported frames at the start, the cut and the end of the first transition |
| `export.mp4` | The exported video |

Options: `--clips a.mp4 b.mp4` uses your own videos instead of the generated test
clips, `--effect <id>` and `--param id=value` (repeatable) set the first transition,
`--duration` sets its length in seconds, `--undo` adds the undo step.

Without `--clips`, the driver writes two test clips into the output folder: a sunset
with a sun moving right (clip A) and a checkerboard with a ball moving down and
right (clip B). They move, so block matching has motion to find.

To check a new effect, run the driver with `--effect <your_effect_id> --export`
and look at the three frames. A plain cut means the effect still renders the
skeleton's placeholder.

## Run without the GUI

```bash
.venv/bin/python -m moshflow.cli effects
.venv/bin/python -m moshflow.cli try --effect crossfade a.mp4 b.mp4 -o test.mp4 --size 640x360
.venv/bin/python -m pytest
```

## Run for a person

```bash
.venv/bin/python run.py
```

This opens the window and waits; it is no use without a screen.

## Gotchas

- The driver sets `QT_QPA_PLATFORM=offscreen` itself. Qt then prints
  `This plugin does not support propagateSizeHints()` and a font notice; both are harmless.
- With `--undo --export`, the export renders the project after the undo, so the
  transition is back to Hard Cut.
- The frames are read back from `export.mp4`, which is lossy (mp4v). Colours can be
  off by a few levels; compare shapes and motion, not exact pixel values.
- The driver may use Qt image functions to save PNG files. It is a development tool
  outside `moshflow/`, so the course library rule and `tests/test_compliance.py` do
  not apply to it. Never import it from `moshflow/`.
