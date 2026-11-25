---
name: check-library-rule
description: Check MoshFlow code against the course library rule before a commit, pull request or submission. Use when adding imports or library calls, reviewing a teammate's effect, or when unsure whether a NumPy, OpenCV or Qt function is allowed.
---

# Check the library rule

Paths are relative to the repository root. The rule: apart from OpenCV video input
and output, no library function may do image or signal processing. Breaking it can
fail the whole project, so check every change to `moshflow/`.

## 1. Run the automatic check

```bash
.venv/bin/python -m pytest tests/test_compliance.py
```

It fails on any `cv2` use outside `moshflow/videoio/`, on banned modules (SciPy,
scikit-image, Pillow, imageio, MoviePy, FFmpeg, PyAV, Numba, scikit-learn), on
`np.fft`, `np.convolve`, `np.correlate`, `np.gradient`, `np.interp`, `np.histogram`,
and on Qt image functions. It cannot catch everything.

## 2. Read the change yourself

```bash
git diff main -- moshflow/
```

For every new import or library call, decide:

| Allowed | Not allowed |
| --- | --- |
| `cv2.VideoCapture`, `cv2.VideoWriter`, `cv2.CAP_PROP_*`, only in `moshflow/videoio/` | Any other `cv2` function, anywhere |
| NumPy arrays: creating, indexing, slicing, `astype`, `reshape`, shapes | NumPy routines that do a processing step for you: filtering, convolution, interpolation, FFT, histograms, image resizing |
| Element-wise `+ - * /`, `abs`, `min`, `max`, `sum`, `clip`, `argmin`, `round` | SciPy, scikit-image, Pillow, imageio, MoviePy, PyAV, FFmpeg filters, Numba |
| PySide6 widgets, layouts, `QPainter` for drawing the UI | `QImage`, `QTransform`, `scaled()` and other Qt functions used on video frames |
| Python standard library for program logic | A new package in `requirements.txt` |

When a call does not fit either column, ask: does the library do the image
processing step that is the point of our project (motion estimation, warping,
blending, resizing, colour conversion)? If yes, it is not allowed; write it by hand.
If still unsure, report it as a question for the team instead of deciding.

## 3. Report

List each finding as `path:line`, the call, and the hand-written replacement (for
example: `np.interp` → bilinear weights as in `toolkit/resize.py`). Say clearly
when nothing was found.

Files outside `moshflow/` (tests, `tools/`, `.claude/`) are not part of the
graded program, but they must never be imported from `moshflow/`.
