"""Block-matching motion estimation, written with plain NumPy arithmetic.

Conventions (see docs/API.md, "Motion vectors"):
- A vector v = (dx, dy) says where a pixel of the current frame came from:
  current(x, y) ~ reference(x + dx, y + dy). An object that moved 2 pixels to
  the right therefore has v = (-2, 0).
- Block vectors: int32 array of shape (rows, cols, 2) with rows = ceil(H / block),
  cols = ceil(W / block); [..., 0] is dx and [..., 1] is dy. The last row and
  column of blocks may be smaller than block x block.
- A field: float32 array of shape (H, W, 2), one vector per pixel, same order.
"""

import numpy as np


def estimate(reference: np.ndarray, current: np.ndarray, block: int, search: int) -> np.ndarray:
    """Full-search block matching on two H x W float32 luma images.

    For every block of `current`, tries every d = (dx, dy) with |dx|, |dy| <= search
    and keeps the one with the smallest SAD(d) = sum |current(x) - reference(x + d)|
    over the block. Candidates that reach outside the image are skipped. Ties go to
    the smaller |dx| + |dy|, so flat areas get (0, 0).

    Returns block vectors (see the module docstring).
    """
    height, width = current.shape
    rows, cols = -(-height // block), -(-width // block)
    top = np.arange(rows) * block
    bottom = np.minimum(top + block, height)
    left = np.arange(cols) * block
    right = np.minimum(left + block, width)

    # One shift of the whole frame per candidate: a slice of the padded reference.
    padded = _pad_edges(reference.astype(np.float32), search)
    current = current.astype(np.float32)
    best_sad = np.full((rows, cols), np.inf, np.float32)
    best = np.zeros((rows, cols, 2), np.int32)
    for dx, dy in _candidates(search):
        shifted = padded[search + dy:search + dy + height, search + dx:search + dx + width]
        sad = _block_sums(np.abs(current - shifted), block, rows, cols)
        inside = (((top + dy >= 0) & (bottom + dy <= height))[:, None]
                  & ((left + dx >= 0) & (right + dx <= width))[None, :])
        better = inside & (sad < best_sad)
        best_sad[better] = sad[better]
        best[better] = (dx, dy)
    return best


def to_field(vectors: np.ndarray, size: tuple[int, int], block: int) -> np.ndarray:
    """Expands block vectors to a per-pixel field: every pixel takes its block's vector.

    size is (width, height). Returns an H x W x 2 float32 field.
    """
    width, height = size
    rows = np.arange(height) // block
    cols = np.arange(width) // block
    return vectors[rows[:, None], cols[None, :]].astype(np.float32)


def smooth_field(vectors: np.ndarray, size: tuple[int, int], block: int, smoothness: float) -> np.ndarray:
    """A per-pixel field without block edges, for Liquid Melt.

    Interpolates the block vectors bilinearly between block centres, then mixes:
    (1 - smoothness) * to_field(...) + smoothness * interpolated. smoothness = 0
    gives exactly to_field(vectors, size, block).
    """
    width, height = size
    rows, cols = vectors.shape[:2]
    # Position of every pixel centre in block units, where block centres sit on whole numbers.
    ys = np.clip((np.arange(height) + 0.5) / block - 0.5, 0, rows - 1)
    xs = np.clip((np.arange(width) + 0.5) / block - 0.5, 0, cols - 1)
    y0, x0 = ys.astype(np.intp), xs.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    wy = (ys - y0).astype(np.float32)[:, None, None]
    wx = (xs - x0).astype(np.float32)[None, :, None]

    v = vectors.astype(np.float32)
    upper = v[y0[:, None], x0[None, :]] * (1 - wx) + v[y0[:, None], x1[None, :]] * wx
    lower = v[y1[:, None], x0[None, :]] * (1 - wx) + v[y1[:, None], x1[None, :]] * wx
    smooth = upper * (1 - wy) + lower * wy

    blocky = to_field(vectors, size, block)
    return ((1 - smoothness) * blocky + smoothness * smooth).astype(np.float32)


def _candidates(search: int) -> list[tuple[int, int]]:
    """Every (dx, dy) in the search range, nearest to (0, 0) first, so ties keep the smaller one."""
    span = range(-search, search + 1)
    return sorted(((dx, dy) for dy in span for dx in span), key=lambda d: abs(d[0]) + abs(d[1]))


def _pad_edges(image: np.ndarray, n: int) -> np.ndarray:
    """The image with n extra pixels on every side, copied from the nearest edge pixel."""
    height, width = image.shape
    ys = np.clip(np.arange(-n, height + n), 0, height - 1)
    xs = np.clip(np.arange(-n, width + n), 0, width - 1)
    return image[ys[:, None], xs[None, :]]


def _block_sums(values: np.ndarray, block: int, rows: int, cols: int) -> np.ndarray:
    """Sum of an H x W array over each block; partial blocks at the edges sum what they have."""
    height, width = values.shape
    padded = np.zeros((rows * block, cols * block), np.float32)
    padded[:height, :width] = values
    return padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
