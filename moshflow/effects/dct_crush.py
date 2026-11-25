"""DCT Crush: a decoder splices clip B's JPEG-style stream into clip A's
(Survey & Plan, 6.4).

Both clips are coded the way JPEG and MPEG code a picture, by hand:

1. Colour: BGR is converted to Y (brightness), Cb and Cr (colour), BT.601.
2. DCT: the frame is cut into block x block squares and each square becomes
   block x block frequency coefficients, F = C . f . C^T, where row u of C is a
   cosine with u half-waves. F[0, 0] (DC) is the block's average; the other 63
   (AC) are its texture and edges.
3. Splice: the blocks form a stream, left to right and top to bottom, cut into
   `tears` strips at fixed random points, like the resync points of a damaged
   file. Every strip switches from A to B at its own moment, earlier near the top:
   - shortly before: A's DC with B's AC, so B's edges appear in A's colours;
   - at the switch: B's DC with A's AC, so A's edges linger in B's colours, and
     B's blocks are read from up to `slide` blocks away in the stream, so the
     strip tears sideways and then slides into place;
   - after that: B's own blocks.
4. Crush: every coefficient is rounded to a step from the standard JPEG table
   times `crush` times a ramp that peaks in the middle of the window, so both clips
   break into flat and gradient blocks with ringing edges. Colour gets `chroma`
   times coarser steps.
5. The inverse DCT, f = C^T . F . C, and the inverse colour conversion give the frame.

At the last frame every strip has switched and settled and nothing is rounded,
so it is exactly B.

One pixel: in a strip that has just switched, its block has B's average colour but
A's 63 texture coefficients, so the pixel is B's colour plus A's edge pattern at its
position. With crush step 80, a coefficient of 130 becomes 160 and one of 30
becomes 0, so fine detail is lost and edges ring.
"""

import math

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color

# Standard JPEG quantization tables (ITU-T T.81, Annex K) for brightness and colour.
LUMA_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float32)
CHROMA_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], np.float32)

SWITCH = 0.28       # share of the window one strip takes from first B edges to settled


@register
class DctCrush(TransitionEffect):
    info = EffectInfo("dct_crush", "DCT Crush", "Casper Bach",
                      "Clip B's JPEG stream is spliced into clip A's: strips tear, mix and slide into B.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("block", "Block size", 8, 4, 32),
            IntParam("tears", "Tears", 10, 1, 200),
            IntParam("slide", "Slide", 24, 0, 128),
            FloatParam("crush", "Crush", 3.0, 0.0, 30.0),
            FloatParam("chroma", "Colour crush", 2.0, 1.0, 8.0),
            FloatParam("order", "Top to bottom", 0.4, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        n = p["block"]
        basis = _dct_matrix(n)
        steps = [_table(LUMA_TABLE, n), _table(CHROMA_TABLE, n) * p["chroma"], _table(CHROMA_TABLE, n) * p["chroma"]]
        is_dc = np.zeros((n, n), np.float32)
        is_dc[0, 0] = 1.0
        a = ctx.outgoing
        held_a = a.frame(ctx.a_window.stop - 1) if len(ctx.a_window) else a.frame(max(ctx.a_window.start - 1, 0))
        first_b = ctx.incoming.frame(ctx.b_window.start if len(ctx.b_window) else 0)
        width, height = ctx.size
        rows, cols = -(-height // n), -(-width // n)
        switch_at, offset = _strips(rows * cols, p, np.random.default_rng(ctx.seed))

        for i, (clip, index) in enumerate(_pairs(ctx)):
            if progress.cancelled:
                return
            t = (i + 1) / ctx.length
            frame_a = a.frame(index) if clip == "a" else held_a
            frame_b = ctx.incoming.frame(index) if clip == "b" else first_b

            # Where every block is in its strip's switch: < 0 before, 0..1 during, > 1 after.
            u = (t - switch_at) / SWITCH
            b_ac = (u >= 0.0).astype(np.float32) * ((u < 0.4) | (u >= 0.7)).astype(np.float32)
            b_ac = np.maximum(b_ac, ((u >= -0.6) & (u < 0.0)).astype(np.float32))
            b_dc = (u >= 0.0).astype(np.float32)
            # B's blocks are read from `offset` blocks away, shrinking to 0 as the strip settles.
            shift = _round(offset * (1.0 - _smooth(u))).astype(np.int32)
            source = np.clip(np.arange(rows * cols) - shift, 0, rows * cols - 1)
            crush = p["crush"] * min(_smooth(t / 0.35), _smooth((1.0 - t) / 0.35))

            planes_a = _to_ycbcr(frame_a.astype(np.float32))
            planes_b = _to_ycbcr(frame_b.astype(np.float32))
            out = []
            for c in range(3):
                flat_a = _dct(planes_a[c], basis).reshape(rows * cols, n, n)
                flat_b = _dct(planes_b[c], basis).reshape(rows * cols, n, n)[source]
                take_b = b_dc[:, None, None] * is_dc[None] + b_ac[:, None, None] * (1.0 - is_dc[None])
                coeff = flat_a + take_b * (flat_b - flat_a)
                if crush > 1e-3:
                    step = steps[c] * crush
                    coeff = _round(coeff / step) * step
                out.append(_idct(coeff.reshape(rows, cols, n, n), basis, planes_a[c].shape))

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(_to_bgr(out))


def _strips(total: int, p, rng) -> tuple[np.ndarray, np.ndarray]:
    """Cut the stream of `total` blocks into p["tears"] strips at random points.
    Returns, per block, when its strip switches to B (as a share of the window) and
    how many blocks away B's blocks are read from at the switch."""
    count = min(p["tears"], total)
    cuts = np.zeros(total, np.int32)
    if count > 1:
        cuts[rng.permutation(total - 1)[:count - 1] + 1] = 1
    strip = np.zeros(total, np.int32)
    for k in range(1, total):                              # strip number of every block
        strip[k] = strip[k - 1] + cuts[k]
    first = np.zeros(count, np.float32)
    for k in range(total - 1, -1, -1):                     # first block of every strip
        first[strip[k]] = k
    place = first / max(total - 1, 1)                      # 0 = top left, 1 = bottom right
    when = p["order"] * place + (1.0 - p["order"]) * rng.random(count).astype(np.float32)
    # The first B edges appear 0.6 * SWITCH early and the strip settles SWITCH later,
    # so switches run from 0.6 * SWITCH + 0.05 to 0.97 - SWITCH.
    low, high = 0.6 * SWITCH + 0.05, 0.97 - SWITCH
    switch_at = (low + when * (high - low)).astype(np.float32)
    offset = ((rng.random(count) * 2.0 - 1.0) * p["slide"]).astype(np.float32)
    return switch_at[strip], offset[strip]


def _smooth(x):
    """0 below 0, 1 above 1, and the smoothstep x * x * (3 - 2x) in between."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _pairs(ctx):
    """("a" or "b", frame index) for every output frame: A's window, then B's."""
    return [("a", i) for i in ctx.a_window] + [("b", i) for i in ctx.b_window]


def _dct_matrix(n: int) -> np.ndarray:
    """The n x n DCT-II matrix: C[u, x] = a(u) * cos((2x + 1) * u * pi / 2n),
    a(0) = sqrt(1/n), a(u) = sqrt(2/n). It is orthonormal, so its inverse is C^T."""
    rows = []
    for u in range(n):
        scale = math.sqrt((1.0 if u == 0 else 2.0) / n)
        rows.append([scale * math.cos((2 * x + 1) * u * math.pi / (2 * n)) for x in range(n)])
    return np.array(rows, np.float32)


def _table(table: np.ndarray, n: int) -> np.ndarray:
    """The 8 x 8 JPEG table stretched or shrunk to n x n by nearest entry."""
    idx = np.arange(n) * 8 // n
    return table[idx[:, None], idx[None, :]]


def _blocks(plane: np.ndarray, n: int) -> np.ndarray:
    """H x W plane as rows x cols x n x n blocks; the edge pixels are repeated to fill
    the last row and column of blocks."""
    height, width = plane.shape
    rows, cols = -(-height // n), -(-width // n)
    ys = np.minimum(np.arange(rows * n), height - 1)
    xs = np.minimum(np.arange(cols * n), width - 1)
    padded = plane[ys[:, None], xs[None, :]]
    return padded.reshape(rows, n, cols, n).transpose(0, 2, 1, 3)


def _times(left: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """left . block for every block: out[.., u, y] = sum_x left[u, x] * block[.., x, y]."""
    return (left[None, None, :, :, None] * blocks[:, :, None, :, :]).sum(axis=3)


def _dct(plane: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """F = C . f . C^T for every block of the plane."""
    blocks = _blocks(plane, basis.shape[0])
    rows_done = _times(basis, blocks)                                 # C . f
    return _times(basis, rows_done.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)   # (C . (C . f)^T)^T


def _idct(coeff: np.ndarray, basis: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """f = C^T . F . C for every block, put back together and cut to shape."""
    inverse = basis.T.copy()
    rows_done = _times(inverse, coeff)
    blocks = _times(inverse, rows_done.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    rows, cols, n = blocks.shape[0], blocks.shape[1], blocks.shape[2]
    plane = blocks.transpose(0, 2, 1, 3).reshape(rows * n, cols * n)
    return plane[:shape[0], :shape[1]]


def _round(values: np.ndarray) -> np.ndarray:
    """Round half away from zero, by hand: sign * floor(|v| + 0.5)."""
    sign = (values >= 0).astype(np.float32) * 2.0 - 1.0
    return sign * (np.abs(values) + 0.5).astype(np.int32).astype(np.float32)


def _to_ycbcr(bgr: np.ndarray) -> list[np.ndarray]:
    """BT.601 full range: Y, Cb, Cr planes from a float BGR image."""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return [y, cb, cr]


def _to_bgr(planes: list[np.ndarray]) -> np.ndarray:
    """Inverse of _to_ycbcr, as an H x W x 3 float BGR image."""
    y, cb, cr = planes[0], planes[1] - 128.0, planes[2] - 128.0
    out = np.zeros(y.shape + (3,), np.float32)
    out[..., 0] = y + 1.772 * cb
    out[..., 1] = y - 0.344136 * cb - 0.714136 * cr
    out[..., 2] = y + 1.402 * cr
    return out
