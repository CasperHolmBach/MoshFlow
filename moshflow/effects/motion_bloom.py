"""Motion Bloom: one motion repeats, so moving pixels swell, and A blooms into B.

Emulates repeating one P-frame many times (Survey & Plan, 13.2). For each clip,
the motion between two neighbouring frames is measured once, as a field F
(a vector per pixel). Repeating that P-frame n times moves every pixel n times
along its vector, so pixel x reads from x + n F(x). Neighbouring pixels with
different vectors drift apart a little more every frame, so moving areas
stretch and swell, while pixels where F is (0, 0) stay frozen.

The whole transition is the bloom:
- clip A blooms out: out_A(x) = A(x + n_A F_A(x)), with n_A growing from 1;
- clip B blooms in, backwards: out_B(x) = B(x - n_B F_B(x)), with n_B shrinking
  to 1 on the last frame, so B's swollen picture settles into its real frame;
- in the middle, while both pictures are most swollen, each pixel switches from
  out_A to out_B (a short blend with "fade", a hard switch with "cut").

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times turns into a random streak. So vectors that do not explain the
motion clearly better than (0, 0) are dropped, and the block vectors are
interpolated between block centres so no hard block edges appear. Pixels are
read with bilinear sampling, so there is no zig-zag from rounding.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

BLOCK = 16          # block size and search range for capturing the motion
SEARCH = 8
CONFIDENCE = 0.7    # keep a block's vector only if it cuts the block's error to below 70 %
BLEND_PART = 0.5    # share of the transition, around the middle, that "fade" blends A into B


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "Clip A's motion repeats until A swells into clip B, which settles back into place.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("source", "Motion from", "own", ("own", "A", "B")),
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("decay", "Decay", 0.0, 0.0, 0.5),
            ChoiceParam("release", "Switch to B", "fade", ("cut", "fade")),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        picture_a = frames.last_clean_a(ctx).astype(np.float32)
        b_index = _first_clean_b(ctx)
        picture_b = ctx.incoming.frame(b_index).astype(np.float32)
        field_a = _field(ctx.outgoing, max(ctx.a_window.start - 1, 0), -1, ctx.size)
        field_b = _field(ctx.incoming, b_index, +1, ctx.size)
        if p["source"] == "A":
            field_b = field_a
        elif p["source"] == "B":
            field_a = field_b

        steps = _repeats(ctx.length, p["gain"], p["decay"])
        for i in range(ctx.length):
            if progress.cancelled:
                return
            n_a, n_b = steps[i], steps[ctx.length - 1 - i]
            w = _switch(i, ctx.length, p["release"])
            out = 0.0
            if w < 1:
                out = (1.0 - w) * warp.warp(picture_a, n_a * field_a, "bilinear")
            if w > 0:
                out = out + w * warp.warp(picture_b, -n_b * field_b, "bilinear")
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _repeats(length: int, gain: float, decay: float) -> list[float]:
    """How many times the P-frame has been repeated at each output frame.

    Repeat k adds gain * (1 - decay) ** k, so with decay > 0 the motion slows down.
    """
    total, out = 0.0, []
    for k in range(length):
        total += gain * (1.0 - decay) ** k
        out.append(total)
    return out


def _switch(i: int, length: int, release: str) -> float:
    """Weight of clip B in output frame i: 0 = only A, 1 = only B."""
    t = (i + 0.5) / length
    if release == "cut":
        return 1.0 if t >= 0.5 else 0.0
    u = min(max((t - 0.5) / BLEND_PART + 0.5, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)   # smoothstep: starts and ends gently


def _first_clean_b(ctx) -> int:
    """B's frame just after the window: the real frame the bloom settles into."""
    return min(ctx.b_window.stop, len(ctx.incoming) - 1)


def _field(clip, index: int, toward: int, size: tuple[int, int]) -> np.ndarray:
    """The motion at frame `index` of a clip, as a smooth per-pixel field (H, W, 2).

    Measured between `index` and its neighbour (the frame before for toward=-1,
    after for +1), always as earlier -> later, so F follows the clip's own motion.
    """
    other = min(max(index + toward, 0), len(clip) - 1)
    if other == index:
        other = min(max(index - toward, 0), len(clip) - 1)
    first, second = sorted((index, other))
    reference = color.bgr_to_y(clip.frame(first))
    current = color.bgr_to_y(clip.frame(second))
    vectors = motion.estimate(reference, current, BLOCK, SEARCH)
    vectors = _confident(vectors, reference, current)
    return motion.smooth_field(vectors, size, BLOCK, 1.0)


def _confident(vectors: np.ndarray, reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Block vectors with the unreliable ones set to (0, 0).

    For each block, error(v) = sum of |current(x) - reference(x + v)| over its pixels.
    A vector is kept only if error(v) < CONFIDENCE * error(0): the match must be
    clearly better than "nothing moved", otherwise it is just noise.
    """
    height, width = current.shape
    field = motion.to_field(vectors, (width, height), BLOCK)
    moved = _block_sums(np.abs(current - warp.warp(reference, field)), BLOCK)
    still = _block_sums(np.abs(current - reference), BLOCK)
    out = vectors.copy()
    out[~(moved < CONFIDENCE * still)] = 0
    return out


def _block_sums(values: np.ndarray, block: int) -> np.ndarray:
    """Sum of an H x W array over each block; partial blocks at the edges sum what they have."""
    height, width = values.shape
    rows, cols = -(-height // block), -(-width // block)
    padded = np.zeros((rows * block, cols * block), np.float32)
    padded[:height, :width] = values
    return padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
