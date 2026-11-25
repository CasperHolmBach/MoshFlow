"""Motion Bloom: clip B's motion drags clip A's picture until B bleeds through.

Emulates a datamosh where clip B's P-frames are applied to clip A's last picture
(Survey & Plan, 13.2). Motion is a field F, one vector per pixel, measured by
block matching on B: F(x) says where B's content at x came from.

- Bloom (A's part of the window): B's first P-frame is repeated. Every frame,
  each pixel of A's picture is moved once more along the same vectors, so moving
  areas stretch and swell while still areas stay frozen.
- Drag (B's part of the window): B's real motion, frame by frame, keeps moving
  what is left of A's picture.
- Bleed: every pixel switches from A to B on its own, earlier where B moves more.
  How much of B a pixel shows is dragged along with the motion too, so B spreads
  out of the moving areas. By the last frame every pixel shows B.

A's picture is never warped twice: each pixel keeps a displacement D (where in A's
picture it reads from), each new P-frame composes onto it,
D_new(x) = F(x) + D(x + F(x)), and the pixel is read once with bilinear sampling.
This avoids the blur of warping again and again and the zig-zag of rounding.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated many
times becomes a random streak, so vectors that do not explain the motion clearly
better than (0, 0) are dropped, and block vectors are interpolated between block
centres so no hard block edges appear.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, warp

BLOCK = 16          # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7    # keep a block's vector only if it cuts the block's error to below 70 %
FAR_PART = 0.15     # a pixel dragged this share of the frame size counts as fully dragged
SWITCH = 0.2        # share of the transition that one pixel takes to turn from A into B
LOOK_AHEAD = 8      # B frame pairs searched for the strongest motion to repeat


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "Clip B's motion drags and swells clip A's last picture until B bleeds through.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("decay", "Decay", 0.0, 0.0, 0.5),
            FloatParam("bleed", "Bleed", 0.6, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        a_picture = _last_clean_a(ctx).astype(np.float32)
        b = ctx.incoming
        first_b = min(ctx.b_window.start, len(b) - 1)
        motion_now = _strongest_motion(b, first_b, ctx.size)
        far = FAR_PART * max(width, height)

        displacement = np.zeros((height, width, 2), np.float32)   # where each pixel reads A
        shown_b = np.zeros((height, width), np.float32)           # 0..1, how much B a pixel shows
        a_frames = len(ctx.a_window)
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < a_frames:
                # Bloom: the same P-frame of B again, a little weaker every frame with decay.
                field = p["gain"] * (1.0 - p["decay"]) ** i * motion_now
                b_frame = b.frame(first_b)
            else:
                # Drag: B's real motion. A repeated (duplicate) frame has no motion of its
                # own, so the previous motion carries on instead of stopping for a frame.
                index = ctx.b_window[i - a_frames]
                if index > 0 and not _same(b.frame(index - 1), b.frame(index)):
                    motion_now = _field(b, index - 1, index, ctx.size)
                field = p["gain"] * motion_now
                b_frame = b.frame(index)

            displacement = field + warp.warp(displacement, field, "bilinear")

            # A pixel starts turning into B at time `start`, earlier the further it has
            # been dragged, so B seeps in through the most smeared areas first.
            dragged = np.minimum((displacement[..., 0] ** 2 + displacement[..., 1] ** 2) ** 0.5 / far, 1.0)
            t = (i + 1) / ctx.length
            start = (1.0 - SWITCH) * (1.0 - p["bleed"] * dragged)
            target = np.clip((t - start) / SWITCH, 0.0, 1.0)
            shown_b = np.maximum(warp.warp(shown_b, field, "bilinear"), target)
            if i == ctx.length - 1:
                shown_b[:] = 1.0

            w = shown_b[..., None]
            out = (1.0 - w) * warp.warp(a_picture, displacement, "bilinear") + w * b_frame
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _last_clean_a(ctx) -> np.ndarray:
    """A's frame just before the window (A's first frame if there is none)."""
    return ctx.outgoing.frame(max(ctx.a_window.start - 1, 0))


def _strongest_motion(clip, start: int, size: tuple[int, int]) -> np.ndarray:
    """The P-frame to repeat: of B's first LOOK_AHEAD frame pairs, the one that moves most.

    Film converted to 30 fps repeats frames, and a repeated frame has no motion,
    so simply taking the first pair can give nothing to repeat.
    """
    width, height = size
    best, best_amount = np.zeros((height, width, 2), np.float32), 0.0
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1)):
        if _same(clip.frame(first), clip.frame(first + 1)):
            continue
        field = _field(clip, first, first + 1, size)
        amount = float((field[..., 0] ** 2 + field[..., 1] ** 2).mean())
        if amount > best_amount:
            best, best_amount = field, amount
    return best


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two frames are (almost) identical, i.e. a repeated frame."""
    return float(np.abs(a.astype(np.float32) - b).mean()) < 0.5


def _field(clip, first: int, second: int, size: tuple[int, int]) -> np.ndarray:
    """Motion from frame `first` to frame `second` of a clip, as a smooth field (H, W, 2).

    Block matching runs at half size (every second pixel, half the block and search)
    to keep it fast, and the vectors are doubled back to full size.
    """
    reference = color.bgr_to_y(clip.frame(first))[::2, ::2]
    current = color.bgr_to_y(clip.frame(second))[::2, ::2]
    vectors = motion.estimate(reference, current, BLOCK // 2, SEARCH // 2)
    vectors = _confident(vectors, reference, current, BLOCK // 2)
    return motion.smooth_field(vectors * 2, size, BLOCK, 1.0)


def _confident(vectors: np.ndarray, reference: np.ndarray, current: np.ndarray, block: int) -> np.ndarray:
    """Block vectors with the unreliable ones set to (0, 0).

    For each block, error(v) = sum of |current(x) - reference(x + v)| over its pixels.
    A vector is kept only if error(v) < CONFIDENCE * error(0): the match must be
    clearly better than "nothing moved", otherwise it is just noise.
    """
    height, width = current.shape
    field = motion.to_field(vectors, (width, height), block)
    moved = _block_sums(np.abs(current - warp.warp(reference, field)), block)
    still = _block_sums(np.abs(current - reference), block)
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
