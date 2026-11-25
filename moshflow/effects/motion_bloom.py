"""Motion Bloom: one motion repeats, so moving pixels swell.

Emulates repeating one P-frame many times (Survey & Plan, 13.2). The motion
between two neighbouring frames is measured once, as a field F. Starting from
the last clean picture R of clip A, every output frame applies F once more.

Instead of warping the picture again and again (which blurs it or, with
nearest-pixel sampling, leaves zig-zag streaks), each pixel follows the flow:
it keeps a displacement D, and every frame steps once more along F from where
it already reads, D(k+1)(x) = D(k)(x) + s F(x + D(k)(x)). The output pixel is then
read once from R: out(x) = R(x + D(k)(x)), with bilinear sampling. A pixel inside a
moving area reads from further and further upstream, so the area stretches and
swells; a pixel where F is (0, 0) stays frozen.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times turns into a random streak. So vectors that do not explain the
motion clearly better than (0, 0) are dropped, and the block vectors are
interpolated between block centres so no hard block edges appear.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

BLOCK = 16          # block size and search range for capturing the motion
SEARCH = 8
CONFIDENCE = 0.7    # keep a block's vector only if it cuts the block's error to below 70 %
FADE_PART = 1 / 3   # share of the transition that the "fade" release blends into clip B


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "One motion repeats on the last picture of clip A, so moving areas swell.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("source", "Motion from", "A", ("A", "B")),
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("decay", "Decay", 0.02, 0.0, 0.5),
            ChoiceParam("release", "Release", "fade", ("cut", "fade")),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        field = _captured_field(ctx, p["source"])
        start = frames.last_clean_a(ctx).astype(np.float32)
        width, height = ctx.size
        displacement = np.zeros((height, width, 2), np.float32)
        fade_length = max(1, round(ctx.length * FADE_PART)) if p["release"] == "fade" else 0
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            # One more step along the same motion, a little weaker every frame.
            strength = p["gain"] * (1.0 - p["decay"]) ** i
            displacement = displacement + strength * warp.warp(field, displacement, "bilinear")
            picture = warp.warp(start, displacement, "bilinear")

            out = picture
            if fade_length:
                # Over the last frames, each pixel moves linearly from the bloom to B:
                # out = (1 - w) * bloom + w * B, with w reaching 1 on the last frame.
                w = min(max((i + 1 - (ctx.length - fade_length)) / fade_length, 0.0), 1.0)
                if w > 0:
                    out = (1.0 - w) * picture + w * _clip_b_frame(ctx, clip, index)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _clip_b_frame(ctx, clip, index: int) -> np.ndarray:
    """The frame of B that output frame shows when there is no effect.

    Inside B's window that is B's own frame; inside A's window it is B's first
    frame, the one the video cuts to right after the transition.
    """
    if clip is ctx.incoming:
        return clip.frame(index).astype(np.float32)
    return ctx.incoming.frame(ctx.b_window.start).astype(np.float32)


def _captured_field(ctx, source: str) -> np.ndarray:
    """The motion between two neighbouring frames, as a smooth per-pixel field (H, W, 2).

    "A": from the second-to-last to the last clean frame of A, so A's own motion
    carries on. "B": from B's first to its second frame, so B's motion drags A's picture.
    """
    if source == "A":
        clip = ctx.outgoing
        second = max(ctx.a_window.start - 1, 0)
        first = max(second - 1, 0)
        if first == second:
            second = min(1, len(clip) - 1)
    else:
        clip = ctx.incoming
        first = ctx.b_window.start
        second = min(first + 1, len(clip) - 1)
    reference = color.bgr_to_y(clip.frame(first))
    current = color.bgr_to_y(clip.frame(second))
    vectors = motion.estimate(reference, current, BLOCK, SEARCH)
    vectors = _confident(vectors, reference, current)
    return motion.smooth_field(vectors, ctx.size, BLOCK, 1.0)


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
