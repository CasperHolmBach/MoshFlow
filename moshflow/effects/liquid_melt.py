"""Liquid Melt: a lost I-frame with smooth motion in place of blocks (Survey & Plan, 13.3).

The picture starts as the last clean frame of clip A. For every output frame, the
block motion of the source video (A's tail, then B's head) is measured, turned into
a smooth per-pixel flow and used to move the picture with bilinear sampling. So
A's picture streams like a liquid along the real motion, without visible blocks,
and over the last `fade` share of the window it blends into the real frames.

One pixel: if the flow at (x, y) is (-1.5, 0.25), the new value there is a bilinear
mix of the four old pixels around (x - 1.5, y + 0.25). Positions move by fractions
of a pixel, so edges smear instead of jumping block by block.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp


@register
class LiquidMelt(TransitionEffect):
    info = EffectInfo("liquid_melt", "Liquid Melt", "Casper Bach",
                      "Clip A's picture flows like a liquid along the video's motion, then fades into B.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("block", "Block size", 16, 8, 64),
            IntParam("search", "Search range", 8, 2, 32),
            FloatParam("strength", "Strength", 1.0, 0.0, 3.0),
            FloatParam("smoothness", "Smoothness", 0.5, 0.0, 1.0),
            FloatParam("fade", "Fade", 0.3, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        picture = frames.last_clean_a(ctx).astype(np.float32)
        fade_frames = int(round(p["fade"] * ctx.length))
        fade_start = ctx.length - fade_frames

        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            current = clip.frame(index).astype(np.float32)

            # Step 1 and 2: the clip's own motion into this frame, applied to our picture.
            # The first frame of a clip has no motion (that is where the I-frame was).
            if index > 0:
                flow = _flow(clip.frame(index - 1), clip.frame(index), ctx.size, p)
                picture = warp.warp(picture, p["strength"] * flow, sampling="bilinear")

            # Step 3: over the last fade_frames frames, blend towards the real frame,
            # reaching it exactly on the last frame of the window.
            out = picture
            if i >= fade_start:
                t = (i - fade_start + 1) / fade_frames
                out = picture + t * (current - picture)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _flow(previous, current, size, p) -> np.ndarray:
    """Smooth H x W x 2 flow from previous to current, as in docs/API.md.

    Block matching runs on every second pixel of the luma (a quarter of the work),
    with half the block and search, and the vectors are doubled back to full size.
    """
    half_block = max(p["block"] // 2, 1)
    half_search = max(p["search"] // 2, 1)
    reference = color.bgr_to_y(previous)[::2, ::2]
    target = color.bgr_to_y(current)[::2, ::2]
    vectors = 2 * motion.estimate(reference, target, half_block, half_search)
    return motion.smooth_field(vectors, size, 2 * half_block, p["smoothness"])
