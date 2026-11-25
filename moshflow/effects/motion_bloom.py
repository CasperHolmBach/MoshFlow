"""Motion Bloom: one motion repeats, so moving pixels swell.

Emulates repeating one P-frame many times (Survey & Plan, 13.2). The motion
between two neighbouring frames is measured once, as a field F. Starting from
the last clean picture R of clip A, every output frame moves the picture along
F again: R(k+1)(x, y) = R(k)(x + s dx, y + s dy). A pixel inside a moving area
keeps being copied from the same direction, so the area stretches and swells,
while pixels where F is (0, 0) stay frozen.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

BLOCK = 16    # block size and search range for capturing the motion
SEARCH = 8


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "One motion repeats on the last picture of clip A, so moving areas swell.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("source", "Motion from", "B", ("A", "B")),
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("decay", "Decay", 0.02, 0.0, 0.5),
            ChoiceParam("release", "Release", "fade", ("cut", "fade")),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        field = _captured_field(ctx, p["source"])
        picture = frames.last_clean_a(ctx).astype(np.float32)
        for i in range(ctx.length):
            if progress.cancelled:
                return
            # The same motion again, a little weaker every frame.
            strength = p["gain"] * (1.0 - p["decay"]) ** i
            picture = warp.warp(picture, strength * field)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _captured_field(ctx, source: str) -> np.ndarray:
    """The motion between two neighbouring frames, as a per-pixel field (H, W, 2).

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
    vectors = motion.estimate(color.bgr_to_y(clip.frame(first)),
                              color.bgr_to_y(clip.frame(second)), BLOCK, SEARCH)
    return motion.to_field(vectors, ctx.size, BLOCK)
