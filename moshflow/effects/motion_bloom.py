"""Motion Bloom: an optional sample skeleton.

Emulates repeating one P-frame many times (Survey & Plan, 13.2): one motion is
applied again and again, so moving areas stretch and swell while static areas
stay frozen.

SKELETON: render() is a plain cut until the TODO steps are written. Finish it,
change it, or delete this file and write your own effect. When the
effect works, remove "(draft)" from its name.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

BLOCK = 16    # block size and search range for capturing the motion; may become parameters
SEARCH = 8


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Sample",
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
        # TODO Step 1: capture one motion field F, between two frames of
        # A's tail (source "A") or of B's head (source "B"), for example:
        #   vectors = motion.estimate(color.bgr_to_y(first), color.bgr_to_y(second), BLOCK, SEARCH)
        #   field = motion.to_field(vectors, ctx.size, BLOCK)
        picture = frames.last_clean_a(ctx).astype(np.float32)
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            current = clip.frame(index).astype(np.float32)

            # TODO Step 2: R(k+1) = warp(R(k), s * F), with the gain s shrinking by
            # `decay` every frame, e.g. s = p["gain"] * (1 - p["decay"]) ** i:
            #   picture = warp.warp(picture, s * field)
            # TODO Step 3: release into clip B at the end ("cut", or "fade" over the last frames).
            picture = current  # placeholder: a plain cut until the steps above exist

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)
