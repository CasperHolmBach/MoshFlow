"""Liquid Melt (owner: Casper Bach).

A lost I-frame with smooth motion in place of blocks (Survey & Plan, 13.3):
clip B's motion becomes a smooth flow, so clip A's picture streams like a
liquid before it fades into clip B.

SKELETON: render() is a plain cut until the TODO steps are written. When the
effect works, remove "(draft)" from its name.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

SEARCH = 8    # search range for the block matching; may become a parameter


@register
class LiquidMelt(TransitionEffect):
    info = EffectInfo("liquid_melt", "Liquid Melt (draft)", "Casper Bach",
                      "Clip A's picture flows like a liquid along clip B's motion, then fades into B.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("block", "Block size", 16, 8, 64),
            FloatParam("strength", "Strength", 1.0, 0.0, 3.0),
            FloatParam("smoothness", "Smoothness", 0.5, 0.0, 1.0),
            FloatParam("fade", "Fade", 0.3, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        picture = frames.last_clean_a(ctx).astype(np.float32)
        previous = picture
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            current = clip.frame(index).astype(np.float32)

            # TODO(Casper) Step 1: block motion between the previous and the current source
            # frame, turned into a smooth per-pixel flow:
            #   vectors = motion.estimate(color.bgr_to_y(previous), color.bgr_to_y(current), p["block"], SEARCH)
            #   flow = motion.smooth_field(vectors, ctx.size, p["block"], p["smoothness"])
            # TODO(Casper) Step 2: move the picture along the flow with bilinear sampling:
            #   picture = warp.warp(picture, p["strength"] * flow, sampling="bilinear")
            # TODO(Casper) Step 3: over the last `fade` share of the window, blend towards clip B.
            picture = current  # placeholder: a plain cut until the steps above exist

            previous = current
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)
