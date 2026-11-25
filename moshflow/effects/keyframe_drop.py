"""Keyframe Drop (owner: Nicolai Hansen).

Emulates deleting the I-frame at the start of clip B (Survey & Plan, 13.1):
clip B's motion drags the last picture of clip A, which slowly heals into B.

SKELETON: render() is a plain cut until the TODO steps are written. When the
effect works, remove "(draft)" from its name.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp


@register
class KeyframeDrop(TransitionEffect):
    info = EffectInfo("keyframe_drop", "Keyframe Drop (draft)", "Nicolai Hansen",
                      "Clip B's motion drags the last picture of clip A, which slowly heals into B.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("block", "Block size", 16, 4, 64),
            IntParam("search", "Search range", 8, 1, 32),
            FloatParam("heal", "Healing", 0.05, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        # Step 1: start from R, the last clean frame of A before the window.
        picture = frames.last_clean_a(ctx).astype(np.float32)
        previous = picture
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            current = clip.frame(index).astype(np.float32)

            # TODO(Nicolai) Step 2: motion between the previous and the current source frame,
            # applied to our picture, then a fraction `heal` of clip B mixed back in:
            #   vectors = motion.estimate(color.bgr_to_y(previous), color.bgr_to_y(current), p["block"], p["search"])
            #   picture = warp.warp(picture, motion.to_field(vectors, ctx.size, p["block"]))
            #   picture = picture + p["heal"] * (current - picture)
            # TODO(Nicolai) Step 3: settle into the real frames of clip B by the end of the window.
            picture = current  # placeholder: a plain cut until the steps above exist

            previous = current
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)
