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
FADE_PART = 1 / 3   # share of the transition that the "fade" release blends into clip B


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
        fade_length = max(1, round(ctx.length * FADE_PART)) if p["release"] == "fade" else 0
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            # The same motion again, a little weaker every frame.
            strength = p["gain"] * (1.0 - p["decay"]) ** i
            picture = warp.warp(picture, strength * field)

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
