"""Liquid Melt: clip A's pixels run sideways out of the frame and clip B's pixels
run in behind them (Survey & Plan, 13.3, a lost I-frame with smooth motion).

Every row of the picture has a front that slides across the frame. Rows start at
slightly different times (`spread`), so the front is ragged like a liquid edge.

- Ahead of the front is clip A's picture, starting from its last clean frame. It is
  still moved by the video's own block motion, as a smooth flow (the datamosh part),
  and is carried sideways by the front.
- Behind the front is clip B, carried in from the other side and settling exactly
  into place when the row's front reaches the far edge.
- Both pictures are smeared along the direction of travel (`smear`) and ripple up
  and down while they move (`wobble`).

One pixel, direction "right": if its row's front is at D = 120 and the pixel is at
x = 200, it shows clip A from x = 80 (200 - 120), so A has moved 120 pixels right.
At x = 50 it shows clip B from x = 50 + (W - 120): B's right edge is still entering.
"""

import numpy as np

from moshflow.api import ChoiceParam, EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

CUT_LEVEL = 40.0    # mean luma difference above which two frames are a scene cut, not motion
SMEAR_SAMPLES = 4   # samples averaged along the direction of travel
EDGE = 4.0          # width in pixels of the soft edge between A and B


@register
class LiquidMelt(TransitionEffect):
    info = EffectInfo("liquid_melt", "Liquid Melt", "Casper Bach",
                      "Clip A's pixels melt sideways out of the frame while clip B's pixels flow in.")

    @classmethod
    def param_specs(cls):
        return [
            ChoiceParam("direction", "Direction", "right", ("right", "left")),
            FloatParam("spread", "Raggedness", 0.3, 0.0, 1.5),
            FloatParam("smear", "Smear", 1.5, 0.0, 4.0),
            FloatParam("wobble", "Wobble", 0.5, 0.0, 3.0),
            FloatParam("strength", "Motion strength", 1.0, 0.0, 3.0),
            IntParam("block", "Block size", 16, 8, 64),
            IntParam("search", "Search range", 8, 2, 32),
            FloatParam("smoothness", "Smoothness", 1.0, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        rng = np.random.default_rng(ctx.seed)
        sign = 1.0 if p["direction"] == "right" else -1.0
        # x measured along the direction of travel: 0 where B enters, width - 1 where A leaves.
        along = np.arange(width, dtype=np.float32)
        if sign < 0:
            along = along[::-1].copy()

        lag = _row_noise(height, rng)                                     # 0..1 per row
        ripple = _smooth_noise(ctx.size, 4 * p["block"], rng)[..., 1]     # -1..1 per pixel
        picture = frames.last_clean_a(ctx).astype(np.float32)
        first_b = ctx.incoming.frame(ctx.b_window.start if len(ctx.b_window) else 0)
        front_before = np.zeros(height, np.float32)

        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            t = (i + 1) / ctx.length

            # The datamosh part: A's picture keeps following the video's motion.
            if index > 0 and p["strength"] > 0:
                flow = _flow(clip.frame(index - 1), clip.frame(index), ctx.size, p)
                picture = warp.warp(picture, p["strength"] * flow, sampling="bilinear")

            # Where every row's front is: rows start at different times, then ease in and out.
            u = np.clip(t * (1.0 + p["spread"]) - p["spread"] * lag, 0.0, 1.0)
            front = (width + EDGE) * u * u * (3.0 - 2.0 * u)
            speed = front - front_before
            front_before = front

            target = clip.frame(index) if clip is ctx.incoming else first_b
            shift_a = -front                      # A moved forward by `front`
            shift_b = width - front               # B still has `width - front` to go
            out_a = _smeared(picture, sign, shift_a, speed, ripple, p)
            out_b = _smeared(target, sign, shift_b, speed, ripple, p)

            # A where the pixel is ahead of its row's front, B behind it, soft edge between.
            alpha = np.clip((along[None, :] - front[:, None]) / EDGE + 0.5, 0.0, 1.0)[..., None]
            out = out_b + alpha * (out_a - out_b)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _smeared(image, sign, shift, speed, ripple, p) -> np.ndarray:
    """The image moved along the direction of travel by `shift` pixels per row,
    averaged over SMEAR_SAMPLES positions trailing behind it (motion blur), with a
    vertical ripple that grows with the row's speed."""
    height, width = image.shape[:2]
    field = np.zeros((height, width, 2), np.float32)
    field[..., 1] = p["wobble"] * ripple * np.minimum(speed, 20.0)[:, None] / 4.0
    samples = SMEAR_SAMPLES if p["smear"] > 0 and speed.max() > 0.5 else 1
    total = np.zeros(image.shape, np.float32)
    for k in range(samples):
        trail = p["smear"] * speed * k / samples
        field[..., 0] = (sign * (shift + trail))[:, None]
        total += warp.warp(image, field, sampling="bilinear")
    return total / samples


def _flow(previous, current, size, p) -> np.ndarray:
    """Smooth H x W x 2 flow from previous to current; zero across a scene cut.

    Block matching runs on every second pixel of the luma, with half the block and
    search, and the vectors are doubled back to full size.
    """
    reference = color.bgr_to_y(previous)[::2, ::2]
    target = color.bgr_to_y(current)[::2, ::2]
    width, height = size
    if np.abs(target - reference).sum() / target.size > CUT_LEVEL:
        return np.zeros((height, width, 2), np.float32)
    half_block = max(p["block"] // 2, 1)
    half_search = max(p["search"] // 2, 1)
    vectors = 2 * motion.estimate(reference, target, half_block, half_search)
    return motion.smooth_field(vectors, size, 2 * half_block, p["smoothness"])


def _smooth_noise(size: tuple[int, int], cell: int, rng) -> np.ndarray:
    """H x W x 2 smooth random field in -1..1: one random vector per `cell` x `cell`
    block, interpolated bilinearly between block centres by motion.smooth_field."""
    width, height = size
    rows, cols = -(-height // cell), -(-width // cell)
    knots = rng.random((rows, cols, 2)).astype(np.float32) * 2.0 - 1.0
    return motion.smooth_field(knots, size, cell, 1.0)


def _row_noise(height: int, rng) -> np.ndarray:
    """A value in 0..1 for every row, so neighbouring rows move together: broad lobes
    (knots every height / 5 rows) plus thin fingers (every height / 24 rows)."""
    broad = _curve(height, max(height // 5, 2), rng)
    fine = _curve(height, max(height // 24, 2), rng)
    return 0.75 * broad + 0.25 * fine


def _curve(length: int, spacing: int, rng) -> np.ndarray:
    """Random knots in 0..1 every `spacing` samples, joined with smoothstep weights
    w * w * (3 - 2w), so the curve has no sharp corners at the knots."""
    knots = rng.random(length // spacing + 2).astype(np.float32)
    position = np.arange(length, dtype=np.float32) / spacing
    low = position.astype(np.intp)
    w = position - low
    w = w * w * (3.0 - 2.0 * w)
    return (1.0 - w) * knots[low] + w * knots[low + 1]
