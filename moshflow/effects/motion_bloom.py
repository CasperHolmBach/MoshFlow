"""Motion Bloom: clip B's P-frames are played on clip A's last picture.

Emulates the classic datamosh transition: clip B's I-frame is dropped, so B's
P-frames are decoded on top of A's last picture, and a P-frame is duplicated (as
moshy's pdupe does on real video files) so the motion blooms. Both clips keep
their real speed: nothing is slowed down or stretched.

A P-frame here is what a codec stores: a motion field F (one vector per pixel,
measured by block matching: F(x) says where the content at x came from) and a
residual E, the colour correction for what the motion could not predict:
E(x) = B_next(x) - B_prev(x + F(x)). Decoding it onto a picture P gives

    P_new(x) = P(x + F(x)) + E(x)

so each pixel is copied from where the motion says and then gets the same
colour difference that B had at that spot.

The transition, frame by frame:
- A's part of the window (before the cut): the picture starts as A's last clean
  frame. A's own P-frames are decoded on it, so A keeps moving at its real speed
  up to the cut. On top, B's opening P-frame (its first real
  motion) is decoded again on every frame: the duplicated P-frame that makes the
  picture bloom along B's motion. Only B's motion is used here; nothing of B's
  picture is shown yet, because B has not started.
- B's part of the window (after the cut): B's own P-frames, one per frame, at B's
  real speed: frame k of the window decodes the P-frame from B's frame k - 1 to k.
  A repeated (duplicate) frame has no P-frame, so the previous motion carries on.
  B's residuals ramp up over the first 8 frames, so B's details do not burst in.
- Emerge: during B's part, each pixel also moves a step towards B's real frame:
  P += h (B - P). The steps follow a planned share of B per pixel: it starts
  rising at the cut where B changes most and 40 % into B's part where B is still,
  eases in and out, and reaches 1 on the last frame. A pixel is dragged only by
  the share of A still left in it, so finished pixels are not cracked by B's
  rough vectors.
- Exit: what is shown is the decoded picture mixed with B's real, sharp frame by
  share^2. The decoded picture is resampled every frame and slowly blurs; at the
  end the viewer sees B's actual footage. Where B's P-frame failed (B moves faster
  than the motion search reaches), B's real frame is shown sooner, like the intra
  blocks an encoder sends when prediction fails.
- With no B window (alignment "before"), B emerges over the last third, towards
  its first frame.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times becomes a random streak, so vectors that do not explain the motion
clearly better than (0, 0) are dropped, and block vectors are interpolated
between block centres so no hard block edges appear. Pixels are read with
bilinear sampling, so fractional vectors do not leave a zig-zag.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, warp

BLOCK = 16           # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %
LOOK_AHEAD = 30      # B frames searched for its opening motion
CHANGE_FULL = 16.0   # residual (in grey levels) at which a pixel emerges earliest
EARLY_START = 0.0    # share of B's part after which pixels where B changes turn into B
LATE_START = 0.4     # ... and pixels where B is still
RESIDUAL_RAMP = 8    # frames after the cut over which B's residuals reach full strength
NO_B_PART = 1 / 3    # with no B window, the share of the end in which B emerges


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "Clip B's P-frames play on clip A's last picture, one repeated, until B emerges.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
            FloatParam("emerge", "Emerge", 1.0, 0.0, 3.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        a, b = ctx.outgoing, ctx.incoming
        width, height = ctx.size
        picture = _last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        opening = _opening_motion(b, first_b, ctx.size)

        a_frames = len(ctx.a_window)
        emerge_from = a_frames if len(ctx.b_window) else round(ctx.length * (1 - NO_B_PART))
        field = opening
        change = np.zeros((height, width), np.float32)
        changed = np.zeros((height, width), np.float32)   # most B has changed at a pixel so far
        shown = np.zeros((height, width), np.float32)     # B's planned share at the last frame
        for i in range(ctx.length):
            if progress.cancelled:
                return
            error = None
            if i < a_frames:
                # Before the cut: B's opening P-frame again (duplicated), motion only.
                field, index = opening, first_b
            else:
                # After the cut: B's own next P-frame, at B's real speed.
                index = ctx.b_window[i - a_frames]
                if index > 0 and not _same(b.frame(index - 1), b.frame(index)):
                    field, error = _p_frame(b, index - 1, index, ctx.size)
                    change = np.minimum(np.abs(color.bgr_to_y(error)) / CHANGE_FULL, 1.0)
            target = b.frame(index).astype(np.float32)

            # Decode. A pixel that has already turned into B is dragged less, by the
            # share of A still left in it: B's real frames bring B's motion there
            # already, and dragging B's picture by B's rough vectors cracks it.
            left = (1.0 - shown)[..., None]
            picture = warp.warp(picture, p["gain"] * left * field, "bilinear")
            if error is not None:
                # B's residuals come in over the first frames after the cut, so B's
                # details do not burst onto A's picture all at once.
                ramp = min((i + 1 - a_frames) / RESIDUAL_RAMP, 1.0)
                picture = picture + p["residual"] * ramp * error

            # A keeps playing into the mosh: during A's window, A's own P-frames are
            # decoded on the same picture too, so A keeps moving at its real speed up
            # to the cut and its movement mixes with B's in the same pixels.
            if i < a_frames:
                a_index = ctx.a_window[i]
                if a_index > 0 and not _same(a.frame(a_index - 1), a.frame(a_index)):
                    a_field, a_error = _p_frame(a, a_index - 1, a_index, ctx.size)
                    picture = warp.warp(picture, a_field, "bilinear") + a_error
            picture = np.clip(picture, 0.0, 255.0)

            # Emerge: B's share of each pixel follows a planned curve over B's part.
            if i >= emerge_from:
                changed = np.maximum(changed, change)
                v = (i + 1 - emerge_from) / (ctx.length - emerge_from)
                share = _b_share(v, changed, p["emerge"])
                if i == ctx.length - 1:
                    h = np.ones_like(share)
                else:
                    h = np.where(share >= 1.0, 1.0,
                                 1.0 - (1.0 - share) / np.maximum(1.0 - shown, 1e-6))
                picture = picture + np.clip(h, 0.0, 1.0)[..., None] * (target - picture)
                shown = share

            # What is shown: the decoded picture, mixed with B's real, sharp frame by
            # share^2 (the decoded picture slowly blurs, so the end shows B's actual
            # footage), and sooner where B's P-frame failed: B moves faster than the
            # motion search reaches, its residual is large and mostly noise, and the
            # decoded picture is speckled and lags. An encoder sends fresh (intra)
            # blocks there. The failure is measured per 16 x 16 block and smoothed
            # between block centres, so no block edges appear.
            failed = _block_average(change, ctx.size)
            sharp = np.maximum(shown * shown, shown * failed)[..., None]
            out = picture + sharp * (target - picture)
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(out)


def _block_average(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Average of an H x W map over each 16 x 16 block, interpolated smoothly between
    block centres (the same bilinear interpolation as motion.smooth_field)."""
    height, width = values.shape
    rows, cols = -(-height // BLOCK), -(-width // BLOCK)
    counts = _block_sums(np.ones_like(values, np.float32), BLOCK)
    means = _block_sums(values.astype(np.float32), BLOCK) / counts
    pair = np.stack([means, means], axis=-1)
    return motion.smooth_field(pair, size, BLOCK, 1.0)[..., 0]


def _b_share(u: float, changed: np.ndarray, emerge: float) -> np.ndarray:
    """How much of B a pixel should show at progress u (0..1), per pixel.

    A pixel starts turning into B at `start`: at EARLY_START of B's part where
    B changes most, at LATE_START where B is still (earlier with a larger `emerge`).
    From there its share follows a smoothstep, r^2 (3 - 2 r) with r = (u - start) /
    (1 - start): it starts gently, and it also arrives gently, so the last frames
    are already almost B and the transition lands without a jump.
    """
    amount = np.minimum(emerge * changed, 1.0)
    start = LATE_START - (LATE_START - EARLY_START) * amount
    ramp = np.clip((u - start) / (1.0 - start), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _last_clean_a(ctx) -> np.ndarray:
    """A's frame just before the window (A's first frame if there is none)."""
    return ctx.outgoing.frame(max(ctx.a_window.start - 1, 0))


def _p_frame(clip, first: int, second: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The P-frame that turns frame `first` of a clip into frame `second`: (F, E).

    E(x) = second(x) - first(x + F(x)), the colour the motion alone cannot explain.
    """
    field = _field(clip, first, second, size)
    predicted = warp.warp(clip.frame(first), field, "bilinear")
    return field, clip.frame(second).astype(np.float32) - predicted


def _opening_motion(clip, start: int, size: tuple[int, int]) -> np.ndarray:
    """B's first real motion: the field of its first non-repeated frame pair.

    Film converted to 30 fps repeats frames, and a repeated frame has no motion,
    so the first pair can be empty; up to LOOK_AHEAD frames are searched.
    """
    width, height = size
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1)):
        if not _same(clip.frame(first), clip.frame(first + 1)):
            return _field(clip, first, first + 1, size)
    return np.zeros((height, width, 2), np.float32)


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
