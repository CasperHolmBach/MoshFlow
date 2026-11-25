"""Motion Bloom: clip B's P-frames, each repeated, are played on clip A's last picture.

Emulates the classic datamosh transition: clip B's I-frame is dropped, so B's
P-frames are decoded on top of A's last picture, and P-frames are duplicated
(as moshy's pdupe and ppulse do on real video files) so the motion blooms.

A P-frame here is what a codec stores: a motion field F (one vector per pixel,
measured by block matching: F(x) says where the content at x came from) and a
residual E, the colour correction for what the motion could not predict:
E(x) = B_next(x) - B_prev(x + F(x)). Decoding it onto a picture P gives

    P_new(x) = P(x + F(x)) + E(x)

so each pixel is copied from where the motion says and then gets the same
colour difference that B had at that spot.

The transition, frame by frame:
- It starts from A's last clean picture.
- B's P-frames, from B's first frame up to the end of B's window, are spread over
  the whole transition: output frame i shows P-frame j = m u^2, with m P-frames
  and u = (i + 1) / length. The first ones are decoded many times in a row;
  repeating the motion makes it bloom, so moving areas keep stretching and
  swelling. Later ones come one per frame, so B is at its normal speed when the
  transition ends, and the picture never stops moving in between. The residual is added only on the first copy;
  adding it again on every copy would burn the colours out.
- Emerge: after every frame, each pixel also moves a step towards B's real frame
  at that point: P += h (B - P). h grows from 0 to 1 over the transition and is
  larger where B's residual is large (where B changes), so B grows out of the
  same motion that smears A, starting where B moves. On the last frame h = 1,
  and the video carries on with B's next real frame.
- With no B window (alignment "before"), B has no P-frames before its first
  frame, so B's cleanest early P-frame is repeated instead.

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
LOOK_AHEAD = 30      # B frame pairs searched for the P-frame to repeat (no B window)
RESIDUAL_COST = 0.3  # how much residual counts against motion when picking that P-frame
CHANGE_FULL = 16.0   # residual (in grey levels) at which a pixel emerges fastest


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "Clip B's P-frames, each repeated, play on clip A's last picture until B emerges.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
            FloatParam("emerge", "Emerge", 1.0, 0.0, 3.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        b = ctx.incoming
        picture = _last_clean_a(ctx).astype(np.float32)
        pairs = _b_pairs(ctx)
        if not pairs:
            field, error = _bloom_p_frame(b, ctx.b_window.start, ctx.size)

        current, change = -1, np.zeros(picture.shape[:2], np.float32)
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if pairs:
                # The P-frame of B that this output frame shows: j = m u^2. Early ones
                # stay for many frames in a row (duplicated, the bloom); towards the end
                # one P-frame per frame, so B reaches its normal speed as it lands.
                j = min(int(len(pairs) * ((i + 1) / ctx.length) ** 2), len(pairs) - 1)
                first_copy = j != current
                if first_copy:
                    current = j
                    field, error = _p_frame(b, *pairs[j], ctx.size)
                    change = np.minimum(np.abs(color.bgr_to_y(error)) / CHANGE_FULL, 1.0)
                target_index = pairs[j][1]
            else:
                first_copy = i == 0
                target_index = min(ctx.b_window.start, len(b) - 1)

            picture = warp.warp(picture, p["gain"] * field, "bilinear")
            if first_copy:
                picture = picture + p["residual"] * error
            picture = np.clip(picture, 0.0, 255.0)

            # Emerge: a step towards B's real frame, bigger later and where B changes.
            u = (i + 1) / ctx.length
            h = np.minimum(p["emerge"] * u * u * (0.25 + 0.75 * change), 1.0)
            if i == ctx.length - 1:
                h = np.ones_like(h)
            picture = picture + h[..., None] * (b.frame(target_index).astype(np.float32) - picture)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _b_pairs(ctx) -> list[tuple[int, int]]:
    """B's P-frames up to the end of its window, as (previous, next) frame pairs.

    Repeated (duplicate) frames carry no P-frame of their own and are skipped.
    """
    b = ctx.incoming
    last = min(ctx.b_window.stop, len(b)) - 1
    pairs, previous = [], 0
    for index in range(1, last + 1):
        if not _same(b.frame(previous), b.frame(index)):
            pairs.append((previous, index))
            previous = index
    return pairs


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


def _bloom_p_frame(clip, start: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The P-frame to repeat: in B's first LOOK_AHEAD frame pairs, the best motion.

    Score = mean vector length - RESIDUAL_COST * mean |residual|: lots of clean
    motion, little that the motion cannot explain. Repeated (duplicate) frames
    have no motion and are skipped; film converted to 30 fps has many.
    """
    width, height = size
    best = (np.zeros((height, width, 2), np.float32), np.zeros((height, width, 3), np.float32))
    best_score = 0.0
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1)):
        if _same(clip.frame(first), clip.frame(first + 1)):
            continue
        field, error = _p_frame(clip, first, first + 1, size)
        score = (float(((field[..., 0] ** 2 + field[..., 1] ** 2) ** 0.5).mean())
                 - RESIDUAL_COST * float(np.abs(error).mean()))
        if score > best_score:
            best, best_score = (field, error), score
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
