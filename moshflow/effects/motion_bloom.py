"""Motion Bloom: clip B's P-frames are played on clip A's last picture.

Emulates the classic datamosh, as moshy's pdupe does on real video files:
clip B's I-frame is dropped, so B's P-frames are decoded on top of A's last
picture, and one P-frame is repeated many times to make the bloom.

A P-frame here is what a codec stores: a motion field F (one vector per pixel,
measured by block matching: F(x) says where the content at x came from) and a
residual E, the colour correction for what the motion could not predict:
E(x) = B_next(x) - B_prev(x + F(x)). Decoding it onto a picture P gives

    P_new(x) = P(x + F(x)) + E(x)

so each pixel is copied from where the motion says and then gets the same
colour difference that B had at that spot.

- Bloom (A's part of the window): one P-frame of B is decoded again and again,
  so moving areas of A's picture keep stretching and swelling. Like a pdupe
  user picking a good frame, it is the P-frame in B's first second with the most
  motion and the least residual, and only a little of its residual is repeated:
  repeating a full residual dozens of times burns the colours out.
- Drag (B's part of the window): B's real P-frames, one per frame, keep moving
  the picture with B's motion and paint B's changes into it. A's ghost survives
  where B is still.
- Emerge: on top of each of B's P-frames, every pixel also moves a step towards
  B's real frame: P += h (B - P). h grows over the drag and is larger where B's
  residual is large (where B changes), so B appears out of the mosh while it is
  already moving, starting where it moves, instead of starting clean. On the
  last frame h = 1, so the transition lands exactly on B.

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
LOOK_AHEAD = 30      # B frame pairs searched for the P-frame to repeat
RESIDUAL_COST = 0.3  # how much residual counts against motion when picking that P-frame
BLOOM_RESIDUAL = 0.1 # share of the residual kept while one P-frame is repeated
CHANGE_FULL = 16.0   # residual (in grey levels) at which a pixel emerges fastest
EMERGE_PART = 1 / 3  # with no B window, the share of the end in which B emerges


@register
class MotionBloom(TransitionEffect):
    info = EffectInfo("motion_bloom", "Motion Bloom (draft)", "Viktor Tost",
                      "Clip B's P-frames play on clip A's last picture; one repeats until it blooms.")

    @classmethod
    def param_specs(cls):
        return [
            FloatParam("gain", "Gain", 1.0, 0.0, 4.0),
            FloatParam("decay", "Decay", 0.0, 0.0, 0.5),
            FloatParam("residual", "Residual", 1.0, 0.0, 2.0),
            FloatParam("emerge", "Emerge", 1.0, 0.0, 3.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        b = ctx.incoming
        picture = _last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        field, error = _bloom_p_frame(b, first_b, ctx.size)
        error = BLOOM_RESIDUAL * error

        a_frames = len(ctx.a_window)
        # B emerges during B's window, or at the end if the transition has none.
        emerge_from = a_frames if len(ctx.b_window) else round(ctx.length * (1 - EMERGE_PART))
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < a_frames:
                # Bloom: the same P-frame again, a little weaker every frame with decay.
                strength = (1.0 - p["decay"]) ** i
                index = first_b
            else:
                # Drag: B's own next P-frame. A repeated (duplicate) frame carries no
                # P-frame of its own, so the previous motion carries on without residual.
                strength = 1.0
                index = ctx.b_window[i - a_frames]
                if index == 0:
                    pass
                elif _same(b.frame(index - 1), b.frame(index)):
                    error = np.zeros_like(error)
                else:
                    field, error = _p_frame(b, index - 1, index, ctx.size)

            picture = (warp.warp(picture, strength * p["gain"] * field, "bilinear")
                       + strength * p["residual"] * error)
            picture = np.clip(picture, 0.0, 255.0)

            if i >= emerge_from:
                target = b.frame(index).astype(np.float32)
                u = (i + 1 - emerge_from) / (ctx.length - emerge_from)
                change = np.minimum(np.abs(color.bgr_to_y(error)) / CHANGE_FULL, 1.0)
                h = np.minimum(p["emerge"] * u * u * (0.25 + 0.75 * change), 1.0)
                if i == ctx.length - 1:
                    h = np.ones_like(h)
                picture = picture + h[..., None] * (target - picture)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


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
