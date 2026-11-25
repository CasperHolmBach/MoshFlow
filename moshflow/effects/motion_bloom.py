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

- Bloom (A's part of the window): B's strongest P-frame among its first frames
  is decoded again and again. Moving areas of A's picture keep stretching, and
  the residual keeps adding B's colour changes, so they swell and burn in.
- Drag (B's part of the window): B's real P-frames, one per frame. Where B
  moves or changes, its residuals paint B into the picture; where B is still,
  nothing corrects the picture and A's ghost survives.
- Intra refresh: over the last part of the transition (`refresh`), blocks are
  replaced one by one, in a random order from ctx.seed, by B's real block, the
  way encoders refresh a few intra blocks per frame. A refreshed block is then
  carried on by B's P-frames, so B takes over block by block; on the last frame
  every block has been refreshed and the video lands on B.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times becomes a random streak, so vectors that do not explain the motion
clearly better than (0, 0) are dropped, and block vectors are interpolated
between block centres so no hard block edges appear. Pixels are read with
bilinear sampling, so fractional vectors do not leave a zig-zag.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register
from moshflow.toolkit import color, motion, warp

BLOCK = 16          # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7    # keep a block's vector only if it cuts the block's error to below 70 %
LOOK_AHEAD = 8      # B frame pairs searched for the strongest P-frame to repeat


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
            FloatParam("refresh", "Refresh", 0.4, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        b = ctx.incoming
        picture = _last_clean_a(ctx).astype(np.float32)
        first_b = min(ctx.b_window.start, len(b) - 1)
        field, error = _strongest_p_frame(b, first_b, ctx.size)

        a_frames = len(ctx.a_window)

        def b_frame_at(i: int) -> np.ndarray:
            index = first_b if i < a_frames else ctx.b_window[i - a_frames]
            return b.frame(index).astype(np.float32)

        # Each 16 x 16 block gets a random moment in the last `refresh` share of the
        # transition at which it is refreshed; on the last frame every block is.
        width, height = ctx.size
        rows, cols = -(-height // BLOCK), -(-width // BLOCK)
        rng = np.random.default_rng(ctx.seed)
        refresh_at = 1.0 - p["refresh"] * rng.random((rows, cols))
        for i in range(ctx.length):
            if progress.cancelled:
                return
            if i < a_frames:
                # Bloom: the same P-frame again, a little weaker every frame with decay.
                strength = (1.0 - p["decay"]) ** i
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

            # Intra refresh: from here on, a block whose turn has come is replaced by B's
            # real block. Later P-frames keep it correct, so B takes over block by block.
            t = (i + 1) / ctx.length
            fresh = _pixels(refresh_at <= t, ctx.size)
            if fresh.any():
                picture[fresh] = b_frame_at(i)[fresh]
            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _pixels(blocks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Expands a (rows, cols) block mask to an (H, W) pixel mask."""
    width, height = size
    return blocks[(np.arange(height) // BLOCK)[:, None], (np.arange(width) // BLOCK)[None, :]]


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


def _strongest_p_frame(clip, start: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The P-frame to repeat: of B's first LOOK_AHEAD frame pairs, the one that moves most.

    Film converted to 30 fps repeats frames, and a repeated frame has no motion,
    so simply taking the first pair can give nothing to repeat.
    """
    width, height = size
    best = (np.zeros((height, width, 2), np.float32), np.zeros((height, width, 3), np.float32))
    best_amount = 0.0
    for first in range(start, min(start + LOOK_AHEAD, len(clip) - 1)):
        if _same(clip.frame(first), clip.frame(first + 1)):
            continue
        field = _field(clip, first, first + 1, size)
        amount = float((field[..., 0] ** 2 + field[..., 1] ** 2).mean())
        if amount > best_amount:
            predicted = warp.warp(clip.frame(first), field, "bilinear")
            best = (field, clip.frame(first + 1).astype(np.float32) - predicted)
            best_amount = amount
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
