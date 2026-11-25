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
- It starts from A's last clean picture. During A's window, A's own P-frames are
  decoded on the same picture as well, with a weight that fades from 1 to 0, so A
  keeps moving inside the mosh and its motion mixes with B's in the same pixels.
- B's P-frames, from B's first frame up to the end of B's window, are spread over
  the whole transition: output frame i is at a fractional position s among the
  P-frames (see _b_position). Early on s moves at 0.1 P-frames per frame, so
  each P-frame's motion is decoded for several frames in a row; repeating the
  motion makes it bloom, so moving areas keep stretching and swelling. Then B
  speeds up, and in the last part s moves one P-frame per frame, so the B that
  becomes visible plays at its real speed and never shows a frame twice.
- Nothing jumps: the motion and B's frame blend smoothly from one P-frame to the
  next, and each P-frame's residual is spread over its copies (the step of s per
  frame), so it adds up to the residual once instead of landing in one frame.
  Adding the full residual on every copy would burn the colours out.
- Emerge: after every frame, each pixel also moves a step towards B's nearest
  real frame at that point: P += h (B - P). The steps follow a planned share of B
  per pixel: it starts rising at 15 % of the transition where B changes most and
  at 45 % where B is still, eases in and out, and reaches 1 on the last frame.
  So B grows out of the same motion that smears A, starting where B moves, and
  A's ghost lasts until the end instead of B finishing early and crawling.
  A pixel is dragged only by the share of A still left in it, so finished pixels
  are not cracked by B's rough vectors.
- Exit: what is shown is the decoded picture mixed with B's real, sharp frame by
  share^2. The decoded picture is resampled every frame and slowly blurs; at the
  end the viewer sees B's actual footage, so the exit is sharp. Where B's P-frame
  failed (B moves faster than the motion search reaches), B's real frame is shown
  sooner, like the intra blocks an encoder sends when prediction fails.
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
CHANGE_FULL = 16.0   # residual (in grey levels) at which a pixel emerges earliest
EARLY_START = 0.15   # share of the transition after which pixels where B changes turn into B
LATE_START = 0.45    # ... and pixels where B is still
SPEED_RAMP = 0.2     # share of the transition over which B speeds up to its real speed
SLOW_SPEED = 0.1     # B's speed (P-frames per output frame) during the bloom at the start


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
        a, b = ctx.outgoing, ctx.incoming
        picture = _last_clean_a(ctx).astype(np.float32)
        pairs = _b_pairs(ctx)
        if pairs:
            cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

            def decoded(j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                """P-frame j as (F, E, change), kept for the current and the next one."""
                j = min(j, len(pairs) - 1)
                if j not in cache:
                    for old in [k for k in cache if k < j - 1]:
                        del cache[old]
                    field, error = _p_frame(b, *pairs[j], ctx.size)
                    change = np.minimum(np.abs(color.bgr_to_y(error)) / CHANGE_FULL, 1.0)
                    cache[j] = (field, error, change)
                return cache[j]
        else:
            still = min(ctx.b_window.start, len(b) - 1)
            field, error = _bloom_p_frame(b, still, ctx.size)
            change = np.minimum(np.abs(color.bgr_to_y(error)) / CHANGE_FULL, 1.0)
            target = b.frame(still).astype(np.float32)

        position = 0.0
        changed = np.zeros(picture.shape[:2], np.float32)   # most B has changed at a pixel so far
        shown = np.zeros(picture.shape[:2], np.float32)     # B's planned share at the last frame
        for i in range(ctx.length):
            if progress.cancelled:
                return
            u = (i + 1) / ctx.length
            if pairs:
                # Where in B's P-frames this output frame is (fractional, see
                # _b_position): first B's first P-frame is repeated (the bloom), then
                # B speeds up, and in the last part it plays at its real speed, so the
                # B that becomes visible never shows a frame twice.
                new_position = _b_position(u, len(pairs), ctx.length)
                step, position = new_position - position, new_position
                j = min(int(position), len(pairs) - 1)
                f = min(position - j, 1.0)
                field_j, error, change = decoded(j)
                field_next = decoded(j + 1)[0]
                # Between two P-frames the motion blends smoothly, so nothing jumps
                # when the next P-frame takes over. The pull goes to B's nearest real
                # frame: blending two frames of a fast pan would show a double image.
                field = (1.0 - f) * field_j + f * field_next
                target = b.frame(pairs[j][1] if f >= 0.5 else pairs[j][0]).astype(np.float32)
            else:
                step = 1.0 / ctx.length

            # Decode: the motion on every frame; the residual spread over the copies
            # of its P-frame (step of it per frame), so the colour arrives evenly
            # and adds up to the residual once.
            # A pixel that has already turned into B is dragged less, by the share of
            # A still left in it: B's real frames bring B's motion there already, and
            # dragging B's picture by B's rough vectors again cracks it into cells.
            left = (1.0 - shown)[..., None]
            picture = (warp.warp(picture, p["gain"] * left * field, "bilinear")
                       + p["residual"] * step * error)

            # A keeps playing into the mosh: during A's window, A's own P-frames are
            # decoded on the same picture too, with a weight that fades from 1 to 0,
            # so A's movement and B's movement mix in the same pixels.
            if i < len(ctx.a_window):
                index = ctx.a_window[i]
                if index > 0 and not _same(a.frame(index - 1), a.frame(index)):
                    a_field, a_error = _p_frame(a, index - 1, index, ctx.size)
                    a_weight = 1.0 - i / len(ctx.a_window)
                    picture = warp.warp(picture, a_weight * a_field, "bilinear") + a_weight * a_error
            picture = np.clip(picture, 0.0, 255.0)

            # Emerge: B's share of each pixel follows a planned curve, so A's ghost
            # lasts until the end of the transition whatever its length.
            changed = np.maximum(changed, change)
            share = _b_share(u, changed, p["emerge"])
            if i == ctx.length - 1:
                h = np.ones_like(share)
            else:
                h = np.where(share >= 1.0, 1.0, 1.0 - (1.0 - share) / np.maximum(1.0 - shown, 1e-6))
            h = np.clip(h, 0.0, 1.0)
            picture = picture + h[..., None] * (target - picture)
            shown = share

            # What is shown: the decoded picture, mixed with B's real, sharp frame by
            # share^2. The decoded picture is resampled every frame and slowly blurs;
            # near the end this shows B's actual footage, so the exit is sharp and
            # moves at B's own speed.
            # Where B's current P-frame failed (B moves faster than the motion search
            # reaches, so its residual is large and mostly noise), the decoded picture
            # is speckled and lags behind B. There B's real frame is shown sooner, up to
            # B's share: what an encoder does when prediction fails, sending fresh
            # (intra) blocks. The failure is measured per 16 x 16 block and smoothed
            # between block centres, so no block edges appear.
            failed = _block_average(change, ctx.size)
            sharp = np.maximum(share * share, share * failed)[..., None]
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


def _b_position(u: float, count: int, length: int) -> float:
    """Position among B's `count` P-frames at progress u (0..1); count at u = 1.

    B's speed (P-frames per output frame) is SLOW_SPEED until u_a, rises linearly
    to 1 at u_b = u_a + SPEED_RAMP, and stays 1 after that. Below 1, each P-frame
    is decoded for several output frames in a row (duplicated): the bloom. At 1,
    B plays at its real speed, so the end shows no frame twice. u_b is chosen so
    that about `count` P-frames are used:
    SLOW_SPEED + (1 - SLOW_SPEED) ((1 - u_b) + SPEED_RAMP / 2) = count / length.
    """
    ramp, slow = SPEED_RAMP, SLOW_SPEED
    fast_part = (count / length - slow) / (1.0 - slow)
    u_b = min(max(1.0 + ramp / 2.0 - fast_part, ramp), 1.0)
    u_a = u_b - ramp

    def frames_used(x: float) -> float:
        extra = 0.0
        if x > u_a:
            extra = (min(x, u_b) - u_a) ** 2 / (2.0 * ramp) + max(x - u_b, 0.0)
        return slow * x + (1.0 - slow) * extra

    return count * frames_used(u) / frames_used(1.0)


def _b_share(u: float, changed: np.ndarray, emerge: float) -> np.ndarray:
    """How much of B a pixel should show at progress u (0..1), per pixel.

    A pixel starts turning into B at `start`: at EARLY_START of the transition where
    B changes most, at LATE_START where B is still (earlier with a larger `emerge`).
    From there its share follows a smoothstep, r^2 (3 - 2 r) with r = (u - start) /
    (1 - start): it starts gently, and it also arrives gently, so the last frames
    are already almost B and the transition lands without a jump.
    """
    amount = np.minimum(emerge * changed, 1.0)
    start = LATE_START - (LATE_START - EARLY_START) * amount
    ramp = np.clip((u - start) / (1.0 - start), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


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
