"""ConformedSource: one clip seen at the project's size and frame rate.

It implements the FrameSource interface on top of any reader that offers
`info` (fps, frame_count) and `read(index)`.
"""

from typing import Protocol

import numpy as np

from moshflow.core.cache import LRUCache
from moshflow.toolkit import resize


class FrameReader(Protocol):
    info: object  # needs .fps and .frame_count

    def read(self, index: int) -> np.ndarray: ...


class ConformedSource:
    def __init__(self, reader: FrameReader, size: tuple[int, int], fps: float,
                 start: float = 0.0, end: float | None = None, cache_size: int = 48):
        self.size = size
        self.fps = fps
        self._reader = reader
        src_fps = reader.info.fps
        total = reader.info.frame_count
        self._first = min(int(round(start * src_fps)), total - 1)
        last = total if end is None else min(total, int(round(end * src_fps)))
        self._src_len = max(1, last - self._first)
        self._step = src_fps / fps
        self._len = max(1, int(round(self._src_len / self._step)))
        self._cache: LRUCache[np.ndarray] = LRUCache(cache_size)

    def __len__(self) -> int:
        return self._len

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self._len:
            raise IndexError(f"frame {index} out of range 0..{self._len - 1}")
        # Nearest-frame resampling from the source frame rate.
        src = self._first + min(int(index * self._step + 0.5), self._src_len - 1)
        return self._cache.get_or_create(src, lambda: self._load(src))

    def _load(self, src: int) -> np.ndarray:
        frame = resize.bilinear(self._reader.read(src), self.size)
        frame = np.ascontiguousarray(frame)
        frame.flags.writeable = False
        return frame
