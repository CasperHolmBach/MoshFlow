"""Reading video properties for the UI, without it depending on OpenCV."""

from moshflow.videoio import VideoInfo, probe

__all__ = ["VideoInfo", "probe"]
