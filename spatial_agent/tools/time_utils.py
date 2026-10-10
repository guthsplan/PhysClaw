"""Time utility module for frame index <-> seconds conversion."""

from spatial_agent.tools.base import CPUTool


def seconds_to_frame_index(seconds: float, fps: float, total_frames: int = 0) -> int:
    """Pure conversion: seconds -> nearest absolute video frame index (clamped).

    Shared by ``TimeUtils.seconds_to_frame`` and ``tools.Video`` so both stay
    in sync — this is the single place that owns the seconds->frame formula.
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"`seconds` must be a number, got {type(seconds).__name__}.")
    if fps <= 0:
        return 0
    idx = int(round(seconds * fps))
    if total_frames > 0:
        idx = max(0, min(idx, total_frames - 1))
    return max(0, idx)


def frame_index_to_seconds(frame_index: int, fps: float) -> float:
    """Pure conversion: absolute frame index -> seconds. See ``seconds_to_frame_index``."""
    if not isinstance(frame_index, (int, float)):
        raise TypeError(f"`frame_index` must be a number, got {type(frame_index).__name__}.")
    if fps <= 0:
        return 0.0
    return frame_index / fps


class TimeUtils(CPUTool):
    """Frame index and timestamp conversion utilities.

    Usage::

        tools.Time.frame_to_seconds(42)       # -> 1.4 (at 30fps)
        tools.Time.seconds_to_frame(1.4)       # -> 42
        tools.Time.frame_range_to_seconds(0, 30)  # -> 1.0
    """

    TOOL_PROMPT_DESCRIPTION = """
### tools.Time — Frame/Time Conversion (CPU)

Pure numeric conversion only — these return bare numbers, not frames. To go
from a time/frame number to an actual frame you can pass to SAM3 /
Reconstruct / TAP, use **`tools.Video`** instead (`tools.Video.frame_at_time(t)`,
`tools.Video.frame(i)`) — `InputImages[tools.Time.seconds_to_frame(t)]` is
WRONG in general, because `InputImages[i]` is a **local** index into the
sampled-frame list, while this returns an **absolute** video frame number;
the two only coincide when every frame was sampled.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `frame_to_seconds` | `(frame_index)` | `float` | Convert absolute frame index to seconds |
| `seconds_to_frame` | `(seconds)` | `int` | Convert seconds to nearest absolute frame index (clamped) |
| `frame_range_to_seconds` | `(start_frame, end_frame)` | `float` | Duration between two absolute frame indices |

Uses the video's FPS and total frame count (auto-configured from Metadata). Only useful when `Metadata.is_video is True` and `Metadata.fps` is set.
"""

    def __init__(self, fps: float, total_frames: int):
        self.fps = fps
        self.total_frames = total_frames

    def frame_to_seconds(self, frame_index: int) -> float:
        """Convert a frame index to seconds."""
        return frame_index_to_seconds(frame_index, self.fps)

    def seconds_to_frame(self, seconds: float) -> int:
        """Convert seconds to the nearest frame index (clamped)."""
        return seconds_to_frame_index(seconds, self.fps, self.total_frames)

    def frame_range_to_seconds(self, start_frame: int, end_frame: int) -> float:
        """Duration in seconds between two frame indices."""
        if self.fps <= 0:
            return 0.0
        return (end_frame - start_frame) / self.fps

    def __repr__(self) -> str:
        return f"TimeUtils(fps={self.fps}, total_frames={self.total_frames})"
