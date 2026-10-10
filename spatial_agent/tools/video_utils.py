"""Video frame-access tool: the only way agent code turns "a time" or "an
absolute frame number" into a FrameImage.

InputImages[i] is indexed LOCALLY (0..len-1 over whatever was sampled);
every absolute frame number in the system (FrameImage.frame_index,
seg.frame_indices, recon.frame_indices) is an ABSOLUTE video frame number.
tools.Video always takes the absolute number / seconds and returns the
right FrameImage, so agent code never converts between the two.
"""
from __future__ import annotations

from typing import List, Optional

from spatial_agent.tools.base import CPUTool
from spatial_agent.tools.time_utils import frame_index_to_seconds, seconds_to_frame_index


class VideoUtils(CPUTool):
    """Frame/window access by time or absolute frame index."""

    TOOL_ABLATION_PREFIX = "tool_video"

    TOOL_PROMPT_DESCRIPTION = """
### tools.Video — Frames & Windows (CPU)

The way to get frames. Every result is a `FrameImage` (or a list of them)
that already knows its absolute frame index, so it can be passed straight
to `tools.SAM3`, `tools.TAP`, `tools.Reconstruct`, `vlm.locate`, `show()`.

| Method | Returns | Description |
|---|---|---|
| `frame_at_time(seconds)` | `FrameImage` | the sampled frame nearest to that time |
| `frame(index)` | `FrameImage` | the sampled frame nearest to that **absolute** frame index |
| `time_of(frame)` | `float` seconds | the frame's actual timestamp — **always use this for `dt`**, not the number you asked for (the nearest sampled frame may differ slightly) |
| `window(...)` | list of `FrameImage` | a span of frames |

`window(...)` — pick the span with **one** of:
`center_time=, radius=` · `start_time=, end_time=` · `start_frame=, end_frame=` · nothing (whole video).
Optionally `num_frames=N` (evenly keep at most N — use this to fit a tool's frame limit) or `step=k` (every k-th).

```python
frame  = tools.Video.frame_at_time(3.0)
window = tools.Video.window(center_time=3.0, radius=0.5, num_frames=16)
track  = tools.SAM3.track("white car", window)
recon  = tools.Reconstruct.run(window)
dt = tools.Video.time_of(window[-1]) - tools.Video.time_of(window[0])
```

`InputImages[i]` is a **local** index into the sampled list, not a frame number — use `tools.Video.frame(...)` when you have a frame number. Multi-video samples: add `video_index=` (1-indexed, matching `InputImages_<N>`).
"""

    def __init__(self, fps: float, total_frames: int, input_images=None,
                 input_images_list: Optional[List] = None):
        self.fps = fps
        self.total_frames = total_frames
        self._input_images = input_images
        self._input_images_list = input_images_list

    def _resolve(self, video_index: int = 1):
        if self._input_images_list is None or len(self._input_images_list) <= 1:
            ii = self._input_images
        else:
            n = len(self._input_images_list)
            if not isinstance(video_index, int) or not (1 <= video_index <= n):
                raise ValueError(
                    f"video_index={video_index!r} is out of range for a sample with {n} videos "
                    f"(1-indexed; video_index=1 -> InputImages_1)."
                )
            ii = self._input_images_list[video_index - 1]
        if ii is None or len(ii) == 0:
            raise RuntimeError("No frames available (InputImages is empty).")
        return ii

    def frame(self, index: int, video_index: int = 1):
        """The sampled `FrameImage` nearest to absolute video frame `index`."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"`index` must be an int absolute frame number, got {type(index).__name__}. "
                f"For seconds use frame_at_time()."
            )
        ii = self._resolve(video_index)
        fis = ii.frame_indices
        if index in fis:
            return ii[fis.index(index)]
        nearest = min(fis, key=lambda f: abs(f - index))
        print(f"[Video] frame {index} is not a sampled frame; returning nearest sampled frame {nearest} "
              f"(use tools.Video.time_of(frame) for its real time).")
        return ii[fis.index(nearest)]

    def frame_at_time(self, seconds: float, video_index: int = 1):
        """The sampled `FrameImage` nearest to time `seconds`."""
        return self.frame(seconds_to_frame_index(seconds, self.fps, self.total_frames), video_index=video_index)

    def time_of(self, frame) -> float:
        """Seconds for a `FrameImage` (or an absolute frame index)."""
        from spatial_agent.kernel_types.frame_image import FrameImage

        idx = frame.frame_index if isinstance(frame, FrameImage) else frame
        return frame_index_to_seconds(idx, self.fps)

    def window(self, center_time=None, radius=None, start_time=None, end_time=None,
               start_frame=None, end_frame=None, num_frames=None, step=None, video_index: int = 1):
        """A span of sampled frames (see class docs for the selection options)."""
        from spatial_agent.kernel_types.input_images import InputImages

        ii = self._resolve(video_index)
        fis = ii.frame_indices
        chosen = [x is not None for x in (center_time, start_time or end_time, start_frame or end_frame)]
        if sum(chosen) > 1:
            raise ValueError(
                "window() takes only one span selector: (center_time, radius) OR (start_time, end_time) "
                "OR (start_frame, end_frame) OR nothing."
            )
        if center_time is not None:
            if radius is None:
                raise ValueError("`radius` (seconds) is required with `center_time`.")
            lo = seconds_to_frame_index(center_time - radius, self.fps, self.total_frames)
            hi = seconds_to_frame_index(center_time + radius, self.fps, self.total_frames)
        elif start_time is not None or end_time is not None:
            lo = seconds_to_frame_index(start_time, self.fps, self.total_frames) if start_time is not None else fis[0]
            hi = seconds_to_frame_index(end_time, self.fps, self.total_frames) if end_time is not None else fis[-1]
        elif start_frame is not None or end_frame is not None:
            lo = start_frame if start_frame is not None else fis[0]
            hi = (end_frame - 1) if end_frame is not None else fis[-1]
        else:
            lo, hi = fis[0], fis[-1]

        subset = [f for f in ii if lo <= f.frame_index <= hi]
        if not subset:
            raise ValueError(
                f"window() selected no frames in absolute range [{lo}, {hi}] (available: [{fis[0]}, {fis[-1]}])."
            )
        if step is not None and num_frames is not None:
            raise ValueError("Pass either `step` or `num_frames`, not both.")
        if step is not None:
            subset = subset[::step]
        elif num_frames is not None and len(subset) > num_frames:
            import numpy as np

            idxs = sorted(set(int(i) for i in np.linspace(0, len(subset) - 1, num_frames).round()))
            subset = [subset[i] for i in idxs]
        return InputImages(subset, frame_indices=[f.frame_index for f in subset])

    def __repr__(self) -> str:
        return f"VideoUtils(fps={self.fps}, total_frames={self.total_frames}; frame, frame_at_time, time_of, window)"
