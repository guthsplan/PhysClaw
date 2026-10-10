"""TAP point-tracking tool -- exposes TAPNext++ as a kernel primitive.
Deliberately has no opinion on strategy: what to seed and how far to track
are the agent's decisions.

Loaded in-process (no GPU-server registry)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from spatial_agent.tools.base import CPUTool
from spatial_agent.tools.time_utils import frame_index_to_seconds

_CKPT_ENV = "TAPNEXTPP_CHECKPOINT"
_DEFAULT_CKPT = "/root/data2/quantiphy_agent/checkpoints/tapnextpp_512.ckpt"
_DEVICE_ENV = "TAPNEXTPP_DEVICE"


class PointTrack:
    """One point tracked across a window (``tools.TAP.track``). Frame indices are absolute."""

    def __init__(self, positions: np.ndarray, visible: np.ndarray, frame_indices, fps: float):
        self.frame_indices = list(frame_indices)
        self._positions = positions  # (T, 2)
        self._visible = visible      # (T,) bool
        self._fps = fps

    def _local(self, frame) -> int:
        from spatial_agent.kernel_types.frame_image import FrameImage

        idx = frame.frame_index if isinstance(frame, FrameImage) else frame
        if idx not in self.frame_indices:
            raise KeyError(f"Frame {idx} is not in this track's frames {self.frame_indices}.")
        return self.frame_indices.index(idx)

    def xy(self, frame) -> tuple:
        """``(x, y)`` pixel position at ``frame`` (``FrameImage`` or absolute index)."""
        return tuple(float(v) for v in self._positions[self._local(frame)])

    def visible(self, frame) -> bool:
        """Whether the point was confidently tracked at ``frame``."""
        return bool(self._visible[self._local(frame)])

    def xy_with_time(self) -> list:
        """``[(t_seconds, (x, y)), ...]`` over visible frames only — input for ``tools.Motion.fit``."""
        return [
            (frame_index_to_seconds(fi, self._fps), tuple(float(v) for v in self._positions[i]))
            for i, fi in enumerate(self.frame_indices) if self._visible[i]
        ]

    def __repr__(self) -> str:
        return (
            f"PointTrack(frames={self.frame_indices[0]}..{self.frame_indices[-1]}, "
            f"visible={int(self._visible.sum())}/{len(self.frame_indices)})\n"
            f"  track.xy(frame) | track.visible(frame) | track.xy_with_time() -> tools.Motion.fit"
        )


class TAPTool(CPUTool):
    TOOL_PROMPT_SECTIONS = None
    TOOL_ABLATION_PREFIX = "tool_tap"

    TOOL_PROMPT_DESCRIPTION = """
### tools.TAP — Point Tracking (in-process)

`tools.TAP.track(point, frames)` → `PointTrack` — follows one pixel point
forward through `frames` (from `tools.Video.window(...)`), starting on
`frames[0]`. `point` is `(x, y)` in pixels on `frames[0]`.

| `PointTrack` | Returns |
|---|---|
| `xy(frame)` | `(x, y)` at a `FrameImage` / absolute frame index |
| `visible(frame)` | `bool` — confidently tracked there? (a non-visible position is meaningless) |
| `xy_with_time()` | `[(t, (x, y)), ...]` over visible frames — feed to `tools.Motion.fit` |
| `.frame_indices` | absolute frame indices covered |

```python
window = tools.Video.window(center_time=3.0, radius=0.5)
seed = tools.SAM3.segment("white ball", window[0]).centroid()   # (x, y) on the first window frame
pt = tools.TAP.track(seed, window)
fit = tools.Motion.fit(pt.xy_with_time(), degree=2)
```
"""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None, fps: float = 1.0):
        self._checkpoint_path = checkpoint_path or os.environ.get(_CKPT_ENV, _DEFAULT_CKPT)
        self._device_str = device
        self._model = None
        self._fps = fps

    def _ensure_model(self):
        if self._model is None:
            import time
            import torch
            from tapnet.tapnextpp.votsp2026.model import TAPNextPP
            # Allow pinning to a specific device via env var (e.g. a build
            # where torch.cuda.is_available() defaults to the wrong/busy
            # GPU). If the requested device turns out to be unusable in this
            # process's torch build (e.g. a CPU-only wheel with no CUDA
            # support), fall back to CPU with a clear message instead of
            # crashing the whole agent step.
            requested = self._device_str or os.environ.get(_DEVICE_ENV) or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(requested)
            t0 = time.monotonic()
            try:
                self._model = TAPNextPP.from_checkpoint(self._checkpoint_path, device=device, input_resolution=512)
            except (RuntimeError, AssertionError) as exc:
                if device.type == "cpu":
                    raise
                print(f"[TAP] device {device} unusable ({exc}); falling back to cpu.")
                device = torch.device("cpu")
                self._model = TAPNextPP.from_checkpoint(self._checkpoint_path, device=device, input_resolution=512)
            print(f"[TAP] loaded model on {device} in {time.monotonic() - t0:.1f}s")
        return self._model

    def _track_points(self, frames_bgr, seed_points_xy, start_frame: int = 0) -> dict:
        """Low-level: multiple seed points over raw BGR arrays. frame_indices are relative to frames_bgr."""
        model = self._ensure_model()
        query_xy = np.asarray(seed_points_xy, dtype=np.float32)
        if query_xy.ndim != 2 or query_xy.shape[1] != 2:
            raise ValueError(f"seed_points_xy must be (N, 2), got shape {query_xy.shape}")

        frames_to_track = frames_bgr[start_frame:]
        positions_all, visible_all = [], []
        state = None
        for j, frame_bgr in enumerate(frames_to_track):
            if j == 0:
                positions, visible, state = model.track_frame(frame_bgr, query_points_xy=query_xy)
            else:
                positions, visible, state = model.track_frame(frame_bgr, state=state)
            positions_all.append(positions.copy())
            visible_all.append(visible.copy())

        return {
            "positions": np.stack(positions_all, axis=0),
            "visible": np.stack(visible_all, axis=0),
            "frame_indices": list(range(start_frame, start_frame + len(frames_to_track))),
        }

    def track(self, point, frames) -> PointTrack:
        """Track one pixel point (defined on ``frames[0]``) forward through ``frames``."""
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise TypeError("`frames` must be a non-empty list of frames, e.g. tools.Video.window(...).")
        if not all(isinstance(f, FrameImage) for f in frames):
            raise TypeError(
                "`frames` must be FrameImage objects (from tools.Video.window(...) or InputImages), "
                "not raw arrays/PIL images."
            )
        if point is None:
            raise ValueError("`point` is None — the seed (e.g. a centroid) was not found; pick another frame or target.")
        pt = np.asarray(point, dtype=np.float32).ravel()
        if pt.shape != (2,):
            raise ValueError(f"`point` must be one (x, y) pair, got shape {pt.shape}.")

        frames_bgr = [np.array(f.image.convert("RGB"))[:, :, ::-1] for f in frames]
        result = self._track_points(frames_bgr, [pt.tolist()], start_frame=0)
        return PointTrack(
            positions=result["positions"][:, 0, :],
            visible=result["visible"][:, 0],
            frame_indices=[f.frame_index for f in frames],
            fps=self._fps,
        )
