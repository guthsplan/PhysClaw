"""SAM3 GPU tool: segmentation and tracking.

Agent-facing surface (four verbs, no video argument — the session's video
is bound at construction):

    tools.SAM3.segment(target, frame)                 -> Mask   (all instances, one frame)
    tools.SAM3.track(target, frames)                  -> Track  (one object per target, a window)
    tools.SAM3.segment_at(frame, point=|box=)         -> Mask   (one instance)
    tools.SAM3.track_from(frame, frames, point=|box=) -> Track

The ``_segment_image_by_*`` / ``_segment_video_by_*`` methods are the
transport layer to the GPU server and are private on purpose: they are not
injected into the kernel, so agent code cannot reach them.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from spatial_agent.kernel_types.per_frame_types import Mask, PerFrameMask, Track
from spatial_agent.tools.base import GPUTool, ensure_image_list


def _normalize_points_per_object(points_per_object):
    """Normalize various point formats to [[[x, y], ...], ...]."""
    if not isinstance(points_per_object, (list, tuple)) or len(points_per_object) == 0:
        return points_per_object

    if (
        len(points_per_object) == 2
        and isinstance(points_per_object[0], (int, float))
        and isinstance(points_per_object[1], (int, float))
    ):
        return [[[points_per_object[0], points_per_object[1]]]]

    if all(
        isinstance(pt, (list, tuple))
        and len(pt) == 2
        and isinstance(pt[0], (int, float))
        and isinstance(pt[1], (int, float))
        for pt in points_per_object
    ):
        return [points_per_object]

    for obj_idx, obj_pts in enumerate(points_per_object):
        if not isinstance(obj_pts, (list, tuple)):
            continue
        if all(
            isinstance(pt, (list, tuple))
            and len(pt) == 1
            and isinstance(pt[0], (int, float))
            for pt in obj_pts
        ) and len(obj_pts) >= 2 and len(obj_pts) % 2 == 0:
            flat = [pt[0] for pt in obj_pts]
            paired = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
            points_per_object = list(points_per_object)
            points_per_object[obj_idx] = paired

    return points_per_object


def _normalize_boxes(boxes):
    """Auto-wrap a flat [x1,y1,x2,y2] into [[x1,y1,x2,y2]]."""
    if not isinstance(boxes, (list, tuple)) or len(boxes) == 0:
        return boxes
    if len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
        return [list(boxes)]
    return boxes


SAM3_VIDEO_METHODS_PROMPT = """
**Limits**: `track`/`track_from` accept at most {sam3_max_video_frames} frames per call — for a longer span use `tools.Video.window(..., num_frames=N)` to subsample. Tracking takes ~15s; `segment` on one frame ~0.5s.
"""


def _frame_indices_of(frames, what: str) -> List[int]:
    """Absolute frame indices of a window, validating it is a list of FrameImage."""
    from spatial_agent.kernel_types.frame_image import FrameImage

    if isinstance(frames, (FrameImage, Image.Image)):
        raise TypeError(
            f"`{what}` must be a window (a list of frames from tools.Video.window(...)), "
            f"but got a single frame. For one frame use tools.SAM3.segment(target, frame) / segment_at(frame, ...)."
        )
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise TypeError(f"`{what}` must be a non-empty window from tools.Video.window(...), got {type(frames).__name__}.")
    if not all(isinstance(f, FrameImage) for f in frames):
        raise TypeError(
            f"`{what}` must contain FrameImage objects (tools.Video.window(...) or an InputImages slice), "
            f"not raw PIL images/arrays."
        )
    fis = [f.frame_index for f in frames]
    if len(fis) < 2:
        raise ValueError(
            f"`{what}` has only one frame — tracking needs at least 2. "
            f"For one frame use tools.SAM3.segment(target, frame)."
        )
    if any(b <= a for a, b in zip(fis, fis[1:])):
        raise ValueError(f"`{what}` must be in increasing frame order (got {fis[:6]}...). Use tools.Video.window(...).")
    return fis


def _check_single_frame(frame, what: str = "frame"):
    from spatial_agent.kernel_types.frame_image import FrameImage

    if isinstance(frame, (list, tuple)):
        raise TypeError(
            f"`{what}` must be ONE frame (e.g. tools.Video.frame_at_time(t) or InputImages[i]), "
            f"but got a list of {len(frame)}. To follow an object across frames use tools.SAM3.track(target, frames)."
        )
    if not isinstance(frame, (FrameImage, Image.Image)):
        raise TypeError(f"`{what}` must be a FrameImage or PIL Image, got {type(frame).__name__}.")


def _check_target(target) -> str:
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"`target` must be a short text description like 'white car', got {target!r}.")
    return target


def _check_point_or_box(point, box):
    if (point is None) == (box is None):
        raise ValueError("Pass exactly one of point=(x, y) or box=(x1, y1, x2, y2), in pixels.")
    if point is not None:
        p = np.asarray(point, dtype=float).ravel()
        if p.shape != (2,):
            raise ValueError(f"`point` must be (x, y) in pixels, got shape {p.shape}. Boxes go in box=.")
        return [float(p[0]), float(p[1])], None
    b = np.asarray(box, dtype=float).ravel()
    if b.shape != (4,):
        raise ValueError(f"`box` must be (x1, y1, x2, y2) in pixels, got shape {b.shape}. Points go in point=.")
    return None, [float(v) for v in b]


class SAM3Tool(GPUTool):
    """Client wrapper for the SAM3 GPU server."""

    TOOL_ABLATION_PREFIX = "tool_sam3"

    TOOL_PROMPT_SECTIONS = OrderedDict([
        ("api", """
### tools.SAM3 — Segmentation & Tracking (GPU)

The video is already bound to this tool: you pass a **frame** or a **window** from `tools.Video`, never `InputImages` as an argument.

| Method | Returns | Use when |
|---|---|---|
| `segment(target, frame)` | `Mask` — every instance of `target` on that frame | "where is / how big is the X in this frame" |
| `track(target, frames)` | `Track` — one object per target, followed across the window | anything over time (velocity, acceleration, path) |
| `segment_at(frame, point=(x, y))` or `segment_at(frame, box=(x1, y1, x2, y2))` | `Mask` with exactly one instance | a specific instance among look-alikes, or `segment` found nothing |
| `track_from(frame, frames, point=...)` or `track_from(frame, frames, box=...)` | `Track` | same, then followed through `frames` (which must contain `frame`) |
| `exists(target, frames)` | `list[bool]`, one per frame | find which frames show X (fast) |

`target`: a short description, `"white car"`. `track` also takes a list of targets to follow several distinct objects.
Points/boxes are **pixels** — get them from `vlm.locate(frame, target).xy` / `vlm.locate_box(frame, target).box`.
Nothing found never raises: `seg.count == 0` / `track.visible_frames() == []`.

**`Mask`** (one frame, `count` instances; `i` = instance index):
`seg.count` · `seg.mask(i)` (H,W) bool · `seg.centroid(i)` (x,y) or None · `seg.bbox(i)` or None · `seg.area(i)` px · `show(seg.overlay())` · `seg.frame_index`

**`Track`** (`frame` = a FrameImage or absolute index; `target` = index or the target string, default 0):
`track.mask(frame, target)` · `track.visible(frame, target)` · `track.visible_frames(target)` · `track.centroid(frame, target)` (x,y) or None · `track.bbox(frame, target)` · `show(track.overlay(frame))` · `track.frame_indices` · `track.targets`

**3D** (with `recon = tools.Reconstruct.run(window)`): `recon.centroid(frame, seg.mask())` or `recon.centroid(frame, track.mask(frame))` → `(3,)` metres or `None`.

```python
frame  = tools.Video.frame_at_time(3.0)
seg    = tools.SAM3.segment("white car", frame)          # seg.count instances
window = tools.Video.window(center_time=3.0, radius=0.5, num_frames=16)
track  = tools.SAM3.track("white car", window)
recon  = tools.Reconstruct.run(window)
for fr in window:
    if track.visible(fr):
        xyz = recon.centroid(fr, track.mask(fr))          # (3,) or None
```
"""),
        ("verify", """
**Verify a mask before measuring with it** — a mask can silently cover the wrong object:
1. `seg.count` (or `track.visible_frames()`): 0 / empty means nothing matched — try another frame, a more specific `target`, or `segment_at` with a point from `vlm.locate`.
2. {verify_visual}
"""),
        ("notes", """
**Frames line up automatically when you reuse one window**: `track = tools.SAM3.track(t, window)` and `recon = tools.Reconstruct.run(window)` share `window.frame_indices`, so `recon.centroid(fr, track.mask(fr))` works for every `fr in window`. Using two different windows gives `KeyError: Frame N not found` when you cross-reference — build the window once.
{sam3_video_methods}
`Mask`/`Track` are returned by SAM3 — do NOT import them.
"""),
    ])

    def __init__(self, video_source: Optional[str] = None, input_images=None,
                 gpu_tool_max_retries: int = 3, sam3_max_video_frames: int = 200,
                 total_video_frames: int = 0, num_videos: int = 1,
                 input_images_list: Optional[List] = None,
                 video_sources_list: Optional[List[str]] = None):
        super().__init__(deployment_name="spatial_SAM3",
                         gpu_tool_max_retries=gpu_tool_max_retries)
        self._video_source = video_source
        self._input_images = input_images
        self._max_video_frames = sam3_max_video_frames
        self._total_video_frames = total_video_frames
        self._num_videos = num_videos
        self._input_images_list = input_images_list
        self._video_sources_list = video_sources_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, target: str, frame) -> Mask:
        """Every instance of ``target`` on one ``frame``. ``Mask`` with ``count`` instances (0 if none)."""
        target = _check_target(target)
        _check_single_frame(frame)
        try:
            pfm = self._segment_image_by_text(frame, target)
        except RuntimeError as exc:
            if "no usable mask" not in str(exc):
                raise
            return self._empty_mask(frame, label=target)
        return Mask._wrap(pfm)

    def track(self, target, frames) -> Track:
        """Follow ``target`` (a string, or a list of strings for several objects) across a window."""
        prompts = [target] if isinstance(target, str) else list(target) if isinstance(target, (list, tuple)) else None
        if not prompts:
            raise ValueError("`target` must be a description like 'white car', or a list of them.")
        prompts = [_check_target(p) for p in prompts]
        fis = _frame_indices_of(frames, "frames")
        pfm = self._segment_video_by_text(
            prompts,
            frame_indices=fis,
            start_frame=fis[0],
            end_frame=fis[-1] + 1,
            video_index=self._infer_video_index(frames),
        )
        return Track._wrap(pfm)

    def segment_at(self, frame, *, point=None, box=None, label: str = "object") -> Mask:
        """One instance at a pixel ``point=(x, y)`` or inside ``box=(x1, y1, x2, y2)`` on ``frame``."""
        _check_single_frame(frame)
        pt, bx = _check_point_or_box(point, box)
        try:
            if pt is not None:
                pfm = self._segment_image_by_points(frame, [pt], [1], label)
            else:
                pfm = self._segment_image_by_box(frame, bx, label)
        except RuntimeError as exc:
            if "produced no mask" not in str(exc):
                raise
            return self._empty_mask(frame, label=label)
        return Mask._wrap(pfm)

    def track_from(self, frame, frames, *, point=None, box=None, label: str = "object") -> Track:
        """Segment one instance at ``point``/``box`` on ``frame``, then follow it through ``frames``."""
        _check_single_frame(frame)
        pt, bx = _check_point_or_box(point, box)
        fis = _frame_indices_of(frames, "frames")
        fi = getattr(frame, "frame_index", None)
        if fi not in fis:
            raise ValueError(
                f"`frame` (frame_index={fi}) must be one of `frames` ({fis[0]}..{fis[-1]}) — "
                f"the point/box is defined on it. Use frames[0], or a window that contains it."
            )
        common = dict(
            prompt_frame_idx=fis.index(fi),
            frame_indices=fis,
            start_frame=fis[0],
            end_frame=fis[-1] + 1,
            video_index=self._infer_video_index(frames),
        )
        if pt is not None:
            pfm = self._segment_video_by_points([[pt]], [[1]], [label], **common)
        else:
            pfm = self._segment_video_by_box([bx], [label], **common)
        return Track._wrap(pfm)

    def exists(self, target: str, frames) -> List[bool]:
        """For each frame in ``frames``, whether ``target`` is detected on it."""
        target = _check_target(target)
        images = [frames] if not isinstance(frames, (list, tuple)) else list(frames)
        return self._is_object_exist(images, target)["exists"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _empty_mask(self, frame, label: str) -> Mask:
        from spatial_agent.kernel_types.frame_image import FrameImage

        fi = getattr(frame, "frame_index", 0)
        plain = frame.image if isinstance(frame, FrameImage) else frame
        W, H = plain.size
        return Mask(
            masks=np.zeros((1, 0, H, W), dtype=bool), labels=[], object_ids=[],
            frame_indices=[fi], frames={fi: plain},
        )

    def _resolve_video(self, video_index: int):
        """(video_source, input_images) for a 1-indexed ``video_index``."""
        if self._num_videos <= 1 or self._video_sources_list is None:
            return self._video_source, self._input_images
        n = self._num_videos
        if not isinstance(video_index, int) or video_index < 1 or video_index > n:
            raise ValueError(
                f"video_index={video_index!r} is out of range for a sample with {n} videos "
                f"(1-indexed; video_index=1 -> InputImages_1)."
            )
        idx = video_index - 1
        vs = self._video_sources_list[idx] if idx < len(self._video_sources_list) else None
        ii = (
            self._input_images_list[idx]
            if self._input_images_list and idx < len(self._input_images_list)
            else None
        )
        if vs is None:
            raise RuntimeError(
                f"Tracking needs a backing video file, but video {video_index} only has pre-extracted "
                f"frames. Use tools.SAM3.segment(target, frame) on its frames instead."
            )
        return vs, ii

    def _infer_video_index(self, frames, default: int = 1) -> int:
        """Which InputImages_<N> a window came from (membership by identity)."""
        if self._num_videos <= 1 or self._input_images_list is None or len(frames) == 0:
            return default
        first = frames[0]
        for idx, ii in enumerate(self._input_images_list, start=1):
            if ii is None or len(ii) == 0:
                continue
            if ii is frames or any(f is first for f in ii):
                return idx
        return default

    def _is_object_exist(self, images: List[Image.Image], object_name: str) -> Dict[str, Any]:
        images = ensure_image_list(images)
        if len(images) == 0:
            raise ValueError("At least one frame is required.")
        from spatial_agent.kernel_types.frame_image import FrameImage

        exists_list, counts_list = [], []
        for img in images:
            plain_img = img.image if isinstance(img, FrameImage) else img
            try:
                raw = self._call_remote("detect", image_source=plain_img, prompt=object_name)
                result = raw.result if hasattr(raw, "result") else raw
                if result is not None and hasattr(result, "masks"):
                    masks = self._to_numpy(result.masks)
                    n = masks.shape[0] if masks.ndim >= 3 else 0
                    exists_list.append(n > 0)
                    counts_list.append(n)
                else:
                    exists_list.append(False)
                    counts_list.append(0)
            except Exception:
                exists_list.append(False)
                counts_list.append(0)
        return {"exists": exists_list, "counts": counts_list}

    def _segment_image_by_points(self, image, points, point_labels, label="object") -> PerFrameMask:
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(image, (Image.Image, FrameImage)):
            raise TypeError(f"`image` must be a single PIL Image or FrameImage, got {type(image).__name__}.")
        if not isinstance(points, (list, tuple)) or len(points) == 0:
            raise ValueError("`points` must be a non-empty list of [x, y] coordinates.")
        if len(points) == 2 and isinstance(points[0], (int, float)):
            points = [points]
        if len(points) != len(point_labels):
            raise ValueError(f"`points` ({len(points)}) and `point_labels` ({len(point_labels)}) must have the same length.")

        frame_index = getattr(image, "frame_index", 0)
        plain_image = image.image if isinstance(image, FrameImage) else image
        W, H = plain_image.size
        norm_pts = [[x / W, y / H] for x, y in points]

        raw = self._call_remote("detect", image_source=plain_image, points=norm_pts,
                                point_labels=point_labels, obj_id=0)
        result = raw.result if hasattr(raw, "result") else raw
        mask = None
        if result is not None and hasattr(result, "masks") and result.masks is not None:
            masks_np = self._to_numpy(result.masks)
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                mask = masks_np[0]
        if mask is None or int(mask.sum()) == 0:
            raise RuntimeError(
                f"SAM3 produced no mask for points={points} on frame {frame_index}. Try a different "
                f"frame, a point more centred on the object, or segment(target, frame) with text."
            )
        return PerFrameMask(
            masks=mask[np.newaxis, np.newaxis].astype(bool), labels=[label], object_ids=[0],
            frame_indices=[frame_index], frames={frame_index: plain_image},
        )

    def _segment_image_by_box(self, image, box, label="object") -> PerFrameMask:
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(image, (Image.Image, FrameImage)):
            raise TypeError(f"`image` must be a single PIL Image or FrameImage, got {type(image).__name__}.")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("`box` must be [x1, y1, x2, y2] in pixel coordinates.")

        frame_index = getattr(image, "frame_index", 0)
        plain_image = image.image if isinstance(image, FrameImage) else image
        W, H = plain_image.size
        x1, y1, x2, y2 = box
        norm_xywh = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]

        raw = self._call_remote("detect", image_source=plain_image,
                                bounding_boxes=[norm_xywh], bounding_box_labels=[1])
        result = raw.result if hasattr(raw, "result") else raw
        mask = None
        if result is not None and hasattr(result, "masks") and result.masks is not None:
            masks_np = self._to_numpy(result.masks)
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                mask = masks_np[0]
        if mask is None or int(mask.sum()) == 0:
            raise RuntimeError(
                f"SAM3 produced no mask for box={box} on frame {frame_index}. Try a different frame, "
                f"a slightly larger box, or segment(target, frame) with text."
            )
        return PerFrameMask(
            masks=mask[np.newaxis, np.newaxis].astype(bool), labels=[label], object_ids=[0],
            frame_indices=[frame_index], frames={frame_index: plain_image},
        )

    def _segment_image_by_text(self, image, prompt: str, label: str = None,
                               confidence_threshold: float = 0.3) -> PerFrameMask:
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(image, (Image.Image, FrameImage)):
            raise TypeError(f"`image` must be a single PIL Image or FrameImage, got {type(image).__name__}.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"`prompt` must be a non-empty string, got {prompt!r}.")
        if label is None:
            label = prompt

        frame_index = getattr(image, "frame_index", 0)
        plain_image = image.image if isinstance(image, FrameImage) else image

        raw = self._call_remote("detect", image_source=plain_image, prompt=prompt)
        result = raw.result if hasattr(raw, "result") else raw
        masks_np = None
        if result is not None and hasattr(result, "masks") and result.masks is not None:
            masks_np = self._to_numpy(result.masks)
            scores = self._to_numpy(result.scores) if getattr(result, "scores", None) is not None else None
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                if scores is not None and len(scores) == masks_np.shape[0]:
                    masks_np = masks_np[scores >= confidence_threshold]
            else:
                masks_np = None
        if masks_np is None or masks_np.shape[0] == 0 or int(masks_np.reshape(masks_np.shape[0], -1).sum(axis=1).max()) == 0:
            raise RuntimeError(
                f"SAM3 produced no usable mask for prompt={prompt!r} on frame {frame_index}."
            )
        n_obj = masks_np.shape[0]
        labels = [label] if n_obj == 1 else [f"{label}_{i}" for i in range(n_obj)]
        return PerFrameMask(
            masks=masks_np[np.newaxis].astype(bool), labels=labels, object_ids=list(range(n_obj)),
            frame_indices=[frame_index], frames={frame_index: plain_image},
        )

    def _segment_video_by_text(self, prompts, labels=None, prompt_frame_idx: int = 0,
                               frame_indices=None, start_frame=None, end_frame=None,
                               video_index: int = 1) -> PerFrameMask:
        video_source, ii = self._resolve_video(video_index)
        if video_source is None:
            raise RuntimeError(
                "Tracking needs a video (Metadata.is_video must be True). For images use "
                "tools.SAM3.segment(target, frame)."
            )
        if labels is None:
            labels = list(prompts)
        self._validate_frame_range(start_frame, end_frame)
        raw = self._call_remote(
            "segment_video", video_source=video_source, prompts=list(prompts),
            prompt_frame_idx=prompt_frame_idx, frame_indices=frame_indices,
            start_frame=start_frame, end_frame=end_frame,
        )
        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"SAM3 tracking failed: {raw.err['msg']}")
        result = raw.result if hasattr(raw, "result") else raw
        return self._build_per_frame_mask(result, labels, frame_indices, input_images=ii)

    def _segment_video_by_points(self, points_per_object, point_labels_per_object, labels,
                                 prompt_frame_idx: int = 0, frame_indices=None,
                                 start_frame=None, end_frame=None, video_index: int = 1) -> PerFrameMask:
        video_source, ii = self._resolve_video(video_index)
        if video_source is None:
            raise RuntimeError("Tracking needs a video (Metadata.is_video must be True).")
        points_per_object = _normalize_points_per_object(points_per_object)
        if len(points_per_object) != len(point_labels_per_object) or len(points_per_object) != len(labels):
            raise ValueError("`points_per_object`, `point_labels_per_object` and `labels` must have the same length.")
        if ii is None or len(ii) == 0:
            raise RuntimeError("Cannot determine image dimensions for coordinate normalization.")
        W, H = ii[0].width, ii[0].height
        norm_pts = [[[x / W, y / H] for x, y in pts] for pts in points_per_object]
        self._validate_frame_range(start_frame, end_frame)
        raw = self._call_remote(
            "segment_video", video_source=video_source, points_per_object=norm_pts,
            point_labels_per_object=point_labels_per_object, obj_ids=list(range(len(labels))),
            prompt_frame_idx=prompt_frame_idx, frame_indices=frame_indices,
            start_frame=start_frame, end_frame=end_frame,
        )
        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"SAM3 tracking failed: {raw.err['msg']}")
        result = raw.result if hasattr(raw, "result") else raw
        return self._build_per_frame_mask(result, labels, frame_indices, input_images=ii)

    def _segment_video_by_box(self, boxes, labels, prompt_frame_idx: int = 0, frame_indices=None,
                              start_frame=None, end_frame=None, video_index: int = 1) -> PerFrameMask:
        video_source, ii = self._resolve_video(video_index)
        if video_source is None:
            raise RuntimeError("Tracking needs a video (Metadata.is_video must be True).")
        boxes = _normalize_boxes(boxes)
        if len(boxes) != len(labels):
            raise ValueError("`boxes` and `labels` must have the same length.")
        if ii is None or len(ii) == 0:
            raise RuntimeError("Cannot determine image dimensions for coordinate normalization.")
        W, H = ii[0].width, ii[0].height
        norm_boxes = [[x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H] for x1, y1, x2, y2 in boxes]
        self._validate_frame_range(start_frame, end_frame)
        raw = self._call_remote(
            "segment_video", video_source=video_source, boxes_per_object=norm_boxes,
            box_labels_per_object=[1] * len(boxes), prompt_frame_idx=prompt_frame_idx,
            frame_indices=frame_indices, start_frame=start_frame, end_frame=end_frame,
        )
        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"SAM3 tracking failed: {raw.err['msg']}")
        result = raw.result if hasattr(raw, "result") else raw
        return self._build_per_frame_mask(result, labels, frame_indices, input_images=ii)

    def _validate_frame_range(self, start_frame, end_frame) -> None:
        if start_frame is not None and end_frame is not None:
            n_frames = end_frame - start_frame
            if n_frames <= 0:
                raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame}).")
        else:
            total = self._total_video_frames
            if total <= 0:
                return
            n_frames = (end_frame if end_frame is not None else total) - (start_frame or 0)
        if n_frames > self._max_video_frames:
            raise ValueError(
                f"This span covers {n_frames} video frames, above the tracking limit of "
                f"{self._max_video_frames}. Subsample the window: "
                f"tools.Video.window(..., num_frames={self._max_video_frames})."
            )

    def _build_per_frame_mask(self, result, labels, frame_indices, input_images=None) -> PerFrameMask:
        ii = input_images if input_images is not None else self._input_images
        masks_np = self._to_numpy(result.masks)  # (T, N_detected, H_vid, W_vid)
        result_fi = list(result.frame_indices) if hasattr(result, "frame_indices") else frame_indices

        if ii is not None and len(ii) > 0:
            target_h, target_w = ii[0].height, ii[0].width
            H_mask, W_mask = masks_np.shape[2], masks_np.shape[3]
            if (H_mask, W_mask) != (target_h, target_w):
                from scipy.ndimage import zoom
                sy, sx = target_h / H_mask, target_w / W_mask
                masks_np = zoom(masks_np.astype(np.float32), (1, 1, sy, sx), order=0) > 0.5

        n_returned, n_labels = masks_np.shape[1], len(labels)
        if n_returned < n_labels:
            T, _, H, W = masks_np.shape
            masks_np = np.concatenate([masks_np, np.zeros((T, n_labels - n_returned, H, W), dtype=masks_np.dtype)], axis=1)
        elif n_returned > n_labels:
            masks_np = masks_np[:, :n_labels]

        frames_dict = {}
        if ii is not None and result_fi is not None:
            fi_set = set(result_fi)
            input_fi = getattr(ii, "frame_indices", None)
            if input_fi is not None:
                for local_idx, abs_idx in enumerate(input_fi):
                    if abs_idx in fi_set and local_idx < len(ii):
                        frames_dict[abs_idx] = ii[local_idx]

        return PerFrameMask(
            masks=masks_np.astype(bool), labels=list(labels), object_ids=list(range(n_labels)),
            frame_indices=result_fi, frames=frames_dict,
        )

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        return np.asarray(arr)

    def __repr__(self) -> str:
        return "SAM3Tool(methods: segment, track, segment_at, track_from, exists)"
