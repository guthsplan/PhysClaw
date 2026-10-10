"""VLMModule: provides ``vlm.locate()`` and ``vlm.ask_with_thinking()``.

Injected into the kernel as ``vlm``. Each method opens an isolated VLM
session: the VLM sees only the images and the question — no conversation
history, no prior reasoning from the main agent.

- ``vlm.locate(frame, target)`` / ``vlm.locate_box(frame, target)`` —
  coordinate grounding on one frame, returns ``Located`` in pixels.
- ``vlm.ask_with_thinking(visual_input, question)`` — visual reasoning
  with extended thinking budget, up to 64 frames per call.

Usage in the Jupyter kernel::

    loc = vlm.locate(InputImages[5], "the car")
    if loc.visible: x, y = loc.xy
    answer = vlm.ask_with_thinking([InputImages[0], InputImages[15], InputImages[30]], "Across these frames, does the person walk toward or away from the camera?")
"""

import asyncio
import concurrent.futures
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from spatial_agent.kernel_types.visual_feedback import VisualFeedback


_LOCATE_MAX_IMAGES = 8
_THINKING_MAX_IMAGES = 64
_NUM = r"-?\d+(?:\.\d+)?"
_PAIR_RE = re.compile(rf"\(?\s*({_NUM})\s*[, ]\s*({_NUM})\s*\)?")
_QUAD_RE = re.compile(rf"\(?\s*({_NUM})\s*[, ]\s*({_NUM})\s*[, ]\s*({_NUM})\s*[, ]\s*({_NUM})\s*\)?")
_ITEM_LINE_RE = re.compile(r"^\s*\**(\d+)\**\s*[:.)\]-]\s*(.*)$")


@dataclass
class Located:
    """Result of ``vlm.locate`` / ``vlm.locate_box``. Coordinates are in **pixels**."""

    visible: bool
    xy: Optional[Tuple[float, float]] = None        # center point (pixels)
    box: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2 (pixels); locate_box only
    raw: str = ""


def _parse_pair(raw: str) -> Optional[Tuple[float, float]]:
    """First adjacent number pair in the response ("420 390", "(420, 390)"), or None."""
    if "not visible" in raw.lower():
        return None
    m = _PAIR_RE.search(raw)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _parse_quad(raw: str) -> Optional[Tuple[float, float, float, float]]:
    if "not visible" in raw.lower():
        return None
    m = _QUAD_RE.search(raw)
    return tuple(float(m.group(i)) for i in range(1, 5)) if m else None


_LOCATE_VERIFY_PREAMBLE = (
    "Before answering, first check whether what the question describes is "
    "clearly visible in the provided image(s). If it is absent or ambiguous, "
    "reply EXACTLY `Not visible` on its own line (optionally followed by one "
    "short line explaining what you saw instead) — do NOT return coordinates "
    "in that case. Only if the requested target is clearly and unambiguously "
    "present, answer the request below.\n\n"
    "Request: "
)


class VLMModule:
    """Injected into the kernel as ``vlm``.

    The main agent calls ``vlm.locate(...)`` for coordinate grounding and
    ``vlm.ask_with_thinking(...)`` for visual reasoning. Both query
    isolated VLM sessions; the VLM has no memory between calls.
    """

    def __init__(
        self,
        llm_client,  # spatial_agent.llm.client.LLMClient
        locate_system_prompt: str,
        thinking_system_prompt: str,
        session_dir: str,
        session_id: str = "",
        locate_role_params=None,    # LLMRoleParams
        thinking_role_params=None,  # LLMRoleParams
    ):
        self._llm_client = llm_client
        self._locate_prompt = locate_system_prompt
        self._thinking_prompt = thinking_system_prompt
        self._session_dir = session_dir
        self._session_id = session_id
        self._locate_role_params = locate_role_params
        self._thinking_role_params = thinking_role_params
        self._queries: List[Dict[str, Any]] = []
        self._query_counter = 0
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Public API (called from kernel code)
    # ------------------------------------------------------------------

    def _grounding_query(self, frame, question: str) -> Tuple[str, int, int]:
        from spatial_agent.kernel_types.frame_image import FrameImage

        plain = frame.image if isinstance(frame, FrameImage) else frame
        if not isinstance(plain, Image.Image):
            raise TypeError(
                f"`frame` must be one FrameImage/PIL Image (e.g. tools.Video.frame_at_time(t)), "
                f"got {type(frame).__name__}."
            )
        if not isinstance(question, str):
            raise TypeError("`target` must be a short text description, e.g. 'the red car'.")
        W, H = plain.size
        raw = self._dispatch(
            visual_input=plain,
            question=_LOCATE_VERIFY_PREAMBLE + question,
            query_type="locate",
            system_prompt=self._locate_prompt,
            role_params=self._locate_role_params,
            max_images=_LOCATE_MAX_IMAGES,
        )
        return raw, W, H

    def locate(self, frame: Union[Image.Image, "FrameImage"], target: str) -> Located:
        """Pixel center of ``target`` in ``frame``.

        Returns ``Located(visible, xy, raw)``; ``xy`` is in **pixels**. Check
        ``.visible`` before using ``.xy`` — it is False when the target is
        absent/ambiguous or the answer could not be parsed.
        """
        raw, W, H = self._grounding_query(
            frame,
            f"Give the (x, y) center coordinates in 0-1000 normalized scale for: {target}. "
            f"Reply with ONLY the two numbers, space-separated.",
        )
        pair = _parse_pair(raw)
        if pair is None:
            return Located(visible=False, raw=raw)
        nx, ny = pair
        return Located(visible=True, xy=(nx / 1000.0 * W, ny / 1000.0 * H), raw=raw)

    def locate_points(self, frame: Union[Image.Image, "FrameImage"], descriptions: List[str]) -> List[Located]:
        """Pixel positions of several *named points* on one frame, in one VLM call.

        Use this for endpoints and edges — "topmost point of the helmet",
        "bottom of the left boot", "left edge of the ball" — where ``locate``
        (an object's center) is the wrong question. Returns one ``Located``
        per description, in order; check each ``.visible`` before ``.xy``.
        """
        if (not isinstance(descriptions, (list, tuple)) or len(descriptions) == 0
                or not all(isinstance(d, str) and d.strip() for d in descriptions)):
            raise ValueError(
                "`descriptions` must be a non-empty list of short point descriptions, "
                "e.g. ['topmost point of the helmet', 'lowest point of the boots']."
            )
        numbered = "\n".join(f"{i + 1}. {d.strip()}" for i, d in enumerate(descriptions))
        question = (
            "For each numbered item below, give the (x, y) position in 0-1000 normalized scale "
            "of exactly that point on the image. Reply with one line per item in the form "
            "'<number>: x y' — or '<number>: Not visible' if that specific point cannot be seen. "
            f"No other text.\n{numbered}"
        )
        raw, W, H = self._grounding_query(frame, question)
        results = [Located(visible=False, raw=raw) for _ in descriptions]
        for line in raw.splitlines():
            m = _ITEM_LINE_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if not (0 <= idx < len(descriptions)):
                continue
            pair = _parse_pair(m.group(2))
            if pair is not None:
                results[idx] = Located(visible=True, xy=(pair[0] / 1000.0 * W, pair[1] / 1000.0 * H), raw=line.strip())
        return results

    def locate_box(self, frame: Union[Image.Image, "FrameImage"], target: str) -> Located:
        """Pixel bounding box of ``target`` in ``frame``.

        Returns ``Located(visible, xy, box, raw)`` with ``box=(x1, y1, x2, y2)``
        and ``xy`` = the box center, both in **pixels**.
        """
        raw, W, H = self._grounding_query(
            frame,
            f"Give the bounding box (x1, y1, x2, y2) in 0-1000 normalized scale for: {target}. "
            f"Reply with ONLY the four numbers, space-separated.",
        )
        quad = _parse_quad(raw)
        if quad is None:
            return Located(visible=False, raw=raw)
        x1, y1, x2, y2 = quad
        box = (x1 / 1000.0 * W, y1 / 1000.0 * H, x2 / 1000.0 * W, y2 / 1000.0 * H)
        return Located(visible=True, xy=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box=box, raw=raw)

    def ask_with_thinking(
        self,
        visual_input: Union[VisualFeedback, Image.Image, List],
        question: str,
    ) -> str:
        """Query the thinking VLM. Returns a text answer about the provided frames.

        Args:
            visual_input: VisualFeedback, PIL.Image, or list of these (max 64).
            question: Self-contained question. The session is independent —
                do not embed prior conclusions; phrase the question on its
                own terms so the answer is formed from the images.

        Returns:
            Text answer from the isolated thinking VLM session.
        """
        return self._dispatch(
            visual_input=visual_input,
            question=question,
            query_type="thinking",
            system_prompt=self._thinking_prompt,
            role_params=self._thinking_role_params,
            max_images=_THINKING_MAX_IMAGES,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        visual_input: Union[VisualFeedback, Image.Image, List],
        question: str,
        query_type: str,
        system_prompt: str,
        role_params,
        max_images: int,
    ) -> str:
        self._query_counter += 1
        query_id = f"vlm_q_{query_type}_{self._query_counter:04d}"

        images = self._resolve_images(visual_input)
        truncated = False
        if len(images) > max_images:
            images = images[:max_images]
            truncated = True

        self._save_query_images(query_id, query_type, images)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                answer = self._executor.submit(
                    asyncio.run,
                    self._llm_client.generate_vision_query(
                        images=images,
                        question=question,
                        system_prompt=system_prompt,
                        role_params=role_params,
                        session_id=self._session_id,
                        usage_session_id=self._session_id,
                    ),
                ).result()
            else:
                answer = loop.run_until_complete(
                    self._llm_client.generate_vision_query(
                        images=images,
                        question=question,
                        system_prompt=system_prompt,
                        role_params=role_params,
                        session_id=self._session_id,
                        usage_session_id=self._session_id,
                    )
                )
        except Exception as exc:
            answer = f"[VLM Error] {exc}"

        source_desc = (
            str(visual_input)
            if not isinstance(visual_input, list)
            else f"list of {len(visual_input)} images"
        )
        self._queries.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "question": question,
                "answer": answer,
                "num_images": len(images),
                "source": source_desc,
            }
        )

        if truncated:
            print(
                f"[vlm.{query_type}] Showing first {max_images} images; "
                f"remaining were truncated."
            )
        print(f"[VLM Q | {query_type}] {question}")
        print(f"[VLM A | {query_type}] {answer}")

        return answer

    def _resolve_images(
        self, visual_input: Union[VisualFeedback, Image.Image, List]
    ) -> List[Image.Image]:
        import numpy as np
        if isinstance(visual_input, list) or (
            hasattr(visual_input, "__iter__")
            and not isinstance(visual_input, (str, np.ndarray, Image.Image, VisualFeedback))
        ):
            return [self._to_pil(v) for v in visual_input]
        return [self._to_pil(visual_input)]

    @staticmethod
    def _to_pil(obj: Any) -> Image.Image:
        from spatial_agent.kernel_types.frame_image import FrameImage
        import numpy as np

        if isinstance(obj, FrameImage):
            return obj.image
        if isinstance(obj, VisualFeedback):
            return obj.image
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            return Image.fromarray(obj)
        raise TypeError(
            f"Cannot convert {type(obj).__name__} to PIL.Image. "
            f"Accepted: FrameImage, VisualFeedback, PIL.Image, uint8 ndarray."
        )

    def _save_query_images(
        self, query_id: str, query_type: str, images: List[Image.Image]
    ) -> None:
        img_dir = os.path.join(self._session_dir, "vlm_queries", query_type)
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(images):
            path = os.path.join(img_dir, f"{query_id}_img_{i}.png")
            img.save(path)

    # ------------------------------------------------------------------
    # Called by feedback_node to retrieve this step's queries
    # ------------------------------------------------------------------

    def get_and_clear_queries(self) -> List[Dict[str, Any]]:
        """Return all queries from the current step and reset the list."""
        queries = self._queries.copy()
        self._queries.clear()
        return queries
