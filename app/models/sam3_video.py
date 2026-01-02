from __future__ import annotations

from loguru import logger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.registry import register_model
from app.schemas.shape import Shape
from .segment_anything_3 import SegmentAnything3


BoxXYXY = Tuple[float, float, float, float]


@register_model("sam3_video")
class Sam3Video(SegmentAnything3):
    """SAM3 "video" wrapper for frame-by-frame usage.

    The server API accepts a single image per request. X-AnyLabeling's video
    auto-labeling typically calls the same model repeatedly across frames.

    This implementation mirrors the pattern used by other "video" models in
    this repo (e.g. `yolo11_track`): it keeps lightweight state across calls
    and supports a `reset_tracker` flag to clear that state.

    Behavior:
    - If `marks` (rect prompts) or `text_prompt` are provided, it runs SAM3 on
      the current frame and stores the resulting boxes as "last prompts".
    - If no prompt is provided but tracking state exists, it reuses the last
      stored boxes as prompts for the next frame.

    Note:
    - This is a prompt-reuse tracker (not full SAM3 temporal propagation).
    - It is designed to "work in the server" with the existing `/v1/predict`
      contract (single image + params).
    """

    def load(self):
        super().load()
        self._reset_tracking_state()

    def _reset_tracking_state(self):
        self._last_boxes_xyxy: List[BoxXYXY] = []
        self._last_frame_wh: Optional[Tuple[int, int]] = None
        self._tracks: Dict[int, BoxXYXY] = {}
        self._track_misses: Dict[int, int] = {}
        self._next_track_id: int = 1

    def reset_tracker(self):
        """Reset internal state across frames."""
        self._reset_tracking_state()

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run SAM3 on a single frame, optionally reusing last prompts."""

        if params.get("reset_tracker") or params.get("reset_session"):
            self.reset_tracker()

        text_prompt = params.get("text_prompt", "")
        marks = params.get("marks", [])

        # If the frame size changes, rescale any stored prompts.
        h, w = int(image.shape[0]), int(image.shape[1])
        if self._last_frame_wh is not None and self._last_boxes_xyxy:
            prev_w, prev_h = self._last_frame_wh
            if (prev_w, prev_h) != (w, h):
                sx = w / max(prev_w, 1)
                sy = h / max(prev_h, 1)
                self._last_boxes_xyxy = [
                    (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                    for (x1, y1, x2, y2) in self._last_boxes_xyxy
                ]

        # Prefer explicit prompts in request.
        if marks or (isinstance(text_prompt, str) and text_prompt.strip()):
            result = super().predict(image, params)
            result = self._apply_track_ids(result, params)
            self._update_tracking_from_shapes(result.get("shapes", []), (w, h))
            return result

        # Otherwise, if we have stored prompts, reuse them.
        if self._last_boxes_xyxy:
            reused_marks = [
                {
                    "type": "rectangle",
                    "label": 1,
                    "data": [float(x1), float(y1), float(x2), float(y2)],
                }
                for (x1, y1, x2, y2) in self._last_boxes_xyxy
            ]

            reused_params = dict(params)
            reused_params["marks"] = reused_marks
            result = self._predict_with_boxes(image, reused_marks, reused_params)
            result = self._apply_track_ids(result, reused_params)
            self._update_tracking_from_shapes(result.get("shapes", []), (w, h))
            return result

        logger.warning("No prompt provided and no tracking state")
        return {"shapes": [], "description": ""}

    def _update_tracking_from_shapes(
        self, shapes: List[Shape], frame_wh: Tuple[int, int]
    ):
        boxes = self._shapes_to_boxes_xyxy(shapes)
        if boxes:
            self._last_boxes_xyxy = boxes
            self._last_frame_wh = frame_wh

    @staticmethod
    def _shapes_to_boxes_xyxy(shapes: List[Shape]) -> List[BoxXYXY]:
        boxes: List[BoxXYXY] = []

        # Prefer rectangles (if present) because they are already boxes.
        rect_shapes = [s for s in shapes if s.shape_type == "rectangle"]
        source_shapes = rect_shapes if rect_shapes else shapes

        for shape in source_shapes:
            bbox = Sam3Video._bbox_from_shape(shape)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            # Filter out degenerate boxes
            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                continue
            boxes.append((x1, y1, x2, y2))

        # Keep deterministic ordering (largest first)
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes

    @staticmethod
    def _bbox_from_shape(shape: Shape) -> Optional[BoxXYXY]:
        if not shape.points:
            return None
        xs = [float(p[0]) for p in shape.points]
        ys = [float(p[1]) for p in shape.points]
        if not xs or not ys:
            return None
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    @staticmethod
    def _iou(a: BoxXYXY, b: BoxXYXY) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def _apply_track_ids(
        self, result: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign stable group_id values to shapes and update track state."""

        shapes: List[Shape] = list(result.get("shapes", []) or [])
        if not shapes:
            # age existing tracks
            self._age_tracks(params)
            return result

        detections = self._group_shapes_into_detections(shapes)
        if not detections:
            self._age_tracks(params)
            return result

        det_boxes = [d["bbox"] for d in detections]
        assignments = self._match_detections_to_tracks(det_boxes, params)

        updated_shapes = shapes[:]
        for det_idx, track_id in assignments.items():
            for shape_index in detections[det_idx]["shape_indices"]:
                updated_shapes[shape_index] = updated_shapes[
                    shape_index
                ].model_copy(update={"group_id": int(track_id)})

        return {**result, "shapes": updated_shapes}

    def _age_tracks(self, params: Dict[str, Any]) -> None:
        max_missed = int(params.get("max_track_missed", self.params.get("max_track_missed", 30)))
        for tid in list(self._tracks.keys()):
            self._track_misses[tid] = int(self._track_misses.get(tid, 0)) + 1
            if self._track_misses[tid] > max_missed:
                self._tracks.pop(tid, None)
                self._track_misses.pop(tid, None)

    def _group_shapes_into_detections(
        self, shapes: List[Shape]
    ) -> List[Dict[str, Any]]:
        """Group shapes into per-object detections.

        Uses rectangles as anchors when present, otherwise each shape becomes
        its own detection.
        """

        rect_indices: List[int] = []
        rect_boxes: List[BoxXYXY] = []
        for i, s in enumerate(shapes):
            if s.shape_type != "rectangle":
                continue
            bbox = self._bbox_from_shape(s)
            if bbox is None:
                continue
            rect_indices.append(i)
            rect_boxes.append(bbox)

        detections: List[Dict[str, Any]] = []
        if not rect_indices:
            for i, s in enumerate(shapes):
                bbox = self._bbox_from_shape(s)
                if bbox is None:
                    continue
                detections.append({"bbox": bbox, "shape_indices": [i]})
            return detections

        # Create one detection per rectangle.
        for idx, bbox in zip(rect_indices, rect_boxes):
            detections.append({"bbox": bbox, "shape_indices": [idx]})

        # Attach non-rect shapes to the best rectangle.
        for i, s in enumerate(shapes):
            if s.shape_type == "rectangle":
                continue
            bbox = self._bbox_from_shape(s)
            if bbox is None:
                continue
            best_j = -1
            best_iou = 0.0
            for j, det in enumerate(detections):
                iou = self._iou(bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            # If we can't reasonably match it, keep it as its own detection.
            if best_j >= 0 and best_iou >= 0.05:
                detections[best_j]["shape_indices"].append(i)
            else:
                detections.append({"bbox": bbox, "shape_indices": [i]})

        return detections

    def _match_detections_to_tracks(
        self, det_boxes: List[BoxXYXY], params: Dict[str, Any]
    ) -> Dict[int, int]:
        """Return mapping det_index -> track_id and update internal tracks."""

        iou_thresh = float(
            params.get(
                "track_iou_threshold", self.params.get("track_iou_threshold", 0.3)
            )
        )
        max_missed = int(params.get("max_track_missed", self.params.get("max_track_missed", 30)))

        track_ids = list(self._tracks.keys())
        used_tracks = set()
        used_dets = set()
        pairs: List[Tuple[float, int, int]] = []  # (iou, det_i, track_j)

        for di, db in enumerate(det_boxes):
            for tj, tid in enumerate(track_ids):
                iou = self._iou(db, self._tracks[tid])
                if iou > 0.0:
                    pairs.append((iou, di, tj))

        pairs.sort(reverse=True, key=lambda t: t[0])
        assignments: Dict[int, int] = {}

        for iou, di, tj in pairs:
            if iou < iou_thresh:
                break
            if di in used_dets:
                continue
            if tj in used_tracks:
                continue
            tid = track_ids[tj]
            assignments[di] = tid
            used_dets.add(di)
            used_tracks.add(tj)

        # New tracks for unassigned detections.
        for di, db in enumerate(det_boxes):
            if di in assignments:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = db
            self._track_misses[tid] = 0
            assignments[di] = tid

        # Update matched track boxes and misses.
        matched_track_ids = set(assignments.values())
        for di, tid in assignments.items():
            self._tracks[tid] = det_boxes[di]
            self._track_misses[tid] = 0

        # Age non-matched tracks.
        for tid in list(self._tracks.keys()):
            if tid in matched_track_ids:
                continue
            self._track_misses[tid] = int(self._track_misses.get(tid, 0)) + 1
            if self._track_misses[tid] > max_missed:
                self._tracks.pop(tid, None)
                self._track_misses.pop(tid, None)

        return assignments

    def unload(self):
        self._reset_tracking_state()
        super().unload()
from __future__ import annotations

from loguru import logger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.registry import register_model
from app.schemas.shape import Shape
from .segment_anything_3 import SegmentAnything3


BoxXYXY = Tuple[float, float, float, float]


@register_model("sam3_video")
class Sam3Video(SegmentAnything3):
    """SAM3 "video" wrapper for frame-by-frame usage.

    The server API accepts a single image per request. X-AnyLabeling's video
    auto-labeling typically calls the same model repeatedly across frames.

    This implementation mirrors the pattern used by other "video" models in
    this repo (e.g. `yolo11_track`): it keeps lightweight state across calls
    and supports a `reset_tracker` flag to clear that state.

    Behavior:
    - If `marks` (rect prompts) or `text_prompt` are provided, it runs SAM3 on
      the current frame and stores the resulting boxes as "last prompts".
    - If no prompt is provided but tracking state exists, it reuses the last
      stored boxes as prompts for the next frame.

    Note:
    - This is a prompt-reuse tracker (not full SAM3 temporal propagation).
    - It is designed to "work in the server" with the existing `/v1/predict`
      contract (single image + params).
    """

    def load(self):
        super().load()
        self._reset_tracking_state()

    def _reset_tracking_state(self):
        self._last_boxes_xyxy: List[BoxXYXY] = []
        self._last_frame_wh: Optional[Tuple[int, int]] = None
        self._tracks: Dict[int, BoxXYXY] = {}
        self._track_misses: Dict[int, int] = {}
        self._next_track_id: int = 1

    def reset_tracker(self):
        """Reset internal state across frames."""
        self._reset_tracking_state()

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run SAM3 on a single frame, optionally reusing last prompts."""

        if params.get("reset_tracker") or params.get("reset_session"):
            self.reset_tracker()

        text_prompt = params.get("text_prompt", "")
        marks = params.get("marks", [])

        # If the frame size changes, rescale any stored prompts.
        h, w = int(image.shape[0]), int(image.shape[1])
        if self._last_frame_wh is not None and self._last_boxes_xyxy:
            prev_w, prev_h = self._last_frame_wh
            if (prev_w, prev_h) != (w, h):
                sx = w / max(prev_w, 1)
                sy = h / max(prev_h, 1)
                self._last_boxes_xyxy = [
                    (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                    for (x1, y1, x2, y2) in self._last_boxes_xyxy
                ]

        # Prefer explicit prompts in request.
        if marks or (isinstance(text_prompt, str) and text_prompt.strip()):
            result = super().predict(image, params)
            result = self._apply_track_ids(result, params)
            self._update_tracking_from_shapes(result.get("shapes", []), (w, h))
            return result

        # Otherwise, if we have stored prompts, reuse them.
        if self._last_boxes_xyxy:
            reused_marks = [
                {
                    "type": "rectangle",
                    "label": 1,
                    "data": [float(x1), float(y1), float(x2), float(y2)],
                }
                for (x1, y1, x2, y2) in self._last_boxes_xyxy
            ]

            reused_params = dict(params)
            reused_params["marks"] = reused_marks
            result = self._predict_with_boxes(image, reused_marks, reused_params)
            result = self._apply_track_ids(result, reused_params)
            self._update_tracking_from_shapes(result.get("shapes", []), (w, h))
            return result

        logger.warning("No prompt provided and no tracking state")
        return {"shapes": [], "description": ""}

    def _update_tracking_from_shapes(
        self, shapes: List[Shape], frame_wh: Tuple[int, int]
    ):
        boxes = self._shapes_to_boxes_xyxy(shapes)
        if boxes:
            self._last_boxes_xyxy = boxes
            self._last_frame_wh = frame_wh

    @staticmethod
    def _shapes_to_boxes_xyxy(shapes: List[Shape]) -> List[BoxXYXY]:
        boxes: List[BoxXYXY] = []

        # Prefer rectangles (if present) because they are already boxes.
        rect_shapes = [s for s in shapes if s.shape_type == "rectangle"]
        source_shapes = rect_shapes if rect_shapes else shapes

        for shape in source_shapes:
            bbox = Sam3Video._bbox_from_shape(shape)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            # Filter out degenerate boxes
            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                continue
            boxes.append((x1, y1, x2, y2))

        # Keep deterministic ordering (largest first)
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes

    @staticmethod
    def _bbox_from_shape(shape: Shape) -> Optional[BoxXYXY]:
        if not shape.points:
            return None
        xs = [float(p[0]) for p in shape.points]
        ys = [float(p[1]) for p in shape.points]
        if not xs or not ys:
            return None
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    @staticmethod
    def _iou(a: BoxXYXY, b: BoxXYXY) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def _apply_track_ids(
        self, result: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign stable group_id values to shapes and update track state."""

        shapes: List[Shape] = list(result.get("shapes", []) or [])
        if not shapes:
            # age existing tracks
            self._age_tracks(params)
            return result

        detections = self._group_shapes_into_detections(shapes)
        if not detections:
            self._age_tracks(params)
            return result

        det_boxes = [d["bbox"] for d in detections]
        assignments = self._match_detections_to_tracks(det_boxes, params)

        updated_shapes = shapes[:]
        for det_idx, track_id in assignments.items():
            for shape_index in detections[det_idx]["shape_indices"]:
                updated_shapes[shape_index] = updated_shapes[
                    shape_index
                ].model_copy(update={"group_id": int(track_id)})

        return {**result, "shapes": updated_shapes}

    def _age_tracks(self, params: Dict[str, Any]) -> None:
        max_missed = int(params.get("max_track_missed", self.params.get("max_track_missed", 30)))
        for tid in list(self._tracks.keys()):
            self._track_misses[tid] = int(self._track_misses.get(tid, 0)) + 1
            if self._track_misses[tid] > max_missed:
                self._tracks.pop(tid, None)
                self._track_misses.pop(tid, None)

    def _group_shapes_into_detections(
        self, shapes: List[Shape]
    ) -> List[Dict[str, Any]]:
        """Group shapes into per-object detections.

        Uses rectangles as anchors when present, otherwise each shape becomes
        its own detection.
        """

        rect_indices: List[int] = []
        rect_boxes: List[BoxXYXY] = []
        for i, s in enumerate(shapes):
            if s.shape_type != "rectangle":
                continue
            bbox = self._bbox_from_shape(s)
            if bbox is None:
                continue
            rect_indices.append(i)
            rect_boxes.append(bbox)

        detections: List[Dict[str, Any]] = []
        if not rect_indices:
            for i, s in enumerate(shapes):
                bbox = self._bbox_from_shape(s)
                if bbox is None:
                    continue
                detections.append({"bbox": bbox, "shape_indices": [i]})
            return detections

        # Create one detection per rectangle.
        for idx, bbox in zip(rect_indices, rect_boxes):
            detections.append({"bbox": bbox, "shape_indices": [idx]})

        # Attach non-rect shapes to the best rectangle.
        for i, s in enumerate(shapes):
            if s.shape_type == "rectangle":
                continue
            bbox = self._bbox_from_shape(s)
            if bbox is None:
                continue
            best_j = -1
            best_iou = 0.0
            for j, det in enumerate(detections):
                iou = self._iou(bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            # If we can't reasonably match it, keep it as its own detection.
            if best_j >= 0 and best_iou >= 0.05:
                detections[best_j]["shape_indices"].append(i)
            else:
                detections.append({"bbox": bbox, "shape_indices": [i]})

        return detections

    def _match_detections_to_tracks(
        self, det_boxes: List[BoxXYXY], params: Dict[str, Any]
    ) -> Dict[int, int]:
        """Return mapping det_index -> track_id and update internal tracks."""

        iou_thresh = float(
            params.get(
                "track_iou_threshold", self.params.get("track_iou_threshold", 0.3)
            )
        )
        max_missed = int(params.get("max_track_missed", self.params.get("max_track_missed", 30)))

        track_ids = list(self._tracks.keys())
        used_tracks = set()
        used_dets = set()
        pairs: List[Tuple[float, int, int]] = []  # (iou, det_i, track_j)

        for di, db in enumerate(det_boxes):
            for tj, tid in enumerate(track_ids):
                iou = self._iou(db, self._tracks[tid])
                if iou > 0.0:
                    pairs.append((iou, di, tj))

        pairs.sort(reverse=True, key=lambda t: t[0])
        assignments: Dict[int, int] = {}

        for iou, di, tj in pairs:
            if iou < iou_thresh:
                break
            if di in used_dets:
                continue
            if tj in used_tracks:
                continue
            tid = track_ids[tj]
            assignments[di] = tid
            used_dets.add(di)
            used_tracks.add(tj)

        # New tracks for unassigned detections.
        for di, db in enumerate(det_boxes):
            if di in assignments:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = db
            self._track_misses[tid] = 0
            assignments[di] = tid

        # Update matched track boxes and misses.
        matched_track_ids = set(assignments.values())
        for di, tid in assignments.items():
            self._tracks[tid] = det_boxes[di]
            self._track_misses[tid] = 0

        # Age non-matched tracks.
        for tid in list(self._tracks.keys()):
            if tid in matched_track_ids:
                continue
            self._track_misses[tid] = int(self._track_misses.get(tid, 0)) + 1
            if self._track_misses[tid] > max_missed:
                self._tracks.pop(tid, None)
                self._track_misses.pop(tid, None)

        return assignments

    def unload(self):
        self._reset_tracking_state()
        super().unload()
