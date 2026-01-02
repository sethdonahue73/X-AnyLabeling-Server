from __future__ import annotations

from loguru import logger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.registry import register_model
from app.schemas.shape import Shape
from .segment_anything_3 import SegmentAnything3


BoxXYXY = Tuple[float, float, float, float]


try:
    import lapx  # type: ignore

    _HAS_LAPX = True
except Exception:
    lapx = None
    _HAS_LAPX = False


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


def _iou_matrix(dets_xyxy: np.ndarray, trks_xyxy: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix for dets vs trks.

    Args:
        dets_xyxy: (M, 4) array
        trks_xyxy: (N, 4) array

    Returns:
        (M, N) IoU matrix.
    """

    if dets_xyxy.size == 0 or trks_xyxy.size == 0:
        return np.zeros((dets_xyxy.shape[0], trks_xyxy.shape[0]), dtype=float)

    dets = dets_xyxy.astype(float, copy=False)
    trks = trks_xyxy.astype(float, copy=False)

    det_x1 = dets[:, 0:1]
    det_y1 = dets[:, 1:2]
    det_x2 = dets[:, 2:3]
    det_y2 = dets[:, 3:4]

    trk_x1 = trks[None, :, 0]
    trk_y1 = trks[None, :, 1]
    trk_x2 = trks[None, :, 2]
    trk_y2 = trks[None, :, 3]

    ix1 = np.maximum(det_x1, trk_x1)
    iy1 = np.maximum(det_y1, trk_y1)
    ix2 = np.minimum(det_x2, trk_x2)
    iy2 = np.minimum(det_y2, trk_y2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    det_area = np.maximum(0.0, det_x2 - det_x1) * np.maximum(0.0, det_y2 - det_y1)
    trk_area = np.maximum(0.0, trk_x2 - trk_x1) * np.maximum(0.0, trk_y2 - trk_y1)
    union = det_area + trk_area - inter

    # Avoid divide-by-zero
    return np.where(union > 0.0, inter / union, 0.0)


def _linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a rectangular linear assignment (Hungarian / LAP).

    Prefers `lapx` if available (already in requirements); otherwise uses a small
    pure-numpy Hungarian implementation.
    """

    if _HAS_LAPX and lapx is not None:
        # lapx exposes lapjv(cost_matrix) returning (cost, x, y)
        # where x[i] is the assigned column for row i, or -1.
        _cost, x, _y = lapx.lapjv(cost_matrix, extend_cost=True)
        row_ind = np.where(x >= 0)[0]
        col_ind = x[row_ind]
        return row_ind.astype(int), col_ind.astype(int)

    return _hungarian_fallback(cost_matrix)


def _hungarian_fallback(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback assignment solver.

    This is a compact implementation sufficient for typical per-frame object
    counts; it minimizes the provided cost matrix.
    """

    cost = np.array(cost_matrix, dtype=float)
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)

    # Pad to square.
    pad = np.zeros((n, n), dtype=float)
    pad[:n_rows, :n_cols] = cost
    if n_rows < n:
        pad[n_rows:, :] = pad.max() + 1.0
    if n_cols < n:
        pad[:, n_cols:] = pad.max() + 1.0

    u = np.zeros(n)
    v = np.zeros(n)
    p = np.zeros(n, dtype=int)
    way = np.zeros(n, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n, np.inf)
        used = np.zeros(n, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j - 1]:
                    continue
                cur = pad[i0 - 1, j - 1] - u[i0 - 1] - v[j - 1]
                if cur < minv[j - 1]:
                    minv[j - 1] = cur
                    way[j - 1] = j0
                if minv[j - 1] < delta:
                    delta = minv[j - 1]
                    j1 = j
            for j in range(0, n + 1):
                if j == 0:
                    continue
                if used[j - 1]:
                    u[p[j] - 1] += delta
                    v[j - 1] -= delta
                else:
                    minv[j - 1] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0 - 1]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = i assigned to column j
    assignment = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1

    row_ind = np.arange(n_rows, dtype=int)
    col_ind = assignment[:n_rows]
    mask = col_ind < n_cols
    return row_ind[mask], col_ind[mask]


class _KalmanBoxTracker:
    """A minimal SORT-style Kalman tracker for a single object."""

    _count = 0

    def __init__(self, bbox: BoxXYXY):
        # state: [cx, cy, s, r, vx, vy, vs]
        self.x = np.zeros((7, 1), dtype=float)
        self.P = np.eye(7, dtype=float) * 10.0
        self.F = np.eye(7, dtype=float)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        self.R = np.eye(4, dtype=float) * 1.0
        self.Q = np.eye(7, dtype=float) * 0.01

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        _KalmanBoxTracker._count += 1
        self.id = _KalmanBoxTracker._count

        self.update(bbox)

    @staticmethod
    def _convert_bbox_to_z(bbox: BoxXYXY) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h
        r = (w / h) if h > 0 else 0.0
        return np.array([[cx], [cy], [s], [r]], dtype=float)

    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray) -> BoxXYXY:
        cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        if s <= 0.0 or r <= 0.0:
            return (0.0, 0.0, 0.0, 0.0)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (float(x1), float(y1), float(x2), float(y2))

    def predict(self) -> BoxXYXY:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: BoxXYXY) -> None:
        z = self._convert_bbox_to_z(bbox)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        # Solve is faster / more stable than inv for small matrices.
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + (K @ y)
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> BoxXYXY:
        return self._convert_x_to_bbox(self.x)


class _SortTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 1):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.trackers: List[_KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self) -> None:
        self.trackers.clear()
        self.frame_count = 0

    def update(self, detections: List[BoxXYXY]) -> Dict[int, int]:
        """Update tracker with detections.

        Returns mapping detection_index -> track_id.
        """

        self.frame_count += 1

        # Predict new locations of tracks.
        if self.trackers:
            predicted = np.asarray([t.predict() for t in self.trackers], dtype=float)
        else:
            predicted = np.zeros((0, 4), dtype=float)

        if len(self.trackers) == 0:
            det_to_tid = {}
            for di, det in enumerate(detections):
                trk = _KalmanBoxTracker(det)
                self.trackers.append(trk)
                det_to_tid[di] = trk.id
            return det_to_tid

        if len(detections) == 0:
            # Age out old tracks.
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            return {}

        det_arr = np.asarray(detections, dtype=float)
        iou_mat = _iou_matrix(det_arr, predicted)

        cost = 1.0 - iou_mat
        row_ind, col_ind = _linear_assignment(cost)

        matched_dets = set()
        matched_trks = set()
        det_to_tid: Dict[int, int] = {}

        for r, c in zip(row_ind, col_ind):
            if r < 0 or c < 0:
                continue
            if iou_mat[r, c] < self.iou_threshold:
                continue
            self.trackers[c].update(detections[int(r)])
            det_to_tid[int(r)] = int(self.trackers[c].id)
            matched_dets.add(int(r))
            matched_trks.add(int(c))

        # Unmatched detections => new tracks.
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            trk = _KalmanBoxTracker(det)
            self.trackers.append(trk)
            det_to_tid[di] = trk.id

        # Age out unmatched tracks.
        survivors: List[_KalmanBoxTracker] = []
        for tj, trk in enumerate(self.trackers):
            if tj in matched_trks:
                survivors.append(trk)
                continue
            if trk.time_since_update <= self.max_age:
                survivors.append(trk)
        self.trackers = survivors

        return det_to_tid


@register_model("sam3_video_sort")
class Sam3VideoSort(SegmentAnything3):
    """SAM3 frame-by-frame with SORT-style tracking IDs.

    Produces normal SAM3 shapes, but assigns stable `Shape.group_id` values
    across frames using a SORT-like Kalman+IoU association.

    This is designed to work with the existing server contract (single image per request).
    """

    def load(self):
        super().load()
        self._reset_state()

    def _reset_state(self) -> None:
        self._last_boxes_xyxy: List[BoxXYXY] = []
        self._last_frame_wh: Optional[Tuple[int, int]] = None
        self._sort = _SortTracker(
            iou_threshold=float(self.params.get("track_iou_threshold", 0.3)),
            max_age=int(self.params.get("max_track_missed", 30)),
            min_hits=int(self.params.get("min_hits", 1)),
        )

    def reset_tracker(self) -> None:
        self._reset_state()

    def predict(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("reset_tracker") or params.get("reset_session"):
            self.reset_tracker()

        text_prompt = params.get("text_prompt", "")
        marks = params.get("marks", [])

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

        if marks or (isinstance(text_prompt, str) and text_prompt.strip()):
            result = super().predict(image, params)
            result = self._apply_sort_ids(result, params)
            self._update_last_boxes(result.get("shapes", []), (w, h))
            return result

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
            result = self._apply_sort_ids(result, reused_params)
            self._update_last_boxes(result.get("shapes", []), (w, h))
            return result

        logger.warning("No prompt provided and no tracking state")
        return {"shapes": [], "description": ""}

    def _apply_sort_ids(self, result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        shapes: List[Shape] = list(result.get("shapes", []) or [])
        if not shapes:
            return result

        bboxes = [_bbox_from_shape(s) for s in shapes]
        detections = self._group_shapes_into_detections(shapes, bboxes)
        if not detections:
            return result

        det_boxes = [d["bbox"] for d in detections]

        # Allow per-request override.
        self._sort.iou_threshold = float(params.get("track_iou_threshold", self._sort.iou_threshold))
        self._sort.max_age = int(params.get("max_track_missed", self._sort.max_age))

        det_to_tid = self._sort.update(det_boxes)

        updated = shapes[:]
        for det_idx, tid in det_to_tid.items():
            for shape_index in detections[det_idx]["shape_indices"]:
                updated[shape_index] = updated[shape_index].model_copy(update={"group_id": int(tid)})

        # Sort output to be deterministic by track id.
        def _shape_sort_key(s: Shape):
            gid = s.group_id or 10**9
            pri = 0 if s.shape_type == "rectangle" else 1
            return (gid, pri)

        updated.sort(key=_shape_sort_key)

        return {**result, "shapes": updated}

    def _group_shapes_into_detections(
        self, shapes: List[Shape], bboxes: List[Optional[BoxXYXY]]
    ) -> List[Dict[str, Any]]:
        rect_indices: List[int] = []
        rect_boxes: List[BoxXYXY] = []
        for i, s in enumerate(shapes):
            if s.shape_type != "rectangle":
                continue
            bbox = bboxes[i]
            if bbox is None:
                continue
            rect_indices.append(i)
            rect_boxes.append(bbox)

        detections: List[Dict[str, Any]] = []
        if not rect_indices:
            for i, _s in enumerate(shapes):
                bbox = bboxes[i]
                if bbox is None:
                    continue
                detections.append({"bbox": bbox, "shape_indices": [i]})
            return detections

        for idx, bbox in zip(rect_indices, rect_boxes):
            detections.append({"bbox": bbox, "shape_indices": [idx]})

        for i, s in enumerate(shapes):
            if s.shape_type == "rectangle":
                continue
            bbox = bboxes[i]
            if bbox is None:
                continue
            best_j = -1
            best_iou = 0.0
            for j, det in enumerate(detections):
                iou = _iou(bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.05:
                detections[best_j]["shape_indices"].append(i)
            else:
                detections.append({"bbox": bbox, "shape_indices": [i]})

        return detections

    def _update_last_boxes(self, shapes: List[Shape], frame_wh: Tuple[int, int]) -> None:
        # Store rectangles (or fallback bboxes) sorted by track id so prompt reuse is stable.
        rects = [s for s in shapes if s.shape_type == "rectangle"]
        src = rects if rects else shapes
        entries: List[Tuple[int, BoxXYXY]] = []
        for s in src:
            bbox = _bbox_from_shape(s)
            if bbox is None:
                continue
            gid = int(s.group_id) if s.group_id else 0
            entries.append((gid, bbox))
        if not entries:
            return
        entries.sort(key=lambda t: (t[0] if t[0] > 0 else 10**9))
        self._last_boxes_xyxy = [b for _, b in entries]
        self._last_frame_wh = frame_wh

    def unload(self):
        self._reset_state()
        super().unload()
