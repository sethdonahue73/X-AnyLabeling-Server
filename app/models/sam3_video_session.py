from __future__ import annotations

from dataclasses import dataclass, field
import json
from loguru import logger
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.registry import register_model
from app.schemas.shape import Shape
from .segment_anything_3 import SegmentAnything3


BoxXYXY = Tuple[float, float, float, float]


@dataclass
class _PromptState:
    boxes_xyxy: List[BoxXYXY] = field(default_factory=list)
    frame_wh: Optional[Tuple[int, int]] = None


@dataclass
class _SessionState:
    last_boxes_xyxy: List[BoxXYXY] = field(default_factory=list)
    last_frame_wh: Optional[Tuple[int, int]] = None
    tracks: Dict[int, BoxXYXY] = field(default_factory=dict)
    track_misses: Dict[int, int] = field(default_factory=dict)
    next_track_id: int = 1

    # Optional per-session recording state
    record_frame_idx: int = 0
    record_video_name: Optional[str] = None
    record_output_dir: Optional[str] = None
    record_writer: Any = None
    record_writer_wh: Optional[Tuple[int, int]] = None
    record_tmp_jsonl_path: Optional[str] = None


@register_model("sam3_video_session")
class Sam3VideoSession(SegmentAnything3):
    """SAM3 "video" wrapper with per-session state.

    Why this exists:
    - The server API is frame-based (one image per request), but video labeling
      sends repeated calls across frames.
    - The original `sam3_video` keeps exactly one tracking state, so multiple
      videos (or interleaved requests) overwrite each other.

    This wrapper supports:
    - `session_id`: isolates state per video/session so you can run multiple
      videos concurrently.
    - `save_prompt_id`: saves the current prompt (boxes) into a shared prompt
      bank.
    - `prompt_id`: loads a saved prompt and uses it as the initial prompt for a
      session/video.

    Parameter summary (all inside request `params`):
    - `session_id` (str): session key; default "default".
    - `reset_session` / `reset_tracker` (bool): clears session tracking state.
    - `prompt_id` (str): name of a saved prompt to load/reuse.
    - `save_prompt_id` (str): name to save the current prompt boxes under.

    Notes:
    - This is still a prompt-reuse tracker (not SAM3 temporal propagation).
    - Prompt bank is in-memory per model instance; restarting the server clears it.
    """

    def load(self):
        super().load()
        self._sessions: Dict[str, _SessionState] = {}
        self._prompt_bank: Dict[str, _PromptState] = {}

    def unload(self):
        for sid in list(self._sessions.keys()):
            self.reset_session(sid)
        self._sessions = {}
        self._prompt_bank = {}
        super().unload()

    def reset_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._cleanup_recording(session)

    def predict(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        session_id = self._get_session_id(params)
        want_reset = bool(params.get("reset_tracker") or params.get("reset_session"))

        session = self._sessions.get(session_id)
        if session is None:
            session = _SessionState()
            self._sessions[session_id] = session

        # Allow one request to both finalize and reset.
        # (Finalize happens after this frame's prediction is recorded.)
        want_finalize = bool(
            params.get("finalize_recording")
            or params.get("finalize_video")
            or params.get("close_session")
        )

        if want_reset and not want_finalize:
            # If reset is requested without finalize, still clean up any open writer.
            self.reset_session(session_id)
            session = _SessionState()
            self._sessions[session_id] = session

        text_prompt = params.get("text_prompt", "")
        marks = params.get("marks", [])
        prompt_id = params.get("prompt_id")
        save_prompt_id = params.get("save_prompt_id")

        h, w = int(image.shape[0]), int(image.shape[1])

        # If the frame size changes, rescale any stored prompts for this session.
        if session.last_frame_wh is not None and session.last_boxes_xyxy:
            prev_w, prev_h = session.last_frame_wh
            if (prev_w, prev_h) != (w, h):
                session.last_boxes_xyxy = self._rescale_boxes(
                    session.last_boxes_xyxy, from_wh=(prev_w, prev_h), to_wh=(w, h)
                )
                session.last_frame_wh = (w, h)

        # If no explicit prompt was provided, but a prompt_id is set, load it.
        if not marks and not (isinstance(text_prompt, str) and text_prompt.strip()):
            if isinstance(prompt_id, str) and prompt_id.strip():
                prompt = self._prompt_bank.get(prompt_id)
                if prompt and prompt.boxes_xyxy:
                    prompt_boxes = prompt.boxes_xyxy
                    if prompt.frame_wh is not None and prompt.frame_wh != (w, h):
                        prompt_boxes = self._rescale_boxes(
                            prompt_boxes, from_wh=prompt.frame_wh, to_wh=(w, h)
                        )
                    session.last_boxes_xyxy = list(prompt_boxes)
                    session.last_frame_wh = (w, h)

        # Prefer explicit prompts in request.
        if marks or (isinstance(text_prompt, str) and text_prompt.strip()):
            result = super().predict(image, params)
            result = self._apply_track_ids(result, params, session)

            shapes = list(result.get("shapes", []) or [])
            self._update_tracking_from_shapes(session, shapes, frame_wh=(w, h))

            # Optionally save the prompt derived from this call.
            if isinstance(save_prompt_id, str) and save_prompt_id.strip():
                self._prompt_bank[save_prompt_id] = _PromptState(
                    boxes_xyxy=list(session.last_boxes_xyxy),
                    frame_wh=session.last_frame_wh,
                )

            self._maybe_record_and_finalize(session_id, session, image, result, params)
            return result

        # Otherwise, if we have stored prompts, reuse them.
        if session.last_boxes_xyxy:
            reused_marks = [
                {
                    "type": "rectangle",
                    "label": 1,
                    "data": [float(x1), float(y1), float(x2), float(y2)],
                }
                for (x1, y1, x2, y2) in session.last_boxes_xyxy
            ]

            reused_params = dict(params)
            reused_params["marks"] = reused_marks
            result = self._predict_with_boxes(image, reused_marks, reused_params)
            result = self._apply_track_ids(result, reused_params, session)

            shapes = list(result.get("shapes", []) or [])
            self._update_tracking_from_shapes(session, shapes, frame_wh=(w, h))

            if isinstance(save_prompt_id, str) and save_prompt_id.strip():
                self._prompt_bank[save_prompt_id] = _PromptState(
                    boxes_xyxy=list(session.last_boxes_xyxy),
                    frame_wh=session.last_frame_wh,
                )

            self._maybe_record_and_finalize(session_id, session, image, result, params)
            return result

        logger.warning(
            f"sam3_video_session: no prompt provided and no state (session_id={session_id})"
        )
        result = {"shapes": [], "description": ""}
        self._maybe_record_and_finalize(session_id, session, image, result, params)
        return result

    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        # Windows + POSIX safe-ish folder name
        cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
        cleaned = cleaned.strip().strip(".")
        return cleaned or "video"

    def _get_recording_config(
        self, session_id: str, params: Dict[str, Any]
    ) -> Tuple[bool, Path, str, float, str]:
        enabled = bool(params.get("save_video") or params.get("record_video"))
        output_dir_raw = params.get("save_video_dir") or params.get("record_output_dir")
        if isinstance(output_dir_raw, str) and output_dir_raw.strip():
            base_dir = Path(output_dir_raw).expanduser()
        else:
            base_dir = Path("logs") / "video_exports"

        video_name_raw = params.get("video_name") or params.get("video_id") or session_id
        if not isinstance(video_name_raw, str):
            video_name_raw = str(video_name_raw)
        video_name = self._sanitize_folder_name(video_name_raw)

        fps = float(params.get("save_video_fps") or params.get("record_fps") or 30.0)
        fourcc = str(params.get("save_video_fourcc") or params.get("record_fourcc") or "mp4v")
        return enabled, base_dir, video_name, fps, fourcc

    def _ensure_recording_started(
        self,
        session: _SessionState,
        base_dir: Path,
        video_name: str,
        frame_wh: Tuple[int, int],
        fps: float,
        fourcc: str,
    ) -> Tuple[Path, Path]:
        out_dir = base_dir / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        video_path = out_dir / "video.mp4"
        tmp_jsonl_path = out_dir / "predictions.tmp.jsonl"

        # Start writer if needed or if size changes.
        if session.record_writer is None or session.record_writer_wh != frame_wh:
            if session.record_writer is not None:
                try:
                    session.record_writer.release()
                except Exception:
                    pass
            w, h = frame_wh
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*fourcc),
                float(fps),
                (int(w), int(h)),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"Failed to open video writer for {video_path}. Try --record_fourcc=avc1 or mp4v."
                )
            session.record_writer = writer
            session.record_writer_wh = frame_wh

        session.record_video_name = video_name
        session.record_output_dir = str(base_dir)
        session.record_tmp_jsonl_path = str(tmp_jsonl_path)
        return out_dir, tmp_jsonl_path

    @staticmethod
    def _serialize_result_for_json(result: Dict[str, Any]) -> Dict[str, Any]:
        shapes = list(result.get("shapes", []) or [])
        serialized_shapes = []
        for s in shapes:
            if hasattr(s, "model_dump"):
                serialized_shapes.append(s.model_dump(exclude_none=True))
            elif isinstance(s, dict):
                serialized_shapes.append(s)
            else:
                serialized_shapes.append({"value": str(s)})
        out: Dict[str, Any] = dict(result)
        out["shapes"] = serialized_shapes
        return out

    def _append_prediction_jsonl(self, tmp_jsonl_path: Path, payload: Dict[str, Any]) -> None:
        with open(tmp_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    def _cleanup_recording(self, session: _SessionState) -> None:
        if session.record_writer is not None:
            try:
                session.record_writer.release()
            except Exception:
                pass
        session.record_writer = None
        session.record_writer_wh = None

    def _finalize_recording(self, session_id: str, session: _SessionState) -> Optional[Path]:
        if not session.record_tmp_jsonl_path:
            self._cleanup_recording(session)
            return None

        tmp_jsonl_path = Path(session.record_tmp_jsonl_path)
        out_dir = tmp_jsonl_path.parent
        out_json_path = out_dir / "predictions.json"

        self._cleanup_recording(session)

        if tmp_jsonl_path.exists():
            items: List[Any] = []
            with open(tmp_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            try:
                tmp_jsonl_path.unlink()
            except Exception:
                pass

        logger.info(
            f"sam3_video_session: finalized recording for session_id={session_id} in {out_dir}"
        )
        return out_dir

    def _maybe_record_and_finalize(
        self,
        session_id: str,
        session: _SessionState,
        image: np.ndarray,
        result: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        enabled, base_dir, video_name, fps, fourcc = self._get_recording_config(session_id, params)
        want_finalize = bool(
            params.get("finalize_recording")
            or params.get("finalize_video")
            or params.get("close_session")
        )
        want_reset = bool(params.get("reset_tracker") or params.get("reset_session"))

        if enabled:
            h, w = int(image.shape[0]), int(image.shape[1])
            out_dir, tmp_jsonl_path = self._ensure_recording_started(
                session=session,
                base_dir=base_dir,
                video_name=video_name,
                frame_wh=(w, h),
                fps=fps,
                fourcc=fourcc,
            )

            # Write frame to video
            session.record_writer.write(image)

            payload = {
                "session_id": session_id,
                "video_name": video_name,
                "frame_index": int(session.record_frame_idx),
                "result": self._serialize_result_for_json(result),
            }
            self._append_prediction_jsonl(Path(tmp_jsonl_path), payload)
            session.record_frame_idx += 1

        if want_finalize and enabled:
            self._finalize_recording(session_id, session)
            if want_reset:
                self.reset_session(session_id)
        elif want_reset and not enabled:
            # Reset requested without recording enabled
            self.reset_session(session_id)

    @staticmethod
    def _get_session_id(params: Dict[str, Any]) -> str:
        session_id = params.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
        return "default"

    @staticmethod
    def _rescale_boxes(
        boxes: List[BoxXYXY], from_wh: Tuple[int, int], to_wh: Tuple[int, int]
    ) -> List[BoxXYXY]:
        from_w, from_h = from_wh
        to_w, to_h = to_wh
        sx = to_w / max(from_w, 1)
        sy = to_h / max(from_h, 1)
        return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (x1, y1, x2, y2) in boxes]

    def _update_tracking_from_shapes(
        self, session: _SessionState, shapes: List[Shape], frame_wh: Tuple[int, int]
    ) -> None:
        boxes = self._shapes_to_boxes_xyxy(shapes)
        if boxes:
            session.last_boxes_xyxy = boxes
            session.last_frame_wh = frame_wh

    @staticmethod
    def _shapes_to_boxes_xyxy(shapes: List[Shape]) -> List[BoxXYXY]:
        boxes: List[BoxXYXY] = []

        # Prefer rectangles (if present) because they are already boxes.
        rect_shapes = [s for s in shapes if s.shape_type == "rectangle"]
        source_shapes = rect_shapes if rect_shapes else shapes

        for shape in source_shapes:
            bbox = Sam3VideoSession._bbox_from_shape(shape)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                continue
            boxes.append((x1, y1, x2, y2))

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
        self, result: Dict[str, Any], params: Dict[str, Any], session: _SessionState
    ) -> Dict[str, Any]:
        shapes: List[Shape] = list(result.get("shapes", []) or [])
        if not shapes:
            self._age_tracks(session, params)
            return result

        detections = self._group_shapes_into_detections(shapes)
        if not detections:
            self._age_tracks(session, params)
            return result

        det_boxes = [d["bbox"] for d in detections]
        assignments = self._match_detections_to_tracks(session, det_boxes, params)

        updated_shapes = shapes[:]
        for det_idx, track_id in assignments.items():
            for shape_index in detections[det_idx]["shape_indices"]:
                updated_shapes[shape_index] = updated_shapes[shape_index].model_copy(
                    update={"group_id": int(track_id)}
                )

        return {**result, "shapes": updated_shapes}

    def _age_tracks(self, session: _SessionState, params: Dict[str, Any]) -> None:
        max_missed = int(
            params.get("max_track_missed", self.params.get("max_track_missed", 30))
        )
        for tid in list(session.tracks.keys()):
            session.track_misses[tid] = int(session.track_misses.get(tid, 0)) + 1
            if session.track_misses[tid] > max_missed:
                session.tracks.pop(tid, None)
                session.track_misses.pop(tid, None)

    def _group_shapes_into_detections(self, shapes: List[Shape]) -> List[Dict[str, Any]]:
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

        for idx, bbox in zip(rect_indices, rect_boxes):
            detections.append({"bbox": bbox, "shape_indices": [idx]})

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
            if best_j >= 0 and best_iou >= 0.05:
                detections[best_j]["shape_indices"].append(i)
            else:
                detections.append({"bbox": bbox, "shape_indices": [i]})

        return detections

    def _match_detections_to_tracks(
        self, session: _SessionState, det_boxes: List[BoxXYXY], params: Dict[str, Any]
    ) -> Dict[int, int]:
        iou_thresh = float(
            params.get(
                "track_iou_threshold", self.params.get("track_iou_threshold", 0.3)
            )
        )
        max_missed = int(
            params.get("max_track_missed", self.params.get("max_track_missed", 30))
        )

        track_ids = list(session.tracks.keys())
        used_tracks = set()
        used_dets = set()
        pairs: List[Tuple[float, int, int]] = []  # (iou, det_i, track_j)

        for di, db in enumerate(det_boxes):
            for tj, tid in enumerate(track_ids):
                iou = self._iou(db, session.tracks[tid])
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

        for di, db in enumerate(det_boxes):
            if di in assignments:
                continue
            tid = session.next_track_id
            session.next_track_id += 1
            session.tracks[tid] = db
            session.track_misses[tid] = 0
            assignments[di] = tid

        matched_track_ids = set(assignments.values())
        for di, tid in assignments.items():
            session.tracks[tid] = det_boxes[di]
            session.track_misses[tid] = 0

        for tid in list(session.tracks.keys()):
            if tid in matched_track_ids:
                continue
            session.track_misses[tid] = int(session.track_misses.get(tid, 0)) + 1
            if session.track_misses[tid] > max_missed:
                session.tracks.pop(tid, None)
                session.track_misses.pop(tid, None)

        return assignments
