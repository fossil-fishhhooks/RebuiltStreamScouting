"""
robot_tracker.py — FRC robot tracker with object permanence.

Detection uses YOLOv8 (ultralytics) on the "person" class as a proxy for
robots — robots are large, upright, vaguely person-shaped blobs from above.
If you have a custom FRC model, point ROBOT_YOLO_MODEL at its weights file.

The RobotTracker / RobotTrack / Kalman stack is unchanged from the original
design; only the detection layer has been replaced.

Requirements:
    pip install ultralytics
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tunables — override in config.py if you add them there
# ---------------------------------------------------------------------------
try:
    from config import (
        ROBOT_MAX_DIST,
        ROBOT_GHOST_FRAMES,
        ROBOT_DORMANT_FRAMES,
        ROBOT_REIDENTIFY_DIST,
        ROBOT_MAX_TRAIL,
        ROBOT_PROCESS_NOISE,
        ROBOT_MEASUREMENT_NOISE,
        ROBOT_YOLO_MODEL,
        ROBOT_YOLO_CONF,
        ROBOT_YOLO_CLASSES,
        ROBOT_YOLO_IMGSZ,
    )
except ImportError:
    ROBOT_MAX_DIST          = 60     # px — max centroid jump per frame for a match
    ROBOT_GHOST_FRAMES      = 12     # frames without detection before track goes dormant
    ROBOT_DORMANT_FRAMES    = 60     # frames to keep dormant track before hard-delete
    ROBOT_REIDENTIFY_DIST   = 120    # px — dormant track search radius for re-ID
    ROBOT_MAX_TRAIL         = 60     # centroid history length
    ROBOT_PROCESS_NOISE     = 5e-2   # Kalman Q scalar
    ROBOT_MEASUREMENT_NOISE = 5e-1   # Kalman R scalar
    # ---- YOLO ----
    ROBOT_YOLO_MODEL   = "yolov8n.pt"  # path to weights; downloads automatically on first run
    ROBOT_YOLO_CONF    = 0.25          # detection confidence threshold
    ROBOT_YOLO_CLASSES = None          # None = all classes; [0] = person only, etc.
    ROBOT_YOLO_IMGSZ   = 640           # inference resolution (px)


# ---------------------------------------------------------------------------
# Lazy YOLO model loader — instantiated once, reused every frame
# ---------------------------------------------------------------------------
_yolo_model = None

def _get_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(ROBOT_YOLO_MODEL)
        _yolo_model.fuse()  # fuse Conv+BN layers for faster CPU inference
    return _yolo_model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_robots(
        frame: np.ndarray,
        alliance: str = "both",  # kept for API compatibility; YOLO is colour-agnostic
) -> List[Tuple[int, int, int, int, str]]:
    """
    Detect FRC robots using YOLO object detection.

    Returns a list of (cx, cy, w, h, alliance_colour) tuples.
    alliance_colour is inferred from bumper colour after detection:
    a small HSV check on the bounding-box region determines red/blue/unknown.

    Parameters
    ----------
    frame    : BGR frame (full crop, no blackouts needed).
    alliance : "red", "blue", or "both" — filters returned detections by
               inferred bumper colour. Pass "both" to keep everything.
    """
    model  = _get_model()
    results = model(
        frame,
        conf=ROBOT_YOLO_CONF,
        classes=ROBOT_YOLO_CLASSES,
        imgsz=ROBOT_YOLO_IMGSZ,
        verbose=False,
    )[0]

    detections: List[Tuple[int, int, int, int, str]] = []
    h_frame, w_frame = frame.shape[:2]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # clamp to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        colour = _infer_alliance(frame, x1, y1, x2, y2)
        if alliance != "both" and colour != alliance and colour != "unknown":
            continue

        detections.append((cx, cy, w, h, colour))

    return detections


def _infer_alliance(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
) -> str:
    """
    Determine alliance colour from bumper pixels inside the bounding box.
    Samples a horizontal strip near the vertical centre of the box where
    bumpers are most likely to appear.
    """
    h = y2 - y1
    # sample middle third vertically — bumpers sit around robot mid-height
    sy1 = y1 + h // 3
    sy2 = y1 + 2 * h // 3
    roi  = frame[sy1:sy2, x1:x2]
    if roi.size == 0:
        return "unknown"

    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red: hue 0-15 or 165-180, sat>150, val>80
    red_mask = (
            cv2.inRange(hsv, (0,   150, 80), (15,  255, 255)) |
            cv2.inRange(hsv, (165, 150, 80), (180, 255, 255))
    )
    # blue: hue 100-115, sat>180, val>100
    blue_mask = cv2.inRange(hsv, (100, 180, 100), (115, 255, 210))

    red_px  = int(cv2.countNonZero(red_mask))
    blue_px = int(cv2.countNonZero(blue_mask))
    total   = roi.shape[0] * roi.shape[1]
    threshold = max(10, int(total * 0.03))  # at least 3% of roi must match

    if red_px > threshold and red_px > blue_px:
        return "red"
    if blue_px > threshold and blue_px > red_px:
        return "blue"
    return "unknown"


# ---------------------------------------------------------------------------
# Kalman filter factory (tuned for robots)
# ---------------------------------------------------------------------------

def _make_robot_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)   # state: [x, y, vx, vy], measurement: [x, y]
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov      = np.eye(4, dtype=np.float32) * ROBOT_PROCESS_NOISE
    kf.measurementNoiseCov  = np.eye(2, dtype=np.float32) * ROBOT_MEASUREMENT_NOISE
    kf.errorCovPost         = np.eye(4, dtype=np.float32)
    return kf


# ---------------------------------------------------------------------------
# RobotTrack
# ---------------------------------------------------------------------------

@dataclass
class RobotTrack:
    """
    Single tracked robot with full object-permanence support.

    States
    ------
    active   — seen recently (ghost_count <= ROBOT_GHOST_FRAMES)
    ghost    — prediction-only; not yet dormant
    dormant  — long-term memory; eligible for re-identification
    """

    id: int
    alliance: str                         # "red" | "blue" | "unknown"

    kf: cv2.KalmanFilter = field(repr=False, default_factory=_make_robot_kalman)

    ghost_count:   int = 0
    dormant_count: int = 0               # frames since track went dormant
    is_dormant:    bool = False

    # Bounding-box size (for drawing, updated on each detection)
    w: int = 60
    h: int = 60

    # History of smoothed centroids (capped at ROBOT_MAX_TRAIL)
    trail: deque = field(default_factory=lambda: deque(maxlen=ROBOT_MAX_TRAIL))

    def __post_init__(self):
        pass  # kf must be initialised externally — see RobotTracker.spawn()

    # ------------------------------------------------------------------
    def predict(self) -> Tuple[int, int]:
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def update(self, cx: int, cy: int, w: int, h: int) -> None:
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        self.ghost_count  = 0
        self.dormant_count = 0
        self.is_dormant   = False
        self.w, self.h    = w, h
        self.trail.append((cx, cy))

    def position(self) -> Tuple[int, int]:
        s = self.kf.statePost
        return int(s[0]), int(s[1])

    def velocity(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[2]), float(s[3])

    @property
    def state(self) -> str:
        if self.is_dormant:
            return "dormant"
        if self.ghost_count > 0:
            return "ghost"
        return "active"


# ---------------------------------------------------------------------------
# RobotTracker
# ---------------------------------------------------------------------------

class RobotTracker:
    """
    Multi-object tracker for FRC robots with object permanence.

    Usage
    -----
        tracker = RobotTracker()

        for each frame:
            detections = detect_robots(crop, alliance="both")
            live, dormant = tracker.update(detections)
            # live    → {track_id: RobotTrack}  (active + ghost)
            # dormant → {track_id: RobotTrack}  (long-term memory)

    Object permanence pipeline
    --------------------------
    1. Active/ghost tracks are predicted forward each frame.
    2. Detections are matched greedily to predictions within ROBOT_MAX_DIST.
    3. Unmatched tracks increment ghost_count.
    4. Once ghost_count > ROBOT_GHOST_FRAMES the track moves to `dormant`.
    5. New detections first try to re-identify a dormant track within
       ROBOT_REIDENTIFY_DIST before spawning a fresh ID.
    6. Dormant tracks age out after ROBOT_DORMANT_FRAMES and are deleted.
    """

    def __init__(self) -> None:
        self.tracks:  Dict[int, RobotTrack] = {}   # live (active + ghost)
        self.dormant: Dict[int, RobotTrack] = {}   # long-term memory
        self.next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
            self,
            detections: List[Tuple[int, int, int, int, str]],
    ) -> Tuple[Dict[int, RobotTrack], Dict[int, RobotTrack]]:
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        detections : list of (cx, cy, w, h, alliance) from detect_robots().

        Returns
        -------
        (live_tracks, dormant_tracks)
        """
        # Step 1: predict all live tracks
        predictions = {tid: t.predict() for tid, t in self.tracks.items()}

        # Step 2: match detections → live tracks
        det_coords  = [(cx, cy) for cx, cy, *_ in detections]
        matched_det, matched_trk = self._greedy_match(det_coords, predictions, ROBOT_MAX_DIST)

        # Step 3: update matched tracks
        matched_trk_ids = set()
        for di, tid in zip(matched_det, matched_trk):
            cx, cy, w, h, alliance = detections[di]
            self.tracks[tid].update(cx, cy, w, h)
            matched_trk_ids.add(tid)

        # Step 4: handle unmatched detections — try re-ID, else spawn
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
        for di in unmatched_dets:
            cx, cy, w, h, alliance = detections[di]
            dormant_id = self._try_reidentify(cx, cy)
            if dormant_id is not None:
                # Resurrect dormant track
                track = self.dormant.pop(dormant_id)
                track.kf.statePost[0] = cx
                track.kf.statePost[1] = cy
                track.update(cx, cy, w, h)
                self.tracks[dormant_id] = track
                matched_trk_ids.add(dormant_id)
            else:
                new_id = self._spawn(cx, cy, w, h, alliance)
                matched_trk_ids.add(new_id)

        # Step 5: age unmatched live tracks → ghost → dormant
        for tid in list(self.tracks.keys()):
            if tid in matched_trk_ids:
                continue
            track = self.tracks[tid]
            track.ghost_count += 1
            if track.ghost_count > ROBOT_GHOST_FRAMES:
                # Move to dormant pool
                track.is_dormant = True
                self.dormant[tid] = track
                del self.tracks[tid]

        # Step 6: age dormant tracks, delete stale ones
        for tid in list(self.dormant.keys()):
            self.dormant[tid].dormant_count += 1
            if self.dormant[tid].dormant_count > ROBOT_DORMANT_FRAMES:
                del self.dormant[tid]

        return dict(self.tracks), dict(self.dormant)

    # ------------------------------------------------------------------
    # Drawing helper
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay tracks onto *frame* in-place and return it.

        Colour coding:
          active  — alliance colour (red/blue) solid box + trail
          ghost   — dim alliance colour, dashed-style box
          dormant — grey, small cross at last known position
        """
        ALLIANCE_BGR = {
            "red":     (0,   0,   220),
            "blue":    (220, 60,  0),
            "unknown": (160, 160, 160),
        }

        # Dormant — just a small marker
        for tid, track in self.dormant.items():
            px, py = track.position()
            cv2.drawMarker(frame, (px, py), (120, 120, 120),
                           cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
            cv2.putText(frame, f"D{tid}", (px + 7, py - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        # Live tracks
        for tid, track in self.tracks.items():
            px, py  = track.position()
            colour  = ALLIANCE_BGR.get(track.alliance, (160, 160, 160))
            is_ghost = track.ghost_count > 0

            # Dim colour for ghosts
            if is_ghost:
                alpha  = 1.0 - track.ghost_count / (ROBOT_GHOST_FRAMES + 1)
                colour = tuple(int(c * alpha) for c in colour)

            # Bounding box
            half_w, half_h = track.w // 2, track.h // 2
            tl = (px - half_w, py - half_h)
            br = (px + half_w, py + half_h)
            thickness = 1 if is_ghost else 2
            cv2.rectangle(frame, tl, br, colour, thickness)

            # Trail
            pts = list(track.trail)
            for i in range(1, len(pts)):
                fade = i / len(pts)
                c    = tuple(int(ch * fade) for ch in colour)
                cv2.line(frame, pts[i - 1], pts[i], c, 1, cv2.LINE_AA)

            # Centroid dot
            cv2.circle(frame, (px, py), 4, colour, -1, cv2.LINE_AA)

            # Label
            label = f"{'G' if is_ghost else ''}{tid}[{track.alliance[0].upper()}]"
            cv2.putText(frame, label, (px + half_w + 3, py - half_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn(self, cx: int, cy: int, w: int, h: int, alliance: str) -> int:
        tid   = self.next_id
        track = RobotTrack(id=tid, alliance=alliance)
        track.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        track.trail.append((cx, cy))
        track.w, track.h   = w, h
        self.tracks[tid]   = track
        self.next_id      += 1
        return tid

    @staticmethod
    def _greedy_match(
            det_coords: List[Tuple[int, int]],
            predictions: Dict[int, Tuple[int, int]],
            max_dist: float,
    ) -> Tuple[List[int], List[int]]:
        """
        Greedy nearest-neighbour matching.  Resolves conflicts by keeping the
        closer detection for each track.

        Returns (matched_det_indices, matched_track_ids) — parallel lists.
        """
        if not det_coords or not predictions:
            return [], []

        # Build (distance, det_idx, track_id) candidates
        candidates = []
        tids = list(predictions.keys())
        for di, (dx, dy) in enumerate(det_coords):
            for tid in tids:
                px, py = predictions[tid]
                dist   = ((px - dx) ** 2 + (py - dy) ** 2) ** 0.5
                if dist <= max_dist:
                    candidates.append((dist, di, tid))

        candidates.sort()               # ascending distance

        used_dets  = set()
        used_trks  = set()
        matched_d  = []
        matched_t  = []

        for dist, di, tid in candidates:
            if di in used_dets or tid in used_trks:
                continue
            matched_d.append(di)
            matched_t.append(tid)
            used_dets.add(di)
            used_trks.add(tid)

        return matched_d, matched_t

    def _try_reidentify(self, cx: int, cy: int) -> Optional[int]:
        """
        Find the closest dormant track within ROBOT_REIDENTIFY_DIST.
        Returns its track_id or None.
        """
        best_id, best_dist = None, ROBOT_REIDENTIFY_DIST
        for tid, track in self.dormant.items():
            px, py = track.position()
            dist   = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id   = tid
        return best_id