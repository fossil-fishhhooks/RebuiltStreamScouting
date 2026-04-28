"""
robot_tracker.py — FRC robot tracker.

Design
------
- Exactly 6 permanent slots (IDs 0-5), pre-allocated at startup, never
  deleted or reassigned.
- Hungarian (optimal) assignment each frame — prevents ID swaps even when
  robots cross.
- All coordinates are in CROP-SPACE.  The caller must convert detections
  before passing them in, and must pass the cropped frame for optical flow.
- Kalman filter predicts forward when a robot is invisible; optical-flow
  (GPU SparsePyrLK if CUDA available, CPU LK fallback) keeps the centroid
  moving between YOLO hits.
- Every slot accumulates an unbounded permanent path drawn as a fading line.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

from robot_detector import detect as _tiled_detect

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from config import (
        ROBOT_MAX_DIST,
        ROBOT_GHOST_FRAMES,
        ROBOT_MAX_TRAIL,
        ROBOT_PROCESS_NOISE,
        ROBOT_MEASUREMENT_NOISE,
    )
except ImportError:
    ROBOT_MAX_DIST          = 60
    ROBOT_GHOST_FRAMES      = 12
    ROBOT_MAX_TRAIL         = 60
    ROBOT_PROCESS_NOISE     = 5e-2
    ROBOT_MEASUREMENT_NOISE = 0.5

NUM_ROBOTS = 6

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

_USE_GPU = _cuda_available()

# ---------------------------------------------------------------------------
# Optical-flow tracker  (operates entirely in whatever frame space it's given)
# ---------------------------------------------------------------------------

class _OpticFlowTracker:
    def __init__(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        self._ok        = False
        self._cx        = (bbox[0] + bbox[2]) // 2
        self._cy        = (bbox[1] + bbox[3]) // 2
        self._prev_gray = None
        self._pts       = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = [max(0, v) for v in bbox]
        x2 = min(x2, gray.shape[1])
        y2 = min(y2, gray.shape[0])
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return

        pts = cv2.goodFeaturesToTrack(roi, maxCorners=30, qualityLevel=0.2,
                                      minDistance=5, blockSize=5)
        if pts is None or len(pts) == 0:
            return

        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1

        self._prev_gray = gray
        self._pts       = pts.astype(np.float32)
        self._ok        = True

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        if not self._ok or self._pts is None or self._prev_gray is None:
            return False, (self._cx, self._cy)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if _USE_GPU:
            try:
                lk       = cv2.cuda.SparsePyrLKOpticalFlow_create(winSize=(21, 21), maxLevel=3)
                prev_gpu = cv2.cuda_GpuMat()
                curr_gpu = cv2.cuda_GpuMat()
                pts_gpu  = cv2.cuda_GpuMat()
                prev_gpu.upload(self._prev_gray)
                curr_gpu.upload(gray)
                pts_gpu.upload(self._pts)
                next_gpu, status_gpu, _ = lk.calc(prev_gpu, curr_gpu, pts_gpu, None)
                next_pts = next_gpu.download()
                status   = status_gpu.download()
            except Exception:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._pts, None)
        else:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._pts, None)

        if next_pts is None or status is None:
            self._ok = False
            return False, (self._cx, self._cy)

        good = next_pts[status.flatten() == 1].reshape(-1, 2)
        if len(good) < 4:
            self._ok = False
            return False, (self._cx, self._cy)

        cx = int(np.median(good[:, 0]))
        cy = int(np.median(good[:, 1]))
        self._cx        = cx
        self._cy        = cy
        self._pts       = good.reshape(-1, 1, 2)
        self._prev_gray = gray
        return True, (cx, cy)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

def _make_robot_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * ROBOT_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * ROBOT_MEASUREMENT_NOISE
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


# ---------------------------------------------------------------------------
# RobotTrack
# ---------------------------------------------------------------------------

# Sentinel: slot has never been matched to a detection yet.
_UNINITIALIZED = object()

@dataclass
class RobotTrack:
    id:       int
    alliance: str

    kf: cv2.KalmanFilter = field(repr=False, default_factory=_make_robot_kalman)

    ghost_count:   int  = 0
    initialized:   bool = False   # False until first real detection
    w:             int  = 60
    h:             int  = 60

    # Raw YOLO box in crop-space — updated every time we get a real detection.
    # Drawn as the primary bounding box so it reflects actual model output.
    last_box: Optional[Tuple[int,int,int,int]] = None   # (x1, y1, x2, y2)

    trail:      deque = field(default_factory=lambda: deque(maxlen=ROBOT_MAX_TRAIL))
    perma_path: list  = field(default_factory=list)

    _of_tracker: object = field(default=None, repr=False)

    def predict(self) -> Tuple[int, int]:
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def update(self, cx: int, cy: int, w: int, h: int,
               alliance: str = "unknown",
               raw_box: Optional[Tuple[int,int,int,int]] = None) -> None:
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        self.ghost_count  = 0
        self.initialized  = True
        self.w, self.h    = w, h
        self.last_box     = raw_box if raw_box is not None else (
            cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        )
        self.trail.append((cx, cy))
        self.perma_path.append((cx, cy))
        if self.alliance == "unknown" and alliance != "unknown":
            self.alliance = alliance

    def update_from_optic_flow(self, cx: int, cy: int) -> None:
        """Soft update — does not reset ghost_count or count as a detection."""
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        self.trail.append((cx, cy))
        self.perma_path.append((cx, cy))

    def position(self) -> Tuple[int, int]:
        s = self.kf.statePost
        return int(s[0]), int(s[1])

    @property
    def state(self) -> str:
        return "ghost" if self.ghost_count > 0 else "active"


# ---------------------------------------------------------------------------
# Alliance detection  (still runs in raw-frame space — caller's job)
# ---------------------------------------------------------------------------

def _infer_alliance(frame: np.ndarray, x1, y1, x2, y2) -> str:
    h    = y2 - y1
    sy1  = y1 + h // 3
    sy2  = y1 + 2 * h // 3
    roi  = frame[sy1:sy2, x1:x2]
    if roi.size == 0:
        return "unknown"
    hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask  = (
        cv2.inRange(hsv, (0,   150, 80), (15,  255, 255)) |
        cv2.inRange(hsv, (165, 150, 80), (180, 255, 255))
    )
    blue_mask = cv2.inRange(hsv, (100, 180, 100), (115, 255, 210))
    red_px    = int(cv2.countNonZero(red_mask))
    blue_px   = int(cv2.countNonZero(blue_mask))
    total     = roi.shape[0] * roi.shape[1]
    thresh    = max(10, int(total * 0.03))
    if red_px > thresh and red_px > blue_px:
        return "red"
    if blue_px > thresh and blue_px > red_px:
        return "blue"
    return "unknown"


def detect_robots(frame: np.ndarray, alliance: str = "both", max_stale_frames: int = 0):
    """Returns detections in raw-frame coords: [(cx, cy, w, h, alliance, x1, y1, x2, y2), ...]"""
    h_frame, w_frame = frame.shape[:2]
    detections = []
    for x1, y1, x2, y2, conf, cls in _tiled_detect(frame, max_stale_frames=max_stale_frames):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        w      = x2 - x1
        h      = y2 - y1
        cx     = x1 + w // 2
        cy     = y1 + h // 2
        colour = _infer_alliance(frame, x1, y1, x2, y2)
        if alliance != "both" and colour != alliance and colour != "unknown":
            continue
        detections.append((cx, cy, w, h, colour, x1, y1, x2, y2))
    return detections


# ---------------------------------------------------------------------------
# Per-slot colours
# ---------------------------------------------------------------------------

_SLOT_COLORS = [
    (0,   80,  255),   # 0
    (255, 100, 0  ),   # 1
    (0,   210, 0  ),   # 2
    (0,   210, 255),   # 3
    (210, 0,   210),   # 4
    (0,   165, 255),   # 5
]


# ---------------------------------------------------------------------------
# RobotTracker
# ---------------------------------------------------------------------------

class RobotTracker:
    """
    Six permanent robot slots, IDs 0-5.

    IMPORTANT: all coordinates passed in must be in CROP-SPACE.
    Pass the cropped frame (not raw_frame) as crop_frame so optical flow
    operates in the same coordinate system as the detections.
    """

    def __init__(self) -> None:
        self.tracks: Dict[int, RobotTrack] = {
            i: RobotTrack(id=i, alliance="unknown") for i in range(NUM_ROBOTS)
        }
        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    def update(
            self,
            detections: List[Tuple],   # (cx, cy, w, h, alliance[, x1, y1, x2, y2]) in CROP-SPACE
            crop_frame: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, RobotTrack], Dict]:
        """
        Parameters
        ----------
        detections : [(cx, cy, w, h, alliance)] or [(cx, cy, w, h, alliance, x1, y1, x2, y2)]
                     in CROP-SPACE.
        crop_frame : the cropped BGR frame (same coord space as detections).
        """
        self._frame_idx += 1

        # Normalise to always have raw_box field (last 4 elements)
        norm_dets = []
        for d in detections:
            cx, cy, w, h, alliance = d[0], d[1], d[2], d[3], d[4]
            if len(d) >= 9:
                raw_box = (d[5], d[6], d[7], d[8])
            else:
                raw_box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            norm_dets.append((cx, cy, w, h, alliance, raw_box))

        # ── Optical-flow update for ghosted, initialized slots ────────────
        if crop_frame is not None:
            for track in self.tracks.values():
                if track.initialized and track.ghost_count > 0 and track._of_tracker is not None:
                    ok, (cx, cy) = track._of_tracker.update(crop_frame)
                    if ok:
                        track.update_from_optic_flow(cx, cy)

        # ── Kalman predict (only initialized slots participate in matching) ─
        predictions = {
            tid: t.predict()
            for tid, t in self.tracks.items()
            if t.initialized
        }

        # Strip to (cx, cy, w, h, alliance) for the matcher
        match_dets = [(d[0], d[1], d[2], d[3], d[4]) for d in norm_dets]

        # ── Hungarian assignment ──────────────────────────────────────────
        matched = self._hungarian_match(match_dets, predictions)
        # matched: {slot_id: det_index}

        # Also let uninitialized slots grab detections not yet claimed
        claimed_dets = set(matched.values())
        for tid, track in self.tracks.items():
            if track.initialized or tid in matched:
                continue
            for di, d in enumerate(norm_dets):
                if di not in claimed_dets:
                    matched[tid] = di
                    claimed_dets.add(di)
                    break

        # ── Update matched slots ──────────────────────────────────────────
        matched_ids = set(matched.keys())
        for tid, di in matched.items():
            cx, cy, w, h, alliance, raw_box = norm_dets[di]
            self.tracks[tid].update(cx, cy, w, h, alliance, raw_box=raw_box)
            if crop_frame is not None:
                x1, y1, x2, y2 = raw_box
                self.tracks[tid]._of_tracker = _OpticFlowTracker(
                    crop_frame, (x1, y1, x2, y2))

        # ── Ghost unmatched initialized slots ─────────────────────────────
        for tid, track in self.tracks.items():
            if tid not in matched_ids and track.initialized:
                track.ghost_count += 1

        return dict(self.tracks), {}

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw all initialized slots onto the cropped frame."""
        for tid, track in self.tracks.items():
            if not track.initialized:
                continue

            slot_color = _SLOT_COLORS[tid % len(_SLOT_COLORS)]
            px, py     = track.position()
            is_ghost   = track.ghost_count > 0

            alpha = max(0.25, 1.0 - track.ghost_count / (ROBOT_GHOST_FRAMES + 1)) \
                    if is_ghost else 1.0
            draw_color = tuple(int(c * alpha) for c in slot_color)

            # ── Permanent path (thin, fades from old → new) ───────────────
            ppath = track.perma_path
            n     = len(ppath)
            if n >= 2:
                for i in range(1, n):
                    fade = 0.15 + 0.85 * (i / n)
                    c    = tuple(int(ch * fade * alpha) for ch in slot_color)
                    cv2.line(frame, ppath[i - 1], ppath[i], c, 1, cv2.LINE_AA)

            # ── Recent trail (thick, bright) ──────────────────────────────
            pts = list(track.trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], draw_color, 2, cv2.LINE_AA)

            # ── Bounding box ──────────────────────────────────────────────
            # Active: use the actual YOLO box stored at last detection.
            # Ghosted: reconstruct from Kalman centroid + last known size.
            if not is_ghost and track.last_box is not None:
                tl = (track.last_box[0], track.last_box[1])
                br = (track.last_box[2], track.last_box[3])
                box_w = track.last_box[2] - track.last_box[0]
                box_h = track.last_box[3] - track.last_box[1]
            else:
                half_w, half_h = track.w // 2, track.h // 2
                tl = (px - half_w, py - half_h)
                br = (px + half_w, py + half_h)
                box_w, box_h = track.w, track.h

            thickness = 1 if is_ghost else 2
            cv2.rectangle(frame, tl, br, draw_color, thickness)

            # ── Centroid dot ──────────────────────────────────────────────
            cx_dot = (tl[0] + br[0]) // 2 if not is_ghost and track.last_box else px
            cy_dot = (tl[1] + br[1]) // 2 if not is_ghost and track.last_box else py
            cv2.circle(frame, (cx_dot, cy_dot), 4, draw_color, -1, cv2.LINE_AA)

            # ── Label (top-right of box) ──────────────────────────────────
            g_tag = "G" if is_ghost else ""
            label = f"{g_tag}{tid}[{track.alliance[0].upper()}]"
            cv2.putText(frame, label, (br[0] + 3, tl[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _hungarian_match(
            detections: List[Tuple[int, int, int, int, str]],
            predictions: Dict[int, Tuple[int, int]],
    ) -> Dict[int, int]:
        """Returns {slot_id: det_index} for pairs within ROBOT_MAX_DIST."""
        if not detections or not predictions:
            return {}

        slot_ids = list(predictions.keys())
        n_slots  = len(slot_ids)
        n_dets   = len(detections)

        cost = np.full((n_slots, n_dets), fill_value=1e6, dtype=np.float64)
        for si, tid in enumerate(slot_ids):
            px, py = predictions[tid]
            for di, (cx, cy, *_) in enumerate(detections):
                cost[si, di] = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

        row_ind, col_ind = linear_sum_assignment(cost)

        result = {}
        for si, di in zip(row_ind, col_ind):
            if cost[si, di] <= ROBOT_MAX_DIST:
                result[slot_ids[si]] = di
        return result