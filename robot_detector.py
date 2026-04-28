"""
robot_detector.py — tiled YOLOv8 inference with optical-flow supplement and
image-based appearance gating.

Key guarantees
--------------
* At most NUM_ROBOTS (6) detections are ever returned.
* Every detection passes ROBOT_MIN_AREA and shape / aspect-ratio filters.
* YOLO runs in a background thread; detect() is non-blocking by default.
* Optical-flow is used as a *primary* measurement source (low noise), feeding
  per-slot Kalman filters directly — not just a trail nudge.
* Before a fresh YOLO box is accepted as the continuation of a tracked slot,
  it must pass an image-appearance check (histogram correlation + template
  NCC) against the slot's stored reference crop.  Detections that fail the
  check are demoted to the fallback pool.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from config import ROBOT_MIN_AREA, ROBOT_MAX_AREA
except ImportError:
    ROBOT_MIN_AREA = 1400
    ROBOT_MAX_AREA = 3000

NUM_ROBOTS = 6

# Tiled-inference settings (mirrored from debug_model.py)
MODEL_PATH         = "YOLOv8n-6k-variated.pt"
TILE_SIZE          = 640
OVERLAP            = 0.47
CONF_THRESH        = 0.4
MAX_BOX_AREA       = 480 * 480 * 0.06
ASPECT_AREA_THRESH = MAX_BOX_AREA * 0.25
ASPECT_MAX_LOOSE   = 2.5
ASPECT_MAX_TIGHT   = 1.35
NMS_IOU            = 0.38

# Appearance gating
APPEARANCE_ACCEPT_THRESH = 0.45   # score below this → demote (not hard-reject)
APPEARANCE_PATCH_SIZE    = (48, 48)

# Optical-flow Kalman noise  — tight = heavy trust in OF
OF_MEASUREMENT_NOISE   = 0.15
YOLO_MEASUREMENT_NOISE = 0.9


# ---------------------------------------------------------------------------
# YOLO model singleton
# ---------------------------------------------------------------------------
_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Tiled inference helpers  (inlined from debug_model.py)
# ---------------------------------------------------------------------------

def _generate_tiles(w: int, h: int) -> List[Tuple[int, int, int, int]]:
    stride = int(TILE_SIZE * (1.0 - OVERLAP))
    xs = list(range(0, w - TILE_SIZE, stride))
    if not xs or xs[-1] != w - TILE_SIZE:
        xs.append(w - TILE_SIZE)
    ys = list(range(0, h - TILE_SIZE, stride))
    if not ys or ys[-1] != h - TILE_SIZE:
        ys.append(h - TILE_SIZE)
    return [(x, y, x + TILE_SIZE, y + TILE_SIZE) for y in ys for x in xs]


def _nms(dets: list, iou_thresh: float) -> list:
    if not dets:
        return []
    d = np.array(dets, dtype=np.float32)
    order = np.argsort(-d[:, 4])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        order = order[1:]
        if not order.size:
            break
        ix1 = np.maximum(d[i, 0], d[order, 0])
        iy1 = np.maximum(d[i, 1], d[order, 1])
        ix2 = np.minimum(d[i, 2], d[order, 2])
        iy2 = np.minimum(d[i, 3], d[order, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = (
            (d[i, 2] - d[i, 0]) * (d[i, 3] - d[i, 1])
            + (d[order, 2] - d[order, 0]) * (d[order, 3] - d[order, 1])
            - inter
        )
        order = order[np.where(union > 0, inter / union, 0) < iou_thresh]
    return d[keep].tolist()


def _passes_shape(gx1: float, gy1: float, gx2: float, gy2: float) -> bool:
    bw, bh = gx2 - gx1, gy2 - gy1
    area   = bw * bh
    if area < ROBOT_MIN_AREA or area > MAX_BOX_AREA:
        return False
    if area > ASPECT_AREA_THRESH:
        aspect  = max(bw, bh) / (min(bw, bh) + 1e-6)
        t       = (area - ASPECT_AREA_THRESH) / (MAX_BOX_AREA - ASPECT_AREA_THRESH)
        allowed = ASPECT_MAX_LOOSE + t * (ASPECT_MAX_TIGHT - ASPECT_MAX_LOOSE)
        if aspect > allowed:
            return False
    return True


def _run_yolo(frame: np.ndarray) -> list:
    h, w         = frame.shape[:2]
    tiles_coords = _generate_tiles(w, h)
    tiles        = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles_coords]
    raw: list    = []
    for result, (x0, y0, _, _) in zip(
        _get_model()(tiles, imgsz=TILE_SIZE, conf=CONF_THRESH, verbose=False, augment=True),
        tiles_coords,
    ):
        if result.boxes is None:
            continue
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            gx1, gy1, gx2, gy2 = box[0]+x0, box[1]+y0, box[2]+x0, box[3]+y0
            if _passes_shape(gx1, gy1, gx2, gy2):
                raw.append([gx1, gy1, gx2, gy2, float(conf), int(cls)])
    return _nms(raw, NMS_IOU)


def _cap_to_num_robots(dets: list) -> list:
    if len(dets) <= NUM_ROBOTS:
        return dets
    return sorted(dets, key=lambda d: d[4], reverse=True)[:NUM_ROBOTS]


# ---------------------------------------------------------------------------
# Appearance store + gating
# ---------------------------------------------------------------------------

def _safe_crop(frame: np.ndarray, box: Tuple) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _make_appearance(frame: np.ndarray, box: Tuple) -> Optional[dict]:
    crop = _safe_crop(frame, box)
    if crop is None:
        return None
    patch = cv2.resize(crop, APPEARANCE_PATCH_SIZE, interpolation=cv2.INTER_LINEAR)
    hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {"patch": patch, "hist": hist.flatten(), "gray": gray}


def _score_appearance(ref: dict, frame: np.ndarray, box: Tuple) -> float:
    """Return similarity in [0,1] between stored appearance ref and box in frame."""
    crop = _safe_crop(frame, box)
    if crop is None:
        return 0.0
    patch    = cv2.resize(crop, APPEARANCE_PATCH_SIZE, interpolation=cv2.INTER_LINEAR)
    hsv      = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist     = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    hist_corr = float(cv2.compareHist(
        ref["hist"].reshape(-1, 1).astype(np.float32),
        hist.flatten().reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))
    hist_score = (hist_corr + 1.0) / 2.0   # [-1,1] → [0,1]

    gray   = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = cv2.matchTemplate(ref["gray"], gray, cv2.TM_CCOEFF_NORMED)
    tmpl   = float(np.clip(result[0, 0], 0.0, 1.0))

    return 0.6 * hist_score + 0.4 * tmpl


# Per-slot appearance templates, updated whenever YOLO fires on that slot
_appearance: List[Optional[dict]] = [None] * NUM_ROBOTS
_appearance_lock = threading.Lock()


def update_appearance(slot_idx: int, frame: np.ndarray, box: Tuple) -> None:
    """Refresh the appearance template for *slot_idx*."""
    app = _make_appearance(frame, box)
    if app is None:
        return
    with _appearance_lock:
        _appearance[slot_idx] = app


def score_detection_vs_slot(slot_idx: int, frame: np.ndarray, box: Tuple) -> float:
    """Return appearance similarity [0,1]; 1.0 if slot has no template yet."""
    with _appearance_lock:
        ref = _appearance[slot_idx]
    if ref is None:
        return 1.0
    return _score_appearance(ref, frame, box)


# ---------------------------------------------------------------------------
# Optical-flow supplement — heavy weight, feeds per-slot Kalman filters
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_USE_GPU = _cuda_available()

_of_states:  List[Optional[dict]]            = [None] * NUM_ROBOTS
_of_kalman:  List[Optional[cv2.KalmanFilter]] = [None] * NUM_ROBOTS
_of_lock     = threading.Lock()


def _make_of_kalman(cx: int, cy: int) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * OF_MEASUREMENT_NOISE
    kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
    kf.statePost           = np.array([[cx],[cy],[0.],[0.]], dtype=np.float32)
    return kf


def _init_of_state(gray: np.ndarray, box: Tuple) -> Optional[dict]:
    x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(gray.shape[1],x2), min(gray.shape[0],y2)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.15,
                                  minDistance=4, blockSize=5)
    if pts is None or len(pts) == 0:
        return None
    pts = pts.astype(np.float32)
    pts[:,0,0] += x1
    pts[:,0,1] += y1
    return {"pts": pts, "prev_gray": gray,
            "cx": (x1+x2)//2, "cy": (y1+y2)//2}


def _step_of(state: dict, gray: np.ndarray) -> Tuple[bool, int, int]:
    pts, prev = state["pts"], state["prev_gray"]
    if _USE_GPU:
        try:
            lk = cv2.cuda.SparsePyrLKOpticalFlow_create(winSize=(21,21), maxLevel=3)
            pg=cv2.cuda_GpuMat(); pg.upload(prev)
            cg=cv2.cuda_GpuMat(); cg.upload(gray)
            tg=cv2.cuda_GpuMat(); tg.upload(pts)
            ng,sg,_ = lk.calc(pg,cg,tg,None)
            npts=ng.download(); status=sg.download()
        except Exception:
            npts,status,_ = cv2.calcOpticalFlowPyrLK(prev,gray,pts,None)
    else:
        npts,status,_ = cv2.calcOpticalFlowPyrLK(prev,gray,pts,None)

    if npts is None or status is None:
        return False, state["cx"], state["cy"]
    good = npts[status.flatten()==1].reshape(-1,2)
    if len(good) < 4:
        return False, state["cx"], state["cy"]
    cx = int(np.median(good[:,0]))
    cy = int(np.median(good[:,1]))
    state["pts"]       = good.reshape(-1,1,2)
    state["prev_gray"] = gray
    state["cx"] = cx; state["cy"] = cy
    return True, cx, cy


def of_update_slots(frame: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Run one OF+Kalman step for all active slots.

    Returns {slot_idx: (cx, cy)} for slots where OF succeeded.
    Should be called every frame by the tracker (not just on YOLO frames).
    The Kalman here uses OF_MEASUREMENT_NOISE which is much tighter than
    YOLO_MEASUREMENT_NOISE, giving optical-flow heavy authority over position.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results: Dict[int, Tuple[int, int]] = {}
    with _of_lock:
        for i in range(NUM_ROBOTS):
            state = _of_states[i]
            kf    = _of_kalman[i]
            if state is None or kf is None:
                continue
            ok, cx, cy = _step_of(state, gray)
            if not ok:
                continue
            kf.predict()
            kf.correct(np.array([[cx],[cy]], dtype=np.float32))
            sp = kf.statePost
            results[i] = (int(sp[0]), int(sp[1]))
    return results


def reinit_slot_of(slot_idx: int, frame: np.ndarray, box: Tuple) -> None:
    """Re-seed the OF + Kalman state for *slot_idx* from a confirmed YOLO box."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    state = _init_of_state(gray, box)
    cx    = (int(box[0]) + int(box[2])) // 2
    cy    = (int(box[1]) + int(box[3])) // 2
    kf    = _make_of_kalman(cx, cy)
    with _of_lock:
        _of_states[slot_idx]  = state
        _of_kalman[slot_idx]  = kf


# ---------------------------------------------------------------------------
# Background YOLO worker thread
# ---------------------------------------------------------------------------

_pending_frame: Optional[np.ndarray] = None
_pending_lock   = threading.Lock()

_latest_result: list = []
_result_lock    = threading.Lock()

_worker_thread: Optional[threading.Thread] = None
_stop_event     = threading.Event()

_yolo_latency_ms: float = 0.0
_yolo_latency_lock      = threading.Lock()
_YOLO_EMA_ALPHA: float  = 0.2

_dispatch_seq: int = 0
_result_seq:   int = 0
_result_ready       = threading.Event()


def get_yolo_latency_ms() -> float:
    with _yolo_latency_lock:
        return _yolo_latency_ms


def _worker_loop() -> None:
    global _pending_frame, _latest_result, _yolo_latency_ms, _result_seq
    while not _stop_event.is_set():
        with _pending_lock:
            frame = _pending_frame
        if frame is None:
            threading.Event().wait(timeout=0.001)
            continue

        t0     = time.perf_counter()
        raw    = _run_yolo(frame)
        result = _cap_to_num_robots(raw)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        with _yolo_latency_lock:
            _yolo_latency_ms = (
                _YOLO_EMA_ALPHA * elapsed_ms + (1.0-_YOLO_EMA_ALPHA)*_yolo_latency_ms
            ) if _yolo_latency_ms else elapsed_ms

        with _result_lock:
            _latest_result = result
            _result_seq   += 1
        _result_ready.set()

        with _pending_lock:
            if _pending_frame is frame:
                _pending_frame = None


def _ensure_worker() -> None:
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True,
                                          name="robot-detector")
        _worker_thread.start()


def stop_worker() -> None:
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(frame: np.ndarray, max_stale_frames: int = 0) -> list:
    """Return up to NUM_ROBOTS (6) YOLO detections — [x1,y1,x2,y2,conf,cls].

    Call of_update_slots() every frame for optical-flow position updates.
    Call score_detection_vs_slot() + reinit_slot_of() + update_appearance()
    from robot_tracker after each Hungarian assignment.
    """
    global _pending_frame, _dispatch_seq
    _ensure_worker()

    with _result_lock:
        seq_before = _result_seq

    _dispatch_seq += 1
    with _pending_lock:
        _pending_frame = frame.copy()

    if max_stale_frames > 0:
        if _dispatch_seq - seq_before > max_stale_frames:
            while True:
                _result_ready.wait(timeout=0.005)
                _result_ready.clear()
                with _result_lock:
                    if _result_seq > seq_before:
                        return list(_latest_result)

    with _result_lock:
        return list(_latest_result)