import threading

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH         = "YOLOv8n-6k-variated.pt"
TILE_SIZE          = 640
OVERLAP            = 0.47
CONF_THRESH        = 0.4
MAX_BOX_AREA       = 480 * 480 * 0.06
ASPECT_AREA_THRESH = MAX_BOX_AREA * 0.25
ASPECT_MAX_LOOSE   = 2.5
ASPECT_MAX_TIGHT   = 1.35
NMS_IOU            = 0.38

_model = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def _generate_tiles(w: int, h: int):
    stride = int(TILE_SIZE * (1.0 - OVERLAP))
    xs = list(range(0, w - TILE_SIZE, stride))
    if xs[-1] != w - TILE_SIZE:
        xs.append(w - TILE_SIZE)
    ys = list(range(0, h - TILE_SIZE, stride))
    if ys[-1] != h - TILE_SIZE:
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


def _passes_shape(gx1, gy1, gx2, gy2) -> bool:
    bw, bh = gx2 - gx1, gy2 - gy1
    area = bw * bh
    if area > MAX_BOX_AREA:
        return False
    if area > ASPECT_AREA_THRESH:
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        t = (area - ASPECT_AREA_THRESH) / (MAX_BOX_AREA - ASPECT_AREA_THRESH)
        allowed = ASPECT_MAX_LOOSE + t * (ASPECT_MAX_TIGHT - ASPECT_MAX_LOOSE)
        if aspect > allowed:
            return False
    return True


def _run_inference(frame: np.ndarray) -> list:
    h, w = frame.shape[:2]
    tiles_coords = _generate_tiles(w, h)
    tiles = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles_coords]

    raw = []
    for result, (x0, y0, _, _) in zip(
        _get_model()(tiles, imgsz=TILE_SIZE, conf=CONF_THRESH, verbose=False, augment=False),
        tiles_coords,
    ):
        if result.boxes is None:
            continue
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            gx1, gy1, gx2, gy2 = box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0
            if _passes_shape(gx1, gy1, gx2, gy2):
                raw.append([gx1, gy1, gx2, gy2, float(conf), int(cls)])

    return _nms(raw, NMS_IOU)



# The main thread writes the latest frame here; the worker thread reads it.
_pending_frame: np.ndarray | None = None
_pending_lock  = threading.Lock()

# The worker thread writes results here; the main thread reads them.
_latest_result: list = []
_result_lock   = threading.Lock()

_worker_thread: threading.Thread | None = None
_stop_event    = threading.Event()


def _worker_loop() -> None:
    global _pending_frame, _latest_result
    while not _stop_event.is_set():
        # Atomically grab-and-clear the pending frame.
        with _pending_lock:
            frame = _pending_frame

        if frame is None:
            # Nothing new yet — yield the GIL briefly and retry.
            threading.Event().wait(timeout=0.001)
            continue

        result = _run_inference(frame)

        with _result_lock:
            _latest_result = result

        # Mark the frame as consumed so we don't re-run on the same frame.
        with _pending_lock:
            # Only clear if no newer frame has already been queued.
            if _pending_frame is frame:
                _pending_frame = None


def _ensure_worker() -> None:
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="robot-detector")
        _worker_thread.start()


def stop_worker() -> None:
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=5)




def detect(frame: np.ndarray) -> list:
   #yayayayyayay non blocking
    global _pending_frame
    _ensure_worker()

    # Hand the latest frame to the worker, overwriting any unprocessed one.
    with _pending_lock:
        _pending_frame = frame

    with _result_lock:
        return list(_latest_result)