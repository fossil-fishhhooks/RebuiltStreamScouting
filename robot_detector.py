

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH         = "YOLOv8n-6k-variated.pt"
TILE_SIZE          = 640
OVERLAP            = 0.47
CONF_THRESH        = 0.4
MAX_BOX_AREA       = 480 * 480 * 0.06
ASPECT_AREA_THRESH = MAX_BOX_AREA * 0.25
ASPECT_MAX_LOOSE   = 2.5
ASPECT_MAX_TIGHT   = 1.35
NMS_IOU            = 0.38          # single threshold; tune or expose both if needed

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



#aspect ratio
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




def detect(frame: np.ndarray) -> list:

    #Run tiled inference on frame
    #Returns a list of [x1, y1, x2, y2, conf, cls] after NMS.

    h, w = frame.shape[:2]
    tiles_coords = _generate_tiles(w, h)
    tiles = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles_coords]

    raw = []
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
            gx1, gy1, gx2, gy2 = box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0
            if _passes_shape(gx1, gy1, gx2, gy2):
                raw.append([gx1, gy1, gx2, gy2, float(conf), int(cls)])

    return _nms(raw, NMS_IOU)