import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

MODEL_PATH    = "best.pt"
VIDEO_PATH    = "Q18.mp4"
OUTPUT_PATH   = "test.avi"
TILE_SIZE     = 640                    # training res
OVERLAP       = 0.45                   # fraction of tile_size that adjacent tiles share
CONF_THRESH   = 0.15                   # minimum confidence for a detection
MAX_BOX_AREA  = 640 * 640 * 0.06      # max allowed box area in pixels
DISPLAY_SCALE = 0.5                    # resize factor for the preview window only

SAVE_OUT      = True

model = YOLO(MODEL_PATH)




def generate_tiles(w, h):
    stride = int(TILE_SIZE * (1.0 - OVERLAP)) # how many pixels to step between tile origins 640*(1-overlap)
    xs = list(range(0, w - TILE_SIZE, stride))
    if xs[-1] != w - TILE_SIZE:
        xs.append(w - TILE_SIZE) #rightmost tile is forced
    ys = list(range(0, h - TILE_SIZE, stride))
    if ys[-1] != h - TILE_SIZE:
        ys.append(h - TILE_SIZE) #rightmost tile is forced
    return [(x, y, x + TILE_SIZE, y + TILE_SIZE) for y in ys for x in xs] # list of (x0,y0,x1,y1) rects


def nms(dets, iou_thresh):
    if not dets: 
        return []
    d = np.array(dets, dtype=np.float32) # each row is [x1,y1,x2,y2,conf,cls]
    order = np.argsort(-d[:, 4]) # indices sorted by confidence descending. INDCIES NOT VALUES
    keep = []
    while order.size:
        i = order[0]
        keep.append(i) #index of best box
        order = order[1:] #remove from order queue
        if not order.size: break # nothing left
        
        ix1 = np.maximum(d[i,0], d[order,0])
        iy1 = np.maximum(d[i,1], d[order,1])   # least top-left of the top left corners
        ix2 = np.minimum(d[i,2], d[order,2])
        iy2 = np.minimum(d[i,3], d[order,3])   # same for bottom rights
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)  #intersect area, or zero
        union = (d[i,2]-d[i,0])*(d[i,3]-d[i,1]) + (d[order,2]-d[order,0])*(d[order,3]-d[order,1]) - inter  # total area (pigeonhole)
        order = order[np.where(union > 0, inter/union, 0) < iou_thresh] # drop any box whose overlap with box i exceeds the threshold
    return d[keep].tolist()  # return real dets in a list


def infer(frame):
    h, w = frame.shape[:2]   # frame dim
    tiles_coords = generate_tiles(w, h)   # list of (x0,y0,x1,y1) covering the full frame
    tiles = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles_coords] # slice pixel data for each tile

    raw = []
    for result, (x0, y0, _, _) in zip(
        model(tiles, imgsz=TILE_SIZE, conf=CONF_THRESH, verbose=False), tiles_coords   # single batched GPU call
    ):
        if result.boxes is None: continue # tile had zero detections
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(), # box coords in tile pixels [x1,y1,x2,y2]
            result.boxes.conf.cpu().numpy(), # confidence score 0-1
            result.boxes.cls.cpu().numpy()  # class index
        ):
            gx1,gy1,gx2,gy2 = box[0]+x0, box[1]+y0, box[2]+x0, box[3]+y0 # coord to full fraame
            if (gx2-gx1) * (gy2-gy1) > MAX_BOX_AREA: continue  # reject oversized boxes
            raw.append([gx1, gy1, gx2, gy2, conf, int(cls)]) # accumulate all surviving detections across all tiles

    return raw, nms(raw, 0.60), nms(raw, 0.45), tiles_coords # raw = all detections; nms50 = loose merge; nms35 = tight merge; coords for tile debug drawing


def draw_boxes(frame, dets, color, thickness=1):
    for x1, y1, x2, y2, conf, cls in dets:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness) # draw box in-place on frame


cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if SAVE_OUT:
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

with tqdm(total=total) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        raw, nms50, nms35, tiles_coords = infer(frame)

        for i, (x0, y0, x1, y1) in enumerate(tiles_coords):
            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          [(0,0,255),(0,255,0),(255,0,0),(0,255,255)][i%4], 1)

        draw_boxes(frame, raw,   (0, 0, 255), 1)
        draw_boxes(frame, nms50, (255, 0, 0), 2)
        draw_boxes(frame, nms35, (0, 255, 0), 2)

        #
        if SAVE_OUT:
            writer.write(frame)
        else:
            cv2.imshow("tiled", cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
            if cv2.waitKey(1) & 0xFF == 27: break                                          # ESC to quit early
        pbar.update(1)

cap.release()
if SAVE_OUT:
    writer.release()
cv2.destroyAllWindows()
