import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO("best.pt")

video_path = "Q18.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tile_size = 640
overlap = 0.2 ## tune for solid detect of stream size frames
stride = int(tile_size * (1 - overlap))

display_scale = 0.5


COLORS = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 128, 255),    # orange
    (255, 0, 128),    # pink
]

def tile_color(ix, iy):
    return COLORS[(ix + iy) % len(COLORS)]

def generate_tiles(w, h, tile_size=640, overlap=0.2):
    stride = int(tile_size * (1 - overlap))

    xs = list(range(0, w, stride))
    ys = list(range(0, h, stride))

    tiles = set()

    for y in ys:
        for x in xs:
            # shift tile if it would go out of frame
            x0 = min(x, w - tile_size)
            y0 = min(y, h - tile_size)

            # ensure valid (in case frame smaller than tile)
            if x0 < 0 or y0 < 0:
                continue

            tiles.add((x0, y0))

    return list(tiles)

def run_tiled_inference(frame):
    h, w, _ = frame.shape
    detections = []

    tiles = generate_tiles(w, h, tile_size=640, overlap=0.2)

    for i, (x, y) in enumerate(tiles):
        tile = frame[y:y+640, x:x+640]

        color = COLORS[i % len(COLORS)]

        # draw tile
        cv2.rectangle(frame, (x, y), (x+640, y+640), color, 2)
        cv2.line(frame, (x, y), (x+640, y+640), color, 2)
        cv2.line(frame, (x+640, y), (x, y+640), color, 2)

        results = model(tile, imgsz=640, conf=0.25, verbose=False)[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                X1 = x1 + x
                Y1 = y1 + y
                X2 = x2 + x
                Y2 = y2 + y

                area = (X2 - X1) * (Y2 - Y1)
                detections.append([X1, Y1, X2, Y2, conf, area])

    return detections

with tqdm(total=total_frames, desc="Processing video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dets = run_tiled_inference(frame)

        # draw
        for x1, y1, x2, y2, conf, area in dets:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            label = f"{conf:.2f} | {int(area)} px"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0,255,0),
                1,
                cv2.LINE_AA
            )

        display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

        cv2.imshow("tiled", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        pbar.update(1)

cap.release()
cv2.destroyAllWindows()