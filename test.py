import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO("best.pt")

video_path = "Q18.mp4"
cap = cv2.VideoCapture(video_path)

# get total frames for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tile_size = 640
overlap = 0.2
stride = int(tile_size * (1 - overlap))

display_scale = 0.5  # shrink window (0.5 = half size)

def run_tiled_inference(frame):
    h, w, _ = frame.shape
    detections = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)

            tile = frame[y:y_end, x:x_end]

            pad = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile

            results = model(pad, imgsz=640, conf=0.25, verbose=False)[0]

            if results.boxes is not None:
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    detections.append([x1 + x, y1 + y, x2 + x, y2 + y])

    return detections

# progress bar
with tqdm(total=total_frames, desc="Processing video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dets = run_tiled_inference(frame)

        # draw boxes
        for x1, y1, x2, y2 in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        # resize for display
        display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

        cv2.imshow("tiled", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        pbar.update(1)

cap.release()
cv2.destroyAllWindows()