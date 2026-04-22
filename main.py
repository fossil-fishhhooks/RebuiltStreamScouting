import cv2
import numpy as np
from collections import defaultdict
import time
import argparse
import os


SKIP_SECONDS    = 4#62
MAX_DIST        = 60

# Regions are tuned against this source resolution and scaled per input video.
REFERENCE_FRAME_SIZE = (1366, 768)
CROP_REF             = (0, 90, 1356, 491)
CROP_REF_SIZE        = (CROP_REF[2] - CROP_REF[0], CROP_REF[3] - CROP_REF[1])
HOLE_REF             = (445, 0, 1356 - 445, 491)
MAX_TRAIL    = 38
TRAIL_DECAY  = 0.95

SCORE_POLYGON_REF_BY_SIDE = {
    "red": [
        (342, 136),  # top-left
        (385, 136),  # top-right
        (405, 165),  # right
        (385, 195),  # bottom-right
        (342, 195),  # bottom-left
        (322, 165),  # left
    ],
    "blue": [
        (974, 126),  # top-left
        (1022, 126), # top-right
        (1046, 155), # right
        (1022, 185), # bottom-right
        (974, 185),  # bottom-left
        (950, 155),  # left
    ],
}

ACTIVE_REGION_REF_BY_SIDE = {
    "red": (0, 0, CROP_REF_SIZE[0] // 2, CROP_REF_SIZE[1]),
    "blue": (CROP_REF_SIZE[0] // 2, 0, CROP_REF_SIZE[0], CROP_REF_SIZE[1]),
}

PARABOLA_MIN_POINTS      = 8
PARABOLA_A_MAX           = 0.003 # kind of a min tho, graphics coords rip 0.003
PARABOLA_R2_MIN          = 0.50
SCORE_COOLDOWN_FRAMES    = 0    # global: min frames between any two scores. Not makeing much sense for volleys
ID_SCORE_COOLDOWN_FRAMES = 10   # per-ID: once scored, can't score again for this long

GHOST_FRAMES = 4 # object permanence
FRAME_SKIP   = 2




# -----------------------------
# KALMAN TRACKER
# -----------------------------
def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


class Track:
    def __init__(self, track_id, x, y):
        self.id          = track_id
        self.kf          = make_kalman()
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.ghost_count = 0

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def update(self, x, y):
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.ghost_count = 0

    def position(self):
        s = self.kf.statePost
        return int(s[0]), int(s[1])


class Tracker:
    def __init__(self):
        self.tracks  = {}
        self.next_id = 0

    def update(self, detections):
        predictions = {tid: t.predict() for tid, t in self.tracks.items()}

        pairs = []
        for di, (dx, dy) in enumerate(detections):
            best_id, best_dist = None, MAX_DIST
            for tid, (px, py) in predictions.items():
                dist = ((px - dx)**2 + (py - dy)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id   = tid
            if best_id is not None:
                pairs.append((di, best_id))

        # Resolve conflicts: keep closest match per track
        used_tracks = {}
        for di, tid in pairs:
            if tid in used_tracks:
                prev_di      = used_tracks[tid]
                px, py       = predictions[tid]
                dx1, dy1     = detections[prev_di]
                dx2, dy2     = detections[di]
                if ((px-dx2)**2+(py-dy2)**2) < ((px-dx1)**2+(py-dy1)**2):
                    used_tracks[tid] = di
            else:
                used_tracks[tid] = di

        matched_det_indices = set(used_tracks.values())
        matched_track_ids   = set()

        for tid, di in used_tracks.items():
            dx, dy = detections[di]
            self.tracks[tid].update(dx, dy)
            matched_track_ids.add(tid)

        # Unmatched detections -> new tracks
        for di, (dx, dy) in enumerate(detections):
            if di not in matched_det_indices:
                self.tracks[self.next_id] = Track(self.next_id, dx, dy)
                self.next_id += 1

        # Unmatched tracks -> ghost or remove
        for tid, track in list(self.tracks.items()):
            if tid not in matched_track_ids:
                track.ghost_count += 1
                if track.ghost_count > GHOST_FRAMES:
                    del self.tracks[tid]

        return {tid: t.position() for tid, t in self.tracks.items()}


# -----------------------------
# HELPERS
# -----------------------------
def in_region(point, region):
    x, y = point
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 <= y <= y2

def scale_polygon(points, sx, sy):
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in points]


def polygon_bounds(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def point_in_polygon(point, polygon):
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def scale_region(region, sx, sy):
    x1, y1, x2, y2 = region
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def clamp_region_for_slice(region, frame_w, frame_h):
    x1, y1, x2, y2 = region
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return x1, y1, x2, y2


def clamp_region(region, frame_w, frame_h):
    x1, y1, x2, y2 = region
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1, min(x2, frame_w - 1))
    y2 = max(y1, min(y2, frame_h - 1))
    return x1, y1, x2, y2


def get_runtime_regions(frame_w, frame_h, side):
    ref_w, ref_h = REFERENCE_FRAME_SIZE
    sx = frame_w / ref_w
    sy = frame_h / ref_h

    crop_region = clamp_region_for_slice(scale_region(CROP_REF, sx, sy), frame_w, frame_h)
    crop_w      = crop_region[2] - crop_region[0]
    crop_h      = crop_region[3] - crop_region[1]

    crop_ref_w, crop_ref_h = CROP_REF_SIZE
    crop_sx = crop_w / crop_ref_w
    crop_sy = crop_h / crop_ref_h

    hole_region = clamp_region(scale_region(HOLE_REF, crop_sx, crop_sy), crop_w, crop_h)

    score_polygon_ref = SCORE_POLYGON_REF_BY_SIDE[side]
    score_polygon = [
        clamp_region_for_slice((x, y, x + 1, y + 1), crop_w, crop_h)[:2]
        for x, y in scale_polygon(score_polygon_ref, crop_sx, crop_sy)
    ]

    active_ref = ACTIVE_REGION_REF_BY_SIDE[side]
    active_region = clamp_region(scale_region(active_ref, crop_sx, crop_sy), crop_w, crop_h)

    return crop_region, hole_region, score_polygon, active_region


def blackout_hole(frame, hole_region):
    x1, y1, x2, y2 = hole_region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return frame


def blackout_outside_active(frame, active_region):
    x1, y1, x2, y2 = active_region
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, max(0, y1 - 1)), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, min(h - 1, y2 + 1)), (w - 1, h - 1), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, y1), (max(0, x1 - 1), y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (min(w - 1, x2 + 1), y1), (w - 1, y2), (0, 0, 0), -1)
    return frame

def is_approximately_yellow(point, frame):
    x, y = point
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    bgr = frame[y, x]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h_val, s_val, v_val = hsv
    return 20 <= h_val <= 40 and s_val >= 120 and v_val >= 120



# PARABOLC CHECK
def fit_parabola(points):
    if len(points) < PARABOLA_MIN_POINTS:
        return None
    xs      = np.array([p[0] for p in points], dtype=float)
    ys      = np.array([p[1] for p in points], dtype=float)
    xs_norm = xs - xs.mean()
    A       = np.column_stack([xs_norm**2, xs_norm, np.ones_like(xs_norm)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = coeffs
    y_pred  = A @ coeffs
    ss_res  = np.sum((ys - y_pred)**2)
    ss_tot  = np.sum((ys - ys.mean())**2)
    if ss_tot > 1e-6:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    return a, b, c, r2


def check_parabola_score(oid, pts, frame_idx, last_score_frame_per_id, last_score_frame, score_polygon):
    if len(pts) < 2:
        return False

    in_score           = point_in_polygon(pts[-1], score_polygon)
    id_cooldown_ok     = (frame_idx - last_score_frame_per_id[oid]) > ID_SCORE_COOLDOWN_FRAMES
    global_cooldown_ok = (frame_idx - last_score_frame) > SCORE_COOLDOWN_FRAMES

    if not in_score or not id_cooldown_ok or not global_cooldown_ok:
        return False

    prev_x, prev_y = pts[-2]
    cur_x, cur_y   = pts[-1]
    x1, y1, x2, y2 = polygon_bounds(score_polygon)

    enters_bucket = prev_y < y1 <= cur_y and x1 <= cur_x <= x2
    falling_down  = cur_y > prev_y

    if not (enters_bucket and falling_down):
        return False

    return True


# -----------------------------
# DETECTION
# -----------------------------
def detect_circles(frame, hole_region, active_region):
    ax1, ay1, ax2, ay2 = active_region
    roi = frame[ay1:ay2 + 1, ax1:ax2 + 1]
    if roi.size == 0:
        return []

    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (20, 120, 120), (40, 255, 255)) |
        cv2.inRange(hsv, (90, 120,  80), (130, 255, 255)) |
        cv2.inRange(hsv, (0,  120,  80), (10,  255, 255)) |
        cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    )
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray     = cv2.GaussianBlur(gray, (9, 9), 2)
    edges    = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(edges, mask)
    blurred  = cv2.GaussianBlur(combined, (9, 9), 2)
    circles  = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=10,
        param1=50, param2=5,
        minRadius=5, maxRadius=10
    )
    detections = []
    if circles is not None:
        for (x, y, r) in np.round(circles[0, :]).astype("int"):
            gx, gy = x + ax1, y + ay1
            if not in_region((gx, gy), hole_region) and is_approximately_yellow((gx, gy), frame):
                detections.append((gx, gy, r))
    return detections



def crop_frame(frame, crop_region):
    x1, y1, x2, y2 = crop_region
    return frame[y1:y2, x1:x2]



def run(video_path, side):
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("Could not read input video dimensions")

    crop_region, hole_region, score_polygon, active_region = get_runtime_regions(frame_w, frame_h, side)

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(SKIP_SECONDS * fps)

    print(
        f"[regions] src={frame_w}x{frame_h} crop={crop_region} "
        f"hole={hole_region} score={score_polygon} active={active_region}"
    )

    tracker  = Tracker()
    trails   = defaultdict(list)
    t_alphas = defaultdict(list)

    score                   = 0
    last_score_frame        = -SCORE_COOLDOWN_FRAMES
    last_score_frame_per_id = defaultdict(lambda: -ID_SCORE_COOLDOWN_FRAMES)

    fps_window      = 30
    frame_times     = []
    display_fps     = 0.0
    last_fps_update = time.time()

    frame_idx = 0

    while True:
        for _ in range(FRAME_SKIP - 1):
            cap.grab()
            frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < skip_frames:
            continue

        t_start = time.perf_counter()

        frame   = crop_frame(frame, crop_region)
        circles = detect_circles(frame, hole_region, active_region)
        objects = tracker.update([(x, y) for (x, y, r) in circles])

        # Update trails — only for non-ghost tracks
        for oid, (x, y) in objects.items():
            track = tracker.tracks.get(oid)
            if track and track.ghost_count == 0:
                trails[oid].append((x, y))
                t_alphas[oid].append(1.0)
                if len(trails[oid]) > MAX_TRAIL:
                    trails[oid].pop(0)
                    t_alphas[oid].pop(0)

        # Decay alphas, prune dead trails
        for oid in list(t_alphas.keys()):
            t_alphas[oid] = [a * TRAIL_DECAY for a in t_alphas[oid]]
            combined = [(p, a) for p, a in zip(trails[oid], t_alphas[oid]) if a > 0.05]
            if combined:
                pts, alps     = zip(*combined)
                trails[oid]   = list(pts)
                t_alphas[oid] = list(alps)
            else:
                trails.pop(oid, None)
                t_alphas.pop(oid, None)

        # Scoring
        for oid, pts in trails.items():
            if check_parabola_score(
                oid, pts, frame_idx, last_score_frame_per_id, last_score_frame, score_polygon
            ):
                score += 1
                last_score_frame_per_id[oid] = frame_idx
                last_score_frame             = frame_idx
                print(f"  [SCORE]  ID {oid} @ frame {frame_idx} -> total: {score}")

        # ----- Visuals -----
        vis = frame.copy()
        vis = blackout_outside_active(vis, active_region)
        vis = blackout_hole(vis, hole_region)

        cv2.polylines(vis, [np.array(score_polygon, dtype=np.int32)], True, (0, 255, 0), 2)

        # Detected circles
        for (x, y, r) in circles:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        # Trails
        for oid in trails:
            pts  = trails[oid]
            alps = t_alphas[oid]
            for i in range(1, len(pts)):
                a     = alps[i - 1]
                color = (0, int(255 * a), int(255 * (1 - a)))
                cv2.line(vis, pts[i - 1], pts[i], color, 2)

        # Parabola preview
        for oid, pts in trails.items():
            if len(pts) >= PARABOLA_MIN_POINTS:
                result = fit_parabola(pts)
                if result:
                    a, b, c, r2 = result
                    xs     = np.array([p[0] for p in pts])
                    x_mean = xs.mean()
                    col    = (0, 200, 255) if r2 > PARABOLA_R2_MIN else (80, 80, 80)
                    for xi in range(int(xs.min()), int(xs.max()), 2):
                        xn  = xi - x_mean
                        yi  = int(a * xn**2 + b * xn + c)
                        xn2 = xi + 2 - x_mean
                        yi2 = int(a * xn2**2 + b * xn2 + c)
                        h, w = vis.shape[:2]
                        if 0 <= yi < h and 0 <= yi2 < h:
                            cv2.line(vis, (xi, yi), (xi + 2, yi2), col, 1)

        # Ghost tracks
        for tid, track in tracker.tracks.items():
            if track.ghost_count > 0:
                px, py = track.position()
                alpha  = 1.0 - track.ghost_count / GHOST_FRAMES
                color  = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                cv2.circle(vis, (px, py), 6, color, 1)

        # Object labels
        for oid, (x, y) in objects.items():
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(oid), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > fps_window:
            frame_times.pop(0)
        now = time.time()
        if now - last_fps_update > 0.5:
            display_fps     = 1.0 / (sum(frame_times) / len(frame_times))
            last_fps_update = now

        cv2.putText(vis, f"Score: {score}", (460, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"{display_fps:.1f}/{(60/FRAME_SKIP):.0f} fps  tracks: {len(tracker.tracks)}",
                    (460, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.imshow("tracking", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track")
    parser.add_argument("--side", type=str, choices=["red", "blue"], default="red")
    parser.add_argument("--frame-drop", type=int)
    parser.add_argument("--video-file", type=str)
    args=parser.parse_args()

    cv2.setNumThreads(os.cpu_count() or 1)

    if args.frame_drop is not None:
        FRAME_SKIP = args.frame_drop

    
        
    print("Initializing...")
    sc = run(args.video_file, args.side)
    print(f"Final score: {sc}")
