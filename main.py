import cv2
import numpy as np
from collections import defaultdict
import time
import argparse


SKIP_SECONDS    = 62
MAX_DIST        = 60

CROP         = (0, 90, 1356, 491)
HOLE         = (445, 0, 1356 - 445, 491)
MAX_TRAIL    = 38
TRAIL_DECAY  = 0.95

SCORE_REGION = (332, 148, 395, 190)

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

def blackout_hole(frame):
    x1, y1, x2, y2 = HOLE
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
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
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0
    return a, b, c, r2


def check_parabola_score(oid, pts, frame_idx, last_score_frame_per_id, last_score_frame):
    if len(pts) < PARABOLA_MIN_POINTS:
        return False

    in_score           = in_region(pts[-1], SCORE_REGION)
    id_cooldown_ok     = (frame_idx - last_score_frame_per_id[oid]) > ID_SCORE_COOLDOWN_FRAMES
    global_cooldown_ok = (frame_idx - last_score_frame) > SCORE_COOLDOWN_FRAMES

    if not in_score or not id_cooldown_ok or not global_cooldown_ok:
        return False

    result = fit_parabola(pts)
    if result is None:
        print(f"  [REJECT] ID {oid}: fit failed")
        return False

    a, b, c, r2 = result

    mid          = len(pts) // 2
    y_early      = np.mean([p[1] for p in pts[:mid]])
    y_late       = np.mean([p[1] for p in pts[mid:]])
    opens_down   = a > PARABOLA_A_MAX
    good_fit     = r2 > PARABOLA_R2_MIN
    falling_down = y_late > y_early

    if not (opens_down and good_fit and falling_down):
        print(
            f"  [REJECT] ID {oid} @ frame {frame_idx}: "
            f"a={a:.5f} (need > {PARABOLA_A_MAX}), "
            f"Rsquare={r2:.3f} (need > {PARABOLA_R2_MIN}), "
            f"y_early={y_early:.1f} y_late={y_late:.1f} "
            f"falling_down={falling_down}, opens_down={opens_down}, good_fit={good_fit}"
        )
        return False

    return True


# -----------------------------
# DETECTION
# -----------------------------
def detect_circles(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (20, 120, 120), (40, 255, 255)) |
        cv2.inRange(hsv, (90, 120,  80), (130, 255, 255)) |
        cv2.inRange(hsv, (0,  120,  80), (10,  255, 255)) |
        cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    )
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            if not in_region((x, y), HOLE) and is_approximately_yellow((x, y), frame):
                detections.append((x, y, r))
    return detections



def crop_frame(frame):
    x1, y1, x2, y2 = CROP
    return frame[y1:y2, x1:x2]



def run(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(SKIP_SECONDS * fps)

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

        frame   = crop_frame(frame)
        circles = detect_circles(frame)
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
            if check_parabola_score(oid, pts, frame_idx, last_score_frame_per_id, last_score_frame):
                score += 1
                last_score_frame_per_id[oid] = frame_idx
                last_score_frame             = frame_idx
                print(f"  [SCORE]  ID {oid} @ frame {frame_idx} -> total: {score}")

        # ----- Visuals -----
        vis = frame.copy()
        vis = blackout_hole(vis)

        sx1, sy1, sx2, sy2 = SCORE_REGION
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

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
    parser.add_argument("--side", type=str)
    parser.add_argument("--frame-drop", type=int)
    parser.add_argument("--video-file", type=str)
    args=parser.parse_args()
    if (args.side == "red"):
        SCORE_REGION = (332, 148, 395, 190)
    elif (args.side == "blue"):
        SCORE_REGION = (953, 148, 1016, 190)

    if args.frame_drop != None:
        FRAME_SKIP = args.frame_drop

    
        
    print("Initializing...")
    sc = run(args.video_file)
    print(f"Final score: {sc}")
