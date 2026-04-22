import cv2
import numpy as np

from config import (
    ACTIVE_REGION_REF_BY_SIDE,
    BOUNCE_OUT_RISE,
    CROP_REF,
    CROP_REF_SIZE,
    ID_SCORE_COOLDOWN_FRAMES,
    HOLE_REF,
    PARABOLA_MIN_POINTS,
    SCORE_COOLDOWN_FRAMES,
    SCORE_MIN_DESCENT,
    SCORE_MIN_INSIDE_POINTS,
    SCORE_POLYGON_REF_BY_SIDE,
    SCORE_TRAIL_WINDOW,
)


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


def _orientation(a, b, c):
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
        min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True

    return False


def trail_hits_polygon(points, polygon):
    if any(point_in_polygon(point, polygon) for point in points):
        return True

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        for j in range(len(polygon)):
            q1 = polygon[j]
            q2 = polygon[(j + 1) % len(polygon)]
            if segments_intersect(p1, p2, q1, q2):
                return True

    return False


def trail_bounced_out_of_polygon(points, polygon):
    recent_pts = points[-SCORE_TRAIL_WINDOW:]
    inside_flags = [point_in_polygon(point, polygon) for point in recent_pts]

    if not any(inside_flags) or inside_flags[-1]:
        return False

    first_inside_idx = next((i for i, inside in enumerate(inside_flags) if inside), None)
    if first_inside_idx is None:
        return False

    post_entry_pts = recent_pts[first_inside_idx:]
    if len(post_entry_pts) < 3:
        return False

    deepest_y = max(p[1] for p in post_entry_pts)
    latest_y = post_entry_pts[-1][1]
    return latest_y + BOUNCE_OUT_RISE < deepest_y


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
    ref_w, ref_h = (1366, 768)
    sx = frame_w / ref_w
    sy = frame_h / ref_h

    crop_region = clamp_region_for_slice(scale_region(CROP_REF, sx, sy), frame_w, frame_h)
    crop_w = crop_region[2] - crop_region[0]
    crop_h = crop_region[3] - crop_region[1]

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
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    xs_norm = xs - xs.mean()
    A = np.column_stack([xs_norm**2, xs_norm, np.ones_like(xs_norm)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = coeffs
    y_pred = A @ coeffs
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    if ss_tot > 1e-6:
        r2 = float(1.0 - np.divide(ss_res, ss_tot))
    else:
        r2 = 0.0
    return a, b, c, r2


def check_parabola_score(oid, pts, frame_idx, last_score_frame_per_id, last_score_frame, score_polygon, scored_track_ids):
    if oid in scored_track_ids or len(pts) < 2:
        return False

    recent_pts = pts[-SCORE_TRAIL_WINDOW:]
    inside_flags = [point_in_polygon(point, score_polygon) for point in recent_pts]
    id_cooldown_ok = (frame_idx - last_score_frame_per_id[oid]) > ID_SCORE_COOLDOWN_FRAMES
    global_cooldown_ok = (frame_idx - last_score_frame) > SCORE_COOLDOWN_FRAMES
    descending = recent_pts[-1][1] >= recent_pts[max(0, len(recent_pts) - 3)][1] + SCORE_MIN_DESCENT
    hit_bucket = trail_hits_polygon(recent_pts, score_polygon)
    stable_inside = inside_flags[-1] and sum(inside_flags[-SCORE_MIN_INSIDE_POINTS:]) >= SCORE_MIN_INSIDE_POINTS

    if trail_bounced_out_of_polygon(recent_pts, score_polygon):
        return False

    if not hit_bucket or not descending or not stable_inside or not id_cooldown_ok or not global_cooldown_ok:
        return False

    return True


def detect_circles(frame, hole_region, active_region):
    ax1, ay1, ax2, ay2 = active_region
    roi = frame[ay1:ay2 + 1, ax1:ax2 + 1]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (20, 120, 120), (40, 255, 255)) |
        cv2.inRange(hsv, (90, 120, 80), (130, 255, 255)) |
        cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) |
        cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    )
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(edges, mask)
    blurred = cv2.GaussianBlur(combined, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=10,
        param1=50, param2=5,
        minRadius=5, maxRadius=10,
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


