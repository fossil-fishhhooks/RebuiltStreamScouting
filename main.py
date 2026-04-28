import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from config import (
    FRAME_SKIP,
    MAX_TRAIL,
    PARABOLA_MIN_POINTS,
    PARABOLA_FIT_ERROR,
    SCORE_COOLDOWN_FRAMES,
    SKIP_SECONDS,
    TRAIL_DECAY,
    SCORE_POLYGON_REF_BY_SIDE,
)
from tracker import Tracker
from vision import (
    blackout_hole,
    blackout_outside_active,
    check_parabola_score,
    crop_frame,
    detect_apriltags,
    detect_circles,
    fit_conic,
    get_runtime_regions,
    sample_conic_curve,
solve_y
)
from robot_tracker import RobotTracker, detect_robots
from robot_detector import get_yolo_latency_ms


def get_frame_at_index(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def polygon_center(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (np.mean(xs), np.mean(ys))


def adjust_polygon_for_apriltag(video_path, side, frame_240_offset=None):
    if frame_240_offset is None:
        try:
            ref_cap = cv2.VideoCapture("Q18.mp4")
            ref_frame = get_frame_at_index(ref_cap, 240)
            ref_cap.release()

            if ref_frame is not None:
                apriltags = detect_apriltags(ref_frame)
                tag11_ref = [tag for tag in apriltags if tag.get("id") == 11]

                if tag11_ref:
                    ref_frame_h, ref_frame_w = ref_frame.shape[:2]
                    crop_region, _, _, _ = get_runtime_regions(ref_frame_w, ref_frame_h, side)
                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region

                    tag11_center = tag11_ref[0]["center"]
                    tag11_in_crop = (tag11_center[0] - crop_x1, tag11_center[1] - crop_y1)

                    ref_polygon = SCORE_POLYGON_REF_BY_SIDE[side]
                    _, _, ref_score_polygon, _ = get_runtime_regions(ref_frame_w, ref_frame_h, side)
                    runtime_center = polygon_center(ref_score_polygon)

                    frame_240_offset = (
                        runtime_center[0] - tag11_in_crop[0],
                        runtime_center[1] - tag11_in_crop[1]
                    )
                    print(f"[AprilTag] Reference offset from Q18.mp4: {frame_240_offset}")
        except Exception as e:
            print(f"[AprilTag] Could not calculate reference offset: {e}")
            return None

    if frame_240_offset is None:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        frame = get_frame_at_index(cap, 240)
        cap.release()

        if frame is None:
            print("[AprilTag] Could not read frame 240 from video")
            return None

        apriltags = detect_apriltags(frame)
        tag11_detections = [tag for tag in apriltags if tag.get("id") == 11]

        if not tag11_detections:
            print("[AprilTag] AprilTag 11 not found at frame 240")
            return None

        frame_h, frame_w = frame.shape[:2]
        crop_region, hole_region, score_polygon, active_region = get_runtime_regions(frame_w, frame_h, side)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region

        tag11_center = tag11_detections[0]["center"]
        tag11_in_crop = (tag11_center[0] - crop_x1, tag11_center[1] - crop_y1)
        target_polygon_center = (
            tag11_in_crop[0] + frame_240_offset[0],
            tag11_in_crop[1] + frame_240_offset[1]
        )

        current_center = polygon_center(score_polygon)
        translation = (
            target_polygon_center[0] - current_center[0],
            target_polygon_center[1] - current_center[1]
        )

        ref_polygon = SCORE_POLYGON_REF_BY_SIDE[side]
        ref_w, ref_h = 1366, 768
        sx = frame_w / ref_w
        sy = frame_h / ref_h

        adjusted_polygon = [
            (int(x * sx + translation[0]), int(y * sy + translation[1]))
            for x, y in ref_polygon
        ]

        print(f"[AprilTag] Found AprilTag 11 at frame 240 center: {tag11_center}")
        print(f"[AprilTag] Adjusted polygon by translation: ({translation[0]:.1f}, {translation[1]:.1f})")

        return adjusted_polygon

    except Exception as e:
        print(f"[AprilTag] Error adjusting polygon: {e}")
        return None

def run(video_path, side, frame_skip=FRAME_SKIP, max_stale_frames=0):
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

    tracker    = Tracker()
    trails     = defaultdict(list)
    t_alphas   = defaultdict(list)
    full_trails = defaultdict(list)  # unbounded, no decay — used for scored snapshots
    first_seen = {}
    origins    = {}
    fit_cache  = {}  # oid -> (trail_len, params, err, x_min, x_max)

    score                   = 0
    last_score_frame        = -SCORE_COOLDOWN_FRAMES
    last_score_frame_per_id = defaultdict(lambda: -10)
    scored_track_ids        = set()
    scored_curves           = {}   # oid -> (trail_pts, curve_pts), drawn permanently

    fps_window      = 30
    frame_times     = []
    display_fps     = 0.0
    last_fps_update = time.time()
    frame_idx       = 0
    crop_x, crop_y  = crop_region[0], crop_region[1]

    while True:
        for _ in range(max(0, frame_skip - 1)):
            cap.grab()
            frame_idx += 1

        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < skip_frames:
            continue

        t_start = time.perf_counter()

        frame = crop_frame(raw_frame, crop_region)
        t_crop = time.perf_counter()

        circles = detect_circles(frame, hole_region, active_region)
        t_circles = time.perf_counter()

        objects = tracker.update([(x, y) for (x, y, r) in circles])
        t_track = time.perf_counter()

        t_before_robots = time.perf_counter()
        robot_dets_full = detect_robots(raw_frame, alliance="both", max_stale_frames=max_stale_frames)
        robot_dets = [
            (cx - crop_x, cy - crop_y, w, h, alliance,
             x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y)
            for (cx, cy, w, h, alliance, x1, y1, x2, y2) in robot_dets_full
        ]
        t_robots = time.perf_counter()
        yolo_waited = max_stale_frames > 0 and (t_robots - t_before_robots) > 0.002

        live, dormant = robot_tracker.update(robot_dets, crop_frame=frame)
        frame = robot_tracker.draw(frame)
        t_rtrack = time.perf_counter()

        for oid, (x, y) in objects.items():
            track = tracker.tracks.get(oid)
            if track and track.ghost_count == 0:
                if oid not in first_seen:
                    first_seen[oid] = (x, y)
                trails[oid].append((x, y))
                t_alphas[oid].append(1.0)
                full_trails[oid].append((x, y))
                if len(trails[oid]) > MAX_TRAIL:
                    trails[oid].pop(0)
                    t_alphas[oid].pop(0)

        # Re-identification: tracker resurrected a dead ID — full_trail is
        # already on the right key so nothing extra needed; but if a *new*
        # id was about to be created and we merged it back, the trail is
        # already continuous. No-op here; the tracker handles it in-place.

        for oid in list(t_alphas.keys()):
            t_alphas[oid] = [a * TRAIL_DECAY for a in t_alphas[oid]]
            combined = [(p, a) for p, a in zip(trails[oid], t_alphas[oid]) if a > 0.05]
            if combined:
                pts, alps = zip(*combined)
                trails[oid]   = list(pts)
                t_alphas[oid] = list(alps)
            else:
                trails.pop(oid, None)
                t_alphas.pop(oid, None)
                fit_cache.pop(oid, None)

        for oid, pts in trails.items():
            track      = tracker.tracks.get(oid)
            track_lost = bool(track and track.ghost_count == 1)
            if check_parabola_score(
                    oid, pts, frame_idx, last_score_frame_per_id,
                    last_score_frame, score_polygon, scored_track_ids,
                    track_lost=track_lost,
            ):
                score += 1
                last_score_frame_per_id[oid] = frame_idx
                last_score_frame             = frame_idx
                scored_track_ids.add(oid)
                if oid not in origins and oid in first_seen:
                    origins[oid] = first_seen[oid]

                if oid not in scored_curves:
                    trail_snap = list(full_trails[oid])
                    scored_curves[oid] = (trail_snap, [])

                print(f"  [SCORE]  ID {oid} @ frame {frame_idx} -> total: {score}")

        # --- post-score trail stitching ---
        # Try to merge any two scored trails where one's endpoint is close
        # to another's start point, indicating a dropped re-id mid-flight.
        STITCH_DIST = 20   # px — max gap to bridge
        merged = True
        while merged:
            merged = False
            keys = list(scored_curves.keys())
            for i, ka in enumerate(keys):
                if ka not in scored_curves:
                    continue
                trail_a, cpts_a = scored_curves[ka]
                if not trail_a:
                    continue
                end_a = trail_a[-1]
                for kb in keys[i+1:]:
                    if kb not in scored_curves:
                        continue
                    trail_b, cpts_b = scored_curves[kb]
                    if not trail_b:
                        continue
                    start_b = trail_b[0]
                    end_b   = trail_b[-1]
                    start_a = trail_a[0]

                    # Check a→b (end of a connects to start of b)
                    d_ab = ((end_a[0]-start_b[0])**2 + (end_a[1]-start_b[1])**2)**0.5
                    # Check b→a (end of b connects to start of a)
                    d_ba = ((end_b[0]-start_a[0])**2 + (end_b[1]-start_a[1])**2)**0.5

                    if d_ab <= STITCH_DIST or d_ba <= STITCH_DIST:
                        if d_ab <= d_ba:
                            merged_trail = trail_a + trail_b
                        else:
                            merged_trail = trail_b + trail_a
                        # keep the lower id, drop the higher
                        keep, drop = (ka, kb) if ka < kb else (kb, ka)
                        scored_curves[keep] = (merged_trail, [])
                        del scored_curves[drop]
                        # re-number is handled by enumerate at draw time
                        merged = True
                        print(f"  [STITCH] merged scored trails {ka} + {kb}")
                        break
                if merged:
                    break
        t_score = time.perf_counter()

        vis = frame.copy()
        cv2.polylines(vis, [np.array(score_polygon, dtype=np.int32)], True, (0, 255, 0), 2)

        for (x, y, r) in circles:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        for oid in trails:
            pts  = trails[oid]
            alps = t_alphas[oid]
            for i in range(1, len(pts)):
                a     = alps[i - 1]
                color = (0, int(255 * a), int(255 * (1 - a)))
                cv2.line(vis, pts[i - 1], pts[i], color, 2)

        # --- parabola fitting disabled ---
        # vis_h = vis.shape[0]
        # vis_w = vis.shape[1]
        # for oid, pts in trails.items():
        #     if len(pts) >= PARABOLA_MIN_POINTS:
        #         trail_len = len(pts)
        #         if oid not in fit_cache or fit_cache[oid][0] != trail_len:
        #             xs = np.array([p[0] for p in pts])
        #             ys = np.array([p[1] for p in pts])
        #             if max(xs.max() - xs.min(), ys.max() - ys.min()) < 20:
        #                 continue
        #             params, err = fit_conic(xs, ys)
        #             a, b, c, theta = params
        #             curve_pts = sample_conic_curve(params, xs, ys, vis_w, vis_h)
        #             fit_cache[oid] = (trail_len, params, err, float(xs.min()), float(xs.max()), curve_pts)
        #         _, params, err, x_min, x_max, curve_pts = fit_cache[oid]
        #         col = (0, 200, 255) if err < PARABOLA_FIT_ERROR else (80, 80, 80)
        #         if len(curve_pts) >= 5:
        #             cv2.polylines(vis, [np.array(curve_pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)

        for tid, track in tracker.tracks.items():
            if track.ghost_count > 0:
                px, py = track.position()
                alpha  = 1.0 - track.ghost_count / 4
                color  = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                cv2.circle(vis, (px, py), 6, color, 1)

        for oid, (x, y) in objects.items():
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(oid), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        t_draw = time.perf_counter()

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > fps_window:
            frame_times.pop(0)
        now = time.time()
        if now - last_fps_update > 0.5 and frame_times:
            display_fps     = 1.0 / (sum(frame_times) / len(frame_times))
            last_fps_update = now

        print(
            f"[time] crop={t_crop-t_start:.3f} "
            f"circles={t_circles-t_crop:.3f} "
            f"track={t_track-t_circles:.3f} "
            f"robots={t_robots-t_before_robots:.3f}{'*' if yolo_waited else ''} "
            f"rtrack={t_rtrack-t_robots:.3f} "
            f"score={t_score-t_rtrack:.3f} "
            f"draw={t_draw-t_score:.3f} "
            f"total={t_end-t_start:.3f}"
        )

        # Draw permanently retained scored-shot trails and parabolas.
        for shot_idx, (oid, (trail_pts, cpts)) in enumerate(scored_curves.items(), start=1):
            color = (0, 165, 255)  # orange
            # trail
            for i in range(1, len(trail_pts)):
                cv2.line(vis, trail_pts[i - 1], trail_pts[i], color, 2, cv2.LINE_AA)
            # parabola fit — disabled
            # if cpts:
            #     cv2.polylines(vis, [np.array(cpts, dtype=np.int32)], False,
            #                   (255, 255, 255), 1, cv2.LINE_AA)
            # label at trail midpoint
            mid = trail_pts[len(trail_pts) // 2]
            cv2.putText(vis, f"#{shot_idx}", (mid[0] + 4, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.putText(vis, f"Score: {score}", (460, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"{display_fps:.1f}/{(60 / max(1, frame_skip)):.0f} fps  tracks: {len(tracker.tracks)}",
                    (460, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        yolo_ms    = get_yolo_latency_ms()
        yolo_color = (0, 255, 180) if yolo_ms < 80 else (0, 140, 255)
        try:
            from robot_tracker import _USE_GPU as _rg
        except ImportError:
            _rg = False
        wait_tag = "  WAIT" if yolo_waited else ""
        cv2.putText(vis, f"YOLO {yolo_ms:.0f} ms  Motion: {'GPU' if _rg else 'CPU'}{wait_tag}",
                    (460, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, yolo_color, 1)

        cv2.imshow("tracking", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return score



robot_tracker = RobotTracker()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track")
    parser.add_argument("--side", type=str, choices=["red", "blue"], default="red")
    parser.add_argument("--frame-drop", type=int)
    parser.add_argument("--max-stale-frames", type=int, default=0,
                        help="Max frames a YOLO result can be stale before it's suppressed (0 = unlimited)")
    parser.add_argument("--video-file", type=str)
    args = parser.parse_args()

    cv2.setNumThreads(os.cpu_count() or 1)

    frame_skip = args.frame_drop if args.frame_drop is not None else FRAME_SKIP

    print("Initializing...")
    sc = run(args.video_file, args.side, frame_skip=frame_skip, max_stale_frames=args.max_stale_frames)
    print(f"Final score: {sc}")