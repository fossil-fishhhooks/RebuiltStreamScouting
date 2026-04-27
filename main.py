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
solve_y
)
from robot_tracker import RobotTracker, detect_robots


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


def run(video_path, side, frame_skip=FRAME_SKIP):
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
    trails   = defaultdict(list)  # smoothed position history per track
    t_alphas = defaultdict(list)  # opacity per trail point, decays over time
    first_seen = {}               # first position ever seen for each track, before trail decay can erase it
    origins    = {}               # launch point per track, only shown once confirmed scored

    score                   = 0
    last_score_frame        = -SCORE_COOLDOWN_FRAMES
    last_score_frame_per_id = defaultdict(lambda: -10)
    scored_track_ids        = set()

    fps_window   = 30
    frame_times  = []
    display_fps  = 0.0
    last_fps_update = time.time()

    frame_idx = 0

    while True:
        # grab() seeks without decoding, much faster than read() for skipped frames
        for _ in range(max(0, frame_skip - 1)):
            cap.grab()
            frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        full_frame = frame

        if frame_idx < skip_frames:
            continue

        t_start = time.perf_counter()

        frame   = crop_frame(frame, crop_region)
        circles = detect_circles(frame, hole_region, active_region)
        objects = tracker.update([(x, y) for (x, y, r) in circles])
        robot_dets = detect_robots(full_frame, alliance="both")
        live, dormant = robot_tracker.update(robot_dets)
        frame = robot_tracker.draw(frame)

        for oid, (x, y) in objects.items():
            track = tracker.tracks.get(oid)
            if track and track.ghost_count == 0:
                if oid not in first_seen:
                    first_seen[oid] = (x, y)
                trails[oid].append((x, y))
                t_alphas[oid].append(1.0)
                if len(trails[oid]) > MAX_TRAIL:
                    trails[oid].pop(0)
                    t_alphas[oid].pop(0)

        # multiply all alphas by decay factor each frame, drop points that have faded out
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

        for oid, pts in trails.items():
            track      = tracker.tracks.get(oid)
            track_lost = bool(track and track.ghost_count == 1)
            if check_parabola_score(
                    oid,
                    pts,
                    frame_idx,
                    last_score_frame_per_id,
                    last_score_frame,
                    score_polygon,
                    scored_track_ids,
                    track_lost=track_lost,
            ):
                score += 1
                last_score_frame_per_id[oid] = frame_idx
                last_score_frame             = frame_idx
                scored_track_ids.add(oid)
                # save the launch point now that we know it was a scored shot
                if oid not in origins and oid in first_seen:
                    origins[oid] = first_seen[oid]
                print(f"  [SCORE]  ID {oid} @ frame {frame_idx} -> total: {score}")

        vis = frame.copy()
        #vis = blackout_outside_active(vis, active_region)
        #vis = blackout_hole(vis, hole_region)

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

        for oid, pts in trails.items():
            if len(pts) >= PARABOLA_MIN_POINTS:
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                params, err = fit_conic(xs, ys)
                a, b, c, theta = params
                print(f"[fit] oid={oid} err={err:.4f} a={a:.4f} b={b:.4f} c={c:.4f} theta={np.degrees(theta):.1f}° pts={len(pts)}")
                print(f"      xs={int(xs.min())}..{int(xs.max())}  ys={int(ys.min())}..{int(ys.max())}")

                col = (0, 200, 255) if err < PARABOLA_FIT_ERROR else (80, 80, 80)
                h, w = vis.shape[:2]

                any_drawn = 0
                for xi in range(int(xs.min()), int(xs.max()), 2):
                    sol = solve_y(a, b, c, theta, float(xi))
                    if sol is None:
                        continue
                    for y in sol:
                        yi = int(round(y))
                        if 0 <= yi < h:
                            cv2.circle(vis, (xi, yi), 1, col, -1)



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

        # # draw launch point markers for scored shots, persists for the rest of the video
        # for oid, (ox, oy) in origins.items():
        #     arm, gap = 8, 3
        #     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        #         cv2.line(
        #             vis,
        #             (ox + dx * gap,         oy + dy * gap),
        #             (ox + dx * (arm + gap), oy + dy * (arm + gap)),
        #             (0, 200, 255), 2, cv2.LINE_AA,
        #         )
        #     cv2.circle(vis, (ox, oy), 3, (0, 200, 255), -1, cv2.LINE_AA)
        #     cv2.putText(vis, f"#{oid}", (ox + arm + gap + 2, oy + 4),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > fps_window:
            frame_times.pop(0)
        now = time.time()
        if now - last_fps_update > 0.5 and frame_times:
            display_fps     = 1.0 / (sum(frame_times) / len(frame_times))
            last_fps_update = now

        cv2.putText(vis, f"Score: {score}", (460, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"{display_fps:.1f}/{(60 / max(1, frame_skip)):.0f} fps  tracks: {len(tracker.tracks)}",
                    (460, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)


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
    parser.add_argument("--video-file", type=str)
    args = parser.parse_args()

    cv2.setNumThreads(os.cpu_count() or 1)

    frame_skip = args.frame_drop if args.frame_drop is not None else FRAME_SKIP

    print("Initializing...")
    sc = run(args.video_file, args.side, frame_skip=frame_skip)
    print(f"Final score: {sc}")