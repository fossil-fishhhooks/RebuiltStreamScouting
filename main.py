import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import bisect

from config import (
    FRAME_SKIP,
    MAX_TRAIL,
    PARABOLA_MIN_POINTS,
    PARABOLA_FIT_ERROR,
    SCORE_COOLDOWN_FRAMES,
    SKIP_SECONDS,
    TRAIL_DECAY,
    SCORE_POLYGON_REF_BY_SIDE,
    ATTRIBUTION_MAX_DIST,
    ATTRIBUTION_TIME_TOL,
    ATTRIBUTION_SEARCH_FRAMES,
    MATCH_PERIODS,
    ROBOT_TRACK_LOSS_OK,
    DEBUG_FRAME_OUTLINES,
    CROP_REF
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
    solve_y,
)
from robot_tracker import RobotTracker, detect_robots, _SLOT_COLORS, NUM_ROBOTS
from robot_detector import get_yolo_latency_ms

from path_stitcher import PathStitcher


# ---------------------------------------------------------------------------
# Robot re-identification UI  (draw-box edition)
# ---------------------------------------------------------------------------

class RobotIDUI:
    """
    Pauses playback and lets the operator draw a bounding box for each robot
    slot that needs re-identification.

    Interaction per slot
    --------------------
    Left-click + drag   → draw / redraw the box for this slot.
    "Confirm" button    → accept the box and advance to the next slot.
    "Not in frame" btn  → mark this slot as absent and advance.
    "Done" button       → visible only when every slot is resolved; close UI.
    ESC                 → abort the whole session (returns None).

    Coordinates
    -----------
    All coordinates returned are in CROP-SPACE.  The caller supplies
    crop_offset=(crop_x, crop_y) so the UI can show the full raw frame while
    internally converting click coords back to crop-space.

    Usage
    -----
    ui = RobotIDUI(robot_tracker)
    result = ui.run(raw_frame, crop_offset, slot_ids, prompt="")
    # result: {slot_id: (crop_cx, crop_cy)} -- absent slots are omitted
    # result is None if the user pressed ESC / aborted
    ui.apply_assignments(result, frame_idx)
    """

    _WINDOW     = "Stupid annoying window now follow directions you bum"
    _BTN_H      = 52      # button bar height (px)
    _BTN_PAD    = 10      # padding inside button bar
    _BANNER_H   = 72      # top info banner height (px)

    # Button definitions: (label, key, bg_color, text_color)
    _BTNS = [
        ("Confirm [C]",      ord("c"), (20, 150,  20), (255, 255, 255)),
        ("Not in frame [N]", ord("n"), (20,  20, 150), (255, 255, 255)),
        ("Skip [ESC]",       27,       (60,  60,  60), (180, 180, 180)),
    ]

    def __init__(self, robot_tracker: "RobotTracker"):
        self._rt       = robot_tracker
        self._cb_state = [None]   # [0] = active state dict, or None = disabled

    # ------------------------------------------------------------------
    def run(
            self,
            display_frame: np.ndarray,
            crop_offset: tuple,
            slot_ids: list,
            prompt: str = "",
    ) -> dict | None:
        """
        Walk through *slot_ids* one at a time.

        Returns
        -------
        dict  {slot_id: (crop_cx, crop_cy)}  -- absent/skipped slots omitted.
        None  if the user aborted with ESC before finishing.
        """
        if not slot_ids:
            return {}

        ox, oy   = crop_offset
        result   = {}          # slot_id -> (crop_cx, crop_cy)
        absent   = set()       # slot_ids marked "not in frame"
        aborted  = False

        cv2.namedWindow(self._WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WINDOW, 1280, 760) # aaaaaaa why is the default like 20 pixels

        # Single persistent callback container -- lives on self so it can never
        # be garbage-collected while OpenCV still holds a reference to _mouse.
        self._cb_state[0] = None   # reset for this run() call

        def _mouse(event, x, y, flags, *args):
            s = self._cb_state[0]
            if s is None:          # callback disabled between slots -- ignore
                return
            s["mouse_pos"] = (x, y)
            h_total  = _canvas_h(display_frame)
            btn_top  = h_total - self._BTN_H
            frame_oy = self._BANNER_H

            # Frame-local y = canvas y minus the banner height.
            # Clamped so the box is always inside the image area.
            frame_y = max(0, min(y - frame_oy, display_frame.shape[0] - 1))

            # Events in the banner or button bar → handle as button clicks only
            in_chrome = (y <= frame_oy) or (y >= btn_top)
            if in_chrome:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self._handle_btn_click(x, y, display_frame, s)
                # Never start or continue a drag in chrome areas
                if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
                    s["dragging"] = False
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                s["box_start"] = (x, frame_y)
                s["box_end"]   = (x, frame_y)
                s["dragging"]  = True
            elif event == cv2.EVENT_MOUSEMOVE and s["dragging"]:
                s["box_end"] = (x, frame_y)
            elif event == cv2.EVENT_LBUTTONUP and s["dragging"]:
                s["box_end"]  = (x, frame_y)
                s["dragging"] = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right-click clears the current box
                s["box_start"] = None
                s["box_end"]   = None

        cv2.setMouseCallback(self._WINDOW, _mouse)

        current_idx = 0
        while current_idx < len(slot_ids):
            slot_id = slot_ids[current_idx]
            alliance_cap = "Red" if slot_id < 3 else "Blue"
            alliance_num = (slot_id % 3) + 1
            label = f"{alliance_cap} {alliance_num}"   # e.g. "Red 1", "Blue 3"
            # BGR: red alliance = bright red, blue alliance = bright blue
            color = (0, 0, 220) if slot_id < 3 else (220, 80, 0)

            # Reset shared state for this slot; keep _cb_state[0] pointing at it
            state = {
                "box_start":    None,
                "box_end":      None,
                "dragging":     False,
                "confirmed":    False,
                "not_in_frame": False,
                "abort":        False,
                "mouse_pos":    (0, 0),
                # stored so _mouse can read them without a closure over loop vars
                "_slot_id":     slot_id,
                "_current_idx": current_idx,
                "_slot_ids":    slot_ids,
                "_result":      result,
                "_absent":      absent,
                "_label":       label,
            }
            self._cb_state[0] = state   # enable callback for this slot

            # Inner repaint loop for this slot
            while not state["confirmed"] and not state["not_in_frame"] \
                    and not state["abort"]:

                canvas = self._build_canvas(
                    display_frame, slot_id, current_idx, slot_ids,
                    color, state, result, absent, ox, oy, prompt
                )
                cv2.imshow(self._WINDOW, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key == 27:                    # ESC → abort
                    state["abort"] = True
                elif key in (ord("c"), 13):      # C or Enter → confirm
                    if state["box_start"] and state["box_end"]:
                        state["confirmed"] = True
                elif key == ord("n"):            # N → not in frame
                    state["not_in_frame"] = True

            # Disable callback before we modify any shared state so a stray
            # mouse event between iterations can never write to a dead dict.
            self._cb_state[0] = None

            if state["abort"]:
                aborted = True
                break

            if state["confirmed"] and state["box_start"] and state["box_end"]:
                bx1 = min(state["box_start"][0], state["box_end"][0])
                by1 = min(state["box_start"][1], state["box_end"][1])
                bx2 = max(state["box_start"][0], state["box_end"][0])
                by2 = max(state["box_start"][1], state["box_end"][1])
                cx  = ((bx1 + bx2) // 2) - ox
                cy  = ((by1 + by2) // 2) - oy
                result[slot_id] = (cx, cy)
                print(f"  [RobotID] Slot {slot_id} → box ({bx1},{by1})-({bx2},{by2})  "
                      f"crop centre ({cx},{cy})")
            elif state["not_in_frame"]:
                absent.add(slot_id)
                print(f"  [RobotID] Slot {slot_id} → not in frame")

            current_idx += 1

        cv2.destroyWindow(self._WINDOW)
        if aborted:
            print("  [RobotID] Aborted by user.")
            return None
        return result

    # ------------------------------------------------------------------
    def _handle_btn_click(self, x, y, display_frame, state):
        """Map a click in the button bar to a state transition."""
        slot_ids  = state["_slot_ids"]
        result    = state["_result"]
        absent    = state["_absent"]
        btn_rects = self._button_rects(display_frame)
        for i, rect in enumerate(btn_rects):
            bx1, by1, bx2, by2 = rect
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                label = self._BTNS[i][0]
                if "Confirm" in label:
                    if state["box_start"] and state["box_end"]:
                        state["confirmed"] = True
                elif "Not in frame" in label:
                    state["not_in_frame"] = True
                elif "Skip" in label:
                    state["abort"] = True
                break

    # ------------------------------------------------------------------
    def _button_rects(self, frame):
        """Return (x1, y1, x2, y2) in canvas coords for each button."""
        h = _canvas_h(frame)
        w = frame.shape[1]
        n       = len(self._BTNS)
        btn_w   = (w - self._BTN_PAD * (n + 1)) // n
        y1      = h - self._BTN_H + self._BTN_PAD // 2
        y2      = h - self._BTN_PAD // 2
        rects   = []
        for i in range(n):
            x1 = self._BTN_PAD + i * (btn_w + self._BTN_PAD)
            rects.append((x1, y1, x1 + btn_w, y2))
        return rects

    # ------------------------------------------------------------------
    def _build_canvas(self, base_frame, slot_id, current_idx, slot_ids,
                      color, state, result, absent, ox, oy, prompt):
        """Composite the annotation layer on top of base_frame."""
        h_base, w = base_frame.shape[:2]
        canvas_h  = _canvas_h(base_frame)
        # Allocate canvas with extra space at top (banner) and bottom (buttons)
        extra_top = self._BANNER_H
        canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        canvas[extra_top: extra_top + h_base] = base_frame

        # Offset everything: frame content starts at y=extra_top
        frame_oy = extra_top

        current_label = state["_label"]

        # ── Draw existing robot positions (already initialized) ───────────
        # Skip slots already confirmed/absent in this session -- they get their
        # own marker drawn in the confirmed-boxes loop below, avoiding duplicates.
        for sid, track in self._rt.tracks.items():
            if not track.initialized:
                continue
            if sid in result or sid in absent:
                continue
            c = (0, 0, 220) if sid < 3 else (220, 80, 0)
            px, py = track.position()
            dpx2 = px + ox
            dpy2 = py + oy + frame_oy
            alpha = max(0.3, 1.0 - track.ghost_count / max(1, ROBOT_TRACK_LOSS_OK))
            dc = tuple(int(ch * alpha) for ch in c)
            cv2.circle(canvas, (dpx2, dpy2), 12, dc, 2, cv2.LINE_AA)
            sid_alliance_cap = track.alliance.capitalize()
            sid_num          = (sid % 3) + 1
            lbl = f"{sid_alliance_cap} {sid_num}"
            # Dark shadow then bright text for readability over any background
            cv2.putText(canvas, lbl, (dpx2 + 14, dpy2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (dpx2 + 14, dpy2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, dc, 1, cv2.LINE_AA)

        # ── Draw confirmed boxes from this session ────────────────────────
        for prev_sid, (pcx, pcy) in result.items():
            # BUG FIX: use prev_sid's own label, not the current slot's label
            prev_alliance_cap = "Red" if prev_sid < 3 else "Blue"
            prev_alliance_num = (prev_sid % 3) + 1
            prev_label = f"{prev_alliance_cap} {prev_alliance_num}"
            c = (0, 0, 220) if prev_sid < 3 else (220, 80, 0)
            px = pcx + ox
            py = pcy + oy + frame_oy
            cv2.circle(canvas, (px, py), 10, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
            ok_lbl = f"OK: {prev_label}"
            cv2.putText(canvas, ok_lbl, (px + 14, py - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, ok_lbl, (px + 14, py - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 1, cv2.LINE_AA)

        # ── Draw "absent" labels -- rendered in top-left corner of image ───
        for ai, asid in enumerate(absent):
            absent_alliance_cap = "Red" if asid < 3 else "Blue"
            absent_alliance_num = (asid % 3) + 1
            absent_label = f"{absent_alliance_cap} {absent_alliance_num}"
            c = (0, 0, 220) if asid < 3 else (220, 80, 0)
            txt = f"{absent_label}: NOT IN FRAME"
            ty_absent = frame_oy + 28 + ai * 26
            cv2.putText(canvas, txt, (12, ty_absent),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, txt, (12, ty_absent),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 1, cv2.LINE_AA)

        # ── Draw the in-progress drag box ─────────────────────────────────
        if state["box_start"] and state["box_end"]:
            bx1 = min(state["box_start"][0], state["box_end"][0])
            by1 = min(state["box_start"][1], state["box_end"][1]) + frame_oy
            bx2 = max(state["box_start"][0], state["box_end"][0])
            by2 = max(state["box_start"][1], state["box_end"][1]) + frame_oy
            # Filled semi-transparent highlight
            overlay = canvas.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
            # Solid border (thicker = more visible)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
            # Centre crosshair
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            cv2.drawMarker(canvas, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 18, 4)
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
            # Size label with shadow
            size_lbl = f"{bx2-bx1}x{by2-by1}"
            cv2.putText(canvas, size_lbl, (bx1 + 2, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, size_lbl, (bx1 + 2, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Top banner ────────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (w, self._BANNER_H), (15, 15, 15), -1)
        # Colored left accent bar for the current slot
        cv2.rectangle(canvas, (0, 0), (6, self._BANNER_H), color, -1)
        title = prompt or "ROBOT RE-IDENTIFICATION"
        cv2.putText(canvas, title, (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        box_status = "[BOX READY]" if state["box_start"] and state["box_end"] else "[NO BOX - drag to draw]"
        slot_info = f"{current_label}  ({current_idx + 1}/{len(slot_ids)})   {box_status}"
        cv2.putText(canvas, slot_info, (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        # Slot colour swatch (larger, with white ring)
        swatch_cx = w - 30
        swatch_cy = self._BANNER_H // 2
        cv2.circle(canvas, (swatch_cx, swatch_cy), 20, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (swatch_cx, swatch_cy), 20, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Bottom button bar ─────────────────────────────────────────────
        btn_area_y = canvas_h - self._BTN_H
        cv2.rectangle(canvas, (0, btn_area_y), (w, canvas_h), (20, 20, 20), -1)
        cv2.line(canvas, (0, btn_area_y), (w, btn_area_y), (60, 60, 60), 1)
        btn_rects    = self._button_rects(base_frame)
        for i, (btn_label, _, bg, fg) in enumerate(self._BTNS):
            bx1, by1, bx2, by2 = btn_rects[i]
            active_bg = bg
            active_fg = fg
            # "Confirm" is greyed if no box yet
            if "Confirm" in btn_label and not (state["box_start"] and state["box_end"]):
                active_bg = (40, 40, 40)
                active_fg = (80, 80, 80)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), active_bg, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (90, 90, 90), 1)
            # Centre the text in the button
            (tw, th), _ = cv2.getTextSize(btn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tx = bx1 + (bx2 - bx1 - tw) // 2
            ty = by1 + (by2 - by1 + th) // 2
            cv2.putText(canvas, btn_label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, active_fg, 1, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    def apply_assignments(self, assignments: dict, frame_idx: int) -> None:
        """Force-feed crop-space centre positions into the Kalman filters."""
        if not assignments:
            return
        for slot_id, (cx, cy) in assignments.items():
            track = self._rt.tracks.get(slot_id)
            if track is None:
                continue
            track.kf.statePost = np.array(
                [[float(cx)], [float(cy)], [0.0], [0.0]], dtype=np.float32
            )
            track.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
            track.ghost_count = 0
            track.initialized = True
            track.trail.append((cx, cy))
            track.perma_path.append((cx, cy, frame_idx))
            print(f"  [RobotID] Slot {slot_id} Kalman reset to crop ({cx},{cy})")

    # ------------------------------------------------------------------
    def _draw_all_robots(self, frame: np.ndarray, ox: int, oy: int) -> None:
        """Draw all initialized robot positions in display-frame coords."""
        for slot_id, track in self._rt.tracks.items():
            if not track.initialized:
                continue
            color  = _SLOT_COLORS[slot_id % len(_SLOT_COLORS)]
            px, py = track.position()
            dpx, dpy = px + ox, py + oy
            alpha  = max(0.3, 1.0 - track.ghost_count / max(1, ROBOT_TRACK_LOSS_OK))
            dc     = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, (dpx, dpy), 12, dc, 2, cv2.LINE_AA)
            if track.last_box is not None:
                x1, y1, x2, y2 = track.last_box
                cv2.rectangle(frame, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), dc, 1)
            alliance_cap = track.alliance.capitalize()
            alliance_num = (slot_id % 3) + 1
            cv2.putText(frame, f"{alliance_cap} {alliance_num}", (dpx + 14, dpy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, dc, 2, cv2.LINE_AA)


def _canvas_h(frame: np.ndarray) -> int:
    """Total canvas height = frame height + banner + button bar."""
    return frame.shape[0] + RobotIDUI._BANNER_H + RobotIDUI._BTN_H



def get_frame_at_index(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def period_for_frame(frame_idx, fps, skip_frames):
    """Return the period name (from MATCH_PERIODS) for a given absolute frame index."""
    elapsed_s = (frame_idx - skip_frames) / max(fps, 1)
    for name, end_s in MATCH_PERIODS:
        if elapsed_s < end_s:
            return name
    return MATCH_PERIODS[-1][0]  # past end of last period -- still last period


def period_names():
    """Ordered list of period name strings from config."""
    return [name for name, _ in MATCH_PERIODS]


def attribute_shot(ball_start_frame: int,
                   ball_start_pos: tuple,
                   robot_tracks: dict,
                   ball_trail: list | None = None,
                   alliance: str | None = None) -> int | None:
    """Return the robot slot ID most likely to have fired this ball, or None.

    Strategy (in order of preference):
    1. Tight window: find the closest robot within ATTRIBUTION_TIME_TOL frames
       of ball_start_frame that is also within ATTRIBUTION_MAX_DIST px.
    2. Extrapolation: if no robot qualifies in the tight window, walk forward
       along the ball's trail (if provided) and for each ball position check
       whether any robot was within ATTRIBUTION_MAX_DIST at the corresponding
       frame.  The robot that appears closest across the most trail points wins.
    3. Wide window: if still unresolved, search each robot's perma_path up to
       ATTRIBUTION_SEARCH_FRAMES frames away and accept the nearest one.

    perma_path entries are (x, y, frame_idx).
    ball_trail, if given, is a list of (x, y) points starting at ball_start_frame.
    """
    bx, by = ball_start_pos

    def _closest_robot_pos(track, target_frame: int):
        """Return (px, py, dt) for the perma_path point nearest to target_frame."""
        if not track.perma_path:
            return None
        frames = [pt[2] for pt in track.perma_path]
        idx = bisect.bisect_left(frames, target_frame)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(track.perma_path)]
        if not candidates:
            return None
        best = min((track.perma_path[i] for i in candidates),
                   key=lambda pt: abs(pt[2] - target_frame))
        return best[0], best[1], abs(best[2] - target_frame)

    eligible = {
        slot_id: track
        for slot_id, track in robot_tracks.items()
        if track.initialized and track.perma_path
        and (not alliance or track.alliance in (alliance, "unknown"))
    }
    if not eligible:
        return None

    # ── Stage 1: tight temporal window ────────────────────────────────────
    best_slot = None
    best_dist = float("inf")
    for slot_id, track in eligible.items():
        res = _closest_robot_pos(track, ball_start_frame)
        if res is None:
            continue
        px, py, dt = res
        if dt > ATTRIBUTION_TIME_TOL:
            continue
        dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
        if dist < ATTRIBUTION_MAX_DIST and dist < best_dist:
            best_dist = dist
            best_slot = slot_id

    if best_slot is not None:
        return best_slot

    # ── Stage 2: trail extrapolation ──────────────────────────────────────
    # Walk along the ball's trail.  For each (frame, ball_pos) pair check which
    # robot was closest at that frame.  Accumulate a score per slot (sum of
    # inverse distances) and return the winner if it is unambiguous.
    if ball_trail and len(ball_trail) >= 2:
        slot_scores: dict[int, float] = {sid: 0.0 for sid in eligible}
        slot_hits:   dict[int, int]   = {sid: 0   for sid in eligible}

        for step, (tbx, tby) in enumerate(ball_trail):
            trail_frame = ball_start_frame + step
            for slot_id, track in eligible.items():
                res = _closest_robot_pos(track, trail_frame)
                if res is None:
                    continue
                px, py, dt = res
                if dt > ATTRIBUTION_SEARCH_FRAMES:
                    continue
                dist = ((tbx - px) ** 2 + (tby - py) ** 2) ** 0.5
                if dist < ATTRIBUTION_MAX_DIST:
                    slot_scores[slot_id] += 1.0 / (dist + 1.0)
                    slot_hits[slot_id]   += 1

        # Accept if one slot has ≥2 hits and clearly dominates (2× next best)
        ranked = sorted(slot_scores.items(), key=lambda kv: kv[1], reverse=True)
        if ranked and ranked[0][1] > 0:
            top_slot, top_score = ranked[0]
            runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if slot_hits[top_slot] >= 2 and top_score >= 2.0 * max(runner_score, 1e-9):
                return top_slot

    # ── Stage 3: wide temporal window ─────────────────────────────────────
    best_slot = None
    best_dist = float("inf")
    for slot_id, track in eligible.items():
        res = _closest_robot_pos(track, ball_start_frame)
        if res is None:
            continue
        px, py, dt = res
        if dt > ATTRIBUTION_SEARCH_FRAMES:
            continue
        dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
        if dist < ATTRIBUTION_MAX_DIST and dist < best_dist:
            best_dist = dist
            best_slot = slot_id

    return best_slot


def polygon_center(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (np.mean(xs), np.mean(ys))


def adjust_polygon_for_apriltag(frame,side, frame_240_offset=None):
    if frame_240_offset is None:
        try:


            ref_frame=frame

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

def run(video_path, side, frame_skip=FRAME_SKIP, max_stale_frames=2):
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    ref_frame = get_frame_at_index(cap, 240)

    adjust_polygon_for_apriltag(ref_frame,side)
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("Could not read input video dimensions")

    crop_region, hole_region, score_polygon, active_region = get_runtime_regions(frame_w, frame_h, side)

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(SKIP_SECONDS * fps)

    print(
        f"[regions] src={frame_w}x{frame_h} crop={crop_region} "
        f"hole={hole_region} score={score_polygon} active={active_region}"
    )

    tracker       = Tracker()
    robot_tracker = RobotTracker()
    trails     = defaultdict(list)
    t_alphas   = defaultdict(list)
    full_trails = defaultdict(list)  # unbounded, no decay -- used for scored snapshots
    first_seen = {}
    origins    = {}
    fit_cache  = {}  # oid -> (trail_len, params, err, x_min, x_max)

    score                   = 0
    last_score_frame        = -SCORE_COOLDOWN_FRAMES
    last_score_frame_per_id = defaultdict(lambda: -10)
    scored_track_ids        = set()
    scored_curves           = {}   # oid -> (trail_pts, curve_pts), drawn permanently

    # Per-robot scores: {slot_id: {"total": int, <period_name>: int, ...}}
    _pnames = period_names()
    robot_scores = {i: {"total": 0, **{p: 0 for p in _pnames}}
                    for i in range(6)}
    # {canon_oid -> slot_id} so we can attribute post-hoc stitched shots
    shot_robot_map: dict = {}

    stitcher        = PathStitcher()
    robot_id_ui     = RobotIDUI(robot_tracker)   # re-identification UI
    _startup_done   = False   # flag: we still need to run the startup ID prompt
    _lost_prompted  = set()   # slot IDs already prompted this loss episode
    # Consecutive-frame counter for each slot being lost; triggers Re-ID once
    # it crosses ROBOT_TRACK_LOSS_OK.  Reset when the slot is re-found.
    _loss_frame_counts: dict = {i: 0 for i in range(NUM_ROBOTS)}
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
        robot_dets=[]
        for (cx, cy, w, h, alliance, x1, y1, x2, y2) in robot_dets_full:
            cx2 = cx-crop_x
            cy2= cy-crop_y
            nx1 = x1-crop_x
            ny1 = y1-crop_y
            nx2 = x2-crop_x
            ny2 = y2-crop_y
            if y2<CROP_REF[3]:
                robot_dets.append((cx2,cy2,w,h,alliance,nx1,ny1,nx2,ny2))

        t_robots = time.perf_counter()
        yolo_waited = max_stale_frames > 0 and (t_robots - t_before_robots) > 0.002

        live, dormant = robot_tracker.update(robot_dets, crop_frame=frame)
        frame = robot_tracker.draw(frame)
        t_rtrack = time.perf_counter()

        # ── Robot re-identification UI ─────────────────────────────────────
        # 1) STARTUP -- always shown on the very first processed frame so the
        #    operator can draw boxes for all 6 slots before playback begins.
        if not _startup_done:
            all_ids = list(range(NUM_ROBOTS))
            print(f"  [RobotID] Startup: prompting for all {NUM_ROBOTS} slots")
            assignments = robot_id_ui.run(
                frame, (0, 0), all_ids,
                prompt="STARTUP: draw a box around each robot to seed tracking.",
            )
            # None means the user pressed ESC/abort -- skip seeding but mark done
            if assignments:
                robot_id_ui.apply_assignments(assignments, frame_idx)
            _startup_done = True
            _lost_prompted.clear()
            # Reset loss counters after startup
            for k in _loss_frame_counts:
                _loss_frame_counts[k] = 0

        # 2) Update per-slot loss counters.
        #    A slot counts as "lost" if:
        #      • it has never been initialized (not seeded and not seen by YOLO), OR
        #      • it is initialized but has ghost_count > 0 (YOLO has lost it), OR
        #      • it was marked "not in frame" by the operator (initialized=False,
        #        ghost_count=0 but still not contributing to active tracking).
        #    A slot is considered *active* only when initialized AND ghost_count==0.
        for tid, track in robot_tracker.tracks.items():
            is_active = track.initialized and track.ghost_count == 0
            if is_active:
                # Robot is live -- reset its loss counter and allow future re-prompts
                if _loss_frame_counts[tid] > 0:
                    _loss_frame_counts[tid] = 0
                _lost_prompted.discard(tid)
            else:
                # Robot is absent, ghosted, or never initialized -- count as lost
                _loss_frame_counts[tid] += 1

        # 3) TRACK-LOSS -- trigger when ≥1 slot has been continuously lost for
        #    ROBOT_TRACK_LOSS_OK frames and hasn't already been prompted this
        #    loss episode.
        #
        #    Key rule: a slot that the operator marked "N" (not in frame) is
        #    never added to _lost_prompted so it will re-trigger after another
        #    ROBOT_TRACK_LOSS_OK frames.  This ensures the system keeps asking
        #    until all 6 robots are genuinely visible and being tracked.
        newly_lost = [
            tid for tid, cnt in _loss_frame_counts.items()
            if cnt >= ROBOT_TRACK_LOSS_OK and tid not in _lost_prompted
        ]
        if newly_lost:
            labels_str = ", ".join(
                f"{'Red' if tid < 3 else 'Blue'} {(tid % 3) + 1}"
                for tid in newly_lost
            )
            print(f"  [RobotID] Track loss threshold reached for {labels_str} "
                  f"(counts: {[_loss_frame_counts[t] for t in newly_lost]})")
            assignments = robot_id_ui.run(
                frame, (0, 0), newly_lost,
                prompt=(
                    f"TRACKING LOST -- {labels_str} missing for "
                    f">={ROBOT_TRACK_LOSS_OK} frames.  Draw a box (or 'Not in frame')."
                ),
            )
            if assignments is not None:   # None = user aborted (ESC)
                robot_id_ui.apply_assignments(assignments, frame_idx)
                # Only silence slots that were confirmed (box drawn).
                # Absent slots ("N") are deliberately NOT added to _lost_prompted
                # so they will re-trigger after another ROBOT_TRACK_LOSS_OK frames.
                confirmed_slots = set(assignments.keys())
                _lost_prompted.update(confirmed_slots)
                # Reset loss counters only for confirmed slots
                for tid in confirmed_slots:
                    _loss_frame_counts[tid] = 0
                # Absent slots: clear their counter so re-trigger starts fresh
                absent_slots = set(newly_lost) - confirmed_slots
                for tid in absent_slots:
                    _loss_frame_counts[tid] = 0   # restart the count from zero

        for oid, (x, y) in objects.items():
            track = tracker.tracks.get(oid)
            if track and track.ghost_count == 0:
                if oid not in first_seen:
                    first_seen[oid] = (x, y, frame_idx)
                trails[oid].append((x, y))
                t_alphas[oid].append(1.0)
                full_trails[oid].append((x, y))
                if len(trails[oid]) > MAX_TRAIL:
                    trails[oid].pop(0)
                    t_alphas[oid].pop(0)

        # Re-identification: tracker resurrected a dead ID -- full_trail is
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

        # ── live trail stitching (parabolic + velocity coherence) ──────────
        stitcher.update(trails, t_alphas, full_trails, tracker, frame_idx)

        # ── trail decay ────────────────────────────────────────────────────
        for oid in list(t_alphas.keys()):
            t_alphas[oid] = [a * TRAIL_DECAY for a in t_alphas[oid]]
            combined = [(p, a) for p, a in zip(trails[oid], t_alphas[oid]) if a > 0.05]
            if combined:
                pts, alps = zip(*combined)
                trails[oid] = list(pts)
                t_alphas[oid] = list(alps)
            else:
                trails.pop(oid, None)
                t_alphas.pop(oid, None)
                # also evict fit cache inside stitcher
                stitcher._fit_cache.pop(oid, None)

        # ── score check (now sees stitched trails) ─────────────────────────
        for oid, pts in trails.items():
            # Resolve canonical ID in case this oid was a continuation
            canon_oid = stitcher.remap.get(oid, oid)
            track = tracker.tracks.get(oid)
            track_lost = bool(track and track.ghost_count == 1)
            if check_parabola_score(
                    canon_oid, pts, frame_idx, last_score_frame_per_id,
                    last_score_frame, score_polygon, scored_track_ids,
                    track_lost=track_lost,
            ):
                score += 1
                last_score_frame_per_id[canon_oid] = frame_idx
                last_score_frame = frame_idx
                scored_track_ids.add(canon_oid)
                if canon_oid not in origins and canon_oid in first_seen:
                    origins[canon_oid] = first_seen[canon_oid][:2]

                if canon_oid not in scored_curves:
                    trail_snap = list(full_trails.get(oid, []))
                    scored_curves[canon_oid] = (trail_snap, [])

                # ── Attribute this shot to the nearest robot ──────────────
                if oid in first_seen:
                    ball_start_pos   = first_seen[oid][:2]
                    ball_start_frame = first_seen[oid][2]
                elif full_trails.get(oid):
                    ball_start_pos   = full_trails[oid][0]
                    ball_start_frame = frame_idx - len(full_trails[oid]) + 1
                else:
                    ball_start_pos   = pts[0]
                    ball_start_frame = frame_idx - len(pts) + 1
                # Pass the full ball trail so attribute_shot can extrapolate
                _ball_trail = full_trails.get(oid) or list(pts)
                slot_id = attribute_shot(ball_start_frame, ball_start_pos,
                                         robot_tracker.tracks,
                                         ball_trail=_ball_trail,
                                         alliance=side)
                if slot_id is not None:
                    period = period_for_frame(ball_start_frame, fps, skip_frames)
                    robot_scores[slot_id]["total"]  += 1
                    robot_scores[slot_id][period]   += 1
                    shot_robot_map[canon_oid] = slot_id
                    print(f"  [SCORE]  ID {canon_oid} @ frame {frame_idx} -> total: {score}"
                          f"  robot slot {slot_id} ({period})")
                else:
                    print(f"  [SCORE]  ID {canon_oid} @ frame {frame_idx} -> total: {score}"
                          f"  (no robot attribution)")

        # ── post-hoc scored-curve stitching (upgraded, replaces old loop) ──
        scored_curves = stitcher.stitch_scored_curves(scored_curves)

        t_score = time.perf_counter()

        vis = frame.copy()

        # ── Debug frame outlines ───────────────────────────────────────────
        # Each processing sub-region gets a colored border + title so it's
        # easy to see exactly what area each pipeline stage operates on.
        # Toggle DEBUG_FRAME_OUTLINES in config.py to show/hide.
        if DEBUG_FRAME_OUTLINES:
            _dbg_font  = cv2.FONT_HERSHEY_SIMPLEX
            _dbg_thick = 2
            _dbg_bg    = (0, 0, 0)   # shadow behind text

            def _dbg_label(img, rect, border_color, title, thickness=2):
                """Draw a colored border + title on img for the given (x1,y1,x2,y2) rect."""
                x1, y1, x2, y2 = rect
                cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(title, _dbg_font, 0.52, 1)
                # Label background pill
                cv2.rectangle(img, (x1, y1), (x1 + tw + 6, y1 + th + 6), _dbg_bg, -1)
                cv2.putText(img, title, (x1 + 3, y1 + th + 2),
                            _dbg_font, 0.52, border_color, 1, cv2.LINE_AA)

            # 1. Full cropped field view (= the whole vis frame)
            _h_vis, _w_vis = vis.shape[:2]
            _dbg_label(vis, (0, 0, _w_vis - 1, _h_vis - 1),
                       (200, 200, 200), "CROP / FIELD VIEW")

            # 2. Active region (where ball detection runs)
            _dbg_label(vis, active_region, (0, 255, 120), "BALL DETECT (active region)")

            # 3. Hole region (blacked out / ignored)
            _dbg_label(vis, hole_region, (40, 40, 200), "HOLE (excluded)")

            # 4. Score polygon bounding box
            _sp_xs = [p[0] for p in score_polygon]
            _sp_ys = [p[1] for p in score_polygon]
            _dbg_label(vis,
                       (min(_sp_xs), min(_sp_ys), max(_sp_xs), max(_sp_ys)),
                       (0, 220, 255), "SCORE ZONE")

            # 5. Robot detection frame (raw frame shown as inset outline in
            #    crop-space: the raw frame covers the full crop area, so the
            #    border is the same as the crop border -- just label it differently)
            _dbg_label(vis, (2, 2, _w_vis - 3, _h_vis - 3),
                       (255, 120, 0), "ROBOT DETECT (YOLO)")

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

        # ── Draw permanently retained scored-shot trails ───────────────────
        for shot_idx, (oid, (trail_pts, cpts)) in enumerate(scored_curves.items(), start=1):
            # Color the trail by which robot shot it (slot color) or orange if unattributed
            r_slot = shot_robot_map.get(oid)
            shot_color = _SLOT_COLORS[r_slot % len(_SLOT_COLORS)] if r_slot is not None \
                else (0, 165, 255)
            for i in range(1, len(trail_pts)):
                cv2.line(vis, trail_pts[i - 1], trail_pts[i], shot_color, 2, cv2.LINE_AA)
            mid = trail_pts[len(trail_pts) // 2]
            cv2.putText(vis, f"#{shot_idx}", (mid[0] + 4, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, shot_color, 1, cv2.LINE_AA)

        # ── Robot score labels next to each live robot box ─────────────────
        for slot_id, track in robot_tracker.tracks.items():
            if not track.initialized:
                continue
            rs = robot_scores[slot_id]
            alliance_cap = track.alliance.capitalize()
            alliance_num = (slot_id % 3) + 1
            robot_label  = f"{alliance_cap} {alliance_num}"
            # Build period breakdown dynamically e.g. "A:2 T:1 E:0"
            period_parts  = " ".join(f"{n[0].upper()}:{rs[n]}" for n, _ in MATCH_PERIODS)
            score_label   = f"{rs['total']}pt  {period_parts}"

            slot_color = _SLOT_COLORS[slot_id % len(_SLOT_COLORS)]
            if track.last_box is not None and track.ghost_count == 0:
                bx2 = track.last_box[2]
                by1 = track.last_box[1]
            else:
                px, py = track.position()
                bx2, by1 = px + track.w // 2, py - track.h // 2

            cv2.putText(vis, score_label, (bx2 + 4, by1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_color, 1, cv2.LINE_AA)

        # ── Current game period indicator ──────────────────────────────────
        current_period = period_for_frame(frame_idx, fps, skip_frames)
        # Cycle through distinct colors for however many periods exist
        _period_colors = [(0,255,255),(0,255,0),(0,140,255),(255,200,0),(200,0,255)]
        _pidx = next((i for i, (n,_) in enumerate(MATCH_PERIODS) if n == current_period), 0)
        period_color = _period_colors[_pidx % len(_period_colors)]
        cv2.putText(vis, current_period.upper(), (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, period_color, 2, cv2.LINE_AA)

        # ── Top-right HUD ──────────────────────────────────────────────────
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

        # ── Live scoreboard table ──────────────────────────────────────────
        # Drawn as a semi-transparent dark panel at bottom-left so it's
        # always readable regardless of field content.
        _pnames     = period_names()
        attributed  = sum(rs["total"] for rs in robot_scores.values())
        t_factor    = attributed / score if score > 0 else 0.0
        col_w       = 44   # px per period column
        row_h       = 16
        n_cols      = 2 + len(_pnames) + 1   # robot | total | periods... | scaled
        table_w     = 88 + col_w * (len(_pnames) + 1)
        n_rows      = 7   # header + 6 robots
        table_h     = row_h * (n_rows + 1) + 8
        tx, ty      = 4, vis.shape[0] - table_h - 4

        # Dark background panel
        overlay = vis.copy()
        cv2.rectangle(overlay, (tx - 2, ty - 2),
                      (tx + table_w, ty + table_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        # Header row
        hdr_y = ty + row_h
        cv2.putText(vis, "ROBOT", (tx + 2, hdr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(vis, "TOT", (tx + 90, hdr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        for pi, pname in enumerate(_pnames):
            cv2.putText(vis, pname[:3].upper(), (tx + 90 + col_w*(pi+1), hdr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(vis, "EST", (tx + 90 + col_w*(len(_pnames)+1), hdr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        # Divider
        div_y = ty + row_h + 3
        cv2.line(vis, (tx, div_y), (tx + table_w, div_y), (100,100,100), 1)

        for row, (slot_id, rs) in enumerate(robot_scores.items()):
            ry = ty + row_h * (row + 2) + 4
            rtrack   = robot_tracker.tracks[slot_id]
            alliance_abbr = {"red": "RED", "blue": "BLU", "unknown": "UNK"}.get(
                rtrack.alliance, "UNK")
            anum      = (slot_id % 3) + 1
            row_label = f"{alliance_abbr}{anum}"   # "RED1", "BLU3" -- compact for table
            slot_col  = _SLOT_COLORS[slot_id % len(_SLOT_COLORS)]

            cv2.putText(vis, row_label, (tx + 2, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)
            cv2.putText(vis, str(rs["total"]), (tx + 90, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)
            for pi, pname in enumerate(_pnames):
                cv2.putText(vis, str(rs[pname]),
                            (tx + 90 + col_w*(pi+1), ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)
            scaled = rs["total"] / t_factor if t_factor > 0 else 0.0
            cv2.putText(vis, f"{scaled:.1f}",
                        (tx + 90 + col_w*(len(_pnames)+1), ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)

        # Footer: tracking factor
        foot_y = ty + table_h - 2
        cv2.putText(vis, f"trk={t_factor:.2f}  attr={attributed}/{score}",
                    (tx + 2, foot_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160,160,160), 1)

        cv2.imshow("Main annoying window that is part of the program thats murdering your system", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── End-of-match summary ───────────────────────────────────────────────
    attributed_total = sum(rs["total"] for rs in robot_scores.values())
    # Tracking error factor: what fraction of scored balls we could attribute
    # to a specific robot.  1.0 = perfect attribution, <1.0 = some shots lost.
    tracking_factor = attributed_total / score if score > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"  MATCH SCORE (ball tracker): {score}")
    print(f"  Attributed shots:           {attributed_total}")
    print(f"  Tracking error factor:      {tracking_factor:.3f}  "
          f"({'perfect' if tracking_factor == 1.0 else 'some shots unattributed'})")
    print("-" * 60)
    print(f"  {'Robot':<12} {'Total':>5}  {'Auto':>5}  {'Teleop':>6}  {'Endgame':>7}  {'Scaled':>7}")
    for slot_id, rs in robot_scores.items():
        track = robot_tracker.tracks[slot_id]
        alliance_cap = track.alliance.capitalize()
        alliance_num = (slot_id % 3) + 1
        label = f"{alliance_cap} {alliance_num} (s{slot_id})"
        # Scale each robot's score by the tracking factor -- gives best estimate
        # of true contribution accounting for missed attributions
        scaled = rs["total"] / tracking_factor if tracking_factor > 0 else 0.0
        print(f"  {label:<16} {rs['total']:>5}  {rs['auto']:>5}  {rs['teleop']:>6}"
              f"  {rs['endgame']:>7}  {scaled:>7.2f}")
    print("=" * 60 + "\n")

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track")
    parser.add_argument("--side", type=str, choices=["red", "blue"], default="red")
    parser.add_argument("--frame-drop", type=int)
    parser.add_argument("--max-stale-frames", type=int, default=5,
                        help="Max frames a YOLO result can be stale before it's suppressed (0 = unlimited)")
    parser.add_argument("--video-file", type=str)
    args = parser.parse_args()

    cv2.setNumThreads(os.cpu_count() or 1)

    frame_skip = args.frame_drop if args.frame_drop is not None else FRAME_SKIP

    print("Initializing...")


    sc = run(args.video_file, args.side, frame_skip=frame_skip, max_stale_frames=args.max_stale_frames)
    print(f"Final score: {sc}")