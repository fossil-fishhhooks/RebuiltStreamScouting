"""
path_stitcher.py — Live and post-hoc trail fusion for ball tracking.

Two complementary strategies are combined:

1. PARABOLIC CONTINUATION  (geometry-first)
   Fit a rotated conic to the older trail.  Extrapolate it forward in time and
   check whether the younger trail's opening points land close to that predicted
   arc.  If they do, the two fragments are the same shot split by a tracking
   dropout and are merged.

2. KALMAN VELOCITY COHERENCE  (physics-first)
   When a Kalman track ends (ghost_count just became 1) and a new track appears
   shortly after within a spatial corridor, test whether the velocity implied by
   the gap is consistent with the dying track's last measured velocity and with
   ballistic deceleration limits.  If it passes, forcibly re-ID the new track
   into the old one and splice the trails.

Both strategies operate on *live* trails so the score-checker always sees a
clean, stitched arc rather than fragments.

A separate post-hoc pass (``stitch_scored_curves``) handles any merges that
were missed at detection time — it supersedes the old proximity-only loop in
main.py.

Usage (in main.py)
------------------
    from path_stitcher import PathStitcher
    stitcher = PathStitcher()

    # inside the per-frame loop, after tracker.update():
    stitcher.update(trails, t_alphas, full_trails, tracker, frame_idx)

    # after scoring:
    scored_curves = stitcher.stitch_scored_curves(scored_curves)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Parabolic continuation
PARA_MIN_POINTS       = 6     # minimum trail length to attempt a conic fit
PARA_FIT_ERR_MAX      = 5.0   # RMS residual threshold — reject noisy fits
PARA_EXTRAP_FRAMES    = 18    # how many frames ahead to extrapolate the arc
PARA_MATCH_DIST       = 14.0  # px — max distance from predicted arc to accept

# Kalman velocity coherence
VEL_MAX_GAP_FRAMES    = 14    # max frame gap between end-of-A and start-of-B
VEL_SPATIAL_CORRIDOR  = 80.0  # px — coarse spatial gate before velocity test
VEL_ANGLE_TOL_DEG     = 38.0  # degrees — max heading change allowed
VEL_SPEED_RATIO_MAX   = 2.8   # new_speed / old_speed must be < this (decel/accel cap)
VEL_MIN_SPEED         = 2.0   # px/frame — below this speed the velocity test is skipped

# Post-hoc scored-curve stitching (supersedes the proximity loop in main.py)
STITCH_DIST           = 28    # px — endpoint proximity gate
STITCH_ANGLE_TOL_DEG  = 45.0  # max heading deviation for post-hoc stitch
STITCH_USE_PARABOLA   = True  # also run parabolic check on scored fragments


# ---------------------------------------------------------------------------
# Conic fitting  (same maths as vision.fit_conic but self-contained)
# ---------------------------------------------------------------------------

def _fit_conic(
    pts: List[Tuple[int, int]],
) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
    """
    Fit a rotated parabola  a·xp² + b·yp + c = 0  (b fixed to -1)
    in a rotated frame defined by angle θ.

    Returns (params=(a,b,c,θ), rms_error) or None if fitting fails.
    """
    if len(pts) < PARA_MIN_POINTS:
        return None

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # Bail out if the trail is basically stationary
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    if span < 15:
        return None

    def residuals(p):
        a, c, theta = p
        b = -1.0
        ct, st = np.cos(theta), np.sin(theta)
        xp = xs * ct - ys * st
        yp = xs * st + ys * ct
        return a * xp ** 2 + b * yp + c

    # Initial guess: axis aligned, slight upward curvature
    p0 = [1e-3, float(ys.mean()), 0.0]
    try:
        result = least_squares(residuals, p0, method="lm", max_nfev=600)
    except Exception:
        return None

    a, c, theta = result.x
    b = -1.0
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return (a, b, c, theta), rms


def _eval_conic_y(
    params: Tuple[float, float, float, float],
    x: float,
    ref_y: float,
) -> Optional[float]:
    """Solve the conic for y given x; return the branch closest to ref_y."""
    a, b, c, theta = params
    ct, st = math.cos(theta), math.sin(theta)
    # xp = x*ct - y*st,  yp = x*st + y*ct
    # a*xp^2 + b*yp + c = 0  →  quadratic in y
    A = a * st ** 2
    B = -2 * a * x * ct * st + b * ct
    C = a * x ** 2 * ct ** 2 + b * x * st + c
    if abs(A) < 1e-10:
        if abs(B) < 1e-10:
            return None
        return -C / B
    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    y1 = (-B + sq) / (2 * A)
    y2 = (-B - sq) / (2 * A)
    return y1 if abs(y1 - ref_y) <= abs(y2 - ref_y) else y2


def _extrapolate_arc(
    params: Tuple[float, float, float, float],
    anchor_pts: List[Tuple[int, int]],
    n_steps: int,
) -> List[Tuple[float, float]]:
    """
    Walk along the fitted conic beyond anchor_pts[-1] for n_steps pixel steps
    in the dominant axis direction.  Returns predicted (x, y) positions.
    """
    if len(anchor_pts) < 2:
        return []

    xs = [p[0] for p in anchor_pts]
    ys = [p[1] for p in anchor_pts]
    dx = xs[-1] - xs[-2]
    dy = ys[-1] - ys[-2]

    # Choose sweep axis
    if abs(dx) >= abs(dy):
        step = math.copysign(1, dx)
        pts = []
        cur_x = float(xs[-1])
        for _ in range(n_steps):
            cur_x += step
            cy = _eval_conic_y(params, cur_x, float(ys[-1] if not pts else pts[-1][1]))
            if cy is None:
                break
            pts.append((cur_x, cy))
        return pts
    else:
        # steep arc — sweep y
        step = math.copysign(1, dy)
        pts = []
        cur_y = float(ys[-1])
        for _ in range(n_steps):
            cur_y += step
            # solve x from y: symmetric math
            a, b, c, theta = params
            ct, st = math.cos(theta), math.sin(theta)
            A = a * ct ** 2
            B = -2 * a * cur_y * ct * st + b * st
            C = a * cur_y ** 2 * st ** 2 + b * cur_y * ct + c
            if abs(A) < 1e-10:
                if abs(B) < 1e-10:
                    break
                cx = -C / B
            else:
                disc = B ** 2 - 4 * A * C
                if disc < 0:
                    break
                sq = math.sqrt(disc)
                x1 = (-B + sq) / (2 * A)
                x2 = (-B - sq) / (2 * A)
                ref_x = float(xs[-1] if not pts else pts[-1][0])
                cx = x1 if abs(x1 - ref_x) <= abs(x2 - ref_x) else x2
            pts.append((cx, cur_y))
        return pts


def _min_dist_to_polyline(
    pt: Tuple[float, float],
    polyline: List[Tuple[float, float]],
) -> float:
    """Minimum Euclidean distance from pt to any point in polyline."""
    if not polyline:
        return float("inf")
    px, py = pt
    return min(math.hypot(px - qx, py - qy) for qx, qy in polyline)


# ---------------------------------------------------------------------------
# Velocity helpers
# ---------------------------------------------------------------------------

def _trail_velocity(pts: List[Tuple[int, int]], window: int = 4) -> Optional[Tuple[float, float]]:
    """
    Estimate (vx, vy) in px/frame from the last `window` points.
    Returns None if trail is too short.
    """
    if len(pts) < 2:
        return None
    tail = pts[-min(window, len(pts)):]
    if len(tail) < 2:
        return None
    vx = (tail[-1][0] - tail[0][0]) / (len(tail) - 1)
    vy = (tail[-1][1] - tail[0][1]) / (len(tail) - 1)
    return vx, vy


def _velocity_coherent(
    vel_a: Tuple[float, float],
    vel_b: Tuple[float, float],
    gap_frames: int,
) -> bool:
    """
    True if the velocity at the start of trail B is consistent with
    trail A's terminal velocity after `gap_frames` of free flight.

    Checks:
      • heading change ≤ VEL_ANGLE_TOL_DEG
      • speed ratio within VEL_SPEED_RATIO_MAX  (allows for some decel)
    """
    ax, ay = vel_a
    bx, by = vel_b
    speed_a = math.hypot(ax, ay)
    speed_b = math.hypot(bx, by)

    if speed_a < VEL_MIN_SPEED:
        return False  # can't infer direction from near-stationary end

    # Heading check
    angle_a = math.atan2(ay, ax)
    angle_b = math.atan2(by, bx)
    delta   = abs(math.degrees(math.atan2(
        math.sin(angle_b - angle_a), math.cos(angle_b - angle_a)
    )))
    if delta > VEL_ANGLE_TOL_DEG:
        return False

    # Speed ratio check (ratio must be ≥ 1/max and ≤ max)
    if speed_b < 1e-3:
        return False
    ratio = speed_b / speed_a
    if ratio > VEL_SPEED_RATIO_MAX or ratio < 1.0 / VEL_SPEED_RATIO_MAX:
        return False

    return True


# ---------------------------------------------------------------------------
# PathStitcher
# ---------------------------------------------------------------------------

class PathStitcher:
    """
    Stateful stitcher that is called once per frame.

    Internal bookkeeping
    --------------------
    _death_log : {oid: (frame_idx, last_pts, vel, fit_result)}
        Snapshot of tracks that just died (ghost_count became 1) so we can
        try to match them against new tracks that appear in the next
        VEL_MAX_GAP_FRAMES frames.

    _remap : {new_oid: old_oid}
        When a new track is identified as the continuation of an old one,
        this mapping lets the caller collapse the IDs before scoring.
    """

    def __init__(self) -> None:
        self._death_log: Dict[int, dict] = {}
        self.remap: Dict[int, int] = {}          # new_oid → canonical_oid
        self._fit_cache: Dict[int, tuple] = {}   # oid → (trail_len, params, rms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        trails: Dict[int, List[Tuple[int, int]]],
        t_alphas: Dict[int, List[float]],
        full_trails: Dict[int, List[Tuple[int, int]]],
        tracker,          # tracker.Tracker instance
        frame_idx: int,
    ) -> None:
        """
        Call once per frame after tracker.update() has run.

        Mutates trails / t_alphas / full_trails in-place when merging.
        Updates self.remap with any new ID collapses.
        """
        # 1. Reap tracks that just started ghosting → log them
        for oid, track in list(tracker.tracks.items()):
            if track.ghost_count == 1 and oid not in self._death_log:
                pts = list(trails.get(oid, []))
                vel = _trail_velocity(pts)
                fit = self._get_fit(oid, pts)
                self._death_log[oid] = {
                    "frame": frame_idx,
                    "pts":   pts,
                    "vel":   vel,
                    "fit":   fit,
                }

        # 2. Expire stale death log entries
        for oid in list(self._death_log.keys()):
            if frame_idx - self._death_log[oid]["frame"] > VEL_MAX_GAP_FRAMES:
                del self._death_log[oid]
                self._fit_cache.pop(oid, None)

        # 3. For each *newly born* track (ghost_count==0, only 1–3 pts),
        #    try to match it to a recently dead track.
        new_oids = [
            oid for oid, track in tracker.tracks.items()
            if track.ghost_count == 0
            and len(trails.get(oid, [])) <= 4
            and oid not in self.remap
        ]

        for new_oid in new_oids:
            new_pts = list(trails.get(new_oid, []))
            if not new_pts:
                continue
            new_vel = _trail_velocity(new_pts)

            best_old, best_score = None, float("inf")
            for old_oid, info in self._death_log.items():
                if old_oid == new_oid:
                    continue
                old_pts = info["pts"]
                if not old_pts:
                    continue
                gap_frames = frame_idx - info["frame"]

                # — Coarse spatial gate —
                end_old   = old_pts[-1]
                start_new = new_pts[0]
                dist = math.hypot(
                    end_old[0] - start_new[0],
                    end_old[1] - start_new[1],
                )
                if dist > VEL_SPATIAL_CORRIDOR:
                    continue

                score = dist  # lower is better; refined below

                # — Parabolic arc test —
                para_pass = False
                fit_result = info.get("fit")
                if fit_result is not None:
                    params, rms = fit_result
                    if rms < PARA_FIT_ERR_MAX:
                        extrap = _extrapolate_arc(params, old_pts, PARA_EXTRAP_FRAMES)
                        if extrap:
                            min_d = _min_dist_to_polyline(
                                (float(start_new[0]), float(start_new[1])), extrap
                            )
                            if min_d < PARA_MATCH_DIST:
                                para_pass = True
                                # Reward good arc alignment
                                score = min_d * 0.5

                # — Velocity coherence test —
                vel_pass = False
                if info["vel"] is not None and new_vel is not None:
                    vel_pass = _velocity_coherent(info["vel"], new_vel, gap_frames)

                # Accept if either geometric test passes
                if not para_pass and not vel_pass:
                    continue

                if score < best_score:
                    best_score = score
                    best_old   = old_oid

            if best_old is not None:
                self._merge(
                    old_oid=best_old,
                    new_oid=new_oid,
                    trails=trails,
                    t_alphas=t_alphas,
                    full_trails=full_trails,
                    tracker=tracker,
                )

    def stitch_scored_curves(
        self,
        scored_curves: Dict[int, tuple],
    ) -> Dict[int, tuple]:
        """
        Post-hoc merge of scored trail fragments.  Supersedes the simple
        proximity loop in main.py.

        Applies three-level checks in order:
          1. Endpoint proximity gate
          2. Heading / velocity coherence
          3. Parabolic continuation (if STITCH_USE_PARABOLA)

        Returns a new dict (same type as scored_curves).
        """
        changed = True
        while changed:
            changed = False
            keys = list(scored_curves.keys())
            for i, ka in enumerate(keys):
                if ka not in scored_curves:
                    continue
                trail_a, cpts_a = scored_curves[ka]
                if len(trail_a) < 2:
                    continue

                for kb in keys[i + 1:]:
                    if kb not in scored_curves:
                        continue
                    trail_b, cpts_b = scored_curves[kb]
                    if len(trail_b) < 2:
                        continue

                    # Try both orderings; pick whichever direction passes
                    order = self._score_stitch_order(trail_a, trail_b)
                    if order is None:
                        continue

                    if order == "ab":
                        merged = trail_a + trail_b
                    else:
                        merged = trail_b + trail_a

                    keep = min(ka, kb)
                    drop = max(ka, kb)
                    scored_curves[keep] = (merged, [])
                    del scored_curves[drop]
                    print(f"  [STITCH-POST] merged scored trails {ka}+{kb} order={order}")
                    changed = True
                    break
                if changed:
                    break

        return scored_curves

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fit(
        self,
        oid: int,
        pts: List[Tuple[int, int]],
    ) -> Optional[Tuple[tuple, float]]:
        """Return cached conic fit or compute a fresh one."""
        cached = self._fit_cache.get(oid)
        if cached is not None and cached[0] == len(pts):
            return cached[1]
        result = _fit_conic(pts)
        self._fit_cache[oid] = (len(pts), result)
        return result

    def _merge(
        self,
        old_oid: int,
        new_oid: int,
        trails: Dict,
        t_alphas: Dict,
        full_trails: Dict,
        tracker,
    ) -> None:
        """
        Splice new_oid's trail onto old_oid's trail.
        Re-IDs new_oid → old_oid in all dicts and in tracker.tracks.
        """
        old_pts   = list(trails.get(old_oid, []))
        new_pts   = list(trails.get(new_oid, []))
        old_alps  = list(t_alphas.get(old_oid, []))
        new_alps  = list(t_alphas.get(new_oid, []))

        # Concatenate — pad alphas if lengths differ (shouldn't happen but be safe)
        merged_pts  = old_pts  + new_pts
        merged_alps = old_alps + new_alps

        trails[old_oid]     = merged_pts
        t_alphas[old_oid]   = merged_alps
        full_trails[old_oid] = list(full_trails.get(old_oid, [])) + \
                               list(full_trails.get(new_oid, []))

        # Remove new_oid entries
        for d in (trails, t_alphas, full_trails):
            d.pop(new_oid, None)

        # Transfer the Kalman track: move new track's state into old slot
        # so the tracker keeps tracking under the original ID.
        old_track = tracker.tracks.get(old_oid)
        new_track = tracker.tracks.get(new_oid)
        if old_track is not None and new_track is not None:
            # Adopt new track's Kalman state (it has fresher measurements)
            old_track.kf         = new_track.kf
            old_track.ghost_count = 0              # alive again
            # Remove the new track slot
            del tracker.tracks[new_oid]

        # Record remap so score-checker and downstream code can canonicalise IDs
        self.remap[new_oid] = old_oid
        # Propagate transitively: anything that pointed to new_oid now points to old_oid
        for k, v in list(self.remap.items()):
            if v == new_oid:
                self.remap[k] = old_oid

        # Evict death log for old track (it's alive again)
        self._death_log.pop(old_oid, None)
        self._fit_cache.pop(old_oid, None)
        self._fit_cache.pop(new_oid, None)

        print(
            f"  [STITCH-LIVE] merged trail {new_oid}→{old_oid} "
            f"({len(old_pts)}+{len(new_pts)} pts)"
        )

    def _score_stitch_order(
        self,
        trail_a: List[Tuple[int, int]],
        trail_b: List[Tuple[int, int]],
    ) -> Optional[str]:
        """
        Determine whether trail_a→trail_b or trail_b→trail_a is a valid
        stitch, and return "ab", "ba", or None.

        Checks (in order): proximity → heading → parabola.
        """
        end_a,   start_a = trail_a[-1], trail_a[0]
        end_b,   start_b = trail_b[-1], trail_b[0]

        d_ab = math.hypot(end_a[0] - start_b[0], end_a[1] - start_b[1])
        d_ba = math.hypot(end_b[0] - start_a[0], end_b[1] - start_a[1])

        candidates: List[Tuple[float, str]] = []
        if d_ab <= STITCH_DIST:
            candidates.append((d_ab, "ab"))
        if d_ba <= STITCH_DIST:
            candidates.append((d_ba, "ba"))

        if not candidates:
            return None

        for dist, order in sorted(candidates):
            src, dst = (trail_a, trail_b) if order == "ab" else (trail_b, trail_a)
            vel_src = _trail_velocity(src)
            vel_dst = _trail_velocity(dst)

            # Heading coherence (loose — scored trails are already filtered)
            heading_ok = True
            if vel_src is not None and vel_dst is not None:
                heading_ok = _velocity_coherent(vel_src, vel_dst, gap_frames=1)

            if not heading_ok:
                continue

            if not STITCH_USE_PARABOLA:
                return order

            # Parabolic check
            fit_result = _fit_conic(src)
            if fit_result is None:
                return order  # can't check, accept based on proximity+heading
            params, rms = fit_result
            if rms >= PARA_FIT_ERR_MAX:
                return order  # fit too noisy, fall back to proximity+heading

            extrap = _extrapolate_arc(params, src, PARA_EXTRAP_FRAMES)
            if not extrap:
                return order

            start_dst = dst[0]
            min_d = _min_dist_to_polyline(
                (float(start_dst[0]), float(start_dst[1])), extrap
            )
            if min_d < PARA_MATCH_DIST * 1.8:   # slightly looser for post-hoc
                return order

        return None


# ---------------------------------------------------------------------------
# Convenience: apply remap to any id-keyed dict
# ---------------------------------------------------------------------------

def apply_remap(d: dict, remap: Dict[int, int]) -> dict:
    """
    Return a new dict where any key that appears in remap is replaced by its
    canonical ID.  Values for the same canonical ID are merged as lists.
    """
    out: dict = {}
    for k, v in d.items():
        canon = remap.get(k, k)
        if canon in out:
            if isinstance(v, list) and isinstance(out[canon], list):
                out[canon] = out[canon] + v
            # else keep existing (first write wins for non-lists)
        else:
            out[canon] = v
    return out