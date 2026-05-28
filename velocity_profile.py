"""velocity_profile.py — Analytical velocity profile computation.

Ported from convoy_exact.py's exact analytical approach:
- Leader-less: waypoint + forward/backward pass + _profile_between
  (exact trapezoidal/triangular profiles between consecutive waypoints)
- Leader-following: CEILING/FREE mode with quadratic dt solving
  (exact follower trajectory against leader's committed_traj)

Inputs:
  - path geometry (segment lengths, max speeds, keys, current offset)
  - kinematic state (v0, v_max, a_max, d_max)
  - plan boundary (must stop at this distance)
  - optional leader trajectory for follower-aware planning

Outputs:
  - traj: list of (t, cum_dist, vel, acc) entries marking phase transitions.
          cum_dist is relative to the vehicle's current position (caller adds
          base offset to convert to committed_traj absolute distance).
  - c_segs: list of (t_enter, t_exit, seg_key, plan_dist) for each segment
            traversed within the plan.
"""
import math
from typing import List, Tuple, Optional

# Type aliases (purely documentation)
TrajEntry = Tuple[float, float, float, float]   # (t, cum_dist, vel, acc)
SegEntry = Tuple[float, float, Tuple[str, str], float]  # (t_enter, t_exit, key, plan_dist)


# ─── Leader trajectory interpolation ───────────────────────────────────────────

def _interp_leader(traj: List[TrajEntry], t: float) -> Tuple[float, float]:
    """Interpolate leader's (cum_dist, vel) at time t from leader.committed_traj.

    leader.committed_traj entries are (t_i, dist_i, vel_i, acc_i) where
    dist_i is leader's plan-base cumulative distance and acc_i is the
    acceleration FROM t_i until the next entry (or until v=0 if decel).

    Returns (dist_at_t, vel_at_t).
    """
    if not traj:
        return 0.0, 0.0

    # Before first entry → freeze at first
    if t <= traj[0][0]:
        return traj[0][1], traj[0][2]

    # Walk to find bracketing entry
    for i in range(len(traj)):
        ti, di, vi, ai = traj[i]
        t_next = traj[i + 1][0] if i + 1 < len(traj) else float('inf')
        if t <= t_next:
            dt = t - ti
            if ai < -1e-9 and vi > 1e-9:
                # Decelerating — clamp at v=0
                t_stop = vi / abs(ai)
                if dt >= t_stop:
                    d = di + vi * t_stop + 0.5 * ai * t_stop * t_stop
                    return d, 0.0
                v = vi + ai * dt
                d = di + vi * dt + 0.5 * ai * dt * dt
                return d, max(0.0, v)
            elif ai > 1e-9:
                v = vi + ai * dt
                d = di + vi * dt + 0.5 * ai * dt * dt
                return d, v
            else:
                # cruise
                v = vi
                d = di + vi * dt
                return d, v

    # Past last entry — extrapolate cruise from last
    last_t, last_d, last_v, _ = traj[-1]
    return last_d + last_v * max(0.0, t - last_t), last_v


# ─── Kinematic helpers (from convoy_exact) ────────────────────────────────────

def _accel_dist(v0: float, v1: float, a: float) -> float:
    """Distance to go from v0 to v1 at acceleration a.  d = (v1²-v0²)/(2a)"""
    if abs(a) < 1e-12:
        return 0.0
    return (v1 * v1 - v0 * v0) / (2.0 * a)


def _triangular_peak(v0: float, v_end: float, a_max: float, d_max: float,
                     dist: float) -> Optional[float]:
    """Peak velocity for accel(a_max) then decel(d_max) over dist.

    v_peak² = (dist + v0²/(2·a_max) + v_end²/(2·d_max)) / (1/(2·a_max) + 1/(2·d_max))
    Returns None if dist is too short.
    """
    denom = 1.0 / (2.0 * a_max) + 1.0 / (2.0 * d_max)
    numer = dist + v0 * v0 / (2.0 * a_max) + v_end * v_end / (2.0 * d_max)
    vp_sq = numer / denom
    if vp_sq < 0:
        return None
    vp = math.sqrt(vp_sq)
    if vp < max(v0, v_end) - 1e-9:
        return None
    return vp


# ─── _profile_between (from convoy_exact) ─────────────────────────────────────

def _profile_between(
    t0: float, x0: float, vi: float, vf: float,
    dist: float, v_cruise: float, a_max: float, d_max: float,
) -> Tuple[List[TrajEntry], float]:
    """Generate TrajEntry phases to go from (x0, vi) to (x0+dist, vf)
    respecting v_cruise as the segment speed limit.

    Returns (phases, t_end) where t_end is the time at end of profile.

    Handles all profile combinations:
      - vi > v_cruise (initial decel to v_cruise, then trapezoidal)
      - trapezoidal (accel + cruise + decel)
      - triangular (accel + decel, can't reach v_cruise)
      - accel only, decel only, cruise only
    """
    phases: List[TrajEntry] = []
    EPS = 1e-9
    t = t0
    x = x0
    v = vi

    # Handle vi > v_cruise: decel to v_cruise first
    if vi > v_cruise + EPS:
        d_initial_decel = (vi * vi - v_cruise * v_cruise) / (2.0 * d_max)
        t_initial_decel = (vi - v_cruise) / d_max
        phases.append((t, x, vi, -d_max))
        t += t_initial_decel
        x += d_initial_decel
        v = v_cruise
        vi = v_cruise
        dist -= d_initial_decel

    # Accel / decel distances for trapezoidal
    if v_cruise > vi + EPS:
        d_accel = (v_cruise * v_cruise - vi * vi) / (2.0 * a_max)
    else:
        d_accel = 0.0
    if v_cruise > vf + EPS:
        d_decel = (v_cruise * v_cruise - vf * vf) / (2.0 * d_max)
    else:
        d_decel = 0.0

    if d_accel + d_decel <= dist + EPS:
        # Trapezoidal (or simpler) fits
        d_cruise_dist = dist - d_accel - d_decel

        # Accel phase
        if d_accel > EPS:
            t_acc = (v_cruise - vi) / a_max
            phases.append((t, x, vi, a_max))
            t += t_acc
            x += d_accel
            v = v_cruise

        # Cruise phase
        if d_cruise_dist > EPS:
            t_cr = d_cruise_dist / v_cruise
            phases.append((t, x, v_cruise, 0.0))
            t += t_cr
            x += d_cruise_dist

        # Decel phase
        if d_decel > EPS:
            t_dec = (v_cruise - vf) / d_max
            phases.append((t, x, v_cruise, -d_max))
            t += t_dec
            x += d_decel
    else:
        # Triangular: can't reach v_cruise
        vp = _triangular_peak(vi, vf, a_max, d_max, dist)
        if vp is not None and vp >= vi - EPS and vp >= vf - EPS:
            vp = min(vp, v_cruise)
            # Accel to vp
            if vp > vi + EPS:
                t_acc = (vp - vi) / a_max
                d_acc = (vp * vp - vi * vi) / (2.0 * a_max)
                phases.append((t, x, vi, a_max))
                t += t_acc
                x += d_acc
            # Decel to vf
            if vp > vf + EPS:
                t_dec = (vp - vf) / d_max
                phases.append((t, x, vp, -d_max))
                t += t_dec
                x += (vp * vp - vf * vf) / (2.0 * d_max)
        else:
            # Direct decel (vi > vf)
            if vi > vf + EPS:
                t_dec = (vi - vf) / d_max
                phases.append((t, x, vi, -d_max))
                t += t_dec
                x = x0 + dist

    return phases, t


# ─── Waypoint construction ────────────────────────────────────────────────────

def _build_segment_waypoints(
    seg_lengths: List[float],
    seg_speeds: List[float],
    seg_offset: float,
    plan_boundary: float,
) -> List[Tuple[float, float]]:
    """Collect (d, v_limit) waypoints from path segment speed changes.

    Waypoints at segment boundaries use min of adjacent segment speeds
    (the vehicle must be at that speed or below when crossing).
    Final waypoint at plan_boundary has v=0 (mandatory stop).
    """
    waypoints: List[Tuple[float, float]] = []
    cum = -seg_offset
    for i in range(len(seg_lengths)):
        d_entry = max(0.0, cum)
        if d_entry > plan_boundary:
            break
        # At segment boundary, speed <= min of this segment and next
        if i > 0 and d_entry > 1e-9:
            v_bnd = min(seg_speeds[i], seg_speeds[i - 1])
        else:
            v_bnd = seg_speeds[i]
        waypoints.append((d_entry, v_bnd))
        cum += seg_lengths[i]
        if cum >= plan_boundary:
            break

    waypoints.append((plan_boundary, 0.0))
    return waypoints


def _merge_waypoints(
    waypoints: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Sort by distance, merge near-coincident entries by taking the min v."""
    waypoints.sort(key=lambda w: (w[0], w[1]))
    merged: List[Tuple[float, float]] = []
    for d, v in waypoints:
        if merged and abs(merged[-1][0] - d) < 1e-6:
            merged[-1] = (d, min(merged[-1][1], v))
        else:
            merged.append((d, v))
    return merged


# ─── Backward pass ─────────────────────────────────────────────────────────────

def _backward_pass(
    waypoints: List[Tuple[float, float]],
    d_max: float,
) -> List[float]:
    """Compute v_cap[i] = max speed at waypoints[i] that still permits
    decelerating to all downstream caps.

    Walks waypoints from last to first; each v_cap is the min of its own
    limit and √(v_cap[i+1]² + 2·d·gap).
    """
    n = len(waypoints)
    v_cap = [w[1] for w in waypoints]
    for i in range(n - 2, -1, -1):
        d_gap = waypoints[i + 1][0] - waypoints[i][0]
        if d_gap <= 1e-9:
            v_cap[i] = min(v_cap[i], v_cap[i + 1])
            continue
        v_allowed_sq = v_cap[i + 1] ** 2 + 2 * d_max * d_gap
        v_allowed = math.sqrt(max(0.0, v_allowed_sq))
        if v_allowed < v_cap[i]:
            v_cap[i] = v_allowed
    return v_cap


# ─── Forward pass (via _profile_between) ─────────────────────────────────────

def _forward_pass(
    waypoints: List[Tuple[float, float]],
    v_cap: List[float],
    v0: float,
    v_max: float,
    a_max: float,
    d_max: float,
    t_now: float,
    seg_speeds: List[float],
    seg_lengths: List[float],
    seg_offset: float,
) -> List[TrajEntry]:
    """Generate (t, x, v, a) phase entries from current state to plan end.

    Uses _profile_between for each waypoint interval — exact trapezoidal/
    triangular profiles that handle all speed transition cases.
    """
    # Build segment speed lookup: for a given plan-relative distance d,
    # what is the segment speed limit?
    seg_bounds: List[Tuple[float, float]] = []
    cum = -seg_offset
    for i in range(len(seg_lengths)):
        seg_bounds.append((max(0.0, cum), seg_speeds[i]))
        cum += seg_lengths[i]

    def _seg_speed_at(d: float) -> float:
        sp = v_max
        for sd, ss in seg_bounds:
            if d >= sd - 1e-6:
                sp = ss
        return sp

    # Forward pass on waypoints: cap by what's reachable from v0
    # AND by the segment speed limit within the interval. The interval
    # cap is essential: without it, v_fwd[i] could be set to the
    # kinematically-reachable value even when seg_speed in the interval
    # prohibits such acceleration (e.g. v_i-1=800, interval seg=800,
    # d_gap=250 → v_reachable=943 but actual attainable v at i is 800).
    # This mismatch makes the next interval start with vi=943 while the
    # previous interval's _profile_between output cruise-at-800, producing
    # a v-discontinuity in the emitted committed_traj entries.
    n = len(waypoints)
    v_fwd = list(v_cap)
    v_fwd[0] = min(v_fwd[0], v0)
    for i in range(1, n):
        d_gap = waypoints[i][0] - waypoints[i - 1][0]
        if d_gap <= 1e-9:
            v_fwd[i] = min(v_fwd[i], v_fwd[i - 1])
            continue
        v_reachable = math.sqrt(max(0.0, v_fwd[i - 1] ** 2 + 2 * a_max * d_gap))
        v_seg_interval = min(v_max, _seg_speed_at(
            (waypoints[i - 1][0] + waypoints[i][0]) / 2.0))
        v_fwd[i] = min(v_fwd[i], v_reachable, v_seg_interval)

    # Generate phases between consecutive waypoints
    phases: List[TrajEntry] = []
    t = t_now

    for i in range(n - 1):
        x_start = waypoints[i][0]
        x_end = waypoints[i + 1][0]
        vi = v_fwd[i]
        vf = v_fwd[i + 1]
        dist = x_end - x_start

        if dist < 1e-12:
            continue

        # v_cruise for this interval = segment speed limit at midpoint,
        # also capped by v_max
        v_cruise = min(v_max, _seg_speed_at((x_start + x_end) / 2.0))

        new_phases, t_end = _profile_between(t, x_start, vi, vf, dist,
                                             v_cruise, a_max, d_max)
        phases.extend(new_phases)
        t = t_end

    # Terminal entry
    final_x = waypoints[-1][0] if waypoints else 0.0
    final_v = v_fwd[-1] if v_fwd else 0.0
    phases.append((t, final_x, final_v, 0.0))
    return phases


# ─── Phase post-processing ────────────────────────────────────────────────────

def _dedupe_phases(phases: List[TrajEntry]) -> List[TrajEntry]:
    """Merge adjacent phases that have the same acceleration.

    Two consecutive entries with the same acc (within tolerance) represent
    the same kinematic phase — keep only the first (it marks the phase start)
    and drop intermediate re-evaluations.  The final terminal entry is always
    kept regardless.
    """
    if len(phases) <= 2:
        return phases
    out = [phases[0]]
    for i in range(1, len(phases) - 1):
        entry = phases[i]
        prev = out[-1]
        # Same acc as previous AND same acc as next → interior of a phase, skip
        nxt = phases[i + 1]
        if (abs(entry[3] - prev[3]) < 1e-6 and abs(entry[3] - nxt[3]) < 1e-6):
            continue
        # Same time as previous → replace. Use the SAME comparison form
        # as the PHASE_DONE post filter in _schedule_plan_events
        # (`t_nxt > t + 1e-9`). Float arithmetic makes `abs(b-a) < 1e-9`
        # NOT equivalent to `not (b > a + 1e-9)` at the 1e-9 boundary —
        # the latter rounds `a + 1e-9` once and a 1.0004e-9 diff fails
        # `abs < 1e-9` but also fails `b > a + 1e-9`. Aligning here lets
        # the filter and dedupe agree on which transitions are degenerate
        # (V#155 seed=99 4275_merge case: a 1.0004e-9-gap accel→decel
        # survived dedupe AND had its PHASE_DONE event dropped — runtime
        # acc stuck on the dead phase).
        if not (entry[0] > prev[0] + 1e-9):
            out[-1] = entry
            continue
        out.append(entry)
    # Always keep terminal
    out.append(phases[-1])
    return out


# ─── Segment crossing computation ─────────────────────────────────────────────

def _solve_dt_to_dist(v0: float, a: float, d: float) -> float:
    """Solve for the smallest non-negative dt such that
    v0·dt + 0.5·a·dt² = d.

    Returns inf if d is unreachable (decelerating and stops before d)."""
    if d <= 1e-9:
        return 0.0
    if abs(a) < 1e-9:
        if v0 < 1e-9:
            return float('inf')
        return d / v0
    # Quadratic: 0.5·a·dt² + v0·dt - d = 0
    A = 0.5 * a
    B = v0
    C = -d
    disc = B * B - 4 * A * C
    if disc < 0:
        return float('inf')
    sqrt_disc = math.sqrt(disc)
    if A > 0:
        dt = (-B + sqrt_disc) / (2 * A)
    else:
        # Decelerating; pick smallest positive root
        dt1 = (-B + sqrt_disc) / (2 * A)
        dt2 = (-B - sqrt_disc) / (2 * A)
        candidates = [x for x in (dt1, dt2) if x >= -1e-9]
        if not candidates:
            return float('inf')
        dt = min(candidates)
    return max(0.0, dt)


def _solve_quad_min_pos(a: float, b: float, c: float) -> float:
    """Smallest positive root of a*t² + b*t + c = 0, or inf."""
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            return float('inf')
        t = -c / b
        return t if t > 1e-12 else float('inf')
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float('inf')
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    candidates = sorted([t for t in [t1, t2] if t > 1e-12])
    return candidates[0] if candidates else float('inf')


def _build_c_segs(
    phases: List[TrajEntry],
    seg_lengths: List[float],
    seg_keys: List[Tuple[str, str]],
    seg_offset: float,
    t_now: float,
) -> List[SegEntry]:
    """Walk through phases, intersecting with segment boundaries to emit
    c_segs entries with enter/exit times."""
    if not seg_keys or not phases:
        return []

    c_segs: List[SegEntry] = []
    # The current segment is seg_keys[0]; we entered it at t_now.
    c_segs.append((t_now, float('inf'), seg_keys[0], 0.0))

    # Distance (in plan-relative coordinates) of each segment START
    seg_starts: List[float] = []
    cum = -seg_offset
    for i in range(len(seg_lengths)):
        seg_starts.append(cum)
        cum += seg_lengths[i]
    # Append the path end so we have a sentinel for the last segment's exit
    seg_starts.append(cum)

    # Iterate phase intervals
    next_seg_idx = 1   # next segment whose START we cross
    for k in range(len(phases) - 1):
        t0, x0, v0, a0 = phases[k]
        t1, x1, v1, _ = phases[k + 1]

        while next_seg_idx < len(seg_starts):
            d_seg_start = seg_starts[next_seg_idx]
            if d_seg_start < x0 - 1e-9:
                # Strictly past — defensively skip
                next_seg_idx += 1
                continue
            if d_seg_start > x1 + 1e-9:
                break    # next seg start is beyond this phase interval
            # Crossing happens at or during [t0, t1]. d_seg_start == x0
            # means the vehicle is *at* the segment boundary at phase
            # start — emit the crossing at t0. (This is the case when
            # seg_offset == seg_lengths[0]: vehicle starts at the end of
            # the first segment, about to enter the second.)
            d_into_phase = max(0.0, d_seg_start - x0)
            dt = _solve_dt_to_dist(v0, a0, d_into_phase)
            t_cross = t0 + dt
            # Close previous c_segs entry
            prev = c_segs[-1]
            c_segs[-1] = (prev[0], t_cross, prev[2], prev[3])
            # Open next c_segs entry (if a key exists)
            if next_seg_idx < len(seg_keys):
                c_segs.append((t_cross, float('inf'),
                               seg_keys[next_seg_idx], d_seg_start))
            next_seg_idx += 1

    return c_segs


# ─── Leader-following: effective segment speeds ─────────────────────────────

def _leader_state_at_dist(leader_traj, d_target):
    """Find (t, v) when leader position equals d_target in leader frame.
    Returns None if d_target is beyond leader's final position.
    """
    n = len(leader_traj)
    for i in range(n):
        t_i, d_i, v_i, a_i = leader_traj[i]
        if i + 1 < n:
            d_phase_end = leader_traj[i + 1][1]
        else:
            if a_i < -1e-9 and v_i > 1e-9:
                t_stop = v_i / abs(a_i)
                d_phase_end = d_i + v_i * t_stop + 0.5 * a_i * t_stop * t_stop
            elif v_i > 1e-9:
                d_phase_end = float('inf')
            else:
                d_phase_end = d_i

        if d_i - 1e-6 <= d_target <= d_phase_end + 1e-6:
            if abs(a_i) < 1e-9:
                if v_i < 1e-9:
                    return None
                dt = (d_target - d_i) / v_i
            else:
                A = 0.5 * a_i
                B = v_i
                C = d_i - d_target
                disc = B * B - 4 * A * C
                if disc < 0:
                    return None
                sq = math.sqrt(disc)
                dt1 = (-B + sq) / (2 * A)
                dt2 = (-B - sq) / (2 * A)
                valid = [x for x in (dt1, dt2) if x >= -1e-9]
                if not valid:
                    return None
                dt = min(valid)
            dt = max(0.0, dt)
            v_at = v_i + a_i * dt
            if v_at < -1e-6:
                continue
            return (t_i + dt, max(0.0, v_at))

    return None


def _compute_follower_trajectory(
    seg_lengths: List[float],
    seg_speeds: List[float],
    seg_offset: float,
    v0: float,
    plan_boundary: float,
    v_max: float,
    a_max: float,
    d_max: float,
    t_now: float,
    leader_traj: List[TrajEntry],
    leader_dist_offset: float,
    h_min: float,
) -> List[TrajEntry]:
    """Compute follower trajectory via effective segment speeds.

    Approach: for each position x, effective_speed(x) = min(seg_speed(x),
    leader_cap(x)), where leader_cap(x) = leader speed when leader reaches
    x + h_min in follower frame.

    Split the follower path at every segment boundary and every leader
    phase boundary. In each sub-interval, use the min (leader start, leader
    end) as the conservative cap. Build waypoints from these effective
    segments and run the standard backward/forward pass.
    """
    x_L0 = leader_traj[0][1]

    # ── Helpers ─────────────────────────────────────────────────────────
    def _seg_speed_at(x: float) -> float:
        cum = -seg_offset
        for i, sl in enumerate(seg_lengths):
            if max(0.0, cum) - 1e-6 <= x < cum + sl + 1e-6:
                return seg_speeds[i]
            cum += sl
        return v_max

    # ── Compute free trajectory first (for timing filter) ─────────────
    seg_wps_free = _build_segment_waypoints(
        seg_lengths, seg_speeds, seg_offset, plan_boundary)
    seg_wps_free = _merge_waypoints(seg_wps_free)
    if not seg_wps_free:
        return [(t_now, 0.0, v0, 0.0)]
    v_cap_free = _backward_pass(seg_wps_free, d_max)
    free_phases = _forward_pass(
        seg_wps_free, v_cap_free, v0, v_max, a_max, d_max, t_now,
        seg_speeds, seg_lengths, seg_offset)
    free_phases = _dedupe_phases(free_phases)

    def _t_free_at(x: float) -> float:
        """Earliest time free trajectory reaches position x."""
        if x <= 0:
            return t_now
        for i in range(len(free_phases) - 1):
            ti, xi, vi, ai = free_phases[i]
            ti1, xi1, _, _ = free_phases[i + 1]
            if xi - 1e-6 <= x <= xi1 + 1e-6:
                dx = x - xi
                if dx <= 1e-9:
                    return ti
                if abs(ai) < 1e-9:
                    if vi < 1e-9:
                        return float('inf')
                    return ti + dx / vi
                A = 0.5 * ai
                B = vi
                C = -dx
                disc = B * B - 4 * A * C
                if disc < 0:
                    continue
                sq = math.sqrt(disc)
                dt1 = (-B + sq) / (2 * A)
                dt2 = (-B - sq) / (2 * A)
                candidates = [d for d in (dt1, dt2) if d >= -1e-9]
                if not candidates:
                    continue
                return ti + min(candidates)
        last = free_phases[-1]
        if last[2] > 1e-6:
            return last[0] + (x - last[1]) / last[2]
        return float('inf')

    def _leader_cap_at(x: float):
        """Leader cap at position x, or None if not binding.

        Timing filter: if follower's free arrival at x is >= ceiling arrival
        time, leader has already moved past, so no cap is needed.
        """
        d_L = x + h_min - leader_dist_offset + x_L0
        # If d_L is at or before leader's start position, the ceiling was
        # here before the plan began — no constraint.
        if d_L <= x_L0 + 1.0:
            return None
        state = _leader_state_at_dist(leader_traj, d_L)
        if state is None:
            return 0.0  # leader never reaches → must stop
        t_c, v_at_tc = state
        t_F = _t_free_at(x)
        if t_F >= t_c - 1e-6:
            return None  # not binding
        return v_at_tc

    # ── Identify cruise vs accel/decel leader phases ──────────────────
    # Cruise phases produce a position-range cap (modify seg speeds in range).
    # Accel/decel phases produce two waypoints (start/end with leader speeds).
    # Leader's final stop forces follower to stop at the corresponding ceiling.
    cruise_ranges: List[Tuple[float, float, float]] = []  # (x_s, x_e, v_cap)
    accel_decel_wps: List[Tuple[float, float]] = []  # (x_ceil, v_cap)
    leader_stop_x: Optional[float] = None  # follower position where leader stops

    # Initial brake-feasibility waypoint at x = leader_pos - h_min with v=v_L_now.
    # The accel_decel_wps loop's dd=0 sample is rejected by the timing filter
    # (t_F >= t_k since leader is already at this position at t_now), but when
    # v0 > v_L_now with small gap excess the follower over-accelerates between
    # samples and breaches h_min before reaching the next accepted cap.
    #
    # Only fires when v0 > v_L_now (catch-up risk). When v0 <= v_L_now the
    # follower is slower or equal; gap doesn't close, so no cap needed. This
    # avoids over-braking when leader is accelerating from rest and follower
    # can safely accelerate alongside.
    _v_L_now = leader_traj[0][2]
    if v0 > _v_L_now + 1e-3:
        _x_init_cap = leader_dist_offset - h_min
        if 0.0 <= _x_init_cap <= plan_boundary:
            accel_decel_wps.append((_x_init_cap, _v_L_now))
        elif _x_init_cap < 0.0:
            # Already inside h_min — must stop immediately.
            accel_decel_wps.append((0.0, 0.0))

    for i in range(len(leader_traj)):
        t_i, d_i, v_i, a_i = leader_traj[i]
        if i + 1 < len(leader_traj):
            t_end, d_end, v_end, _ = leader_traj[i + 1]
        else:
            # Last entry — extrapolate
            if a_i < -1e-9 and v_i > 1e-9:
                t_stop = v_i / abs(a_i)
                d_end = d_i + v_i * t_stop + 0.5 * a_i * t_stop * t_stop
                v_end = 0.0
                t_end = t_i + t_stop
            elif v_i > 1e-9:
                # Cruise without end — extend to cover entire plan
                d_end = d_i + v_i * 1e6
                v_end = v_i
                t_end = t_i + 1e6
            else:
                # Already stopped — leader is permanently at d_i
                d_end = d_i
                v_end = 0.0
                t_end = t_i + 1e6

        x_s = leader_dist_offset + (d_i - x_L0) - h_min
        x_e = leader_dist_offset + (d_end - x_L0) - h_min

        if abs(a_i) < 1e-9 and v_i > 1e-6:
            # Cruise phase: cap effective segments in [x_s, x_e]
            x_s_clip = max(0.0, x_s)
            x_e_clip = min(plan_boundary, x_e)
            if x_e_clip > x_s_clip + 1e-6:
                cruise_ranges.append((x_s_clip, x_e_clip, v_i))
        else:
            # Accel/decel phase: sample leader's velocity curve along its
            # progress so backward pass sees the actual curve, not just
            # endpoints. Endpoint-only approximation gives a concave v_cap
            # above leader's true speed mid-phase — follower then over-
            # accelerates between waypoints and closes the gap.
            SAMPLE_STEP_MM = 250.0
            dd_total = d_end - d_i

            # FIX: Add phase-boundary waypoints (start + end velocities) when
            # v0 > v_phase_boundary (catch-up risk). The timing filter rejects
            # samples where t_F >= t_k, which is exactly the v_F > v_L
            # catch-up case where a cap is most needed (e.g., V#59↔V#152:
            # leader's decel phase end at v=245 must cap follower, but t_k is
            # in the past so filter rejects). Phase boundaries represent
            # leader's true velocity at known positions.
            #
            # The v0 > v_b guard avoids over-capping when follower is slower
            # than leader's boundary velocity (no catch-up risk; e.g., V#46
            # at v0=0 with leader V#122 starting from rest — capping at v=0
            # would freeze V#46 indefinitely though leader is about to move).
            v_e_phase = math.sqrt(max(0.0, v_i * v_i + 2 * a_i * dd_total))
            for x_b, v_b in ((x_s, v_i), (x_s + dd_total, v_e_phase)):
                if 0 <= x_b <= plan_boundary and v0 > v_b + 1e-3:
                    accel_decel_wps.append((x_b, v_b))

            if dd_total <= 1e-6:
                pass  # boundary waypoints already covered above
            else:
                n_samples = max(1, int(dd_total / SAMPLE_STEP_MM))
                for k in range(n_samples + 1):
                    dd = dd_total * k / n_samples
                    x_k = x_s + dd
                    if not (0 <= x_k <= plan_boundary):
                        continue
                    v_k_sq = v_i * v_i + 2 * a_i * dd
                    if v_k_sq < 0:
                        continue
                    v_k = math.sqrt(v_k_sq)
                    if abs(a_i) > 1e-9:
                        t_k = t_i + (v_k - v_i) / a_i
                    else:
                        t_k = t_i + (dd / v_i if v_i > 1e-6 else 0.0)
                    t_F = _t_free_at(x_k)
                    if t_F < t_k - 1e-6:
                        accel_decel_wps.append((x_k, v_k))

    # Leader's final stop position (if leader stops at finite position)
    last_t, last_d, last_v, _ = leader_traj[-1]
    if last_v < 1e-6:
        x_ceil_stop = leader_dist_offset + (last_d - x_L0) - h_min
        if 0 <= x_ceil_stop <= plan_boundary:
            leader_stop_x = x_ceil_stop

    # ── Build effective segment splits ────────────────────────────────
    # Splits = original seg boundaries + cruise range boundaries + leader stop
    splits = {0.0, plan_boundary}
    cum = -seg_offset
    for sl in seg_lengths:
        if 1e-6 < cum < plan_boundary - 1e-6:
            splits.add(cum)
        cum += sl
        if 1e-6 < cum < plan_boundary - 1e-6:
            splits.add(cum)
    for x_s, x_e, _ in cruise_ranges:
        if 1e-6 < x_s < plan_boundary - 1e-6:
            splits.add(x_s)
        if 1e-6 < x_e < plan_boundary - 1e-6:
            splits.add(x_e)
    if leader_stop_x is not None and 1e-6 < leader_stop_x < plan_boundary - 1e-6:
        splits.add(leader_stop_x)

    splits_sorted = sorted(splits)

    # ── Build effective segments with cruise caps ──────────────────────
    eff_lengths: List[float] = []
    eff_speeds: List[float] = []
    eff_plan_boundary = plan_boundary
    if leader_stop_x is not None:
        eff_plan_boundary = min(eff_plan_boundary, leader_stop_x)

    for i in range(len(splits_sorted) - 1):
        x_s = splits_sorted[i]
        x_e = splits_sorted[i + 1]
        if x_e > eff_plan_boundary + 1e-6:
            break
        length = x_e - x_s
        if length <= 1e-6:
            continue
        mid = (x_s + x_e) / 2.0
        v_seg = _seg_speed_at(mid)
        # Cruise cap: lowest cruise leader speed whose range contains mid
        # AND that's binding (free arrival earlier than leader's arrival).
        v_cruise_cap = v_max
        for cr_s, cr_e, cr_v in cruise_ranges:
            if cr_s - 1e-6 <= mid <= cr_e + 1e-6:
                # Timing filter: only binding if free would catch ceiling
                d_L = mid + h_min - leader_dist_offset + x_L0
                state = _leader_state_at_dist(leader_traj, d_L)
                if state is None:
                    continue
                t_c = state[0]
                t_F = _t_free_at(mid)
                if t_F < t_c - 1e-6:
                    v_cruise_cap = min(v_cruise_cap, cr_v)
        v_eff = min(v_seg, v_cruise_cap, v_max)
        if v_eff <= 1.0:
            eff_plan_boundary = x_s
            break
        eff_lengths.append(length)
        eff_speeds.append(v_eff)

    if not eff_lengths:
        return [(t_now, 0.0, v0, 0.0)]

    # ── Build waypoint list ───────────────────────────────────────────
    # Base waypoints from effective segments (cruise caps already merged in).
    waypoints = _build_segment_waypoints(
        eff_lengths, eff_speeds, 0.0, eff_plan_boundary)

    # Add accel/decel leader phase waypoints. Backward/forward pass will
    # interpolate naturally between consecutive caps via _profile_between.
    for x_wp, v_wp in accel_decel_wps:
        if 0 <= x_wp <= eff_plan_boundary + 1e-6:
            waypoints.append((x_wp, min(v_wp, v_max)))

    waypoints = _merge_waypoints(waypoints)
    if not waypoints:
        return [(t_now, 0.0, v0, 0.0)]

    v_cap = _backward_pass(waypoints, d_max)
    phases = _forward_pass(
        waypoints, v_cap, v0, v_max, a_max, d_max, t_now,
        eff_speeds, eff_lengths, 0.0)
    phases = _dedupe_phases(phases)

    return phases


# ─── Public entry point ───────────────────────────────────────────────────────

def compute_velocity_profile(
    seg_lengths: List[float],
    seg_speeds: List[float],
    seg_keys: List[Tuple[str, str]],
    seg_offset: float,
    v0: float,
    plan_boundary: float,
    v_max: float,
    a_max: float,
    d_max: float,
    t_now: float,
    leader_traj: Optional[List[TrajEntry]] = None,
    leader_dist_offset: float = 0.0,
    h_min: float = 1150.0,
) -> Tuple[List[TrajEntry], List[SegEntry]]:
    """Compute analytical velocity profile from current state to plan_boundary.

    Without leader: waypoint + backward/forward pass + _profile_between
    With leader: CEILING/FREE mode forward computation

    Returns:
        traj: list of (t, cum_dist_rel, vel, acc) — cum_dist is relative to
              the vehicle's current position (caller adds base offset).
        c_segs: list of (t_enter, t_exit, seg_key, plan_dist_rel) for each
                segment traversed within the plan.
    """
    if plan_boundary <= 0:
        return [(t_now, 0.0, v0, 0.0)], []
    horizon = plan_boundary

    if leader_traj:
        phases = _compute_follower_trajectory(
            seg_lengths, seg_speeds, seg_offset, v0, horizon,
            v_max, a_max, d_max, t_now,
            leader_traj, leader_dist_offset, h_min,
        )
        phases = _dedupe_phases(phases)
    else:
        # Waypoint-based exact profile (no leader)
        waypoints = _build_segment_waypoints(
            seg_lengths, seg_speeds, seg_offset, horizon)
        waypoints = _merge_waypoints(waypoints)
        if not waypoints:
            return [(t_now, 0.0, v0, 0.0)], []
        v_cap = _backward_pass(waypoints, d_max)
        phases = _forward_pass(waypoints, v_cap, v0, v_max, a_max, d_max,
                               t_now, seg_speeds, seg_lengths, seg_offset)
        phases = _dedupe_phases(phases)

    # Segment crossing times
    c_segs = _build_c_segs(phases, seg_lengths, seg_keys, seg_offset, t_now)

    return phases, c_segs
