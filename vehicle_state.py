"""vehicle_state.py — Canonical lookup of a vehicle's state at arbitrary time.

state_at(v, t) reads ONLY v.committed_traj / v.committed_segs / v.path.
It is independent of v.t_ref / v.vel / v.acc (runtime cache for last
fired event's phase).

Why a separate canonical lookup:
  - Runtime (v.t_ref/vel/acc) represents a SINGLE phase. A committed plan
    typically has multiple phases (accel→cruise→decel→stop). vel_at(t)
    using runtime is correct only within the current phase.
  - Different vehicles' runtimes have been advance_position'd to different
    times depending on dispatch order. Cross-vehicle queries (gap, leader
    lookahead, follower planning) need a side-effect-free, dispatch-order-
    independent answer.
  - Future-time queries (commit-end-start replan) need to look past the
    runtime's single phase. state_at(v, t_future) walks committed_traj.

Invariants assumed of committed_traj:
  - Sorted by t (strictly non-decreasing).
  - Each entry (t_i, d_i, vel_i, acc_i) marks the START of a phase with
    constant acc_i. The phase runs until (t_{i+1}, d_{i+1}, ...). Within
    the phase: vel(τ) = vel_i + acc_i * (τ - t_i), clamped at 0.
  - Last entry's phase extends forward (cruise/decel) until either:
      * decel hits vel=0 (then stays at d_stop)
      * a future _replan extends committed_traj.

Invariants assumed of committed_segs:
  - Sorted by t_enter.
  - Entry (t_enter, t_exit, seg_key, plan_dist):
      * vehicle is on seg_key during [t_enter, t_exit]
      * plan_dist = absolute cumulative distance at the START of seg_key
"""
from __future__ import annotations
from collections import namedtuple
from typing import Optional


VehicleState = namedtuple(
    'VehicleState',
    ['t', 'dist', 'vel', 'acc', 'seg_key', 'seg_offset', 'path_idx']
)
"""Snapshot of a vehicle at a specific time.

  t          — query time (echoed back)
  dist       — absolute cumulative distance
  vel        — velocity at t (>= 0, clamped)
  acc        — acceleration of the phase containing t
  seg_key    — (from_node, to_node) of segment containing t, or None
  seg_offset — distance into seg_key (0 <= seg_offset < seg_length)
  path_idx   — index of seg_key in v.path, or -1 if not found
"""


def state_at(v, t: float) -> Optional[VehicleState]:
    """Lookup v's state at time t. Pure function — no side effects.

    Returns None if v has no committed plan yet (committed_traj empty).
    """
    traj = v.committed_traj
    if not traj:
        return None

    n = len(traj)

    # ── 1. Phase lookup via binary search ────────────────────────────
    # Largest i with traj[i][0] <= t.
    if t <= traj[0][0]:
        # Query before first committed entry — return first entry's state.
        i = 0
        dt_eff = 0.0
        t_i, d_i, vel_i, acc_i = traj[0]
        vel = vel_i
        dist = d_i
        acc = acc_i
    else:
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if traj[mid][0] <= t + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        t_i, d_i, vel_i, acc_i = traj[i]
        dt = t - t_i

        # Clamp dt to within this phase. If next entry exists, that's the
        # phase boundary. (Caller asked for t > traj[i+1][0]? Then i would
        # have stepped to i+1 above. So this clamp only matters for the
        # final entry's phase, which extrapolates forward.)
        if i + 1 < n:
            dt = min(dt, traj[i + 1][0] - t_i)

        # Clamp dt at decel-to-zero point (vehicle stays parked after).
        if acc_i < -1e-12:
            dt_to_zero = -vel_i / acc_i
            dt_eff = min(dt, max(0.0, dt_to_zero))
        else:
            dt_eff = dt

        vel = max(0.0, vel_i + acc_i * dt_eff)
        dist = d_i + vel_i * dt_eff + 0.5 * acc_i * dt_eff * dt_eff
        acc = acc_i

    # ── 2. Segment lookup via binary search on t_enter ───────────────
    segs = v.committed_segs
    seg_key = None
    seg_offset = 0.0
    plan_dist_at_seg = 0.0
    if segs:
        # Largest j with segs[j][0] <= t.
        lo, hi = 0, len(segs) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if segs[mid][0] <= t + 1e-9:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if idx >= 0:
            _t_enter, _t_exit, sk, plan_dist = segs[idx]
            seg_key = sk
            plan_dist_at_seg = plan_dist
            seg_offset = max(0.0, dist - plan_dist)

    # ── 3. path_idx lookup (cheap when v.path_idx is a good hint) ────
    path_idx = -1
    if seg_key is not None:
        # Try v.path_idx first; otherwise scan forward then back.
        hint = getattr(v, 'path_idx', 0)
        path = v.path
        plen = len(path) - 1
        if 0 <= hint < plen and (path[hint], path[hint + 1]) == seg_key:
            path_idx = hint
        else:
            # Walk forward from hint
            for pi in range(max(0, hint), plen):
                if (path[pi], path[pi + 1]) == seg_key:
                    path_idx = pi
                    break
            if path_idx == -1:
                # Walk back from hint (rare — shouldn't happen for forward
                # motion, but committed_segs may include older history)
                for pi in range(min(hint, plen) - 1, -1, -1):
                    if (path[pi], path[pi + 1]) == seg_key:
                        path_idx = pi
                        break

    return VehicleState(t, dist, vel, acc, seg_key, seg_offset, path_idx)


def vel_at(v, t: float) -> float:
    """Convenience: just the velocity at t. Returns 0.0 if no plan."""
    s = state_at(v, t)
    return s.vel if s is not None else 0.0


def dist_at(v, t: float) -> float:
    """Convenience: just the cumulative distance at t. Returns 0.0 if no plan."""
    s = state_at(v, t)
    return s.dist if s is not None else 0.0
