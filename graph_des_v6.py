"""
graph_des_v6.py — Pure DES engine for OHT networks.

Architecture:
- Pure DES: events are self-sustaining, committed, and NEVER cancelled.
- Two types of follower relationships:
  1. Path follower: same-route leader's committed events guarantee safe distance.
  2. ZCU follower:  ZCU lock holder's SEG_END at exit node releases the lock.
- ZCU lock system:
  - Merge:  boundary nodes = predecessor nodes (A, B). Exit = merge node SEG_END.
  - Diverge: boundary node = diverge node. Exit = successor node SEG_END.
  - Merge+Diverge: both zones handled independently with unique lock IDs.
- No event invalidation. No _notify_followers. All events are committed trajectories.
- Rendering completely separated via query_positions(t).
"""

from __future__ import annotations
import math, heapq, collections
from typing import Dict, List, Optional, Tuple, Set

from graph_des_v5 import (
    ZCUZone, MapNode, MapSegment, GraphMap,
    random_safe_path, _interp_path,
)
from velocity_profile import compute_velocity_profile
from vehicle_state import state_at

# ── Vehicle states ────────────────────────────────────────────────────────────

IDLE    = 'IDLE'
ACCEL   = 'ACCEL'
CRUISE  = 'CRUISE'
DECEL   = 'DECEL'
STOP    = 'STOP'
LOADING = 'LOADING'

# ── Event types ───────────────────────────────────────────────────────────────

EV_START      = 'START'
EV_REPLAN     = 'REPLAN'       # re-evaluate at plan boundary (leader/dest)
EV_SEG_END    = 'SEG_END'      # reached end of current segment
EV_PHASE_DONE = 'PHASE_DONE'   # reached target speed (accel complete)
EV_STOPPED    = 'STOPPED'      # decelerated to v=0
EV_ZCU_GRANT  = 'ZCU_GRANT'   # ZCU lock released, waiting vehicle may proceed
EV_BOUNDARY   = 'BOUNDARY'    # reached braking point before plan boundary
EV_ZCU_EXIT   = 'ZCU_EXIT'    # vehicle reached planned ZCU exit node — release lock
EV_TIMEOUT    = 'TIMEOUT'     # ZCU stuck V: reroute 시도 trigger (Automod-style)

# Job dispatch events
EV_JOB_CREATE   = 'JOB_CREATE'    # system event: spawn a new job (vid=-1)
EV_LOAD_DONE    = 'LOAD_DONE'     # per-vehicle: pickup dwell complete
EV_UNLOAD_DONE  = 'UNLOAD_DONE'   # per-vehicle: drop dwell complete


# ── Tolerance / margin constants ─────────────────────────────────────────────
# Category A — float drift in discrete state derivation (position-threshold).
# These exist because advance_position derives the discrete "current segment"
# from the continuous seg_offset; without drift tolerance a vehicle at the
# analytic exit time would occasionally report seg_offset < seg_len.
SEG_CROSS_EPS = 0.01      # mm — tolerance on `seg_offset >= seg_len`

# Category B — semantic boundary: a vehicle stopped at the boundary node
# is considered to be at the END of the pre-zone segment (seg_offset =
# seg_len), NOT in the zone. The zone is entered only when the vehicle
# actually traverses into the next (zone) segment with seg_offset > 0,
# which is gated by lock acquisition. advance_position's drift tolerance
# is vel-gated (see Vehicle.advance_position) so a stopped vehicle never
# auto-crosses without an actual cross event.
#
# Historical note: this used to be enforced via a 1 mm ZCU_STOP_MARGIN
# that backed brake plans off the boundary. That margin caused the V#52
# deadlock at idle_n200_seed99_disp — when a port (dispatch dest) sat on
# a ZCU merge node, the brake target was 1 mm short of dest, the
# trajectory never produced an EV_SEG_END for that crossing, path_idx
# never advanced, dest_reached never fired, and the merge lock was held
# forever. The new vel-gated SEG_CROSS_EPS replaces the margin.
ZCU_ARRIVE_EPS = SEG_CROSS_EPS + 0.5          # mm — "at boundary" slack

# Category C — kinematic "effectively stopped / effectively no distance"
VEL_ZERO = 0.1            # mm/s — below this we treat vel as zero
DIST_ZERO = 0.1           # mm — below this we treat distance as zero

# Category D — Automod-style timeout reroute (cycle deadlock recovery)
REROUTE_TIMEOUT = 15.0    # sim seconds — ZCU stuck 후 reroute 시도까지
MAX_REROUTE_PER_TASK = 3  # 한 task 의 reroute 최대 횟수 (= 무한 reroute 방지)


# ── Kinematics helper ────────────────────────────────────────────────────────

def _time_to_boundary_during_accel(v0: float, a_acc: float, d_max: float,
                                   dist: float, v_max: float) -> float:
    """Time at which braking must begin to stop at `dist`, while accelerating.

    Vehicle is accelerating at a_acc from v0. We need to find the time t₁
    at which to switch to deceleration (-d_max) so that the vehicle stops
    exactly at distance `dist`.

    Triangular profile: accel(a_acc) for t₁, then decel(-d_max) to v=0.
      v_peak = v0 + a_acc * t₁  (capped at v_max)
      accel_dist = v0*t₁ + 0.5*a_acc*t₁²
      brake_dist = v_peak² / (2*d_max)
      accel_dist + brake_dist = dist

    Substituting v_peak = v0 + a_acc*t₁:
      v0*t₁ + 0.5*a_acc*t₁² + (v0 + a_acc*t₁)² / (2*d_max) = dist

    This is a quadratic in t₁:
      (0.5*a_acc + a_acc²/(2*d_max)) * t₁²
      + (v0 + a_acc*v0/d_max) * t₁
      + (v0²/(2*d_max) - dist) = 0

    Returns time t₁ (from now) or inf if impossible.
    """
    if dist <= 0:
        return 0.0
    # Check if already need to brake (current brake_dist >= dist)
    if v0 * v0 / (2 * d_max) >= dist:
        return 0.0

    # Quadratic coefficients: A*t² + B*t + C = 0
    A = 0.5 * a_acc + a_acc * a_acc / (2 * d_max)
    B = v0 + a_acc * v0 / d_max
    C = v0 * v0 / (2 * d_max) - dist

    disc = B * B - 4 * A * C
    if disc < 0:
        return float('inf')

    t1 = (-B + math.sqrt(disc)) / (2 * A)
    if t1 < 0:
        return 0.0

    # Check v_max cap: if v_peak would exceed v_max, use trapezoidal profile
    v_peak = v0 + a_acc * t1
    if v_peak > v_max:
        # Time to reach v_max
        t_accel = (v_max - v0) / a_acc
        d_accel = v0 * t_accel + 0.5 * a_acc * t_accel ** 2
        d_brake = v_max * v_max / (2 * d_max)
        d_cruise_needed = dist - d_accel - d_brake
        if d_cruise_needed < 0:
            # Can't reach v_max and still stop — original t1 was correct
            # but capped scenario means we brake earlier
            return t1
        # Cruise phase at v_max, then brake
        t_cruise = d_cruise_needed / v_max
        return t_accel + t_cruise

    return t1


def _time_to_speed_limit_during_accel(
        v0: float, a_acc: float, d_max: float,
        dist: float, next_spd: float, v_max: float) -> float:
    """가속 중 slow 세그먼트(next_spd)에 맞춰 제동 시작해야 하는 시각.

    가속(a_acc)으로 t초 주행 후 -d_max로 전환 시 scan_dist 후
    next_spd에 정확히 도달하는 t를 구함.

    A*t² + B*t + C = 0
      A = a_acc/2 * (1 + a_acc/d_max)
      B = v0 * (1 + a_acc/d_max)
      C = (v0² - next_spd²) / (2*d_max) - dist
    """
    if dist <= 0:
        return 0.0
    # 이미 next_spd 이하면 제동 불필요
    if v0 <= next_spd + 1e-9:
        return float('inf')
    # 현재 제동거리만으로 충분한지 확인
    brake_d = (v0 * v0 - next_spd * next_spd) / (2 * d_max)
    if brake_d >= dist:
        return 0.0

    A = 0.5 * a_acc + a_acc * a_acc / (2 * d_max)
    B = v0 + a_acc * v0 / d_max
    C = (v0 * v0 - next_spd * next_spd) / (2 * d_max) - dist

    disc = B * B - 4 * A * C
    if disc < 0:
        return float('inf')

    t1 = (-B + math.sqrt(disc)) / (2 * A)
    if t1 < 0:
        return 0.0

    # v_max 도달 여부 확인
    v_peak = v0 + a_acc * t1
    if v_peak <= v_max:
        return t1

    # v_max에서 크루즈 후 제동하는 경우
    t_accel = (v_max - v0) / a_acc
    d_accel = v0 * t_accel + 0.5 * a_acc * t_accel ** 2
    d_brake = (v_max * v_max - next_spd * next_spd) / (2 * d_max)
    d_cruise_needed = dist - d_accel - d_brake
    if d_cruise_needed < 0:
        return t1
    t_cruise = d_cruise_needed / v_max
    return t_accel + t_cruise


def _time_to_travel(v0: float, acc: float, dist: float, v_max: float) -> float:
    if dist <= 0:
        return 0.0
    if acc > 0:
        t_cap = (v_max - v0) / acc if v0 < v_max else 0.0
        d_cap = v0 * t_cap + 0.5 * acc * t_cap ** 2
        if dist <= d_cap:
            disc = v0 * v0 + 2 * acc * dist
            return (-v0 + math.sqrt(max(0, disc))) / acc if disc >= 0 else float('inf')
        else:
            return t_cap + (dist - d_cap) / v_max
    elif acc < 0:
        d_stop = v0 * v0 / (2 * abs(acc))
        if dist > d_stop:
            return float('inf')
        disc = v0 * v0 + 2 * acc * dist
        if disc < 0:
            return float('inf')
        return (-v0 + math.sqrt(disc)) / acc
    else:
        return dist / v0 if v0 > 0 else float('inf')


# ── Vehicle ───────────────────────────────────────────────────────────────────

class Vehicle:
    def __init__(self, vid: int, gmap: GraphMap, path: List[str],
                 color=(200, 200, 200)):
        self.id = vid
        self.gmap = gmap
        self.color = color

        self.path: List[str] = path
        self.path_idx: int = 0
        self.seg_offset: float = 0.0

        self.vel: float = 0.0
        self.acc: float = 0.0
        self.t_ref: float = 0.0

        self.v_max: float = 3600.0
        self.a_max: float = 500.0
        self.d_max: float = 500.0
        # Vehicle dimensions come from the map's vehicleModel JSON when
        # available (KaistTB: 1108mm). Hardcoding 750 made the engine plan
        # h_min=1150 between vehicle heads, but the map / safety layer
        # treat vehicles as 1108mm long, so heads 1149mm apart actually
        # overlap — physical collision class.
        # h_min adds 200mm safety margin past the physical length.
        _gmap_vlen = getattr(gmap, 'vehicle_length', None)
        self.length: float = float(_gmap_vlen) if _gmap_vlen else 750.0
        self.h_min: float = self.length + 200

        self.state: str = IDLE
        self.next_event_t: float = 0.0

        # Path leader (vehicle ahead whose committed trajectory blocks me).
        # leader_dist = path-distance from this vehicle's current position to
        # the leader's current/committed position. Computed by _update_leader
        # and reused by gap() so detection and distance stay consistent.
        self.leader: Optional[Vehicle] = None
        self.leader_dist: float = float('inf')

        # X marker
        self.stop_dist: Optional[float] = None
        self.stop_reason: Optional[str] = None  # 'zcu', 'leader', 'dest', None(free)
        # X marker: visualization-only, pinned at the commit horizon
        # (the last point of committed trajectory under fixed
        # constraints — bnd / dest / path_end). Excludes dynamic leader
        # caps so the marker is monotonic across replans (does not
        # retreat when a closer leader appears).
        self.x_marker_pidx: int = 0
        self.x_marker_offset: float = 0.0
        self.x_marker_node: Optional[str] = None
        # Boundary-event semantic: ZCU node to attempt at the next
        # EV_BOUNDARY firing. None ⇒ leader-only brake (no lock attempt).
        # Decoupled from x_marker_node so the visual commit horizon does
        # not get hijacked when a leader caps the physical trajectory
        # short of the boundary.
        self.next_zcu_node: Optional[str] = None

        # Destination
        self.dest_node: Optional[str] = None
        self.dest_reached: bool = False

        # Job dispatch
        self.job: Optional[object] = None      # Job from dispatch.py
        self.job_state: str = 'IDLE'           # IDLE | TO_PICKUP | LOADING | TO_DROP | UNLOADING

        # ZCU state
        self.waiting_at_zcu: Optional[str] = None   # zone lock_id if waiting
        self.passed_zcu: Set[str] = set()            # boundary nodes granted

        # Reroute (Automod-style timeout): 현재 task 의 reroute 발동 횟수.
        # MAX_REROUTE_PER_TASK 초과 시 더 이상 reroute 시도 안 함.
        # Job arrival 시 reset.
        self.reroute_count: int = 0

        # Forward path scan result from _update_leader: distance to the
        # closest STOP vehicle on v's forward path (within leader walk
        # cap). Used as an additional plan_boundary cap so that v never
        # commits past a known stationary vehicle, even when the chosen
        # leader is a different (closer, moving) vehicle that may later
        # leave v's path (cross-branch divergence).
        self.forward_stop_cap: float = float('inf')

        # Push (idle-leader wakeup) — see DES._try_push
        self.last_push_t: float = -1e9
        # True iff current path was assigned by a push (vs JobManager
        # job dispatch). Set/cleared by _assign_destination's via_push
        # arg. Pure visualization flag — not used by simulation logic.
        self.via_push: bool = False

        # Last sim_time at which a leader-notify EV_REPLAN was posted for
        # this vehicle. Used by _notify_followers to dedup: if multiple
        # leaders' plans change at the *same* sim_time, posting one
        # EV_REPLAN is enough (the replan reads all leader state freshly).
        # Without dedup, N=200 with cascading replans piles dozens of
        # identical EV_REPLANs at the same t — heap and memory blow up.
        self.last_notify_post_t: float = -1e9

        # Committed trajectory: full velocity profile for follower_plan
        # list of (t, dist_from_plan_start, vel, acc)
        self.committed_traj: List = []
        self.committed_traj_t0: float = 0.0

        # Committed segment entries: [(t_enter, seg_key, offset_in_plan), ...]
        # Which segments this vehicle will occupy as part of its committed plan.
        self.committed_segs: List = []

        # Commit horizon — path index up to which this vehicle is *physically
        # committed* to traverse. dispatch / push / reroute MUST preserve
        # path[path_idx : commit_end_idx + 1] when calling _assign_destination,
        # because:
        #   - lock acquisitions may have been made for boundaries within this range
        #   - EV_ZCU_EXIT events scheduled depend on these segments existing in path
        #   - leader-follower trajectories of trailing vehicles reference this
        # Set in _replan based on commit_horizon_dist = min(brake_dist_v_max, bnd_dist).
        # Default = path_idx (no commitment beyond current segment).
        self.commit_end_idx: int = 0
        self.commit_horizon_dist: float = 0.0
        # Time of the last event posted by the most recent
        # _schedule_plan_events call (= upper bound of committed event
        # range). Survives _trim_committed (unlike committed_traj which
        # gets future-wiped). Used to:
        #   - measure commit-invariant violations (events posted at
        #     t < commit_end_t mean a new plan is overwriting committed
        #     events that the prior plan posted)
        #   - eventually drive _schedule_plan_events to skip posting
        #     overlapping events (extend-only mode)
        self.commit_end_t: float = 0.0

        # Render cache
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0
        self.gap_to_leader: float = float('inf')

        self._seg_lengths: List[float] = []
        self._seg_speeds: List[float] = []
        self._rebuild_seg_cache()

    def _rebuild_seg_cache(self):
        self._seg_lengths = []
        self._seg_speeds = []
        for i in range(len(self.path) - 1):
            seg = self.gmap.segment_between(self.path[i], self.path[i + 1])
            if seg:
                self._seg_lengths.append(seg.length)
                self._seg_speeds.append(seg.max_speed)
            else:
                self._seg_lengths.append(0.0)
                self._seg_speeds.append(self.v_max)

    @property
    def seg_from(self) -> str:
        # When at the end of the path (path_idx == len(path)-1), report
        # the START of the LAST real segment so followers can locate this
        # vehicle via _seg_occupants on the actual last seg, not a phantom
        # self-loop.
        if self.path_idx >= len(self.path) - 1 and len(self.path) >= 2:
            return self.path[-2]
        return self.path[self.path_idx]

    @property
    def seg_to(self) -> str:
        if not self.path:
            return ''
        if self.path_idx >= len(self.path) - 1:
            return self.path[-1]
        return self.path[self.path_idx + 1]

    def current_segment(self) -> Optional[MapSegment]:
        return self.gmap.segment_between(self.seg_from, self.seg_to)

    def current_seg_length(self) -> float:
        if self.path_idx < len(self._seg_lengths):
            return self._seg_lengths[self.path_idx]
        seg = self.current_segment()
        return seg.length if seg else 0.0

    def current_seg_speed(self) -> float:
        if self.path_idx < len(self._seg_speeds):
            return self._seg_speeds[self.path_idx]
        return self.v_max

    def dist_to_seg_end(self) -> float:
        return max(0.0, self.current_seg_length() - self.seg_offset)

    def _dist_traveled(self, dt: float) -> float:
        if dt <= 0:
            return 0.0
        if self.acc > 0:
            t_cap = max(0, (self.v_max - self.vel) / self.acc) if self.vel < self.v_max else 0
            if dt <= t_cap:
                return self.vel * dt + 0.5 * self.acc * dt ** 2
            else:
                d_cap = self.vel * t_cap + 0.5 * self.acc * t_cap ** 2
                return d_cap + self.v_max * (dt - t_cap)
        elif self.acc < 0:
            t_stop = -self.vel / self.acc if self.acc != 0 else float('inf')
            if dt <= t_stop:
                return self.vel * dt + 0.5 * self.acc * dt ** 2
            else:
                return self.vel * t_stop + 0.5 * self.acc * t_stop ** 2
        else:
            return self.vel * dt

    def vel_at(self, t: float) -> float:
        dt = max(0, t - self.t_ref)
        if self.acc > 0:
            return min(self.v_max, self.vel + self.acc * dt)
        elif self.acc < 0:
            return max(0.0, self.vel + self.acc * dt)
        return self.vel

    def braking_distance(self, from_vel: float = -1) -> float:
        v = from_vel if from_vel >= 0 else self.vel
        return v * v / (2 * self.d_max) if v > 0 else 0.0

    def advance_position(self, t: float) -> List[str]:
        """Advance to time t. Returns list of nodes crossed (arrived at).

        SEG_CROSS_EPS (drift tolerance) is applied ONLY when the vehicle is
        in motion. A stopped vehicle's seg_offset is the committed brake-
        target — applying drift tolerance would auto-advance a vehicle that
        deliberately parked at a boundary, pushing it across without a lock
        acquisition (the V#52 idle_n200_seed99_disp deadlock case: brake
        plan ended at offset≈seg_len, advance_position auto-crossed into
        the locked zone, but path_idx incremented out of sync with the
        physical leader-walk view, leaving the vehicle stranded with the
        merge lock held forever).

        For stopped vehicles, the boundary node is where seg_offset ==
        seg_len of the pre-zone segment; the vehicle is logically at the
        boundary node but on the *pre-zone* segment, hence not in any
        zone yet. Lock acquisition is what actually transitions it
        physically into the next (zone) segment, via the next plan's
        EV_SEG_END at the trajectory's analytic crossing time.
        """
        dist = self._dist_traveled(t - self.t_ref)
        self.vel = self.vel_at(t)
        self.t_ref = t
        self.seg_offset += dist
        crossed = []
        while self.path_idx < len(self.path) - 1:
            seg_len = self.current_seg_length()
            if seg_len <= 0:
                self.path_idx += 1
                self.seg_offset = 0.0
                continue
            # Drift tolerance only for moving vehicles. Stopped vehicles
            # parked at seg_offset == seg_len are *at the boundary node*
            # but still on the pre-zone segment by definition (zone entry
            # requires lock, lock entry requires plan that drives into the
            # next segment). Auto-cross of a parked car into a zone it
            # never locked is a SIMULTANEOUS violation source.
            if self.vel > VEL_ZERO:
                cross = self.seg_offset >= seg_len - SEG_CROSS_EPS
            else:
                # Stopped — never auto-cross. _brake_to_stop's decel
                # gets clipped to d_max when the requested stop_dist is
                # tighter than physical reach, which makes the actual
                # kinematic d_stop overshoot seg_len by ~1mm (V#65
                # seed=12 case: stop_dist=299mm with vel=547mm/s
                # needed decel=502mm/s², clipped to 500, d_stop=300mm).
                # A naive `seg_offset > seg_len + 1e-9` then fires and
                # drifts V across an un-owned ZCU boundary. Clamp
                # seg_offset to seg_len and refuse the cross — V is
                # logically at the boundary node on the pre-zone
                # segment. Lock acquisition (granting motion past) is
                # the only way V transitions into the next zone segment.
                if self.seg_offset > seg_len:
                    self.seg_offset = seg_len
                cross = False
            if cross:
                # Hold at the end of the LAST real segment instead of
                # advancing past it; this keeps seg_from/seg_to reporting
                # the actual final segment with the correct end-offset, so
                # followers can locate this vehicle.
                if self.path_idx == len(self.path) - 2:
                    crossed.append(self.seg_from)
                    self.seg_offset = seg_len
                    break
                self.seg_offset -= seg_len
                self.path_idx += 1
                if self.seg_offset < 0:
                    self.seg_offset = 0.0
                crossed.append(self.seg_from)
            else:
                break
        return crossed

    def set_state(self, t: float) -> List[str]:
        crossed = self.advance_position(t)
        self.acc = 0.0
        return crossed

    def update_render(self, t: float):
        dist = self._dist_traveled(t - self.t_ref)
        offset = self.seg_offset + dist
        pidx = self.path_idx
        while pidx < len(self.path) - 1:
            seg_len = self._seg_lengths[pidx] if pidx < len(self._seg_lengths) else 0.0
            if seg_len <= 0:
                pidx += 1; offset = 0.0; continue
            if offset >= seg_len - SEG_CROSS_EPS and pidx < len(self.path) - 2:
                offset -= seg_len; pidx += 1
                if offset < 0: offset = 0.0
            else:
                break
        if pidx < len(self.path) - 1:
            seg = self.gmap.segment_between(self.path[pidx], self.path[pidx + 1])
            if seg and seg.path_points:
                self.x, self.y, self.theta = _interp_path(seg.path_points, max(0, offset))
                return
        nid = self.path[min(pidx, len(self.path) - 1)]
        node = self.gmap.nodes.get(nid)
        if node:
            self.x, self.y = node.x, node.y

    def needs_path_extension(self) -> bool:
        # Need enough lookahead to fit a worst-case brake from v_max down to
        # the slowest reasonable segment speed. If path tail is shorter than
        # the brake distance, the planner can't pre-decelerate in time and
        # the vehicle enters the slow segment over the limit.
        # Margin: brake distance + 30 segments slack.
        brake_d = self.v_max * self.v_max / (2 * self.d_max)   # mm
        # Approximate seg length ~ 1000mm; convert distance to seg-count.
        min_segs_ahead = int(brake_d / 800.0) + 30
        return self.path_idx >= len(self.path) - min_segs_ahead

    def extend_path(self, new_nodes: List[str]) -> bool:
        """Returns True if the path was actually extended."""
        # Compute the effective extension first; if there's nothing to add,
        # do not trim either (trimming without adding leaves the path shorter
        # than the planner expects and eventually pushes path_idx out of
        # bounds, producing a phantom self-loop seg).
        if new_nodes and new_nodes[0] == self.path[-1]:
            new_nodes = new_nodes[1:]
        if not new_nodes:
            return False
        if self.path_idx > 5:
            trim = self.path_idx - 2
            self.path = self.path[trim:]
            self.path_idx -= trim
            self._seg_lengths = self._seg_lengths[trim:]
            self._seg_speeds = self._seg_speeds[trim:]
            self.x_marker_pidx = max(-1, self.x_marker_pidx - trim)
        self.path.extend(new_nodes)
        old_len = len(self._seg_lengths)
        for i in range(old_len, len(self.path) - 1):
            seg = self.gmap.segment_between(self.path[i], self.path[i + 1])
            if seg:
                self._seg_lengths.append(seg.length)
                self._seg_speeds.append(seg.max_speed)
            else:
                self._seg_lengths.append(0.0)
                self._seg_speeds.append(self.v_max)
        return True


# ── Event ─────────────────────────────────────────────────────────────────────

class Event:
    __slots__ = ('t', 'kind', 'vid', 'data', 'seq')

    _seq_counter = 0

    def __init__(self, t: float, kind: str, vid: int, data=None):
        self.t = t
        self.kind = kind
        self.vid = vid
        self.data = data   # optional payload (e.g. lock_id for EV_ZCU_EXIT)
        Event._seq_counter += 1
        self.seq = Event._seq_counter

    def __lt__(self, other):
        if self.t != other.t:
            return self.t < other.t
        return self.seq < other.seq


# ── Pure DES Engine ───────────────────────────────────────────────────────────

class GraphDESv6:
    """Pure DES engine — all events are committed, never cancelled."""

    def __init__(self, gmap: GraphMap, rng_seed: Optional[int] = None):
        self.gmap = gmap
        self.vehicles: Dict[int, Vehicle] = {}
        self.heap: List[Event] = []
        self.sim_time: float = 0.0
        self.event_count: int = 0
        # Job dispatcher (set externally by dispatch.JobManager)
        self.job_mgr: Optional[object] = None
        # When True, vehicles with no active job extend their path via
        # random_safe_path as the tail runs out (legacy "wander forever"
        # behavior, used by the visualizer). When False (default), such
        # vehicles run out of path and stop, becoming pushable IDLEs.
        self.auto_extend_idle: bool = False
        # Optional deterministic rng for path extensions. When None, the
        # global random module is used (preserves legacy behavior).
        # Pass a seed to get reproducible runs for diagnostics/regression.
        if rng_seed is not None:
            import random as _random_mod
            self._rng = _random_mod.Random(rng_seed)
        else:
            self._rng = None

        # Segment occupancy (for gap computation & rendering)
        self._seg_occupants: Dict[Tuple[str, str], List[Vehicle]] = \
            collections.defaultdict(list)

        # ── ZCU lock system ──────────────────────────────────────────────
        # lock_id = (zone.node_id, zone.kind) to handle merge+diverge at same node
        # boundary_node → list of (zone, lock_id)  (a node can be boundary for multiple zones)
        self._boundary_to_zones: Dict[str, List[Tuple[ZCUZone, str]]] = \
            collections.defaultdict(list)
        self._zone_lock: Dict[str, Optional[Vehicle]] = {}       # lock_id → holder
        self._zone_waiters: Dict[str, List[Vehicle]] = \
            collections.defaultdict(list)                         # lock_id → waiters
        self._boundary_nodes: Set[str] = set()
        # exit_node → list of (zone, lock_id)  (to release lock on SEG_END)
        self._exit_to_zones: Dict[str, List[Tuple[ZCUZone, str]]] = \
            collections.defaultdict(list)

        # Leader → Follower reverse mapping for resume notification
        self._followers: Dict[int, Set[Vehicle]] = collections.defaultdict(set)

        # ZCU violation tracking
        self.zcu_violation_count: int = 0
        self.zcu_violation_log: list = []   # max 100: [(t, lock_id, type, detail)]


        # Speed violation tracking
        self.speed_violation_count: int = 0
        self.speed_violation_log: list = []   # max 100: [(t, vid, seg, vel, limit, excess)]

        # Push instrumentation
        self.push_count: int = 0
        self.push_log: list = []   # [(t, pusher_id, pushee_id, target_node, mode)]

        # seg → zone reverse mapping (fast entry check)
        self._seg_to_zone: Dict[Tuple[str, str], List[Tuple[object, str]]] = {}

        self._build_zcu_locks()
        self._build_corridors()
        # Corridor occupants: corridor_id -> set of vehicles currently on
        # any segment within that corridor. Updated in _update_occupancy
        # and add_vehicle. ZCU-zone segments are not in any corridor, so
        # vehicles inside a ZCU zone are not in _corridor_occupants.
        self._corridor_occupants: Dict[int, Set[Vehicle]] = \
            collections.defaultdict(set)

    def _build_zcu_locks(self):
        # Reverse map: lock_id -> set of exit nodes (used by _on_zcu_exit
        # to validate "v has just crossed the exit").
        self._lock_exit_nodes: Dict[str, Set[str]] = collections.defaultdict(set)

        for zone in self.gmap.zcu_zones:
            lock_id = f"{zone.node_id}_{zone.kind}"
            self._zone_lock[lock_id] = None

            if zone.kind == 'merge':
                # Boundary = predecessor nodes (entry points before merge)
                for seg_key in zone.all_segs():
                    pred_node = seg_key[0]
                    self._boundary_to_zones[pred_node].append((zone, lock_id))
                    self._boundary_nodes.add(pred_node)
                # Exit = merge node itself
                self._exit_to_zones[zone.node_id].append((zone, lock_id))
                self._lock_exit_nodes[lock_id].add(zone.node_id)

            elif zone.kind == 'diverge':
                # Boundary = diverge node itself
                self._boundary_to_zones[zone.node_id].append((zone, lock_id))
                self._boundary_nodes.add(zone.node_id)
                # Exit = any successor node
                for seg_key in zone.all_segs():
                    succ_node = seg_key[1]
                    self._exit_to_zones[succ_node].append((zone, lock_id))
                    self._lock_exit_nodes[lock_id].add(succ_node)

        # NOTE: merge exit = merge node only. diverge exit = successor nodes only.
        # No blanket "all successors of boundary" — _relevant_zones already
        # filters out merge locks for directions the OHT doesn't take,
        # so there's no need to release at wrong exits.

        # Build seg → zone reverse mapping
        for zone in self.gmap.zcu_zones:
            lock_id = f"{zone.node_id}_{zone.kind}"
            for seg_key in zone.all_segs():
                if seg_key not in self._seg_to_zone:
                    self._seg_to_zone[seg_key] = []
                self._seg_to_zone[seg_key].append((zone, lock_id))

        print(f"ZCU locks: {len(self._zone_lock)} zones, "
              f"{len(self._boundary_nodes)} boundary nodes, "
              f"{len(self._exit_to_zones)} exit nodes")

    def _build_corridors(self):
        """Group consecutive non-ZCU, non-branch segments into directed corridors.

        A 'control point' is a graph node where vehicles can change direction
        or must synchronize:
          - ZCU boundary node (lock acquisition)
          - ZCU exit node (lock release)
          - Branch node (multiple successors or predecessors)

        A corridor is a maximal directed path of segments such that every
        interior node has exactly one predecessor and one successor AND is
        not a ZCU node. Each corridor starts at a control point and ends at
        a control point (or a dead-end node).

        ZCU-zone segments (those that are part of a ZCU zone's segs) are NOT
        included in any corridor — they're governed by lock semantics
        instead.

        Sets self._seg_to_corridor[(from, to)] -> corridor_id and
        self._corridors[corridor_id] -> ordered list of (from, to).
        Logs summary stats. No behavior change at this phase.
        """
        # In-degree map (used to detect merge points)
        indeg = collections.defaultdict(int)
        for node, succs in self.gmap.adj.items():
            for succ in succs:
                indeg[succ] += 1

        # Control points: ZCU boundaries + ZCU exits + branch nodes
        zcu_exits = set(self._exit_to_zones.keys())
        control_points = set(self._boundary_nodes) | zcu_exits
        for node in self.gmap.nodes:
            if (indeg.get(node, 0) > 1
                    or len(self.gmap.adj.get(node, [])) > 1):
                control_points.add(node)

        self._seg_to_corridor: Dict[Tuple[str, str], int] = {}
        self._corridors: Dict[int, List[Tuple[str, str]]] = {}
        # Cumulative distance from corridor start to the START of each
        # segment. A vehicle at (seg, off) inside corridor C has corridor
        # position = self._corridor_offset[seg] + off — comparable across
        # all vehicles in the same corridor since corridors are directed
        # linear paths.
        self._corridor_offset: Dict[Tuple[str, str], float] = {}
        cid = 0

        # Walk corridors starting from each control point's outgoing edges.
        # Sorted iteration for deterministic corridor IDs across runs.
        for cp in sorted(control_points):
            for succ in self.gmap.adj.get(cp, []):
                seg = (cp, succ)
                if seg in self._seg_to_corridor:
                    continue
                # ZCU-zone segments are not part of any corridor.
                if self._seg_to_zone.get(seg):
                    continue
                corridor_segs: List[Tuple[str, str]] = []
                cur, nxt = cp, succ
                while True:
                    edge = (cur, nxt)
                    if edge in self._seg_to_corridor:
                        break
                    if self._seg_to_zone.get(edge):
                        break
                    self._seg_to_corridor[edge] = cid
                    corridor_segs.append(edge)
                    if nxt in control_points:
                        break
                    succs_n = self.gmap.adj.get(nxt, [])
                    if len(succs_n) != 1:
                        break
                    cur, nxt = nxt, succs_n[0]
                if corridor_segs:
                    self._corridors[cid] = corridor_segs
                    cum = 0.0
                    for seg in corridor_segs:
                        self._corridor_offset[seg] = cum
                        seg_obj = self.gmap.segments.get(seg)
                        cum += seg_obj.length if seg_obj else 0.0
                    cid += 1

        # Stats
        if self._corridors:
            lens = [len(c) for c in self._corridors.values()]
            total_segs_in_corridors = sum(lens)
            total_non_zcu_segs = sum(
                1 for s in self.gmap.segments
                if not self._seg_to_zone.get(s))
            print(
                f"Corridors: {len(self._corridors)} corridors, "
                f"avg {sum(lens)/len(lens):.1f} segs, "
                f"max {max(lens)} segs, min {min(lens)} segs "
                f"({total_segs_in_corridors}/{total_non_zcu_segs} "
                f"non-ZCU segs grouped)")
        else:
            print("Corridors: none built")

    # ── Vehicle management ────────────────────────────────────────────────

    def add_vehicle(self, v: Vehicle):
        self.vehicles[v.id] = v
        seg_key = (v.seg_from, v.seg_to)
        self._seg_occupants[seg_key].append(v)
        cid = self._seg_to_corridor.get(seg_key)
        if cid is not None:
            self._corridor_occupants[cid].add(v)
        # Zone-internal start guard: if v starts on a segment that lives
        # *inside* a ZCU zone, auto-acquire that zone's lock. Otherwise
        # the lock system never sees v's presence — another vehicle can
        # later acquire the same lock from a different boundary and
        # physically collide on the merge node (seed=7 V#28<->V#111
        # at 1788_merge case).
        for _zone, lock_id in self._seg_to_zone.get(seg_key, []):
            existing = self._zone_lock.get(lock_id)
            if existing is None:
                self._zone_lock[lock_id] = v
            elif existing is not v:
                # Two vehicles starting inside the same zone — setup bug.
                # Print a clear warning rather than silently overwrite.
                print(f"[WARN] add_vehicle V#{v.id} on seg {seg_key} "
                      f"in zone {lock_id} but already held by "
                      f"V#{existing.id} — likely setup placed two "
                      f"vehicles in same zone.")

    _REPLAN_PRIORITY_DELTA = 1e-9

    def _post(self, t: float, kind: str, v: Vehicle, data=None):
        if kind == EV_REPLAN:
            t = t + GraphDESv6._REPLAN_PRIORITY_DELTA
        v.next_event_t = t
        heapq.heappush(self.heap, Event(t, kind, v.id, data))

    def _post_system(self, t: float, kind: str, data=None):
        """Post a system event (no specific vehicle). vid=-1 sentinel."""
        heapq.heappush(self.heap, Event(t, kind, -1, data))

    # ── Public API ────────────────────────────────────────────────────────

    def run_until(self, t_end: float):
        while self.heap and self.heap[0].t <= t_end:
            ev = heapq.heappop(self.heap)
            # System event (vid=-1): route to job_mgr
            if ev.vid == -1:
                self.sim_time = ev.t
                self.event_count += 1
                if self.job_mgr is not None and ev.kind == EV_JOB_CREATE:
                    self.job_mgr.on_create_event(ev.t)
                continue
            v = self.vehicles.get(ev.vid)
            if v is None:
                continue
            self.sim_time = ev.t
            self.event_count += 1
            self._dispatch(ev, v)

    def query_positions(self, t: float):
        for v in self.vehicles.values():
            v.update_render(t)
            if v.leader is not None:
                g, _ = self.gap(v, t)
                v.gap_to_leader = g
            else:
                v.gap_to_leader = float('inf')

    def step(self, t_now: float):
        self.run_until(t_now)
        self.query_positions(t_now)

    def start_all(self):
        # Sort segment occupant queues by offset (highest first = furthest ahead)
        for queue in self._seg_occupants.values():
            queue.sort(key=lambda v: -v.seg_offset)
        self.assign_leaders()

        # Post START events in leader-first topological order so that each
        # follower's first plan can already see its leader's committed_traj.
        # Without this, followers planning before their leaders end up with
        # short stop plans and need extra notify-driven replans to converge.
        visited: Set[int] = set()
        order: List[Vehicle] = []

        def _visit(v: Vehicle):
            if v.id in visited:
                return
            if v.leader is not None and v.leader.id not in visited:
                _visit(v.leader)
            visited.add(v.id)
            order.append(v)

        for v in self.vehicles.values():
            _visit(v)

        for v in order:
            self._post(0.0, EV_START, v)

    # ── Event dispatch ────────────────────────────────────────────────────

    # Debug tracing (set to False for production)
    DEBUG_TRACE = False
    DEBUG_VID = 0  # vehicle ID to trace (-1 for all)
    DEBUG_VID_SET: Set[int] = set()  # if non-empty, restrict trace to these vids
    _trace_log: list = []

    def _trace(self, msg: str):
        if self.DEBUG_TRACE:
            self._trace_log.append(msg)
            print(msg)

    def _trace_match(self, vid: int) -> bool:
        """Gate for per-vehicle trace lines.

        DEBUG_VID_SET (when non-empty) takes precedence over DEBUG_VID.
        DEBUG_VID == -1 means trace all.
        """
        if not self.DEBUG_TRACE:
            return False
        if self.DEBUG_VID_SET:
            return vid in self.DEBUG_VID_SET
        return self.DEBUG_VID < 0 or vid == self.DEBUG_VID

    def _dispatch(self, ev: Event, v: Vehicle):
        if self._trace_match(ev.vid):
            self._trace(f"[EVT] t={ev.t:.4f} V#{ev.vid} {ev.kind} "
                        f"state={v.state} v={v.vel:.1f} a={v.acc:.1f} "
                        f"pidx={v.path_idx} off={v.seg_offset:.1f} "
                        f"seg={v.seg_from}->{v.seg_to}")
        if ev.kind == EV_ZCU_EXIT:
            self._on_zcu_exit(ev.t, v, ev.data)
            return
        if ev.kind == EV_BOUNDARY:
            self._on_boundary(ev.t, v, ev.data)
            return
        if ev.kind == EV_TIMEOUT:
            self._on_timeout(ev.t, v, ev.data)
            return
        handler = {
            EV_START:      self._replan,
            EV_REPLAN:     self._replan,
            EV_SEG_END:    self._on_seg_end,
            EV_PHASE_DONE: self._on_phase_done,
            EV_STOPPED:    self._on_stopped,
            EV_ZCU_GRANT:  self._on_zcu_grant,
            EV_LOAD_DONE:  self._on_load_done,
            EV_UNLOAD_DONE:self._on_unload_done,
        }.get(ev.kind)
        if handler:
            handler(ev.t, v)

    def _on_load_done(self, t: float, v: Vehicle):
        if self.job_mgr is not None:
            self.job_mgr.on_load_done(t, v)
        self._wake_lock_waiters_at_dwell_end(t, v)

    def _on_unload_done(self, t: float, v: Vehicle):
        if self.job_mgr is not None:
            self.job_mgr.on_unload_done(t, v)
        self._wake_lock_waiters_at_dwell_end(t, v)

    def _wake_lock_waiters_at_dwell_end(self, t: float, v: Vehicle) -> None:
        """Dwell이 ZCU 점유 상태에서 종료될 수 있다. 이 차량은 dwell 중에
        도 lock을 들고 있고, waiter들은 reason='zcu'로 정지해 있으며 EV_
        ZCU_GRANT은 자기 차례 lock release시까지 오지 않는다. 그 동안 다
        른 OHT가 이 dwell V를 push로 비키게 해 lock을 풀고 진행해야 할
        수 있는데 — push는 _replan tail의 _find_idle_in_forward에서 발
        견되므로, waiter에게 EV_REPLAN을 명시적으로 post해 재탐색을 트
        리거한다 (polling이 아닌 dwell-종료 1회성 wake)."""
        held = [lid for lid, h in self._zone_lock.items() if h is v]
        for lid in held:
            for waiter in list(self._zone_waiters.get(lid, ())):
                self._post(t, EV_REPLAN, waiter)

    # ── Destination assignment ──────────────────────────────────────────

    def _assign_destination(self, t: float, v: Vehicle,
                            new_path: List[str], dst_node: str,
                            via_push: bool = False) -> None:
        """Replace v.path with caller-supplied node sequence and trigger
        the standard replan cycle. Caller guarantees:
          new_path[0] == v.seg_from   (current segment from)
          new_path[1] == v.seg_to     (current segment to)
        passed_zcu is preserved so that locks held for the current
        segment are not falsely treated as "physically crossed" by
        _release_passed_diverge_locks.

        Stale-waiter cleanup: a waiter entry is a one-way pointer that the
        commit-horizon protocol does NOT cover. If the new path no longer
        traverses the awaited zone, the prior waiter would receive an
        unwanted ZCU_GRANT on the next release and silently hold a lock
        whose EV_ZCU_EXIT will never fire (V#115 / 1666_merge leak case).
        Clear all of v's waiter registrations here; the new path's
        _on_boundary will re-register if still relevant.

        Stale-lock cleanup (Phase 1 of commit-immutability fix):
        EV_ZCU_EXIT events are scheduled by _post_zcu_exit_events from
        the new plan's c_segs — but only if the exit node appears in
        the new path. When a held lock's exit is not in new_path, the
        old EV_ZCU_EXIT gets stale-skipped (plan_gen bump) and no new
        one is posted → lock is held forever (V#198 / 1567_merge case
        @ idle_n200_seed99_disp). Release such locks here, BEFORE the
        path swap, unless the vehicle is currently inside one of the
        zone's segments (releasing while in-zone would let another
        vehicle enter and produce a SIMULTANEOUS violation).
        """
        # Phase 1: release stale locks. Only for non-push reassign for
        # now — push-time release at t=859 V#49 caused notify cascade
        # storm @ idle_n200_seed99_disp (V#18/V#47/V#56 each 11k+
        # replans). Push case needs separate handling (Phase 4).
        new_path_set = set(new_path)
        cur_seg = (v.seg_from, v.seg_to)
        stale_to_release = []
        for lid, holder in list(self._zone_lock.items()):
            if holder is not v:
                continue
            exits = [n for n, zones in self._exit_to_zones.items()
                     if any(zlid == lid for _z, zlid in zones)]
            if not exits:
                continue
            if any(e in new_path_set for e in exits):
                continue
            in_zone = any(cur_seg == sk
                          for sk, zones in self._seg_to_zone.items()
                          for _z, zlid in zones if zlid == lid)
            if in_zone:
                continue
            stale_to_release.append(lid)
        for lid in stale_to_release:
            # New path doesn't traverse the zone exit, and v is outside
            # the zone (in_zone filter above). v will never trigger
            # EV_ZCU_EXIT for this lock and the _update_occupancy safety
            # net also won't fire (no exit cross). Must release explicitly.
            # _zone_release does path-aware grant — stale waiters are
            # skipped, so this can't transfer to a vehicle that doesn't
            # need the lock.
            self._zone_release(t, lid)

        # NOTE: do NOT clear waiter registrations here. Waiter state is
        # event-driven (registered by _zone_wait at boundary, removed by
        # grant or by passing the zone exit). Path change is not such an
        # event. Stale waiters are filtered at grant time in _zone_release
        # — if the next waiter's current path no longer requires the lock,
        # skip and advance to the next eligible waiter.
        # passed_zcu cleanup: boundaries the new path will cross AGAIN
        # must be re-armed for lock acquisition. The original purpose of
        # preserving passed_zcu (line 845-851 comment above) is to avoid
        # false-positive crossings on the current segment — that's still
        # respected because new_path[0] = current seg_from. But any
        # boundary forward in new_path needs lock attempt on cross;
        # leaving it in passed_zcu makes _find_first_boundary skip it →
        # plan has no EV_BOUNDARY for that node → V enters zone without
        # a lock (V#136 entered 1511_merge / 4588 case).
        # Path-swap commit metadata fix (V#194 seed=99 t=12247, V#123
        # seed=1 t=1842 cases):
        # old commit_end_idx/commit_end_t reference the OLD path's indices/
        # times. After swap, _replan's commit-alive early-exit mis-fires
        # and skips _find_first_boundary, leaving ZCU locks un-acquired.
        # Fix: truncate committed_traj/segs at v.commit_end_t (prev
        # commit's actual end time, already correct under phase 2.1
        # semantics) then swap path and reset commit metadata.
        # _replan from t_cut uses the NEW path for boundary scanning.
        #
        # Subtle: using v.commit_end_t (NOT committed_segs's t_exit) is
        # key. With phase 2.1 truncation v.commit_end_t = brake-start time,
        # which is also when subsequent ZCU lock grants typically fire.
        # Using committed_segs's segment t_exit overshoots into the future
        # — _replan(caller='boundary_lock_grant') after a mid-path lock
        # grant would then see commit still alive and skip extension,
        # losing the next boundary's lock (V#123 case).
        old_path_idx = v.path_idx
        old_cei = v.commit_end_idx
        do_truncate = (
            v.commit_end_t > t + 1e-9
            and 0 <= old_cei < len(v.path)
            and old_cei > old_path_idx)
        new_cei = max(0, old_cei - old_path_idx)
        if new_cei >= len(new_path) - 1:
            new_cei = max(0, len(new_path) - 2)

        if do_truncate:
            self._truncate_commit_at(v, v.commit_end_t)

        new_fwd_nodes = set(new_path[1:])
        v.passed_zcu = {n for n in v.passed_zcu if n not in new_fwd_nodes}
        v.path = new_path
        v.path_idx = 0
        v.dest_node = dst_node
        v.dest_reached = False
        v.via_push = via_push
        v._rebuild_seg_cache()
        if do_truncate:
            v.commit_end_idx = new_cei
            v.next_zcu_node = None   # force fresh boundary scan
        # Caller (dispatch._reroute / push) preserves the committed prefix
        # in new_path. Truncation above ensures _replan reads commit-state
        # consistent with the new path. When not truncating (commit empty
        # or unmappable), fall through to legacy behavior — _replan reads
        # the existing committed_traj/segs and extends from there.
        self._replan(t, v, caller='assign_dest')
        # v's path just changed — vehicles registered as v's followers may
        # now have v on a completely different branch (push/dispatch can
        # move v off their forward path entirely). Without a wake, they
        # keep v as their leader forever (stale leader → STOP/leader with
        # leader >> leader_walk_cap away, V#196/V#178 case). Targeted
        # notify on the existing _followers registration lets each
        # follower replan and re-pick its real leader via _update_leader.
        self._notify_followers(t, v)

    # ── Push (wake idle leader so pusher isn't blocked) ────────────────

    PUSH_PTP_MARGIN    = 100.0    # mm — minimum clearance past diverge so
                                  # pusher's cross-branch peek doesn't latch
                                  # onto the pushee
    PUSH_CASCADE_MAX   = 8
    PUSH_COOLDOWN_S    = 0.5      # sec — skip Case-1 PTP recompute if pushee
                                  # was just pushed; avoids _compute_ptp
                                  # (BFS up to 1500 nodes) duplicate cost
                                  # under congestion. Pushee replan is in
                                  # flight; result will be visible to next
                                  # pusher after cooldown expires.

    def _is_push_target(self, v: Vehicle) -> bool:
        """이 차량이 PTP 를 받아 출발할 수 있는 cascade 의 *근원* 인가.

        Push 의 의미: path 끝에 정지한 OHT (다른 차량의 진행을 막고
        있는 IDLE) 를 비키게 함. lock 대기/leader 정지는 lock 시스템/
        leader 해소로 자연 진행 가능하므로 push 대상 아님. 단순 waiter
        (stop_reason='zcu') 를 push 로 강제 reroute 하면 path 가 일시
        적으로 lock zone 을 우회 → grant 시점에 path-aware grant 가
        stale 로 판정하여 waiter 자격을 영구히 잃음 (V#9 1181_merge
        케이스). lock holder 가 zone 안에서 stuck 인 경우 (V#11) 도
        push 가 아니라 lock waiter chain / deadlock detection 으로
        해소되어야 한다.
        """
        if v.state != STOP:                              return False
        if v.stop_reason not in (None, 'dest'):          return False
        if v.job is not None and v.job_state in ('LOADING', 'UNLOADING'):
            return False
        return True

    def _is_chain_traversable(self, v: Vehicle) -> bool:
        """이 차량이 cascade 가 *통과* 할 수 있는 중간 노드인가.
        STOP 인 차량은 모두 통과 가능 (leader/zcu/dest/None 무관) —
        외부 force 없이는 안 움직이므로, 그 leader 의 leader 를 풀면 본
        차량도 _notify_followers 경로로 자연 wake 된다."""
        return v.state == STOP

    def _find_idle_in_forward(self, v: Vehicle):
        """First push-target vehicle on v's forward path. Used to drive
        cascade. Returns the vehicle or None.

        No segment cap — chain root may be far ahead, and bounding the
        search drops cascade for distant chains. Cascade depth is
        independently bounded by PUSH_CASCADE_MAX in _try_push."""
        for i in range(v.path_idx + 1, len(v.path) - 1):
            seg = (v.path[i], v.path[i + 1])
            for occ in self._seg_occupants.get(seg, []):
                if occ is v: continue
                if self._is_push_target(occ):
                    return occ
                # Non-target occupant — chain breaks here regardless
                return None
        return None

    def _compute_ptp(self, pushee: Vehicle, pusher: Vehicle):
        """Return (new_path, target_node) or None.

        Policy: pushee 가 pusher 와 *절대 마주치지 않을* 가장 가까운 위치.
          1) Diverge bypass — pushee 의 forward 를 따라 walk, 첫 diverge
             에서 pusher_fwd 에 없는 alt edge 선택. alt 후 h_min + margin
             거리 노드를 target. 거리 보정으로 cross-branch peek 안전.
          2) Fallback — diverge bypass 못 찾으면 BFS 로 pusher_fwd 와
             분리된 + ZCU-clear 노드. 이 경우 거리 보정 없이 첫 자격 노드.
        둘 다 segment 수 cap 없음. pusher_fwd 는 *전체* forward path.

        Commit prefix preservation: when pushee has a committed plan that
        extends past its current segment (commit_end_idx > path_idx),
        start the walk from commit_end_node so the new path begins where
        the committed motion finishes. Caller prepends the committed
        prefix so committed_traj/segs/SEG_END events remain consistent.
        Mirrors dispatch._reroute's pattern.
        """
        from collections import deque
        # Walk-start segment = (commit_end_idx - 1, commit_end_idx) when
        # pushee has prefix to preserve, else current segment.
        if pushee.commit_end_idx > pushee.path_idx and pushee.commit_end_idx < len(pushee.path) - 1:
            walk_from_idx = pushee.commit_end_idx
            seg_from = pushee.path[pushee.commit_end_idx - 1]
            seg_to   = pushee.path[pushee.commit_end_idx]
        else:
            walk_from_idx = pushee.path_idx + 1
            seg_from = pushee.seg_from
            seg_to   = pushee.seg_to
        # pusher 전체 forward path — cap 없음 (반복 push 방지)
        pusher_fwd = set(pusher.path[pusher.path_idx:])

        # ── Stage 1: Diverge bypass (cap 없음) ──────────────────────
        chain = [seg_from, seg_to]
        visited_walk = {seg_from, seg_to}
        cur = seg_to
        WALK_CAP = 1500    # graph nodes < 4000, 안전 상한
        for _ in range(WALK_CAP):
            succ = self.gmap.adj.get(cur, [])
            if not succ: break
            if cur in self.gmap.diverge_nodes and len(succ) >= 2:
                # Exclude spurs (dead-end pendants like 10001) so push targets
                # never land on a U-turn detour.
                spurs = getattr(self.gmap, 'spur_nodes', set())
                alts = [s for s in succ if s not in pusher_fwd
                                       and s not in visited_walk
                                       and s not in spurs]
                if alts:
                    alt = alts[0]
                    tail = random_safe_path(self.gmap, alt, length=20,
                                             rng=self._rng)
                    new_path = chain + [alt] + tail[1:]
                    target = self._node_at_dist_along(
                        new_path, len(chain),
                        pushee.h_min + self.PUSH_PTP_MARGIN,
                        avoid_zcu_interior=True)
                    if target is not None:
                        # Fix A: truncate path at target so v.dest_node ==
                        # path[-1]. Without this, the 20-step lookahead in
                        # random_safe_path extends path past the push target,
                        # breaking the dest_node-is-path-end invariant and
                        # causing _update_leader walks to traverse irrelevant
                        # segments (V#100 stale-leader case).
                        ti = new_path.index(target)
                        new_path = new_path[:ti + 1]
                        return new_path, target
                    # alt 는 찾았지만 거리 보정 실패 (모두 zone 내부) →
                    # 다음 forward step 에서 다시 시도
            nxt = succ[0]
            if nxt in visited_walk: break
            chain.append(nxt); visited_walk.add(nxt); cur = nxt

        # ── Stage 2: BFS fallback ──────────────────────────────────
        # diverge bypass 못 찾았을 때. pusher_fwd 와 분리된 + ZCU-clear
        # 노드 중 hop 거리 최소. 거리 보정 없음 — fallback 의 의미가
        # "어디든 pusher path 밖으로" 이므로 정밀도 양보.
        start = seg_to
        prev_node = seg_from
        parent: Dict[str, str] = {start: prev_node}
        queue = deque([start])
        target = None
        BFS_CAP = 1500
        explored = 0
        while queue and explored < BFS_CAP:
            cur_n = queue.popleft()
            explored += 1
            par = parent[cur_n]
            if cur_n != start:
                in_prev_zone = bool(self._seg_to_zone.get((par, cur_n)))
                if (cur_n not in pusher_fwd
                        and cur_n not in self._boundary_nodes
                        and not in_prev_zone):
                    target = cur_n
                    break
            for nb in self.gmap.adj.get(cur_n, []):
                if nb not in parent:
                    parent[nb] = cur_n
                    queue.append(nb)
        if target is None:
            return None
        rev = [target]
        while rev[-1] != start:
            rev.append(parent[rev[-1]])
        rev.append(prev_node)
        new_path = list(reversed(rev))
        return new_path, target

    def _node_at_dist_along(self, path, start_idx: int,
                            target_dist: float,
                            avoid_zcu_interior: bool = False):
        """Walk path from path[start_idx] forward; return first node whose
        cumulative segment distance from path[start_idx] reaches target_dist.

        avoid_zcu_interior=True: do NOT stop on a ZCU boundary node or while
        the *next* segment is still inside a ZCU zone. Pushees are otherwise
        prone to landing on ZCU exit nodes, holding the lock or blocking
        the post-ZCU segment for following traffic. The walk continues
        until landing on a node whose entering AND exiting segments are
        ZCU-free.

        None if path too short to satisfy either constraint."""
        d = 0.0
        for i in range(start_idx, len(path) - 1):
            seg = self.gmap.segment_between(path[i], path[i + 1])
            if seg is None: return None
            d += seg.length
            if d < target_dist:
                continue
            cand = path[i + 1]
            if not avoid_zcu_interior:
                return cand
            # Reject if cand is a ZCU boundary, OR if the segment we just
            # traversed (incoming to cand) is inside a ZCU zone, OR if the
            # next segment (outgoing from cand) is inside a ZCU zone —
            # any of these means cand is mid-ZCU.
            in_zcu = bool(self._seg_to_zone.get((path[i], path[i + 1])))
            if (cand not in self._boundary_nodes
                    and not in_zcu):
                # also check the next outgoing segment if we have one
                if i + 2 < len(path):
                    nxt_in_zcu = bool(self._seg_to_zone.get(
                        (path[i + 1], path[i + 2])))
                    if not nxt_in_zcu:
                        return cand
                else:
                    return cand
        return None

    def _push_or_schedule_for_idle_leader(self, t: float,
                                            v: Vehicle) -> None:
        """v just stopped because of leader OR ZCU lock. Try to push.
        - leader: 그 leader 가 push target 이거나 chain traversable 이면
          _try_push 가 chain 을 따라 근원 IDLE 까지 거슬러 올라가 push.
        - zcu: lock 대기 중. leader 가 None 일 수 있고, 있다면 그 chain
          끝의 IDLE 도 풀어야 자기가 풀린다. _try_push 가 정합성 검사."""
        if v.stop_reason not in ('leader', 'zcu'): return
        L = v.leader
        if L is None: return
        self._try_push(t, pusher=v, pushee=L)

    def _try_push_lock_holder(self, t: float, waiter: Vehicle,
                               lock_id: str) -> bool:
        """waiter just registered as a waiter on lock_id. _try_push 가
        holder 가 push target 인 경우와 chain traversable 인 경우 (STOP/
        leader holder 의 leader chain 을 따라 근원 IDLE 까지 거슬러
        올라가는 경우) 를 모두 처리한다."""
        holder = self._zone_lock.get(lock_id)
        if holder is None or holder is waiter: return False
        return self._try_push(t, pusher=waiter, pushee=holder)

    def _try_push(self, t: float, pusher: Vehicle, pushee: Vehicle,
                  depth: int = 0) -> bool:
        """3-case cascade:
          Case 1) pushee 가 push target → 정규 cascade. forward IDLE 먼저
                  (front-of-chain wakes earliest) 처리 후 자신의 PTP 산출.
          Case 2) pushee 가 chain traversable (STOP/leader, STOP/zcu, …)
                  → PTP 부여 대상은 *그 leader* 로 재귀 진입. traverse
                  노드 자체는 leader 가 풀리면 _notify_followers 로 wake.
          Case 3) 둘 다 아님 (주행 중 / job 처리 중) → 거절.

        Cooldown (Case 1 only): skip PTP recompute if pushee was pushed
        within PUSH_COOLDOWN_S — its replan is in flight, the new path is
        about to land. Repeat _compute_ptp here is wasted BFS work under
        congestion. Case 2 (chain traversal) is unaffected: chain hops are
        cheap and we still want to walk to the IDLE root.
        """
        if depth >= self.PUSH_CASCADE_MAX:                    return False

        # ── Case 1: push target ────────────────────────────────────
        if self._is_push_target(pushee):
            if t - pushee.last_push_t < self.PUSH_COOLDOWN_S:
                return False
            fwd = self._find_idle_in_forward(pushee)
            if fwd is not None and fwd is not pusher:
                self._try_push(t, pusher, fwd, depth + 1)

            result = self._compute_ptp(pushee, pusher)
            if result is None: return False
            ptp_path, target_node = result
            # Prepend the committed prefix so the path continues from
            # where pushee's committed motion finishes. ptp_path starts
            # with [path[commit_end_idx-1], path[commit_end_idx], ...]
            # (the walk-start segment), so prefix = path[path_idx :
            # commit_end_idx] ends at path[commit_end_idx-1] = ptp_path[0].
            # Drop ptp_path[0] to avoid duplicating that node.
            if (pushee.commit_end_idx > pushee.path_idx
                    and pushee.commit_end_idx < len(pushee.path) - 1):
                prefix = list(pushee.path[pushee.path_idx
                                          : pushee.commit_end_idx])
                new_path = prefix + ptp_path[1:]
            else:
                new_path = ptp_path
            pushee.last_push_t = t
            self.push_count += 1
            self.push_log.append((t, pusher.id, pushee.id, target_node))
            self._assign_destination(t, pushee, new_path, target_node,
                                     via_push=True)
            return True

        # ── Case 2: chain traversable → leader 로 재귀 ─────────────
        if self._is_chain_traversable(pushee):
            next_target = pushee.leader
            if next_target is None:                           return False
            if next_target is pusher:                         return False
            if next_target is pushee:                         return False  # self-leader 방어
            return self._try_push(t, pusher, next_target, depth + 1)

        # ── Case 3: 주행 중 / job 처리 중 ──────────────────────────
        return False

    # ── Crossed-node processing ─────────────────────────────────────────

    def _process_crossed_nodes(self, t: float, v: Vehicle, crossed: List[str]):
        """Clear passed_zcu cleanup for all crossed nodes.

        ZCU exit lock release is handled by EV_ZCU_EXIT events posted at
        plan time (see _post_zcu_exit_events). Node crossing is no longer
        a state-change event in this engine.
        """
        for node in crossed:
            v.passed_zcu.discard(node)

    def _on_zcu_exit(self, t: float, v: Vehicle, data):
        """Vehicle reached planned ZCU exit node — release the lock.

        data is the lock_id (str). Multiple ZCU_EXIT events for the same
        lock may accumulate when successive _replans re-post (each
        _post_zcu_exit_events posts without canceling prior). Validate
        at fire time that v has actually JUST crossed this lock's exit
        node — otherwise this is a stale event from an obsolete plan.

        Exit-node criterion: v.seg_from is the most recently crossed
        node. If it equals an exit node of this zone (= a key in
        _exit_to_zones containing this lock_id), v has just crossed.

        For merge zones: exit_node = merge node itself.
        For diverge zones: exit_node ∈ {successors of diverge}.
        Both subsume the prior "still inside zone" check (when v is
        inside, seg_from is the diverge node OR the pre-merge predecessor,
        neither of which is in exit_to_zones for this lock).
        """
        lock_id = data
        if self._zone_lock.get(lock_id) is not v:
            return  # not holder (or stale double-release)
        # Advance runtime to current time so seg_from/seg_to reflect
        # actual position at event firing.
        old_key = (v.seg_from, v.seg_to)
        crossed = v.advance_position(t)
        new_key = (v.seg_from, v.seg_to)
        if old_key != new_key:
            # Retrospective NO_LOCK validation MUST run before
            # _update_occupancy. _update_occupancy's safety-net
            # (line ~3301) releases any held lock whose exit was in
            # the crossed_set — for a multi-seg jump that enters and
            # exits the zone in a single advance (V#35 seed=99 1643→
            # 1644→exit case under reroute), the release happens
            # before mid_key checks run. Those mid_keys point at
            # zone segments that V *did* hold the lock for while
            # traversing, but the post-release holder check fires a
            # false NO_LOCK. Verifying first with the still-held
            # state captures the correct invariant; the actual
            # release lands a few lines below.
            for k in range(len(crossed) - 1):
                mid_key = (crossed[k], crossed[k + 1])
                self._check_zcu_entry(t, v, seg_key=mid_key)
            self._check_zcu_entry(t, v, seg_key=new_key)
            self._update_occupancy(v, old_key, new_key, t, crossed=crossed)
            # passed_zcu cleanup: every crossed boundary node must be
            # discarded so a later re-occurrence of the same node along
            # the path re-arms lock acquisition. Without this, an
            # advance done inside _on_zcu_exit silently keeps the prior
            # boundary node in passed_zcu — V#7 seed=300 1500 case:
            # passed_zcu retained 1500 from the first acquire, and
            # _find_first_boundary skipped 1500 forever, allowing a
            # silent merge entry at the second occurrence of 1500 going
            # into 4381.
            self._process_crossed_nodes(t, v, crossed)
        # Validate: v.seg_from must be an exit node of this lock's zone.
        exit_nodes_for_lock = self._lock_exit_nodes.get(lock_id)
        if not exit_nodes_for_lock or v.seg_from not in exit_nodes_for_lock:
            return  # stale event — v hasn't crossed exit (or moved past)
        self._zone_release(t, lock_id)

    # ── ZCU violation detection ──────────────────────────────────────────

    def _check_zcu_entry(self, t: float, v: Vehicle, seg_key: tuple = None):
        """Check for ZCU violations when vehicle enters a new segment.

        seg_key: explicit segment to check (for multi-seg jumps); defaults
        to v's current segment.
        """
        if seg_key is None:
            seg_key = (v.seg_from, v.seg_to)
        zones_here = self._seg_to_zone.get(seg_key, [])
        if not zones_here:
            return

        for zone, lock_id in zones_here:
            holder = self._zone_lock.get(lock_id)

            # Type 1: entered zone segment without holding the lock — hard
            # invariant violation. Under the new event design (EV_BOUNDARY
            # carries bnd_node payload, posted only when plan reaches the
            # boundary), every zone entry must pass through a successful
            # _try_acquire_all_zones. Reaching here means a silent cross
            # slipped past acquire — fail fast so the regression is
            # caught immediately rather than producing a stale-collision
            # trace much later.
            if holder is not v:
                detail = (f"V#{v.id} entered {seg_key} "
                          f"lock={lock_id} holder={'V#'+str(holder.id) if holder else 'None'} "
                          f"waiters={[w.id for w in self._zone_waiters.get(lock_id,[])]}")
                self._log_zcu_violation(t, lock_id, 'NO_LOCK', detail)
                # log-only — physical collision 은 gap_violation 으로 잡음.

            # Type 2: multiple vehicles inside the same zone
            inside = []
            for s in zone.all_segs():
                for other in self._seg_occupants.get(s, []):
                    if other not in inside:
                        inside.append(other)
            if len(inside) > 1:
                detail = (f"zone={lock_id} "
                          f"vehicles={sorted(x.id for x in inside)} "
                          f"holder={'V#'+str(holder.id) if holder else 'None'}")
                self._log_zcu_violation(t, lock_id, 'SIMULTANEOUS', detail)

    def _log_zcu_violation(self, t: float, lock_id: str,
                           vtype: str, detail: str):
        self.zcu_violation_count += 1
        if len(self.zcu_violation_log) < 100:
            self.zcu_violation_log.append((t, lock_id, vtype, detail))
        if self.zcu_violation_count <= 10:
            print(f"[ZCU VIOLATION t={t:.3f}] {vtype}: {detail}")

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_seg_end(self, t: float, v: Vehicle):
        """Segment boundary crossed — occupancy/exit only, NO replan."""
        old_key = (v.seg_from, v.seg_to)
        crossed = v.advance_position(t)
        new_key = (v.seg_from, v.seg_to)

        # ── 속도제한 위반 감지 ──────────────────────────────────
        new_limit = v.current_seg_speed()
        if v.vel > new_limit + 100:   # 100mm/s 허용 오차
            excess = v.vel - new_limit
            lookahead_v, _ = self._lookahead_speed(v)
            path_remaining = len(v.path) - v.path_idx
            max_look = v.v_max * v.v_max / (2 * v.d_max) + 2000

            self.speed_violation_count += 1
            entry = (t, v.id, new_key, v.vel, new_limit, excess,
                     lookahead_v, path_remaining)
            if len(self.speed_violation_log) < 100:
                self.speed_violation_log.append(entry)
            if self.speed_violation_count <= 20:
                print(f"[SPEED_VIOL t={t:.2f}] V#{v.id} "
                      f"entered {new_key[0]}->{new_key[1]} "
                      f"vel={v.vel:.0f} limit={new_limit:.0f} excess={excess:.0f}mm/s "
                      f"| lookahead_v={lookahead_v:.0f} "
                      f"path_remaining={path_remaining} "
                      f"max_look={max_look:.0f}mm")

        # advance_position이 한 번에 여러 seg를 건너면 중간 seg에서도
        # occupancy를 정리해야 한다. crossed 리스트에는 지나친 노드들이
        # 들어 있고, 이를 이용해 (crossed[k], crossed[k+1]) 쌍으로 중간
        # seg key를 복원해 제거한다. 마지막은 (last_crossed, new_seg_to)
        # = old_key의 인접한 다음 seg부터 new_key 직전까지.
        if len(crossed) > 1:
            # 모든 중간 seg에서 v를 제거 (old_key는 _update_occupancy가 처리)
            for k in range(len(crossed) - 1):
                mid_key = (crossed[k], crossed[k + 1])
                if v in self._seg_occupants[mid_key]:
                    self._seg_occupants[mid_key].remove(v)
                # Diagnostic: this segment was traversed mid-jump.
                self._check_zcu_entry(t, v, seg_key=mid_key)
            # 마지막 crossed 노드와 new_key의 from_node를 연결하는 seg
            last_crossed = crossed[-1]
            if last_crossed != new_key[0]:
                gap_key = (last_crossed, new_key[0])
                if v in self._seg_occupants[gap_key]:
                    self._seg_occupants[gap_key].remove(v)
                self._check_zcu_entry(t, v, seg_key=gap_key)

        self._update_occupancy(v, old_key, new_key, t, crossed=crossed)
        self._check_zcu_entry(t, v)

        self._process_crossed_nodes(t, v, crossed)

        # (C) Automod decelerate_ok-style: every node is a control point.
        # On EVERY node crossing (not just ZCU boundary), re-evaluate the
        # leader. If it changed, replan from this new vantage so the next
        # commit window picks up an emerging closer leader. Bounded cost:
        # replan only triggered when leader identity actually changes.
        # Replaces the prior ZCU-only check that left leader-change
        # detection lagged across intermediate non-ZCU nodes (V#106/V#126
        # seed=99 t=1207 root cause: V#126 became visible only AFTER V#106
        # had already committed to a fast accel through an intermediate
        # node).
        if crossed:
            old_leader = v.leader
            self._update_leader(v, t)
            # Refresh STOPped followers' leader pointer when v just crossed
            # a ZCU boundary or exit node. STOPped followers never run
            # _replan on their own when their leader drifts away (V#172
            # case: STOP at path-end, no events fire even though leader v
            # was notified-posting EV_REPLAN — _replan early-exits before
            # _update_leader). Restricting to STOPped followers keeps the
            # _update_leader caller contract satisfied (vel=0 ⇒ seg_offset
            # is current without an advance_position call, which would
            # otherwise change seg_from/to outside _update_occupancy and
            # desync ZCU lock state).
            crossed_control = any(
                n in self._boundary_nodes or n in self._exit_to_zones
                for n in crossed)
            if crossed_control:
                for f in list(self._followers.get(v.id, ())):
                    if f.state == STOP:
                        old_f_leader = f.leader
                        self._update_leader(f, t)
                        # If the relationship changed (esp. cleared to
                        # None), the STOPped follower's stop_reason and
                        # plan are stale — post EV_REPLAN so it can
                        # re-evaluate and resume motion (V#113 phantom-
                        # stop case: leader=None after refresh but
                        # reason='leader' kept STOP frozen).
                        if f.leader is not old_f_leader:
                            self._post(t, EV_REPLAN, f)
            if v.leader is not old_leader:
                self._replan(t, v, skip_set_state=True,
                             caller='seg_end_leader_change')
                return

        # NO acc sync here — acc transitions are now driven solely by EV_PHASE_DONE
        # events bulk-posted from committed_traj. NO ZCU exit release here either —
        # EV_ZCU_EXIT events handle that.

        # NOTE: path extension is handled exclusively in _replan (line ~942).
        # SEG_END is only for occupancy + speed-limit checks; extending here
        # would either duplicate work (extension is redundant before next
        # replan) or trigger spurious replans that interrupt the current plan
        # mid-execution. The replan at the next BOUNDARY/STOPPED naturally
        # picks up the extension when the path tail actually shrinks below
        # threshold.

    def _advance_with_occupancy(self, v: Vehicle, t: float) -> List[str]:
        """Wrapper for v.advance_position that also reconciles seg_occupants
        for any segments crossed during the advance. Returns the crossed
        node list (same as advance_position)."""
        old_key = (v.seg_from, v.seg_to)
        crossed = v.advance_position(t)
        new_key = (v.seg_from, v.seg_to)
        if old_key != new_key:
            # Remove from any intermediate segs (multi-seg jump)
            if len(crossed) > 1:
                for k in range(len(crossed) - 1):
                    mid_key = (crossed[k], crossed[k + 1])
                    if v in self._seg_occupants[mid_key]:
                        self._seg_occupants[mid_key].remove(v)
                    # Diagnostic: this segment was traversed mid-jump,
                    # so a violation here would otherwise go uncounted.
                    self._check_zcu_entry(t, v, seg_key=mid_key)
                last_crossed = crossed[-1]
                if last_crossed != new_key[0]:
                    gap_key = (last_crossed, new_key[0])
                    if v in self._seg_occupants[gap_key]:
                        self._seg_occupants[gap_key].remove(v)
                    self._check_zcu_entry(t, v, seg_key=gap_key)
            self._update_occupancy(v, old_key, new_key, t, crossed=crossed)
            self._check_zcu_entry(t, v, seg_key=new_key)
        return crossed

    def _on_phase_done(self, t: float, v: Vehicle):
        """Kinematic phase complete — sync acc from committed_traj, NO replan."""
        crossed = self._advance_with_occupancy(v, t)
        new_acc = self._lookup_committed_acc(v, t)
        v.acc = new_acc if new_acc is not None else 0.0
        if v.acc > 0.001:
            v.state = ACCEL
        elif v.acc < -0.001:
            v.state = DECEL
        else:
            v.state = CRUISE
        self._process_crossed_nodes(t, v, crossed)
        # No replan — BOUNDARY or next phase events are already in the heap.

    def _on_stopped(self, t: float, v: Vehicle):
        crossed = v.set_state(t)
        v.vel = 0.0
        v.acc = 0.0
        v.state = STOP
        self._push_or_schedule_for_idle_leader(t, v)
        # V just confirmed-stopped. Any follower whose plan was committed
        # against V's worst-case trajectory may now want to recompute
        # (commit was safe under synthetic decel-to-stop; actual stop is
        # at or before that worst-case position). More importantly, BL-
        # stopped followers (commit dead, sitting with stop_reason='leader')
        # have no live events to drive forward — this is their wake.
        self._notify_followers(t, v)
        # Reset stop_dist — the planned brake distance has been consumed.
        # Leaving it stale breaks _notify_followers's threshold check
        # (new_leader_free > old_stop + h_min/2), which keeps a stopped
        # follower asleep while its leader moves far ahead.
        v.stop_dist = 0.0
        self._process_crossed_nodes(t, v, crossed)

        # NEW: stopped-at-DIVERGE-exit release (narrow).
        # V 가 보유한 DIVERGE lock 의 마지막 sub-seg (= seg_to ∈ exit_nodes)
        # 의 끝에 stop 한 경우 release. Diverge 의 exit_node = succ_X 는
        # 분기 이후 branch 의 노드 → 다른 V_Y 가 같은 zone 의 *다른 branch*
        # 진입해도 V_X 와 물리적 분리 → 충돌 X.
        # ⚠️ MERGE 는 절대 release 금지. Merge 의 exit_node = merge_node 는
        # 모든 branch 의 합류점 → 다른 V_Y 가 진입 시 merge_node 에서 충돌.
        cur_seg_key = (v.seg_from, v.seg_to)
        cur_seg_lids = {lid for _z, lid in
                         self._seg_to_zone.get(cur_seg_key, [])}
        if cur_seg_lids:
            seg_len_at_stop = v.current_seg_length()
            at_seg_end = (seg_len_at_stop > 0
                          and v.seg_offset >= seg_len_at_stop - SEG_CROSS_EPS)
            if at_seg_end:
                for lid in list(cur_seg_lids):
                    if not lid.endswith('_diverge'):
                        continue   # MERGE 는 release 금지
                    if self._zone_lock.get(lid) is not v:
                        continue
                    if v.seg_to not in self._lock_exit_nodes.get(lid, set()):
                        continue
                    self._zone_release(t, lid)

        # Job arrival: if v has a job and reached end of path, trigger dwell.
        # Only fires when v is on the LAST segment at/past its end.
        if v.job is not None and self.job_mgr is not None:
            last_seg_idx = len(v.path) - 2
            at_end_of_path = (v.path_idx >= last_seg_idx
                              and last_seg_idx >= 0
                              and v.seg_offset >= v.current_seg_length() - 1.0)
            if at_end_of_path:
                self.job_mgr.on_arrive(t, v)
                return

        if self._trace_match(v.id):
            self._trace(f"  [STOPPED] t={t:.4f} V#{v.id} pidx={v.path_idx} "
                        f"off={v.seg_offset:.1f} seg={v.seg_from}->{v.seg_to} "
                        f"stop_reason={v.stop_reason}")

        # Check if at a ZCU boundary — register as waiter for event-driven
        # wakeup. ZCU_ARRIVE_EPS = SEG_CROSS_EPS + 0.5mm: a vehicle that
        # brakes-to-stop at a boundary node has bnd_dist ≈ 0 (margin-free
        # design); we still allow a sub-mm slack for any residual float
        # drift. Without this the waiter registration is skipped and the
        # vehicle never gets woken (seed=4 V#1 case).
        bnd_dist, _, bnd_node = self._find_first_boundary(v)
        if bnd_node and bnd_dist < ZCU_ARRIVE_EPS:
            # Stopped right at boundary → try lock or wait
            zones = self._relevant_zones(v, bnd_node)
            for zone, lock_id in zones:
                if not self._zone_request(v, lock_id):
                    self._zone_wait(v, lock_id)
                    self._try_push_lock_holder(t, v, lock_id)
                    return
            # All granted
            v.passed_zcu.add(bnd_node)
            self._post(t + 0.01, EV_REPLAN, v)
            return

        # No polling. Wake-up only via event-driven triggers (leader's
        # _notify_followers or ZCU grant).

    def _on_zcu_grant(self, t: float, v: Vehicle):
        v.waiting_at_zcu = None
        # At grant time we already hold the lock we waited on. Try to
        # acquire free siblings; if any is held, chain-wait on it while
        # keeping the one we have. We do NOT release the just-granted
        # lock here (that would thrash FIFO and caused measurable
        # collision regression in n=100 stress tests). The partial-hold
        # risk is handled at the _on_boundary side (atomic acquire) so
        # this grant path stays conservative.
        bnd_dist, _, bnd_node = self._find_first_boundary(v)
        if bnd_node and bnd_dist < v.h_min + 5000:
            zones = self._relevant_zones(v, bnd_node)
            denied_lock_id = None
            for _zone, lock_id in zones:
                if self._zone_lock.get(lock_id) is v:
                    continue
                if not self._zone_request(v, lock_id):
                    if denied_lock_id is None:
                        denied_lock_id = lock_id
            if denied_lock_id is not None:
                self._zone_wait(v, denied_lock_id)
                self._try_push_lock_holder(t, v, denied_lock_id)
            elif zones:
                v.passed_zcu.add(bnd_node)
        self._check_zcu_entry(t, v)
        # V was waiting at boundary (vel=0). Prior commit ended in stop;
        # extension-only replan is safe given _on_zcu_exit's exit-node
        # validation (stale ZCU_EXIT events from any earlier post for
        # the same lock will now self-skip at fire time).
        self._replan(t, v, caller='zcu_grant')

    def _relevant_zones(self, v: Vehicle, bnd_node: str) -> List[Tuple[ZCUZone, str]]:
        """Determine which zones at a boundary node this OHT must lock.

        - Diverge: always lock (physical node is shared, direction-independent)
        - Merge: lock only if the FIRST occurrence of bnd_node in the
                 vehicle's forward path goes into the merge core. Later
                 re-occurrences (from path extensions looping back) are
                 handled by their own boundary-crossing events — the
                 vehicle's path_idx will have advanced past the first
                 occurrence by then, so this scan naturally finds the
                 re-entry at that later time. Without the "first
                 occurrence only" rule, a future re-entry's direction
                 can trigger over-locking at a pass-through crossing
                 that doesn't actually enter the merge (seed=5 case).
        """
        result = []
        for zone, lock_id in self._boundary_to_zones.get(bnd_node, []):
            if zone.kind == 'diverge':
                result.append((zone, lock_id))
            elif zone.kind == 'merge':
                for i in range(v.path_idx, min(v.path_idx + 40, len(v.path) - 1)):
                    if v.path[i] == bnd_node:
                        if v.path[i + 1] == zone.node_id:
                            result.append((zone, lock_id))
                        break
        return result

    def _brake_to_stop(self, t: float, v: Vehicle, stop_dist: float):
        """Decelerate to stop at stop_dist ahead. Posts SEG_END + STOPPED.

        stop_dist is the PHYSICAL brake-end (may be < commit horizon when
        a leader caps the brake earlier). v.stop_dist is updated to track
        the physical end, but X marker is NOT modified — caller (_replan
        / pre-acquire refusal) has already pinned X at the commit horizon
        and the brake should not retreat the visual commit point.
        """
        if v.vel > VEL_ZERO and stop_dist > DIST_ZERO:
            decel = max(1.0, v.vel * v.vel / (2 * stop_dist))
            decel = min(decel, v.d_max)
            v.acc = -decel
            v.state = DECEL
            v.stop_dist = stop_dist
            t_stop = v.vel / decel
            # Phase 2.1: base_dist = vehicle's CURRENT cumulative distance.
            # With Phase 2.1 truncation, committed_traj's last entry may
            # be the cruise-start point (not vehicle's current position).
            # state_at interpolates correctly across the cruise phase.
            s_now = state_at(v, t)
            base_dist = (s_now.dist if s_now is not None
                         else (v.committed_traj[-1][1]
                               if v.committed_traj else 0.0))
            v.committed_traj.append((t, base_dist, v.vel, -decel))
            d_stop = base_dist + v.vel * t_stop - 0.5 * decel * t_stop ** 2
            v.committed_traj.append((t + t_stop, d_stop, 0.0, 0.0))
            # Brake-stop is now committed up to the stop time; otherwise a
            # force=False replan that fires before V actually stops would
            # extend with entries earlier than the in-traj stop, corrupting
            # time-order (out-of-order committed_traj breaks state_at's
            # binary search).
            v.commit_end_t = max(v.commit_end_t, t + t_stop)
            # SEG_END for each boundary crossed during the brake. Without
            # these, EV_STOPPED fires while advance_position is several
            # segments behind and processes a multi-seg jump in one event,
            # leaving stale _seg_occupants entries on intermediate segments.
            # Use ACTUAL brake distance (post d_max clamp), not the caller-
            # requested stop_dist: when stop_dist is tighter than d_max
            # allows, decel was clamped to d_max and the actual brake
            # distance = v.vel * t_stop - 0.5 * decel * t_stop² is larger
            # than requested stop_dist. Using requested stop_dist here
            # missed SEG_END posts for legitimate cross-during-brake
            # (seed=99 V#122 case: stop_dist=106.5 but actual=475 with
            # decel clamped from 2230 to 500, brake crossed pidx=89→90
            # boundary but no SEG_END fired → STOPPED handler clamped V
            # at pidx=89 off=seg_len; followers saw leader 300mm closer
            # than committed → spurious gap-violation reports at t=7830).
            actual_brake_dist = v.vel * t_stop - 0.5 * decel * t_stop ** 2
            if v.path_idx < len(v._seg_lengths):
                cum = max(0.0, v._seg_lengths[v.path_idx] - v.seg_offset)
            else:
                cum = 0.0
            next_idx = v.path_idx + 1
            while next_idx < len(v.path) - 1 and cum < actual_brake_dist - 1e-6:
                disc = v.vel * v.vel - 2 * decel * cum
                if disc < 0:
                    break
                t_exit_rel = (v.vel - math.sqrt(disc)) / decel
                if t_exit_rel > t_stop - 1e-9 or t_exit_rel < 1e-9:
                    break
                self._post(t + t_exit_rel, EV_SEG_END, v)
                if next_idx >= len(v._seg_lengths):
                    break
                cum += v._seg_lengths[next_idx]
                next_idx += 1
            self._post(t + t_stop, EV_STOPPED, v)
            if self._trace_match(v.id):
                self._trace(f"  [BRAKE] t={t:.4f} V#{v.id} stop_dist={stop_dist:.1f} "
                            f"vel={v.vel:.1f} decel={decel:.1f} t_stop={t+t_stop:.4f} "
                            f"d_stop={d_stop:.1f} followers={sorted(f.id for f in self._followers.get(v.id, set()))}")
        else:
            v.vel = 0.0; v.acc = 0.0; v.state = STOP
            self._pin_marker_at_dist(v, 0)
            v.stop_dist = 0.0
            self._commit_state(v, t)

    def _on_boundary(self, t: float, v: Vehicle, data=None):
        """Reached braking point before a ZCU boundary — lock attempt.

        bnd_node is read from event payload (set at post time in
        _schedule_plan_events). Event is only posted when the plan
        actually reaches the boundary (stop_reason=='zcu'), so zones
        must be non-empty here; otherwise the I2 invariant is broken.
        """
        self._trim_committed(v, t)
        bnd_node = (data or {}).get('bnd_node')
        if bnd_node is None:
            raise RuntimeError(
                f"EV_BOUNDARY without bnd_node payload V#{v.id} t={t:.4f}")

        zones = self._relevant_zones(v, bnd_node)
        if not zones:
            # I2 invariant: EV_BOUNDARY is only posted when plan reaches an
            # un-owned boundary that has relevant zones. If we get here, the
            # boundary was claimed/granted elsewhere between post and fire
            # (path changed, lock auto-granted via _zone_release, etc.). Re-
            # check via a fresh replan rather than fail — that's recoverable.
            if self._trace_match(v.id):
                self._trace(f"  [BND_STALE] t={t:.4f} V#{v.id} bnd={bnd_node} "
                            f"no relevant zones — replan")
            self._replan(t, v, caller='boundary_stale')
            return

        if self._trace_match(v.id):
            self._trace(f"  [BND] t={t:.4f} V#{v.id} bnd_node={bnd_node} "
                        f"zones={[(z.kind, lid) for z,lid in zones]} "
                        f"vel={v.vel:.1f} leader={v.leader.id if v.leader else None} "
                        f"stop_reason={v.stop_reason}")

        # PRE-ACQUIRE SAFETY: peek past bnd_node for a leader sitting
        # within h_min beyond it. _update_leader normally walks only up
        # to the un-claimed boundary (since v's plan can't legally extend
        # past), so a vehicle just past the boundary is invisible until
        # v locks. By that time v has already cruised across the boundary
        # and physically overlaps that vehicle (V#1↔V#6 case at seed=1
        # t=297.80 — V#1 was already past 2 ZCUs when V#6 acquired the
        # entry, leader walk never saw V#1). Refuse the lock if a leader
        # is too close; brake at boundary; await its _notify_followers
        # wake-up. EV_BOUNDARY re-fires on EV_REPLAN (stop_reason='zcu')
        # to retry the acquire once the leader has cleared.
        v.passed_zcu.add(bnd_node)
        self._update_leader(v, t)
        pre_leader = v.leader
        pre_gap = float('inf')
        if pre_leader is not None:
            pre_gap, _ = self.gap(v, t)
        v.passed_zcu.discard(bnd_node)
        # Refuse only when pre_leader is stationary (parked past
        # boundary). Moving leaders will clear naturally as v
        # approaches; the post-acquire replan's compute_velocity_profile
        # uses h_min-from-leader to brake correctly. The collision case
        # (V#1↔V#6) was a stationary V#1 — that's what we need to catch.
        # Refusing for moving leaders creates needless EV_REPLAN
        # cascades through follower cycles (event storm seen at
        # idle_n200_seed99_disp t≈346 — sim_t froze, GUI looked
        # deadlocked).
        too_close = (pre_leader is not None
                     and pre_gap < v.h_min
                     and pre_leader.vel < VEL_ZERO)

        if too_close:
            # Mirror the lock-denied brake path but skip _zone_wait /
            # _try_push_lock_holder — wake source here is the leader's
            # _notify_followers, not zone grant. Keep v.leader pointing
            # at pre_leader (set by the peek's _update_leader) so the
            # follower registration is in place for that wake.
            old_key = (v.seg_from, v.seg_to)
            crossed = v.set_state(t)
            new_key = (v.seg_from, v.seg_to)
            if old_key != new_key:
                self._update_occupancy(v, old_key, new_key, t, crossed=crossed)
            self._process_crossed_nodes(t, v, crossed)

            bnd_dist_pa, _, _ = self._find_first_boundary(v)
            if bnd_dist_pa == float('inf'):
                bnd_dist_pa = 0.0

            # If pre_leader is stationary, tighten target so we stop
            # h_min back from the parked leader rather than at the
            # boundary node (boundary node would put us 0mm from
            # pre_leader if pre_leader is sitting just past it).
            if pre_leader.vel < VEL_ZERO:
                gap_d_now, _ = self.gap(v, t)
                h_min_dist = max(0.0, gap_d_now - v.h_min)
                bnd_dist_pa = min(bnd_dist_pa, h_min_dist)

            if v.vel > VEL_ZERO and bnd_dist_pa > DIST_ZERO:
                self._brake_to_stop(t, v, bnd_dist_pa)
            else:
                v.vel = 0.0; v.acc = 0.0; v.state = STOP
                self._pin_marker_at_dist(v, 0)
                self._commit_state(v, t)
            v.stop_reason = 'zcu'

            # Register as waiter on every relevant zone lock that is
            # currently held. Without this, the only wake source is
            # pre_leader's _notify_followers — which fails after the
            # next _replan calls _update_leader without passed_zcu
            # peek (leader walks stop at the un-claimed boundary,
            # v.leader → None, _sync_followers drops v from
            # pre_leader's _followers, all wake links severed —
            # V#61 idle_n200_seed99_disp deadlock).
            for _zone, lid in zones:
                if self._zone_lock.get(lid) is not None:
                    self._zone_wait(v, lid)

            if self._trace_match(v.id):
                self._trace(
                    f"  [PRE_ACQ_REFUSE] t={t:.4f} V#{v.id} bnd={bnd_node} "
                    f"pre_leader=V#{pre_leader.id} pre_gap={pre_gap:.1f} "
                    f"h_min={v.h_min} bnd_dist={bnd_dist_pa:.1f}")
            return

        # All-or-nothing acquisition. If any sibling lock at this
        # boundary is held by another vehicle, roll back any we
        # newly acquired in this call and wait for the denied one.
        # Holding partial sibling locks while waiting for the rest
        # is a classic deadlock source (observed in the 4-way cycle
        # at 3556/3558/3609/3611 in multi_n100_seed99 at t=110).
        denied_lock_id = self._try_acquire_all_zones(v, zones, t)

        if denied_lock_id is None:
            v.passed_zcu.add(bnd_node)
            # Phase 2.1: opt-in extension. Commit truncated at brake_start
            # so old decel/stop events don't exist — replan extension
            # adds cruise continuation past current brake_start.
            self._replan(t, v, skip_set_state=True,
                         caller='boundary_lock_grant')
            return

        # Lock denied → brake to boundary, then wait for ZCU_GRANT
        old_key = (v.seg_from, v.seg_to)
        crossed = v.set_state(t)
        new_key = (v.seg_from, v.seg_to)
        if old_key != new_key:
            self._update_occupancy(v, old_key, new_key, t, crossed=crossed)
        self._process_crossed_nodes(t, v, crossed)

        bnd_dist, _, _ = self._find_first_boundary(v)
        if bnd_dist == float('inf'):
            bnd_dist = 0.0
        # Brake target = boundary node exactly. vel-gated SEG_CROSS_EPS
        # (advance_position) ensures a stopped vehicle at offset=seg_len
        # is logically still on the pre-zone segment, not auto-crossed.

        # If a stationary leader sits just past the un-claimed boundary
        # (V#115/V#152 case), tighten brake target so we stop h_min
        # back from the parked leader.
        self._update_leader(v, t)
        if v.leader is not None and v.leader.vel < VEL_ZERO:
            gap_d, _ = self.gap(v, t)
            h_min_dist = max(0.0, gap_d - v.h_min)
            bnd_dist = min(bnd_dist, h_min_dist)

        # Register as waiter IMMEDIATELY — not after the brake. If
        # the holder releases while we're still braking, a naive
        # implementation would leave the lock free, and a trailing
        # vehicle could arrive later and grab it ahead of us (FIFO
        # violation — seed 278 exhibited this: V#1 was denied while
        # moving and was never queued, so V#2 later stole the lock).
        # _zone_wait is idempotent on the dedup check.
        self._zone_wait(v, denied_lock_id)
        self._try_push_lock_holder(t, v, denied_lock_id)
        if v.vel > VEL_ZERO and bnd_dist > DIST_ZERO:
            self._brake_to_stop(t, v, bnd_dist)
        else:
            v.vel = 0.0; v.acc = 0.0; v.state = STOP
            self._pin_marker_at_dist(v, 0)
            self._commit_state(v, t)
        # Release any diverge lock we're still holding whose diverge
        # node is already in passed_zcu — we're physically past that
        # node and stopped at the next boundary, so keeping the lock
        # only creates circular hold-wait deadlocks on tight U-curves.
        self._release_passed_diverge_locks(t, v)

    # ── Core: _replan() ───────────────────────────────────────────────────

    def _replan(self, t: float, v: Vehicle, skip_set_state: bool = False,
                caller: str = 'unknown'):
        # Commit-aware early exit. If v has a live commit (events past t
        # already in heap), the committed plan governs forward motion. New
        # plans would either duplicate events (heap chaos) or overwrite
        # committed events (invariant violation). Skip and let committed
        # events fire. Schedule a retry once commit ends so callers (e.g.
        # zcu_grant arriving mid-brake) don't get silently dropped. Dedup
        # by retry target time — multiple early-exits for the same V
        # converge to the same commit_end_t and posting many duplicates
        # would storm at fire time.
        if v.commit_end_t > t + 1e-6:
            self._replan_skip_commit_alive = getattr(
                self, '_replan_skip_commit_alive', 0) + 1
            retry_t = v.commit_end_t + 1e-9
            # Dedup with tolerance: with the EV_REPLAN +1e-9 priority offset
            # in _post, retry_t can drift by 1ns per cycle, breaking strict
            # equality dedup and triggering an infinite ping-pong cascade
            # when mutual leader links exist (V#11/V#29 SPLIT case). Allow
            # 1e-6 tolerance so cumulative epsilon drift is still treated
            # as "same retry".
            prev_retry = getattr(v, '_retry_replan_at', -1.0)
            if abs(prev_retry - retry_t) > 1e-6:
                v._retry_replan_at = retry_t
                self._post(retry_t, EV_REPLAN, v)
            return
        self._trim_committed(v, t)
        old_key = (v.seg_from, v.seg_to)
        if skip_set_state:
            crossed = v.advance_position(t)
        else:
            crossed = v.set_state(t)
        new_key = (v.seg_from, v.seg_to)
        if old_key != new_key:
            # Retrospective NO_LOCK validation BEFORE _update_occupancy
            # so the safety-net release in _update_occupancy can't drop
            # the lock V was holding while traversing each mid_key.
            # Mirrors the fix applied in _on_zcu_exit.
            for k in range(len(crossed) - 1):
                mid_key = (crossed[k], crossed[k + 1])
                self._check_zcu_entry(t, v, seg_key=mid_key)
            if crossed:
                last_crossed = crossed[-1]
                if last_crossed != new_key[0]:
                    self._check_zcu_entry(t, v,
                                          seg_key=(last_crossed, new_key[0]))
            self._check_zcu_entry(t, v, seg_key=new_key)
            self._update_occupancy(v, old_key, new_key, t, crossed=crossed)
        self._process_crossed_nodes(t, v, crossed)

        # Auto-extend only when no job assigned AND legacy wander mode is on.
        # Job-driven paths are final (Dijkstra to src/dst); extending would
        # invalidate the destination. With auto_extend_idle=False (default),
        # job-less vehicles run out of path and stop, becoming pushable IDLEs.
        if (self.auto_extend_idle and v.job is None
                and v.needs_path_extension()):
            ext = random_safe_path(self.gmap, v.path[-1], length=100,
                                   rng=self._rng)
            v.extend_path(ext)

        # Arrival handling: end-of-path AND dest-reached are merged so a
        # vehicle that is physically at its dest (boundary coincidence:
        # path[-1] == dest_node, seg_to == dest_node, seg_offset == seg_len)
        # gets dest_reached and stop_reason='dest' set even when the prior
        # brake fired with reason='leader'/'zcu'. Previously the two were
        # separate branches and end-of-path returned first, leaving the
        # arrival metadata stale → vehicle marked stop_reason='leader',
        # not pushable, blocking the corridor indefinitely
        # (V#172/V#147 phantom-stop case).
        #
        # at_path_end: physically at path[-1] (1mm tolerance)
        # at_dest: position coincides with dest_node — three forms:
        #   1. seg_from == dest    : already crossed into dest's seg
        #   2. path[path_idx] == dest : redundant with (1)
        #   3. seg_to == dest AND offset at end : stopped at boundary
        #                            node that is dest (V#52 port-at-
        #                            merge case, V#147 phantom case)
        last_seg_idx = len(v.path) - 2
        at_path_end = (v.path_idx >= last_seg_idx and last_seg_idx >= 0
                       and v.seg_offset >= v.current_seg_length() - 1.0)
        at_dest = False
        if v.dest_node and not v.dest_reached:
            at_dest = (v.seg_from == v.dest_node
                       or (v.path_idx > 0
                           and v.path[v.path_idx] == v.dest_node)
                       or (v.seg_to == v.dest_node
                           and v.seg_offset
                           >= v.current_seg_length() - SEG_CROSS_EPS))
        if at_path_end or at_dest or (v.dest_node and v.dest_reached):
            v.vel = 0.0; v.acc = 0.0; v.state = STOP
            v.next_zcu_node = None
            v.commit_end_idx = v.path_idx
            v.commit_horizon_dist = 0.0
            if at_dest or v.dest_reached:
                v.dest_reached = True
                # Clear stale stop_reason from the brake that brought v
                # here (was 'leader'/'zcu') so push_target accepts v as
                # a movable parked vehicle.
                v.stop_reason = 'dest'
            self._commit_state(v, t)
            if v.job is not None and self.job_mgr is not None:
                self.job_mgr.on_arrive(t, v)
                return
            # No job: V is now pushable. Wake followers.
            self._notify_followers(t, v)
            return

        # ── Constraints ──────────────────────────────────────────────────
        # target_v는 _go의 초기 휴리스틱 용도. 실제 plan 최적화는
        # compute_velocity_profile가 path 전체 segment 속도제한을 envelope
        # 으로 흡수해서 처리한다. 따라서 여기서는 현재 segment 속도만
        # 본다 (lookahead나 ZCU peek 보정 불필요).
        seg_speed = v.current_seg_speed()
        target_v = min(v.v_max, seg_speed)

        # Phase 1: commit-immutability plan start. If committed_traj has
        # entries beyond t, the new plan starts from the END of committed
        # (so extension only — already-committed kinematics are immutable).
        # Otherwise plan starts from vehicle's current state.
        (t_start, v0_start, abs_dist_start,
         path_idx_start, seg_offset_start) = self._get_plan_start(v, t)

        # ZCU boundary — find first un-granted boundary FROM plan start
        # (not from vehicle current). When start = current, this is the
        # same as the legacy _find_first_boundary call.
        #
        # Iterate past currently-non-relevant boundaries (e.g., a merge
        # whose first-occurrence in v.path doesn't enter the merge core)
        # via a LOCAL skip set, NOT v.passed_zcu. Poisoning passed_zcu
        # here masks a later re-occurrence of the same node where it
        # would be relevant — V#7 seed=300 1500/4381 case: 1500 appears
        # twice in the path, first time exits the zone, second time
        # enters; v.passed_zcu.add(1500) on the first scan made the
        # second scan skip and produced a silent merge entry.
        _local_skip: Set[str] = set()
        bnd_dist, bnd_pi, bnd_node = self._find_first_boundary_from(
            v, path_idx_start, seg_offset_start)
        for _ in range(20):  # safety limit
            if bnd_dist >= 100000 or not bnd_node:
                break
            zones = self._relevant_zones(v, bnd_node)
            if zones:
                break
            _local_skip.add(bnd_node)
            bnd_dist, bnd_pi, bnd_node = self._find_first_boundary_from(
                v, path_idx_start, seg_offset_start, skip=_local_skip)

        # Path leader — refresh at each replan
        self._update_leader(v, t)
        leader = v.leader

        # Push: if our leader is push target or a chain-traversable
        # STOP, _try_push will handle traversal to the chain root.
        # _assign_destination triggers that root's own _replan, which
        # will _notify_followers and bring us back through here in the
        # next tick with an active leader.
        if leader is not None:
            self._try_push(t, pusher=v, pushee=leader)
            # Leader was just rerouted; refresh the reference (its
            # plan_gen advanced but identity didn't change).

        leader_free = float('inf')
        leader_traj_end_x = float('inf')
        if leader is not None:
            gap_d, _ = self.gap(v, t)
            remaining = self._leader_committed_remaining(leader, t)
            leader_free = gap_d + remaining - v.h_min
            # Explicit "never plan beyond leader's committed horizon"
            # cap (follower frame, minus h_min). For decel-to-stop-
            # ending trajs, traj[-1] is the stop position, so this
            # equals the follower's steady-state approach limit. For
            # cruise/accel-ending trajs, traj[-1] is the last phase's
            # START — once leader is past it, remaining collapses to
            # 0 and this cap becomes "leader's current position -
            # h_min". Prevents the follower from planning into the
            # cruise-forever extrapolation window that
            # compute_velocity_profile uses internally.
            leader_traj_end_x = self._leader_traj_end_x(leader, t,
                                                        gap_d, v.h_min)

        # Dest — distance from plan-start position
        dest_dist = self._dist_to_dest_from(v, path_idx_start, seg_offset_start)

        # Path-end distance — from plan-start position
        path_end_dist = float('inf')
        last_seg_idx = len(v.path) - 2
        if path_idx_start <= last_seg_idx and last_seg_idx >= 0:
            seg_len_s = (v._seg_lengths[path_idx_start]
                         if path_idx_start < len(v._seg_lengths) else 0.0)
            d = max(0.0, seg_len_s - seg_offset_start)
            for k in range(path_idx_start + 1, last_seg_idx + 1):
                d += v._seg_lengths[k] if k < len(v._seg_lengths) else 0
            path_end_dist = d

        # Leader too close → immediate stop (before plan_boundary calculation).
        # Wake-up via leader's _notify_followers when it moves.
        if leader is not None and leader_free <= 0:
            if self._trace_match(v.id):
                self._trace(f"  [TELEPORT-leader] t={t:.4f} V#{v.id} "
                            f"leader_free={leader_free!r} vel_was={v.vel:.1f}")
            v.vel = 0.0; v.acc = 0.0; v.state = STOP; v.stop_dist = 0.0
            v.stop_reason = 'leader'
            v.next_zcu_node = None
            v.commit_end_idx = v.path_idx
            v.commit_horizon_dist = 0.0
            self._pin_marker_at_dist(v, 0)
            self._commit_state(v, t)
            self._push_or_schedule_for_idle_leader(t, v)
            self._notify_followers(t, v)
            return

        # Fix A: V currently stopped (v≈0) AND gap_d < h_min.
        # Leader's committed plan will open the gap to h_min eventually
        # (leader_free > 0 else caught above). Instead of starting motion
        # from a too-close initial gap (transit would dip below h_min
        # physically — V#18↔V#181 case near ZCU-1697), stay stopped and
        # schedule an EV_REPLAN at the exact time gap reaches h_min per
        # leader's committed motion. When v is still cruising (not stopped),
        # skip this branch and let the regular plan_boundary + velocity_
        # profile caps handle the constraint.
        if (leader is not None and gap_d < v.h_min
                and v.vel < VEL_ZERO):
            v.vel = 0.0; v.acc = 0.0; v.state = STOP; v.stop_dist = 0.0
            v.stop_reason = 'leader'
            v.next_zcu_node = None
            v.commit_end_idx = v.path_idx
            v.commit_horizon_dist = 0.0
            self._pin_marker_at_dist(v, 0)
            self._commit_state(v, t)
            wake_t = self._compute_wake_time_for_h_min(v, leader, t, gap_d)
            if wake_t is not None and wake_t > t + 1e-6:
                self._post(wake_t, EV_REPLAN, v)
            self._notify_followers(t, v)
            return

        # Plan boundary — brake target IS the boundary node exactly.
        # advance_position's vel-gated SEG_CROSS_EPS keeps a stopped
        # vehicle on the pre-zone segment (not auto-crossed) without
        # needing a setback margin. See module-level comment.
        bnd_dist_adj = bnd_dist

        # Commit horizon — physical commit point. Events / locks are
        # scheduled within this distance only. Bounded by:
        #   - brake distance from v_max (we can always stop within this)
        #   - next ZCU entrance (locks beyond it are not yet acquired)
        # Leader / dest / path_end are velocity caps, NOT commit boundaries:
        # a closer leader does not retract committed events for our own
        # locked path; a dispatch reroute beyond commit_end is allowed.
        brake_dist_v_max = v.v_max * v.v_max / (2 * v.d_max)
        # Effective brake distance — *실제 도달 가능 속도* 기반.
        # commit/lock 범위가 actual segment speed 에 맞게 축소된다 (KAIST 처럼
        # segment speed(1000) << v_max(3600) 인 경우 13m→1m). v_max-based 범위
        # 안의 max segment speed 가 vehicle 이 거기서 실제로 낼 수 있는 최대
        # 속도 (= segment speed cap) 이므로 그 brake_dist 면 충분히 정지 가능.
        eff_v_max = (v._seg_speeds[v.path_idx]
                     if v.path_idx < len(v._seg_speeds) else v.v_max)
        _acc_s = max(0.0, v.current_seg_length() - v.seg_offset)
        _ks = v.path_idx + 1
        while _ks < len(v.path) - 1 and _acc_s <= brake_dist_v_max:
            sp = (v._seg_speeds[_ks]
                  if _ks < len(v._seg_speeds) else v.v_max)
            if sp > eff_v_max:
                eff_v_max = sp
            _acc_s += (v._seg_lengths[_ks]
                       if _ks < len(v._seg_lengths) else 0.0)
            _ks += 1
        eff_v_max = min(eff_v_max, v.v_max)
        eff_brake_dist = eff_v_max * eff_v_max / (2 * v.d_max)
        commit_horizon_dist = min(eff_brake_dist, bnd_dist_adj)
        v.commit_horizon_dist = commit_horizon_dist
        # Walk forward to find commit_end_idx: largest k such that the
        # cumulative distance from current pos to path[k] ≤ commit_horizon.
        dist_to_next_node = max(0.0, v.current_seg_length() - v.seg_offset)
        if commit_horizon_dist >= dist_to_next_node:
            accumulated = dist_to_next_node
            commit_end_idx = v.path_idx + 1
            while commit_end_idx < len(v.path) - 1:
                seg_len = (v._seg_lengths[commit_end_idx]
                           if commit_end_idx < len(v._seg_lengths) else 0.0)
                if accumulated + seg_len > commit_horizon_dist:
                    break
                accumulated += seg_len
                commit_end_idx += 1
            v.commit_end_idx = commit_end_idx
        else:
            v.commit_end_idx = v.path_idx

        # Commit horizon: the trajectory's COMMIT endpoint, bounded by
        # fixed constraints (bnd / dest / path_end) AND v's physical
        # reach (brake_dist_v_max + h_min). The physical-reach cap is
        # critical for safety — without it, v can plan to brake at a
        # boundary far beyond what leader_walk sees, and collide with
        # another vehicle already parked at that boundary
        # (V#76/V#53 at 3185_diverge case). Leader walk cap and plan
        # endpoint must agree: anything inside V's physical reach must
        # also be inside V's leader-visibility window. Excludes leader
        # caps so commit doesn't retreat across replans when a new
        # closer leader appears. X marker pins here.
        physical_reach = brake_dist_v_max + v.h_min
        commit_dist = min(bnd_dist_adj, dest_dist, path_end_dist, physical_reach)

        # Plan boundary: the trajectory's PHYSICAL end. Includes leader
        # caps because compute_velocity_profile must terminate where the
        # follower can actually reach. May be < commit_dist when the
        # leader is closer.
        # forward_stop_cap: closest STOP vehicle on forward path (from
        # _update_leader full walk). Caps plan past a stationary obstacle
        # that the moving leader may later leave behind (V#119/V#113).
        forward_stop_bound = (v.forward_stop_cap - v.h_min
                              if v.forward_stop_cap < float('inf')
                              else float('inf'))
        plan_boundary = min(commit_dist, leader_free, leader_traj_end_x,
                            forward_stop_bound)

        if self._trace_match(v.id):
            self._trace(f"  [PLAN] t={t:.4f} V#{v.id} vel={v.vel:.1f} "
                        f"bnd={bnd_dist:.1f}({bnd_node}) dest={dest_dist:.1f} "
                        f"leader_free={leader_free:.1f} "
                        f"traj_end={leader_traj_end_x:.1f} "
                        f"→ commit={commit_dist:.1f} pb={plan_boundary:.1f} "
                        f"target_v={target_v:.1f} passed_zcu={v.passed_zcu}")

        # Determine stop reason from what binds plan_boundary, then
        # pin X marker at commit_dist (NOT plan_boundary) so the
        # visualization shows the commit horizon, not the
        # leader-truncated physical end.
        if plan_boundary < 100000:
            # next_zcu_node is informational only — always reflects the
            # next un-owned boundary from _find_first_boundary, independent
            # of stop_reason. Lock-attempt trigger is no longer bound to
            # this field; EV_BOUNDARY is posted with bnd_node in its event
            # payload only when stop_reason=='zcu' (plan actually reaches
            # the boundary). Keeping the info populated lets diagnostics
            # and waiter-registration paths see the next boundary even
            # when leader/dest caps shorter.
            commit_at_bnd = (bnd_node is not None
                             and abs(commit_dist - bnd_dist_adj) < 1e-6)
            brake_at_bnd = (commit_at_bnd
                            and abs(plan_boundary - bnd_dist_adj) < 1e-6)
            if brake_at_bnd:
                v.stop_reason = 'zcu'
            elif (plan_boundary <= leader_free + 1e-6
                  and plan_boundary <= leader_traj_end_x + 1e-6
                  and (leader_free < commit_dist - 1e-6
                       or leader_traj_end_x < commit_dist - 1e-6)):
                v.stop_reason = 'leader'
            else:
                v.stop_reason = 'dest'
            v.next_zcu_node = bnd_node

            # X marker at commit_dist (visualization commit horizon)
            if commit_at_bnd:
                self._pin_marker(v, bnd_pi, bnd_node)
            else:
                self._pin_marker_at_dist(v, commit_dist)
            v.stop_dist = plan_boundary
        else:
            v.stop_dist = None
            v.stop_reason = None
            v.next_zcu_node = None
            self._pin_marker_at_dist(v, v.dist_to_seg_end() + 5000)

        # ── Scheduling ───────────────────────────────────────────────────

        # If V is essentially at an un-owned ZCU boundary (stopped, within
        # ZCU_ARRIVE_EPS) we MUST NOT start a micro-motion plan that would
        # drift across the boundary node via advance_position's vel-gated
        # SEG_CROSS_EPS slack (V#56 seed=20 case: brake-to-stop landed at
        # seg_offset=999.998 ≈ seg_len, the next replan computed
        # plan_boundary=0.002mm, _go set acc=500, the brake_start EV_BOUNDARY
        # fired 2ms later, set_state bumped vel to 0.95mm/s — above
        # VEL_ZERO — and the liberal cross check let V slip into the zone
        # without a lock). Route to the same at-boundary acquire/wait
        # logic _on_stopped uses, so this REPLAN converges to either a
        # legitimate cruise extension past the boundary (granted) or a
        # waiter parked at the boundary (denied).
        v_essentially_stopped = (v.vel < VEL_ZERO)
        at_unowned_bnd = (bnd_node is not None
                          and bnd_dist < ZCU_ARRIVE_EPS
                          and v_essentially_stopped)
        if at_unowned_bnd:
            zones_here = self._relevant_zones(v, bnd_node)
            if zones_here:
                denied_lock_id = self._try_acquire_all_zones(v, zones_here, t)
                v.vel = 0.0; v.acc = 0.0; v.state = STOP; v.stop_dist = 0.0
                v.commit_end_idx = v.path_idx
                v.commit_horizon_dist = 0.0
                self._pin_marker_at_dist(v, 0)
                self._commit_state(v, t)
                if denied_lock_id is None:
                    # All locks granted at this boundary. Mark passed and
                    # post a follow-up REPLAN so the now-unblocked plan
                    # extends past the boundary.
                    v.passed_zcu.add(bnd_node)
                    v.next_zcu_node = None
                    v.stop_reason = None
                    self._post(t + 1e-6, EV_REPLAN, v)
                else:
                    self._zone_wait(v, denied_lock_id)
                    self._try_push_lock_holder(t, v, denied_lock_id)
                    v.stop_reason = 'zcu'
                    v.next_zcu_node = bnd_node
                return

        if plan_boundary <= 0:
            if self._trace_match(v.id):
                self._trace(f"  [TELEPORT-pb] t={t:.4f} V#{v.id} "
                            f"pb={plan_boundary!r} vel_was={v.vel:.1f}")
            v.vel = 0.0; v.acc = 0.0; v.state = STOP; v.stop_dist = 0.0
            v.next_zcu_node = None
            v.commit_end_idx = v.path_idx
            v.commit_horizon_dist = 0.0
            self._pin_marker_at_dist(v, 0)
            self._commit_state(v, t)
            # No lock attempts in _replan — locks are handled by _on_boundary
            self._push_or_schedule_for_idle_leader(t, v)
            self._notify_followers(t, v)
            return

        # Set initial motion and let _schedule_plan_events handle the full profile
        # including leader-following, curve speed, and ZCU boundary braking.
        #
        # at_horizon_cap: stop_reason='dest' has three sub-cases —
        #   (i)   commit_dist == dest_dist        → V truly stops at job dest
        #   (ii)  commit_dist == path_end_dist    → V truly stops at path end (IDLE)
        #   (iii) commit_dist == physical_reach   → transient stop at commit
        #         horizon (~14m); next replan extends cruise. NOT a real stop.
        # Phase 2.1 truncate was originally meant for (iii) as well — V never
        # actually stops at horizon, so committing decel/stop and posting
        # EV_STOPPED is a future stale event waiting to happen (V#195↔V#83
        # seed=99 t=972 case: stale EV_STOPPED reset v.acc=0 after replan
        # extended cruise). Detect (iii) and treat it like leader/zcu for
        # truncation purposes.
        at_horizon_cap = (
            v.stop_reason == 'dest'
            and commit_dist < dest_dist - 1e-6
            and commit_dist < path_end_dist - 1e-6
        )
        self._go(t, v, target_v)
        self._schedule_plan_events(t, v, target_v, plan_boundary,
                                   t_start=t_start, v0_start=v0_start,
                                   path_idx_start=path_idx_start,
                                   seg_offset_start=seg_offset_start,
                                   abs_dist_start=abs_dist_start,
                                   at_horizon_cap=at_horizon_cap)
        # X marker stays at commit_dist (set above). _schedule_plan_events
        # no longer overwrites it.

    # ── Event scheduling ──────────────────────────────────────────────────

    def _schedule_plan_events(self, t: float, v: Vehicle, target_v: float,
                              plan_boundary: float,
                              t_start: Optional[float] = None,
                              v0_start: Optional[float] = None,
                              path_idx_start: Optional[int] = None,
                              seg_offset_start: Optional[float] = None,
                              abs_dist_start: Optional[float] = None,
                              _commit_obs: bool = True,
                              at_horizon_cap: bool = False):
        """Generate events from plan-start state to plan boundary.

        Phase 1 plan_start params override the legacy "from-current-state"
        defaults. When committed_traj has future entries, _replan computes
        the start state via _get_plan_start (commit end) and passes it.
        For empty/exhausted committed_traj, defaults fall through to
        vehicle's current state (legacy behavior).
        """
        # Default to current vehicle state when params not provided
        if t_start is None:
            t_start = t
        if v0_start is None:
            v0_start = v.vel
        if path_idx_start is None:
            path_idx_start = v.path_idx
        if seg_offset_start is None:
            seg_offset_start = v.seg_offset
        if abs_dist_start is None:
            abs_dist_start = (v.committed_traj[-1][1]
                              if v.committed_traj else 0.0)

        # ── Leader committed_traj at t_start ──────────────────────────
        # Slice from t_start (not t) so leader's near-future range maps
        # correctly when the plan starts in the future.
        leader = v.leader
        leader_traj = None
        leader_dist_offset = 0.0
        if leader is not None:
            gap_d, _ = self.gap(v, t)   # current gap (approximation)
            if gap_d < 100000:
                leader_traj = self._leader_traj_from_now(leader, t_start)
                leader_dist_offset = gap_d

        # ── plan_start 부터 path 슬라이스 ──────────────────────────
        seg_lengths = list(v._seg_lengths[path_idx_start:])
        seg_speeds = list(v._seg_speeds[path_idx_start:])
        seg_keys = [(v.path[i], v.path[i + 1])
                    for i in range(path_idx_start, len(v.path) - 1)]

        # ── Analytical profile 계산 (start = plan_start) ─────────────
        traj_rel, c_segs_rel = compute_velocity_profile(
            seg_lengths=seg_lengths,
            seg_speeds=seg_speeds,
            seg_keys=seg_keys,
            seg_offset=seg_offset_start,
            v0=v0_start,
            plan_boundary=plan_boundary,
            v_max=v.v_max,
            a_max=v.a_max,
            d_max=v.d_max,
            t_now=t_start,
            leader_traj=leader_traj,
            leader_dist_offset=leader_dist_offset,
            h_min=v.h_min,
        )

        if not traj_rel:
            # Empty profile = degenerate input (real bug if it fires —
            # investigate compute_velocity_profile inputs). Force the
            # vehicle into a deterministic, observable state so the rest
            # of the system can recover instead of silently freezing.
            print(f"[WARN] empty profile V#{v.id} t={t:.4f} v0={v.vel:.1f} pb={plan_boundary:.1f}")
            v.vel = 0.0
            v.acc = 0.0
            v.state = STOP
            self._commit_state(v, t)
            # If we're parked at a ZCU boundary, register as waiter on
            # every held relevant zone lock so a future release can wake
            # us. Mirrors the pre-acquire refusal path; without this an
            # empty-profile fall-through at a boundary leaves no wake
            # source if the leader registration also got cleared.
            if v.stop_reason == 'zcu':
                bnd_dist_w, _, bnd_node_w = self._find_first_boundary(v)
                if (bnd_node_w is not None
                        and bnd_dist_w < ZCU_ARRIVE_EPS):
                    for _zone, lid in self._relevant_zones(v, bnd_node_w):
                        if self._zone_lock.get(lid) is not None:
                            self._zone_wait(v, lid)
            return

        # ── traj 상대 거리 → committed_traj 절대 거리 변환 ──────────
        # Phase 1: base_dist = abs_dist_start (== plan-start abs distance,
        # which is commit end when committed_traj has future entries, else
        # vehicle's current cumulative). traj_rel[0] is at (t_start, 0, ...)
        # so its absolute counterpart is (t_start, abs_dist_start, ...).
        base_dist = abs_dist_start
        abs_traj = [(ti, base_dist + xi, vi, ai)
                    for (ti, xi, vi, ai) in traj_rel]
        abs_segs = [(te, tx, sk, base_dist + pd)
                    for (te, tx, sk, pd) in c_segs_rel]

        # ── Phase 2.1: kinematic truncation at brake_start ───────────
        # User intent: commit only [accel, cruise] up to brake_start
        # for DECISION-point stops (zcu, leader). decel/stop tail is
        # uncommitted; followers infer worst-case via _leader_traj_*
        # synthetic decel. dest stops are real commitments (vehicle
        # actually stops there for dwell), no truncation.
        # Skip when plan starts with decel (no cruise prelude).
        bs_t_pre = None
        for i in range(len(abs_traj) - 1, -1, -1):
            ai = abs_traj[i][3]
            if ai < -1e-6:
                bs_t_pre = abs_traj[i][0]
            else:
                if bs_t_pre is not None:
                    break
        # Truncate at brake_start when stop is TRANSIENT (not a real terminal
        # stop). Real terminal stops:
        #   - actual job destination (commit_dist == dest_dist)
        #   - true path end with no dest (commit_dist == path_end_dist)
        # Transient stops (truncate):
        #   - 'zcu': waiting at boundary for lock
        #   - 'leader': following a leader that may move
        #   - 'dest' + at_horizon_cap: commit was capped by physical_reach,
        #     not by an actual destination. Next replan will extend cruise.
        # Posting EV_STOPPED + decel tail for transient stops creates stale
        # events that fire after the extension replan and reset v.acc/vel
        # (V#83 seed=99 t=962.78 case).
        truncate_commit = (bs_t_pre is not None
                           and bs_t_pre > t + 1e-6
                           and (v.stop_reason in ('zcu', 'leader')
                                or at_horizon_cap))
        if truncate_commit:
            abs_traj_committed = [e for e in abs_traj
                                  if e[0] < bs_t_pre - 1e-9]
            abs_segs_committed = [e for e in abs_segs
                                  if e[0] < bs_t_pre - 1e-9]
            if not abs_traj_committed:
                abs_traj_committed = [(t, base_dist, v.vel, 0.0)]
        else:
            abs_traj_committed = abs_traj
            abs_segs_committed = abs_segs

        # committed_traj append. When committed has future entries, abs_traj
        # starts AT the existing last entry (same t & d) — replace last,
        # extend with rest. When committed is exhausted/empty, normal extend.
        if (v.committed_traj and abs_traj_committed and
                abs(v.committed_traj[-1][0] - abs_traj_committed[0][0]) < 1e-9):
            v.committed_traj[-1] = abs_traj_committed[0]
            v.committed_traj.extend(abs_traj_committed[1:])
        else:
            v.committed_traj.extend(abs_traj_committed)
        v.committed_segs.extend(abs_segs_committed)

        # ── 런타임 v.acc/vel/t_ref 를 profile 첫 phase에 sync ─────────
        # _go가 미리 설정한 v.acc는 target_v 기반 휴리스틱이지만, profile은
        # v_cap envelope을 따르므로 둘이 다를 수 있다 (예: target_v보다
        # plan_boundary 제약이 더 강해 profile은 cruise/brake로 시작).
        # 런타임 kinematics가 profile을 정확히 따르도록 첫 phase로 강제 sync.
        #
        # Skip degenerate leading phases (duration ≤ 1e-9 s). compute_velocity_
        # profile can emit a 0-duration accel→decel sequence when v0² ≈ 2·d_max
        # ·plan_boundary (V#155 seed=99 t=5082.7168 4275_merge case). The
        # PHASE_DONE for such a transition is filtered by the `t_nxt > t + 1e-9`
        # guard below, so syncing to the degenerate phase leaves runtime acc
        # stuck (V#155 stayed at acc=+500 across the boundary, accelerating
        # through 1491 without acquiring 4275_merge — NO_LOCK fault).
        # Use the SAME comparison as the PHASE_DONE post filter below
        # (`t_nxt > t + 1e-9`). Float arithmetic makes `(t_nxt - t_cur) <= 1e-9`
        # NOT equivalent to `not (t_nxt > t_cur + 1e-9)` — the latter rounds
        # `t_cur + 1e-9` once, the former subtracts directly. Stick to the
        # filter form so we skip exactly the phases whose PHASE_DONE the
        # filter drops.
        first_idx = 0
        while (first_idx + 1 < len(abs_traj)
               and not (abs_traj[first_idx + 1][0]
                        > abs_traj[first_idx][0] + 1e-9)):
            first_idx += 1
        first = abs_traj[first_idx]
        v.t_ref = first[0]
        v.vel = first[2]
        v.acc = first[3]
        if v.acc > 0.001:
            v.state = ACCEL
        elif v.acc < -0.001:
            v.state = DECEL
        else:
            v.state = CRUISE if v.vel > 0.001 else STOP

        if self._trace_match(v.id):
            first_phase = abs_traj[0] if abs_traj else None
            self._trace(f"  [PROF] t={t:.4f} V#{v.id} phases={len(traj_rel)} "
                        f"c_segs={len(c_segs_rel)} pb={plan_boundary:.1f} "
                        f"v0={v.vel:.1f} target_v={target_v:.1f} "
                        f"first_phase={first_phase} "
                        f"last_traj_dist={abs_traj[-1][1] if abs_traj else None:.1f}")

        # ── plan 종료 시점 식별 ──────────────────────────────────────
        last_t, last_d, last_v_, _ = abs_traj[-1]
        brake_start_t = bs_t_pre   # 위에서 이미 계산
        # Phase 2.1 — event posting horizon (= commit truncation horizon)
        event_horizon_t = brake_start_t if truncate_commit else float('inf')

        # ── EV_PHASE_DONE bulk-post ──────────────────────────────────
        # Filter: skip if t_nxt >= brake_start_t (= decel transition).
        # PHASE_DONE for accel→cruise / cruise→cruise within prelude
        # have t_nxt < brake_start_t, OK.
        for i in range(len(abs_traj) - 1):
            t_cur, _, _, a_cur = abs_traj[i]
            t_nxt, _, _, a_nxt = abs_traj[i + 1]
            if abs(a_nxt - a_cur) > 1e-6 and t_nxt > t + 1e-9:
                if t_nxt >= event_horizon_t - 1e-6:
                    continue
                self._post(t_nxt, EV_PHASE_DONE, v)

        # ── EV_SEG_END post ──────────────────────────────────────────
        for (t_enter, t_exit, seg_key, plan_dist) in abs_segs:
            if t_exit < float('inf') and t_exit > t + 1e-9:
                if t_exit > event_horizon_t + 1e-6:
                    continue
                self._post(t_exit, EV_SEG_END, v)

        # EV_BOUNDARY: lock-attempt trigger only. Posted when this plan
        # actually reaches the un-owned boundary (stop_reason == 'zcu').
        # bnd_node is carried in event payload so the handler does not
        # depend on v.next_zcu_node (which is now informational and may
        # have been overwritten by a concurrent replan before fire).
        #
        # EV_REPLAN: brake-start extension trigger for truncated leader/
        # dest plans. Replaces the EV_BOUNDARY case-2 path that used to
        # fire here for the same purpose. Keeps lock-attempt and plan-
        # extension semantically separate.
        if plan_boundary < 100000:
            t_bnd = brake_start_t if brake_start_t is not None else last_t
            # v.next_zcu_node is the first un-owned boundary from this plan's
            # _find_first_boundary; set unconditionally upstream so it's safe
            # to read here regardless of stop_reason.
            bnd_node_for_post = v.next_zcu_node
            if v.stop_reason == 'zcu' and bnd_node_for_post is not None:
                t_bnd = max(t_bnd, t + 1e-6)
                self._post(t_bnd, EV_BOUNDARY, v,
                           data={'bnd_node': bnd_node_for_post})
            elif truncate_commit and t_bnd > t + 1e-9:
                t_bnd = max(t_bnd, t + 1e-6)
                self._post(t_bnd, EV_REPLAN, v)
        if (last_v_ < 1e-6 and last_t > t + 1e-9 and not truncate_commit):
            self._post(last_t, EV_STOPPED, v)

        # ── ZCU exit 이벤트 ──────────────────────────────────────────
        self._post_zcu_exit_events(v, abs_segs)

        # ── 자기 지속: plan_boundary == inf → REPLAN at last SEG_END ──
        if plan_boundary >= 100000 and abs_segs:
            last_seg_t = abs_segs[-1][1]
            if last_seg_t < float('inf') and last_seg_t > t + 1e-9:
                self._post(last_seg_t, EV_REPLAN, v)

        # ── stop_dist 갱신 (physical brake target) ───────────────────
        # X marker is set in _replan at commit_dist (commit horizon) and
        # NOT overwritten here. stop_dist tracks physical trajectory end
        # (which may be < commit when leader caps the brake earlier).
        if abs_traj:
            final_dist_rel = traj_rel[-1][1]
            v.stop_dist = final_dist_rel if final_dist_rel < 100000 else None

        # ── Diagnostic: classify plan termination ──────────────────
        # Category: what kind of "tail" does this plan have? Captures
        # the SEMANTIC plan terminator (the event that marks the end
        # of forward commitment), not just the latest event in time
        # (PHASE_DONE/STOPPED tie at last_t for stop-ending plans).
        if plan_boundary >= 100000:
            cat = 'INFINITE_REPLAN'   # cruise indefinitely, self-sustaining
        elif last_v_ < 1e-6:
            cat = 'STOP_END'          # plan ends with vel=0 (decel-to-stop)
        elif brake_start_t is not None:
            cat = 'BOUNDARY_BRAKE'    # plan ends at brake-start (lock attempt
                                      # at boundary; vehicle still moving)
        else:
            cat = 'CRUISE_END'        # plan ends mid-cruise (no decel phase)
        log = getattr(self, '_plan_term_log', [])
        log.append((cat, last_v_, plan_boundary, len(abs_traj),
                    last_t - t))
        self._plan_term_log = log
        cnt = getattr(self, '_plan_term_count', collections.Counter())
        cnt[cat] += 1
        self._plan_term_count = cnt

        # ── Phase 1.1/1.2: commit_end_t observation ──────────────────
        # Snapshot prev commit_end_t (the upper bound of the events the
        # PRIOR _schedule_plan_events posted). v.commit_end_t survives
        # _trim_committed (it's not derived from committed_traj).
        if _commit_obs:
            prev_commit_end_t = v.commit_end_t

            # New commit_end_t = latest committed event time. With Phase
            # 2.1 truncation (zcu/leader stops), the last commit event is
            # EV_BOUNDARY at brake_start_t. For non-truncated plans
            # (dest stops, decel-only), it's last_t (= EV_STOPPED time).
            if truncate_commit:
                new_commit_end_t = brake_start_t
            else:
                new_commit_end_t = abs_traj[-1][0] if abs_traj else t
            v.commit_end_t = new_commit_end_t

            # Count commit-invariant violations: events the OLD commit
            # was going to fire that this NEW plan is overwriting.
            # For each NEW posted event with t_ev in [t, prev_commit_end_t)
            # we're effectively replacing the OLD plan's commitment.
            if prev_commit_end_t > t + 1e-6:
                inv_cnt = getattr(self, '_commit_invariant_violations',
                                  collections.Counter())

                # PHASE_DONE events
                for i in range(len(abs_traj) - 1):
                    t_cur, _, _, a_cur = abs_traj[i]
                    t_nxt, _, _, a_nxt = abs_traj[i + 1]
                    if (abs(a_nxt - a_cur) > 1e-6
                            and t_nxt > t + 1e-9
                            and t_nxt < prev_commit_end_t - 1e-6):
                        inv_cnt['PHASE_DONE'] += 1

                # SEG_END events
                for (t_enter, t_exit, sk, pd) in abs_segs:
                    if (t_exit < float('inf')
                            and t_exit > t + 1e-9
                            and t_exit < prev_commit_end_t - 1e-6):
                        inv_cnt['SEG_END'] += 1

                # BOUNDARY (already computed t_bnd above; recompute cheaply)
                if plan_boundary < 100000:
                    bs_t = None
                    for i in range(len(abs_traj) - 1, -1, -1):
                        if abs_traj[i][3] < -1e-6:
                            bs_t = abs_traj[i][0]
                        elif bs_t is not None:
                            break
                    cand_bnd = bs_t if bs_t is not None else last_t
                    cand_bnd = max(cand_bnd, t + 1e-6)
                    if cand_bnd < prev_commit_end_t - 1e-6:
                        inv_cnt['BOUNDARY'] += 1

                # STOPPED
                if (last_v_ < 1e-6 and last_t > t + 1e-9
                        and last_t < prev_commit_end_t - 1e-6):
                    inv_cnt['STOPPED'] += 1

                self._commit_invariant_violations = inv_cnt

        self._notify_followers(t, v)

        # No trailing REPLAN: SEG_END/PHASE_DONE chains are self-sustaining,
        # and BOUNDARY/STOPPED already terminate the plan deterministically.

    # ── Leader ────────────────────────────────────────────────────────────

    def _find_leader_on_path(self, v: Vehicle, fwd_segs: dict,
                              exclude_id: int = -1,
                              t: float = 0.0) -> Optional[Vehicle]:
        """Find nearest vehicle on forward segments using segment occupancy queues.

        O(forward_segments × occupants_per_segment) instead of O(N_vehicles).
        Single-track guarantee: queue order = position order within segment.
        """
        best_leader = None
        best_dist = float('inf')
        self_key = None
        self_off = 0.0
        for sk, sd in fwd_segs.items():
            if sd < -0.01:
                self_key = sk
                self_off = -sd
                break

        for seg_key, seg_dist in fwd_segs.items():
            for other in self._seg_occupants.get(seg_key, []):
                if other.id == exclude_id:
                    continue
                d = seg_dist + other.seg_offset
                if seg_key == self_key and other.seg_offset <= self_off:
                    continue  # behind me on same segment
                if 0 < d < best_dist:
                    best_dist = d
                    best_leader = other

        return best_leader

    def _update_leader(self, v: Vehicle, t: float = 0.0):
        """Find nearest vehicle ahead, including cross-branch leaders at
        diverge nodes (path-through-diverge distance).

        Walk stops at the first un-claimed ZCU boundary with relevant zones:
        v's plan cannot legally extend past a boundary it hasn't locked, so
        any vehicle beyond that point cannot constrain v's current plan
        window. Boundaries v already holds or that have no relevant zones
        for v are traversed. Loop-back and a 50-path-node safety cap also
        bound the walk.

        Cross-branch leaders: at each forward node fn (including diverge),
        also look at occupants of (fn, succ) for succ != tn. Distance is
        cur_dist (v → fn along v's path) + occupant's segment offset on
        the other branch. This catches the case where leader peeled off
        onto a different post-diverge branch — Euclidean is bounded below
        by 1/√2 of path-through-fn distance for ≤90° diverges, so h_min
        path-distance enforcement gives ~813mm Euclidean separation at
        h_min=1150 (above the 750mm OHT collision threshold).
        """
        old_leader = v.leader
        cur_key = (v.seg_from, v.seg_to)

        # Fix B: dest_reached vehicles are "done" — waiting for dispatcher
        # to assign a new job. They don't drive further on their current
        # path so they don't need a leader. Returning leader=None prevents
        # stale leader links over the path's tail (which may extend past
        # dest in lookahead scenarios) and breaks the mutual-leader cycle
        # potential (V#11/V#29 case, V#100 stale leader case).
        if v.dest_reached:
            if v.leader is not None:
                v.leader = None
                v.leader_dist = float('inf')
                v.forward_stop_cap = float('inf')
                self._sync_followers(v, old_leader)
            return

        # Same-segment leader (someone on my current seg ahead of me).
        # Compare EXTRAPOLATED offsets: other.seg_offset is a snapshot
        # taken at other's last event (replan / SEG_END / BOUNDARY), so if
        # another vehicle just crossed into this seg via EV_SEG_END its
        # stored seg_offset is 0 even though it has already moved forward
        # per its committed plan. Using the stored value misidentifies
        # that vehicle as being behind v and drops it from the leader
        # search, letting v over-accelerate into it.
        # Same-segment leader is always closer than any cross-branch
        # candidate (which is at distance ≥ remaining of current segment),
        # so early return is safe.
        # Caller contract: advance_position(t) must run before _update_leader,
        # so v.seg_offset is current. We compare other.extrapolated_offset
        # against v.seg_offset (snapshot, equal to current under contract).
        best = None; best_off = float('inf')
        for other in self._seg_occupants.get(cur_key, []):
            if other is v:
                continue
            other_off = (other.seg_offset
                         + other._dist_traveled(t - other.t_ref))
            if other_off > v.seg_offset and other_off < best_off:
                best_off = other_off
                best = other
        if best is not None:
            v.leader = best
            v.leader_dist = max(0.0, best_off - v.seg_offset)
            # Same-seg STOP also fills forward_stop_cap so plan caps to
            # its position (other_off - h_min, since other is on same seg
            # ahead of v).
            if best.state == STOP:
                v.forward_stop_cap = best.seg_offset + best._dist_traveled(t - best.t_ref) - v.seg_offset
            else:
                v.forward_stop_cap = float('inf')
            self._sync_followers(v, old_leader)
            return

        def _ext_off(o):
            return o.seg_offset + o._dist_traveled(t - o.t_ref)

        # Corridor-based ahead-of-v candidate (Phase 2 of the corridor
        # refactor). Within a directed corridor, vehicles' order is well-
        # defined by `corridor_offset[seg] + offset`. A W with corridor_pos
        # > v's corridor_pos is ahead of v and a leader candidate. This
        # catches OHTs on adjacent graph segments inside the same corridor
        # that the per-segment `_seg_occupants` forward walk below misses
        # only via the boundary peek / path-end peek (V#0/V#33 seed=99
        # t=1436 case: V#33 on (1647,1648) at corridor_offset=0, V#0 on
        # (1648,1649) — same corridor 109, but V#33 was BEHIND V#0 in the
        # corridor, so this check correctly does NOT pick V#33 as V#0's
        # leader. Real leaders ahead of v in the same corridor would be
        # picked up here without forward-walking each intermediate seg.)
        corridor_best = None
        corridor_dist = float('inf')
        v_cid = self._seg_to_corridor.get(cur_key)
        if v_cid is not None:
            v_corr_pos = self._corridor_offset[cur_key] + v.seg_offset
            for W in self._corridor_occupants.get(v_cid, ()):
                if W is v:
                    continue
                W_seg = (W.seg_from, W.seg_to)
                W_corr_off = self._corridor_offset.get(W_seg)
                if W_corr_off is None:
                    continue
                W_corr_pos = W_corr_off + _ext_off(W)
                d = W_corr_pos - v_corr_pos
                if d > 0 and d < corridor_dist:
                    corridor_dist = d
                    corridor_best = W

        # cur_dist = path-distance from v's current physical position to
        # path[i] (the start of the segment examined this iteration).
        # Initial value: remaining of v's current segment to reach path[v.path_idx + 1].
        if v.path_idx < len(v._seg_lengths):
            cur_dist = max(0.0, v._seg_lengths[v.path_idx] - v.seg_offset)
        else:
            cur_dist = 0.0

        # Same-path candidates and (narrowly-scoped) cross-branch fallback.
        # Cross-branch peek is performed ONLY at the un-claimed ZCU boundary
        # where the walk would otherwise stop with no leader — this is the
        # specific scenario the user reported (V_lead diverged onto another
        # branch right at the boundary v is approaching). Peeking at every
        # forward diverge causes spurious leader assignments that shift
        # downstream timing and produce marginal overlap regressions
        # elsewhere (V#6↔V#61 657mm at seg 1681-1682). Narrow scope keeps
        # the fix targeted.
        same_best = None
        same_dist = float('inf')
        cross_best = None
        cross_dist = float('inf')
        # Closest STOP vehicle dist (regardless of whether it's the chosen
        # leader). Captured during the full walk; used to cap plan_boundary
        # past a moving leader. Final value written to v.forward_stop_cap.
        forward_stop_dist = float('inf')
        walk_completed = True   # False if loop broke early (boundary, etc.)

        # Distance cap: a leader beyond this distance cannot constrain v's
        # current commit (commit_horizon = brake_dist_v_max). Adding h_min
        # gives the leader 1.15m margin so v at v_max can brake to stop
        # h_min behind a stationary leader. Caps spurious far-away
        # dependencies that otherwise produce closed leader cycles —
        # idle_n200_seed99_disp had ~77% of STOP/leader vehicles in cycles
        # at T=600 before this bound was introduced.
        brake_dist_v_max = v.v_max * v.v_max / (2 * v.d_max)
        leader_walk_cap = brake_dist_v_max + v.h_min

        for i in range(v.path_idx + 1,
                       min(v.path_idx + 50, len(v.path) - 1)):
            if cur_dist > leader_walk_cap:
                # Past physical relevance — any further candidate cannot
                # bind v's commit. Mark walk as incomplete so path-end-peek
                # (which assumes cur_dist measures distance to path[-1])
                # does not fire with an intermediate cur_dist.
                walk_completed = False
                break
            fn = v.path[i]
            tn = v.path[i + 1] if i + 1 < len(v.path) else None
            if tn is None:
                walk_completed = False
                break

            # Boundary stop: plan can't extend past. Peek cross-branch
            # successors of this boundary node (only here) and break.
            if fn in self._boundary_nodes and fn not in v.passed_zcu:
                zones = self._relevant_zones(v, fn)
                if any(self._zone_lock.get(lid) is not v
                       for _z, lid in zones):
                    # Peek ALL successors of fn (including same-direction tn).
                    # Walk-stop prevents seeing the same-path leader (fn, tn)
                    # in the normal forward branch check below — without this
                    # peek, follower brakes at the boundary even when the
                    # leader is just past it on v's planned branch
                    # (V#115/V#152 case at 1778_diverge).
                    for succ in self.gmap.adj.get(fn, []):
                        for other in self._seg_occupants.get((fn, succ), []):
                            if other is v:
                                continue
                            d = cur_dist + _ext_off(other)
                            if 0 < d < cross_dist:
                                cross_dist = d
                                cross_best = other
                    walk_completed = False
                    break

            fwd_key = (fn, tn)
            if fwd_key == cur_key:
                # Loop detected — v cannot be its own leader
                v.leader = None
                v.leader_dist = float('inf')
                self._sync_followers(v, old_leader)
                return

            for other in self._seg_occupants.get(fwd_key, []):
                if other is v:
                    continue
                d = cur_dist + _ext_off(other)
                if 0 < d < same_dist:
                    same_dist = d
                    same_best = other
                # Track closest STOP vehicle (regardless of whether it's
                # the leader) so plan_boundary can cap past a moving
                # leader to a stationary obstacle further ahead.
                if other.state == STOP and 0 < d < forward_stop_dist:
                    forward_stop_dist = d

            seg_len = (v._seg_lengths[i] if i < len(v._seg_lengths)
                       else (self.gmap.segment_between(fn, tn).length
                             if self.gmap.segment_between(fn, tn) else 0))
            # Do NOT break on first leader. Walk to the cap so a STOP
            # vehicle further ahead (V#113 past leader V#191 case) is
            # captured in forward_stop_dist; otherwise plan_boundary
            # depends only on the closest moving leader, which can
            # cross-branch off v's path and leave v committed past a
            # known stationary obstacle.
            cur_dist += seg_len

        # Path-end peek: vehicles parked AT or just past v.path[-1] are not
        # on any segment of v's forward path, so the loop above never sees
        # them, but they will physically collide with us at end-of-path.
        # Walk past path[-1] up to PUSH_PEEK_HORIZON additional distance
        # and inspect each (cur, succ) segment's occupants. This catches
        # both "parked exactly on dest node" (same case as 1-segment peek)
        # and "parked one+ segments past dest with co-located coordinates"
        # (e.g. nodes 335→3350001 are 220mm apart in space — separate
        # graph nodes but the same physical point).
        # Guard: only when the walk completed naturally, so cur_dist
        # actually measures the distance to path[-1]. If the walk broke
        # at a ZCU boundary, cur_dist is short and a peek would falsely
        # measure something far away as close.
        if (same_best is None and walk_completed
                and len(v.path) >= 2):
            PEEK_HORIZON = v.h_min + 500.0   # mm past path[-1]
            extra = 0.0
            cur = v.path[-1]
            # Only block re-entering v.path's FUTURE (already scanned by the
            # forward walk above). Including historical nodes (set(v.path))
            # over-blocks: V#82 seed=99 t=6993.5 case — v.path's historical
            # part contained 33990001, so peek bailed at iter 0 and missed
            # V#165 sitting on (33990001, 33990003) just past path[-1]=3399.
            seen = set(v.path[v.path_idx:])
            for _ in range(10):   # bounded depth as safety
                for succ in self.gmap.adj.get(cur, []):
                    for other in self._seg_occupants.get((cur, succ), []):
                        if other is v:
                            continue
                        d = cur_dist + extra + _ext_off(other)
                        if 0 < d < cross_dist and d < cur_dist + PEEK_HORIZON:
                            cross_dist = d
                            cross_best = other
                # Step one segment forward (primary successor)
                succs = self.gmap.adj.get(cur, [])
                if not succs:
                    break
                nxt = succs[0]
                if nxt in seen:
                    break
                seg = self.gmap.segment_between(cur, nxt)
                if seg is None:
                    break
                extra += seg.length
                if extra >= PEEK_HORIZON:
                    break
                seen.add(nxt)
                cur = nxt

        # Prefer same-path leader; fall back to cross-branch only if absent.
        # Corridor-based candidate (Phase 2): if it's closer than same_best,
        # use it. This catches OHTs ahead of v in the same corridor that
        # forward walk's per-segment _seg_occupants query happened to miss
        # — typically a future-merging vehicle that hasn't physically
        # entered v's path yet but already shares a corridor.
        if corridor_best is not None and corridor_dist < same_dist:
            same_best = corridor_best
            same_dist = corridor_dist
            if corridor_best.state == STOP and corridor_dist < forward_stop_dist:
                forward_stop_dist = corridor_dist
        if same_best is not None:
            v.leader = same_best
            v.leader_dist = same_dist
        elif cross_best is not None:
            v.leader = cross_best
            v.leader_dist = cross_dist
        else:
            v.leader = None
            v.leader_dist = float('inf')
        # Publish forward stop cap (closest STOP V on forward path).
        # plan_boundary uses this to avoid committing past a stationary
        # obstacle that the moving leader may eventually leave behind
        # (V#119 stacking on V#113 past V#191 case).
        v.forward_stop_cap = forward_stop_dist
        self._sync_followers(v, old_leader)

        # Phantom-stop guard: V 가 STOP 상태이고 stop_reason='leader' 인데
        # 새로 계산된 leader 가 None 이면 *옛 leader 가 사라진 phantom-stop*.
        # stop_reason 만 그대로 두면 V 가 *옛 leader 가 여전히 막고 있다*
        # 고 인식해 영구 stop. stop_reason reset + EV_REPLAN 으로 깨움.
        # (V#173 timeout-reroute 사례에서 노출됨)
        if v.state == STOP and v.stop_reason == 'leader' and v.leader is None:
            v.stop_reason = None
            self._post(t, EV_REPLAN, v)

    def _sync_followers(self, v: Vehicle, old_leader: Optional[Vehicle]):
        """Update _followers reverse mapping when v.leader changes."""
        if old_leader is v.leader:
            return
        if old_leader is not None:
            self._followers[old_leader.id].discard(v)
        if v.leader is not None:
            self._followers[v.leader.id].add(v)
        if (self._trace_match(v.id)
                or (old_leader is not None and self._trace_match(old_leader.id))
                or (v.leader is not None and self._trace_match(v.leader.id))):
            old_id = old_leader.id if old_leader is not None else None
            new_id = v.leader.id if v.leader is not None else None
            self._trace(f"  [LEADER_CHG] t={self.sim_time:.4f} V#{v.id} "
                        f"leader: V#{old_id} -> V#{new_id}")

    def _notify_followers(self, t: float, leader: Vehicle):
        """Wake every follower of `leader` to replan.

        Phase 1.7 finding: blanket-disabling notify breaks BL-stopped
        followers (commit_end_t <= t = dead commit). With dead commit,
        the follower has nothing in heap to drive forward; it needs an
        EV_REPLAN to wake up when leader moves away.

        Live-commit followers receive EV_REPLAN but _replan early-exits
        cheaply (8k+ skips/run, ~no cost). The post itself is the wake
        mechanism we cannot remove.

        Sort by id for determinism (set iteration is hash-seed-dependent;
        Event.seq breaks t-ties so post order matters).
        """
        followers = sorted(self._followers.get(leader.id, set()),
                           key=lambda f: f.id)
        if self._trace_match(leader.id):
            self._trace(f"  [NOTIFY_F] t={t:.4f} leader=V#{leader.id} "
                        f"followers={[f.id for f in followers]}")
        leader_followers_of_leader = self._followers.get(leader.id, set())
        for follower in followers:
            # Mutual-link guard: if leader is in follower's _followers set
            # (i.e., leader is *also* a follower of `follower`), this is a
            # mutual leader cycle. Notifying would trigger an infinite
            # ping-pong cascade (V#11/V#29 SPLIT case). Skip the post.
            if leader in self._followers.get(follower.id, set()):
                if self._trace_match(follower.id) or self._trace_match(leader.id):
                    self._trace(f"    [NOTIFY_MUTUAL_SKIP] leader=V#{leader.id} "
                                f"<-> follower=V#{follower.id}")
                continue
            # Dedup: only one EV_REPLAN per follower per sim_time.
            if follower.last_notify_post_t == t:
                if self._trace_match(follower.id) or self._trace_match(leader.id):
                    self._trace(f"    [NOTIFY_DEDUP] V#{follower.id} "
                                f"already notified at t={t:.4f}")
                continue
            follower.last_notify_post_t = t
            self._post(t, EV_REPLAN, follower)

    def assign_leaders(self):
        """Initial leader assignment using sorted segment queues."""
        for v in self.vehicles.values():
            self._update_leader(v, 0.0)

    def gap_from_pos(self, v: Vehicle, pidx: int, offset: float,
                     leader: Vehicle, t: float) -> Tuple[float, float]:
        """Compute gap from a simulated position (pidx, offset) to leader at time t."""
        # Leader position (extrapolated). Cap walk at last real segment —
        # same phantom-self-loop trap as gap() (V#79/V#154 case).
        l_off = leader.seg_offset + leader._dist_traveled(t - leader.t_ref)
        l_pidx = leader.path_idx
        while l_pidx < len(leader.path) - 2:
            sl = leader._seg_lengths[l_pidx] if l_pidx < len(leader._seg_lengths) else 0
            if sl <= 0 or l_off < sl - SEG_CROSS_EPS:
                break
            l_off -= sl
            l_pidx += 1
        l_seg_from = leader.path[l_pidx] if l_pidx < len(leader.path) else leader.path[-1]
        l_seg_to = leader.path[l_pidx + 1] if l_pidx + 1 < len(leader.path) else leader.path[-1]

        # Walk forward from (pidx, offset) to find leader's segment
        dist = 0.0
        if pidx < len(v._seg_lengths):
            dist += v._seg_lengths[pidx] - offset
        for i in range(pidx + 1, min(pidx + 80, len(v.path) - 1)):
            fn = v.path[i]
            tn = v.path[i + 1] if i + 1 < len(v.path) else None
            if tn is None:
                break
            # fn==l_seg_from covers both same-branch (tn==l_seg_to) and
            # cross-branch via diverge at fn (tn != l_seg_to). In both
            # cases dist+l_off is the path-through-fn distance to leader.
            if fn == l_seg_from:
                return max(0, dist + l_off), leader.vel_at(t)
            if i < len(v._seg_lengths):
                dist += v._seg_lengths[i]
            else:
                seg = self.gmap.segment_between(fn, tn)
                dist += seg.length if seg else 0
            if dist > 100000:
                break

        # Also check committed_segs of leader
        for (*_t, seg_key, plan_dist) in leader.committed_segs:
            for i in range(pidx, min(pidx + 80, len(v.path) - 1)):
                fn = v.path[i]
                tn = v.path[i + 1] if i + 1 < len(v.path) else None
                if tn is None:
                    break
                if (fn, tn) == seg_key:
                    d = 0.0
                    if i == pidx:
                        d = -offset
                    else:
                        d = v._seg_lengths[pidx] - offset if pidx < len(v._seg_lengths) else 0
                        for j in range(pidx + 1, i):
                            if j < len(v._seg_lengths):
                                d += v._seg_lengths[j]
                    return max(0, d), leader.vel_at(t)

        return float('inf'), 0.0

    def gap(self, follower: Vehicle, t: float) -> Tuple[float, float]:
        """Path-distance from follower to its leader.

        Reuses `follower.leader_dist` cached by `_update_leader` — both functions
        must agree on the leader's path-distance (V#124/V#63 seed=1 case:
        path-end peek in _update_leader detected V#63 at 110mm but the
        independent gap() walk returned inf, so leader_free was inf and V#124
        drove into V#63). Reusing the cache makes detection and distance
        consistent by construction.

        Caller contract: _update_leader was called recently for the same t.
        Returns (path_distance, leader_vel_at_t). When no leader: (inf, 0).
        """
        leader = follower.leader
        if leader is None:
            return float('inf'), 0.0
        if follower.leader_dist < float('inf'):
            return follower.leader_dist, leader.vel_at(t)
        # Fallback: recompute (defensive; should rarely hit if _update_leader
        # ran first).

        f_off = follower.seg_offset + follower._dist_traveled(t - follower.t_ref)
        l_off = leader.seg_offset + leader._dist_traveled(t - leader.t_ref)

        # Advance follower through segments to handle extrapolation overshoot.
        # Stop at the LAST real segment (path[len-2]→path[len-1]). Advancing
        # past it produces a phantom (path[-1], path[-1]) self-loop whose
        # seg_from is the dest_node — and the forward-walk below excludes
        # path[len-1], so follower-at-dest cases would mis-report gap.
        f_pidx = follower.path_idx
        while f_pidx < len(follower.path) - 2:
            sl = follower._seg_lengths[f_pidx] if f_pidx < len(follower._seg_lengths) else 0
            if sl <= 0 or f_off < sl - SEG_CROSS_EPS:
                break
            f_off -= sl
            f_pidx += 1

        # Advance leader through segments similarly. Same last-segment cap
        # as the follower walk above — without it, a leader parked at its
        # dest (l_off == last_seg_len) walks into a (path[-1], path[-1])
        # phantom and l_seg_from becomes the dest_node itself; the
        # follower's forward walk's `range(f_pidx+1, len-1)` excludes
        # path[len-1], so the search for fn==l_seg_from never matches and
        # gap returns inf (V#79/V#154 seed=99 t=4633.7 case: V#79 ran
        # straight INTO parked V#154 because leader_free was inf).
        l_pidx = leader.path_idx
        l_seg_from = leader.seg_from
        l_seg_to = leader.seg_to
        while l_pidx < len(leader.path) - 2:
            sl = leader._seg_lengths[l_pidx] if l_pidx < len(leader._seg_lengths) else 0
            if sl <= 0 or l_off < sl - SEG_CROSS_EPS:
                break
            l_off -= sl
            l_pidx += 1
            l_seg_from = leader.path[l_pidx]
            l_seg_to = leader.path[l_pidx + 1]

        # Same segment check (after virtual advance)
        f_seg_from = follower.path[f_pidx] if f_pidx < len(follower.path) else follower.path[-1]
        f_seg_to = follower.path[f_pidx + 1] if f_pidx + 1 < len(follower.path) else follower.path[-1]
        if f_seg_from == l_seg_from and f_seg_to == l_seg_to:
            return max(0, l_off - f_off), leader.vel_at(t)

        # Walk forward from follower's virtual position
        dist = 0.0
        if f_pidx < len(follower._seg_lengths):
            dist += follower._seg_lengths[f_pidx] - f_off
        else:
            seg = self.gmap.segment_between(f_seg_from, f_seg_to)
            dist += (seg.length if seg else 0) - f_off

        for i in range(f_pidx + 1,
                       min(f_pidx + 80, len(follower.path) - 1)):
            fn = follower.path[i]
            tn = follower.path[i + 1] if i + 1 < len(follower.path) else None
            if tn is None:
                break
            # fn==l_seg_from covers both same-branch (tn==l_seg_to) and
            # cross-branch via diverge at fn (tn != l_seg_to). In both
            # cases dist+l_off is the path-through-fn distance.
            if fn == l_seg_from:
                return max(0, dist + l_off), leader.vel_at(t)
            if i < len(follower._seg_lengths):
                dist += follower._seg_lengths[i]
            else:
                seg = self.gmap.segment_between(fn, tn)
                dist += seg.length if seg else 0
            if dist > 200000:
                break

        return float('inf'), leader.vel_at(t)

    def _leader_traj_end_x(self, leader: Vehicle, t: float,
                            gap_d: float, h_min: float) -> float:
        """Follower-frame position corresponding to leader's commit endpoint.

        Phase 2.1: leader's commit ends at brake_start (cruise vel > 0).
        Apply synthetic worst-case decel-to-stop = brake_start_d + decel_dist.
        For decel/stop-ending trajs (last_v ≈ 0), this is just last_d.
        """
        traj = leader.committed_traj
        if not traj:
            return float('inf')
        leader_dist_now = 0.0
        for i in range(len(traj)):
            ti, di, vi, ai = traj[i]
            t_next = traj[i + 1][0] if i + 1 < len(traj) else ti + 200
            if t <= t_next + 1e-9:
                dt = max(0.0, t - ti)
                if ai < 0 and vi > 0:
                    dt = min(dt, vi / abs(ai))
                leader_dist_now = di + vi * dt + 0.5 * ai * dt * dt
                break
        else:
            leader_dist_now = traj[-1][1]
        last_t_, last_d_, last_v_, last_a_ = traj[-1]
        if last_v_ > 1e-6:
            decel_dist = last_v_ * last_v_ / (2 * leader.d_max)
            commit_end_d = last_d_ + decel_dist
        else:
            commit_end_d = last_d_
        remaining = max(0.0, commit_end_d - leader_dist_now)
        return gap_d + remaining - h_min

    def _compute_wake_time_for_h_min(self, v: Vehicle, leader: Vehicle,
                                      t_now: float,
                                      gap_d_now: float) -> Optional[float]:
        """Return the earliest time at which gap reaches v.h_min, assuming
        v stays stationary at its current position and leader follows its
        committed_traj. Returns None if leader's current plan doesn't move
        it far enough to reach that gap.

        Used by _replan's "gap currently < h_min" branch to schedule a
        precise wake without relying on _notify_followers threshold logic.
        """
        import math as _m
        need_dist = v.h_min - gap_d_now
        if need_dist <= 0:
            return t_now
        traj = leader.committed_traj
        if not traj:
            return None

        # Locate leader's phase containing t_now
        idx = 0
        for i, e in enumerate(traj):
            if e[0] <= t_now + 1e-9:
                idx = i
            else:
                break
        t_i, d_i, v_i, a_i = traj[idx]
        dt = t_now - t_i
        if dt < 0: dt = 0.0
        # Clamp decel phase if it would go below v=0
        if a_i < 0 and v_i > 0:
            t_stop_phase = v_i / abs(a_i)
            if dt > t_stop_phase:
                dt = t_stop_phase
        d_at_now = d_i + v_i * dt + 0.5 * a_i * dt * dt
        v_at_now = max(0.0, v_i + a_i * dt)
        target_d = d_at_now + need_dist

        # Walk phases forward including partial current phase
        for pi in range(idx, len(traj)):
            if pi == idx:
                t_start = t_now
                d_start = d_at_now
                v_start = v_at_now
                a = a_i
            else:
                t_start = traj[pi][0]
                d_start = traj[pi][1]
                v_start = traj[pi][2]
                a = traj[pi][3]

            if pi + 1 < len(traj):
                t_end = traj[pi + 1][0]
                d_end_phase = traj[pi + 1][1]
            else:
                # Last phase — extrapolate
                if a < -1e-9 and v_start > 1e-9:
                    t_stop = v_start / abs(a)
                    t_end = t_start + t_stop
                    d_end_phase = d_start + v_start * t_stop + 0.5 * a * t_stop * t_stop
                elif v_start > 1e-9:
                    t_end = float('inf')
                    d_end_phase = float('inf')
                else:
                    return None   # stopped, can't progress

            # Does this phase reach target_d?
            if d_end_phase < target_d - 1e-9:
                continue

            # Solve: d_start + v_start·τ + 0.5·a·τ² = target_d, τ ≥ 0
            dd = target_d - d_start
            if dd <= 1e-9:
                return t_start
            if abs(a) < 1e-9:
                if v_start < 1e-9:
                    continue
                return t_start + dd / v_start
            A = 0.5 * a
            B = v_start
            C = -dd
            disc = B * B - 4.0 * A * C
            if disc < 0:
                continue
            sq = _m.sqrt(disc)
            candidates = [(-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A)]
            candidates = [x for x in candidates if x >= -1e-9]
            if not candidates:
                continue
            return t_start + max(0.0, min(candidates))

        return None

    def _leader_committed_remaining(self, leader: Vehicle, t: float) -> float:
        """How much further the leader will travel based on its committed_traj.

        Returns the remaining committed distance from the leader's current
        position to the end of its committed plan. If committed_traj is empty,
        falls back to next_event_t based estimate.
        """
        traj = leader.committed_traj
        if not traj:
            # Fallback: use next_event_t
            t_event = leader.next_event_t
            if t_event <= t:
                return 0.0
            dist_now = leader._dist_traveled(t - leader.t_ref)
            dist_event = leader._dist_traveled(t_event - leader.t_ref)
            return max(0, dist_event - dist_now)

        # Find leader's current distance within committed_traj
        leader_dist_now = 0.0
        for i in range(len(traj)):
            ti, di, vi, ai = traj[i]
            t_next = traj[i + 1][0] if i + 1 < len(traj) else ti + 200
            if t <= t_next + 1e-9:
                dt = max(0, t - ti)
                if ai < 0 and vi > 0:
                    dt = min(dt, vi / abs(ai))
                leader_dist_now = di + vi * dt + 0.5 * ai * dt * dt
                break
        else:
            leader_dist_now = traj[-1][1]

        # Total committed distance = last entry's distance
        # (already includes braking phase if plan ends with BOUNDARY→STOPPED)
        leader_dist_end = traj[-1][1]

        return max(0, leader_dist_end - leader_dist_now)

    # ── ZCU lock ──────────────────────────────────────────────────────────

    def _zone_request(self, v: Vehicle, lock_id: str) -> bool:
        holder = self._zone_lock.get(lock_id)
        if holder is None or holder is v:
            self._zone_lock[lock_id] = v
            if self._trace_match(v.id):
                self._trace(f"  [LOCK_ACQ] t={self.sim_time:.4f} V#{v.id} "
                            f"lock={lock_id} (prev={'V#'+str(holder.id) if holder else 'None'})")
            return True
        if self._trace_match(v.id) or (holder is not None
                                        and self._trace_match(holder.id)):
            self._trace(f"  [LOCK_DENY] t={self.sim_time:.4f} V#{v.id} "
                        f"lock={lock_id} held_by=V#{holder.id}")
        return False

    def _try_acquire_all_zones(self, v: Vehicle, zones, t: float):
        """Atomic all-or-nothing acquisition across zones at a boundary.

        Tries to acquire every lock in `zones`. If any is denied, rolls
        back all locks that were newly acquired in this call and returns
        the denied lock_id. Returns None on full success.

        Partial-hold (holding only some of the zone locks at a shared
        boundary) is a classic deadlock pattern: vehicle can't proceed
        (needs all) but blocks others who need what it's holding.
        Atomic acquire prevents this.
        """
        newly_acquired = []
        for _zone, lock_id in zones:
            if self._zone_lock.get(lock_id) is v:
                continue   # already held from before this call
            if self._zone_request(v, lock_id):
                newly_acquired.append(lock_id)
            else:
                for lid in newly_acquired:
                    self._zone_release(t, lid)
                return lock_id
        return None

    def _zone_holder_exit_time(self, lock_id: str) -> float:
        """Get the committed exit time of the current lock holder.

        The holder's next_event_t is committed (never cancelled).
        Walk the holder's committed events to estimate when it reaches
        the exit node. Conservative: use next_event_t + braking time
        as the earliest possible exit.
        """
        holder = self._zone_lock.get(lock_id)
        if holder is None:
            return 0.0
        # The holder's committed trajectory: it will at minimum travel
        # until next_event_t. At that point it has vel_at(next_event_t),
        # and may continue further. The exit happens when holder reaches
        # the exit node via SEG_END.
        # Best estimate: holder's next_event_t (conservative lower bound)
        return holder.next_event_t

    def _zone_release(self, t: float, lock_id: str):
        prev = self._zone_lock.get(lock_id)
        waiters = self._zone_waiters.get(lock_id, [])
        # Path-aware grant: skip stale waiters whose current path no
        # longer requires this lock (e.g. they were rerouted via push).
        # Without this filter, granting to a stale waiter leaks the lock
        # because they never enter the zone to trigger EV_ZCU_EXIT.
        exit_nodes = self._lock_exit_nodes.get(lock_id, set())
        while waiters:
            next_v = waiters.pop(0)
            cur_path_set = set(next_v.path[next_v.path_idx:])
            if exit_nodes and not (exit_nodes & cur_path_set):
                # Stale: path no longer crosses this lock's zone.
                next_v.waiting_at_zcu = None
                if self._trace_match(next_v.id):
                    self._trace(f"  [LOCK_STALE_SKIP] t={t:.4f} lock={lock_id} "
                                f"skipped stale V#{next_v.id}")
                continue
            self._zone_lock[lock_id] = next_v  # direct transfer, no gap
            next_v.waiting_at_zcu = None
            self._post(t, EV_ZCU_GRANT, next_v)
            if (self._trace_match(next_v.id)
                    or (prev is not None and self._trace_match(prev.id))):
                self._trace(f"  [LOCK_XFER] t={t:.4f} lock={lock_id} "
                            f"V#{prev.id if prev else '?'}->V#{next_v.id} "
                            f"queue_left={[w.id for w in waiters]}")
            return
        # No eligible waiter
        self._zone_lock[lock_id] = None
        if prev is not None and self._trace_match(prev.id):
            self._trace(f"  [LOCK_REL] t={t:.4f} V#{prev.id} "
                        f"lock={lock_id} (no waiters)")

    def _zone_wait(self, v: Vehicle, lock_id: str):
        v.waiting_at_zcu = lock_id
        if v not in self._zone_waiters[lock_id]:
            self._zone_waiters[lock_id].append(v)
            if self._trace_match(v.id):
                holder = self._zone_lock.get(lock_id)
                self._trace(f"  [LOCK_WAIT] t={self.sim_time:.4f} V#{v.id} "
                            f"lock={lock_id} held_by="
                            f"{'V#'+str(holder.id) if holder else 'None'} "
                            f"queue={[w.id for w in self._zone_waiters[lock_id]]}")
        # Automod-style timeout reroute: V 가 lock_id 의 waiter 로 등록된
        # 후 REROUTE_TIMEOUT 동안 grant 못 받으면 alt branch 로 reroute 시도.
        # data = (lock_id, reroute_count) tuple — fire 시 stale check 에 사용.
        # reroute_count 가 *그 사이 reroute 발동* 시 변화 → 옛 EV_TIMEOUT
        # 자동 stale.
        if v.reroute_count < MAX_REROUTE_PER_TASK:
            self._post(self.sim_time + REROUTE_TIMEOUT, EV_TIMEOUT, v,
                       data=(lock_id, v.reroute_count))

    # ── Timeout-based reroute (Automod-style) ────────────────────────────

    def _on_timeout(self, t: float, v: Vehicle, data):
        """EV_TIMEOUT handler. ZCU stuck V 의 reroute 시도.
        data = (lock_id, reroute_count) tuple. reroute_count 변하면 stale."""
        # data 가 tuple — backward compat 안 둠 (= 새 EV_TIMEOUT 만 처리)
        if not isinstance(data, tuple):
            return
        post_lock_id, post_count = data
        # ── Stale check (state-based, DES 표준 패턴) ────────────────
        if v.state != STOP:
            return   # V 가 이미 진행 중
        if v.waiting_at_zcu != post_lock_id:
            return   # V 가 다른 lock 기다림 또는 이미 grant 받음
        if v.reroute_count != post_count:
            return   # V 가 이미 reroute 함 (이 EV_TIMEOUT 은 옛 token)
        if v.reroute_count >= MAX_REROUTE_PER_TASK:
            return   # 무한 reroute 방지

        # ── 진짜 timeout — alt branch 시도 ──────────────────────────
        self._try_reroute_at_diverge(t, v)

    def _try_reroute_at_diverge(self, t: float, v: Vehicle):
        """현재 위치 다음 분기점에서 alt branch 로 reroute 시도.
        다음 조건 모두 만족 시 발동:
          1. cur_seg 의 다음 노드 (= seg_to) 가 분기점 (succ ≥ 2)
          2. alt branch 가 graph 상 dest 까지 도달 가능
          3. alt branch 의 zone lock 이 다른 V 에 잡혀있지 않음
        """
        if v.path_idx + 2 >= len(v.path):
            return  # path 끝 근처 — 다음 분기 없음
        fork_node = v.seg_to
        succs = list(self.gmap.adj.get(fork_node, []))
        if len(succs) < 2:
            return  # 분기 없음

        cur_next = v.path[v.path_idx + 2]
        alt_candidates = [s for s in succs if s != cur_next]
        if not alt_candidates:
            return

        dest_node = v.path[-1] if v.path else None
        if dest_node is None:
            return

        # 각 alt 검사 — lock free + occupancy + reachability
        for alt in alt_candidates:
            alt_seg = (fork_node, alt)
            zone_lids = [lid for _z, lid in
                          self._seg_to_zone.get(alt_seg, [])]
            # alt 의 zone 이 *다른* V 에 잡혔으면 skip (= 같은 cycle 위험)
            blocked = any(self._zone_lock.get(lid) not in (None, v)
                          for lid in zone_lids)
            if blocked:
                continue
            # alt 에서 dest 까지 path 계산.
            # cur_next 방향을 차단하여 reroute 경로가 *cur_next 다시 통과*
            # 안 하도록. 그렇지 않으면 BFS 가 cycle 형태 path 반환 가능.
            new_tail = self._shortest_path_bfs(alt, dest_node,
                                                blocked={cur_next})
            if new_tail is None:
                continue
            # 새 path 전방 안전거리(h_min) 검사: reroute 는 정상 plan(CVP)
            # 의 h_min 보장을 우회하므로, V 를 *전방 OHT 의 h_min 안에*
            # 삽입하면 즉시 gap violation. alt 첫 seg 만이 아니라 새 path
            # 의 h_min+margin 거리 내 전방 OHT 가 있으면 그 alt 거부.
            if not self._reroute_path_clear(v, fork_node, new_tail):
                continue
            # 성공 — reroute apply
            self._apply_reroute(t, v, fork_node, alt, new_tail)
            return
        # 모든 alt 실패 — 그냥 둠

    def _reroute_path_clear(self, v: Vehicle, fork_node: str,
                            new_tail: list) -> bool:
        """reroute 후 V 가 (fork → new_tail) 를 따라갈 때, 전방
        h_min+margin 거리 안에 다른 OHT 가 있으면 False (= 그 alt 거부).
        정상 plan(CVP)이 leader.committed_traj 기준으로 h_min 을 보장하는
        것처럼, reroute 도 새 path 의 전방 차와 h_min 이 확보되는 alt 만
        선택하도록 한다."""
        MARGIN = 1000.0
        horizon = v.h_min + MARGIN
        # V → fork 까지 거리 (V 는 fork=seg_to 직전에 정지/접근 중)
        cur_len = v.current_seg_length()
        cum = max(0.0, cur_len - v.seg_offset)
        prev = fork_node
        for nxt in new_tail:           # new_tail[0] = alt
            if cum > horizon:
                break
            seg = (prev, nxt)
            seg_obj = self.gmap.segments.get(seg)
            if seg_obj is None:
                break
            for o in self._seg_occupants.get(seg, []):
                if o is v:
                    continue
                # o 의 V 기준 전방 거리 = (V→fork) + (fork→o)
                o_dist = cum + o.seg_offset
                if o_dist < v.h_min:
                    return False       # h_min 이내 전방 OHT → reroute 거부
            cum += seg_obj.length
            prev = nxt
        return True

    def _shortest_path_bfs(self, from_node: str, to_node: str, blocked=None):
        """간단 BFS — graph 거리 최단. 도달 불가 시 None.
        blocked: 방문 금지 노드 set (= reroute 시 cur_next 방향 차단)."""
        if from_node == to_node:
            return [from_node]
        visited = {from_node}
        if blocked:
            visited.update(blocked)
        queue = collections.deque([(from_node, [from_node])])
        while queue:
            cur, path = queue.popleft()
            for succ in self.gmap.adj.get(cur, []):
                if succ in visited:
                    continue
                new_path = path + [succ]
                if succ == to_node:
                    return new_path
                visited.add(succ)
                queue.append((succ, new_path))
        return None

    def _apply_reroute(self, t: float, v: Vehicle, fork_node: str,
                       alt: str, new_tail):
        """waiter queue 정리 + path 갱신 + EV_REPLAN."""
        # 1. waiter queue 에서 V 제거
        old_lock = v.waiting_at_zcu
        if old_lock:
            queue = self._zone_waiters.get(old_lock)
            if queue and v in queue:
                queue.remove(v)
            v.waiting_at_zcu = None

        # 2. path 의 path_idx+2 이후를 new_tail 로 교체
        # new_tail 의 첫 노드 = alt. 기존 path 의 path_idx+2 자리에 들어감.
        new_path = list(v.path[:v.path_idx + 2]) + list(new_tail)
        v.path = new_path
        # _seg_lengths cache rebuild
        if hasattr(v, '_rebuild_seg_cache'):
            v._rebuild_seg_cache()
        else:
            v._seg_lengths = [
                self.gmap.segments[(new_path[i], new_path[i + 1])].length
                if (new_path[i], new_path[i + 1]) in self.gmap.segments else 0.0
                for i in range(len(new_path) - 1)
            ]

        # 3. stop_reason reset + counter increment
        v.stop_reason = None
        v.reroute_count += 1

        # 4. EV_REPLAN trigger (= 새 plan 생성 + ACCEL)
        self._post(t, EV_REPLAN, v)

        # 5. Follower wake-up — V 의 path 변경으로 옛 follower 들의 leader
        # pointer 가 stale. _notify_followers 가 EV_REPLAN post 해서 그들의
        # _replan 안에서 _update_leader 가 호출되어 stale leader 정리.
        # 이 누락 시 follower 들이 stop_reason='leader' + leader=None 의
        # 영구 stop 상태로 빠짐 (V#173 case).
        self._notify_followers(t, v)

        print(f"[REROUTE t={t:.1f}] V#{v.id} fork={fork_node} "
              f"original_next={'?'} alt={alt} "
              f"new_path_len={len(new_path)} (count={v.reroute_count}/{MAX_REROUTE_PER_TASK})",
              flush=True)

    def _release_passed_diverge_locks(self, t: float, v: Vehicle):
        """Release diverge locks held by v for diverge ZONES v is fully
        past — v is stopped at a downstream boundary waiting on another
        lock.

        Subtlety: a diverge ZONE includes the segments AFTER the diverge
        node (zone.all_segs() = (diverge, succ) pairs).

        Correct predicate (current): the diverge node is NOT in v's
        forward path. If it IS forward, v will still need to cross it
        and must keep the lock.

        Why not v.passed_zcu (prior heuristic)? _assign_destination's
        passed_zcu cleanup (line ~920) removes forward nodes from
        passed_zcu on path change so each upcoming boundary re-arms
        for lock acquisition. That cleanup makes "node not in
        passed_zcu" ambiguous — could mean "already crossed" OR
        "freshly re-armed after reroute, not yet crossed". The latter
        case caused V#64 seed=1 to release 1187_diverge while still on
        the pre-zone segment (1186, 1187), then advance into (1187,
        1189) without a holder → NO_LOCK at t=1364. Use forward-path
        membership instead, which is unambiguous.
        """
        held = [lid for lid, h in self._zone_lock.items() if h is v]
        cur_seg = (v.seg_from, v.seg_to)
        zones_here = {zlid for _z, zlid in self._seg_to_zone.get(cur_seg, [])}
        fwd_path = set(v.path[v.path_idx:])
        for lock_id in held:
            if not lock_id.endswith('_diverge'):
                continue
            if lock_id in zones_here:
                continue  # still inside this zone
            node_id = lock_id[:-len('_diverge')]
            if node_id not in fwd_path:
                self._zone_release(t, lock_id)

    # ── Segment occupancy ─────────────────────────────────────────────────

    def _update_occupancy(self, v: Vehicle, old_key: Tuple[str, str],
                          new_key: Tuple[str, str], t: float,
                          crossed: Optional[List[str]] = None):
        if old_key == new_key:
            return
        if v in self._seg_occupants[old_key]:
            self._seg_occupants[old_key].remove(v)
        if v not in self._seg_occupants[new_key]:
            self._seg_occupants[new_key].append(v)
        # Corridor membership: update when v crosses between corridors. ZCU-
        # zone segments are not in any corridor (None), so transitions into
        # a ZCU remove v from corridor occupants; transitions back out add
        # to the new corridor. Same-corridor transitions (most common) are
        # no-ops here.
        old_cid = self._seg_to_corridor.get(old_key)
        new_cid = self._seg_to_corridor.get(new_key)
        if old_cid != new_cid:
            if old_cid is not None:
                self._corridor_occupants[old_cid].discard(v)
            if new_cid is not None:
                self._corridor_occupants[new_cid].add(v)
        # Safety net: release any held lock whose exit node v just
        # crossed. EV_ZCU_EXIT is scheduled at plan time and discarded
        # after first fire; if that fire was stale, no further release
        # attempt happens → lock leaks (V#193 case). Catch actual cross.
        # Multi-seg jumps pass `crossed` so exit nodes traversed in the
        # middle of the jump (V#28: 4777 between (4775,4777) and
        # (47770001,47770003)) are not missed.
        crossed_set = {new_key[0]}
        if crossed:
            crossed_set |= set(crossed)
        held = [lid for lid, h in self._zone_lock.items() if h is v]
        for lid in held:
            exit_nodes = self._lock_exit_nodes.get(lid)
            if exit_nodes and (exit_nodes & crossed_set):
                self._zone_release(t, lid)

    # ── Committed trajectory management ──────────────────────────────────

    def _post_zcu_exit_events(self, v: Vehicle, c_segs: list):
        """Post EV_ZCU_EXIT for every held lock's exit node in c_segs."""
        held = [lid for lid, holder in self._zone_lock.items() if holder is v]
        if not held:
            return
        held_set = set(held)
        exit_targets: Dict[str, Set[str]] = {}
        for ex_node, zones in self._exit_to_zones.items():
            for _zone, zlid in zones:
                if zlid in held_set:
                    exit_targets.setdefault(ex_node, set()).add(zlid)
        if not exit_targets:
            return
        for (t_enter, _t_exit, seg_key, _plan_dist) in c_segs:
            from_node = seg_key[0]
            if from_node in exit_targets:
                # sorted() for determinism: set iteration order is
                # hash-seed-dependent, but EV_ZCU_EXIT posting order is
                # captured by Event.seq and affects dispatch tiebreaking.
                for lid in sorted(exit_targets[from_node]):
                    self._post(t_enter + 1e-6, EV_ZCU_EXIT, v, data=lid)
                exit_targets.pop(from_node, None)
                if not exit_targets:
                    return

    def _lookup_committed_acc(self, v: Vehicle, t: float) -> Optional[float]:
        """Return planned acc value at time t from committed_traj.

        Used by _on_phase_done / _on_seg_end to sync runtime acc with the
        committed plan, so the simulation loop can change sim_acc without
        re-entering _replan.

        Now backed by vehicle_state.state_at — same result, O(log n)
        binary search, and exercises state_at on the production path.
        """
        s = state_at(v, t)
        return s.acc if s is not None else None

    def _leader_traj_from_now(self, leader: Vehicle, t: float):
        """Return leader.committed_traj sliced to entries at/after t.

        Phase 2.1: leader's commit ends at brake_start (cruise vel > 0).
        Append synthetic worst-case decel-to-stop tail so follower's
        compute_velocity_profile assumes leader will stop. Conservative —
        leader may extend cruise instead (FASTER), but commit invariant
        guarantees gap stays safe under either outcome.
        """
        traj = leader.committed_traj
        if not traj:
            return None
        idx = 0
        for i, e in enumerate(traj):
            if e[0] <= t + 1e-9:
                idx = i
            else:
                break
        t_i, d_i, v_i, a_i = traj[idx]
        dt = t - t_i
        if dt <= 1e-9:
            sliced = list(traj[idx:])
        else:
            d_at_t = d_i + v_i * dt + 0.5 * a_i * dt * dt
            v_at_t = v_i + a_i * dt
            if v_at_t < 0:
                v_at_t = 0.0
            first = (t, d_at_t, v_at_t, a_i)
            sliced = [first] + list(traj[idx + 1:])
        # Synthetic decel-to-stop tail when commit ends mid-motion (cruise
        # OR still-accelerating). Without this, _compute_follower_trajectory
        # extrapolates the last phase (e.g. accel a=500) over 1e6 s,
        # producing dd_total ~10^9 mm and 5M+ sampling iterations
        # (stuttering source).
        # Conservative: assume leader at last_t_ has reached last_v_ and
        # will brake to stop. May be too restrictive if leader actually
        # continues to accelerate (= less restrictive for follower), but
        # safety wins over efficiency.
        if sliced:
            last_t_, last_d_, last_v_, last_a_ = sliced[-1]
            if last_v_ > 1e-6 and last_a_ > -1e-6:
                d_max = leader.d_max
                t_stop_dur = last_v_ / d_max
                d_stop = (last_d_ + last_v_ * t_stop_dur
                          - 0.5 * d_max * t_stop_dur * t_stop_dur)
                sliced[-1] = (last_t_, last_d_, last_v_, -d_max)
                sliced.append((last_t_ + t_stop_dur, d_stop, 0.0, 0.0))
        return sliced

    def _trim_committed(self, v: Vehicle, t: float):
        """Phase 1.4 — keep ALL future entries (they are the commit).
        Only trim old history (>30s) for memory bound.

        Pairs with _replan early-exit when v.commit_end_t > t: when
        commit is alive, we don't replan, so future entries stay valid
        and old in-heap events drive the vehicle forward."""
        if not v.committed_traj:
            v.committed_traj_t0 = t
            return
        # Trim old history to bound memory (keep last 30s)
        cutoff = t - 30.0
        if v.committed_traj and v.committed_traj[0][0] < cutoff:
            v.committed_traj = [e for e in v.committed_traj if e[0] >= cutoff]
            v.committed_segs = [e for e in v.committed_segs if e[0] >= cutoff]
        if not v.committed_traj:
            v.committed_traj_t0 = t

    def _commit_state(self, v: Vehicle, t: float):
        """Record current state as a committed trajectory entry.

        base_dist must reflect v's CURRENT physical absolute distance, not
        the last committed entry's x (which is stale once v has advanced).
        Use state_at canonical lookup; when committed_traj is empty or last
        entry t equals current t (overwrite case), fall back to last entry x.
        """
        if v.committed_traj and abs(v.committed_traj[-1][0] - t) < 1e-9:
            # Overwrite same-t entry: position hasn't changed since last commit
            base_dist = v.committed_traj[-1][1]
            v.committed_traj[-1] = (t, base_dist, v.vel, v.acc)
        else:
            s_now = state_at(v, t)
            base_dist = (s_now.dist if s_now is not None
                         else (v.committed_traj[-1][1]
                               if v.committed_traj else 0.0))
            v.committed_traj.append((t, base_dist, v.vel, v.acc))

    def _truncate_commit_at(self, v: Vehicle, t_cut: float) -> None:
        """Truncate committed_traj/segs so commit ends exactly at t_cut.

        - committed_traj: drop phase entries strictly after t_cut. If the
          last kept entry is before t_cut, append a synthetic endpoint
          (t_cut, interpolated_dist, interpolated_vel, 0.0) so future
          _get_plan_start reads the correct state at t_cut.
        - committed_segs: drop entries that start after t_cut. If the
          last kept entry's t_exit > t_cut, clip its t_exit to t_cut.
        - v.commit_end_t = t_cut.

        Used by _assign_destination when v.path is swapped: the old commit
        beyond the prefix-end is no longer valid against the new path, so
        we cut it cleanly at the prefix boundary. _replan then extends
        from t_cut using the new path."""
        # Guard against invalid t_cut (would propagate NaN through
        # committed_traj into compute_velocity_profile via leader_traj).
        if not (t_cut == t_cut and t_cut != float('inf')
                and t_cut != float('-inf')):
            return   # NaN/inf → no-op
        # 1. committed_traj
        keep_traj = []
        for entry in v.committed_traj:
            if entry[0] <= t_cut + 1e-9:
                keep_traj.append(entry)
            else:
                break
        if keep_traj:
            last_ti, last_di, last_vi, last_ai = keep_traj[-1]
            if t_cut > last_ti + 1e-9:
                dt = t_cut - last_ti
                # When the last kept phase is a decel-to-stop and t_cut is
                # past the stop time, clamp dt at the natural stop time so
                # v_at doesn't go negative (max-clamped to 0 but d_at would
                # over-shoot into deceleration past stop).
                if last_ai < -1e-9 and last_vi > 1e-9:
                    dt_to_stop = -last_vi / last_ai
                    dt = min(dt, max(0.0, dt_to_stop))
                # Interpolate state under constant acc within phase
                d_at = last_di + last_vi * dt + 0.5 * last_ai * dt * dt
                v_at = max(0.0, last_vi + last_ai * dt)
                # acc=0 sentinel: this is an endpoint, not a phase boundary
                keep_traj.append((t_cut, d_at, v_at, 0.0))
        v.committed_traj = keep_traj

        # 2. committed_segs
        keep_segs = []
        for (t_enter, t_exit, sk, plan_dist) in v.committed_segs:
            if t_enter > t_cut + 1e-9:
                break
            if t_exit > t_cut + 1e-9:
                # straddles t_cut → clip
                keep_segs.append((t_enter, t_cut, sk, plan_dist))
                break
            keep_segs.append((t_enter, t_exit, sk, plan_dist))
        v.committed_segs = keep_segs

        # 3. metadata
        v.commit_end_t = t_cut

    # ── Boundary / ZCU helpers ────────────────────────────────────────────

    def _find_first_boundary(self, v: Vehicle) -> Tuple[float, int, Optional[str]]:
        return self._find_first_boundary_from(v, v.path_idx, v.seg_offset)

    def _find_first_boundary_from(self, v: Vehicle, path_idx: int,
                                   seg_offset: float,
                                   skip: Optional[Set[str]] = None
                                   ) -> Tuple[float, int, Optional[str]]:
        """Same as _find_first_boundary but starting from arbitrary
        (path_idx, seg_offset). Used by commit-end-start replan path.

        `skip` is an OPTIONAL transient skip set, layered over
        v.passed_zcu. Callers use this to iterate past
        currently-non-relevant boundaries without poisoning
        v.passed_zcu (which would mask a future re-occurrence of the
        same node where it IS relevant — V#7 1500/4381 case).
        """
        if path_idx >= len(v.path) - 1:
            return float('inf'), -1, None
        seg_len = (v._seg_lengths[path_idx]
                   if path_idx < len(v._seg_lengths) else 0)
        dist = max(0.0, seg_len - seg_offset)
        pi = path_idx
        while dist < 100000 and pi + 1 < len(v.path) - 1:
            next_node = v.path[pi + 1]
            if next_node in self._boundary_nodes and \
               next_node not in v.passed_zcu and \
               (skip is None or next_node not in skip):
                return dist, pi, next_node
            tn = v.path[pi + 2] if pi + 2 < len(v.path) else None
            if tn is None:
                break
            seg = self.gmap.segments.get((next_node, tn))
            if seg:
                dist += seg.length
            else:
                break
            pi += 1
        return float('inf'), -1, None

    def _get_plan_start(self, v: Vehicle, t: float):
        """Determine plan start state for compute_velocity_profile.

        Phase 1 (commit immutability): if v.committed_traj has entries
        beyond t, the plan must START from end-of-committed instead of
        the vehicle's current state. This preserves the already-committed
        kinematics — the next traj generated will EXTEND past the commit
        end, not overwrite it.

        Returns: (t_start, v0_start, abs_dist_start, path_idx_start,
                  seg_offset_start). When no future committed entries,
                  returns the vehicle's current state (t, v.vel, ...).
        """
        if (not v.committed_traj
                or v.committed_traj[-1][0] <= t + 1e-9):
            # Use state_at canonical lookup: when t > last_traj[0], v has
            # physically advanced past the last committed entry (advance_position
            # updated path_idx + seg_offset). The new plan must start from the
            # CURRENT physical absolute distance, not the stale last entry x.
            # Otherwise committed_traj gets entries with x lagging physical
            # position by the un-recorded motion (seed=99 t=6589.0 V#136 case:
            # last entry x=409164.51 was 403mm behind physical 409567.91 →
            # new phase x values understated → leader-frame computations off).
            # _brake_to_stop (line ~1816) already uses this canonical pattern.
            s_now = state_at(v, t)
            base_dist = (s_now.dist if s_now is not None
                         else (v.committed_traj[-1][1]
                               if v.committed_traj else 0.0))
            return (t, v.vel, base_dist, v.path_idx, v.seg_offset)

        last_traj = v.committed_traj[-1]
        t_start = last_traj[0]
        v0_start = last_traj[2]
        abs_dist_start = last_traj[1]

        # Find segment v ends on at t_start. committed_segs's last entry
        # tells us. seg_key + plan_dist + (abs_dist - plan_dist) = offset.
        if not v.committed_segs:
            return (t, v.vel, abs_dist_start, v.path_idx, v.seg_offset)
        last_seg = v.committed_segs[-1]
        _t_enter, _t_exit, seg_key, plan_dist = last_seg
        seg_offset_start = max(0.0, abs_dist_start - plan_dist)

        path_idx_start = None
        for i in range(v.path_idx, len(v.path) - 1):
            if (v.path[i], v.path[i + 1]) == seg_key:
                path_idx_start = i
                break
        if path_idx_start is None:
            # path was reassigned beyond commit end — fallback
            return (t, v.vel, abs_dist_start, v.path_idx, v.seg_offset)

        # Clamp seg_offset to segment length (drift safety)
        seg_len_start = (v._seg_lengths[path_idx_start]
                         if path_idx_start < len(v._seg_lengths) else 0.0)
        seg_offset_start = min(seg_offset_start, seg_len_start)

        return (t_start, v0_start, abs_dist_start,
                path_idx_start, seg_offset_start)

    def _pin_marker(self, v: Vehicle, pi: int, node_id: str):
        if pi < len(v._seg_lengths):
            seg_len = v._seg_lengths[pi]
        else:
            seg = v.gmap.segment_between(v.path[pi], v.path[pi + 1])
            seg_len = seg.length if seg else 0.0
        v.x_marker_pidx = pi
        v.x_marker_offset = seg_len
        v.x_marker_node = node_id

    def _pin_marker_at_dist(self, v: Vehicle, dist: float):
        remaining = dist
        pi = v.path_idx
        off = v.seg_offset + remaining
        while pi < len(v.path) - 1:
            seg_len = v._seg_lengths[pi] if pi < len(v._seg_lengths) else 0.0
            if off <= seg_len or pi >= len(v.path) - 2:
                v.x_marker_pidx = pi
                v.x_marker_offset = min(off, seg_len)
                v.x_marker_node = None
                return
            off -= seg_len
            pi += 1
        v.x_marker_pidx = max(0, len(v.path) - 2)
        v.x_marker_offset = v._seg_lengths[v.x_marker_pidx] \
            if v.x_marker_pidx < len(v._seg_lengths) else 0.0
        v.x_marker_node = None

    # ── Dest / lookahead / go ─────────────────────────────────────────────

    def _dist_to_dest(self, v: Vehicle) -> float:
        return self._dist_to_dest_from(v, v.path_idx, v.seg_offset)

    def _dist_to_dest_from(self, v: Vehicle, path_idx: int,
                            seg_offset: float) -> float:
        """Same as _dist_to_dest but starting from arbitrary
        (path_idx, seg_offset)."""
        if v.dest_node is None or v.dest_reached:
            return float('inf')
        if path_idx >= len(v.path) - 1:
            return float('inf')
        seg_len = (v._seg_lengths[path_idx]
                   if path_idx < len(v._seg_lengths) else 0)
        dist = max(0.0, seg_len - seg_offset)
        pi = path_idx
        while pi + 1 < len(v.path):
            if v.path[pi + 1] == v.dest_node:
                if dist < 1.0:
                    # Only mark reached when querying from CURRENT state
                    # (path_idx == v.path_idx). Future state queries
                    # shouldn't toggle the flag.
                    if path_idx == v.path_idx and seg_offset == v.seg_offset:
                        v.dest_reached = True
                return max(0.0, dist)
            pi += 1
            if pi + 1 < len(v.path):
                dist += v._seg_lengths[pi] if pi < len(v._seg_lengths) else 0
            if dist > 200000:
                break
        return float('inf')

    def _lookahead_speed(self, v: Vehicle) -> Tuple[float, float]:
        """DEPRECATED — analytical profile (compute_velocity_profile) replaces
        this for plan generation. Still used by _on_seg_end's SPEED_VIOL
        diagnostic to display "lookahead_v" in violation logs. Do not use for
        new plan logic; modify the velocity profile envelope instead."""
        cur_speed = v.current_seg_speed()
        dist = v.current_seg_length() - v.seg_offset
        pi = v.path_idx
        best_v = cur_speed
        best_dist = float('inf')
        max_look = v.v_max * v.v_max / (2 * v.d_max) + 2000
        while dist < max_look and pi + 1 < len(v.path) - 1:
            pi += 1
            seg_spd = v._seg_speeds[pi] if pi < len(v._seg_speeds) else v.v_max
            if seg_spd < best_v:
                # 현재 vel에서 d_max로 제동 시 해당 세그먼트 진입 시 속도
                v_entry_sq = max(0.0, v.vel * v.vel - 2 * v.d_max * dist)
                v_entry = math.sqrt(v_entry_sq)

                if v_entry > seg_spd:
                    # 이미 현재 vel로는 seg_spd에 맞춰 진입 불가
                    # 즉시 seg_spd를 target으로 설정 (최대한 빨리 감속 유도)
                    v_safe = seg_spd
                else:
                    # 정상 계산: 이 속도에서 제동 시작하면 진입 시 정확히 seg_spd
                    v_safe = math.sqrt(seg_spd * seg_spd + 2 * v.d_max * dist)

                if v_safe < best_v:
                    best_v = v_safe
                    if best_dist == float('inf'):
                        best_dist = dist
            dist += v._seg_lengths[pi] if pi < len(v._seg_lengths) else 0
        return best_v, best_dist

    def _go(self, t: float, v: Vehicle, target_v: float):
        # Use a tolerance band to prevent oscillation when lookahead
        # speed changes slightly between replans
        tol_accel = 50.0
        tol_decel = 10.0
        if v.vel < target_v - tol_accel:
            v.acc = v.a_max
            v.state = ACCEL
        elif v.vel > target_v + tol_decel:
            v.acc = -v.d_max
            v.state = DECEL
        else:
            # Within tolerance — cruise at current speed (don't snap)
            v.acc = 0.0
            v.state = CRUISE
