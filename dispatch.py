"""dispatch.py — Job creation (Poisson), nearest-IDLE matching, Dijkstra path.

Integrates with graph_des_v6: JobManager hooks to DES events.
- EV_JOB_CREATE (system): Poisson-scheduled job spawn
- EV_LOAD_DONE / EV_UNLOAD_DONE (per-vehicle): dwell completion

Vehicle lifecycle with job:
  IDLE --(assign)--> TO_PICKUP --(arrive src)--> LOADING --(load_dwell)-->
  TO_DROP --(arrive dst)--> UNLOADING --(unload_dwell)--> IDLE
"""
from __future__ import annotations
import heapq
import json
import math
import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from graph_des_v5 import GraphMap
    from graph_des_v6 import GraphDESv6, Vehicle


# ── Port ──────────────────────────────────────────────────────────────────────

@dataclass
class Port:
    id: str
    node_id: str
    kind: str           # 'Buffer' | 'Station'
    offset_x: float
    offset_y: float
    use_handshake: bool


def load_ports(map_path: str) -> Dict[str, Port]:
    with open(map_path) as f:
        data = json.load(f)
    ports: Dict[str, Port] = {}
    for p in data.get('ports', []):
        nid = p.get('nodeId')
        if nid is None:
            continue
        pos = p.get('position', {}) or {}
        port = Port(
            id=p['id'],
            node_id=nid,
            kind=p.get('kind', 'Buffer'),
            offset_x=float(pos.get('x', 0.0)),
            offset_y=float(pos.get('y', 0.0)),
            use_handshake=bool(p.get('useHandshake', False)),
        )
        ports[port.id] = port
    return ports


# ── Job ───────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    src_port_id: str
    dst_port_id: str
    src_node_id: str
    dst_node_id: str
    spawn_t: float
    state: str = 'PENDING'   # PENDING | ASSIGNED | CARRYING | DONE
    assigned_to: Optional[int] = None
    assign_t: Optional[float] = None
    pickup_t: Optional[float] = None
    drop_t: Optional[float] = None


# ── Dijkstra ──────────────────────────────────────────────────────────────────

def dijkstra_path(gmap, src: str, dst: str) -> Optional[List[str]]:
    """Shortest path from src to dst by segment length. Returns [src, ..., dst]
    or None if unreachable."""
    if src == dst:
        return [src]
    dist: Dict[str, float] = {src: 0.0}
    prev: Dict[str, str] = {}
    pq: List[Tuple[float, str]] = [(0.0, src)]
    found = False
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            found = True
            break
        if d > dist.get(u, math.inf):
            continue
        for nb in gmap.adj.get(u, []):
            seg = gmap.segment_between(u, nb)
            if not seg:
                continue
            nd = d + seg.length
            if nd < dist.get(nb, math.inf):
                dist[nb] = nd
                prev[nb] = u
                heapq.heappush(pq, (nd, nb))
    if not found:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return list(reversed(path))


def dijkstra_dist_to(gmap, dst: str) -> Dict[str, float]:
    """Returns dict node → shortest path distance FROM that node TO dst.
    Computed by reverse Dijkstra using adj_rev."""
    dist: Dict[str, float] = {dst: 0.0}
    pq: List[Tuple[float, str]] = [(0.0, dst)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        for pred in gmap.adj_rev.get(u, []):
            seg = gmap.segment_between(pred, u)
            if not seg:
                continue
            nd = d + seg.length
            if nd < dist.get(pred, math.inf):
                dist[pred] = nd
                heapq.heappush(pq, (nd, pred))
    return dist


# ── JobManager ────────────────────────────────────────────────────────────────

class JobManager:
    """Generates jobs via Poisson process and dispatches them to nearest IDLE
    OHT (by Dijkstra path-distance from vehicle's seg_to to job src_node).

    Hook events from DES:
      - on_create_event(t): EV_JOB_CREATE fired → spawn job + reschedule + dispatch
      - on_arrive(t, v):    end-of-path → start LOAD or UNLOAD dwell
      - on_load_done(t, v): EV_LOAD_DONE → reroute to drop
      - on_unload_done(t, v): EV_UNLOAD_DONE → mark done, free vehicle, dispatch next
    """

    def __init__(self, des: 'GraphDESv6', gmap: 'GraphMap',
                 ports: Dict[str, Port], lambda_rate: float,
                 load_dwell: float, unload_dwell: float,
                 rng_seed: Optional[int] = None,
                 valid_node_only: bool = True):
        self.des = des
        self.gmap = gmap
        # filter ports to those whose node exists in gmap and is reachable
        self.ports: List[Port] = []
        for p in ports.values():
            if not valid_node_only or p.node_id in gmap.nodes:
                self.ports.append(p)
        self.lambda_rate = lambda_rate
        self.load_dwell = load_dwell
        self.unload_dwell = unload_dwell
        self._rng = _random.Random(rng_seed) if rng_seed is not None else _random.Random()

        self.pending: List[Job] = []
        self.completed: List[Job] = []
        self.assigned: Dict[str, Job] = {}   # job.id → Job
        self._next_job_id = 0

        des.job_mgr = self

    # ── Public lifecycle ─────────────────────────────────────────────

    def start(self, t: float = 0.0):
        """Schedule first job creation."""
        self._schedule_next_creation(t)

    # ── Event hooks ──────────────────────────────────────────────────

    def on_create_event(self, t: float):
        from graph_des_v6 import EV_JOB_CREATE  # noqa: F401
        self._create_job(t)
        self._schedule_next_creation(t)

    def on_arrive(self, t: float, v: 'Vehicle'):
        """Called when v reaches end of its assigned path."""
        from graph_des_v6 import EV_LOAD_DONE, EV_UNLOAD_DONE, LOADING
        if v.job is None:
            return
        if v.job_state == 'TO_PICKUP':
            v.job_state = 'LOADING'
            v.state = LOADING
            self.des._post(t + self.load_dwell, EV_LOAD_DONE, v)
        elif v.job_state == 'TO_DROP':
            v.job_state = 'UNLOADING'
            v.state = LOADING
            self.des._post(t + self.unload_dwell, EV_UNLOAD_DONE, v)

    def on_load_done(self, t: float, v: 'Vehicle'):
        if v.job is None:
            return
        v.job.pickup_t = t
        v.job.state = 'CARRYING'
        v.job_state = 'TO_DROP'
        self._reroute(t, v, v.job.dst_node_id)

    def on_unload_done(self, t: float, v: 'Vehicle'):
        from graph_des_v6 import STOP
        if v.job is None:
            return
        job = v.job
        job.drop_t = t
        job.state = 'DONE'
        self.completed.append(job)
        self.assigned.pop(job.id, None)
        v.job = None
        v.job_state = 'IDLE'
        # Vehicle physical state: stopped at the drop port.
        # Setting STOP + stop_reason='dest' makes the vehicle pushable
        # by other OHTs (was state=IDLE, which _is_idle_pushable rejects)
        # and triggers nearby followers to retry their plan.
        v.state = STOP
        v.stop_reason = 'dest'
        # Reroute counter reset: task 완료 후 새 task 의 reroute 시도 가능.
        if hasattr(v, 'reroute_count'):
            v.reroute_count = 0
        # V is now pushable. Wake its registered followers explicitly.
        self.des._notify_followers(t, v)
        self._try_dispatch(t)

    # ── Internals ────────────────────────────────────────────────────

    def _schedule_next_creation(self, t: float):
        if self.lambda_rate <= 0:
            return
        from graph_des_v6 import EV_JOB_CREATE
        u = max(1e-12, self._rng.random())
        dt = -math.log(u) / self.lambda_rate
        self.des._post_system(t + dt, EV_JOB_CREATE)

    def _create_job(self, t: float):
        if len(self.ports) < 2:
            return
        src = self._rng.choice(self.ports)
        # pick dst at a different node
        for _ in range(10):
            dst = self._rng.choice(self.ports)
            if dst.node_id != src.node_id:
                break
        else:
            return
        job = Job(
            id=f'j{self._next_job_id}',
            src_port_id=src.id,
            dst_port_id=dst.id,
            src_node_id=src.node_id,
            dst_node_id=dst.node_id,
            spawn_t=t,
        )
        self._next_job_id += 1
        self.pending.append(job)
        self._try_dispatch(t)

    def _current_node(self, v: 'Vehicle') -> str:
        """The node v is heading toward (= seg_to). Used as Dijkstra source."""
        return v.seg_to

    def _try_dispatch(self, t: float):
        """For each PENDING job (FIFO), find nearest IDLE OHT and assign."""
        if not self.pending:
            return
        # iterate snapshot of pending
        for job in list(self.pending):
            idle_vehicles = [v for v in self.des.vehicles.values() if v.job is None]
            if not idle_vehicles:
                return
            # one reverse-Dijkstra to src_node, lookup each idle vehicle's seg_to
            dist_to_src = dijkstra_dist_to(self.gmap, job.src_node_id)
            best: Optional['Vehicle'] = None
            best_d = math.inf
            for v in idle_vehicles:
                cn = self._current_node(v)
                d = dist_to_src.get(cn, math.inf)
                if d < best_d:
                    best_d = d
                    best = v
            if best is None or best_d == math.inf:
                continue   # no reachable idle, leave pending
            self.pending.remove(job)
            self._assign(t, best, job)

    def _assign(self, t: float, v: 'Vehicle', job: Job):
        job.state = 'ASSIGNED'
        job.assigned_to = v.id
        job.assign_t = t
        self.assigned[job.id] = job
        v.job = job
        v.job_state = 'TO_PICKUP'
        self._reroute(t, v, job.src_node_id)

    def _reroute(self, t: float, v: 'Vehicle', dst_node: str):
        """Build Dijkstra(commit_end_node → dst_node) and prepend the
        committed prefix.

        Prefix preservation is now unconditional (was previously gated on
        `has_lock`). Reasoning: with Phase 2.1 + force=False extension in
        _assign_destination, the committed plan must remain valid after
        the path swap — every segment v is committed to traverse must
        appear in v.path. Preserving prefix=v.path[path_idx..commit_end_idx]
        guarantees this. Held-lock exit nodes are also covered (this was
        the original orphan-lock guard rationale for the V#199/5148 case).

        Earlier collision concern from prefix-always-preserved
        (~3 follower-gap regressions at seed=99): predates Phase 2.1; was
        an artifact of force=True wipe + plan_gen bump. With force=False
        the committed_traj is preserved and follower reads stay consistent.
        """
        if v.commit_end_idx > v.path_idx + 1:
            commit_end_node = v.path[v.commit_end_idx]
            prefix = list(v.path[v.path_idx : v.commit_end_idx + 1])
            # Push-honor: if v is mid-push (via_push=True) and the push
            # target lies past commit_end, route THROUGH the push target so
            # the original push intent (clear pusher's path up to that
            # node) is fulfilled before heading to the new job dest. Just
            # rerouting from commit_end to dst_node would abandon the
            # push midway and re-block the pusher (V#94 seed=99 t=8874
            # case: push target '1286' was 4 segments past commit_end='1463'
            # but a job dispatch redirected V#94 from '1463' straight to
            # '349'. The new direction at '1463' required a different ZCU
            # lock that V#94 didn't hold — NO_LOCK fault — and V#52
            # (pusher) was left blocked).
            push_mid: List[str] = []
            push_anchor = commit_end_node
            if v.via_push and v.dest_node and v.dest_node in v.path:
                pt_idx = v.path.index(v.dest_node)
                if pt_idx > v.commit_end_idx:
                    push_mid = list(v.path[v.commit_end_idx + 1 : pt_idx + 1])
                    push_anchor = v.dest_node
            tail = dijkstra_path(self.gmap, push_anchor, dst_node)
            if tail is None:
                print(f"[DISP t={t:.1f}] no path V#{v.id} from "
                      f"{push_anchor} to {dst_node}")
                return
            new_path = prefix + push_mid + tail[1:]   # tail[0] == push_anchor
        else:
            tail = dijkstra_path(self.gmap, v.seg_to, dst_node)
            if tail is None:
                print(f"[DISP t={t:.1f}] no path V#{v.id} from {v.seg_to} to {dst_node}")
                return
            new_path = [v.seg_from] + tail   # tail[0] == seg_to
        self.des._assign_destination(t, v, new_path, dst_node)

    # ── Reporting ────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        return {
            'pending': len(self.pending),
            'assigned': len(self.assigned),
            'completed': len(self.completed),
            'total_created': self._next_job_id,
        }
