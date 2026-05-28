"""
graph_des_v5.py — Graph-based DES engine for OHT networks.

Based on graph_des_v3.py with the following key changes:
- X marker is ALWAYS placed, even when no leader OHT exists ahead.
- X marker maximum range = distance to the next ZCU node on the path.
- ZCU nodes = merge/diverge nodes (nodes with in-degree >= 2 or out-degree >= 2).
- If a leader exists within that range, x marker = min(leader-based, ZCU distance).
- _free_dist_with_zcu() renamed to _dist_to_next_zcu_node() for clarity.
- _plan() always sets stop_dist (never None) — always bounded by ZCU node distance.
"""

from __future__ import annotations
import math, heapq, random, json, collections
from typing import Dict, List, Optional, Tuple


# ── Map structures ────────────────────────────────────────────────────────────

class ZCUZone:
    """Zone Control Unit at a merge or diverge point."""

    def __init__(self, node_id: str, kind: str,
                 curve_segs: set, straight_segs: set):
        self.node_id = node_id
        self.kind = kind
        self.curve_segs = curve_segs
        self.straight_segs = straight_segs

    def is_curve(self, seg_key: Tuple[str, str]) -> bool:
        return seg_key in self.curve_segs

    def all_segs(self) -> set:
        return self.curve_segs | self.straight_segs


class MapNode:
    __slots__ = ('id', 'x', 'y')
    def __init__(self, nid: str, x: float, y: float):
        self.id = nid
        self.x = x
        self.y = y


class MapSegment:
    __slots__ = ('id', 'from_id', 'to_id', 'length', 'max_speed', 'path_points', '_curve_group')
    def __init__(self, sid, from_id: str, to_id: str, length: float,
                 max_speed: float = 2000.0,
                 path_points: List[Tuple[float, float]] = None):
        self.id = sid
        self.from_id = from_id
        self.to_id = to_id
        self.length = length
        self.max_speed = max_speed
        self.path_points = path_points or []
        self._curve_group = None


def _build_path_points(parts: list, start_xy: Tuple[float, float],
                       n_arc: int = 16) -> List[Tuple[float, float]]:
    """Convert segment parts (Straight / Arc) into a dense polyline."""
    pts: List[Tuple[float, float]] = [start_xy]
    cur = start_xy
    for part in parts:
        raw = [(p['x'], p['y']) for p in part['points']]
        if part['kind'] == 'Straight':
            pts.append(raw[0])
            cur = raw[0]
        elif part['kind'] == 'Arc':
            p0, p1, p2 = cur, raw[0], raw[1]
            for i in range(1, n_arc + 1):
                t = i / n_arc
                x = (1-t)**2 * p0[0] + 2*t*(1-t) * p1[0] + t**2 * p2[0]
                y = (1-t)**2 * p0[1] + 2*t*(1-t) * p1[1] + t**2 * p2[1]
                pts.append((x, y))
            cur = p2
    return pts


def _polyline_length(pts: List[Tuple[float, float]]) -> float:
    return sum(math.hypot(pts[i+1][0] - pts[i][0],
                          pts[i+1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def _interp_path(pts: List[Tuple[float, float]],
                 pos_mm: float) -> Tuple[float, float, float]:
    """Given a polyline and a distance along it, return (x, y, theta)."""
    remaining = pos_mm
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if remaining <= seg_len or i == len(pts) - 2:
            t = (remaining / seg_len) if seg_len > 0 else 1.0
            t = min(t, 1.0)
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            return x, y, theta
        remaining -= seg_len
    p = pts[-1]
    theta = math.atan2(pts[-1][1] - pts[-2][1],
                       pts[-1][0] - pts[-2][0]) if len(pts) >= 2 else 0.0
    return p[0], p[1], theta


# ── Graph Map ─────────────────────────────────────────────────────────────────

class GraphMap:
    """Loads an xmsmap-v4.0 JSON and provides graph traversal.

    area_filter (옵션): JSON 의 nodes[].area 값으로 sub-network 만 로드.
        예: area_filter='OHT_A' → KaistTB_map.json 의 OHT_A 영역만 사용.
    """

    def __init__(self, json_path: str, area_filter: Optional[str] = None):
        with open(json_path, 'r', encoding='utf-8') as f:
            d = json.load(f)

        if area_filter is not None:
            keep = {n['id'] for n in d['nodes'] if n.get('area') == area_filter}
            self.nodes: Dict[str, MapNode] = {
                n['id']: MapNode(n['id'], n['x'], n['y'])
                for n in d['nodes'] if n['id'] in keep
            }
        else:
            self.nodes: Dict[str, MapNode] = {
                n['id']: MapNode(n['id'], n['x'], n['y'])
                for n in d['nodes']
            }

        self.segments: Dict[Tuple[str, str], MapSegment] = {}
        self.adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for s in d['segments']:
            fn, tn = s['startNodeId'], s['endNodeId']
            if fn not in self.nodes or tn not in self.nodes:
                continue
            n1, n2 = self.nodes[fn], self.nodes[tn]
            start_xy = (n1.x, n1.y)
            parts = s.get('parts')
            if parts:
                path_pts = _build_path_points(parts, start_xy)
                length = _polyline_length(path_pts)
            else:
                path_pts = [start_xy, (n2.x, n2.y)]
                length = math.hypot(n2.x - n1.x, n2.y - n1.y)
            if length < 0.1:
                continue
            seg = MapSegment(s['id'], fn, tn, length,
                             s.get('speed', 2000), path_pts)
            self.segments[(fn, tn)] = seg
            self.adj[fn].append(tn)

        # Reverse adjacency
        self.adj_rev: Dict[str, List[str]] = collections.defaultdict(list)
        for (fn, tn) in self.segments:
            self.adj_rev[tn].append(fn)

        # Split U-turn curves at midpoint
        self._split_uturns()

        # Rebuild adj_rev after split
        self.adj_rev = collections.defaultdict(list)
        for (fn, tn) in self.segments:
            self.adj_rev[tn].append(fn)

        # Compute main loop (largest SCC)
        self.main_loop = self._compute_main_scc()

        # Spur nodes: dead-end pendants whose only outgoing edge points back
        # to a predecessor (e.g. 10001↔10002). They sit inside the main SCC
        # because of the 2-cycle, so main_loop membership doesn't filter them.
        # Path-generation code (random_safe_path, push planners, OHT spawn)
        # should exclude these to avoid wasted U-turn detours.
        self.spur_nodes = {
            n for n in self.nodes
            if len(self.adj.get(n, [])) == 1
               and self.adj[n][0] in self.adj_rev.get(n, [])
        }

        # Vehicle params from JSON
        oht_vm = next((v for v in d.get('vehicleModels', []) if v['id'] == 'OHT'), {})
        dim = oht_vm.get('dimension', {})
        self.vehicle_length = dim.get('length', 750)
        self.vehicle_width = dim.get('width', 500)

        # Bounding box
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        pad = 2000
        self.bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

        # Port nodes
        self.port_nodes = {
            p['nodeId'] for p in d.get('ports', [])
            if p.get('nodeId') in self.nodes
        }

        # ── ZCU: classify merge/diverge segments ─────────────────────────
        self._build_zcu()

    def _is_uturn(self, fn: str, tn: str) -> bool:
        in_dirs = []
        for pred in self.adj_rev.get(fn, []):
            seg = self.segments.get((pred, fn))
            if seg and len(seg.path_points) <= 2:
                pn = self.nodes[pred]
                n = self.nodes[fn]
                dx, dy = n.x - pn.x, n.y - pn.y
                l = math.hypot(dx, dy)
                if l > 1:
                    in_dirs.append((dx / l, dy / l))
        out_dirs = []
        for succ in self.adj.get(tn, []):
            seg = self.segments.get((tn, succ))
            if seg and len(seg.path_points) <= 2:
                n = self.nodes[tn]
                sn = self.nodes[succ]
                dx, dy = sn.x - n.x, sn.y - n.y
                l = math.hypot(dx, dy)
                if l > 1:
                    out_dirs.append((dx / l, dy / l))
        if not in_dirs or not out_dirs:
            return False
        dot = in_dirs[0][0] * out_dirs[0][0] + in_dirs[0][1] * out_dirs[0][1]
        angle = math.degrees(math.acos(max(-1, min(1, dot))))
        return angle > 120

    def _split_uturns(self):
        to_split = []
        for (fn, tn), seg in list(self.segments.items()):
            if seg.path_points and len(seg.path_points) > 2:
                if self._is_uturn(fn, tn):
                    to_split.append((fn, tn, seg))

        self.curve_groups = collections.defaultdict(list)
        mid_id = 0
        for fn, tn, seg in to_split:
            pts = seg.path_points
            total_len = _polyline_length(pts)
            if total_len < 100:
                continue

            half = total_len / 2.0
            accum = 0.0
            mid_x, mid_y = pts[-1]
            split_idx = len(pts) - 1
            for i in range(len(pts) - 1):
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
                sl = math.hypot(dx, dy)
                if accum + sl >= half:
                    t = (half - accum) / sl if sl > 0 else 0.5
                    mid_x = pts[i][0] + dx * t
                    mid_y = pts[i][1] + dy * t
                    split_idx = i + 1
                    break
                accum += sl

            mid_nid = f"_curve_mid_{mid_id}"
            mid_id += 1
            self.nodes[mid_nid] = MapNode(mid_nid, mid_x, mid_y)
            self.adj[mid_nid] = []
            self.adj_rev[mid_nid] = []

            pts_first = list(pts[:split_idx]) + [(mid_x, mid_y)]
            pts_second = [(mid_x, mid_y)] + list(pts[split_idx:])
            len_first = _polyline_length(pts_first)
            len_second = _polyline_length(pts_second)

            del self.segments[(fn, tn)]
            self.adj[fn].remove(tn)
            self.adj_rev[tn].remove(fn)

            seg1 = MapSegment(f"{seg.id}_a", fn, mid_nid, len_first,
                              seg.max_speed, pts_first)
            seg2 = MapSegment(f"{seg.id}_b", mid_nid, tn, len_second,
                              seg.max_speed, pts_second)
            self.segments[(fn, mid_nid)] = seg1
            self.segments[(mid_nid, tn)] = seg2
            self.adj[fn].append(mid_nid)
            self.adj[mid_nid].append(tn)
            self.adj_rev[mid_nid].append(fn)
            self.adj_rev[tn].append(mid_nid)

            gid = f"_cg_{mid_id - 1}"
            seg1._curve_group = gid
            seg2._curve_group = gid
            self.curve_groups[gid].append((fn, mid_nid))
            self.curve_groups[gid].append((mid_nid, tn))

        if to_split:
            print(f"U-turn splits: {len(to_split)} -> "
                  f"{len(self.nodes)} nodes, {len(self.segments)} segments")

    def _is_curve_seg(self, fn: str, tn: str) -> bool:
        seg = self.segments.get((fn, tn))
        return seg is not None and seg.path_points and len(seg.path_points) > 2

    def _build_zcu(self):
        """Build ZCU zones at merge/diverge points.
        v5: also builds self.zcu_nodes — the unified set of all ZCU boundary nodes."""
        self.zcu_zones: List[ZCUZone] = []
        self.seg_to_zone: Dict[Tuple[str,str], ZCUZone] = {}
        self.merge_nodes = set()
        self.diverge_nodes = set()
        self.zcu_nodes: set = set()  # v5: all ZCU nodes (merge + diverge)
        self.merge_curve_entries = set()
        self.merge_straight_entries = set()
        self.diverge_curve_exits = set()
        self.diverge_straight_exits = set()

        # Merge zones
        for nid in self.nodes:
            preds = self.adj_rev.get(nid, [])
            if len(preds) < 2:
                continue
            entries = [(p, nid) for p in preds if (p, nid) in self.segments]
            if len(entries) < 2:
                continue
            curves = [k for k in entries if self._is_curve_seg(*k)]
            straights = [k for k in entries if not self._is_curve_seg(*k)]
            if not curves:
                continue

            zone = ZCUZone(nid, 'merge', set(curves), set(straights))
            self.zcu_zones.append(zone)
            self.merge_nodes.add(nid)
            self.zcu_nodes.add(nid)
            for k in curves:
                self.seg_to_zone[k] = zone
                self.merge_curve_entries.add(k)
            for k in straights:
                self.seg_to_zone[k] = zone
                self.merge_straight_entries.add(k)

        # Diverge zones
        for nid in self.nodes:
            succs = self.adj.get(nid, [])
            if len(succs) < 2:
                continue
            exits = [(nid, s) for s in succs if (nid, s) in self.segments]
            if len(exits) < 2:
                continue
            curves = [k for k in exits if self._is_curve_seg(*k)]
            straights = [k for k in exits if not self._is_curve_seg(*k)]

            zone = ZCUZone(nid, 'diverge', set(curves), set(straights))
            self.zcu_zones.append(zone)
            self.diverge_nodes.add(nid)
            self.zcu_nodes.add(nid)
            for k in curves:
                self.seg_to_zone[k] = zone
                self.diverge_curve_exits.add(k)
            for k in straights:
                self.seg_to_zone[k] = zone
                self.diverge_straight_exits.add(k)

        print(f"ZCU: {len(self.merge_nodes)} merge, "
              f"{len(self.diverge_nodes)} diverge, "
              f"{len(self.zcu_zones)} zones total, "
              f"{len(self.zcu_nodes)} ZCU boundary nodes")

    def _compute_main_scc(self) -> set:
        order = []
        vis = set()
        for start in self.nodes:
            if start in vis:
                continue
            stack = [(start, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    order.append(node)
                    continue
                if node in vis:
                    continue
                vis.add(node)
                stack.append((node, True))
                for nb in self.adj.get(node, []):
                    if nb not in vis:
                        stack.append((nb, False))

        radj = collections.defaultdict(list)
        for (fn, tn) in self.segments:
            radj[tn].append(fn)

        vis2 = set()
        largest = []
        for node in reversed(order):
            if node in vis2:
                continue
            comp = []
            stack = [node]
            while stack:
                n = stack.pop()
                if n in vis2:
                    continue
                vis2.add(n)
                comp.append(n)
                for nb in radj.get(n, []):
                    if nb not in vis2:
                        stack.append(nb)
            if len(comp) > len(largest):
                largest = comp
        return sorted(largest)

    def segment_between(self, from_id: str, to_id: str) -> Optional[MapSegment]:
        return self.segments.get((from_id, to_id))


def random_safe_path(gmap: GraphMap, start_node: str, length: int = 200,
                     rng=None) -> List[str]:
    _rand = rng if rng is not None else random
    main = set(gmap.main_loop)
    spurs = getattr(gmap, 'spur_nodes', set())
    path = [start_node]
    cur = start_node
    for _ in range(length):
        neighbors = gmap.adj.get(cur, [])
        # Exclude spurs (e.g. 10001) — they're in main_loop SCC due to back-
        # pointer 2-cycles but represent dead-end U-turn detours.
        safe = [n for n in neighbors if n in main and n not in spurs]
        if not safe:
            break
        visited = set(path[-20:])
        unvisited = [n for n in safe if n not in visited]
        nxt = _rand.choice(unvisited) if unvisited else _rand.choice(safe)
        path.append(nxt)
        cur = nxt
    return path


# ── Vehicle states ────────────────────────────────────────────────────────────

IDLE    = 'IDLE'
ACCEL   = 'ACCEL'
CRUISE  = 'CRUISE'
DECEL   = 'DECEL'
STOP    = 'STOP'
LOADING = 'LOADING'


# ── Vehicle ───────────────────────────────────────────────────────────────────

class Vehicle:
    def __init__(self, vid: int, gmap: GraphMap, path: List[str], color=(200, 200, 200)):
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
        # h_min=1150 between vehicle heads, but the safety/visualization
        # layer (and reality) treats vehicles as 1108mm long, so heads
        # 1149mm apart actually overlap — false collision class.
        # h_min adds 200mm safety margin past the physical length.
        _gmap_vlen = getattr(gmap, 'vehicle_length', None)
        self.length: float = float(_gmap_vlen) if _gmap_vlen else 750.0
        self.h_min: float = self.length + 200

        self.state: str = IDLE
        self.leader: Optional[Vehicle] = None
        self.token: int = 0
        self.next_event_t: float = 0.0

        self.stop_dist: Optional[float] = None
        self.x_marker_abs: float = 0.0

        # v5: X marker absolute position on path (pinned to ZCU node)
        self.x_marker_pidx: int = 0       # path_idx of the segment containing the marker
        self.x_marker_offset: float = 0.0 # offset within that segment
        self.x_marker_node: Optional[str] = None  # ZCU node ID the marker is pinned to

        # Destination stop: node ID where vehicle must stop (None = free running)
        self.dest_node: Optional[str] = None
        self.dest_reached: bool = False

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
        if self.path_idx < len(self.path):
            return self.path[self.path_idx]
        return self.path[-1]

    @property
    def seg_to(self) -> str:
        if self.path_idx + 1 < len(self.path):
            return self.path[self.path_idx + 1]
        return self.path[-1]

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

    # ── Kinematics ────────────────────────────────────────────────────────

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
        dt = t - self.t_ref
        if dt < 0:
            dt = 0
        if self.acc > 0:
            return min(self.v_max, self.vel + self.acc * dt)
        elif self.acc < 0:
            return max(0.0, self.vel + self.acc * dt)
        return self.vel

    def dist_at(self, t: float) -> float:
        return self._dist_traveled(t - self.t_ref)

    def advance_position(self, t: float):
        dist = self.dist_at(t)
        self.vel = self.vel_at(t)
        self.t_ref = t
        self.seg_offset += dist
        while self.path_idx < len(self.path) - 1:
            seg_len = self.current_seg_length()
            if seg_len <= 0:
                self.path_idx += 1
                self.seg_offset = 0.0
                continue
            if self.seg_offset >= seg_len - 0.01:
                self.seg_offset -= seg_len
                self.path_idx += 1
                if self.seg_offset < 0:
                    self.seg_offset = 0.0
            else:
                break

    def set_state(self, t: float):
        self.advance_position(t)
        self.acc = 0.0

    def abs_path_dist(self) -> float:
        d = 0.0
        for i in range(min(self.path_idx, len(self._seg_lengths))):
            d += self._seg_lengths[i]
        d += self.seg_offset
        return d

    def braking_distance(self, from_vel: float = -1) -> float:
        v = from_vel if from_vel >= 0 else self.vel
        if v <= 0:
            return 0.0
        return v * v / (2 * self.d_max)

    def dist_ahead_on_path(self, n_segs: int = 50) -> float:
        if self.path_idx >= len(self.path) - 1:
            return 0.0
        total = self.current_seg_length() - self.seg_offset
        for i in range(self.path_idx + 1, min(self.path_idx + n_segs, len(self.path) - 1)):
            if i < len(self._seg_lengths):
                total += self._seg_lengths[i]
            else:
                seg = self.gmap.segment_between(self.path[i], self.path[i + 1])
                if seg:
                    total += seg.length
        return total

    def update_render(self, t: float):
        dist = self._dist_traveled(t - self.t_ref)
        offset = self.seg_offset + dist
        pidx = self.path_idx

        while pidx < len(self.path) - 1:
            if pidx < len(self._seg_lengths):
                seg_len = self._seg_lengths[pidx]
            else:
                seg = self.gmap.segment_between(self.path[pidx], self.path[pidx + 1])
                seg_len = seg.length if seg else 0.0
            if seg_len <= 0:
                pidx += 1
                offset = 0.0
                continue
            if offset >= seg_len - 0.01 and pidx < len(self.path) - 2:
                offset -= seg_len
                pidx += 1
                if offset < 0:
                    offset = 0.0
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
        return self.path_idx >= len(self.path) - 10

    def extend_path(self, new_nodes: List[str]):
        if self.path_idx > 5:
            trim = self.path_idx - 2
            self.path = self.path[trim:]
            self.path_idx -= trim
            self._seg_lengths = self._seg_lengths[trim:]
            self._seg_speeds = self._seg_speeds[trim:]

        if new_nodes and new_nodes[0] == self.path[-1]:
            new_nodes = new_nodes[1:]
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


# ── Events ────────────────────────────────────────────────────────────────────

class Event:
    __slots__ = ('t', 'kind', 'vid', 'token')
    def __init__(self, t: float, kind: str, vid: int, token: int):
        self.t = t
        self.kind = kind
        self.vid = vid
        self.token = token

    def __lt__(self, other):
        return self.t < other.t


# ── DES Engine ────────────────────────────────────────────────────────────────

class GraphDESv5:
    """v5: X marker always placed at next ZCU node, regardless of leader existence."""

    def __init__(self, gmap: GraphMap, stop_prob: float = 0.02,
                 stop_dur: Tuple[float, float] = (2.0, 5.0)):
        self.gmap = gmap
        self.vehicles: Dict[int, Vehicle] = {}
        self.heap: List[Event] = []
        self.event_count: int = 0
        self.stop_prob = stop_prob
        self.stop_dur = stop_dur
        self.stops_executed: int = 0

        self._seg_occupants: Dict[Tuple[str, str], List[Vehicle]] = collections.defaultdict(list)

    def add_vehicle(self, v: Vehicle):
        self.vehicles[v.id] = v
        key = (v.seg_from, v.seg_to)
        self._seg_occupants[key].append(v)

    def _post(self, t: float, kind: str, v: Vehicle):
        v.next_event_t = t
        heapq.heappush(self.heap, Event(t, kind, v.id, v.token))

    def _invalidate(self, v: Vehicle):
        v.token += 1

    # ── Gap computation ───────────────────────────────────────────────────

    def gap(self, follower: Vehicle, t: float) -> Tuple[float, float]:
        leader = follower.leader
        if leader is None:
            return float('inf'), 0.0

        f_pidx = follower.path_idx
        f_off = follower.seg_offset + follower._dist_traveled(t - follower.t_ref)
        l_pidx = leader.path_idx
        l_off = leader.seg_offset + leader._dist_traveled(t - leader.t_ref)

        f_from, f_to = follower.seg_from, follower.seg_to
        l_from, l_to = leader.seg_from, leader.seg_to

        if f_from == l_from and f_to == l_to:
            gap_d = l_off - f_off
            if gap_d < 0:
                gap_d = 0
            return gap_d, leader.vel_at(t)

        dist = 0.0
        if f_pidx < len(follower._seg_lengths):
            dist += follower._seg_lengths[f_pidx] - f_off
        else:
            seg = follower.current_segment()
            dist += (seg.length if seg else 0) - f_off

        max_look = min(f_pidx + 80, len(follower.path) - 1)
        for i in range(f_pidx + 1, max_look):
            fn = follower.path[i]
            tn = follower.path[i + 1] if i + 1 < len(follower.path) else None
            if tn is None:
                break
            if fn == l_from and tn == l_to:
                dist += l_off
                return dist, leader.vel_at(t)
            if i < len(follower._seg_lengths):
                dist += follower._seg_lengths[i]
            else:
                seg = self.gmap.segment_between(fn, tn)
                dist += seg.length if seg else 0

            if dist > 200000:
                break

        fx, fy = follower.x, follower.y
        lx, ly = leader.x, leader.y
        euc = math.hypot(lx - fx, ly - fy)
        return max(euc, 1.0), leader.vel_at(t)

    # ── Core: _plan() ────────────────────────────────────────────────────

    def _leader_extra_dist(self, leader: Vehicle, t: float) -> float:
        t_confirmed = getattr(leader, 'next_event_t', t)
        if t_confirmed <= t:
            return leader.braking_distance(leader.vel_at(t))
        dist_now = leader._dist_traveled(t - leader.t_ref)
        dist_event = leader._dist_traveled(t_confirmed - leader.t_ref)
        confirmed_travel = dist_event - dist_now
        confirmed_vel = leader.vel_at(t_confirmed)
        return confirmed_travel + leader.braking_distance(confirmed_vel)

    _in_notify = False

    def _maybe_notify(self, t: float, v: Vehicle, old_acc: float, old_vel: float):
        if self._in_notify:
            return
        if (old_acc <= 0 and v.acc > 0) or (old_acc >= 0 and v.acc < 0) or \
           abs(v.vel - old_vel) > 100:
            self._in_notify = True
            self._notify_followers(t)
            self._in_notify = False

    def _plan(self, t: float, v: Vehicle):
        """v5: X marker is ALWAYS placed at the next ZCU node boundary.

        Key difference from v3:
        - v3: leader 없으면 zcu_limit > 100000일 때 stop_dist = None (x marker 없음)
        - v5: leader 유무 관계없이 항상 stop_dist = 다음 ZCU node까지의 거리
              ZCU node = merge/diverge node
        """
        old_acc = v.acc
        old_vel = v.vel
        old_key = (v.seg_from, v.seg_to)
        v.set_state(t)
        new_key = (v.seg_from, v.seg_to)

        # Update segment occupancy if changed
        if old_key != new_key:
            if v in self._seg_occupants[old_key]:
                self._seg_occupants[old_key].remove(v)
            if v not in self._seg_occupants[new_key]:
                self._seg_occupants[new_key].append(v)
            if old_key[0] is not None:
                old_zone = self.gmap.seg_to_zone.get(old_key)
                if old_zone:
                    still_occupied = any(
                        self._seg_occupants.get(sk)
                        for sk in old_zone.all_segs()
                        if sk != new_key
                    )
                    if not still_occupied:
                        self._notify_followers(t)

        # Extend path if needed
        if v.needs_path_extension():
            last_node = v.path[-1]
            ext = random_safe_path(self.gmap, last_node, length=100)
            v.extend_path(ext)

        # End of path
        if v.path_idx >= len(v.path) - 1:
            v.vel = 0.0
            v.acc = 0.0
            v.state = STOP
            self._post(t + 1.0, 'RESUME', v)
            return

        # Destination: check if we've reached or passed the dest node
        if v.dest_node is not None and not v.dest_reached:
            # Check if dest node is the start of current segment (we just passed it)
            if v.seg_from == v.dest_node:
                v.dest_reached = True
            # Also check previous node in path
            elif v.path_idx > 0 and v.path[v.path_idx] == v.dest_node:
                v.dest_reached = True

        if v.dest_node is not None and v.dest_reached:
            v.vel = 0.0
            v.acc = 0.0
            v.state = STOP
            self._post(t + 0.5, 'RESUME', v)
            self._maybe_notify(t, v, old_acc, old_vel)
            return

        seg_speed = v.current_seg_speed()
        target_v = min(v.v_max, seg_speed)

        # ── Look-ahead braking for upcoming slow segments (curves) ───────
        lookahead_v, dist_to_slow = self._lookahead_speed(v)
        target_v = min(target_v, lookahead_v)

        leader = v.leader

        # ── v5: Always compute ZCU-bounded free distance ─────────────────
        zcu_limit, marker_dist = self._dist_to_next_zcu_node(v)

        if leader is not None:
            gap_d, leader_vel = self.gap(v, t)
            leader_extra = self._leader_extra_dist(leader, t)
            free_from_confirmed = gap_d + leader_extra - v.h_min
            free_from_gap = gap_d - v.h_min
            free_dist = min(free_from_confirmed, free_from_gap, zcu_limit)
            leader_vel_now = leader_vel
        else:
            free_dist = zcu_limit
            leader_vel_now = 0.0

        # ── Destination stop: limit free_dist to dest node ───────────────
        dest_dist = self._dist_to_dest(v)
        if dest_dist < free_dist:
            free_dist = dest_dist

        v.stop_dist = free_dist if free_dist < 100000 else None

        # Unlimited free distance AND no curve ahead: cruise, recheck periodically
        if free_dist > 100000 and dist_to_slow > 100000:
            self._go(t, v, target_v)
            # Schedule RESUME: also consider braking distance to any upcoming
            # speed change so we start decelerating in time.
            brake_look = v.v_max * v.v_max / (2 * v.d_max)  # max braking dist
            replan_dist = marker_dist
            # Check if a slow segment is within 2x braking distance
            # (so next replan will start the approach)
            if dist_to_slow < brake_look * 2:
                replan_dist = min(replan_dist, dist_to_slow)
            if replan_dist < 100000:
                effective_v = max(v.vel, target_v, 1.0)
                t_to = replan_dist / effective_v
                self._post(t + max(0.05, min(1.0, t_to * 0.5)), 'RESUME', v)
            else:
                self._post(t + 1.0, 'RESUME', v)
            self._maybe_notify(t, v, old_acc, old_vel)
            return

        if free_dist <= 0:
            v.acc = 0.0
            v.vel = 0.0
            v.state = STOP
            v.stop_dist = 0.0
            self._post(t + 0.3, 'RESUME', v)
            self._maybe_notify(t, v, old_acc, old_vel)
            return

        # ── v_safe (leader/ZCU: brake to 0) ──────────────────────────────
        if free_dist < 100000:
            v_safe = math.sqrt(max(0, 2 * v.d_max * free_dist))
        else:
            v_safe = v.v_max

        # ── Combine: look-ahead (brake to curve speed) + gap (brake to 0) ─
        v_target = min(target_v, v_safe)

        if leader_vel_now > 0.1 and leader_vel_now <= v_safe:
            v_target = min(target_v, max(v_target, leader_vel_now))

        # ── Set motion ───────────────────────────────────────────────────
        # Determine the effective braking constraint:
        # - For leader/ZCU: brake to 0 at free_dist
        # - For curves: brake to curve_speed at dist_to_slow
        # Use whichever is more constraining.
        if free_dist < 100000:
            # Leader/ZCU constraint: standard brake-to-zero logic
            brake_dist = v.braking_distance()

            if free_dist > brake_dist:
                self._go(t, v, v_target)
                future_brake = v_target * v_target / (2 * v.d_max)
                coast_dist = max(0, free_dist - future_brake)
                effective_v = v_target if v_target > 0 else 1
                if v.acc > 0:
                    t_accel = max(0, (v_target - v.vel) / v.a_max)
                    d_accel = v.vel * t_accel + 0.5 * v.a_max * t_accel ** 2
                    t_next = t_accel + max(0, coast_dist - d_accel) / effective_v
                else:
                    t_next = coast_dist / effective_v
                self._post(t + max(0.05, t_next), 'RESUME', v)
            else:
                if v.vel > 0.1 and free_dist > 0.1:
                    decel = min(v.vel * v.vel / (2 * free_dist), v.d_max)
                    v.acc = -decel
                    v.state = DECEL
                    t_stop = v.vel / decel
                    self._post(t + t_stop, 'STOPPED', v)
                else:
                    v.acc = 0.0
                    v.vel = 0.0
                    v.state = STOP
                    v.stop_dist = 0.0
                    self._post(t + 0.2, 'RESUME', v)
        else:
            # No leader/ZCU constraint, but approaching a slow segment (curve).
            # v_target already includes look-ahead braking via _lookahead_speed.
            self._go(t, v, v_target)
            # Short replan interval during curve approach for precise braking.
            # Replan every ~0.15s or when decel completes, whichever is sooner.
            if v.vel > v_target + 1:
                t_decel = (v.vel - v_target) / v.d_max
                self._post(t + max(0.05, min(0.15, t_decel)), 'RESUME', v)
            else:
                replan_dist = min(marker_dist, dist_to_slow)
                if replan_dist < 100000:
                    effective_v = max(v.vel, v_target, 1.0)
                    t_to = replan_dist / effective_v
                    self._post(t + max(0.05, min(0.2, t_to * 0.3)), 'RESUME', v)
                else:
                    self._post(t + 1.0, 'RESUME', v)

        self._maybe_notify(t, v, old_acc, old_vel)

    def _is_zcu_blocked(self, zone: ZCUZone, seg_key: Tuple[str, str],
                        v: Vehicle) -> bool:
        """Is this ZCU segment blocked for vehicle v?"""
        is_curve = zone.is_curve(seg_key)

        if seg_key[0].startswith('_curve_mid_'):
            is_curve = False

        if is_curve:
            occ = [o for o in self._seg_occupants.get(seg_key, []) if o is not v]
            if occ:
                return True

        seg_obj = self.gmap.segments.get(seg_key)
        if seg_obj:
            cg = getattr(seg_obj, '_curve_group', None)
            if cg:
                is_first_half = not seg_key[0].startswith('_curve_mid_')
                if is_first_half:
                    for sk in self.gmap.curve_groups.get(cg, []):
                        if sk == seg_key:
                            continue
                        occ = [o for o in self._seg_occupants.get(sk, []) if o is not v]
                        if occ:
                            return True

        for other_key in zone.all_segs():
            if other_key == seg_key:
                continue
            occupants = self._seg_occupants.get(other_key, [])
            occ = [o for o in occupants if o is not v]
            if not occ:
                continue
            other_is_curve = zone.is_curve(other_key)
            if other_key[0].startswith('_curve_mid_'):
                other_is_curve = False
            if is_curve:
                return True
            elif other_is_curve:
                return True
        return False

    def _find_next_zcu_node(self, v: Vehicle) -> Tuple[float, int, str]:
        """v5: Walk path forward to find the next ZCU node.

        Returns (dist, zcu_path_idx, zcu_node_id):
        - dist: distance from current position to the ZCU node
        - zcu_path_idx: the path index where the ZCU node sits
              (ZCU node = v.path[zcu_path_idx + 1], end of segment zcu_path_idx)
        - zcu_node_id: the ZCU node ID

        If no ZCU found, returns (inf, -1, None).
        """
        dist = v.current_seg_length() - v.seg_offset
        pi = v.path_idx

        while dist < 100000 and pi + 1 < len(v.path) - 1:
            next_node = v.path[pi + 1]

            if next_node in self.gmap.zcu_nodes:
                return dist, pi, next_node

            # Not ZCU: advance to next segment
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

    def _dist_to_next_zcu_node(self, v: Vehicle) -> Tuple[float, float]:
        """v5: Compute free distance with ZCU-hop logic.

        Returns (free_dist, marker_dist):
        - free_dist: motion constraint (inf if no blocked ZCU)
        - marker_dist: distance to the marker's ZCU node (for RESUME scheduling)

        Logic:
        1. Marker always starts at the nearest ZCU node ahead.
        2. When within braking distance of that ZCU:
           - blocked → free_dist = distance to this ZCU (stop here)
           - open → hop marker to the NEXT ZCU, free_dist extends too
        3. When still far from the nearest ZCU:
           - Marker stays at that ZCU, free_dist = inf (don't slow down)
        """
        brake_dist = v.braking_distance()

        dist = v.current_seg_length() - v.seg_offset
        pi = v.path_idx
        first_zcu_pi = -1
        first_zcu_node = None
        first_zcu_dist = 0.0

        while dist < 100000 and pi + 1 < len(v.path) - 1:
            next_node = v.path[pi + 1]

            if next_node in self.gmap.zcu_nodes:
                if first_zcu_node is None:
                    first_zcu_pi = pi
                    first_zcu_node = next_node
                    first_zcu_dist = dist

                if dist > brake_dist:
                    # Far: pin marker at nearest ZCU, no speed limit
                    self._pin_marker_at_node(v, first_zcu_pi, first_zcu_node)
                    return float('inf'), first_zcu_dist

                # Within braking distance: check if blocked
                blocked = self._is_zcu_node_blocked(v, pi, next_node)

                if blocked:
                    self._pin_marker_at_node(v, pi, next_node)
                    return dist, dist

                # Open & within braking distance: hop marker forward
                first_zcu_pi = pi
                first_zcu_node = next_node
                first_zcu_dist = dist

            # Advance to next segment
            tn = v.path[pi + 2] if pi + 2 < len(v.path) else None
            if tn is None:
                break
            seg = self.gmap.segments.get((next_node, tn))
            if seg:
                dist += seg.length
            else:
                break
            pi += 1

        # Pin marker at the latest ZCU we reached
        if first_zcu_node is not None:
            self._pin_marker_at_node(v, first_zcu_pi, first_zcu_node)
            return float('inf'), first_zcu_dist

        v.x_marker_pidx = -1
        v.x_marker_offset = 0.0
        v.x_marker_node = None
        return float('inf'), float('inf')

    def _is_zcu_node_blocked(self, v: Vehicle, pi: int, node_id: str) -> bool:
        """Check if a ZCU node is blocked for vehicle v.
        Checks both the incoming and outgoing segments at that node."""
        seg_key_in = (v.path[pi], node_id)
        zone = self.gmap.seg_to_zone.get(seg_key_in)
        if zone and self._is_zcu_blocked(zone, seg_key_in, v):
            return True
        if pi + 2 < len(v.path):
            seg_key_out = (node_id, v.path[pi + 2])
            zone = self.gmap.seg_to_zone.get(seg_key_out)
            if zone and self._is_zcu_blocked(zone, seg_key_out, v):
                return True
        return False

    def _pin_marker_at_node(self, v: Vehicle, pi: int, node_id: str):
        """Pin the X marker exactly at a ZCU node position.
        The node is at the END of segment pi (= start of segment pi+1).
        So marker is at path_idx=pi, offset=seg_length (end of that segment)."""
        if pi < len(v._seg_lengths):
            seg_len = v._seg_lengths[pi]
        else:
            seg = v.gmap.segment_between(v.path[pi], v.path[pi + 1])
            seg_len = seg.length if seg else 0.0
        v.x_marker_pidx = pi
        v.x_marker_offset = seg_len  # end of segment = at the node
        v.x_marker_node = node_id

    def _dist_to_dest(self, v: Vehicle) -> float:
        """Distance from current position to the vehicle's dest_node.
        Returns inf if no destination or already reached."""
        if v.dest_node is None or v.dest_reached:
            return float('inf')

        dist = v.current_seg_length() - v.seg_offset
        pi = v.path_idx

        while pi + 1 < len(v.path):
            next_node = v.path[pi + 1]
            if next_node == v.dest_node:
                # Mark reached when very close
                if dist < 1.0:
                    v.dest_reached = True
                return max(0.0, dist)

            pi += 1
            if pi + 1 < len(v.path):
                if pi < len(v._seg_lengths):
                    dist += v._seg_lengths[pi]
                else:
                    seg = v.gmap.segment_between(v.path[pi], v.path[pi + 1])
                    dist += seg.length if seg else 0
            if dist > 200000:
                break

        return float('inf')

    def _lookahead_speed(self, v: Vehicle) -> Tuple[float, float]:
        """Look ahead along the path for slower segments (e.g. curves).

        Returns (max_vel_now, dist_to_slow):
        - max_vel_now: the max speed the vehicle should be going RIGHT NOW
                       so it can brake in time for any slower segment ahead.
                       Formula: sqrt(v_slow^2 + 2 * d_max * dist_to_slow)
        - dist_to_slow: distance to the nearest speed-reducing segment
                        (for RESUME scheduling). inf if none found.

        Also checks the CURRENT segment speed as the baseline —
        the vehicle must never exceed its own segment's limit.
        """
        cur_speed = v.current_seg_speed()
        dist = v.current_seg_length() - v.seg_offset
        pi = v.path_idx
        best_v = cur_speed   # never exceed current segment's limit
        best_dist = float('inf')

        # Look ahead up to braking distance from v_max
        max_look = v.v_max * v.v_max / (2 * v.d_max) + 2000

        while dist < max_look and pi + 1 < len(v.path) - 1:
            pi += 1
            if pi < len(v._seg_speeds):
                seg_spd = v._seg_speeds[pi]
            else:
                seg = v.gmap.segment_between(v.path[pi], v.path[pi + 1])
                seg_spd = seg.max_speed if seg else v.v_max

            if seg_spd < best_v:
                # Slower segment at distance `dist`.
                # Subtract a safety margin so braking starts early enough
                # to account for discrete replanning intervals.
                # Safety margin: ~0.5s of travel at current speed.
                # Ensures braking starts early enough for discrete replanning.
                margin = max(500, v.vel * 0.5)
                safe_dist = max(0.0, dist - margin)
                v_safe_here = math.sqrt(seg_spd * seg_spd + 2 * v.d_max * safe_dist)
                if v_safe_here < best_v:
                    best_v = v_safe_here
                    if best_dist == float('inf'):
                        best_dist = dist

            # Add this segment's length
            if pi < len(v._seg_lengths):
                dist += v._seg_lengths[pi]
            else:
                seg = v.gmap.segment_between(v.path[pi], v.path[pi + 1])
                dist += seg.length if seg else 0

        return best_v, best_dist

    def _go(self, t: float, v: Vehicle, target_v: float):
        if v.vel < target_v - 1:
            v.acc = v.a_max
            v.state = ACCEL
        elif v.vel > target_v + 1:
            v.acc = -v.d_max
            v.state = DECEL
        else:
            v.vel = target_v
            v.acc = 0.0
            v.state = CRUISE

    # ── Event handlers ────────────────────────────────────────────────────

    def step(self, t_now: float):
        while self.heap and self.heap[0].t <= t_now:
            ev = heapq.heappop(self.heap)
            v = self.vehicles.get(ev.vid)
            if v is None or ev.token != v.token:
                continue
            self.event_count += 1

            if ev.kind == 'START':
                self._plan(ev.t, v)
            elif ev.kind == 'RESUME':
                self._plan(ev.t, v)
            elif ev.kind == 'STOPPED':
                v.set_state(ev.t)
                v.vel = 0.0
                v.acc = 0.0
                v.state = STOP
                self._post(ev.t + 0.2, 'RESUME', v)

        for v in self.vehicles.values():
            v.update_render(t_now)
            g, _ = self.gap(v, t_now)
            v.gap_to_leader = g

    def _notify_followers(self, t: float):
        for v in self.vehicles.values():
            if v.leader is None or v.state == LOADING:
                continue
            if v.state in (STOP, IDLE):
                continue
            self._invalidate(v)
            self._plan(t, v)

    # ── Leader assignment ─────────────────────────────────────────────────

    def assign_leaders(self):
        seg_vehs: Dict[Tuple[str, str], List[Tuple[float, Vehicle]]] = collections.defaultdict(list)
        for v in self.vehicles.values():
            key = (v.seg_from, v.seg_to)
            seg_vehs[key].append((v.seg_offset, v))

        for v in self.vehicles.values():
            best_leader = None
            best_dist = float('inf')

            key = (v.seg_from, v.seg_to)
            for off, other in seg_vehs[key]:
                if other.id == v.id:
                    continue
                if off > v.seg_offset:
                    d = off - v.seg_offset
                    if d < best_dist:
                        best_dist = d
                        best_leader = other

            if best_leader is None:
                dist_accum = v.current_seg_length() - v.seg_offset
                for i in range(v.path_idx + 1, min(v.path_idx + 30, len(v.path) - 1)):
                    fn = v.path[i]
                    tn = v.path[i + 1] if i + 1 < len(v.path) else None
                    if tn is None:
                        break
                    fwd_key = (fn, tn)
                    if fwd_key in seg_vehs:
                        candidates = seg_vehs[fwd_key]
                        if candidates:
                            candidates_sorted = sorted(candidates, key=lambda x: x[0])
                            off, other = candidates_sorted[0]
                            if other.id != v.id:
                                d = dist_accum + off
                                if d < best_dist:
                                    best_dist = d
                                    best_leader = other
                                break
                    seg = self.gmap.segment_between(fn, tn)
                    dist_accum += seg.length if seg else 0
                    if dist_accum > 100000:
                        break

            v.leader = best_leader

    def start_all(self):
        self.assign_leaders()
        for v in self.vehicles.values():
            self._post(0.0, 'START', v)

    def reassign_leaders_periodic(self, t: float):
        self.assign_leaders()
        for v in self.vehicles.values():
            if v.state in (ACCEL, CRUISE, DECEL):
                self._invalidate(v)
                self._plan(t, v)
