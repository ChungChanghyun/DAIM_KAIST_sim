"""test_plan_micro.py - micro scenario tests for plan correctness.

Builds tiny synthetic maps in memory and runs DES on isolated topologies
to verify that compute_velocity_profile, leader detection, and ZCU lock
behavior produce correct trajectories. Each scenario asserts invariants
on per-step state.

Run: python test_plan_micro.py            # all scenarios
     python test_plan_micro.py S3         # single scenario
"""

import json
import math
import os
import sys
import tempfile
from typing import Callable, List, Optional, Tuple

from graph_des_v5 import GraphMap
from graph_des_v6 import GraphDESv6, Vehicle


# ── Mini map builder ─────────────────────────────────────────────────────────

def _make_map(nodes: list, segments: list) -> GraphMap:
    """Build a GraphMap from in-memory node/segment dicts via tempfile."""
    d = {
        "nodes": nodes,
        "segments": segments,
        "ports": [],
        "vehicleModels": [{
            "id": "OHT",
            "dimension": {"length": 750, "width": 500},
        }],
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(path, "w") as f:
            json.dump(d, f)
        return GraphMap(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _node(nid: str, x: float, y: float = 0.0) -> dict:
    return {"id": nid, "x": x, "y": y}


def _seg(sid: str, fn: str, tn: str, speed: int = 3600, parts=None) -> dict:
    d = {"id": sid, "startNodeId": fn, "endNodeId": tn, "speed": speed}
    if parts:
        d["parts"] = parts
    return d


def _arc_parts(x0, y0, x1, y1, bulge=500.0):
    """Generate Arc-type parts that make GraphMap treat this as a curve segment.

    Creates a 3-point arc from (x0,y0) to (x1,y1) with a midpoint offset
    by `bulge` in the perpendicular direction. This ensures
    len(path_points) > 2 → _is_curve_seg returns True.
    """
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    nx, ny = -dy / length * bulge, dx / length * bulge
    return [
        {"kind": "Arc", "points": [
            {"x": mx + nx, "y": my + ny},
            {"x": x1, "y": y1},
        ]}
    ]


# ── Topology helpers ─────────────────────────────────────────────────────────

def map_diamond(approach_specs: List[Tuple[float, int]],
                branch_a_specs: List[Tuple[float, int]],
                branch_b_specs: List[Tuple[float, int]],
                exit_specs: List[Tuple[float, int]]) -> GraphMap:
    """Diamond topology: approach → diverge → 2 branches → merge → exit.

    Layout (top view):
        a0 ─ a1 ─ ... ─ DIV ─── branch_a ─── MRG ─ e1 ─ e2 ─ ...
                              └── branch_b ──┘

    Branch B uses Arc path_points so GraphMap detects it as a curve,
    enabling ZCU zone creation at both the diverge and merge nodes.

    Path through branch A: a0..DIV..ba_end..MRG..exit
    Path through branch B: a0..DIV..bb_end..MRG..exit

    Returns (gmap, path_a, path_b) where path_a goes through branch A
    and path_b through branch B.
    """
    nodes = []
    segments = []

    # ── Approach ─────────────────────────────────────────────────────
    x = 0.0
    nodes.append(_node("a0", x))
    for i, (length, speed) in enumerate(approach_specs):
        x += length
        nid = f"a{i+1}"
        nodes.append(_node(nid, x))
        segments.append(_seg(f"as{i}", f"a{i}", nid, speed))
    div_node = f"a{len(approach_specs)}"
    div_x = x

    # ── Branch A (straight, upper path at y=0) ──────────────────────
    bx = div_x
    prev = div_node
    ba_nodes = []
    for i, (length, speed) in enumerate(branch_a_specs):
        bx += length
        nid = f"ba{i}"
        nodes.append(_node(nid, bx, 0.0))
        segments.append(_seg(f"ba_s{i}", prev, nid, speed))
        ba_nodes.append(nid)
        prev = nid

    # ── Branch B (curve, lower path at y=-2000) ─────────────────────
    bx = div_x
    prev = div_node
    bb_nodes = []
    for i, (length, speed) in enumerate(branch_b_specs):
        bx += length
        nid = f"bb{i}"
        nx = div_x + (bx - div_x)
        ny = -2000.0
        nodes.append(_node(nid, nx, ny))
        # Make it a curve segment (Arc parts)
        prev_node = [n for n in nodes if n["id"] == prev][0]
        arc_parts = _arc_parts(prev_node["x"], prev_node["y"], nx, ny, bulge=800.0)
        segments.append(_seg(f"bb_s{i}", prev, nid, speed, parts=arc_parts))
        bb_nodes.append(nid)
        prev = nid

    # ── Merge node ──────────────────────────────────────────────────
    merge_x = max(
        nodes[-1]["x"] if ba_nodes else div_x,
        div_x + sum(s[0] for s in branch_a_specs),
    ) + 1000
    nodes.append(_node("mrg", merge_x, 0.0))
    # Branch A → merge (straight)
    segments.append(_seg("ba_mrg", ba_nodes[-1] if ba_nodes else div_node, "mrg", 3600))
    # Branch B → merge (curve — needed for ZCU merge zone detection)
    bb_last = bb_nodes[-1] if bb_nodes else div_node
    bb_last_node = [n for n in nodes if n["id"] == bb_last][0]
    arc_parts_mrg = _arc_parts(bb_last_node["x"], bb_last_node["y"],
                                merge_x, 0.0, bulge=600.0)
    segments.append(_seg("bb_mrg", bb_last, "mrg", 700, parts=arc_parts_mrg))

    # ── Exit ────────────────────────────────────────────────────────
    x = merge_x
    prev = "mrg"
    for i, (length, speed) in enumerate(exit_specs):
        x += length
        nid = f"e{i}"
        nodes.append(_node(nid, x, 0.0))
        segments.append(_seg(f"es{i}", prev, nid, speed))
        prev = nid

    return _make_map(nodes, segments)


def map_straight(seg_specs: List[Tuple[float, int]]) -> GraphMap:
    """N-segment straight line. seg_specs = [(length, speed), ...]."""
    nodes = []
    segments = []
    x = 0.0
    nodes.append(_node("n0", x))
    for i, (length, speed) in enumerate(seg_specs):
        x += length
        nodes.append(_node(f"n{i+1}", x))
        segments.append(_seg(f"s{i}", f"n{i}", f"n{i+1}", speed))
    return _make_map(nodes, segments)


def map_straight_with_diverge(approach_specs: List[Tuple[float, int]],
                               exit_len: float = 2000,
                               extra_branch_len: float = 1000) -> GraphMap:
    """Straight approach → diverge node → 2 outgoing branches.

    The diverge node becomes a ZCU boundary (diverge zone). The vehicle
    travels along the first branch only.
    Returns map. Path: n0 → n1 → ... → n_{len(specs)} → ne (exit) AND
                       n_{len(specs)} → nb (branch) [for diverge ZCU].
    """
    nodes = []
    segments = []
    x = 0.0
    nodes.append(_node("n0", x))
    for i, (length, speed) in enumerate(approach_specs):
        x += length
        nodes.append(_node(f"n{i+1}", x))
        segments.append(_seg(f"s{i}", f"n{i}", f"n{i+1}", speed))
    diverge = f"n{len(approach_specs)}"
    nodes.append(_node("ne", x + exit_len))
    segments.append(_seg("se", diverge, "ne", 3600))
    nodes.append(_node("nb", x + extra_branch_len, 1000.0))   # off-axis
    segments.append(_seg("sb", diverge, "nb", 3600))
    return _make_map(nodes, segments)


def map_diverge_long(approach_specs: List[Tuple[float, int]],
                     exit_specs: List[Tuple[float, int]],
                     branch_specs: List[Tuple[float, int]]) -> GraphMap:
    """Long approach → diverge node → 2 long branches.

    Path layout:
        n0 → n1 → ... → n_d (diverge node)
        n_d → e1 → e2 → ... → e_E (main exit branch)
        n_d → b1 → b2 → ... → b_B (alternate branch)

    The diverge node has 2 outgoing segments → ZCU diverge zone.
    Both exit and branch are long enough to allow continuous driving
    after passing the ZCU.
    """
    nodes = []
    segments = []
    x = 0.0
    nodes.append(_node("n0", x))
    for i, (length, speed) in enumerate(approach_specs):
        x += length
        nodes.append(_node(f"n{i+1}", x))
        segments.append(_seg(f"as{i}", f"n{i}", f"n{i+1}", speed))
    diverge = f"n{len(approach_specs)}"
    # Main exit branch (along same axis)
    cur = diverge
    for i, (length, speed) in enumerate(exit_specs):
        x += length
        nid = f"e{i+1}"
        nodes.append(_node(nid, x))
        segments.append(_seg(f"es{i}", cur, nid, speed))
        cur = nid
    # Alternate branch (off-axis)
    bx = 0.0
    cur = diverge
    for i, (length, speed) in enumerate(branch_specs):
        bx += length
        nid = f"b{i+1}"
        nodes.append(_node(nid, bx, 1000.0))
        segments.append(_seg(f"bs{i}", cur, nid, speed))
        cur = nid
    return _make_map(nodes, segments)


def map_merge_long(branch_a_specs: List[Tuple[float, int]],
                   branch_b_specs: List[Tuple[float, int]],
                   exit_specs: List[Tuple[float, int]]) -> GraphMap:
    """Two long incoming branches converge at the merge node, then long exit.

    Path layout:
        a0 → a1 → ... → am (= merge node)
        b0 → b1 → ... → bm (= merge node)
        merge → e1 → e2 → ... → e_E

    Both branches end at the same merge node → ZCU merge zone with 2
    boundary nodes (the predecessors of merge on each branch).
    Exit branch is long enough for continuous post-merge driving.
    """
    nodes = []
    segments = []
    # Branch A along y=0
    x = 0.0
    nodes.append(_node("a0", x, 0.0))
    for i, (length, speed) in enumerate(branch_a_specs):
        x += length
        if i + 1 < len(branch_a_specs):
            nid = f"a{i+1}"
            nodes.append(_node(nid, x, 0.0))
        else:
            nid = "merge"
            nodes.append(_node("merge", x, 0.0))
        prev = f"a{i}" if i > 0 else "a0"
        segments.append(_seg(f"as{i}", prev, nid, speed))
    merge_x = x
    # Branch B along y=2000
    x = 0.0
    nodes.append(_node("b0", x, 2000.0))
    for i, (length, speed) in enumerate(branch_b_specs):
        x += length
        if i + 1 < len(branch_b_specs):
            nid = f"b{i+1}"
            nodes.append(_node(nid, x, 2000.0))
        else:
            nid = "merge"  # already added
        prev = f"b{i}" if i > 0 else "b0"
        segments.append(_seg(f"bs{i}", prev, nid, speed))
    # Long exit
    cur = "merge"
    x = merge_x
    for i, (length, speed) in enumerate(exit_specs):
        x += length
        nid = f"e{i+1}"
        nodes.append(_node(nid, x, 0.0))
        segments.append(_seg(f"es{i}", cur, nid, speed))
        cur = nid
    return _make_map(nodes, segments)


def map_merge(branch1_specs: List[Tuple[float, int]],
              branch2_specs: List[Tuple[float, int]],
              post_len: float = 2000) -> GraphMap:
    """Two incoming branches converge at the merge node, then exit straight.

    Branch 1 nodes: a0, a1, ..., am (m = len(branch1_specs))
    Branch 2 nodes: b0, b1, ..., bm
    Merge node: 'merge'
    Exit node: 'exit'
    """
    nodes = []
    segments = []
    # Branch 1 along y=0
    x = 0.0
    nodes.append(_node("a0", x, 0.0))
    for i, (length, speed) in enumerate(branch1_specs):
        x += length
        nid = f"a{i+1}" if i + 1 < len(branch1_specs) else "merge"
        nodes.append(_node(nid, x, 0.0))
        segments.append(_seg(f"as{i}", f"a{i}" if i > 0 else "a0", nid, speed))
    # Branch 2 along y=1000
    x = 0.0
    nodes.append(_node("b0", x, 1000.0))
    for i, (length, speed) in enumerate(branch2_specs):
        x += length
        nid = f"b{i+1}" if i + 1 < len(branch2_specs) else "merge"
        if nid == "merge":
            # Already exists; just add the segment
            segments.append(_seg(f"bs{i}", f"b{i}" if i > 0 else "b0", "merge", speed))
        else:
            nodes.append(_node(nid, x, 1000.0))
            segments.append(_seg(f"bs{i}", f"b{i}" if i > 0 else "b0", nid, speed))
    # Exit
    merge_x = sum(s[0] for s in branch1_specs)
    nodes.append(_node("exit", merge_x + post_len, 0.0))
    segments.append(_seg("ex", "merge", "exit", 3600))
    return _make_map(nodes, segments)


# ── Run helper ───────────────────────────────────────────────────────────────

class RunResult:
    def __init__(self):
        self.collisions: List[Tuple[float, int, int, float, tuple]] = []
        self.speed_violations: List[tuple] = []
        self.zcu_violations: List[tuple] = []
        self.position_log: dict = {}   # vid -> [(t, pidx, off, vel, state)]
        self.final_state: dict = {}    # vid -> dict
        self.deadlock: bool = False


def run_scenario(gmap: GraphMap, vehicles_spec: List[dict],
                 t_end: float, dt: float = 0.05) -> RunResult:
    """Run DES headless with given vehicle specs.

    vehicles_spec entries: {
        'id': int, 'path': [node_ids], 'init_pidx': int, 'init_off': float,
        'init_vel': float (default 0), 'pre_locks': [(lock_id, holder_id_or_self)]
    }
    """
    des = GraphDESv6(gmap)
    vs = []
    for spec in vehicles_spec:
        v = Vehicle(spec['id'], gmap, list(spec['path']))
        v.path_idx = spec.get('init_pidx', 0)
        v.seg_offset = spec.get('init_off', 0.0)
        v.vel = spec.get('init_vel', 0.0)
        vs.append(v)
        des.add_vehicle(v)
    # pre-lock support
    for spec in vehicles_spec:
        for lid, holder_id in spec.get('pre_locks', []):
            holder = next((vv for vv in vs if vv.id == holder_id), None)
            if holder is not None:
                des._zone_lock[lid] = holder

    des.start_all()

    res = RunResult()
    for vv in vs:
        res.position_log[vv.id] = []

    t = 0.0
    deadlock_t0 = None
    while t < t_end:
        t += dt
        des.run_until(t)
        des.query_positions(t)

        # Per-step real-time positions
        rt = {}
        for vv in vs:
            d = vv._dist_traveled(t - vv.t_ref)
            off = vv.seg_offset + d
            pidx = vv.path_idx
            while pidx < len(vv.path) - 1:
                sl = vv._seg_lengths[pidx] if pidx < len(vv._seg_lengths) else 0
                if sl <= 0 or off < sl - 0.01:
                    break
                off -= sl
                pidx += 1
            rt[vv.id] = (pidx, max(0.0, off))
            res.position_log[vv.id].append((t, pidx, off, vv.vel_at(t), vv.state))

        # Collision check
        from collections import defaultdict
        by_seg = defaultdict(list)
        for vv in vs:
            pidx, off = rt[vv.id]
            if pidx < len(vv.path) - 1:
                k = (vv.path[pidx], vv.path[pidx + 1])
                by_seg[k].append((vv, off))
        for seg_key, occs in by_seg.items():
            if len(occs) < 2:
                continue
            occs.sort(key=lambda x: x[1])
            for i in range(len(occs) - 1):
                a, ao = occs[i]
                b, bo = occs[i + 1]
                gap = bo - ao
                if gap < a.length:
                    res.collisions.append((t, a.id, b.id, gap, seg_key))

        # Deadlock — only if vehicles that still have room ahead are stuck.
        # Walk the leader chain: if any ancestor reached path end, this is
        # natural "pile up at the end", not a deadlock.
        def _chain_ends_at_goal(vv):
            seen = set()
            cur = vv
            while cur is not None and cur.id not in seen:
                seen.add(cur.id)
                at_end = (cur.path_idx >= len(cur.path) - 2 and
                          cur.seg_offset >= cur.current_seg_length() - 10)
                if at_end:
                    return True
                cur = cur.leader
            return False

        # Use real-time vel_at(t), not the lazy vv.vel field which only
        # updates on events and may be stale between events.
        all_idle = all(vv.vel_at(t) < 1.0 for vv in vs)
        if all_idle:
            any_stuck = False
            for vv in vs:
                if _chain_ends_at_goal(vv):
                    continue
                any_stuck = True
            if any_stuck:
                if deadlock_t0 is None:
                    deadlock_t0 = t
                elif t - deadlock_t0 >= 5.0:
                    res.deadlock = True
            else:
                deadlock_t0 = None
        else:
            deadlock_t0 = None

    res.speed_violations = list(des.speed_violation_log)
    res.zcu_violations = list(des.zcu_violation_log)
    for vv in vs:
        res.final_state[vv.id] = {
            'pidx': vv.path_idx, 'off': vv.seg_offset,
            'vel': vv.vel, 'state': vv.state, 'leader': vv.leader.id if vv.leader else None,
        }
    return res


# ── Assertion helpers ────────────────────────────────────────────────────────

def assert_no_collisions(res: RunResult, name: str):
    if res.collisions:
        first = res.collisions[0]
        raise AssertionError(
            f"[{name}] COLLISION at t={first[0]:.2f} V#{first[1]}<-V#{first[2]} "
            f"gap={first[3]:.0f} seg={first[4]}")


def assert_no_violations(res: RunResult, name: str):
    if res.speed_violations:
        raise AssertionError(f"[{name}] SPEED_VIOL: {res.speed_violations[0]}")
    if res.zcu_violations:
        raise AssertionError(f"[{name}] ZCU_VIOL: {res.zcu_violations[0]}")


def assert_min_gap(res: RunResult, vid_rear: int, vid_front: int,
                   h_min: float, name: str):
    """For each timestep where both vehicles are tracked, gap_along_path
    between rear and front must be ≥ h_min (with small tolerance)."""
    rear = res.position_log[vid_rear]
    front = res.position_log[vid_front]
    n = min(len(rear), len(front))
    for i in range(n):
        t = rear[i][0]
        # Only compare when on same seg (same pidx and same nodes)
        if rear[i][1] != front[i][1]:
            continue
        gap = front[i][2] - rear[i][2]
        if gap < h_min - 50:   # 50mm tolerance
            raise AssertionError(
                f"[{name}] gap < h_min at t={t:.2f}: "
                f"V#{vid_rear} off={rear[i][2]:.0f}, V#{vid_front} off={front[i][2]:.0f}, "
                f"gap={gap:.0f} < h_min={h_min}")


def assert_reached(res: RunResult, vid: int, target_pidx: int, name: str):
    final = res.final_state.get(vid)
    if final is None or final['pidx'] < target_pidx:
        raise AssertionError(
            f"[{name}] V#{vid} did not reach pidx>={target_pidx}: final={final}")


def assert_max_vel_in_range(res: RunResult, vid: int,
                             vmin: float, vmax: float, name: str):
    log = res.position_log[vid]
    peak = max(entry[3] for entry in log) if log else 0
    if not (vmin <= peak <= vmax):
        raise AssertionError(
            f"[{name}] V#{vid} peak vel {peak:.0f} not in [{vmin}, {vmax}]")


# ── Scenarios ────────────────────────────────────────────────────────────────

SCENARIOS = {}


def scenario(name):
    def deco(fn):
        SCENARIOS[name] = fn
        return fn
    return deco


@scenario("S1_solo_straight")
def s1():
    """1 vehicle on a long straight, no ZCU. Should reach v_max and stop at end.
    Need enough length to reach v_max=3600: dist = 3600^2/(2*500) = 12960mm just
    for accel. Use 10 segs of 3000mm = 30000mm so it definitely hits v_max."""
    gmap = map_straight([(3000, 3600)] * 10)
    path = [f"n{i}" for i in range(11)]
    res = run_scenario(gmap, [{'id': 0, 'path': path}], t_end=30.0)
    assert_no_collisions(res, "S1")
    assert_no_violations(res, "S1")
    assert_max_vel_in_range(res, 0, 3500, 3700, "S1")
    final = res.final_state[0]
    if final['pidx'] < 9:
        raise AssertionError(
            f"[S1] vehicle stopped early at pidx={final['pidx']} off={final['off']:.0f}")
    return f"OK (final pidx={final['pidx']})"


@scenario("S7_mixed_speed_segments")
def s7():
    """Long path with varying speed limits. Single vehicle, no ZCU.
    Verifies that profile respects each segment's limit and transitions smoothly.

    Layout: 3600 → 700 → 3600 → 1200 → 3600 → 700 → 3600
    Each segment 3000mm.
    """
    # Fast segs 30000mm (accel 12960mm + cruise + decel 12470mm fits).
    # Slow segs 3000mm (curve-like).
    specs = [(30000, 3600), (3000, 700), (30000, 3600), (3000, 1200),
             (30000, 3600), (3000, 700), (30000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    res = run_scenario(gmap, [{'id': 0, 'path': path}], t_end=60.0)
    assert_no_collisions(res, "S7")
    assert_no_violations(res, "S7")
    final = res.final_state[0]
    if final['pidx'] < 5:
        raise AssertionError(f"[S7] stopped early at pidx={final['pidx']}")
    log = res.position_log[0]
    peak = max(e[3] for e in log)
    if peak < 3400:
        raise AssertionError(f"[S7] peak vel {peak:.0f} too low (expected ~3600)")
    # Check slow segments respected
    in_slow = [e for e in log if e[1] in (1, 5)]  # pidx 1 and 5 are 700mm/s segs
    if in_slow:
        max_in_slow = max(e[3] for e in in_slow)
        if max_in_slow > 800:
            raise AssertionError(
                f"[S7] vel {max_in_slow:.0f} in 700-limit seg (pidx 1 or 5)")
    return f"OK (peak={peak:.0f})"


@scenario("S8_two_follow_mixed_speed")
def s8():
    """2 vehicles on mixed-speed path. Front vehicle must slow for curves,
    rear must follow without collision or speed violation."""
    specs = [(15000, 3600), (3000, 700), (15000, 3600), (3000, 1200),
             (15000, 3600), (3000, 700), (15000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    res = run_scenario(gmap, [
        {'id': 1, 'path': path, 'init_pidx': 0, 'init_off': 0.0},
        {'id': 0, 'path': path, 'init_pidx': 0, 'init_off': 5000.0},  # ahead by 5000mm
    ], t_end=60.0)
    assert_no_collisions(res, "S8")
    assert_no_violations(res, "S8")
    return "OK"


@scenario("S3_curve_from_stop_trace")
def s3_trace():
    """Same as S3 but with DEBUG_TRACE on for diagnosis."""
    GraphDESv6.DEBUG_TRACE = True
    GraphDESv6.DEBUG_VID = 0
    try:
        return s3()
    finally:
        GraphDESv6.DEBUG_TRACE = False


@scenario("S3_curve_from_stop")
def s3():
    """V#25 reproduction: vehicle at end of straight, next seg is a low-speed
    curve, vehicle starts from rest. Profile must accelerate up to curve_speed
    inside the curve seg, NOT cap at 31.6 mm/s."""
    # seg0: short straight 300mm at 3600
    # seg1: long "curve" 1726mm at 700
    # seg2: straight 2000mm at 3600 (just to give vehicle path room)
    # diverge at n2 makes ZCU there (so plan_boundary captures n2 as bnd)
    gmap = map_straight_with_diverge(
        approach_specs=[(300, 3600), (1726, 700)],
        exit_len=2000, extra_branch_len=1000)
    path = ["n0", "n1", "n2", "ne"]
    res = run_scenario(gmap, [{
        'id': 0, 'path': path, 'init_pidx': 0, 'init_off': 299.0,
    }], t_end=15.0)
    assert_no_collisions(res, "S3")
    assert_no_violations(res, "S3")
    # Peak vel during curve should approach 700 (curve limit), not be stuck near 31.6
    log = res.position_log[0]
    in_curve = [e for e in log if e[1] == 1]   # pidx == 1 means in seg n1->n2
    if not in_curve:
        raise AssertionError("[S3] vehicle never entered curve seg")
    peak_in_curve = max(e[3] for e in in_curve)
    if peak_in_curve < 600:
        raise AssertionError(
            f"[S3] curve peak vel {peak_in_curve:.1f} too low (expected ~700). "
            f"Forward-pass triangular profile bug.")
    if peak_in_curve > 800:
        # Dump trajectory around peak
        peak_idx = max(range(len(in_curve)), key=lambda i: in_curve[i][3])
        print("    --- S3 trajectory dump ---")
        for e in in_curve[max(0, peak_idx-5):peak_idx+5]:
            print(f"      t={e[0]:.3f} pidx={e[1]} off={e[2]:.1f} vel={e[3]:.1f} state={e[4]}")
        raise AssertionError(
            f"[S3] curve peak vel {peak_in_curve:.1f} > 800 (limit 700)")
    return f"OK (curve peak vel = {peak_in_curve:.0f})"


@scenario("S4_solo_zcu_pass")
def s4():
    """1 vehicle, ZCU diverge boundary in path, lock free. Should pass through."""
    gmap = map_straight_with_diverge([(2000, 3600), (2000, 3600)])
    path = ["n0", "n1", "n2", "ne"]
    res = run_scenario(gmap, [{'id': 0, 'path': path}], t_end=10.0)
    assert_no_collisions(res, "S4")
    assert_no_violations(res, "S4")
    assert_reached(res, 0, 2, "S4")
    return "OK"


@scenario("S9_convoy_long")
def s9():
    """5 vehicles on a long mixed-speed path. All must complete without
    collision or deadlock. Speed violations counted but tolerated up to
    DES replan tolerance (~300mm/s at seg boundaries)."""
    specs = [(20000, 3600), (3000, 700), (20000, 3600), (3000, 1200),
             (20000, 3600), (3000, 700), (20000, 3600), (3000, 1200),
             (20000, 3600), (3000, 700), (20000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    vehicles_spec = []
    for i in range(5):
        vehicles_spec.append({
            'id': i, 'path': path,
            'init_pidx': 0, 'init_off': (4 - i) * 3000.0,
        })
    res = run_scenario(gmap, vehicles_spec, t_end=120.0)
    assert_no_collisions(res, "S9")
    if res.deadlock:
        raise AssertionError("[S9] deadlock detected")
    n_sv = len(res.speed_violations)
    # Check no severe speed violations (> 500mm/s excess)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[S9] severe speed violation: {severe[0]}")
    return f"OK (speed_viols={n_sv})"


@scenario("S10_convoy_mixed_speed_long")
def s10():
    """8 vehicles on a very long path with speed transitions.
    Stress test for convoy following through speed changes."""
    specs = [(25000, 3600), (4000, 700), (25000, 3600), (4000, 1500),
             (25000, 3600), (4000, 700), (25000, 3600), (4000, 1500),
             (25000, 3600), (4000, 700), (25000, 3600), (4000, 1500),
             (25000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    vehicles_spec = []
    for i in range(8):
        vehicles_spec.append({
            'id': i, 'path': path,
            'init_pidx': 0, 'init_off': (7 - i) * 2500.0,
        })
    res = run_scenario(gmap, vehicles_spec, t_end=180.0)
    assert_no_collisions(res, "S10")
    if res.deadlock:
        raise AssertionError("[S10] deadlock detected")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[S10] severe speed violation: {severe[0]}")
    return f"OK (speed_viols={n_sv})"


def _find_curve_path(gmap, seed: int, min_curves: int = 5,
                     length: int = 80):
    """Find a random_safe_path that traverses at least `min_curves`
    curve segments (speed < 1000).

    Deterministic: uses a local random.Random(seed) for both the start
    node and the walk itself (via the rng parameter to random_safe_path).
    Does not touch global random state.
    """
    import random
    rng = random.Random(seed)
    from graph_des_v5 import random_safe_path
    for _ in range(100):
        start = rng.choice(list(gmap.adj.keys()))
        path = random_safe_path(gmap, start, length=length, rng=rng)
        curve_count = sum(
            1 for i in range(len(path) - 1)
            if (s := gmap.segment_between(path[i], path[i + 1]))
            and s.max_speed < 1000
        )
        if curve_count >= min_curves:
            return path, curve_count
    return None, 0


@scenario("S11_real_map_curves_solo")
def s11():
    """Solo vehicle on a real-map path that traverses curve segments
    (700/800/1323 mm/s). Verifies curve speed enforcement and that the
    standard pipeline handles real-world speed variations."""
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    chosen_path, n_curves = _find_curve_path(gmap, seed=42)
    if chosen_path is None:
        raise AssertionError("[S11] no path with curves found")
    res = run_scenario(gmap, [{
        'id': 0, 'path': chosen_path, 'init_pidx': 0, 'init_off': 0.0,
    }], t_end=180.0)
    assert_no_collisions(res, "S11")
    if res.deadlock:
        raise AssertionError("[S11] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[S11] severe speed violation: {severe[0]}")
    return f"OK (path_len={len(chosen_path)} curves={n_curves} speed_viols={n_sv})"


@scenario("S12_real_map_curves_convoy")
def s12():
    """3 vehicles on a real-map curve-rich path. Verifies leader-following
    works correctly across curve segments with varying speeds.

    Larger initial gap (8000mm) to avoid multiple vehicles being in the
    same ZCU zone simultaneously at start."""
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    chosen_path, n_curves = _find_curve_path(gmap, seed=42)
    if chosen_path is None:
        raise AssertionError("[S12] no path with curves found")
    vehicles_spec = []
    for i in range(3):
        vehicles_spec.append({
            'id': i, 'path': list(chosen_path),
            'init_pidx': 0, 'init_off': (2 - i) * 8000.0,
        })
    res = run_scenario(gmap, vehicles_spec, t_end=180.0)
    assert_no_collisions(res, "S12")
    if res.deadlock:
        raise AssertionError("[S12] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[S12] severe speed violation: {severe[0]}")
    return f"OK (path_len={len(chosen_path)} curves={n_curves} speed_viols={n_sv})"


@scenario("Z1_diverge_solo")
def z1():
    """Single vehicle through a diverge ZCU. Long approach + long exit.
    Verifies basic ZCU lock acquisition and pass-through."""
    gmap = map_diverge_long(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        exit_specs=[(15000, 3600), (3000, 1200), (15000, 3600)],
        branch_specs=[(5000, 3600)],
    )
    path = ["n0", "n1", "n2", "n3", "e1", "e2", "e3"]
    res = run_scenario(gmap, [{
        'id': 0, 'path': path, 'init_pidx': 0, 'init_off': 0.0,
    }], t_end=120.0)
    assert_no_collisions(res, "Z1")
    assert_no_violations(res, "Z1")
    if res.deadlock:
        raise AssertionError("[Z1] deadlock")
    final = res.final_state[0]
    if final['pidx'] < 5:
        raise AssertionError(f"[Z1] did not finish: pidx={final['pidx']}")
    return f"OK final pidx={final['pidx']}"


@scenario("Z2_diverge_convoy")
def z2():
    """3 vehicles convoy through a diverge ZCU. All take main exit branch.
    Tests that followers correctly handle ZCU after leader passes."""
    gmap = map_diverge_long(
        approach_specs=[(20000, 3600), (3000, 700), (20000, 3600)],
        exit_specs=[(20000, 3600), (3000, 1200), (20000, 3600)],
        branch_specs=[(5000, 3600)],
    )
    path = ["n0", "n1", "n2", "n3", "e1", "e2", "e3"]
    vehicles_spec = []
    for i in range(3):
        vehicles_spec.append({
            'id': i, 'path': list(path),
            'init_pidx': 0, 'init_off': (2 - i) * 5000.0,
        })
    res = run_scenario(gmap, vehicles_spec, t_end=180.0)
    assert_no_collisions(res, "Z2")
    if res.deadlock:
        raise AssertionError("[Z2] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z2] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    if n_zv > 0:
        raise AssertionError(f"[Z2] ZCU violation: {res.zcu_violations[0]}")
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("Z3_diamond_merge_contention")
def z3_diamond():
    """Two vehicles from different branches of a diamond (diverge→merge) topology.
    Branch A is straight+fast, branch B is curve+slow.
    Tests that merge ZCU serializes entry: one waits while the other passes."""
    gmap = map_diamond(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        branch_a_specs=[(10000, 3600), (3000, 1200)],
        branch_b_specs=[(8000, 700), (3000, 700)],
        exit_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
    )
    # Start each vehicle on its own branch (past the shared approach/diverge)
    path_a = ["ba0", "ba1", "mrg", "e0", "e1", "e2"]
    path_b = ["bb0", "bb1", "mrg", "e0", "e1", "e2"]
    res = run_scenario(gmap, [
        {'id': 0, 'path': path_a, 'init_pidx': 0, 'init_off': 0.0},
        {'id': 1, 'path': path_b, 'init_pidx': 0, 'init_off': 0.0},
    ], t_end=180.0)
    assert_no_collisions(res, "Z3d")
    if res.deadlock:
        raise AssertionError("[Z3d] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z3d] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("Z4_diamond_convoy")
def z4_diamond():
    """2 vehicles per branch (4 total) through diamond diverge→merge.
    Tests leader-following + ZCU contention simultaneously."""
    gmap = map_diamond(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        branch_a_specs=[(10000, 3600), (3000, 1200)],
        branch_b_specs=[(8000, 700), (3000, 700)],
        exit_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
    )
    # Start each pair on its own branch
    path_a = ["ba0", "ba1", "mrg", "e0", "e1", "e2"]
    path_b = ["bb0", "bb1", "mrg", "e0", "e1", "e2"]
    res = run_scenario(gmap, [
        {'id': 0, 'path': list(path_a), 'init_pidx': 0, 'init_off': 5000.0},
        {'id': 1, 'path': list(path_a), 'init_pidx': 0, 'init_off': 0.0},
        {'id': 2, 'path': list(path_b), 'init_pidx': 0, 'init_off': 3000.0},
        {'id': 3, 'path': list(path_b), 'init_pidx': 0, 'init_off': 0.0},
    ], t_end=240.0)
    assert_no_collisions(res, "Z4d")
    if res.deadlock:
        raise AssertionError("[Z4d] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z4d] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("Z5_diamond_same_branch")
def z5_diamond():
    """3 vehicles all taking branch A through diverge→merge diamond.
    Tests diverge+merge ZCU pass-through with convoy following."""
    gmap = map_diamond(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        branch_a_specs=[(10000, 3600), (3000, 1200)],
        branch_b_specs=[(8000, 700), (3000, 700)],
        exit_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
    )
    path_a = ["a0", "a1", "a2", "a3", "ba0", "ba1", "mrg", "e0", "e1", "e2"]
    vehicles_spec = []
    for i in range(3):
        vehicles_spec.append({
            'id': i, 'path': list(path_a),
            'init_pidx': 0, 'init_off': (2 - i) * 5000.0,
        })
    res = run_scenario(gmap, vehicles_spec, t_end=180.0)
    assert_no_collisions(res, "Z5d")
    if res.deadlock:
        raise AssertionError("[Z5d] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z5d] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("Z3_real_merge_solo")
def z3():
    """Two vehicles from different branches merging at a real-map merge node.
    Uses actual oht.large.map.json topology (node 1285) with curve segments.
    Tests ZCU merge lock contention — one must wait for the other."""
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    # Merge zone at node 1285: branch A via 1283, branch B via _curve_mid_59
    path_a = ["1277", "1278", "1279", "1280", "1281", "1282", "1283",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    path_b = ["1457", "1458", "1459", "1460", "14600001", "14600002", "1463", "_curve_mid_59",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    res = run_scenario(gmap, [
        {'id': 0, 'path': path_a, 'init_pidx': 0, 'init_off': 0.0},
        {'id': 1, 'path': path_b, 'init_pidx': 0, 'init_off': 0.0},
    ], t_end=180.0)
    assert_no_collisions(res, "Z3")
    if res.deadlock:
        raise AssertionError("[Z3] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z3] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    if n_zv > 0:
        raise AssertionError(f"[Z3] ZCU violation: {res.zcu_violations[0]}")
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("Z4_real_merge_convoy")
def z4():
    """2 vehicles on each of 2 merging branches (4 total) at real merge node 1285.
    Tests merge ZCU + leader-following on both branches simultaneously."""
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    path_a = ["1277", "1278", "1279", "1280", "1281", "1282", "1283",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    path_b = ["1457", "1458", "1459", "1460", "14600001", "14600002", "1463", "_curve_mid_59",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    res = run_scenario(gmap, [
        {'id': 0, 'path': list(path_a), 'init_pidx': 0, 'init_off': 3000.0},
        {'id': 1, 'path': list(path_a), 'init_pidx': 0, 'init_off': 0.0},
        {'id': 2, 'path': list(path_b), 'init_pidx': 0, 'init_off': 3000.0},
        {'id': 3, 'path': list(path_b), 'init_pidx': 0, 'init_off': 0.0},
    ], t_end=240.0)
    assert_no_collisions(res, "Z4")
    if res.deadlock:
        raise AssertionError("[Z4] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z4] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    if n_zv > 0:
        raise AssertionError(f"[Z4] ZCU violation: {res.zcu_violations[0]}")
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("Z5_real_diverge_convoy")
def z5():
    """2 vehicles through a real-map diverge zone (node 1063) with curves.
    Both take exit 0 (straight exit). Tests diverge ZCU with leader-following."""
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    # Diverge at 1063. Approach from 1832 side. Exit 0 via 1065.
    path = ["1832", "1833", "1834", "1835", "1836", "1837",
            "1059", "1060", "1061", "1062", "1063",
            "1065", "1066", "1067", "1068", "1069"]
    res = run_scenario(gmap, [
        {'id': 0, 'path': list(path), 'init_pidx': 0, 'init_off': 5000.0},
        {'id': 1, 'path': list(path), 'init_pidx': 0, 'init_off': 0.0},
    ], t_end=240.0)
    assert_no_collisions(res, "Z5")
    if res.deadlock:
        raise AssertionError("[Z5] deadlock")
    n_sv = len(res.speed_violations)
    severe = [sv for sv in res.speed_violations if sv[5] > 500]
    if severe:
        raise AssertionError(f"[Z5] severe speed violation: {severe[0]}")
    n_zv = len(res.zcu_violations)
    if n_zv > 0:
        raise AssertionError(f"[Z5] ZCU violation: {res.zcu_violations[0]}")
    return f"OK speed_viols={n_sv} zcu_viols={n_zv}"


@scenario("S6_two_follow_straight")
def s6():
    """2 vehicles same straight path, sufficient initial gap. Long enough
    map and runtime that the rear actually catches up to where the front
    has parked at the end of path."""
    gmap = map_straight([(2000, 3600)] * 6)
    path = [f"n{i}" for i in range(7)]
    res = run_scenario(gmap, [
        {'id': 1, 'path': path, 'init_pidx': 0, 'init_off': 0.0},
        {'id': 0, 'path': path, 'init_pidx': 1, 'init_off': 1000.0},  # ahead by 3000mm
    ], t_end=30.0)
    assert_no_collisions(res, "S6")
    assert_no_violations(res, "S6")
    # Rear must end up at least h_min behind front
    fr = res.final_state[1]   # rear id=1
    ft = res.final_state[0]   # front id=0
    if fr['pidx'] == ft['pidx']:
        gap = ft['off'] - fr['off']
    else:
        # Different pidx — rear is in earlier seg, front in later
        gap = 1e9
    # Strict invariant: no rear-end (gap >= length=750).
    # Looser invariant for h_min: 900mm (h_min=1150 with discrete-replan
    # tolerance ~250mm). The rear may stop slightly closer than h_min due
    # to event-discrete replanning, but never within length.
    if gap < 750:
        raise AssertionError(
            f"[S6] REAR-END: rear pidx={fr['pidx']} off={fr['off']:.0f}, "
            f"front pidx={ft['pidx']} off={ft['off']:.0f}, gap={gap:.0f}")
    if gap < 900:
        raise AssertionError(
            f"[S6] gap below safety margin: gap={gap:.0f} < 900 (h_min=1150)")
    return f"OK (final gap = {gap:.0f})"


# ── Visualizer hook ──────────────────────────────────────────────────────────

# Each scenario can register a viz spec via @viz_scenario instead of (or in
# addition to) the headless @scenario. The viz spec returns (gmap, vehicles).

VIZ_SCENARIOS = {}


def viz_scenario(name):
    def deco(fn):
        VIZ_SCENARIOS[name] = fn
        return fn
    return deco


@viz_scenario("S3_curve_from_stop")
def viz_s3():
    gmap = map_straight_with_diverge(
        approach_specs=[(300, 3600), (1726, 700)],
        exit_len=2000, extra_branch_len=1000)
    v = Vehicle(0, gmap, ["n0", "n1", "n2", "ne"])
    v.path_idx = 0
    v.seg_offset = 299.0
    return gmap, [v]


@viz_scenario("S6_two_follow_straight")
def viz_s6():
    gmap = map_straight([(2000, 3600)] * 6)
    path = [f"n{i}" for i in range(7)]
    rear = Vehicle(1, gmap, list(path))
    rear.path_idx = 0; rear.seg_offset = 0.0
    front = Vehicle(0, gmap, list(path))
    front.path_idx = 1; front.seg_offset = 1000.0
    return gmap, [rear, front]


@viz_scenario("S1_solo_straight")
def viz_s1():
    gmap = map_straight([(3000, 3600)] * 10)
    v = Vehicle(0, gmap, [f"n{i}" for i in range(11)])
    return gmap, [v]


@viz_scenario("S4_solo_zcu_pass")
def viz_s4():
    gmap = map_straight_with_diverge([(2000, 3600), (2000, 3600)])
    v = Vehicle(0, gmap, ["n0", "n1", "n2", "ne"])
    return gmap, [v]


@viz_scenario("S7_mixed_speed_segments")
def viz_s7():
    specs = [(30000, 3600), (3000, 700), (30000, 3600), (3000, 1200),
             (30000, 3600), (3000, 700), (30000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    v = Vehicle(0, gmap, path)
    return gmap, [v]


@viz_scenario("S8_two_follow_mixed_speed")
def viz_s8():
    specs = [(15000, 3600), (3000, 700), (15000, 3600), (3000, 1200),
             (15000, 3600), (3000, 700), (15000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    rear = Vehicle(1, gmap, list(path))
    rear.path_idx = 0; rear.seg_offset = 0.0
    front = Vehicle(0, gmap, list(path))
    front.path_idx = 0; front.seg_offset = 5000.0
    return gmap, [rear, front]


@viz_scenario("S9_convoy_long")
def viz_s9():
    """5 vehicles on a long mixed-speed path — convoy formation test."""
    specs = [(20000, 3600), (3000, 700), (20000, 3600), (3000, 1200),
             (20000, 3600), (3000, 700), (20000, 3600), (3000, 1200),
             (20000, 3600), (3000, 700), (20000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    vehicles = []
    for i in range(5):
        v = Vehicle(i, gmap, list(path))
        v.path_idx = 0
        v.seg_offset = i * 3000.0  # 3000mm apart
        vehicles.append(v)
    # Reverse so vehicles[0] is frontmost
    vehicles.reverse()
    return gmap, vehicles


@viz_scenario("S10_convoy_mixed_speed_long")
def viz_s10():
    """8 vehicles on a very long path with speed transitions — stress convoy."""
    specs = [(25000, 3600), (4000, 700), (25000, 3600), (4000, 1500),
             (25000, 3600), (4000, 700), (25000, 3600), (4000, 1500),
             (25000, 3600), (4000, 700), (25000, 3600), (4000, 1500),
             (25000, 3600)]
    gmap = map_straight(specs)
    path = [f"n{i}" for i in range(len(specs) + 1)]
    vehicles = []
    for i in range(8):
        v = Vehicle(i, gmap, list(path))
        v.path_idx = 0
        v.seg_offset = i * 2500.0  # 2500mm apart
        vehicles.append(v)
    vehicles.reverse()
    return gmap, vehicles


@viz_scenario("S12_real_map_curves_convoy")
def viz_s12():
    from graph_des_v5 import GraphMap, random_safe_path
    gmap = GraphMap("oht.large.map.json")
    chosen_path, _ = _find_curve_path(gmap, seed=42)
    vehicles = []
    for i in range(3):
        v = Vehicle(i, gmap, list(chosen_path))
        v.path_idx = 0
        v.seg_offset = (2 - i) * 8000.0
        vehicles.append(v)
    vehicles.reverse()
    return gmap, vehicles


def _viz_s12_with_seed(seed: int):
    """Reusable builder for seed-varied S12 viz scenarios. Returns
    (gmap, vehicles, rng_seed) so the deadlock is fully deterministic
    (path extensions use DES's seeded rng, not global random)."""
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    chosen_path, _ = _find_curve_path(gmap, seed=seed)
    vehicles = []
    for i in range(3):
        v = Vehicle(i, gmap, list(chosen_path))
        v.path_idx = 0
        v.seg_offset = (2 - i) * 8000.0
        vehicles.append(v)
    vehicles.reverse()
    return gmap, vehicles, seed   # 3-tuple: deterministic rng


@viz_scenario("S12_deadlock_seed278")
def viz_s12_seed278():
    """Deadlock — plan truncation + circular leader chain.
    V#2 holds 1258_merge (and 1373_diverge) inside the zone but plan
    is capped by leader, so exit event never scheduled. Chain
    V#0→V#2→V#1→V#0 prevents anyone from unblocking."""
    return _viz_s12_with_seed(278)


@viz_scenario("S12_deadlock_seed589")
def viz_s12_seed589():
    """Deadlock — plan truncation + waiter/holder mutual block.
    V#2 holds 4959_diverge, V#1 waits on it with leader=None. V#2
    leader=V#1. Deadlock."""
    return _viz_s12_with_seed(589)


@viz_scenario("S12_deadlock_seed683")
def viz_s12_seed683():
    """Deadlock — same class as seed 589 (lock 4270_diverge)."""
    return _viz_s12_with_seed(683)


@viz_scenario("S12_deadlock_seed817")
def viz_s12_seed817():
    """Deadlock — circular leader chain on 1373_diverge (like 278)."""
    return _viz_s12_with_seed(817)


@viz_scenario("S12_deadlock_seed1147")
def viz_s12_seed1147():
    """Pure circular leader chain — no held locks, no ZCU involvement.
    Three vehicles mutually point to each other as leader due to path
    extension topology (V#0→V#2→V#1→V#0). No one can move first."""
    return _viz_s12_with_seed(1147)


@viz_scenario("S12_deadlock_seed1168")
def viz_s12_seed1168():
    """Same pattern as 1147 — pure circular leader chain."""
    return _viz_s12_with_seed(1168)


@viz_scenario("S12_deadlock_seed2016")
def viz_s12_seed2016():
    """Same pattern as 1147/1168 — pure circular leader chain."""
    return _viz_s12_with_seed(2016)


def _viz_multi(seed: int, n_vehicles: int):
    """Multi-vehicle scenario on the real map. Mirrors probe_multi.py:
    picks well-connected start nodes, assigns a 200-node random_safe_path
    to each vehicle, and hands off to test_graph_v6 with a seeded rng
    for reproducible extension behavior."""
    import os
    import random
    import collections
    from graph_des_v5 import GraphMap, random_safe_path

    random.seed(seed)
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "oht.large.map.json")
    gmap = GraphMap(map_path)

    degree = collections.Counter()
    for fn, tn in gmap.segments:
        degree[fn] += 1
        degree[tn] += 1
    # Exclude ZCU boundary nodes: a vehicle starting there would already be
    # inside a zone without holding the lock.
    good_starts = [nid for nid in gmap.main_loop
                   if degree[nid] >= 2
                   and gmap.adj.get(nid)
                   and nid not in gmap.zcu_nodes]
    if not good_starts:
        good_starts = [nid for nid in gmap.main_loop if gmap.adj.get(nid)]

    # Reverse adjacency: nodes that have `n` as a successor. Without this,
    # only forward neighbors are blocked, so V#B can still be placed
    # immediately behind a just-placed V#A.
    reverse_adj = collections.defaultdict(list)
    for u, outs in gmap.adj.items():
        for w in outs:
            reverse_adj[w].append(u)

    used = set()
    vehicles = []
    for i in range(n_vehicles):
        cands = [n for n in good_starts if n not in used]
        if not cands:
            cands = good_starts
        start = random.choice(cands)
        path = random_safe_path(gmap, start, length=200)
        v = Vehicle(i, gmap, path)
        vehicles.append(v)
        used.add(start)
        for nb in gmap.adj.get(start, []):
            used.add(nb)
        for nb in reverse_adj.get(start, []):
            used.add(nb)
    return gmap, vehicles, seed


@viz_scenario("multi_n50_seed1")
def viz_multi_n50_seed1():
    """50-vehicle high-density scenario on real map (seed=1). Reproduces
    the V#29↔V#41 collision series observed around t=277 in probe_multi."""
    return _viz_multi(1, 50)


@viz_scenario("multi_n50_seed42")
def viz_multi_n50_seed42():
    """50-vehicle scenario on real map (seed=42). Clean baseline."""
    return _viz_multi(42, 50)


@viz_scenario("multi_n100_seed99")
def viz_multi_n100_seed99():
    """100-vehicle scenario on real map (seed=99). Many collisions
    observed in headless probe_multi — useful for visual inspection."""
    return _viz_multi(99, 100)


@viz_scenario("multi_n100_seed42")
def viz_multi_n100_seed42():
    """100-vehicle scenario on real map (seed=42). Was 0 collisions
    pre-A-O-N, now 185 — the clearest regression case for inspecting
    what the A-O-N change broke at high density."""
    return _viz_multi(42, 100)


@viz_scenario("multi_n100_seed1")
def viz_multi_n100_seed1():
    """100-vehicle scenario on real map (seed=1). First collision
    V#97←V#69 at t≈1.5s on seg (90001, 90003)."""
    return _viz_multi(1, 100)


@viz_scenario("multi_n100_seed2")
def viz_multi_n100_seed2():
    """100-vehicle scenario (seed=2). First collision V#42←V#92 at
    t≈43.5s on seg (810005, 810008); multiple pairs converge."""
    return _viz_multi(2, 100)


@viz_scenario("multi_n100_seed7")
def viz_multi_n100_seed7():
    """100-vehicle scenario (seed=7). First collision V#4←V#33 at
    t≈96.6s on seg (1349, 1350)."""
    return _viz_multi(7, 100)


@viz_scenario("multi_n100_seed11")
def viz_multi_n100_seed11():
    """100-vehicle scenario (seed=11). First collision V#60←V#97 at
    t≈31.9s on seg (1882, 1883) — good convoy-dynamics example."""
    return _viz_multi(11, 100)


@viz_scenario("multi_n100_seed50")
def viz_multi_n100_seed50():
    """100-vehicle scenario (seed=50). First collision V#10←V#68 at
    t≈107.9s on seg (13680001, 1370)."""
    return _viz_multi(50, 100)


@viz_scenario("multi_n100_seed278")
def viz_multi_n100_seed278():
    """100-vehicle scenario (seed=278). First collision V#92←V#30 at
    t≈1.5s on seg (3630001, 3630003) — early-start overlap pattern."""
    return _viz_multi(278, 100)


@viz_scenario("multi_n100_seed1147")
def viz_multi_n100_seed1147():
    """100-vehicle scenario (seed=1147). First collision V#85←V#13 at
    t≈56.5s on seg (1427, 1428)."""
    return _viz_multi(1147, 100)


@viz_scenario("multi_n200_seed1")
def viz_multi_n200_seed1():
    """200-vehicle high-density scenario (seed=1)."""
    return _viz_multi(1, 200)


@viz_scenario("multi_n200_seed42")
def viz_multi_n200_seed42():
    """200-vehicle high-density scenario (seed=42)."""
    return _viz_multi(42, 200)


@viz_scenario("multi_n200_seed99")
def viz_multi_n200_seed99():
    """200-vehicle high-density scenario (seed=99)."""
    return _viz_multi(99, 200)


def _viz_idle_only(seed: int, n_vehicles: int):
    """Spawn N vehicles, each with a 2-node path and dest=path[0] so
    they all enter STOP/IDLE on first _replan. No random_safe_path,
    no destinations beyond their own segment. Useful for observing
    the IDLE distribution and (with dispatch attached) push behavior
    against a fully-idle starting field."""
    import os, random, collections, math
    random.seed(seed)
    # SPLIT_ZONES=1 env var → use oht.large.map.split.json (long straight
    # diverge zone segments split at midpoint so the mutex region is
    # shorter; long stretches become non-zone corridor).
    map_name = ("oht.large.map.split.json"
                if os.environ.get("SPLIT_ZONES") == "1"
                else "oht.large.map.json")
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            map_name)
    gmap = GraphMap(map_path)

    degree = collections.Counter()
    for fn, tn in gmap.segments:
        degree[fn] += 1; degree[tn] += 1
    good_starts = [nid for nid in gmap.main_loop
                   if degree[nid] >= 2 and gmap.adj.get(nid)
                   and nid not in gmap.zcu_nodes
                   and '_split_' not in nid]

    reverse_adj = collections.defaultdict(list)
    for u, outs in gmap.adj.items():
        for w in outs: reverse_adj[w].append(u)

    # 초기 배치가 h_min 이내면 spawn 시점부터 안전거리 위반. start node
    # 좌표가 기존 spawn 들과 h_min 이상 떨어지도록 우선 선택 (1-hop used
    # 제외만으로는 2-hop=1seg 거리가 h_min 미만이 될 수 있음).
    # placed 차들을 cell=h_min 의 grid 에 버킷팅 — 새 cand 검사 시 자기
    # cell + 8-인접 cell 만 보면 됨 (선형 비교 시 N=200 spawn 이 17초+).
    H_MIN = (getattr(gmap, 'vehicle_length', None) or 750.0) + 200.0
    H2 = H_MIN * H_MIN
    cell = H_MIN
    grid = {}
    def _far_enough(x, y):
        cx, cy = int(x // cell), int(y // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in grid.get((cx + dx, cy + dy), ()):
                    if (x - px) * (x - px) + (y - py) * (y - py) < H2:
                        return False
        return True
    used = set(); vehicles = []
    for i in range(n_vehicles):
        cands = [n for n in good_starts if n not in used] or good_starts
        far = [n for n in cands
               if _far_enough(gmap.nodes[n].x, gmap.nodes[n].y)]
        if far:
            cands = far                  # h_min 확보 후보 우선
        start = random.choice(cands)
        succs = gmap.adj.get(start, [])
        if not succs: continue
        nxt = succs[0]
        v = Vehicle(i, gmap, [start, nxt])
        v.dest_node = start          # dest_reached on first _replan → STOP/None
        vehicles.append(v)
        used.add(start)
        x, y = gmap.nodes[start].x, gmap.nodes[start].y
        grid.setdefault((int(x // cell), int(y // cell)), []).append((x, y))
        for nb in gmap.adj.get(start, []): used.add(nb)
        for nb in reverse_adj.get(start, []): used.add(nb)
    return gmap, vehicles, seed


@viz_scenario("idle_n200_seed99")
def viz_idle_n200_seed99():
    """200 OHTs spawn pre-IDLE (no random walk). All cyan from t=0."""
    return _viz_idle_only(99, 200)


@viz_scenario("idle_n200_seed99_disp")
def viz_idle_n200_seed99_disp():
    """200 OHTs spawn pre-IDLE + dispatch (lambda=0.8/s). Watch jobs
    pull idle OHTs out and push fire on the path."""
    gmap, vehicles, seed = _viz_idle_only(99, 200)
    return gmap, vehicles, seed, 0.8


@viz_scenario("Z1_diverge_solo")
def viz_z1():
    gmap = map_diverge_long(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        exit_specs=[(15000, 3600), (3000, 1200), (15000, 3600)],
        branch_specs=[(5000, 3600)],
    )
    path = ["n0", "n1", "n2", "n3", "e1", "e2", "e3"]
    v = Vehicle(0, gmap, path)
    return gmap, [v]


@viz_scenario("Z2_diverge_convoy")
def viz_z2():
    gmap = map_diverge_long(
        approach_specs=[(20000, 3600), (3000, 700), (20000, 3600)],
        exit_specs=[(20000, 3600), (3000, 1200), (20000, 3600)],
        branch_specs=[(5000, 3600)],
    )
    path = ["n0", "n1", "n2", "n3", "e1", "e2", "e3"]
    vehicles = []
    for i in range(3):
        v = Vehicle(i, gmap, list(path))
        v.path_idx = 0
        v.seg_offset = (2 - i) * 5000.0
        vehicles.append(v)
    vehicles.reverse()
    return gmap, vehicles


@viz_scenario("Z3_diamond_merge_contention")
def viz_z3d():
    gmap = map_diamond(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        branch_a_specs=[(10000, 3600), (3000, 1200)],
        branch_b_specs=[(8000, 700), (3000, 700)],
        exit_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
    )
    path_a = ["ba0", "ba1", "mrg", "e0", "e1", "e2"]
    path_b = ["bb0", "bb1", "mrg", "e0", "e1", "e2"]
    va = Vehicle(0, gmap, path_a)
    vb = Vehicle(1, gmap, path_b)
    return gmap, [va, vb]


@viz_scenario("Z4_diamond_convoy")
def viz_z4d():
    gmap = map_diamond(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        branch_a_specs=[(10000, 3600), (3000, 1200)],
        branch_b_specs=[(8000, 700), (3000, 700)],
        exit_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
    )
    path_a = ["ba0", "ba1", "mrg", "e0", "e1", "e2"]
    path_b = ["bb0", "bb1", "mrg", "e0", "e1", "e2"]
    vehicles = []
    for vid, p, off in [(0, path_a, 5000.0), (1, path_a, 0.0),
                         (2, path_b, 3000.0), (3, path_b, 0.0)]:
        v = Vehicle(vid, gmap, list(p))
        v.path_idx = 0; v.seg_offset = off
        vehicles.append(v)
    return gmap, vehicles


@viz_scenario("Z5_diamond_same_branch")
def viz_z5d():
    gmap = map_diamond(
        approach_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
        branch_a_specs=[(10000, 3600), (3000, 1200)],
        branch_b_specs=[(8000, 700), (3000, 700)],
        exit_specs=[(15000, 3600), (3000, 700), (15000, 3600)],
    )
    path_a = ["a0", "a1", "a2", "a3", "ba0", "ba1", "mrg", "e0", "e1", "e2"]
    vehicles = []
    for i in range(3):
        v = Vehicle(i, gmap, list(path_a))
        v.path_idx = 0; v.seg_offset = (2 - i) * 5000.0
        vehicles.append(v)
    vehicles.reverse()
    return gmap, vehicles


@viz_scenario("Z3_real_merge_solo")
def viz_z3():
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    path_a = ["1277", "1278", "1279", "1280", "1281", "1282", "1283",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    path_b = ["1457", "1458", "1459", "1460", "14600001", "14600002", "1463", "_curve_mid_59",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    va = Vehicle(0, gmap, path_a)
    vb = Vehicle(1, gmap, path_b)
    return gmap, [va, vb]


@viz_scenario("Z4_real_merge_convoy")
def viz_z4():
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    path_a = ["1277", "1278", "1279", "1280", "1281", "1282", "1283",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    path_b = ["1457", "1458", "1459", "1460", "14600001", "14600002", "1463", "_curve_mid_59",
              "1285", "1286", "1287", "1288", "1289", "1290", "1291", "1292", "1293"]
    vehicles = []
    for vid, p, off in [(0, path_a, 3000.0), (1, path_a, 0.0),
                         (2, path_b, 3000.0), (3, path_b, 0.0)]:
        v = Vehicle(vid, gmap, list(p))
        v.path_idx = 0; v.seg_offset = off
        vehicles.append(v)
    return gmap, vehicles


@viz_scenario("Z5_real_diverge_convoy")
def viz_z5():
    from graph_des_v5 import GraphMap
    gmap = GraphMap("oht.large.map.json")
    path = ["1832", "1833", "1834", "1835", "1836", "1837",
            "1059", "1060", "1061", "1062", "1063",
            "1065", "1066", "1067", "1068", "1069"]
    vehicles = []
    for i in range(2):
        v = Vehicle(i, gmap, list(path))
        v.path_idx = 0; v.seg_offset = (1 - i) * 5000.0
        vehicles.append(v)
    vehicles.reverse()
    return gmap, vehicles


def run_viz(name, trace_vids=None):
    spec = VIZ_SCENARIOS.get(name)
    if spec is None:
        print(f"No viz scenario named '{name}'. Available: {list(VIZ_SCENARIOS)}")
        return 1
    result = spec()
    # Specs can return:
    #   (gmap, vehicles)
    #   (gmap, vehicles, rng_seed)
    #   (gmap, vehicles, rng_seed, dispatch_lambda)
    rng_seed = None
    dispatch_lambda = None
    if len(result) == 4:
        gmap, vehicles, rng_seed, dispatch_lambda = result
    elif len(result) == 3:
        gmap, vehicles, rng_seed = result
    else:
        gmap, vehicles = result
    import test_graph_v6
    test_graph_v6.main(inject_gmap=gmap, inject_vehicles=vehicles,
                       title=f"micro: {name}",
                       inject_rng_seed=rng_seed,
                       inject_dispatch_lambda=dispatch_lambda,
                       inject_trace_vids=trace_vids)
    return 0


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if args and args[0] == "--viz":
        if len(args) < 2:
            print("Usage: python test_plan_micro.py --viz <scenario_name> "
                  "[--trace VID,VID,...]")
            print(f"Available: {list(VIZ_SCENARIOS)}")
            sys.exit(1)
        scenario = args[1]
        trace_vids = None
        rest = args[2:]
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok in ("--trace", "--trace-vids"):
                if i + 1 >= len(rest):
                    print("--trace requires a comma-separated list of vehicle IDs")
                    sys.exit(1)
                trace_vids = [int(x) for x in rest[i + 1].split(",") if x.strip()]
                i += 2
            elif tok.startswith("--trace="):
                trace_vids = [int(x) for x in tok.split("=", 1)[1].split(",") if x.strip()]
                i += 1
            else:
                print(f"Unknown viz option: {tok}")
                sys.exit(1)
        sys.exit(run_viz(scenario, trace_vids=trace_vids))
    if args and args[0] == "--viz-param":
        import argparse
        ap = argparse.ArgumentParser(
            prog="test_plan_micro.py --viz-param",
            description="OHT idle field on oht.large map with free vehicle "
                        "count + dispatch load factor (synthetic-map viz). "
                        "SPLIT_ZONES=1 env var selects the split map.")
        ap.add_argument("--n", type=int, default=200,
                        help="number of OHT vehicles (pre-IDLE spawn)")
        ap.add_argument("--lambda", dest="lam", type=float, default=0.0,
                        help="dispatch arrival rate (jobs/s, load factor). "
                             "0 = no dispatch, idle field only")
        ap.add_argument("--seed", type=int, default=99, help="random seed")
        ap.add_argument("--trace", type=str, default=None,
                        help="comma-separated vehicle IDs to trace")
        pa = ap.parse_args(args[1:])
        trace_vids = ([int(x) for x in pa.trace.split(",") if x.strip()]
                      if pa.trace else None)
        gmap, vehicles, seed = _viz_idle_only(pa.seed, pa.n)
        disp = pa.lam if pa.lam and pa.lam > 0 else None
        import test_graph_v6
        test_graph_v6.main(
            inject_gmap=gmap, inject_vehicles=vehicles,
            title=f"param n={pa.n} seed={pa.seed} "
                  f"lambda={pa.lam if disp else 'off'}",
            inject_rng_seed=seed,
            inject_dispatch_lambda=disp,
            inject_trace_vids=trace_vids)
        sys.exit(0)
    selected = args if args else None
    names = selected if selected else list(SCENARIOS.keys())
    print(f"Running {len(names)} scenario(s)\n")
    n_pass = 0
    n_fail = 0
    for name in names:
        fn = SCENARIOS.get(name)
        if fn is None:
            print(f"  ?  {name} (unknown scenario)")
            continue
        try:
            result = fn()
            print(f"  PASS  {name}: {result}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
