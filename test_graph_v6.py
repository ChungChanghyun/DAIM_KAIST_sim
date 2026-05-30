"""
test_graph_v6.py — Graph DES v6 (Pure DES) visualizer for oht.large.map.json.

Controls:
  RClick-drag / Arrow keys : Pan
  Mouse wheel              : Zoom
  Left-click               : Select vehicle
  Space                    : Pause / Resume
  +/-                      : Speed up / slow down sim
  1~9                      : Set sim speed multiplier
  R                        : Reset view
  N                        : Toggle node dots
  Z                        : Toggle ZCU markers
  O                        : Toggle port markers (Buffer / Station)
  L                        : Toggle leader arrows (all)
  V                        : Toggle violation overlay
  T                        : Toggle destination markers (orange=push, green=dest)
  X                        : Toggle committed trajectory (MOVING 차량 전부 선 + 끝점 ✕, 선택 차량 굵게)
  D                        : Dump stopped/waiting vehicles + leader/lock chain
  F (hold)                 : Follow selected vehicle
  ESC                      : Quit
"""

import sys, os, math, random, collections, re, time
import pygame

from graph_des_v5 import (
    GraphMap, random_safe_path, _interp_path,
)
from graph_des_v6 import (
    GraphDESv6, Vehicle,
    IDLE, ACCEL, CRUISE, DECEL, STOP, LOADING,
)

# ── Colors ────────────────────────────────────────────────────────────────────

BG        = (18, 18, 24)
TRACK_C   = (50, 100, 170)
NODE_C    = (60, 60, 80)
TEXT      = (200, 200, 200)
DIM       = (120, 120, 150)
ZCU_MERGE_C  = (255, 60, 60)
ZCU_DIVERGE_C = (255, 180, 60)
ZCU_CURVE_C  = (255, 100, 50)
ZCU_STRAIGHT_C = (80, 220, 255)

# Lock status colors for ZCU markers
ZCU_FREE_C   = (60, 200, 60)
ZCU_LOCKED_C = (255, 60, 60)
ZCU_WAIT_C   = (255, 200, 60)   # has waiters

# Ports (mcs_unified style: Buffer = shelf, Station = process tool)
PORT_BUFFER_C  = (255, 140, 40)
PORT_STATION_C = (60, 220, 220)

# Violation / stuck
VIOL_C  = (255, 0, 80)
STUCK_C = (255, 0, 255)

MOVING_C   = (60, 200, 60)      # 작업(job) 수행을 위한 주행 — green
IDLE_MOV_C = (80, 140, 255)     # job 없는 OHT의 주행 (push로 비키는 경우 포함) — blue

STATE_C = {
    IDLE:    (100, 100, 100),
    ACCEL:   MOVING_C,
    CRUISE:  MOVING_C,
    DECEL:   MOVING_C,
    STOP:    (255, 60, 60),
    LOADING: (200, 60, 200),
}

# Sub-classification of STOP by stop_reason: distinguishes idle (pushable)
# vs blocked (waiting on leader / ZCU).
STOP_REASON_C = {
    None:     (90, 200, 230),   # IDLE_FREE: cyan — free, pushable
    'dest':   (60, 220, 220),   # IDLE_DEST: cyan-green — at destination
    'leader': (255, 90,  60),   # BLOCK_LEADER: orange-red
    'zcu':    (255, 60, 180),   # BLOCK_ZCU: magenta-red
}

STATE_LABEL = {
    IDLE: "IDLE", ACCEL: "ACCEL", CRUISE: "CRUISE",
    DECEL: "DECEL", STOP: "STOP", LOADING: "LOAD",
}

VEHICLE_COLORS = [
    (255, 80,  80),
    (80,  200, 255),
    (255, 200, 60),
    (100, 255, 130),
    (255, 130, 200),
    (200, 130, 255),
    (255, 160, 80),
    (130, 255, 220),
    (180, 180, 255),
    (255, 255, 150),
]

N_VEHICLES = 150

def _draw_panel(screen, x, y, w, h, alpha=200):
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((20, 20, 35, alpha))
    screen.blit(panel, (x, y))


def main(inject_gmap=None, inject_vehicles=None, title=None,
         inject_rng_seed=None, inject_dispatch_lambda=None,
         inject_trace_vids=None):
    """Run the visualizer.

    Default mode: load oht.large.map.json and spawn N_VEHICLES random OHTs.
    Injection mode: caller provides gmap + already-constructed Vehicle list,
    main() skips the random setup and visualizes the provided scenario.
    Used by test_plan_micro.py for micro-scenario visualization.

    inject_rng_seed: optional seed for DES path-extension rng (for
    reproducible deadlock viz). None = legacy global-random behavior.
    """
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "oht.large.map.json")
    if inject_gmap is not None:
        gmap = inject_gmap
        des = GraphDESv6(gmap, rng_seed=inject_rng_seed)
        if inject_trace_vids:
            des.DEBUG_TRACE = True
            des.DEBUG_VID = -1            # don't fall back to single-vid mode
            des.DEBUG_VID_SET = set(inject_trace_vids)
            print(f"[viz] DEBUG trace enabled for V#{sorted(des.DEBUG_VID_SET)}")
        vehicles = list(inject_vehicles or [])
        for v in vehicles:
            v.color = VEHICLE_COLORS[v.id % len(VEHICLE_COLORS)]
            des.add_vehicle(v)
        des.start_all()
        print(f"[viz] Injected {len(vehicles)} vehicles, "
              f"{len(gmap.nodes)} nodes, {len(gmap.segments)} segs")
        if inject_dispatch_lambda is not None:
            import dispatch as dp
            ports = dp.load_ports(map_path)
            mgr = dp.JobManager(des, gmap, ports,
                                lambda_rate=inject_dispatch_lambda,
                                load_dwell=5.0, unload_dwell=5.0,
                                rng_seed=inject_rng_seed)
            mgr.start(0.0)
            print(f"[viz] Dispatch attached: lambda={inject_dispatch_lambda}/s, "
                  f"{len(mgr.ports)} ports")
    else:
        random.seed(42)

        # ── Load map ──────────────────────────────────────────────────────────
        gmap = GraphMap(map_path)
        print(f"Loaded: {len(gmap.nodes)} nodes, {len(gmap.segments)} segments, "
              f"main loop: {len(gmap.main_loop)} nodes")

        degree = collections.Counter()
        for fn, tn in gmap.segments:
            degree[fn] += 1
            degree[tn] += 1
        good_starts = [nid for nid in gmap.main_loop
                       if degree[nid] >= 2 and gmap.adj.get(nid)]
        if not good_starts:
            good_starts = [nid for nid in gmap.main_loop if gmap.adj.get(nid)]
        print(f"Good start nodes: {len(good_starts)}")

        # ── Create DES engine ─────────────────────────────────────────────────
        des = GraphDESv6(gmap)

        used_nodes = set()
        vehicles = []

        def spawn_vehicles(count):
            nonlocal used_nodes
            for i in range(count):
                vid = len(vehicles)
                candidates = [n for n in good_starts if n not in used_nodes]
                if not candidates:
                    candidates = good_starts
                start = random.choice(candidates)
                path = random_safe_path(gmap, start, length=200)
                color = VEHICLE_COLORS[vid % len(VEHICLE_COLORS)]
                v = Vehicle(vid, gmap, path, color)
                vehicles.append(v)
                des.add_vehicle(v)
                used_nodes.add(start)
                for nb in gmap.adj.get(start, []):
                    used_nodes.add(nb)

        spawn_vehicles(N_VEHICLES)
        des.start_all()
        print(f"Spawned {N_VEHICLES} vehicles (pure DES v6)")

    # ── Pygame init ───────────────────────────────────────────────────────
    pygame.init()
    disp = pygame.display.Info()
    SW = min(1600, disp.current_w - 100)
    SH = min(900, disp.current_h - 100)
    screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)
    pygame.display.set_caption(title or "Graph DES v6 — Pure DES + ZCU wait-queue")
    clock = pygame.time.Clock()
    font_s = pygame.font.SysFont("consolas", 12)

    # ── Camera ────────────────────────────────────────────────────────────
    bx0, by0, bx1, by1 = gmap.bbox
    world_cx = (bx0 + bx1) / 2
    world_cy = (by0 + by1) / 2
    world_w = bx1 - bx0
    world_h = by1 - by0
    scale = min(SW / (world_w * 1.1), SH / (world_h * 1.1))
    cam_x, cam_y = world_cx, world_cy
    init_scale, init_cx, init_cy = scale, cam_x, cam_y

    node_xy = {nid: (n.x, n.y) for nid, n in gmap.nodes.items()}

    # ── Ports (Buffer / Station, mcs_unified style) ─────────────────────────
    # Each entry: (kind, world_x, world_y). World pos = node + position offset.
    ports_world: list = []
    try:
        import json as _json
        with open(map_path) as _pf:
            _pdata = _json.load(_pf)
        for _p in _pdata.get('ports', []):
            _nid = _p.get('nodeId')
            if _nid not in node_xy:
                continue
            _kind = _p.get('kind', 'Buffer')
            _pos = _p.get('position', {}) or {}
            _ox = float(_pos.get('x', 0.0))
            _oy = float(_pos.get('y', 0.0))
            _nx, _ny = node_xy[_nid]
            ports_world.append((_kind, _nx + _ox, _ny + _oy))
        _n_buf = sum(1 for k, *_ in ports_world if k == 'Buffer')
        _n_sta = sum(1 for k, *_ in ports_world if k == 'Station')
        print(f"[viz] Ports: {_n_buf} Buffer, {_n_sta} Station")
    except Exception as _e:
        print(f"[viz] Port load failed: {_e}")

    # ── State ─────────────────────────────────────────────────────────────
    sim_time = 0.0
    sim_speed = 3.0
    paused = True

    # ── VIZ_BENCH: headless-vs-viz overhead 측정용 임시 hook ──────────
    # VIZ_BENCH=1 → 자동 시작 + 60fps cap 무시 + 고정 sim step, sim_time
    # 이 VIZ_BENCH_TEND 도달 시 wall 출력 후 종료. SDL dummy 와 함께 쓰면
    # background 에서 render 포함 throughput 측정 가능.
    _bench = os.environ.get("VIZ_BENCH") == "1"
    _bench_tend = float(os.environ.get("VIZ_BENCH_TEND", "86400"))
    _bench_t0 = None
    if _bench:
        paused = False
    show_nodes = True
    show_zcu = True
    show_ports = True
    show_leaders = False
    show_violations = False   # ZCU violation overlay/panel 기본 off (무해 진단) — V 키로 toggle
    show_dests = True   # destination markers (T toggle)
    show_commit = False  # committed trajectory 끝점 marker (X toggle)
    selected = None
    dragging = False
    drag_start = None
    hovered_zcu_node = None
    total_viol = 0
    min_gap = float('inf')
    gap_hist = [0] * 20

    # Violation flash tracking
    violation_vehicles = {}    # vid -> expiry sim_time
    prev_viol_count = 0

    # Collision tracking: rear-end detection triggers auto-pause + persistent
    # red highlight on the involved vehicles so users can rewind & inspect.
    collision_vehicles = set()   # vids currently flagged as collided
    collision_pairs = set()      # frozenset({a_id, b_id}) dedupe
    collision_log = []           # [(t, a_id, b_id, gap, seg)]
    collision_autopause = True

    def w2s(wx, wy):
        return (int((wx - cam_x) * scale + SW / 2),
                int(-(wy - cam_y) * scale + SH / 2))

    def s2w(sx, sy):
        return ((sx - SW / 2) / scale + cam_x,
                -(sy - SH / 2) / scale + cam_y)

    def _vehicle_locks():
        """vid -> list of lock_id."""
        vl = collections.defaultdict(list)
        for lock_id, holder in des._zone_lock.items():
            if holder is not None:
                vl[holder.id].append(lock_id)
        return vl

    # ── Main loop ─────────────────────────────────────────────────────────
    running = True
    while running:
        if _bench:
            if _bench_t0 is None:
                _bench_t0 = time.time()
            dt_real = 0.0   # cap 무시 — 최대 속도
        else:
            dt_real = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    cam_x, cam_y, scale = init_cx, init_cy, init_scale
                    selected = None
                elif event.key == pygame.K_n:
                    show_nodes = not show_nodes
                elif event.key == pygame.K_z:
                    show_zcu = not show_zcu
                elif event.key == pygame.K_o:
                    show_ports = not show_ports
                elif event.key == pygame.K_l:
                    show_leaders = not show_leaders
                elif event.key == pygame.K_v:
                    show_violations = not show_violations
                elif event.key == pygame.K_t:
                    show_dests = not show_dests
                elif event.key == pygame.K_x:
                    show_commit = not show_commit
                    print(f"Commit-end marker: {show_commit}")
                elif event.key == pygame.K_c:
                    # Toggle collision auto-pause
                    collision_autopause = not collision_autopause
                    print(f"Collision auto-pause: {collision_autopause}")
                elif event.key == pygame.K_p:
                    print(f"\n=== Speed violation log ({des.speed_violation_count}건) ===")
                    seg_counts = {}
                    for entry in des.speed_violation_log:
                        t_log, vid, seg, vel, limit, excess = entry[:6]
                        lookahead_v = entry[6] if len(entry) > 6 else -1
                        path_rem    = entry[7] if len(entry) > 7 else -1
                        seg_counts[seg] = seg_counts.get(seg, 0) + 1
                        print(f"  t={t_log:.2f} V#{vid} {seg[0]}->{seg[1]} "
                              f"vel={vel:.0f} limit={limit:.0f} excess={excess:.0f}mm/s "
                              f"lookahead={lookahead_v:.0f} path_rem={path_rem}")
                    print(f"  세그먼트별 집계: {sorted(seg_counts.items(), key=lambda x:-x[1])[:10]}")
                    if not des.speed_violation_log:
                        print("  (없음)")
                elif event.key == pygame.K_d:
                    # Dump stopped/waiting vehicles + lock & leader chain
                    print(f"\n=== Stop/wait dump @ t={sim_time:.2f}s ===")
                    stopped = [vv for vv in vehicles if vv.vel < 1.0]
                    print(f"  stopped count: {len(stopped)}/{len(vehicles)}")
                    # Build holder reverse map
                    holder_locks = collections.defaultdict(list)
                    for lid, h in des._zone_lock.items():
                        if h is not None:
                            holder_locks[h.id].append(lid)
                    waiter_locks = collections.defaultdict(list)
                    for lid, ws in des._zone_waiters.items():
                        for w in ws:
                            waiter_locks[w.id].append(lid)
                    for vv in stopped:
                        ld = vv.leader.id if vv.leader else None
                        ld_state = ""
                        if vv.leader:
                            ld_state = (f" leader_state={vv.leader.state}"
                                        f" leader_vel={vv.leader.vel:.0f}"
                                        f" leader_stop_reason={vv.leader.stop_reason}"
                                        f" leader_wait={vv.leader.waiting_at_zcu}")
                        print(f"  V#{vv.id} state={vv.state} stop_reason={vv.stop_reason} "
                              f"wait={vv.waiting_at_zcu} pidx={vv.path_idx} "
                              f"seg={vv.seg_from}->{vv.seg_to}")
                        print(f"      leader={ld}{ld_state}")
                        print(f"      holds={holder_locks.get(vv.id, [])}")
                        print(f"      waits_on={waiter_locks.get(vv.id, [])}")
                    # Detect leader chain cycles among stopped
                    print(f"  --- leader chains ---")
                    for vv in stopped:
                        chain = [vv.id]
                        cur = vv.leader
                        seen = {vv.id}
                        while cur is not None and cur.id not in seen:
                            chain.append(cur.id)
                            seen.add(cur.id)
                            cur = cur.leader
                        if cur is not None:
                            chain.append(f"CYCLE→{cur.id}")
                        print(f"    {' → '.join(str(x) for x in chain)}")
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    sim_speed = max(0.5, sim_speed / 1.5)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    sim_speed = min(50.0, sim_speed * 1.5)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    sim_speed = float(event.key - pygame.K_0)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    dragging = True
                    drag_start = event.pos
                elif event.button == 1:
                    mx, my = event.pos
                    best, bd = None, 25
                    for v in vehicles:
                        sx, sy = w2s(v.x, v.y)
                        d = math.hypot(sx - mx, sy - my)
                        if d < bd:
                            best, bd = v, d
                    selected = best
                elif event.button == 4:
                    mx, my = event.pos
                    wx, wy = s2w(mx, my)
                    scale *= 1.15
                    cam_x = wx - (mx - SW / 2) / scale
                    cam_y = wy + (my - SH / 2) / scale
                elif event.button == 5:
                    mx, my = event.pos
                    wx, wy = s2w(mx, my)
                    scale /= 1.15
                    cam_x = wx - (mx - SW / 2) / scale
                    cam_y = wy + (my - SH / 2) / scale

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    cam_x -= dx / scale
                    cam_y += dy / scale
                    drag_start = event.pos
                elif show_zcu:
                    mx, my = event.pos
                    hovered_zcu_node = None
                    best_d = 20
                    all_zcu = gmap.zcu_nodes | des._boundary_nodes
                    for nid in all_zcu:
                        if nid not in node_xy:
                            continue
                        nx, ny = node_xy[nid]
                        sx, sy = w2s(nx, ny)
                        d = math.hypot(sx - mx, sy - my)
                        if d < best_d:
                            best_d = d
                            hovered_zcu_node = nid

            elif event.type == pygame.VIDEORESIZE:
                SW, SH = event.w, event.h
                screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)

        # Arrow key pan
        keys = pygame.key.get_pressed()
        pan = 500 / scale
        if keys[pygame.K_LEFT]:  cam_x -= pan
        if keys[pygame.K_RIGHT]: cam_x += pan
        if keys[pygame.K_UP]:    cam_y += pan
        if keys[pygame.K_DOWN]:  cam_y -= pan

        if selected and keys[pygame.K_f]:
            cam_x, cam_y = selected.x, selected.y

        # ── Sim step ──────────────────────────────────────────────────────
        if not paused:
            if _bench:
                dt_sim = 0.5   # 고정 step (probe 간격과 무관, 측정 일관성)
            else:
                dt_sim = min(dt_real * sim_speed, 0.3)
            sim_time += dt_sim

            des.run_until(sim_time)
            des.query_positions(sim_time)
            if _bench and sim_time >= _bench_tend:
                _wall = time.time() - _bench_t0
                _spd = sim_time / _wall if _wall > 0 else 0
                print(f"[VIZ_BENCH] N={len(vehicles)} sim={sim_time:.0f}s "
                      f"wall={_wall:.1f}s speedup={_spd:.1f}x "
                      f"events={des.event_count}", flush=True)
                running = False
                continue

            # Gap stats
            for v in vehicles:
                g = v.gap_to_leader
                if g < float('inf') and g < 200000:
                    b = min(int(g / 100), 19)
                    if b >= 0:
                        gap_hist[b] += 1
                    min_gap = min(min_gap, g)
                    # violation = safety distance(h_min) 침범. 물리 겹침
                    # (gap<length)은 아래 collision detection 이 별도로 잡음.
                    # 1mm eps: plan 이 h_min 을 정확히 유지할 때 runtime
                    # gap 이 949.x 로 미세하게 부족한 경계 케이스 제외.
                    if g < v.h_min - 1.0:
                        total_viol += 1

            # Physical rear-end collision detection (gap < vehicle length).
            # Group vehicles by real-time segment, sort by offset, and
            # check consecutive pairs. Same logic as probe_multi.py.
            rt_pos = {}
            for vv in vehicles:
                d = vv._dist_traveled(sim_time - vv.t_ref)
                off = vv.seg_offset + d
                pidx = vv.path_idx
                while pidx < len(vv.path) - 1:
                    sl = vv._seg_lengths[pidx] if pidx < len(vv._seg_lengths) else 0
                    if sl <= 0 or off < sl - 0.01:
                        break
                    off -= sl
                    pidx += 1
                rt_pos[vv.id] = (pidx, max(0.0, off))
            by_seg = collections.defaultdict(list)
            for vv in vehicles:
                pidx, off = rt_pos[vv.id]
                if pidx < len(vv.path) - 1:
                    by_seg[(vv.path[pidx], vv.path[pidx + 1])].append((vv, off))
            new_collision = False
            for seg_key, occs in by_seg.items():
                if len(occs) < 2:
                    continue
                occs.sort(key=lambda x: x[1])
                for i in range(len(occs) - 1):
                    a, a_off = occs[i]
                    b, b_off = occs[i + 1]
                    gap = b_off - a_off
                    if gap < a.length:
                        pair = frozenset({a.id, b.id})
                        if pair not in collision_pairs:
                            collision_pairs.add(pair)
                            collision_log.append((sim_time, a.id, b.id, gap, seg_key))
                            collision_vehicles.add(a.id)
                            collision_vehicles.add(b.id)
                            new_collision = True
            if new_collision and collision_autopause:
                paused = True

            # Detect new ZCU violations -> flash
            new_vc = des.zcu_violation_count
            if new_vc > prev_viol_count:
                for entry in des.zcu_violation_log[prev_viol_count:new_vc]:
                    detail = entry[3] if len(entry) > 3 else ""
                    for m in re.finditer(r'V#(\d+)', detail):
                        violation_vehicles[int(m.group(1))] = sim_time + 3.0
                prev_viol_count = new_vc

            # Expire flashes
            expired = [vid for vid, t in violation_vehicles.items() if sim_time > t]
            for vid in expired:
                del violation_vehicles[vid]

        # Per-frame lookups
        v_locks = _vehicle_locks()

        stuck_holders = set()
        for lid, holder in des._zone_lock.items():
            if holder and holder.state == STOP:
                stuck_holders.add(holder.id)

        # ── Draw ──────────────────────────────────────────────────────────
        screen.fill(BG)

        # Visible world rect for culling
        vl, vt = s2w(0, 0)
        vr, vb = s2w(SW, SH)
        margin = 5000
        vl -= margin; vr += margin
        vt_y = min(vt, vb) - margin
        vb_y = max(vt, vb) + margin

        # Segments
        lw = max(1, min(3, int(scale * 80)))
        for (fn, tn), seg in gmap.segments.items():
            if not seg.path_points:
                continue
            p0 = seg.path_points[0]
            pn = seg.path_points[-1]
            mid_x = (p0[0] + pn[0]) / 2
            mid_y = (p0[1] + pn[1]) / 2
            if mid_x < vl or mid_x > vr or mid_y < vt_y or mid_y > vb_y:
                continue
            pts_s = [w2s(p[0], p[1]) for p in seg.path_points]
            if len(pts_s) >= 2:
                pygame.draw.lines(screen, TRACK_C, False, pts_s, lw)

        # Nodes
        if show_nodes:
            nr = max(2, int(scale * 60))
            for nid, (nx, ny) in node_xy.items():
                if nx < vl or nx > vr or ny < vt_y or ny > vb_y:
                    continue
                # TEMP: split-zone midpoints (from split_long_zones.py) in red
                col = (220, 30, 30) if '_split_' in nid else NODE_C
                pygame.draw.circle(screen, col, w2s(nx, ny), nr)

        # Ports — small squares offset from node position
        if show_ports and ports_world:
            buf_r = max(2, int(scale * 80))
            sta_r = max(3, int(scale * 160))
            for _kind, _wx, _wy in ports_world:
                if _wx < vl or _wx > vr or _wy < vt_y or _wy > vb_y:
                    continue
                _sx, _sy = w2s(_wx, _wy)
                if _kind == 'Station':
                    pygame.draw.rect(screen, PORT_STATION_C,
                                     (_sx - sta_r, _sy - sta_r,
                                      sta_r * 2, sta_r * 2))
                else:  # Buffer
                    pygame.draw.rect(screen, PORT_BUFFER_C,
                                     (_sx - buf_r, _sy - buf_r,
                                      buf_r * 2, buf_r * 2))

        # ── ZCU markers (lock-status coloring) ───────────────────────────
        if show_zcu:
            mr = max(3, int(scale * 120))

            # Highlight ALL segments of locked diverge/merge zones (so users
            # see that holding the lock blocks every direction, not just the
            # direction the holder is going).
            for zone in gmap.zcu_zones:
                lock_id = f"{zone.node_id}_{zone.kind}"
                holder = des._zone_lock.get(lock_id)
                if holder is None:
                    continue
                for seg_key in zone.all_segs():
                    seg = gmap.segments.get(seg_key)
                    if not seg or not seg.path_points:
                        continue
                    pts = [w2s(px, py) for px, py in seg.path_points]
                    if len(pts) >= 2:
                        pygame.draw.lines(screen, ZCU_LOCKED_C, False, pts,
                                           max(2, int(scale * 30)))

            # Merge nodes  --  circle, colored by lock status
            for nid in gmap.merge_nodes:
                nx, ny = node_xy[nid]
                if nx < vl or nx > vr or ny < vt_y or ny > vb_y:
                    continue
                lock_id = f"{nid}_merge"
                holder = des._zone_lock.get(lock_id)
                waiters = des._zone_waiters.get(lock_id, [])
                c = ZCU_WAIT_C if (holder and waiters) else (ZCU_LOCKED_C if holder else ZCU_FREE_C)
                pygame.draw.circle(screen, c, w2s(nx, ny), mr)
                if scale > 0.02:
                    sx, sy = w2s(nx, ny)
                    screen.blit(font_s.render("M", True, (255,255,255)), (sx-3, sy-6))

            # Diverge nodes  --  diamond, colored by lock status
            for nid in gmap.diverge_nodes:
                nx, ny = node_xy[nid]
                if nx < vl or nx > vr or ny < vt_y or ny > vb_y:
                    continue
                lock_id = f"{nid}_diverge"
                holder = des._zone_lock.get(lock_id)
                waiters = des._zone_waiters.get(lock_id, [])
                c = ZCU_WAIT_C if (holder and waiters) else (ZCU_LOCKED_C if holder else ZCU_FREE_C)
                sx, sy = w2s(nx, ny)
                dm = [(sx, sy-mr), (sx+mr, sy), (sx, sy+mr), (sx-mr, sy)]
                pygame.draw.polygon(screen, c, dm)
                if scale > 0.02:
                    screen.blit(font_s.render("D", True, (255,255,255)), (sx-3, sy-6))

            # Boundary nodes (small diamond)
            br = max(2, int(scale * 80))
            for nid in des._boundary_nodes:
                if nid in gmap.merge_nodes or nid in gmap.diverge_nodes:
                    continue
                if nid not in node_xy:
                    continue
                nx, ny = node_xy[nid]
                if nx < vl or nx > vr or ny < vt_y or ny > vb_y:
                    continue
                bsx, bsy = w2s(nx, ny)
                is_locked = any(des._zone_lock.get(lid) is not None
                                for _, lid in des._boundary_to_zones.get(nid, []))
                bc = (255, 255, 60) if is_locked else (120, 120, 60)
                dm = [(bsx, bsy-br), (bsx+br, bsy), (bsx, bsy+br), (bsx-br, bsy)]
                if is_locked:
                    pygame.draw.polygon(screen, bc, dm)
                else:
                    pygame.draw.polygon(screen, bc, dm, 1)

            # ── Hovered ZCU detail ────────────────────────────────────
            if hovered_zcu_node:
                zcu_lw = max(3, min(6, int(scale * 200)))
                hn = node_xy[hovered_zcu_node]
                hsx, hsy = w2s(*hn)

                r = max(6, int(scale * 300))
                is_merge = hovered_zcu_node in gmap.merge_nodes
                is_diverge = hovered_zcu_node in gmap.diverge_nodes
                is_boundary = hovered_zcu_node in des._boundary_nodes

                if is_merge:
                    diamond_c = ZCU_MERGE_C
                elif is_diverge:
                    diamond_c = ZCU_DIVERGE_C
                else:
                    diamond_c = (200, 200, 60)
                diamond = [(hsx, hsy-r), (hsx+r, hsy), (hsx, hsy+r), (hsx-r, hsy)]
                pygame.draw.polygon(screen, diamond_c, diamond)
                pygame.draw.polygon(screen, (255,255,255), diamond, 2)

                if is_merge:
                    for pred in gmap.adj_rev.get(hovered_zcu_node, []):
                        seg_key = (pred, hovered_zcu_node)
                        seg = gmap.segments.get(seg_key)
                        if seg and seg.path_points:
                            pts_s = [w2s(p[0], p[1]) for p in seg.path_points]
                            if len(pts_s) >= 2:
                                is_curve = seg_key in gmap.merge_curve_entries
                                c = ZCU_CURVE_C if is_curve else ZCU_STRAIGHT_C
                                pygame.draw.lines(screen, c, False, pts_s, zcu_lw)
                        if pred in node_xy:
                            pygame.draw.circle(screen, (255,200,60), w2s(*node_xy[pred]), mr+2, 2)
                if is_diverge:
                    for succ in gmap.adj.get(hovered_zcu_node, []):
                        seg_key = (hovered_zcu_node, succ)
                        seg = gmap.segments.get(seg_key)
                        if seg and seg.path_points:
                            pts_s = [w2s(p[0], p[1]) for p in seg.path_points]
                            if len(pts_s) >= 2:
                                is_curve = seg_key in gmap.diverge_curve_exits
                                c = ZCU_CURVE_C if is_curve else ZCU_STRAIGHT_C
                                pygame.draw.lines(screen, c, False, pts_s, zcu_lw)
                        if succ in node_xy:
                            pygame.draw.circle(screen, (255,200,60), w2s(*node_xy[succ]), mr+2, 2)

                # Info panel
                info_lines = []
                types = []
                if is_merge: types.append("MERGE")
                if is_diverge: types.append("DIVERGE")
                if is_boundary: types.append("BOUNDARY")
                info_lines.append(f"{'+'.join(types) if types else 'NODE'} {hovered_zcu_node}")

                for zone in gmap.zcu_zones:
                    if zone.node_id == hovered_zcu_node:
                        lock_id = f"{zone.node_id}_{zone.kind}"
                        holder = des._zone_lock.get(lock_id)
                        waiters = des._zone_waiters.get(lock_id, [])
                        holder_str = f"V#{holder.id}" if holder else "FREE"
                        wait_str = ",".join(f"V#{w.id}" for w in waiters) if waiters else "-"
                        info_lines.append(f"  {zone.kind} lock={holder_str} wait=[{wait_str}]")

                for zone, lock_id in des._boundary_to_zones.get(hovered_zcu_node, []):
                    holder = des._zone_lock.get(lock_id)
                    waiters = des._zone_waiters.get(lock_id, [])
                    holder_str = f"V#{holder.id}" if holder else "FREE"
                    wait_str = ",".join(f"V#{w.id}" for w in waiters) if waiters else "-"
                    info_lines.append(f"  bnd->{zone.node_id}_{zone.kind} lock={holder_str} wait=[{wait_str}]")

                for zone, lock_id in des._exit_to_zones.get(hovered_zcu_node, []):
                    holder = des._zone_lock.get(lock_id)
                    if holder:
                        info_lines.append(f"  exit<-{lock_id} holder=V#{holder.id}")

                waiting_here = [v for v in vehicles
                                if v.waiting_at_zcu and
                                hovered_zcu_node in (v.waiting_at_zcu, )]
                for v in vehicles:
                    if v.waiting_at_zcu and hovered_zcu_node in v.waiting_at_zcu \
                       and v not in waiting_here:
                        waiting_here.append(v)
                if waiting_here:
                    info_lines.append(f"  OHTs waiting: {[f'V#{v.id}' for v in waiting_here]}")

                # Arrows to lock holders
                drawn_holders = set()
                all_zone_entries = list(des._boundary_to_zones.get(hovered_zcu_node, []))
                for zone in gmap.zcu_zones:
                    if zone.node_id == hovered_zcu_node:
                        lid = f"{zone.node_id}_{zone.kind}"
                        all_zone_entries.append((zone, lid))
                for zone, lock_id in all_zone_entries:
                    holder = des._zone_lock.get(lock_id)
                    if holder and holder.id not in drawn_holders:
                        drawn_holders.add(holder.id)
                        hx2, hy2 = w2s(holder.x, holder.y)
                        pygame.draw.line(screen, (255, 100, 255), (hsx, hsy), (hx2, hy2), 2)
                        pygame.draw.circle(screen, (255, 100, 255), (hx2, hy2),
                                           max(5, int(scale * 250)), 2)
                        lbl = font_s.render(f"V#{holder.id}", True, (255, 100, 255))
                        screen.blit(lbl, (hx2 + 8, hy2 - 14))

                if info_lines:
                    pw = max(len(line) * 7 + 10 for line in info_lines)
                    ph = 14 * len(info_lines) + 8
                    px = hsx + r + 8
                    py = hsy - ph // 2
                    _draw_panel(screen, px, py, pw, ph, 220)
                    for i, line in enumerate(info_lines):
                        c = (255, 255, 255) if i == 0 else (200, 200, 200)
                        if "FREE" in line:
                            c = (80, 255, 80)
                        elif "V#" in line and "lock=" in line:
                            c = (255, 200, 60)
                        screen.blit(font_s.render(line, True, c), (px + 4, py + 4 + i * 14))

        # ── Leader arrows (all, toggle with L) ───────────────────────────
        if show_leaders:
            for v in vehicles:
                if not v.leader:
                    continue
                sx1, sy1 = w2s(v.x, v.y)
                sx2, sy2 = w2s(v.leader.x, v.leader.y)
                if (sx1 < -50 or sx1 > SW+50 or sy1 < -50 or sy1 > SH+50) and \
                   (sx2 < -50 or sx2 > SW+50 or sy2 < -50 or sy2 > SH+50):
                    continue
                g = v.gap_to_leader
                if g < v.length:
                    lc = VIOL_C
                elif g < v.h_min:
                    lc = (255, 200, 60)
                else:
                    lc = (60, 120, 60)
                pygame.draw.line(screen, lc, (sx1, sy1), (sx2, sy2), 1)

        # ── Draw vehicles ────────────────────────────────────────────────
        vhl, vhw = 750 / 2, 500 / 2
        for v in vehicles:
            sx, sy = w2s(v.x, v.y)
            if sx < -50 or sx > SW + 50 or sy < -50 or sy > SH + 50:
                continue

            th = v.theta
            ct, st_v = math.cos(th), math.sin(th)
            corners = [w2s(v.x + lx * ct - ly * st_v, v.y + lx * st_v + ly * ct)
                       for lx, ly in [(vhl, vhw), (vhl, -vhw), (-vhl, -vhw), (-vhl, vhw)]]

            if v.state == STOP:
                clr = STOP_REASON_C.get(v.stop_reason, STATE_C[STOP])
            elif v.state == LOADING:
                clr = STATE_C[LOADING]
            else:
                # Moving (ACCEL/CRUISE/DECEL). Distinguish purpose:
                #   - has an active retrieve/deliver job  → green (own work)
                #   - no job, just being pushed aside     → blue
                # via_push alone doesn't capture the case where an IDLE V
                # was assigned a temporary destination but its job stays
                # None — use job presence as the source of truth.
                if v.job is not None and v.job_state in ('TO_PICKUP', 'TO_DROP'):
                    clr = MOVING_C
                else:
                    clr = IDLE_MOV_C
            # Override to bright red for vehicles involved in a collision
            if v.id in collision_vehicles:
                clr = (255, 40, 40)
            is_sel = selected and v.id == selected.id

            # Violation flash (pulsing red outline)
            if show_violations and v.id in violation_vehicles:
                pulse = int(128 + 127 * math.sin(sim_time * 10))
                pygame.draw.polygon(screen, (255, pulse // 2, pulse // 4), corners, 4)

            # Stuck holder: magenta outline
            if v.id in stuck_holders:
                pygame.draw.polygon(screen, STUCK_C, corners, 3)

            # Collision: persistent thick yellow outline
            if v.id in collision_vehicles:
                pygame.draw.polygon(screen, (255, 230, 0), corners, 3)

            pygame.draw.polygon(screen, clr, corners)
            if is_sel:
                pygame.draw.polygon(screen, (255, 255, 100), corners, 2)

            # Committed trajectory 끝점 (commit horizon = x_marker) marker.
            # 노란 X 로 표시 — V 가 현재 committed plan 으로 도달할 최종 지점.
            if show_commit:
                pidx = v.x_marker_pidx
                if 0 <= pidx < len(v.path) - 1:
                    cseg = gmap.segments.get((v.path[pidx], v.path[pidx + 1]))
                    if cseg and cseg.path_points:
                        cmx, cmy, _cth = _interp_path(cseg.path_points,
                                                      v.x_marker_offset)
                        cmsx, cmsy = w2s(cmx, cmy)
                        pygame.draw.line(screen, (255, 255, 0),
                                         (cmsx - 5, cmsy - 5),
                                         (cmsx + 5, cmsy + 5), 2)
                        pygame.draw.line(screen, (255, 255, 0),
                                         (cmsx - 5, cmsy + 5),
                                         (cmsx + 5, cmsy - 5), 2)

            # Load indicator: small yellow square at center when carrying a load
            # (after pickup, before drop). job_state TO_DROP/UNLOADING means
            # the vehicle has physical load aboard.
            if getattr(v, 'job_state', None) in ('TO_DROP', 'UNLOADING'):
                lsize = max(2, int(scale * 200))
                pygame.draw.rect(screen, (255, 230, 0),
                                 (sx - lsize, sy - lsize, lsize * 2, lsize * 2))

            # Front dot
            fx = v.x + vhl * 0.7 * ct
            fy = v.y + vhl * 0.7 * st_v
            pygame.draw.circle(screen, (255, 255, 255), w2s(fx, fy), max(2, int(scale * 120)))

            # Vehicle ID + lock count badge
            if scale > 0.015:
                n_locks = len(v_locks.get(v.id, []))
                if n_locks > 0 and scale > 0.025:
                    lbl_txt = f"{v.id}[{n_locks}]"
                    lbl_c = (255, 200, 60)
                else:
                    lbl_txt = str(v.id)
                    lbl_c = TEXT
                screen.blit(font_s.render(lbl_txt, True, lbl_c), (sx - 4, sy + 8))

            # Selected vehicle: show velocity next to OHT (in addition to HUD panel)
            if is_sel:
                vel_now = v.vel_at(sim_time)
                vel_txt = f"v={vel_now:.0f}mm/s"
                screen.blit(font_s.render(vel_txt, True, (255, 255, 100)),
                            (sx - 4, sy + 22))

        # Committed trajectory 경로선. X 토글 시 commit horizon 이 현재 위치보다
        # 앞선(= MOVING) 차량 전부에 얇은 노란 선, 선택 차량은 굵은 선.
        # vehicle loop 밖에서 그려야 다른 OHT polygon 에 안 가려짐 (맨 위 z-order).
        if show_commit:
            def _commit_pts(v):
                end_pidx = v.x_marker_pidx
                if not (0 <= end_pidx < len(v.path) - 1):
                    return None
                # forward commit 없으면(= idle/정지, horizon == 현재위치) skip
                if not (end_pidx > v.path_idx
                        or v.x_marker_offset > v.seg_offset + 1.0):
                    return None
                pts = [w2s(v.x, v.y)]
                for pi in range(v.path_idx + 1, end_pidx + 1):
                    nd = gmap.nodes.get(v.path[pi])
                    if nd:
                        pts.append(w2s(nd.x, nd.y))
                cseg = gmap.segments.get((v.path[end_pidx], v.path[end_pidx + 1]))
                if cseg and cseg.path_points:
                    ex, ey, _eth = _interp_path(cseg.path_points,
                                                v.x_marker_offset)
                    pts.append(w2s(ex, ey))
                return pts if len(pts) >= 2 else None
            # 비선택 MOVING 차량: 얇은 dim 노란
            for v in vehicles:
                if v is selected:
                    continue
                pts = _commit_pts(v)
                if pts:
                    pygame.draw.lines(screen, (170, 170, 40), False, pts, 1)
            # 선택 차량: 굵은 bright 노란 (맨 위)
            if selected is not None:
                pts = _commit_pts(selected)
                if pts:
                    pygame.draw.lines(screen, (255, 255, 0), False, pts, 3)

        # Destination marker — connects current pos to dest_node (straight
        # line). Color-coded: push target = orange, normal dest = green.
        # Toggle with T.
        if show_dests:
            PUSH_COL = (255, 140, 0)    # orange
            DEST_COL = (0, 220, 120)    # green
            for v in vehicles:
                if not v.dest_node or v.dest_reached:
                    continue
                dest_xy = node_xy.get(v.dest_node)
                if dest_xy is None:
                    continue
                dsx, dsy = w2s(dest_xy[0], dest_xy[1])
                if not (0 <= dsx <= SW and 0 <= dsy <= SH):
                    continue
                col = PUSH_COL if v.via_push else DEST_COL
                vsx, vsy = w2s(v.x, v.y)
                pygame.draw.line(screen, col, (vsx, vsy), (dsx, dsy), 1)
                dr = max(3, int(scale * 100))
                pygame.draw.circle(screen, col, (dsx, dsy), dr, 2)
                if scale > 0.02:
                    lbl = font_s.render(v.dest_node, True, col)
                    screen.blit(lbl, (dsx + dr + 2, dsy - 6))

        # ── Selected OHT: path + leader arrow ───────────────────────────
        if selected:
            v = selected
            path_lw = max(1, int(scale * 60))
            for i in range(v.path_idx, min(v.path_idx + 30, len(v.path) - 1)):
                seg = v.gmap.segment_between(v.path[i], v.path[i + 1])
                if seg and seg.path_points:
                    pts_s = [w2s(p[0], p[1]) for p in seg.path_points]
                    if len(pts_s) >= 2:
                        pygame.draw.lines(screen, v.color[:3], False, pts_s, path_lw)

        if selected and selected.leader:
            s1 = w2s(selected.x, selected.y)
            s2 = w2s(selected.leader.x, selected.leader.y)
            pygame.draw.line(screen, (255, 255, 100), s1, s2, 2)
            lx, ly = w2s(selected.leader.x, selected.leader.y)
            pygame.draw.circle(screen, (255, 255, 100), (lx, ly), max(4, int(scale * 200)), 2)
            screen.blit(font_s.render(f"Leader:#{selected.leader.id}", True, (255, 255, 100)),
                         (lx + 6, ly - 16))

        # ══════════════════════════════════════════════════════════════════
        # HUD
        # ══════════════════════════════════════════════════════════════════

        sc = collections.Counter(v.state for v in vehicles)
        zcu_waiting = sum(1 for v in vehicles if v.waiting_at_zcu is not None)
        total_locks_held = sum(1 for h in des._zone_lock.values() if h is not None)
        total_locks = len(des._zone_lock)

        # ── Top-left: status ─────────────────────────────────────────────
        lines = [
            f"t={sim_time:.1f}s  x{sim_speed:.1f}  {'PAUSED' if paused else 'RUNNING'}",
            f"Events:{des.event_count}  DES-t:{des.sim_time:.2f}",
            f"OHT:{len(vehicles)}  Segs:{len(gmap.segments)}  ZCU:{len(gmap.zcu_nodes)}",
            f"ACCEL:{sc.get(ACCEL,0)} CRUISE:{sc.get(CRUISE,0)} DECEL:{sc.get(DECEL,0)}"
            f" STOP:{sc.get(STOP,0)} LOAD:{sc.get(LOADING,0)} IDLE:{sc.get(IDLE,0)}",
            f"Locks:{total_locks_held}/{total_locks}  ZCU-wait:{zcu_waiting}  Stuck:{len(stuck_holders)}",
            f"GapViol:{total_viol} ZcuViol:{des.zcu_violation_count} MinGap:{min_gap:.0f}mm",
            f"SpeedViol:{des.speed_violation_count}",
            f"Collisions:{len(collision_log)}"
            f"{'  AUTO-PAUSED' if collision_autopause and collision_log else ''}",
        ]
        # Push counter (always shown — 0 if no push activity)
        lines.append(f"Push:{des.push_count}")
        # Dispatch stats (only when JobManager is attached)
        if des.job_mgr is not None:
            mgr = des.job_mgr
            mstats = mgr.stats()
            lines.append(f"Jobs created:{mstats['total_created']} "
                         f"pending:{mstats['pending']} "
                         f"asgn:{mstats['assigned']} "
                         f"done:{mstats['completed']}")
        if collision_log:
            lines.append("--- Recent collisions ---")
            for e in collision_log[-3:]:
                lines.append(f"t={e[0]:.2f} V#{e[1]}<-V#{e[2]} gap={e[3]:.0f}")

        # ── Selected vehicle detail ──────────────────────────────────────
        if selected:
            v = selected
            lines.append("")
            lines.append(f"--- OHT #{v.id} ({STATE_LABEL.get(v.state, v.state)}) ---")
            lines.append(f"v={v.vel_at(sim_time):.0f}  a={v.acc:.0f}")
            lines.append(f"Seg: {v.seg_from} -> {v.seg_to}  off={v.seg_offset:.0f}")
            lines.append(f"PathIdx: {v.path_idx}/{len(v.path)}")

            # Leader with gap status
            if v.leader:
                g = v.gap_to_leader
                if g < v.length:
                    gap_s = "VIOLATION"
                elif g < v.h_min:
                    gap_s = "CLOSE"
                else:
                    gap_s = "OK"
                lines.append(f"Leader: #{v.leader.id} [{STATE_LABEL.get(v.leader.state,'')}]"
                             f"  gap={g:.0f}({gap_s})")
            else:
                lines.append("Leader: -")

            if v.stop_dist is not None:
                lines.append(f"Stop: dist={v.stop_dist:.0f}  reason={v.stop_reason or 'free'}")
            if v.x_marker_node:
                lines.append(f"XMarker: @{v.x_marker_node}")
            if v.waiting_at_zcu:
                lines.append(f"WaitZCU: {v.waiting_at_zcu}")

            held = v_locks.get(v.id, [])
            if held:
                lines.append(f"Locks({len(held)}):")
                for lid in held[:6]:
                    lines.append(f"  {lid}")
                if len(held) > 6:
                    lines.append(f"  +{len(held)-6} more")
            else:
                lines.append("Locks: -")

            if v.passed_zcu:
                lines.append(f"Passed: {list(v.passed_zcu)[:4]}")
            if v.dest_node:
                lines.append(f"Dest: {v.dest_node} {'REACHED' if v.dest_reached else ''}")

            # ── Plan / committed trajectory (남은 action) ────────────────
            traj = v.committed_traj or []
            future = [e for e in traj if e[0] >= sim_time - 1e-3]
            if future:
                lines.append(f"Plan({len(future)}/{len(traj)} phases):")
                for (pt, pd, pv, pa) in future[:5]:
                    dt = pt - sim_time
                    if abs(pa) < 1:    phase = 'CRZ'
                    elif pa > 0:        phase = 'ACC'
                    else:               phase = 'DEC'
                    lines.append(f"  +{dt:5.2f}s  v={pv:5.0f}  a={pa:+5.0f}  {phase}  d={pd:.0f}")
                if len(future) > 5:
                    lines.append(f"  +{len(future)-5} more")
            else:
                lines.append("Plan: empty (no committed action)")

            # ── Pending event (heap 의 다음 event) ─────────────────────
            if v.next_event_t is not None and v.next_event_t < float('inf'):
                lines.append(f"NextEvent: t={v.next_event_t:.2f} "
                             f"(in {v.next_event_t - sim_time:+.2f}s)")
            # heap 에서 vid 의 다음 event kind 찾기 (선형 스캔, 작은 비용)
            next_evs = sorted(
                ((ev.t, ev.kind) for ev in des.heap if ev.vid == v.id),
                key=lambda x: x[0])[:3]
            if next_evs:
                lines.append("Heap events:")
                for (et, ek) in next_evs:
                    lines.append(f"  t={et:.2f}  {ek}")

            # ── Followers (뒤에 누가) ──────────────────────────────────
            followers = sorted(des._followers.get(v.id, set()),
                               key=lambda f: f.id)
            if followers:
                lines.append(f"Followers({len(followers)}): "
                             f"{[f'V#{f.id}' for f in followers[:6]]}")

            # ── Push diagnostic ───────────────────────────────────────
            if v.last_push_t > -1e8:
                lines.append(f"LastPushT: {v.last_push_t:.2f} "
                             f"({sim_time - v.last_push_t:.1f}s ago)")

        pw = 440
        ph = 16 * len(lines) + 16
        _draw_panel(screen, 8, 8, pw, ph)
        for i, line in enumerate(lines):
            if line.startswith("---"):
                c = (120, 200, 255)
            elif "VIOLATION" in line:
                c = VIOL_C
            elif "CLOSE" in line and "gap=" in line:
                c = (255, 200, 60)
            else:
                c = TEXT
            screen.blit(font_s.render(line, True, c), (14, 14 + i * 16))

        # ── Bottom-left: violation log ───────────────────────────────────
        if show_violations and des.zcu_violation_log:
            recent = des.zcu_violation_log[-8:]
            viol_lines = ["ZCU Violations (recent):"]
            for entry in reversed(recent):
                t_v, lid, vtype, detail = entry
                viol_lines.append(f"  t={t_v:.1f} {vtype} {lid}")
            vpw = 400
            vph = 14 * len(viol_lines) + 8
            vpy = SH - vph - 30
            _draw_panel(screen, 8, vpy, vpw, vph)
            for i, line in enumerate(viol_lines):
                c = (255, 200, 200) if i == 0 else (255, 120, 120)
                screen.blit(font_s.render(line, True, c), (14, vpy + 4 + i * 14))

        # ── Bottom-right: gap histogram ──────────────────────────────────
        hx, hy, hw, hh = SW - 220, SH - 160, 200, 140
        _draw_panel(screen, hx - 5, hy - 5, hw + 10, hh + 10, 180)
        screen.blit(font_s.render("Gap(mm)", True, TEXT), (hx, hy - 2))
        mx_c = max(gap_hist) if any(gap_hist) else 1
        bw2 = hw // 20
        for i, c in enumerate(gap_hist):
            bh = int((c / mx_c) * (hh - 20)) if mx_c > 0 else 0
            bc = ((255, 0, 0) if i * 100 < 750 else
                  (255, 200, 60) if i * 100 < 1150 else
                  (60, 200, 60))
            pygame.draw.rect(screen, bc, (hx + i * bw2, hy + hh - bh, bw2 - 1, bh))

        # ── Legend ───────────────────────────────────────────────────────
        legend = [
            (ZCU_FREE_C,                "ZCU free"),
            (ZCU_LOCKED_C,              "ZCU locked"),
            (ZCU_WAIT_C,                "locked+waiters"),
            (STUCK_C,                   "Stuck holder"),
            (VIOL_C,                    "Violation"),
            (MOVING_C,                  "Moving (own job)"),
            (STOP_REASON_C[None],       "STOP idle (pushable)"),
            (STOP_REASON_C['leader'],   "STOP blocked by leader"),
            (STOP_REASON_C['zcu'],      "STOP blocked by ZCU"),
        ]
        ly_start = hy - 82 - 4 * 15
        lx_start = SW - 180
        for idx, (clr, label) in enumerate(legend):
            ly_pos = ly_start + idx * 15
            pygame.draw.circle(screen, clr, (lx_start, ly_pos + 5), 5)
            screen.blit(font_s.render(label, True, DIM), (lx_start + 10, ly_pos - 1))

        # Controls hint
        screen.blit(font_s.render(
            "Space:Pause +/-:Speed R:Reset N:Nodes Z:ZCU L:Leaders V:Violations P:SpeedViol F:Follow",
            True, DIM), (10, SH - 20))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
