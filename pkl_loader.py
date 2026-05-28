"""
pkl_loader.py — Load a collision-profile .pkl into a MapGraph + region overlays.

The pkl contains:
  Stop_state    : {state_id -> State}  (S,node_idx,angle)
  Move_state    : {state_id -> State}  (M,from_idx,to_idx)
  stop_regions  : {state_id -> Shapely Polygon}   agent footprint at each Stop
  move_regions  : {state_id -> Shapely Polygon}   swept area for each Move

Node positions are inferred as the centroid of each stop_region polygon.
Edge angles are computed from node-centroid geometry.
Edge max_speed defaults to DEFAULT_SPEED (1000 mm/s) — consistent with pkl data.
"""
from __future__ import annotations
import math
import pickle
import sys
import os
from typing import Dict, Tuple, List

# Need the State class from the solver to unpickle
def _load_pkl(filepath: str) -> dict:
    """Load pkl, injecting solver State class into __main__ for unpickling."""
    solver_dir = os.path.join(os.path.dirname(__file__), 'solvers')
    if solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
    try:
        from ACS_graph_grid_focal_crisscross_heapcost_backup1028_General_Queuing import (
            State, Point, PointType, Direction
        )
        import __main__
        __main__.State     = State
        __main__.Point     = Point
        __main__.PointType = PointType
        __main__.Direction = Direction

        # pkl이 Generalized_251012_Affect_state 모듈로 직렬화된 경우 대응
        import types
        for mod_name in ('Generalized_251012_Affect_state',
                         'Generalized_251012_Affect_state_no_rot'):
            if mod_name not in sys.modules:
                fake = types.ModuleType(mod_name)
                fake.State     = State
                fake.Point     = Point
                fake.PointType = PointType
                fake.Direction = Direction
                sys.modules[mod_name] = fake
    except ImportError as e:
        raise ImportError(f"Cannot import solver State class: {e}")

    with open(filepath, 'rb') as f:
        return pickle.load(f)


DEFAULT_SPEED = 1000.0   # mm/s  (all edges in the test map use 1 m/s)


# ── Minimal node / edge containers (compatible with MapGraph interface) ────────

class PklNode:
    def __init__(self, node_id: str, x: float, y: float):
        self.id   = node_id
        self.x    = x       # mm
        self.y    = y       # mm
        self.kind = 'Normal'

    def __repr__(self):
        return f'Node({self.id}, {self.x/1000:.2f}m, {self.y/1000:.2f}m)'


class PklEdge:
    def __init__(self, from_id: str, to_id: str,
                 from_node: PklNode, to_node: PklNode,
                 max_speed: float = DEFAULT_SPEED):
        self.id        = f"M,{from_id},{to_id}"
        self.from_id   = from_id
        self.to_id     = to_id
        dx             = to_node.x - from_node.x
        dy             = to_node.y - from_node.y
        self.length    = math.hypot(dx, dy)          # mm
        self.angle     = math.atan2(dy, dx)          # radians (map-space, y-up)
        self.max_speed = max_speed                   # mm/s

    def __repr__(self):
        return f'Edge({self.from_id}→{self.to_id}, L={self.length:.0f}mm)'


class _LoadState:
    """MCS LOADING/UNLOADING dwell 을 path-plan 의 일부로 표현하는 state.

    State graph 의 stop/move/rotate state 와 동일한 interface (cost, next_state,
    affect_state) 를 제공해 SIPP planner / TAPG runtime 이 동일하게 처리.

    L,node_id 형식의 sid 로 식별. AGV 가 port 에 도착 후 이 state 에서 cost
    (= dwell_time) 만큼 정지. 그 동안 자기 노드 + 인접 M edges 점유.
    """
    def __init__(self, node_id: str, cost: float,
                 next_state: List[str], affect_state: List[str]):
        self.id           = f'L,{node_id}'
        self.cost         = cost
        self.next_state   = next_state
        self.affect_state = affect_state
        self.type         = 'Load'
        # State 와 호환을 위한 dummy 필드 (실제로는 사용 안 됨)
        self.center       = None
        self.heading      = 0.0
        self.end_id       = node_id
        self.start_id     = node_id
        self.offset       = 0.0
        self.segment_id   = -1
        self.interval_list   = []
        self.rsv_time_table  = {}
        self.rsv_veh_list    = []
        self.sc           = None
        self.sr           = None
        self.split_interval = []


class PklMapGraph:
    """
    MapGraph-compatible graph built from a collision-profile .pkl.

    Additional attributes (not in plain MapGraph)
    ──────────────────────────────────────────────
    stop_regions  : {state_id: Shapely Polygon}
    move_regions  : {state_id: Shapely Polygon}
    stop_states   : raw State dict from pkl
    move_states   : raw State dict from pkl
    """

    def __init__(self, pkl_path: str):
        self._load(pkl_path)

    def _load(self, pkl_path: str):
        data = _load_pkl(pkl_path)

        self.stop_regions: Dict[str, object] = data.get('stop_regions', {})
        self.move_regions: Dict[str, object] = data.get('move_regions', {})
        self.stop_states_raw   = data.get('Stop_state',   {})
        self.move_states_raw   = data.get('Move_state',   {})
        self.rotate_states_raw = data.get('Rotate_state', {})
        # Rotate state cost 정규화: 항상 *최단 회전* 으로 override.
        # pkl 원본은 일부 R state 가 긴 방향 (예: 0->357 = 357°) 으로 cost 계산
        # 되어 planning 이 과대평가. runtime _start_rotate 는 이미 최단 방향
        # 으로 회전하므로 cost 만 일치시켜 정합성 확보.
        # 90 deg/sec 가정 (= ANGULAR_SPEED, env_tapg.py:34).
        for rs_id, rs in self.rotate_states_raw.items():
            parts = rs_id.split(',')   # R,node,from,to
            if len(parts) >= 4:
                try:
                    from_deg = float(parts[2])
                    to_deg = float(parts[3])
                    diff_ccw = (to_deg - from_deg) % 360
                    shortest = min(diff_ccw, 360.0 - diff_ccw)
                    # 90 deg/sec 기준. 기존 cost 의 unit 추정해서 비례 유지.
                    rs.cost = shortest / 90.0
                except (ValueError, IndexError, AttributeError):
                    pass
        # Load/Unload state — MCS dwell 을 path-plan 의 일부로 표현.
        # `build_load_states(dwell_time, port_nodes)` 호출 시 생성. plan/runtime
        # 이 L state 를 만나면 cost 만큼 정지 (= LOADING/UNLOADING dwell).
        self.load_states_raw: Dict[str, '_LoadState'] = {}

        # ── Node positions from stop_region centroids ──────────────────────
        self.nodes: Dict[str, PklNode] = {}
        for state_id, poly in self.stop_regions.items():
            parts = state_id.split(',')
            if parts[0] != 'S':
                continue
            nid = parts[1]
            if nid not in self.nodes:
                c = poly.centroid
                self.nodes[nid] = PklNode(nid, c.x, c.y)

        # ── Edges from move_state keys ─────────────────────────────────────
        # Edge max_speed 는 M state 의 cost (= dist / speed, gen 시 저장) 와
        # node 간 유클리드 거리에서 역산. pkl 의 timing 의도를 그대로 사용해
        # layout 단위 (예: 송도의 px=mm, speed 10 px/s) 가 자동 반영됨.
        self.edges: Dict[Tuple, PklEdge] = {}
        self.adj:   Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for state_id, m_state in self.move_states_raw.items():
            parts = state_id.split(',')
            if parts[0] != 'M':
                continue
            fn, tn = parts[1], parts[2]
            if fn not in self.nodes or tn not in self.nodes:
                continue
            # speed = dist / cost
            dx = self.nodes[tn].x - self.nodes[fn].x
            dy = self.nodes[tn].y - self.nodes[fn].y
            dist = math.hypot(dx, dy)
            m_cost = getattr(m_state, 'cost', 0)
            if m_cost and m_cost > 0:
                edge_speed = dist / m_cost
            else:
                edge_speed = DEFAULT_SPEED
            edge = PklEdge(fn, tn, self.nodes[fn], self.nodes[tn], edge_speed)
            self.edges[(fn, tn)] = edge
            if tn not in self.adj.get(fn, []):
                self.adj.setdefault(fn, []).append(tn)

        # ── Vehicle dimensions from stop polygon edge lengths ──────────────
        self.vehicle_length = 2000.0   # mm default
        self.vehicle_width  = 1300.0   # mm default
        if self.stop_regions:
            sample_poly = next(iter(self.stop_regions.values()))
            coords = list(sample_poly.exterior.coords)[:-1]
            if len(coords) >= 2:
                d01 = math.hypot(coords[1][0] - coords[0][0],
                                 coords[1][1] - coords[0][1])
                d12 = math.hypot(coords[2][0] - coords[1][0],
                                 coords[2][1] - coords[1][1])
                self.vehicle_length = max(d01, d12)
                self.vehicle_width  = min(d01, d12)

        # ── Ports: nodes that only appear in stop states (no outgoing edges)
        # or nodes explicitly marked — use od_pairs destinations if available
        self.ports: Dict[str, str] = {}
        od = data.get('od_pairs', [])
        for i, (src, dst) in enumerate(od):
            self.ports[str(i)] = str(dst)

    # ── Load state 빌드 ────────────────────────────────────────────────────────

    def build_load_states(self, dwell_time: float,
                          port_nodes: List[str] = None):
        """각 port 노드에 L state 생성.

        port_nodes 가 None 이면 self.ports.values() 사용. 명시 시 외부 (예:
        sidings) 도 포함 가능.

        L state 의 properties:
          - cost = dwell_time
          - next_state = 같은 노드의 stop states (LOADING 끝나면 stop 으로)
          - affect_state = 같은 노드의 stop states + 인접 M edges (자기 점유 표현)
        """
        if port_nodes is None:
            port_nodes = list(set(self.ports.values()))

        self.load_states_raw.clear()
        for nid in port_nodes:
            if nid not in self.nodes:
                continue
            # 같은 노드의 stop states 수집
            node_stops = [sid for sid in self.stop_states_raw
                          if sid.split(',')[1] == nid]
            # 인접 M edges (IN + OUT)
            adj_moves = []
            for sid in self.move_states_raw:
                parts = sid.split(',')
                if len(parts) >= 3 and (parts[1] == nid or parts[2] == nid):
                    adj_moves.append(sid)
            # affect_state = 자기 노드 stops + 인접 M edges
            affect = node_stops + adj_moves
            # next_state = stop states (어떤 heading 으로든 transition 가능)
            next_st = list(node_stops)
            self.load_states_raw[f'L,{nid}'] = _LoadState(
                nid, dwell_time, next_st, affect)

        # stop state 에 L 로의 transition 추가 — port stop 에서 LOADING 진입 가능
        for sid in list(self.stop_states_raw.keys()):
            parts = sid.split(',')
            nid = parts[1] if len(parts) >= 2 else None
            if nid in self.load_states_raw or f'L,{nid}' in self.load_states_raw:
                s = self.stop_states_raw[sid]
                lsid = f'L,{nid}'
                if hasattr(s, 'next_state') and lsid not in s.next_state:
                    # next_state 가 list 인 경우만 (state graph mutability)
                    if isinstance(s.next_state, list):
                        s.next_state.append(lsid)

    # ── MapGraph-compatible helpers ────────────────────────────────────────────

    def get_edge(self, from_id: str, to_id: str) -> 'PklEdge | None':
        return self.edges.get((from_id, to_id))

    def neighbors(self, node_id: str) -> List[str]:
        return self.adj.get(node_id, [])

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        return min(xs), min(ys), max(xs), max(ys)

    def __repr__(self):
        return (f'PklMapGraph({len(self.nodes)} nodes, '
                f'{len(self.edges)} edges, '
                f'{len(self.stop_regions)} stop_regions, '
                f'{len(self.move_regions)} move_regions)')
