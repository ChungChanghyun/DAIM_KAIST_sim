"""
env_oht_v6_adapter.py — OHT engine 을 graph_des_v6 (GraphDESv6) 로 교체.

기존 env_oht_des.OHTEnvironmentDES / OHTAgent 인터페이스를 graph_des_v6 의
GraphDESv6 / Vehicle 위에 wrapping 하여, vis_mcs_unified 측 코드 변경을 최소화.

대응:
  OHTMap(json, area='OHT_A')      → GraphMap(json, area_filter='OHT_A') + bfs / nearby
  OHTAgent(...)                    → wraps graph_des_v6.Vehicle
  OHTEnvironmentDES(map, ...)      → wraps GraphDESv6
  agent.state ∈ {IDLE,MOVING,..,DONE} → 엔진 state 에서 매핑

State 매핑:
  엔진 IDLE / STOP / LOADING + 경로끝(path 끝까지 도달) → DONE  (= MCS free)
  엔진 ACCEL / CRUISE / DECEL                          → MOVING
  엔진 STOP (중도)                                       → BLOCKED
"""
from __future__ import annotations
import collections
import math
from typing import Dict, List, Optional, Set, Tuple

from graph_des_v5 import GraphMap, MapNode, MapSegment, ZCUZone as V5ZCUZone


class _ZCUZoneShim:
    """env_oht_des.ZCUZone 호환 외피 (entry_segs / exit_segs 노출).

    graph_des_v5 의 ZCUZone 은 curve/straight 분류, env_oht_des 는 entry/exit.
    의미는 다르지만 vis 렌더링은 "이 zone 에 속한 모든 segment" 만 알면 충분.
    entry_segs = exit_segs = all_segs() 로 두어 union 결과가 곧 zone 전체 segs.
    """
    def __init__(self, zone: V5ZCUZone):
        self._z = zone
        # graph_des_v6 의 lock_id 와 일치하는 zone id 형식
        self.id        = f"{zone.node_id}_{zone.kind}"
        self.zone_id   = self.id
        # entry_segs = 모든 zone segs 의 union (vis 가 union 결과만 사용)
        self.entry_segs = frozenset(zone.curve_segs | zone.straight_segs)
        self.exit_segs  = frozenset()
        # graph_des_v5 호환 (passthrough)
        self.node_id      = zone.node_id
        self.kind         = zone.kind
        self.curve_segs   = zone.curve_segs
        self.straight_segs = zone.straight_segs

    def all_segs(self):
        return self._z.all_segs()

    def __getattr__(self, name):
        return getattr(self._z, name)
from graph_des_v6 import (
    GraphDESv6, Vehicle as V6Vehicle,
    IDLE as V6_IDLE, ACCEL as V6_ACCEL, CRUISE as V6_CRUISE,
    DECEL as V6_DECEL, STOP as V6_STOP, LOADING as V6_LOADING,
)


# ── env_oht_des 호환 state 상수 ──────────────────────────────────────────────
IDLE      = 'IDLE'
MOVING    = 'MOVING'
FOLLOWING = 'FOLLOWING'
BLOCKED   = 'BLOCKED'
DONE      = 'DONE'


def _bfs_path(adj: Dict[str, List[str]], src: str, dst: str = None,
              length: int = 200) -> Optional[List[str]]:
    """BFS shortest path. dst=None 이면 src 근처에서 length 노드 길이의 random path."""
    if dst is None:
        # 단순 random walk (env_oht_des.OHTMap.bfs_path 의 기본 동작 흉내)
        import random as _rnd
        path = [src]
        cur = src
        for _ in range(length - 1):
            nbrs = adj.get(cur, [])
            if not nbrs:
                break
            cur = _rnd.choice(nbrs)
            path.append(cur)
        return path
    # 실제 BFS shortest path
    if src == dst:
        return [src]
    seen = {src: None}
    q = collections.deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in seen:
                continue
            seen[v] = u
            if v == dst:
                # backtrace
                path = [v]
                while seen[path[-1]] is not None:
                    path.append(seen[path[-1]])
                return list(reversed(path))
            q.append(v)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OHTMap — GraphMap 위에 env_oht_des.OHTMap 호환 인터페이스 layering
# ─────────────────────────────────────────────────────────────────────────────

class OHTMap:
    """env_oht_des.OHTMap 호환 외피. 내부적으로 GraphMap(area_filter='OHT_A')."""

    def __init__(self, json_path: str, area: str = 'OHT_A'):
        self.gmap = GraphMap(json_path, area_filter=area)
        self.area = area

        # 호환용 별칭 — 기존 코드가 직접 접근
        self.nodes    = self.gmap.nodes
        self.segments = self.gmap.segments
        self.adj      = self.gmap.adj

        # vehicle 파라미터 (env_oht_des 와 동일 키)
        self.vehicle_length = self.gmap.vehicle_length
        self.vehicle_width  = self.gmap.vehicle_width
        self.h_min = self.vehicle_length + 200   # mm
        self.accel = 500.0
        self.decel = 500.0
        max_v = max((s.max_speed for s in self.gmap.segments.values()),
                    default=1000.0)
        self.max_brake = max_v * max_v / (2.0 * self.decel)

        # ports
        self.port_nodes = set(self.gmap.port_nodes)
        # ZCU zones (graph_des_v5/v6 내부에서 자동 빌드 → shim 으로 wrap)
        self.zcu_zones = [_ZCUZoneShim(z)
                          for z in getattr(self.gmap, 'zcu_zones', [])]
        # ZCU 노드 (vis 표시용)
        self.zcu_node_ids: Set[str] = set(getattr(self.gmap, 'zcu_nodes', set()))
        # entry/exit 호환 (env_oht_des 의 OHTMap 에 있던 것)
        for zone in self.zcu_zones:
            for fn, tn in getattr(zone, 'entry_segs', []) or []:
                self.zcu_node_ids.add(tn)
            for fn, tn in getattr(zone, 'exit_segs', []) or []:
                self.zcu_node_ids.add(fn)

        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        self.bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0,0,0,0)

    def bfs_path(self, src: str, dst: str = None,
                 length: int = 200) -> Optional[List[str]]:
        return _bfs_path(self.adj, src, dst, length=length)

    def nearby_nodes(self, nid: str, radius: float) -> Set[str]:
        """env_oht_des.OHTMap.nearby_nodes 호환 (nid 로부터 radius mm 이내 노드)."""
        out = set()
        n = self.nodes.get(nid)
        if n is None:
            return out
        for other_id, other in self.nodes.items():
            if math.hypot(other.x - n.x, other.y - n.y) <= radius:
                out.add(other_id)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# OHTAgent — Vehicle 외피
# ─────────────────────────────────────────────────────────────────────────────

class OHTAgent:
    """env_oht_des.OHTAgent 호환 — graph_des_v6 Vehicle 을 wrap.

    주: Vehicle 은 path 가 최소 2 노드 필요. 단일 노드 starting 시 가짜 self-loop
    또는 임의 random_safe_path 로 채움 (재assign 호출 시 교체됨).
    """

    def __init__(self, aid: int, color, path: List[str],
                 vehicle_length: float):
        self.id = aid
        self.color = color
        self.length = vehicle_length

        # path 는 호출자가 제공 (≥2 노드)
        # 실제 Vehicle 객체는 attach() 에서 gmap 받아 생성
        self._init_path = list(path)
        self.vehicle: Optional[V6Vehicle] = None

        # 외부에서 참조하는 attribute 호환
        self.node_path = list(path)   # 첫 path
        self._dwelling = False
        self._t_dwell_start = 0.0

    def attach(self, gmap: GraphMap):
        """env.add_agent 시점에 호출 — Vehicle 인스턴스 생성.

        MCS 모드: vehicle.job 에 sentinel 을 set 해서 graph_des_v6 의 auto-extend
        (random_safe_path 로 path 자동 연장) 를 차단. → MCS dispatch 가 올 때만
        reassign 으로 경로 받음. 비활성 시 OHT 는 idle (path 끝에서 정차).
        """
        if self.vehicle is None:
            self.vehicle = V6Vehicle(self.id, gmap, self._init_path, self.color)
            self.vehicle.job = 'MCS'   # sentinel — auto-extend 차단

    # ── env_oht_des.OHTAgent 와 동일한 attribute 노출 ────────────────────
    @property
    def state(self) -> str:
        v = self.vehicle
        if v is None:
            return IDLE
        # dest 도달 = DONE (MCS 도착 통보 트리거). path 가 goal 너머 extend 된
        # 경우 dest 가 path 끝이 아니므로 last_seg_idx 검사로는 부족하다.
        # 두 가지 방식으로 검사:
        #  (1) v.dest_reached: v6 가 path_idx advance 시 set
        #  (2) self.cur_node == dest_node: seg 끝 도달 (어댑터 cur_node 정의)
        # (2) 가 (1) 의 fallback — v6 내부에서 dest_reached set 누락된 경우에도
        # cur_node 가 dest_node 와 같으면 도달로 인정 (시각적으로 그 자리).
        if v.dest_node is not None and v.vel < 0.5:
            if v.dest_reached:
                return DONE
            if self.cur_node == v.dest_node:
                return DONE
        # path 끝 도달 (dest 미설정인 차량 또는 extended 없는 케이스)
        last_seg_idx = len(v.path) - 2
        at_path_end = False
        if last_seg_idx >= 0:
            seg_len = 0.0
            try:
                seg_len = v.current_seg_length()
            except Exception:
                pass
            if v.path_idx >= last_seg_idx and v.seg_offset >= seg_len - 1.0:
                at_path_end = True
        elif v.path_idx >= len(v.path) - 1:
            at_path_end = True
        if at_path_end and v.vel < 0.5:
            return DONE
        if v.state in (V6_ACCEL, V6_CRUISE, V6_DECEL):
            return MOVING
        if v.state == V6_LOADING:
            return MOVING   # MCS dwell 처리
        if v.state == V6_STOP:
            # path 끝이 아니면서 STOP = leader/ZCU 대기 등 진짜 BLOCKED
            return BLOCKED
        return IDLE

    @property
    def cur_node(self) -> Optional[str]:
        v = self.vehicle
        if v is None or not v.path:
            return self._init_path[0] if self._init_path else None
        pidx = v.path_idx
        if pidx >= len(v.path):
            return v.path[-1]
        # 현재 seg 끝에 도달했으면 다음 노드를 cur_node 로 (path_idx 가 아직
        # 갱신되기 전 시점에서도 정확한 위치 반영). v6 의 STOP commit 시
        # path_idx == last_seg_idx 인 채로 seg_offset == seg_len 이라
        # path[pidx] = 직전 노드를 가리킴 → path[pidx+1] 이 실제 위치.
        if pidx < len(v.path) - 1:
            try:
                seg_len = v.current_seg_length()
                if seg_len > 0 and v.seg_offset >= seg_len - 1.0:
                    return v.path[pidx + 1]
            except Exception:
                pass
        return v.path[pidx]

    @property
    def path_idx(self) -> int:
        return self.vehicle.path_idx if self.vehicle else 0

    @property
    def raw_path(self) -> List[str]:
        return self.vehicle.path if self.vehicle else self._init_path

    @property
    def x(self) -> float:
        return self.vehicle.x if self.vehicle else 0.0

    @property
    def y(self) -> float:
        return self.vehicle.y if self.vehicle else 0.0

    @property
    def theta(self) -> float:
        return self.vehicle.theta if self.vehicle else 0.0

    @property
    def vel(self) -> float:
        return self.vehicle.vel if self.vehicle else 0.0

    @property
    def v(self) -> float:
        """env_oht_des 호환 alias."""
        return self.vel


# ─────────────────────────────────────────────────────────────────────────────
# OHTEnvironmentDES — GraphDESv6 외피
# ─────────────────────────────────────────────────────────────────────────────

class OHTEnvironmentDES:
    """env_oht_des.OHTEnvironmentDES 호환 외피. 내부 엔진 = GraphDESv6."""

    def __init__(self, oht_map: OHTMap, cross_segment: bool = True):
        self.oht_map = oht_map
        self.gmap    = oht_map.gmap
        self.des     = GraphDESv6(self.gmap)
        self.agents: List[OHTAgent] = []
        self._started = False

    def add_agent(self, agent: OHTAgent, t_start: float = 0.0):
        agent.attach(self.gmap)
        self.des.add_vehicle(agent.vehicle)
        self.agents.append(agent)

    def remove_agent(self, aid: int):
        # graph_des_v6 has no built-in vehicle removal — best-effort detach.
        self.agents = [a for a in self.agents if a.id != aid]
        v = self.des.vehicles.pop(aid, None)
        if v is not None:
            # 차량이 점유 중인 segment 큐에서도 제거
            try:
                key = (v.seg_from, v.seg_to)
                if v in self.des._seg_occupants.get(key, []):
                    self.des._seg_occupants[key].remove(v)
            except Exception:
                pass

    # ── ZCU lock 상태 노출 (vis 가 _zcu_holders/_zcu_waitlists 로 접근) ──
    @property
    def _zcu_holders(self):
        return self.des._zone_lock        # Dict[lock_id, Optional[Vehicle]]

    @property
    def _zcu_waitlists(self):
        return self.des._zone_waiters     # Dict[lock_id, List[Vehicle]]

    def reassign(self, agent: OHTAgent, new_path: List[str], t: float):
        """MCS dispatch 시 호출. dispatch.py._reroute 와 동일하게 v6 의 canonical
        path-swap API `_assign_destination` 을 통해 처리.

        new_path: caller 가 제공하는 path. cur_node 부터 또는 임의 노드부터 시작
        가능. 내부에서 `[seg_from, seg_to, ...tail]` contract 에 맞춰 재구성.

        효과 (= _assign_destination 안에서 자동 처리):
        - commit prefix 보존 (= path_idx..commit_end_idx 까지 그대로 통과)
        - stale ZCU lock release
        - passed_zcu cleanup (= 다시 거치는 boundary re-arm)
        - _release_passed_diverge_locks
        - _truncate_commit_at (= commit_end_t 에서 committed_traj 잘라줌)
        - _replan 호출

        Fallback: _assign_destination contract 불충족 시 (e.g. seg_to 없음) 또는
        예외 발생 시 기존 동작 (path 직접 수정 + _replan).
        """
        v = agent.vehicle
        if v is None or not new_path:
            return

        # --- _assign_destination contract 에 맞춰 path 구성 ---
        # new_path 의 fragment 들로부터 [seg_from, seg_to, ...] 형태 산출.
        seg_from = v.seg_from
        seg_to = v.seg_to
        dst_node = new_path[-1]
        full_path = None

        if seg_from is not None and seg_to is not None and seg_from != seg_to:
            # tail = seg_to 부터 dst 까지 BFS. 이미 new_path 가 seg_to 거치면
            # 그 위치부터 trim.
            if seg_to in new_path:
                idx = new_path.index(seg_to)
                tail = list(new_path[idx:])   # tail[0] == seg_to
            else:
                # BFS bridge seg_to → new_path[0], 그 후 new_path 이어붙임
                bridge = _bfs_path(self.gmap.adj, seg_to, new_path[0])
                if bridge:
                    # bridge[0]=seg_to, bridge[-1]=new_path[0]
                    tail = list(bridge) + list(new_path[1:])
                else:
                    tail = None
            if tail and tail[0] == seg_to:
                full_path = [seg_from] + tail
        # full_path None 이면 fallback (= legacy 직접 수정 경로) 사용

        if full_path is not None:
            try:
                self.des._assign_destination(t, v, full_path, dst_node)
                agent.node_path = list(v.path)
                return
            except Exception as e:
                # _assign_destination 실패 → fallback
                print(f'[REASSIGN-FAIL] V{v.id} _assign_destination raised: {e}'
                      f' — falling back to legacy path manipulation')

        # --- Fallback: legacy 직접 수정 (= seg_to 없거나 contract 불충족 시) ---
        if v.path and v.path[-1] == new_path[0]:
            v.extend_path(new_path[1:])
        else:
            cur_node = v.path[v.path_idx] if v.path_idx < len(v.path) else v.path[-1]
            if cur_node != new_path[0]:
                bridge = _bfs_path(self.gmap.adj, cur_node, new_path[0])
                if bridge:
                    new_path = bridge[:-1] + list(new_path)
            v.path = v.path[:v.path_idx + 1]
            v.extend_path(new_path[1:] if new_path[0] == cur_node else new_path)
        v.dest_node = new_path[-1]
        v.dest_reached = False
        agent.node_path = list(v.path)
        try:
            self.des._replan(t, v)
        except Exception:
            try:
                from graph_des_v6 import EV_START
                self.des._post(t, EV_START, v)
            except Exception:
                pass

    def step(self, t_now: float):
        if not self._started and self.agents:
            self.des.start_all()
            self._started = True
        self.des.step(t_now)

    # ── 호환 (env_oht_des 의 일부 인터페이스) ────────────────────────────
    @property
    def vehicles(self) -> Dict[int, V6Vehicle]:
        return self.des.vehicles
