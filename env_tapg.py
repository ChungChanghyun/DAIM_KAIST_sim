"""
env_tapg.py - TAPG (Temporal Action Precedence Graph) DES environment.

CBS time-based plan을 action-dependency DAG로 변환합니다.
각 에이전트는 모든 cross-agent 선행 의존 작업이 완료된 시점에만 M/R 액션을
실행하므로, 실제 주행 시간의 차이가 있어도 충돌이 발생하지 않습니다.

TAPG 노드
─────────
  (state_id, agv_id, cbs_start_time)  - ALL 상태 포함 (S, M, R)

엣지
────
  순차 엣지   : 같은 에이전트 내 연속 상태 사이
  교차 에이전트 : non-S 상태 쌍 중 affect_state 충돌 + t1 <= t2 조건 만족 시
                agent j의 path에서 t2 >= t1인 가장 첫 번째 non-S 충돌 상태만 연결

클레임 조건
────────────
  nk의 모든 predecessor 중 다른 에이전트의 것이 모두 completed_nodes에 있을 때
"""
from __future__ import annotations
import math
import heapq
import networkx as nx
INF = math.inf
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from pkl_loader import PklMapGraph

# ── 상수 ──────────────────────────────────────────────────────────────────────
ROTATION_TIME_90  = 1.0                                 # seconds per 90° turn (= 90°/s, matches gen_songdo_pkl.py rotation_speed)
ANGULAR_SPEED     = math.radians(90.0) / ROTATION_TIME_90  # rad/s

IDLE     = 'idle'
MOVING   = 'moving'
ROTATING = 'rotating'
WAITING  = 'waiting'
DONE     = 'done'


# ── 이벤트 ────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class Event:
    time:     float
    seq:      int
    kind:     str  = field(compare=False)
    agent_id: int  = field(compare=False)
    data:     dict = field(compare=False, default_factory=dict)


# ── TAPG 에이전트 ─────────────────────────────────────────────────────────────

class TAPGAgent:
    """
    CBS raw_path [(state_id, cbs_cost), ...] 를 따라 이동하는 에이전트.
    S 상태는 즉시 통과하고, M/R 상태는 TAPG 선행 조건을 확인합니다.
    """

    def __init__(self, agent_id: int, color: tuple, raw_path: list):
        self.id       = agent_id
        self.color    = color
        self.raw_path = raw_path   # [(state_id, cbs_cost), ...]
        self.path_idx = 0
        self.state    = IDLE

        # 위치 / 자세
        self.x:     float = 0.0
        self.y:     float = 0.0
        self.theta: float = 0.0
        self.v:     float = 0.0

        # MOVING - 위치 기반 도달 감지
        self.from_x:        float = 0.0
        self.from_y:        float = 0.0
        self.to_x:          float = 0.0
        self.to_y:          float = 0.0
        self.dist_traveled: float = 0.0   # 현재 엣지에서 이동한 거리 (mm)
        self.edge_length:   float = 0.0   # 현재 엣지 전체 길이 (mm)
        self.max_speed:     float = 0.0   # 현재 엣지 최고 속도 (mm/s)

        # ROTATING - 각도 기반 완료 감지
        self.from_theta:      float = 0.0
        self.to_theta:        float = 0.0
        self.angle_traversed: float = 0.0   # 이번 회전에서 회전한 각도 (rad)
        self.angle_total:     float = 0.0   # 목표 회전 각도 (rad)

        # 현재 실행 중인 TAPG 노드 (도달 시 complete 처리에 사용)
        self._tapg_node: tuple = None

        # Claim: 실제 주행이 확정된 구간의 끝 인덱스 (exclusive)
        # path_idx ~ claim_idx 구간은 멈출 수 없음
        self.claim_idx: int = 0
        # Coarse mode cycle prevention: AGV 가 *다음에 진입하려는 node*.
        # claim_idx 끝의 다음 state 의 dest node. Cycle check 에 사용.
        self._wanting: 'str | None' = None
        # WAITING 진입 시각 (FIFO claim 순서용). None = 현재 WAITING 아님.
        self._wait_start_t: 'float | None' = None
        # Claim 우선순위 (작을수록 먼저). cycle-push victim 은 -1 (= 최우선).
        # MOVING 진입 시 0 으로 reset.
        self._priority: int = 0

    @property
    def cur_state_id(self) -> str:
        if self.path_idx < len(self.raw_path):
            return self.raw_path[self.path_idx][0]
        return self.raw_path[-1][0] if self.raw_path else ''

    @property
    def node_path(self) -> List[str]:
        """시각화용 node_id 경로 (S 상태 기반, 연속 중복 제거)."""
        nodes: List[str] = []
        for sid, _ in self.raw_path:
            if sid.startswith('S,'):
                nid = sid.split(',')[1]
                if not nodes or nodes[-1] != nid:
                    nodes.append(nid)
        return nodes


# ── TAPG 환경 ─────────────────────────────────────────────────────────────────

class TAPGEnvironment:
    """
    TAPG 기반 DES 실행 환경.

    setup(raw_paths, agent_ids, colors) 으로 초기화하고
    step(sim_time) 을 매 프레임 호출합니다.
    """

    PERIODIC_INTERVAL = 0.5   # sim-seconds: WAITING 에이전트 재확인 주기

    def __init__(self, graph: PklMapGraph,
                 accel: float = INF,
                 decel: float = INF):
        """
        Parameters
        ----------
        accel : mm/s²  가속도. INF → 즉시 최고속도(상수 속도 모드)
        decel : mm/s²  감속도. INF → 즉시 정지(위치 도달 순간 스냅)
        """
        self.graph           = graph
        self.accel           = accel
        self.decel           = decel
        self._eq: list       = []
        self._seq            = 0
        self.agents:         Dict[int, TAPGAgent]  = {}
        self.sim_time        = 0.0
        self._last_check     = 0.0
        self.event_count     = 0   # 처리한 이벤트 누적 (벤치마크용)

        # TAPG
        self.G:               nx.DiGraph            = nx.DiGraph()
        self.wait_queues:     Dict[Any, List[int]]  = {}  # tapg_node -> [agent_id]

        # Coarse mode (송도용) - 시간 기반 cross-edge 비활성화, live occupancy
        # 검사 + cut node admission rule 적용.
        self._coarse_mode: bool = False
        self._cut_nodes: set = set()
        self._cut_to_port: dict = {}
        self._checkpoints: set = set()   # branching ∪ ports ∪ sidings
        self._rest_places: set = set()   # ports ∪ sidings (= 정차 가능 지점)
        # Debug log toggle. True 면 [CLAIM-OK/DENY/CUT/REVOKE/BUG/OVERLAP] print.
        # 매 claim 시도마다 출력되어 sim 속도에 큰 영향. 기본 False.
        self._coarse_debug: bool = False

    def _add_edge(self, u, v):
        """self-loop 방지 래퍼. DAG에 자기 참조 엣지는 절대 추가하지 않는다."""
        if u == v:
            import traceback
            print(f'  [TAPG] self-loop BLOCKED: {u}')
            traceback.print_stack(limit=4)
            return
        self.G.add_edge(u, v)

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def setup(self, agents: list, t_start: float = 0.0):
        """
        agents: 외부에서 생성한 TAPGAgent 객체 목록 (이 객체를 직접 업데이트함).
        같은 객체를 시각화쪽에서 참조해야 위치/상태가 반영된다.
        """
        self.agents = {a.id: a for a in agents}
        self.wait_queues.clear()
        self.G.clear()
        self._eq        = []
        self._seq       = 0
        self.sim_time   = t_start
        self._last_check = t_start

        raw_paths = [a.raw_path for a in agents]
        agent_ids = [a.id      for a in agents]

        # 에이전트 상태 초기화 + 시작 위치 설정
        for a in agents:
            a.path_idx = 0
            a.claim_idx = 0
            a.state    = IDLE
            a.v        = 0.0
            if a.raw_path:
                nid = self._node_of(a.raw_path[0][0])
                if nid and nid in self.graph.nodes:
                    n = self.graph.nodes[nid]
                    a.x, a.y = n.x, n.y
                a.theta = self._heading_of(a.raw_path[0][0])

        self._build_tapg(raw_paths, agent_ids)

        for aid in agent_ids:
            self._schedule(t_start, 'TRY_ADVANCE', aid)

    # ── TAPG 구성 ─────────────────────────────────────────────────────────────

    def _build_tapg(self, raw_paths: list, agent_ids: list):
        """
        Automod prioritized solver의 construct_temporal_graph 와 동일한 논리.

        노드 키: (state_id, agv_id, cbs_start_time)
        """
        # ① 모든 상태 노드 + 같은 에이전트 내 순차 엣지
        for aid, path in zip(agent_ids, raw_paths):
            for k, (sid, t) in enumerate(path):
                nk = self._nk(sid, aid, t)
                duration = (float('inf') if k == len(path) - 1
                            else path[k + 1][1] - t)
                self.G.add_node(nk, agv_id=aid, start_time=t, duration=duration)
                if k > 0:
                    prev_sid, prev_t = path[k - 1]
                    self._add_edge(self._nk(prev_sid, aid, prev_t), nk)

        # ② 교차-에이전트 엣지 (S/M/R 모두 포함)
        # Coarse mode 에선 time-based cross-edge 생성 건너뜀. claim 시점에
        # *live occupancy* 검사로 대체 (_is_claimable_coarse).
        if self._coarse_mode:
            return

        for i, (ai, pi) in enumerate(zip(agent_ids, raw_paths)):
            for j, (aj, pj) in enumerate(zip(agent_ids, raw_paths)):
                if i == j:
                    continue
                for k1 in range(len(pi) - 1, -1, -1):
                    s1, t1 = pi[k1]
                    affect1 = self._state_affect_set(s1)
                    if not affect1:
                        continue

                    for k2, (s2, t2) in enumerate(pj):
                        if t2 <= t1:
                            continue
                        affect2 = self._state_affect_set(s2)
                        if s2 in affect1 or s1 in affect2:
                            self._add_edge(self._nk(s1, ai, t1), self._nk(s2, aj, t2))
                            break

    # ── 메인 루프 ──────────────────────────────────────────────────────────────

    def step(self, sim_time: float):
        dt            = max(0.0, sim_time - self.sim_time)
        self.sim_time = sim_time

        # TRY_ADVANCE 등 예약 이벤트 처리
        while self._eq and self._eq[0].time <= sim_time:
            ev = heapq.heappop(self._eq)
            self.event_count += 1
            self._process(ev)

        # 위치 기반 이동/도달 처리 (센서 시뮬레이션)
        self._update_positions(dt, sim_time)

        # 주기적으로 WAITING 에이전트 재확인 (event miss 방어)
        if sim_time - self._last_check >= self.PERIODIC_INTERVAL:
            self._last_check = sim_time
            self._periodic_wakeup(sim_time)

        # Coarse mode: physical overlap detector only (revoke 제거 - claim 은
        # immutable, 다른 AGV 가 기존 claim 참고해 less claim 해야 함).
        if self._coarse_mode:
            if sim_time - getattr(self, '_last_overlap_check', 0) >= 1.0:
                self._last_overlap_check = sim_time
                self._verify_no_physical_overlap(sim_time)
                if self._coarse_debug:
                    for a in self.agents.values():
                        if a.state == MOVING and a.claim_idx <= a.path_idx:
                            print(f'  [BUG] V{a.id-100} MOVING but claim_idx={a.claim_idx} '
                                  f'<= path_idx={a.path_idx}. sid={a.raw_path[a.path_idx][0]}')

    # ── 이벤트 처리 ────────────────────────────────────────────────────────────

    def _process(self, ev: Event):
        handler = {
            'TRY_ADVANCE': self._on_try_advance,
            'LOAD_DONE':   self._on_load_done,
        }.get(ev.kind)
        if handler:
            handler(ev)

    def _on_load_done(self, ev: Event):
        """L state (LOADING/UNLOADING) dwell 완료. path_idx 를 L 너머로 advance.

        L 노드를 G 에서 제거 (대기자 깨우기 포함). 그 후 TRY_ADVANCE 즉시 fire
        해서 final_S 도달 처리 이어감.
        """
        agent = self.agents.get(ev.agent_id)
        if agent is None or agent.state == DONE:
            return
        # 현재 path_idx 가 L state 인지 확인
        if agent.path_idx >= len(agent.raw_path):
            return
        sid, cbs_t = agent.raw_path[agent.path_idx]
        if not sid.startswith('L,'):
            # 다른 이벤트가 먼저 path_idx 를 advance 시켰을 수 있음 - 안전하게 skip
            self._schedule(ev.time, 'TRY_ADVANCE', agent.id)
            return
        # L 노드 G 에서 제거 + wait_queue 깨우기
        nk = self._nk(sid, agent.id, cbs_t)
        if self.G.has_node(nk):
            for wid in self.wait_queues.pop(nk, []):
                wa = self.agents.get(wid)
                if wa and wa.state == WAITING:
                    wa.state = IDLE
                    self._schedule(ev.time + 1e-9, 'TRY_ADVANCE', wid)
            self.G.remove_node(nk)
        agent.path_idx += 1
        agent.state = IDLE
        self._schedule(ev.time, 'TRY_ADVANCE', agent.id)
        # Coarse mode: LOAD 완료 -> port 점유 풀림. 다른 WAITING agent 깨우기.
        # _complete_node 의 broadcast 와 동일 mechanism.
        if self._coarse_mode:
            for a in self.agents.values():
                if a.id != agent.id and a.state == WAITING:
                    a.state = IDLE
                    self._schedule(ev.time + 1e-9, 'TRY_ADVANCE', a.id)

    def _on_try_advance(self, ev: Event):
        agent = self.agents.get(ev.agent_id)
        if agent is None or agent.state in (MOVING, ROTATING, DONE):
            return

        # S 상태는 연속으로 즉시 통과
        while (agent.path_idx < len(agent.raw_path)
               and agent.raw_path[agent.path_idx][0].startswith('S,')):
            sid = agent.raw_path[agent.path_idx][0]
            nid = self._node_of(sid)
            if nid and nid in self.graph.nodes:
                n = self.graph.nodes[nid]
                agent.x, agent.y = n.x, n.y
            agent.theta = self._heading_of(sid)

            # 다음이 M/R이면: claimed 범위 내면 무조건 통과, 아니면 체크
            next_idx = agent.path_idx + 1
            if next_idx < len(agent.raw_path):
                next_sid, next_t = agent.raw_path[next_idx]
                if not next_sid.startswith('S,'):
                    if next_idx < agent.claim_idx:
                        pass  # 이미 claimed → 무조건 통과
                    else:
                        next_nk = self._nk(next_sid, agent.id, next_t)
                        if not self._is_claimable(next_nk, agent.id):
                            break
                        if not self._try_claim_next(agent):
                            break

            agent.path_idx += 1

        if agent.path_idx >= len(agent.raw_path):
            agent.state = DONE
            agent.v     = 0.0
            return

        sid, cbs_t = agent.raw_path[agent.path_idx]
        nk = self._nk(sid, agent.id, cbs_t)

        # L state (MCS LOADING/UNLOADING) - cost(dwell_time) 만큼 정지 후 진행.
        # path_idx 는 유지 (현재 L state 에 있음). LOAD_DONE 이벤트가 cost 시점에
        # fire 하면 path_idx 를 advance + 다음 state 처리.
        if sid.startswith('L,'):
            l_state = self._get_state_obj(sid)
            cost = l_state.cost if l_state else 0.0
            agent.state = WAITING  # LOADING/UNLOADING 동안 idle 표현
            agent.v     = 0.0
            self._schedule(ev.time + cost, 'LOAD_DONE', agent.id)
            return

        # S state에서 대기 → WAITING 등록
        # 차단 노드는 path_idx+1 이 아닐 수도 있음 (M 자체는 clear 인데
        # 그 destination S 가 막혀서 claim 실패한 경우 등). 앞으로 walk 하면서
        # 첫 cross-pred 가 있는 노드를 찾아 그 노드의 preds 에 등록.
        if sid.startswith('S,'):
            for fwd_idx in range(agent.path_idx + 1,
                                 min(len(agent.raw_path), agent.path_idx + 10)):
                fs_sid, fs_t = agent.raw_path[fwd_idx]
                fs_nk = self._nk(fs_sid, agent.id, fs_t)
                if fs_nk in self.G:
                    cross_preds = [p for p in self.G.predecessors(fs_nk)
                                   if p[1] != agent.id]
                    if cross_preds:
                        for pred in cross_preds:
                            q = self.wait_queues.setdefault(pred, [])
                            if agent.id not in q:
                                q.append(agent.id)
                        break
            if agent.state != WAITING:
                agent._wait_start_t = ev.time
            agent.state = WAITING
            agent.v     = 0.0
            return

        # Claimed 범위 내면 무조건 실행
        if agent.path_idx < agent.claim_idx:
            if sid.startswith('M,'):
                self._start_move(agent, sid, cbs_t, ev.time)
            elif sid.startswith('R,'):
                self._start_rotate(agent, sid, cbs_t, ev.time)
            else:
                self._complete_node(nk, ev.time)
                agent.path_idx += 1
                self._schedule(ev.time, 'TRY_ADVANCE', agent.id)
            return

        # Claimed 범위 밖: claimable 체크 + claim 확장
        if not self._is_claimable(nk, agent.id):
            if agent.state != WAITING:
                agent._wait_start_t = ev.time
            agent.state = WAITING
            agent.v     = 0.0
            for pred in self.G.predecessors(nk):
                if pred[1] != agent.id:
                    q = self.wait_queues.setdefault(pred, [])
                    if agent.id not in q:
                        q.append(agent.id)
            return

        if not self._try_claim_next(agent):
            if agent.state != WAITING:
                agent._wait_start_t = ev.time
            agent.state = WAITING
            agent.v     = 0.0
            return

        # 액션 실행
        if sid.startswith('M,'):
            self._start_move(agent, sid, cbs_t, ev.time)
        elif sid.startswith('R,'):
            self._start_rotate(agent, sid, cbs_t, ev.time)
        else:
            # 알 수 없는 상태: 건너뜀
            self._complete_node(nk, ev.time)
            agent.path_idx += 1
            self._schedule(ev.time, 'TRY_ADVANCE', agent.id)

    def _start_move(self, agent: TAPGAgent, sid: str,
                    cbs_t: float, cur_time: float,
                    v_init: float = None):
        """
        v_init: 연속 주행 체이닝 시 이전 엣지에서 이어받는 속도.
                None이면 가속도 설정에 따라 0 또는 max_speed에서 출발.
        """
        parts        = sid.split(',')
        from_n, to_n = parts[1], parts[2]
        edge         = self.graph.get_edge(from_n, to_n)
        if edge is None:
            nk = self._nk(sid, agent.id, cbs_t)
            self._complete_node(nk, cur_time)
            agent.path_idx += 1
            self._schedule(cur_time, 'TRY_ADVANCE', agent.id)
            return

        fn = self.graph.nodes[from_n]
        tn = self.graph.nodes[to_n]

        agent.from_x        = fn.x
        agent.from_y        = fn.y
        agent.to_x          = tn.x
        agent.to_y          = tn.y
        # Heading: M 상태의 start_id (=정의된 facing 방향) 우선.
        # edge.angle (물리 atan2) 은 레이아웃의 reverse/sideways override 를
        # 반영하지 못하므로 fallback 으로만 사용.
        m_state = (self.graph.move_states_raw.get(sid)
                   if hasattr(self.graph, 'move_states_raw') else None)
        if m_state and getattr(m_state, 'start_id', None):
            agent.theta = self._heading_of(m_state.start_id)
        else:
            agent.theta = edge.angle
        agent.max_speed     = edge.max_speed
        if v_init is not None:
            # 연속 주행: 이전 속도 유지 (max_speed 초과 방지)
            agent.v = min(v_init, edge.max_speed)
        elif math.isfinite(self.accel):
            agent.v = 0.0           # 유한 가속도: 정지 상태에서 출발
        else:
            agent.v = edge.max_speed  # 즉시 최고속도
        agent.dist_traveled = 0.0
        agent.edge_length   = edge.length
        agent._tapg_node    = self._nk(sid, agent.id, cbs_t)
        agent.state         = MOVING
        agent._priority     = 0   # MOVING 시 cycle-victim priority reset
        # ACTION_DONE 이벤트 없음 - _update_positions에서 위치 기반으로 도달 감지

    def _start_rotate(self, agent: TAPGAgent, sid: str,
                      cbs_t: float, cur_time: float):
        parts    = sid.split(',')
        from_deg = float(parts[2])
        to_deg   = float(parts[3])

        # 최단 경로 방향의 부호 있는 각도 계산
        # CCW(+) vs CW(-) 중 짧은 쪽을 선택
        diff_ccw = (to_deg - from_deg) % 360   # 반시계 방향으로 얼마나 돌아야 하는지
        if diff_ccw <= 180:
            signed_diff = math.radians(diff_ccw)   # CCW (+)
        else:
            signed_diff = math.radians(diff_ccw - 360)  # CW (-)

        from_rad = math.radians(from_deg)

        agent.from_theta      = from_rad
        agent.to_theta        = from_rad + signed_diff  # 최단 경로 기준 도달 각도
        agent.theta           = from_rad
        agent.v               = 0.0
        agent.angle_traversed = 0.0
        agent.angle_total     = abs(signed_diff)
        agent._tapg_node      = self._nk(sid, agent.id, cbs_t)
        agent.state           = ROTATING
        agent._priority       = 0   # ROTATING 시 cycle-victim priority reset
        # ACTION_DONE 이벤트 없음 - _update_positions에서 각도 기반으로 완료 감지

    # ── 주기적 재확인 ──────────────────────────────────────────────────────────

    def _periodic_wakeup(self, sim_time: float):
        """
        WAITING 에이전트 중 선행 조건이 해소된 에이전트를 깨웁니다.
        event-driven wakeup이 누락되는 경우를 방어합니다.
        """
        for agent in list(self.agents.values()):
            if agent.state != WAITING:
                continue
            if agent.path_idx >= len(agent.raw_path):
                agent.state = DONE
                continue
            # 현재 state 부터 다음 non-S state 까지 walk 하면서 전부
            # claimable 해야 진행 가능. 중간에 하나라도 막혀있으면 WAITING 유지.
            # (단순히 현재 S 만 claimable 체크하면 다음 M 이 막혀있어도 IDLE 로
            #  전이시켜 _on_try_advance ↔ _periodic_wakeup 토글 deadlock 발생.)
            idx = agent.path_idx
            can_progress = False
            while idx < len(agent.raw_path):
                sid, cbs_t = agent.raw_path[idx]
                nk = self._nk(sid, agent.id, cbs_t)
                if not self._is_claimable(nk, agent.id):
                    break
                if not sid.startswith('S,'):
                    can_progress = True
                    break
                idx += 1
            else:
                # 끝까지 모두 S 였고 모두 claimable → 종료 가능
                can_progress = True
            if can_progress:
                agent.state = IDLE
                self._schedule(sim_time, 'TRY_ADVANCE', agent.id)

    # ── 완료 처리 ──────────────────────────────────────────────────────────────

    def _complete_node(self, nk: tuple, cur_time: float):
        """TAPG 노드를 DAG에서 제거하고 대기 중인 에이전트를 깨웁니다."""
        # 대기 중인 agent 깨우기 (제거 전에 처리)
        waiters = self.wait_queues.pop(nk, [])

        # DAG에서 노드 제거 → 후행 노드의 in-edge가 자동으로 사라짐
        if self.G.has_node(nk):
            self.G.remove_node(nk)

        for wid in waiters:
            wa = self.agents.get(wid)
            if wa and wa.state == WAITING:
                wa.state = IDLE
                wa._wait_start_t = None
                self._schedule(cur_time + 1e-9, 'TRY_ADVANCE', wid)

        # Coarse mode: claim 영역 변화 → 모든 WAITING agent 즉시 retry.
        # FIFO 순서 (_wait_start_t 오름차순) — 먼저 WAITING 진입한 AGV 가 먼저
        # claim 시도. agents.values() 의 id 순보다 공정.
        # (wait_queue 메커니즘은 cross-edge 기반이라 coarse 에선 무효)
        if self._coarse_mode:
            waiting = [a for a in self.agents.values()
                       if a.state == WAITING]
            # 정렬 key: (_priority, _wait_start_t). priority 작을수록 우선
            # (cycle victim = -1), 동일하면 먼저 WAITING 진입한 게 우선.
            waiting.sort(key=lambda a: (
                a._priority,
                a._wait_start_t if a._wait_start_t is not None
                else float('inf')))
            for a in waiting:
                a.state = IDLE
                a._wait_start_t = None
                self._schedule(cur_time + 1e-9, 'TRY_ADVANCE', a.id)

    # ── TAPG 클레임 확인 ───────────────────────────────────────────────────────

    def _is_claimable(self, nk: tuple, agent_id: int) -> bool:
        """다른 에이전트의 선행 TAPG 노드가 DAG에 없으면(제거됨=완료) True.
        Coarse mode 에선 *live occupancy* 검사 - 다른 AGV 의 현재 claim 영역
        과 *물리적 overlap (affect_state)* 검사.
        """
        if self._coarse_mode:
            return self._is_claimable_coarse(nk, agent_id)
        if nk not in self.G:
            return True
        for pred in self.G.predecessors(nk):
            if pred[1] != agent_id:
                return False
        return True

    def _state_occupied_nodes(self, sid: str, executing: bool = True) -> set:
        """state 가 *물리적으로 점유* 하는 node 집합 (heading-independent).

        executing: AGV 가 이 state 를 *실제 실행 중* 인지.
        - True (= state MOVING/ROTATING): M -> {source, destination} 양 endpoint
        - False (= state IDLE 이지만 path[path_idx]=M, 미시작): M -> {source} 만

        S/R/L: 항상 단일 node.
        """
        parts = sid.split(',')
        if parts[0] == 'M' and len(parts) >= 3:
            if executing:
                return {parts[1], parts[2]}
            return {parts[1]}   # idle at source
        return {parts[1]} if len(parts) >= 2 else set()

    def _is_claimable_coarse(self, nk: tuple, agent_id: int) -> bool:
        """Coarse: 다른 AGV 가 *현재* claim 한 [path_idx, claim_idx) 범위 의
        states 중 nk 의 sid 와 *node-level overlap* 있으면 차단.

        Node-level (heading-independent) - affect_state 의 heading 누락으로
        놓치던 conflict 도 잡음.

        Release-timing 보정: AGV 의 path_idx 가 M/R 인 경우 *직전 S* 도 검사.
        S walk 가 path_idx 를 *S 떠나기 전에* M idx 로 advance 시키므로 그
        짧은 구간 동안 S 의 물리적 점유가 range 에서 빠지는 것 방지.
        """
        sid = nk[0]
        my_nodes = self._state_occupied_nodes(sid)
        my_affect = self._state_affect_set(sid) | {sid}
        for other in self.agents.values():
            if other.id == agent_id:
                continue
            if not other.raw_path:
                continue
            # DONE AGV (path_idx >= len): path[-1] (= 물리적 last 위치) 점유.
            # 안 잡으면 DONE 이 invisible -> 다른 AGV 가 그 자리로 entry 시도.
            if other.path_idx >= len(other.raw_path):
                last_sid = other.raw_path[-1][0]
                last_nodes = self._state_occupied_nodes(last_sid,
                                                        executing=False)
                if my_nodes & last_nodes:
                    self._last_block_info = (
                        f'V{agent_id-100}@{sid} blocked by DONE '
                        f'V{other.id-100}@{last_sid} '
                        f'(node overlap: {my_nodes & last_nodes})')
                    return False
                continue
            # 기본 범위
            lo = other.path_idx
            hi = max(other.claim_idx, other.path_idx + 1)
            hi = min(hi, len(other.raw_path))
            # 보정: path_idx 가 non-S (M/R) 이면 직전 S 도 포함
            # (AGV 가 물리적으로 그 S 에 있을 가능성)
            if (other.path_idx > 0
                    and other.path_idx < len(other.raw_path)
                    and not other.raw_path[other.path_idx][0].startswith('S,')):
                lo = other.path_idx - 1
            for k in range(lo, hi):
                other_sid = other.raw_path[k][0]
                # Other 의 state 별 executing 판단:
                #   k == path_idx: state 에 따라 (IDLE 면 미출발, source 만)
                #   k != path_idx: future claim = atomic, both endpoints
                if k == other.path_idx:
                    is_exec = other.state in (MOVING, ROTATING)
                else:
                    is_exec = True
                # 1. Node-level overlap (heading-independent, primary)
                other_nodes = self._state_occupied_nodes(other_sid, is_exec)
                if my_nodes & other_nodes:
                    self._last_block_info = (
                        f'V{agent_id-100}@{sid} blocked by '
                        f'V{other.id-100}@{other_sid}[idx={k}] '
                        f'(node overlap: {my_nodes & other_nodes})')
                    return False
                # 2. Affect_state secondary check.
                # 단, IDLE AGV 의 미출발 M state 는 affect 검사 skip
                # (= AGV 가 물리적으로 source 에 있으므로 M 의 swept affect 무관)
                if (k == other.path_idx
                        and other.state in (IDLE, WAITING)
                        and other_sid.startswith('M,')):
                    continue   # source only via node check; affect skip
                if other_sid in my_affect:
                    self._last_block_info = (
                        f'V{agent_id-100}@{sid} blocked by '
                        f'V{other.id-100}@{other_sid}[idx={k}] '
                        f'(other_sid in my_affect)')
                    return False
                other_affect = self._state_affect_set(other_sid)
                if sid in other_affect:
                    self._last_block_info = (
                        f'V{agent_id-100}@{sid} blocked by '
                        f'V{other.id-100}@{other_sid}[idx={k}] '
                        f'(my_sid in other_affect)')
                    return False
        return True

    def _verify_no_physical_overlap(self, sim_time: float):
        """모든 AGV 의 *현재 state* 가 서로 affect_state overlap 없는지 검사.
        Overlap 발견 시 즉시 print (collision detection 전 단계 디버그)."""
        states = []
        for a in self.agents.values():
            if a.path_idx < len(a.raw_path):
                states.append((a.id, a.raw_path[a.path_idx][0]))
        for i in range(len(states)):
            id1, sid1 = states[i]
            aff1 = self._state_affect_set(sid1)
            for j in range(i+1, len(states)):
                id2, sid2 = states[j]
                aff2 = self._state_affect_set(sid2)
                if sid2 in aff1 or sid1 in aff2:
                    if self._coarse_debug:
                        print(f'  [OVERLAP-BUG] t={sim_time:.2f} '
                              f'V{id1-100}@{sid1} ↔ V{id2-100}@{sid2}')
                    return True
        return False

    def _would_claim_create_cycle(self, agent: 'TAPGAgent',
                                    new_end_idx: int) -> bool:
        """가상으로 agent 가 [claim_idx, new_end_idx) 를 claim 했을 때
        다른 AGV 의 _wanting 과 묶여 cycle 형성하는지 확인.

        Cycle 조건:
          - agent 의 new _wanting = path[new_end_idx]'s dest node
          - 다른 AGV B 의 _wanting 이 agent 가 점유한 (또는 점유할) node
          - 따라가는 chain 이 agent 로 돌아오면 cycle

        dep_map: aid -> blocker_aid (= aid 가 다음 진입하려는 node 의 현재 owner)
        """
        if not self._coarse_mode:
            return False
        # 1. agent 의 hypothetical _wanting = new claim end 다음 state 의 dest node
        path = agent.raw_path
        hypo_wanting = None
        if new_end_idx < len(path):
            sid = path[new_end_idx][0]
            hypo_wanting = self._dest_node_of(sid)
        if hypo_wanting is None:
            return False   # path 끝 - 더 wanting 없음
        # 2. 모든 AGV 의 점유 node 맵 (= node -> owner aid)
        #    Hypothetical: agent 의 점유 범위 = [path_idx, new_end_idx)
        node_owner = {}
        for other in self.agents.values():
            if other.id == agent.id:
                # agent 의 가상 점유 범위
                lo, hi = other.path_idx, new_end_idx
            else:
                lo = other.path_idx
                hi = max(other.claim_idx, other.path_idx + 1)
                hi = min(hi, len(other.raw_path))
            for k in range(lo, hi):
                if k >= len(other.raw_path):
                    break
                s = other.raw_path[k][0]
                if ',' in s:
                    n = s.split(',')[1]
                    if n not in node_owner:
                        node_owner[n] = other.id
        # 3. dep_map: AGV → (그 AGV 의 wanting node 점유 owner)
        #    Wanting = path[claim_idx] 의 dest node (= 다음 claim 시도 대상).
        #    _wanting attribute 는 dry-run 시점에 stale 할 수 있으므로 path 에서
        #    직접 derive.
        dep = {}
        for other in self.agents.values():
            if other.id == agent.id:
                w = hypo_wanting
            else:
                # other 의 next claim 시도 대상 = path[claim_idx]'s dest
                if (other.raw_path
                        and other.claim_idx < len(other.raw_path)):
                    next_sid = other.raw_path[other.claim_idx][0]
                    w = self._dest_node_of(next_sid)
                else:
                    w = None
            if w and w in node_owner:
                own = node_owner[w]
                if own != other.id:
                    dep[other.id] = own
        # 4. agent 에서 시작해 chain 따라가서 agent 로 돌아오면 cycle.
        #    Cycle 일 경우 first blocker (= agent 가 직접 의존하는 AGV) 를
        #    _last_block_info 에 기록 -> waiter loop 의 regex 가 추출 가능.
        first_blocker = dep.get(agent.id)
        visited = set()
        cur = agent.id
        while cur in dep:
            if cur in visited:
                if cur == agent.id and first_blocker is not None:
                    self._set_cycle_block_info(agent, first_blocker,
                                                hypo_wanting)
                return cur == agent.id
            visited.add(cur)
            cur = dep[cur]
            if cur == agent.id:
                if first_blocker is not None:
                    self._set_cycle_block_info(agent, first_blocker,
                                                hypo_wanting)
                return True
        return False

    def _set_cycle_block_info(self, agent: 'TAPGAgent',
                                blocker_aid: int, want_node: str):
        """Cycle 일 때 _last_block_info 에 blocker 정보 기록.
        Waiter loop 의 regex `blocked by V<n>@<sid>` 와 호환."""
        # blocker agent 의 현재 위치 sid (= 첫 path idx)
        blocker = self.agents.get(blocker_aid)
        if blocker and blocker.raw_path and blocker.path_idx < len(blocker.raw_path):
            b_sid = blocker.raw_path[blocker.path_idx][0]
        else:
            b_sid = f'S,{want_node},0'   # fallback
        self._last_block_info = (
            f'V{agent.id-100}@{want_node} blocked by V{blocker_aid-100}'
            f'@{b_sid}[cycle] (would create cycle)')

    def _try_claim_next(self, agent: TAPGAgent) -> bool:
        """다음 M/R action까지 claim 확장 시도.
        TAPG claimable 체크를 통과하면 claim 확장.

        중요: M/R 직후 destination S 도 claimable 해야 claim 진행.
        그렇지 않으면 M 실행 후 차량이 destination 에 도착했지만 S 진입이
        막혀, 물리적으로 destination 위치에 머물면서 다른 agent 와 충돌 가능.

        Coarse mode 추가 규칙: 다음 노드가 cut node (port 종속) 면 admission
        rule - port 까지 atomic claim OR cut zone 통과 claim 둘 중 하나 만족.
        """
        target_idx = agent.claim_idx
        while target_idx < len(agent.raw_path):
            sid = agent.raw_path[target_idx][0]
            if sid.startswith('M,') or sid.startswith('R,'):
                # Coarse mode: cut node admission rule 검사
                if self._coarse_mode:
                    dest_node = self._dest_node_of(sid)
                    if dest_node in self._cut_nodes:
                        end_idx = self._cut_admission_end_idx(agent, target_idx)
                        port = self._cut_to_port.get(dest_node, '?')
                        if end_idx is None:
                            if self._coarse_debug:
                                print(f'  [CUT-DENY] V{agent.id-100} → cut={dest_node} '
                                      f'(port={port}) - neither port-claim nor through-claim'
                                      f' available')
                            return False
                        ok = self._claim_range(agent, target_idx, end_idx)
                        if not ok:
                            if self._coarse_debug:
                                blk = getattr(self, '_last_block_info', '?')
                                print(f'  [CUT-DENY] V{agent.id-100} → cut={dest_node} '
                                      f'(port={port}) range [{target_idx},{end_idx}) - {blk}')
                            # Wanting 기록 (= 시도했다가 막힌 node)
                            agent._wanting = self._dest_node_of(
                                agent.raw_path[target_idx][0])
                            return False
                        # Cycle 발생 여부 확인 - claim 으로 인해 cycle 형성하면 거부
                        if self._would_claim_create_cycle(agent, end_idx):
                            if self._coarse_debug:
                                print(f'  [CUT-CYCLE-DENY] V{agent.id-100} → cut={dest_node} '
                                      f'(port={port}) would create dep cycle')
                            agent._wanting = self._dest_node_of(
                                agent.raw_path[target_idx][0])
                            return False
                        if self._coarse_debug:
                            print(f'  [CUT-OK] V{agent.id-100} → cut={dest_node} '
                                  f'(port={port}) claim [{target_idx},{end_idx})')
                        agent.claim_idx = end_idx
                        # Claim 성공 - wanting 갱신 (next node beyond claim)
                        if end_idx < len(agent.raw_path):
                            agent._wanting = self._dest_node_of(
                                agent.raw_path[end_idx][0])
                        else:
                            agent._wanting = None
                        return True

                # 일반 claim
                # Coarse mode: *next checkpoint 까지 atomic claim* (= 회색
                # corridor 끝까지 lock). SIPP mode: 기존 M/R + 연속 S 1단계.
                if self._coarse_mode and self._checkpoints:
                    end_idx = self._coarse_segment_end(agent, target_idx)
                else:
                    end_idx = target_idx + 1
                    while end_idx < len(agent.raw_path) and agent.raw_path[end_idx][0].startswith('S,'):
                        end_idx += 1

                if not self._claim_range_test(agent, target_idx, end_idx):
                    if self._coarse_mode and self._coarse_debug:
                        blk = getattr(self, '_last_block_info', '?')
                        print(f'  [CLAIM-DENY] V{agent.id-100} range=[{target_idx},{end_idx}) - {blk}')
                    if self._coarse_mode:
                        agent._wanting = self._dest_node_of(
                            agent.raw_path[target_idx][0])
                    return False

                # Cycle 발생 여부 확인 (coarse mode 만)
                if self._coarse_mode and self._would_claim_create_cycle(agent, end_idx):
                    if self._coarse_debug:
                        print(f'  [CLAIM-CYCLE-DENY] V{agent.id-100} sid={sid} '
                              f'range=[{target_idx},{end_idx}) would create dep cycle')
                    agent._wanting = self._dest_node_of(
                        agent.raw_path[target_idx][0])
                    return False

                if self._coarse_mode and self._coarse_debug:
                    print(f'  [CLAIM-OK] V{agent.id-100} sid={sid} '
                          f'range=[{target_idx},{end_idx})')
                agent.claim_idx = end_idx
                # Claim 성공 - wanting 갱신
                if self._coarse_mode:
                    if end_idx < len(agent.raw_path):
                        agent._wanting = self._dest_node_of(
                            agent.raw_path[end_idx][0])
                    else:
                        agent._wanting = None
                return True
            target_idx += 1

        return False

    def _cut_admission_end_idx(self, agent: TAPGAgent, target_idx: int):
        """Cut node 진입 시 claim 끝 idx 결정.

        (a) target_idx 부터 *port S 직후* 까지 (= port 도달 + L + post-S):
            claim 가능하면 end_idx 반환
        (b) target_idx 부터 *cut zone 벗어난 다음 회색 corridor S* 까지:
            claim 가능하면 end_idx 반환
        둘 다 못 하면 None.
        """
        path = agent.raw_path
        n = len(path)
        ports_set = (self.graph.ports.values() if self.graph.ports else set())
        # (a) port S 까지 walk. M state 는 *destination* node 가 port/cut 인지 봐야 함.
        # _dest_node_of: M -> parts[2] (도달지), S/R/L -> parts[1].
        port_end_idx = None
        for k in range(target_idx, n):
            sid = path[k][0]
            node = self._dest_node_of(sid)
            if node and node in ports_set:
                # port found - walk 좀 더 (L + post-S 같은 node)
                end = k + 1
                while end < n and (path[end][0].startswith('L,') or path[end][0].startswith('S,')):
                    end_node = self._dest_node_of(path[end][0])
                    if end_node != node:
                        break
                    end += 1
                port_end_idx = end
                break
            # cut node 아니거나 port 아닌데 도달 못 함
            if node and node not in self._cut_nodes and not (k == target_idx):
                # cut zone 벗어남
                break
        if port_end_idx and self._claim_range_test(agent, target_idx, port_end_idx):
            return port_end_idx
        # (b) cut zone 벗어난 다음 corridor 노드 까지
        post_cut_idx = None
        for k in range(target_idx, n):
            sid = path[k][0]
            node = self._dest_node_of(sid)
            if node and node not in self._cut_nodes:
                # 이 노드는 cut zone 벗어남. S state 면 거기까지.
                if sid.startswith('S,'):
                    post_cut_idx = k + 1
                    break
        if post_cut_idx and self._claim_range_test(agent, target_idx, post_cut_idx):
            return post_cut_idx
        return None

    def _coarse_segment_end(self, agent: TAPGAgent, target_idx: int) -> int:
        """Coarse mode: target_idx 부터 *첫 정차 가능 지점* 까지 atomic claim.

        Cut node (= port 진입로) 만 stop 금지. Branching/rest/corridor 모두
        정차 가능. Cycle 발생 방지는 별도 cycle detection (push trigger 와
        cycle replan 단계) 에서 처리.
        """
        path = agent.raw_path
        n = len(path)
        end = target_idx + 1
        while end < n:
            sid = path[end][0]
            if sid.startswith('S,'):
                node = self._node_of(sid)
                if node in self._cut_nodes:
                    # Cut node (port 진입로) 만 통과 - port 차단 위험.
                    pass
                else:
                    return end + 1
            end += 1
        return n

    def _claim_range_test(self, agent: TAPGAgent, start: int, end: int) -> bool:
        """[start, end) 의 모든 state 가 claimable 한지 검사 (실제 claim 안 함)."""
        for k in range(start, end):
            sid, t = agent.raw_path[k]
            nk = self._nk(sid, agent.id, t)
            if not self._is_claimable(nk, agent.id):
                return False
        return True

    def _claim_range(self, agent: TAPGAgent, start: int, end: int) -> bool:
        """[start, end) 의 모든 state claim 시도. 모두 가능하면 True."""
        return self._claim_range_test(agent, start, end)

    # ── 위치 기반 이동 / 도달 감지 (센서 시뮬레이션) ──────────────────────────

    def _update_positions(self, dt: float, sim_time: float):
        """
        dt 동안 각 에이전트를 물리적으로 전진시키고,
        목적지 도달 여부를 거리/각도로 판단합니다.
        CBS 플래닝 타임과 무관하게 실제 이동량 기준으로 완료를 결정합니다.
        """
        if dt <= 0.0:
            return

        arrivals = []

        for agent in self.agents.values():
            if agent.state == MOVING:
                # 사다리꼴 속도 프로파일 (accel/decel이 유한할 때만 적용)
                d_rem   = agent.edge_length - agent.dist_traveled
                d_brake = (agent.v ** 2 / (2.0 * self.decel)
                           if math.isfinite(self.decel) else 0.0)
                if math.isfinite(self.decel) and d_rem <= d_brake + 1e-6:
                    # 제동 구간 - 하지만 다음 Move가 이미 claimable이면 감속 생략
                    if (math.isfinite(self.accel)
                            and self._next_move_claimable(agent)):
                        # 연속 주행: 감속 없이 통과
                        if agent.v < agent.max_speed:
                            agent.v = min(agent.max_speed,
                                          agent.v + self.accel * dt)
                    else:
                        agent.v = max(0.0, agent.v - self.decel * dt)
                elif math.isfinite(self.accel) and agent.v < agent.max_speed:
                    # 가속 구간
                    agent.v = min(agent.max_speed, agent.v + self.accel * dt)
                # else: 최고속도 유지 (cruise)

                agent.dist_traveled += agent.v * dt
                if agent.dist_traveled >= agent.edge_length - 1e-6:
                    # 도달 - 목적지로 스냅 (센서 트리거)
                    agent.x = agent.to_x
                    agent.y = agent.to_y
                    arrivals.append(agent)
                else:
                    frac    = agent.dist_traveled / agent.edge_length
                    agent.x = agent.from_x + (agent.to_x - agent.from_x) * frac
                    agent.y = agent.from_y + (agent.to_y - agent.from_y) * frac

            elif agent.state == ROTATING:
                agent.angle_traversed += ANGULAR_SPEED * dt
                if agent.angle_total <= 0 or agent.angle_traversed >= agent.angle_total:
                    # 회전 완료 - 목표 각도로 스냅
                    agent.theta = agent.to_theta
                    arrivals.append(agent)
                else:
                    frac        = agent.angle_traversed / agent.angle_total
                    agent.theta = (agent.from_theta
                                   + (agent.to_theta - agent.from_theta) * frac)

        for agent in arrivals:
            self._handle_arrival(agent, sim_time)

    def _handle_arrival(self, agent: TAPGAgent, sim_time: float):
        """에이전트가 목적지(또는 회전 목표)에 도달했을 때 호출됩니다.
        완료된 M/R action + 그 이전의 S state들을 DAG에서 한꺼번에 제거."""
        nk               = agent._tapg_node
        carry_v          = agent.v
        agent._tapg_node = None
        agent.path_idx  += 1
        agent.state      = IDLE

        # 현재 완료된 action(M/R) + 그 이전의 모든 S state를 DAG에서 제거
        if nk:
            # path에서 현재 path_idx 이전의 모든 state 제거
            for i in range(agent.path_idx):
                sid_i, t_i = agent.raw_path[i]
                nk_i = self._nk(sid_i, agent.id, t_i)
                if self.G.has_node(nk_i):
                    # wait_queues에서 대기자 깨우기
                    for wid in self.wait_queues.pop(nk_i, []):
                        wa = self.agents.get(wid)
                        if wa and wa.state == WAITING:
                            wa.state = IDLE
                            self._schedule(sim_time + 1e-9, 'TRY_ADVANCE', wid)
                    self.G.remove_node(nk_i)

        # Coarse mode: 도착 = 다른 AGV 의 claim 영역 변경. 모든 WAITING agent
        # 깨우기 (= _complete_node 의 broadcast 와 동일).
        if self._coarse_mode:
            for a in self.agents.values():
                if a.id != agent.id and a.state == WAITING:
                    a.state = IDLE
                    self._schedule(sim_time + 1e-9, 'TRY_ADVANCE', a.id)

        # 유한 가속도 모드 + Move 완료 시 연속 주행 시도
        if math.isfinite(self.accel) and carry_v > 0 and nk and nk[0].startswith('M,'):
            if self._try_chain_move(agent, carry_v, sim_time):
                return

        agent.v = 0.0
        self._schedule(sim_time, 'TRY_ADVANCE', agent.id)

    # ── 연속 주행 헬퍼 ────────────────────────────────────────────────────────

    def _peek_next_move(self, agent: TAPGAgent):
        """현재 path_idx 이후의 첫 번째 M 상태 (S 상태 건너뜀) 반환. 없으면 None."""
        idx = agent.path_idx + 1
        while idx < len(agent.raw_path):
            sid, t = agent.raw_path[idx]
            if sid.startswith('S,'):
                idx += 1
                continue
            return (sid, t) if sid.startswith('M,') else None
        return None

    def _next_move_claimable(self, agent: TAPGAgent) -> bool:
        """다음 M 상태가 이미 claimed이거나 claim 가능한지 확인."""
        nm = self._peek_next_move(agent)
        if nm is None:
            return False
        # 이미 claimed 범위 내면 OK
        nm_idx = None
        for i in range(agent.path_idx + 1, len(agent.raw_path)):
            if agent.raw_path[i][0] == nm[0] and abs(agent.raw_path[i][1] - nm[1]) < 1e-6:
                nm_idx = i
                break
        if nm_idx is not None and nm_idx < agent.claim_idx:
            return True
        # Claim 확장 시도
        return self._try_claim_next(agent)

    def _try_chain_move(self, agent: TAPGAgent,
                        carry_v: float, sim_time: float) -> bool:
        """
        도달 직후 S 상태를 inline으로 처리하고,
        다음 M이 claimable이면 멈추지 않고 바로 _start_move 진입.

        성공하면 True, 체이닝 불가(R 상태·claimable 아님·경로 끝)이면 False.

        Coarse mode invariant: *atomic claim 보장* 되기 전엔 path_idx advance
        금지. 즉 S walk 전에 next M 의 atomic claim 검증 먼저, 성공한 경우만
        path_idx 진행. 실패 시 path_idx 유지 (= AGV 가 *현재 S 에 있음*).
        """
        if self._coarse_mode:
            # 1) peek 후보 next M (S sequence 너머 첫 M)
            scan_idx = agent.path_idx
            while (scan_idx < len(agent.raw_path)
                   and agent.raw_path[scan_idx][0].startswith('S,')):
                scan_idx += 1
            if scan_idx >= len(agent.raw_path):
                agent.state = DONE
                agent.v = 0.0
                return True
            target_sid = agent.raw_path[scan_idx][0]
            if not target_sid.startswith('M,'):
                return False   # R state 등 - chain 안 함

            # 2) atomic claim 검증 (path_idx 는 *아직 advance 안 함*)
            saved_claim = agent.claim_idx
            if agent.claim_idx <= scan_idx:
                # 임시로 claim_idx 를 scan_idx 까지 set 후 _try_claim_next 호출
                agent.claim_idx = scan_idx
                if not self._try_claim_next(agent):
                    agent.claim_idx = saved_claim  # 복원
                    return False
            # claim 이 scan_idx 의 M 을 포함하는지
            if scan_idx >= agent.claim_idx:
                agent.claim_idx = saved_claim
                return False

            # 3) claim 성공 -> 이제 S walk + advance path_idx
            while (agent.path_idx < scan_idx):
                sid, t = agent.raw_path[agent.path_idx]
                nid = self._node_of(sid)
                if nid and nid in self.graph.nodes:
                    n = self.graph.nodes[nid]
                    agent.x, agent.y = n.x, n.y
                agent.theta = self._heading_of(sid)
                agent.path_idx += 1

            # 4) _start_move
            next_t = agent.raw_path[scan_idx][1]
            self._start_move(agent, target_sid, next_t, sim_time, v_init=carry_v)
            return True

        # SIPP mode: 기존 동작
        while (agent.path_idx < len(agent.raw_path)
               and agent.raw_path[agent.path_idx][0].startswith('S,')):
            sid, t = agent.raw_path[agent.path_idx]
            nid = self._node_of(sid)
            if nid and nid in self.graph.nodes:
                n = self.graph.nodes[nid]
                agent.x, agent.y = n.x, n.y
            agent.theta = self._heading_of(sid)

            next_idx = agent.path_idx + 1
            if next_idx < len(agent.raw_path):
                next_sid, next_t = agent.raw_path[next_idx]
                if not next_sid.startswith('S,'):
                    next_nk = self._nk(next_sid, agent.id, next_t)
                    if not self._is_claimable(next_nk, agent.id):
                        return False

            agent.path_idx += 1

        if agent.path_idx >= len(agent.raw_path):
            agent.state = DONE
            agent.v     = 0.0
            return True

        next_sid, next_t = agent.raw_path[agent.path_idx]
        next_nk = (next_sid, agent.id, next_t)

        if next_sid.startswith('M,') and self._is_claimable(next_nk, agent.id):
            self._start_move(agent, next_sid, next_t, sim_time, v_init=carry_v)
            return True

        return False

    # ── 헬퍼 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _round_t(t: float) -> float:
        """TAPG 노드 키의 시간을 반올림하여 부동소수점 불일치를 방지."""
        return round(t, 6)

    @staticmethod
    def _nk(sid: str, aid: int, t: float) -> tuple:
        """TAPG 노드 키 생성 - 시간을 반올림하여 부동소수점 불일치 방지."""
        return (sid, aid, round(t, 6))

    def _get_state_obj(self, state_id: str):
        # R 상태는 별도 dict 없이 Move_state에 같이 저장됨
        if state_id.startswith('M,') or state_id.startswith('R,'):
            return self.graph.move_states_raw.get(state_id)
        if state_id.startswith('S,'):
            return self.graph.stop_states_raw.get(state_id)
        if state_id.startswith('L,'):
            return getattr(self.graph, 'load_states_raw', {}).get(state_id)
        return None

    def _state_affect_set(self, state_id: str) -> set:
        """state_id의 affect_state set 반환. 호출 빈도가 높아 캐시."""
        cache = getattr(self, '_affect_set_cache', None)
        if cache is None:
            cache = {}
            self._affect_set_cache = cache
        s = cache.get(state_id)
        if s is None:
            obj = self._get_state_obj(state_id)
            s = frozenset(getattr(obj, 'affect_state', [])) if obj else frozenset()
            cache[state_id] = s
        return s

    def _effective_state_ids(self, state_id: str) -> list:
        """충돌 체크에서 이 state가 상대방 affect_set에 있는지 확인할 ID 목록."""
        return [state_id]

    @staticmethod
    def _node_of(state_id: str) -> Optional[str]:
        parts = state_id.split(',')
        return parts[1] if len(parts) >= 2 else None

    @staticmethod
    def _dest_node_of(state_id: str) -> Optional[str]:
        """state 가 '도달' 시 AGV 가 위치할 node.
        - M,from,to -> to (destination)
        - R,node,h1,h2 -> node
        - S,node,h -> node
        - L,node -> node
        """
        parts = state_id.split(',')
        if parts[0] == 'M' and len(parts) >= 3:
            return parts[2]
        return parts[1] if len(parts) >= 2 else None

    @staticmethod
    def _heading_of(state_id: str) -> float:
        parts = state_id.split(',')
        if parts[0] == 'S' and len(parts) >= 3:
            try:
                return math.radians(float(parts[2]))
            except ValueError:
                pass
        return 0.0

    def _schedule(self, time: float, kind: str, agent_id: int, **data):
        ev = Event(time, self._seq, kind, agent_id, data)
        self._seq += 1
        heapq.heappush(self._eq, ev)

    # ── 상태 조회 ──────────────────────────────────────────────────────────────

    # ── 동적 경로 확장 (incremental replan) ─────────────────────────────────

    def extend_agents_batch(self, agent_paths: dict, t_start: float):
        """
        여러 DONE agent에 새 경로를 한번에 부여하고 TAPG DAG를 확장합니다.

        1) 모든 새 경로의 agent 상태 초기화 + DAG 노드/순차엣지 추가
        2) 모든 새 경로가 등록된 후 cross-agent 엣지 빌드
        3) TRY_ADVANCE 스케줄

        Parameters
        ----------
        agent_paths : {agent_id: new_raw_path} - 새 경로를 받을 agent들
        t_start     : 현재 sim_time
        """
        new_paths = {}  # aid → raw_path (등록된 것만)

        # ── Phase 0: 이전 경로의 DAG 노드 정리 ──────────────────────────
        for aid, new_raw_path in agent_paths.items():
            agent = self.agents.get(aid)
            if agent is None or not new_raw_path:
                continue
            # 이전 경로의 남은 노드를 DAG에서 제거 (DONE 시 정리 안 된 것 포함)
            old_path = agent.raw_path
            if old_path:
                for k, (sid, t) in enumerate(old_path):
                    old_nk = self._nk(sid, aid, t)
                    if self.G.has_node(old_nk):
                        self.G.remove_node(old_nk)
                # wait_queues에서도 제거
                for nk, waiters in list(self.wait_queues.items()):
                    if aid in waiters:
                        waiters.remove(aid)
                    if not waiters:
                        self.wait_queues.pop(nk, None)

        # ── Phase 1: agent 상태 초기화 + DAG 노드/순차엣지 추가 ──────────
        for aid, new_raw_path in agent_paths.items():
            agent = self.agents.get(aid)
            if agent is None or not new_raw_path:
                continue

            agent.raw_path = new_raw_path
            agent.path_idx = 0
            agent.claim_idx = 0
            agent.state    = IDLE
            agent.v        = 0.0

            if new_raw_path:
                nid = self._node_of(new_raw_path[0][0])
                if nid and nid in self.graph.nodes:
                    n = self.graph.nodes[nid]
                    agent.x, agent.y = n.x, n.y
                agent.theta = self._heading_of(new_raw_path[0][0])

            for k, (sid, t) in enumerate(new_raw_path):
                nk = self._nk(sid, aid, t)
                # ── DEBUG: Phase1 노드가 이미 그래프에 있는지 확인 ──
                if nk in self.G:
                    old_edges_in = list(self.G.predecessors(nk))
                    old_edges_out = list(self.G.successors(nk))
                    if old_edges_in or old_edges_out:
                        print(f'  [TAPG WARN] Phase1: node {nk} already exists '
                              f'with in={len(old_edges_in)} out={len(old_edges_out)}')
                        for p in old_edges_in[:3]:
                            print(f'    pred: A{p[1]-100 if isinstance(p[1],int) else p[1]} {p[0]} t={p[2]:.2f}')
                        for s in old_edges_out[:3]:
                            print(f'    succ: A{s[1]-100 if isinstance(s[1],int) else s[1]} {s[0]} t={s[2]:.2f}')
                duration = (float('inf') if k == len(new_raw_path) - 1
                            else new_raw_path[k + 1][1] - t)
                self.G.add_node(nk, agv_id=aid, start_time=t, duration=duration)
                if k > 0:
                    prev_sid, prev_t = new_raw_path[k - 1]
                    prev_nk = self._nk(prev_sid, aid, prev_t)
                    if prev_nk != nk:  # self-loop 방지
                        self._add_edge(prev_nk, nk)

            new_paths[aid] = new_raw_path

        # ── Phase 2: cross-agent 엣지 빌드 ──────────────────────────────
        # 새 경로들 + 기존 active 경로들 전체를 대상으로 빌드
        # Coarse mode: time-based cross-edge skip (live occupancy 가 대체)
        if self._coarse_mode:
            for aid in new_paths:
                self._schedule(t_start, 'TRY_ADVANCE', aid)
            return

        all_paths = {}
        for aid, agent in self.agents.items():
            if agent.raw_path:
                all_paths[aid] = agent.raw_path  # 새 경로 포함 (Phase 1에서 교체됨)

        for ai, pi in new_paths.items():
            for aj, pj in all_paths.items():
                if ai == aj:
                    continue
                # pi vs pj: pi의 state가 pj를 block
                for k1 in range(len(pi) - 1, -1, -1):
                    s1, t1 = pi[k1]
                    affect1 = self._state_affect_set(s1)
                    if not affect1:
                        continue
                    for k2, (s2, t2) in enumerate(pj):
                        if t2 <= t1:
                            continue
                        affect2 = self._state_affect_set(s2)
                        if s2 in affect1 or s1 in affect2:
                            nk1 = self._nk(s1, ai, t1)
                            nk2 = self._nk(s2, aj, t2)
                            if nk1 == nk2:
                                print(f'  [TAPG SELF-LOOP] Phase2-A: {nk1}')
                            if nk1 in self.G and nk2 in self.G:
                                self._add_edge(nk1, nk2)
                            break

                # pj vs pi: pj의 state가 pi를 block
                if aj in new_paths:
                    continue
                for k1 in range(len(pj) - 1, -1, -1):
                    s1, t1 = pj[k1]
                    affect1 = self._state_affect_set(s1)
                    if not affect1:
                        continue
                    for k2, (s2, t2) in enumerate(pi):
                        if t2 <= t1:
                            continue
                        affect2 = self._state_affect_set(s2)
                        if s2 in affect1 or s1 in affect2:
                            nk1 = self._nk(s1, aj, t1)
                            nk2 = self._nk(s2, ai, t2)
                            if nk1 == nk2:
                                print(f'  [TAPG SELF-LOOP] Phase2-B: {nk1}')
                            if nk1 in self.G and nk2 in self.G:
                                self._add_edge(nk1, nk2)
                            break

        # ── Phase 3: TRY_ADVANCE 스케줄 ─────────────────────────────────
        for aid in new_paths:
            self._schedule(t_start, 'TRY_ADVANCE', aid)

    def append_agents_batch(self, agent_paths: dict, t_start: float):
        """각 agent의 기존 raw_path 뒤에 새 plan 을 이어붙인다 (replace 아님).

        의도:
          agent 의 `raw_path[-1]` (inf-claim 지점)을 replan 시작점으로 삼으므로,
          새 plan 은 `raw_path[-1]` 과 동일한 state 로 시작해야 한다. 그 duplicate
          첫 state 는 skip 하고 나머지만 append.

          path_idx / x / y / state 는 건드리지 않음 → agent 는 현재 실행 중인
          경로를 완주한 뒤 자연스럽게 새 구간으로 진입. 순간이동 없음.

          단 이전에 DONE 상태였던 agent 는 새 state 가 생겼으므로 IDLE 로 풀고
          TRY_ADVANCE 재스케줄.

        Parameters
        ----------
        agent_paths : {aid: new_raw_path}
            new_raw_path[0] 은 agent.raw_path[-1] 과 동일한 state여야 함
        t_start : float
            TRY_ADVANCE 재스케줄 시각 (보통 현재 sim_time)
        """
        ext_info = []

        for aid, new_raw_path in agent_paths.items():
            agent = self.agents.get(aid)
            if agent is None or not new_raw_path or len(new_raw_path) < 2:
                continue
            if not agent.raw_path:
                continue  # 빈 raw_path 는 append 불가, setup 경로로 처리해야 함

            ext = new_raw_path[1:]  # duplicate first state skip
            new_lo = len(agent.raw_path)

            # 이전 마지막 노드의 duration 갱신 (inf → 다음 state 까지의 간격).
            # recompute_earliest_schedule 에서만 읽히므로 정확성 유지용.
            prev_last_sid, prev_last_t = agent.raw_path[-1]
            prev_last_nk = self._nk(prev_last_sid, aid, prev_last_t)
            if self.G.has_node(prev_last_nk):
                self.G.nodes[prev_last_nk]['duration'] = ext[0][1] - prev_last_t

            agent.raw_path.extend(ext)
            new_hi = len(agent.raw_path)

            # 새 state 노드 + 같은 agent sequential edge 추가
            for k in range(new_lo, new_hi):
                sid, t = agent.raw_path[k]
                nk = self._nk(sid, aid, t)
                duration = (float('inf') if k == new_hi - 1
                            else agent.raw_path[k + 1][1] - t)
                self.G.add_node(nk, agv_id=aid, start_time=t, duration=duration)
                if k > 0:
                    prev_sid, prev_t = agent.raw_path[k - 1]
                    prev_nk = self._nk(prev_sid, aid, prev_t)
                    if prev_nk != nk:
                        self._add_edge(prev_nk, nk)

            ext_info.append((aid, new_lo, new_hi))

            # DONE agent 는 새 state 로 진입할 수 있도록 unblock.
            # path_idx 를 *마지막 도달 state* (= old path 끝, AGV 의 현재 물리
            # 위치) 로 reset. 그러지 않으면 path_idx 가 새 ext[0] (= M) 을
            # 가리켜 'AGV 정지 중인데 path[path_idx]=M' invariant 위반.
            if agent.state == DONE:
                agent.state = IDLE
                agent.v = 0.0
                # path_idx 가 new_lo (= ext[0] idx) 이면 그 직전 (= 현재 S) 로 back
                if agent.path_idx >= new_lo:
                    agent.path_idx = new_lo - 1
                self._schedule(t_start, 'TRY_ADVANCE', aid)

        # cross-agent TAPG edge 보강 (새 구간 ↔ 다른 agent 전체 경로, 양방향)
        # Coarse mode: skip - live occupancy 가 대체
        if ext_info and not self._coarse_mode:
            self.add_cross_edges_for_extensions(ext_info)

    def add_cross_edges_for_extensions(self, ext_info: list):
        """경로 확장 후 cross-agent TAPG edge를 추가한다.

        Edge A → B 는 "A 의 plan 종료시각 ≤ B 의 plan 시작시각" 일 때만 추가.
        그렇지 않으면 plan timing 과 모순되는 dependency 가 되어 실행 시
        lock-step deadlock 을 유발할 수 있음.

        Coarse mode: skip - live occupancy 가 대체. 호출자 모두 (외부 wrapper
        포함) 가 일관되게 skip 되도록 함수 진입 시 guard.

        Parameters
        ----------
        ext_info : list[tuple[int, int, int]]
            (agent_id, new_start_idx, new_end_idx_exclusive) 리스트
        """
        if self._coarse_mode:
            return
        all_paths = {a.id: a.raw_path for a in self.agents.values()
                     if a.raw_path}

        def _finish_time(path, k):
            """path[k] 의 plan 종료시각 = path[k+1].t (없으면 inf)."""
            if k + 1 < len(path):
                return path[k + 1][1]
            return float('inf')

        for ai, lo, hi in ext_info:
            pi = all_paths.get(ai)
            if not pi:
                continue

            for aj, pj in all_paths.items():
                if aj == ai:
                    continue

                # Direction A: ai의 새 state가 pj의 어떤 state를 block (ai → aj)
                for k1 in range(hi - 1, lo - 1, -1):
                    s1, t1 = pi[k1]
                    affect1 = self._state_affect_set(s1)
                    if not affect1:
                        continue
                    s1_finish = _finish_time(pi, k1)
                    for k2, (s2, t2) in enumerate(pj):
                        if t2 < s1_finish:
                            continue
                        if t2 == s1_finish and ai >= aj:
                            continue
                        affect2 = self._state_affect_set(s2)
                        if s2 in affect1 or s1 in affect2:
                            nk1 = self._nk(s1, ai, t1)
                            nk2 = self._nk(s2, aj, t2)
                            if nk1 in self.G and nk2 in self.G:
                                self._add_edge(nk1, nk2)
                            # target 이 S 인 경우, 그 S 에 진입하는 직전 M/R 에도
                            # edge 추가 - 차량이 destination 에 도착하기 전에 막아야
                            # 물리적 충돌을 방지할 수 있음.
                            if (s2.startswith('S,') and k2 > 0):
                                prev_sid, prev_t = pj[k2 - 1]
                                if (prev_sid.startswith('M,') or
                                        prev_sid.startswith('R,')):
                                    prev_nk = self._nk(prev_sid, aj, prev_t)
                                    if prev_nk in self.G and nk1 in self.G:
                                        self._add_edge(nk1, prev_nk)
                            break

                # Direction B: pj의 어떤 state가 ai의 새 state를 block (aj → ai)
                for k1 in range(len(pj) - 1, -1, -1):
                    s1, t1 = pj[k1]
                    affect1 = self._state_affect_set(s1)
                    if not affect1:
                        continue
                    s1_finish = _finish_time(pj, k1)
                    for k2 in range(lo, hi):
                        s2, t2 = pi[k2]
                        if t2 < s1_finish:
                            continue
                        if t2 == s1_finish and aj >= ai:
                            continue
                        affect2 = self._state_affect_set(s2)
                        if s2 in affect1 or s1 in affect2:
                            nk1 = self._nk(s1, aj, t1)
                            nk2 = self._nk(s2, ai, t2)
                            if nk1 in self.G and nk2 in self.G:
                                self._add_edge(nk1, nk2)
                            # target 이 S 인 경우 직전 M/R 에도 edge 추가
                            if (s2.startswith('S,') and k2 > lo):
                                prev_sid, prev_t = pi[k2 - 1]
                                if (prev_sid.startswith('M,') or
                                        prev_sid.startswith('R,')):
                                    prev_nk = self._nk(prev_sid, ai, prev_t)
                                    if prev_nk in self.G and nk1 in self.G:
                                        self._add_edge(nk1, prev_nk)
                            break

    def recompute_earliest_schedule(self, current_time=0.0):
        """
        TAPG DAG의 earliest start schedule을 재계산합니다.

        1) 모든 agent의 DAG 시작 노드 시간 = current_time
        2) 위상 정렬 순서대로 earliest start 전파
        3) M/R의 duration = state.cost (고정)
           S의 duration = 다음 action의 earliest_start - S의 earliest_start
        """
        import networkx as nx

        if self.G.number_of_nodes() == 0:
            return {}

        try:
            topo_order = list(nx.topological_sort(self.G))
        except nx.NetworkXUnfeasible:
            return {}

        # 각 agent의 DAG 시작 노드 찾기 (in-degree 0 중 같은 agent)
        start_nodes = set()
        for v in topo_order:
            same_agent_preds = [p for p in self.G.predecessors(v) if p[1] == v[1]]
            if not same_agent_preds:
                start_nodes.add(v)

        def _pred_dur(u):
            """Predecessor finish 계산용 duration.

            - M/R: state.cost (rigid)
            - S(is_dwell=True): DAG 노드의 duration 보존 (LOADING/UNLOADING dwell)
            - S(일반 대기): 0 - cross-pred 제약에 따라 동적으로 늘어남
            """
            u_sid = u[0]
            u_state = self._get_state_obj(u_sid)
            if u_sid.startswith('S,'):
                if self.G.nodes[u].get('is_dwell'):
                    old_dur = self.G.nodes[u].get('duration', 0)
                    return 0 if old_dur == float('inf') else old_dur
                return 0.0
            return u_state.cost if u_state and u_state.cost else 0.0

        # Earliest start 계산
        new_start = {}
        for v in topo_order:
            state_id = v[0]

            # Cross-agent predecessor의 earliest finish
            cross_pred_finish = []
            for u in self.G.predecessors(v):
                if u[1] != v[1] and u in new_start:
                    cross_pred_finish.append(new_start[u] + _pred_dur(u))

            # Same-agent predecessor의 earliest finish
            same_pred_finish = []
            for u in self.G.predecessors(v):
                if u[1] == v[1] and u in new_start:
                    same_pred_finish.append(new_start[u] + _pred_dur(u))

            if v in start_nodes:
                # 시작 노드: current_time, cross-agent 제약 반영
                t_v = current_time
                if cross_pred_finish:
                    t_v = max(t_v, max(cross_pred_finish))
            else:
                all_finish = same_pred_finish + cross_pred_finish
                if all_finish:
                    t_v = max(all_finish)
                else:
                    t_v = current_time

            new_start[v] = t_v

        # 새 그래프 구성
        H = nx.DiGraph()
        old_to_new = {}
        for old_node in topo_order:
            if old_node not in new_start:
                continue
            t_v = new_start[old_node]
            state_id, agv_i, _old_t = old_node
            new_node = self._nk(state_id, agv_i, t_v)
            old_to_new[old_node] = new_node

            attrs = dict(self.G.nodes[old_node])
            attrs['start_time'] = t_v
            # duration 갱신:
            #   M/R = state.cost
            #   S(is_dwell): 기존 DAG duration 보존 (LOADING/UNLOADING 시간 유지)
            #   S(일반 대기): 0 - 후속 노드에 의해 결정 (elastic)
            state_obj = self._get_state_obj(state_id)
            if state_obj and state_id.startswith('S,'):
                if attrs.get('is_dwell'):
                    old_dur = self.G.nodes[old_node].get('duration', 0)
                    attrs['duration'] = old_dur if old_dur != float('inf') else 0
                else:
                    attrs['duration'] = 0
            elif state_obj:
                attrs['duration'] = state_obj.cost
            H.add_node(new_node, **attrs)

        for u, v in self.G.edges():
            if u in old_to_new and v in old_to_new:
                if old_to_new[u] != old_to_new[v]:
                    H.add_edge(old_to_new[u], old_to_new[v])

        self.G = H

        # agent raw_path + _tapg_node 갱신
        for agent in self.agents.values():
            new_raw_path = []
            for idx, (sid, old_t) in enumerate(agent.raw_path):
                old_key = self._nk(sid, agent.id, old_t)
                if old_key in old_to_new:
                    new_raw_path.append((sid, round(old_to_new[old_key][2], 6)))
                else:
                    new_raw_path.append((sid, round(old_t, 6)))
            agent.raw_path = new_raw_path

            if agent._tapg_node and agent._tapg_node in old_to_new:
                agent._tapg_node = old_to_new[agent._tapg_node]

        # wait_queues 키 갱신
        new_wq = {}
        for old_key, waiters in self.wait_queues.items():
            new_key = old_to_new.get(old_key, old_key)
            new_wq[new_key] = waiters
        self.wait_queues = new_wq

        return new_start

    def all_done(self) -> bool:
        return all(a.state == DONE for a in self.agents.values())

    def tapg_stats(self) -> dict:
        total     = self.G.number_of_nodes()
        edges     = self.G.number_of_edges()
        seq_edges = sum(1 for u, v in self.G.edges()
                        if self.G.nodes[u].get('agv_id') == self.G.nodes[v].get('agv_id'))
        return {
            'total':      total,
            'done':       0,
            'remaining':  total - done,
            'edges':      edges,
            'cross_edges': edges - seq_edges,
        }
