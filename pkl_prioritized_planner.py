"""
pkl_prioritized_planner.py — Dependency-graph based prioritized SIPP planner
for PklMapGraph (state-level planning with collision profiles).

Works on the state graph directly:
  Stop states: S,node_id,heading  (next_state → Move states)
  Move states: M,from_id,to_id   (next_state → Stop states)

Output: [(state_id, time), ...] per agent — directly consumable by TAPGEnvironment.
"""
from __future__ import annotations
import math, heapq, time, random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None

from pkl_loader import PklMapGraph


class SippFailure(RuntimeError):
    """Raised when SIPP fails and `halt_on_sipp_fail` is enabled.

    Caller (시뮬레이터 상위 루프) 가 catch 해서 상세 상태 덤프 후 종료하는
    용도. 기본 동작은 fallback 시도이므로 이 예외는 디버그 플래그 ON 일 때만
    발생.
    """
    def __init__(self, agent_id, start_node, goal_node, constraints_count):
        self.agent_id = agent_id
        self.start_node = start_node
        self.goal_node = goal_node
        self.constraints_count = constraints_count
        super().__init__(
            f"SIPP failed: agent {agent_id}, {start_node} → {goal_node} "
            f"(constraints={constraints_count})"
        )


# ── SIPP node ────────────────────────────────────────────────────────────────

class SIPPNode:
    __slots__ = ('state', 'g', 'h', 'f', 'time', 'interval')

    def __init__(self, state: str, g: float, time: float,
                 interval: Tuple[float, float]):
        self.state    = state
        self.g        = g
        self.time     = time
        self.interval = interval
        self.h        = 0.0
        self.f        = g

    def __lt__(self, other):
        # tie-breakers: g (lower-g first preferred), then state id, then interval.
        # 같은 f 일 때 heap 의 array position 이 결정짓는 implicit ordering 을
        # 피하기 위함. CPython heapq 는 < 만 쓰는데 ties 시 array offset 으로
        # implicit tiebreak 이 일어나면, 같은 sim 입력이라도 메모리 할당 순서나
        # 호출 sequence 에 따라 비결정적 결과가 나올 수 있음.
        if self.f != other.f: return self.f < other.f
        if self.g != other.g: return self.g < other.g
        if self.state != other.state: return self.state < other.state
        return self.interval < other.interval


# ── Interval helpers ─────────────────────────────────────────────────────────

def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _safe_intervals(t_lo, t_hi, blocked):
    merged = _merge_intervals(blocked)
    result = []
    cur = t_lo
    for bs, be in merged:
        if bs > cur:
            result.append((cur, bs))
        cur = max(cur, be)
    if cur < t_hi:
        result.append((cur, t_hi))
    return result


# ── Heuristic (reverse Dijkstra on state graph) ─────────────────────────────

def _build_state_heuristic(graph: PklMapGraph, goal_node: str) -> Dict[str, float]:
    """Reverse Dijkstra from any S,goal_node,* to all reachable states.

    Stop / Move / Rotate / Load 모든 state 포함. Load (L) state 는 stop 으로의
    transition 만 있고 cost = dwell. heuristic 은 goal 까지의 최소 traversal
    cost — 실제 dwell duration 까지 포함되도록 L 도 edge 로 취급.
    """
    # Build reverse adjacency: for each state, who can transition TO it?
    radj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for sid, s in graph.stop_states_raw.items():
        for ns in s.next_state:
            # Stop → Move/Load: cost of the stop state (0 for stops)
            radj[ns].append((sid, s.cost))
    for sid, s in graph.move_states_raw.items():
        for ns in s.next_state:
            # Move → Stop: cost of the move state
            radj[ns].append((sid, s.cost))
    # Load → Stop transitions (반대 방향으로 stop ← load 등록)
    for sid, s in getattr(graph, 'load_states_raw', {}).items():
        for ns in s.next_state:
            radj[ns].append((sid, s.cost))

    # Initialize from all stop states at goal_node
    dist: Dict[str, float] = {}
    heap = []
    for sid in graph.stop_states_raw:
        parts = sid.split(',')
        if parts[1] == goal_node:
            dist[sid] = 0.0
            heapq.heappush(heap, (0.0, sid))

    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, float('inf')):
            continue
        for prev_sid, cost in radj.get(cur, []):
            nd = d + cost
            if nd < dist.get(prev_sid, float('inf')):
                dist[prev_sid] = nd
                heapq.heappush(heap, (nd, prev_sid))

    return dist


# ── Planner ──────────────────────────────────────────────────────────────────

class PklPrioritizedPlanner:
    """Prioritized SIPP planner operating on PklMapGraph state graph."""

    def __init__(self, graph: PklMapGraph,
                 push_extras: Optional[List[str]] = None):
        """
        push_extras : 추가 push 후보 노드 (예: Tier-A sidings). port 가 아닌
            안전한 임시 정차 지점을 simulator 에서 주입. 점유는 SIPP constraint
            로 자연 반영됨. 기본 None 이면 port 만 사용.
        """
        self.graph = graph
        self._port_nodes: List[str] = list(set(graph.ports.values()))
        # push 후보 풀 = ports ∪ 외부 주입 sidings.
        # _find_empty_ports / _find_alternate_goal 가 사용.
        self._push_pool: List[str] = list(self._port_nodes)
        if push_extras:
            for n in push_extras:
                if n in graph.nodes and n not in self._push_pool:
                    self._push_pool.append(n)
        self._h_cache: Dict[str, Dict[str, float]] = {}
        # push_extras 만 별도 보관 (= strict cut-safe sidings only, ports 제외)
        self._push_extras_only: List[str] = list(push_extras) if push_extras else []
        # If True, raise SippFailure immediately on primary SIPP fail (no fallback).
        # For debugging — let the caller dump state before exit.
        self.halt_on_sipp_fail: bool = False

        # dwell_time = pkl 의 L state 의 cost (= LOADING/UNLOADING 시간).
        # _is_goal_constrained 의 dwell window 검사에 사용.
        self.dwell_time: float = 3.0
        if graph.load_states_raw:
            sample = next(iter(graph.load_states_raw.values()))
            self.dwell_time = float(sample.cost)
        # 재귀 가드 — _is_goal_constrained 안의 exit-feasibility SIPP 가
        # 다시 _is_goal_constrained 호출 시 dwell-window 검사로 fallback.
        self._in_exit_check: bool = False
        # True 면 원래 strict 검사 (도착 후 영원 free) 로 회귀 — 비교/디버그용.
        self._goal_check_strict: bool = False
        # _is_goal_constrained 가 exit feasibility 검증에 성공했을 때 그
        # path 를 저장. 외부 _sipp_search 가 main path 끝에 L state + 이 exit
        # path 를 attach 한다.
        self._last_accepted_exit_path: Optional[List[Tuple[str, float]]] = None
        # plan() 안에서 현재 SIPP search 중인 agent id — _is_goal_constrained
        # 가 endpoint allocator 의 preferred siding 을 query 할 때 사용.
        self._current_planning_aid: Optional[int] = None
        # Endpoint allocator (외부 주입). exit candidate 선정 시 활용.
        # None 이면 self._push_pool 전체에서 nearest-free.
        self._endpoint_allocator = None
        # Endpoint candidates — strict cut-safe sidings only (ports 제외).
        # set_endpoint_allocator() 시 자동 세팅.
        self._endpoint_candidates: Optional[List[str]] = None

    def set_endpoint_allocator(self, allocator):
        """외부 endpoint allocator 주입. exit candidate 선정 시 활용."""
        self._endpoint_allocator = allocator
        self._endpoint_candidates = list(allocator.candidates) if allocator else None

    def _heuristic(self, goal_node: str) -> Dict[str, float]:
        if goal_node not in self._h_cache:
            self._h_cache[goal_node] = _build_state_heuristic(self.graph, goal_node)
        return self._h_cache[goal_node]

    # ── State lookup ─────────────────────────────────────────────────────────

    def _get_state(self, state_id: str):
        if state_id.startswith('S,'):
            return self.graph.stop_states_raw.get(state_id)
        elif state_id.startswith('M,') or state_id.startswith('R,'):
            return self.graph.move_states_raw.get(state_id)
        elif state_id.startswith('L,'):
            return self.graph.load_states_raw.get(state_id)
        return None

    def _node_from_state(self, state_id: str) -> str:
        parts = state_id.split(',')
        if parts[0] == 'S':
            return parts[1]
        elif parts[0] == 'M':
            return parts[2]
        # R,node_id,...  or  L,node_id
        return parts[1]

    def _is_stop_at_node(self, state_id: str, node_id: str) -> bool:
        """Check if state_id is a Stop state at the given node."""
        parts = state_id.split(',')
        return parts[0] == 'S' and parts[1] == node_id

    def _find_stop_state(self, node_id: str) -> Optional[str]:
        """Find any stop state at node_id (prefer heading 0)."""
        best = None
        for sid in self.graph.stop_states_raw:
            parts = sid.split(',')
            if parts[1] == node_id:
                if parts[2] == '0':
                    return sid
                if best is None:
                    best = sid
        return best

    # ── Constraint building ──────────────────────────────────────────────────

    def _build_constraints(self, planned_path: List[Tuple[str, float]],
                           agent_id: int) -> List[dict]:
        constraints = []
        for idx, (state_id, t_start) in enumerate(planned_path):
            state = self._get_state(state_id)
            if state is None:
                continue

            t_end = planned_path[idx + 1][1] if idx < len(planned_path) - 1 else float('inf')
            if t_start == t_end and idx < len(planned_path) - 1:
                continue

            # Block the state itself
            constraints.append({
                'agent': agent_id, 'loc': state_id,
                'timestep': (t_start, t_end),
            })

            # Block all affect_states
            for aff_id in state.affect_state:
                aff = self._get_state(aff_id)
                aff_cost = aff.cost if aff else 0.0
                c_start = max(0.0, t_start - aff_cost)
                constraints.append({
                    'agent': agent_id, 'loc': aff_id,
                    'timestep': (c_start, t_end),
                })

        return constraints

    def _make_constraint_table(self, all_constraints, exclude_agent):
        table: Dict[str, List[Tuple[float, float]]] = {}
        for c in all_constraints:
            if c['agent'] == exclude_agent:
                continue
            loc = c['loc']
            if loc not in table:
                table[loc] = []
            table[loc].append(c['timestep'])
        return table

    # ── SIPP search on state graph ───────────────────────────────────────────

    def _get_successors(self, state_id: str, cur_time: float,
                        interval: Tuple[float, float],
                        goal_node: str,
                        c_table: Dict[str, List]) -> List[SIPPNode]:
        """Automod SIPP 방식: 부모 interval 하나에 대해서만 successor 생성."""
        state = self._get_state(state_id)
        if state is None:
            return []

        successors = []
        cost = state.cost
        start_t = cur_time + cost   # S: cost=0, M/R: traversal time
        end_t = interval[1]

        # 현재 상태의 노드 추출 (같은 노드 내 전이 판별용)
        cur_node = self._node_of_state(state_id)

        for neighbor in state.next_state:
            # goal node면 end_t를 inf로 (목적지에서 무한 대기 가능)
            local_end = float('inf') if self._is_stop_at_node(neighbor, goal_node) else end_t

            # M→S, R→S, L→S 전이: 이미 해당 노드에 도착/존재하므로 arrive 지연 불가
            # safe interval이 start_t를 포함하지 않으면 사용 불가
            # S→M, S→R, S→L 전이: S에서 대기 후 출발 가능하므로 제약 없음
            must_arrive_now = (state_id.startswith(('M,', 'R,', 'L,'))
                               and neighbor.startswith('S,'))

            # neighbor의 safe intervals: 부모 interval 범위 내에서 계산
            blocked = c_table.get(neighbor, [])
            safe = _safe_intervals(interval[0], local_end, blocked)

            for si in safe:
                if si[0] > local_end or si[1] <= start_t:
                    continue
                arrive = max(start_t, si[0])

                # M→S, R→S: 도착 시점을 늦출 수 없음
                if must_arrive_now and arrive > start_t + 1e-6:
                    continue

                successors.append(SIPPNode(neighbor, arrive, arrive, si))

        return successors

    @staticmethod
    def _node_of_state(state_id: str) -> Optional[str]:
        """state_id에서 노드 ID 추출. S,node,angle → node / R,node,a,b → node /
        L,node → node / M,from,to → None"""
        parts = state_id.split(',')
        if parts[0] in ('S', 'R', 'L') and len(parts) >= 2:
            return parts[1]
        return None  # M 상태는 두 노드 간 이동이므로 same-node 판별 대상 아님

    def _is_goal_constrained(self, state_id, t, c_table):
        """Goal arrival 의 수용 여부 결정.

        Lifelong 환경에서 원래의 "도착 후 영원 free" 검사가 over-strict 하여
        매우 긴 wait plan 을 유발. 다음 검사로 대체:

        1) Dwell window [t, t+dwell] 동안 goal state 가 free 인가? (False 면 reject)
        2) Dwell 종료 후 가까운 siding 으로 SIPP 탈출 path 가 존재하는가?
           (= 영원 점유 대신 dwell 후 자발적 exit 가능성으로 안전 보장)

        Inner SIPP (= exit feasibility 검사) 의 재귀를 피하기 위해
        `self._in_exit_check` 플래그로 단순 dwell-window 검사로 fallback.

        검사 비활성화 시 (`self._goal_check_strict = True`) 원래 strict 검사 사용.
        """
        if getattr(self, '_goal_check_strict', False):
            for ts, te in c_table.get(state_id, []):
                if te > t:
                    return True
            return False

        dwell = getattr(self, 'dwell_time', 3.0)

        # 1) Dwell window 검사 — 도착 + dwell 동안 어떤 block 과도 겹치면 reject
        for ts, te in c_table.get(state_id, []):
            if ts < t + dwell and te > t:
                if not getattr(self, '_in_exit_check', False):
                    self._dbg_reject_reason = 'DWELL_WINDOW'
                    self._dbg_reject_at = state_id
                    self._dbg_reject_blocker = (ts, te)
                return True

        # 1.5) Post-dwell inf-claim 검사 — dwell 후 이 agent 가 goal 에 inf-park
        # 한다고 가정. 누군가의 future constraint (te > t+dwell) 와 overlap 하면
        # 그 agent 와 충돌 → reject. SIPP 가 더 늦은 도착 시각 (after future
        # constraint ends) 으로 시도하도록 유도.
        if not getattr(self, '_in_exit_check', False):
            t_exit = t + dwell
            for ts, te in c_table.get(state_id, []):
                if te > t_exit:
                    # 누군가 t_exit 이후까지 점유 — A 의 inf-claim 과 충돌
                    self._dbg_reject_reason = 'POST_DWELL_FUTURE'
                    self._dbg_reject_at = state_id
                    self._dbg_reject_blocker = (ts, te)
                    return True

        # 재귀 중이면 (= exit feasibility 검증 안의 SIPP 가 다시 goal 도달):
        # dwell-window 검사만 통과시키고 exit 추가 검증은 skip.
        if getattr(self, '_in_exit_check', False):
            return False

        # 2) Exit feasibility — siding 으로 SIPP path 존재?
        goal_node = self._node_of_state(state_id)
        if goal_node is None:
            return False
        exit_time = t + dwell
        h = self._heuristic(goal_node)

        # Endpoint candidate pool 결정 — allocator 가 있으면 그 candidates
        # (= strict cut-safe sidings only), 없으면 push_extras_only 로 fallback.
        # Ports 는 후보에서 제외 — task target 이라 inf-park 부적합.
        if self._endpoint_candidates is not None:
            candidate_pool = self._endpoint_candidates
        else:
            candidate_pool = self._push_extras_only or self._push_pool

        # Allocator 가 이 agent 에게 preferred endpoint 를 가지고 있으면
        # 우선 candidate. 다른 free siding 보다 먼저 시도.
        preferred = None
        if (self._endpoint_allocator is not None
                and self._current_planning_aid is not None):
            preferred = self._endpoint_allocator.get(self._current_planning_aid)

        # 다른 agent 가 이미 endpoint 로 예약한 sidings (= 자기 자신 제외)
        occupied_by_others = set()
        if self._endpoint_allocator is not None:
            for aid, sid in self._endpoint_allocator.assignments.items():
                if aid != self._current_planning_aid:
                    occupied_by_others.add(sid)

        candidates = []

        def _check_future_blocked(siding_node):
            """siding 의 S state 또는 affect_state 가 미래 점유되어 inf-park
            안전하지 않은지 검사."""
            sid_s = self._find_stop_state(siding_node)
            if sid_s is None:
                return True, None
            for ts, te in c_table.get(sid_s, []):
                if te > exit_time:
                    return True, sid_s
            sid_state = self._get_state(sid_s)
            if sid_state is not None:
                for aff in sid_state.affect_state:
                    for ts, te in c_table.get(aff, []):
                        if te > exit_time:
                            return True, sid_s
            return False, sid_s

        # 1) Preferred 우선
        if preferred and preferred != goal_node:
            fb, sid_s = _check_future_blocked(preferred)
            if not fb and sid_s is not None:
                d = h.get(sid_s, float('inf'))
                if d != float('inf'):
                    candidates.append((d, preferred))

        # 2) 나머지 candidate_pool — other agent endpoint 예약 노드 + future
        # blocked 노드 제외
        for siding in candidate_pool:
            if siding == goal_node or siding == preferred:
                continue
            if siding in occupied_by_others:
                continue
            fb, sid_s = _check_future_blocked(siding)
            if fb or sid_s is None:
                continue
            d = h.get(sid_s, float('inf'))
            if d != float('inf'):
                candidates.append((d, siding))

        # 거리순 정렬 — preferred 는 이미 candidates[0] 에 들어있음
        # (preferred 가 있으면 그것이 첫 시도. 그 뒤로 nearest-free 순)
        if preferred and candidates and candidates[0][1] == preferred:
            head, tail = candidates[0], candidates[1:]
            tail.sort()
            candidates = [head] + tail
        else:
            candidates.sort()

        TOP_K = 5
        INNER_TIMEOUT = 1.0

        self._in_exit_check = True
        try:
            for _, siding in candidates[:TOP_K]:
                path = self._sipp_search(goal_node, siding, c_table,
                                          start_time=exit_time,
                                          timeout=INNER_TIMEOUT)
                if path is not None:
                    self._last_accepted_exit_path = path
                    return False
        finally:
            self._in_exit_check = False
        self._dbg_reject_reason = 'EXIT_INFEASIBLE'
        self._dbg_reject_at = state_id
        self._dbg_reject_blocker = (len(candidates), candidates[:5])
        return True

    def _sipp_search(self, start_node: str, goal_node: str,
                     c_table: Dict, start_time: float = 0.0,
                     timeout: float = 10.0,
                     start_sid: Optional[str] = None
                     ) -> Optional[List[Tuple[str, float]]]:
        """SIPP A* on state graph from start_node to goal_node.

        start_sid: 외부에서 명시한 시작 stop state (e.g., 'S,Port1,91'). None
        이면 _find_stop_state 로 자동 선택. ACS_focal 와 호환을 위해 추가됨
        — ACS_focal 은 특정 heading 의 start state 를 받음.
        """
        t0 = time.time()
        h_table = self._heuristic(goal_node)
        # [DBG] tracking
        self._dbg_states_expanded = 0
        self._dbg_goal_reached = 0
        self._dbg_goal_rejected = 0
        self._dbg_max_g = 0.0  # 가장 멀리 간 시간
        self._dbg_timed_out = False

        # Find best start stop state at start_node (외부 명시 우선)
        if start_sid is None:
            start_sid = self._find_stop_state(start_node)
        if start_sid is None or start_sid not in self.graph.stop_states_raw:
            return None

        blocked = c_table.get(start_sid, [])
        safe = _safe_intervals(start_time, float('inf'), blocked)
        if not safe:
            return None
        init_interval = safe[0]
        actual_start = max(start_time, init_interval[0])

        s0 = SIPPNode(start_sid, actual_start, actual_start, init_interval)
        s0.h = h_table.get(start_sid, float('inf'))
        s0.f = s0.g + s0.h

        open_list = [s0]
        came_from = {(start_sid, init_interval): None}
        cost_so_far = {(start_sid, init_interval): s0.g}

        while open_list:
            if time.time() - t0 > timeout:
                self._dbg_timed_out = True
                return None

            current = heapq.heappop(open_list)
            key = (current.state, current.interval)
            self._dbg_states_expanded += 1
            if current.g > self._dbg_max_g:
                self._dbg_max_g = current.g

            # Goal check: any Stop state at goal_node, unconstrained
            if self._is_stop_at_node(current.state, goal_node):
                self._dbg_goal_reached += 1
                # _is_goal_constrained 호출 전에 exit-path slot 비움 → 검사
                # 안에서 exit feasibility 성공 시 그 path 가 저장됨.
                self._last_accepted_exit_path = None
                goal_blocked = self._is_goal_constrained(current.state, current.time, c_table)
                if goal_blocked:
                    self._dbg_goal_rejected += 1
                if not goal_blocked:
                    main_path = self._reconstruct(came_from, cost_so_far, key)
                    # Exit path 는 *검사용* 으로만 사용 — 실제 plan 에 append 안 함.
                    # 이전 동작은 *항상* siding 으로 빠지게 만들어서 1 AGV 일
                    # 때도 불필요 이동. 다른 AGV 가 port 필요한 contention 은
                    # wrapper 의 idle blocker 가 잡음 (target == agent end →
                    # 그 agent push).
                    self._last_accepted_exit_path = None
                    return main_path

            if current.g > cost_so_far.get(key, float('inf')):
                continue

            succs = self._get_successors(current.state, current.time,
                                         current.interval, goal_node, c_table)

            for s in succs:
                sk = (s.state, s.interval)
                g_new = s.time
                h_val = h_table.get(s.state, float('inf'))
                f_val = g_new + h_val

                if sk not in cost_so_far or g_new < cost_so_far[sk]:
                    cost_so_far[sk] = g_new
                    came_from[sk] = key
                    s.g = g_new
                    s.h = h_val
                    s.f = f_val
                    heapq.heappush(open_list, s)

        return None

    def _reconstruct(self, came_from, cost_so_far, goal_key):
        path = []
        cur = goal_key
        while cur is not None:
            state_id = cur[0]
            t = cost_so_far[cur]
            path.append((state_id, t))
            cur = came_from.get(cur)
        path.reverse()
        return path

    # ── Dependency graph ─────────────────────────────────────────────────────

    def _build_dependency_graph(self, agent_nodes, agent_goals):
        G = nx.DiGraph()
        for aid in agent_nodes:
            G.add_node(aid)
        for i, goal_i in agent_goals.items():
            for j, cur_j in agent_nodes.items():
                if i != j and goal_i == cur_j:
                    G.add_edge(i, j)
        return G

    # ── Path utilities ───────────────────────────────────────────────────────

    @staticmethod
    def path_to_nodes(state_path: List[Tuple[str, float]]) -> List[str]:
        nodes = []
        for state_id, _ in state_path:
            parts = state_id.split(',')
            if parts[0] == 'S':
                nid = parts[1]
            elif parts[0] == 'M':
                nid = parts[2]
            else:
                continue
            if not nodes or nodes[-1] != nid:
                nodes.append(nid)
        return nodes

    # ── Main planning interface ──────────────────────────────────────────────

    def plan(self, agent_positions: Dict[int, str],
             agent_goals: Dict[int, str],
             existing_constraints: Optional[List[dict]] = None,
             start_times: Optional[Dict[int, float]] = None,
             timeout_per_agent: float = 30.0,
             disable_alternate_goal: bool = False,
             quiet_fail: bool = False,
             non_task_agents: Optional[set] = None) -> Optional['PklPlanResult']:
        """
        Plan conflict-free paths for all agents.

        Parameters
        ----------
        agent_positions : {agent_id: node_id}
        agent_goals     : {agent_id: node_id}  (port destinations)
        non_task_agents : set, optional
            LOADING/UNLOADING 진행 중이 아닌 agent set (= idle / pushed).
            Phase 2/3 cycle-break push 시 이 agent 들만 후보로 사용 — task
            agent 의 dwell semantics 보존. 모든 후보가 task agent 면 push 를
            skip 하고 다음 tick 으로 미룬다.

        Returns
        -------
        PklPlanResult with .paths (state-level) and .node_paths
        """
        if nx is None:
            raise ImportError("networkx is required")

        all_constraints = list(existing_constraints or [])
        agent_nodes = dict(agent_positions)
        goals = dict(agent_goals)
        starts = dict(start_times or {})
        non_task = set(non_task_agents) if non_task_agents is not None else None

        # Validate goals
        for aid, goal in list(goals.items()):
            if goal not in self.graph.nodes:
                print(f"[WARN] Goal {goal} not in graph for agent {aid}")
                goals.pop(aid)

        at_goal = {aid for aid, g in goals.items() if agent_nodes.get(aid) == g}
        planned: Dict[int, List[Tuple[str, float]]] = {}
        remaining = set(goals.keys())

        for aid in at_goal:
            t = starts.get(aid, 0.0)
            sid = self._find_stop_state(goals[aid])
            if sid:
                planned[aid] = [(sid, t)]
                cs = self._build_constraints(planned[aid], aid)
                all_constraints.extend(cs)
            remaining.discard(aid)

        max_iter = len(remaining) * 3 + 10

        for iteration in range(max_iter):
            if not remaining:
                break

            G_dep = self._build_dependency_graph(agent_nodes, goals)

            # Phase 1: Z set (out-degree 0, not yet planned)
            Z = [a for a in G_dep.nodes
                 if G_dep.out_degree(a) == 0 and a in remaining]

            best_aid, best_goal = None, None
            best_dist = float('inf')

            for aid in Z:
                goal_n = goals[aid]
                cur_n  = agent_nodes[aid]
                h = self._heuristic(goal_n)
                start_sid = self._find_stop_state(cur_n)
                dist = h.get(start_sid, float('inf')) if start_sid else float('inf')
                if dist < best_dist:
                    best_dist = dist
                    best_aid  = aid
                    best_goal = goal_n

            # Phase 2: blockers
            if best_aid is None:
                planned_set = set(planned.keys())
                blockers = [a for a in G_dep.nodes
                            if G_dep.in_degree(a) > 0 and a in planned_set]
                empty_ports = self._find_empty_ports(agent_nodes, goals)
                if blockers and empty_ports:
                    # 모든 blocker push 가능 — task agent 도 path extension 으로
                    # 후속 agent 의 goal port 비워줌. 실제 dwell 은 task agent 의
                    # 원래 path 따라 진행되고, 그 후 siding 으로 빠지는 효과.
                    candidates = blockers
                    for aid in candidates:
                        cur_n = agent_nodes[aid]
                        start_sid = self._find_stop_state(cur_n)
                        for port in empty_ports:
                            h = self._heuristic(port)
                            dist = h.get(start_sid, float('inf')) if start_sid else float('inf')
                            if dist < best_dist:
                                best_dist = dist
                                best_aid  = aid
                                best_goal = port

            # Phase 3: cycle breaking
            if best_aid is None:
                cycles = list(nx.simple_cycles(G_dep))
                if not cycles:
                    break
                empty_ports = self._find_empty_ports(agent_nodes, goals)
                if not empty_ports:
                    break
                # 모든 cycle 의 모든 agent push 후보 (task/non-task 무관)
                planned_set = set(planned.keys())
                cyc_agents = [a for cyc in cycles for a in cyc]
                prefer = [a for a in cyc_agents if a in planned_set] or cyc_agents
                for aid in prefer:
                    cur_n = agent_nodes[aid]
                    start_sid = self._find_stop_state(cur_n)
                    for port in empty_ports:
                        h = self._heuristic(port)
                        dist = h.get(start_sid, float('inf')) if start_sid else float('inf')
                        if dist < best_dist:
                            best_dist = dist
                            best_aid  = aid
                            best_goal = port

            if best_aid is None:
                break

            # ── SIPP search ──────────────────────────────────────────────────
            c_table = self._make_constraint_table(all_constraints, best_aid)
            t_start = starts.get(best_aid, 0.0)
            if best_aid in planned:
                t_start = planned[best_aid][-1][1]

            # Endpoint 할당 — allocator 가 있으면 agent 에게 endpoint 부여.
            # 다른 agent 가 이미 plan 받은 endpoints 는 자동 제외.
            if self._endpoint_allocator is not None:
                self._endpoint_allocator.allocate(
                    best_aid, agent_nodes[best_aid])

            # _is_goal_constrained 가 어느 agent 의 search 인지 알 수 있게.
            self._current_planning_aid = best_aid
            try:
                path = self._sipp_search(
                    agent_nodes[best_aid], best_goal, c_table,
                    start_time=t_start, timeout=timeout_per_agent
                )
            finally:
                self._current_planning_aid = None

            if path is None:
                if not quiet_fail:
                    expanded = getattr(self, '_dbg_states_expanded', 0)
                    reached = getattr(self, '_dbg_goal_reached', 0)
                    rejected = getattr(self, '_dbg_goal_rejected', 0)
                    max_g = getattr(self, '_dbg_max_g', 0)
                    timed_out = getattr(self, '_dbg_timed_out', False)
                    if timed_out:
                        cause = f'TIMEOUT (10s wall) max_g_explored={max_g:.1f}s'
                    elif reached == 0:
                        cause = f'NEVER_REACHED_GOAL (search exhausted, max_g={max_g:.1f}s)'
                    elif rejected == reached:
                        rr = getattr(self, '_dbg_reject_reason', '?')
                        rat = getattr(self, '_dbg_reject_at', '?')
                        rbl = getattr(self, '_dbg_reject_blocker', '?')
                        cause = f'GOAL_REJECTED {rejected}/{reached} | reason={rr} at={rat} blocker={rbl}'
                    else:
                        cause = f'PARTIAL reached={reached} rej={rejected}'
                    print(f"[FAIL] SIPP agent={best_aid} "
                          f"{agent_nodes[best_aid]} → {best_goal} | "
                          f"expanded={expanded} | {cause}")
                # NEVER_REACHED_GOAL → planner 입력 dump (offline 재현용)
                if not timed_out and reached == 0:
                    import pickle, os, time as _time
                    os.makedirs('logs', exist_ok=True)
                    fn = f'logs/sipp_exhausted_{int(_time.time())}_a{best_aid}.pkl'
                    snap = {
                        'agent_positions': dict(agent_nodes),
                        'agent_goals': dict(goals),
                        'starts': dict(starts),
                        'all_constraints': list(all_constraints),
                        'planned_paths': {a: list(p) for a, p in planned.items()},
                        'failed_agent': best_aid,
                        'failed_goal': best_goal,
                        'failed_start_time': t_start,
                        'non_task_agents': (list(non_task) if non_task else None),
                        'disable_alternate_goal': disable_alternate_goal,
                    }
                    try:
                        with open(fn, 'wb') as f:
                            pickle.dump(snap, f)
                        print(f"  [DUMP] planner state → {fn}")
                    except Exception as e:
                        print(f"  [DUMP] failed to save: {e}")
                if self.halt_on_sipp_fail:
                    raise SippFailure(
                        best_aid, agent_nodes[best_aid], best_goal,
                        len(all_constraints))
                # SIPP fail 시 alt goal 으로 substitute 안 함 (task semantic 보존).
                # 그 agent 만 skip + continue → 다른 agent plan 은 계속, 다음 tick
                # 에서 재시도.
                remaining.discard(best_aid)
                continue

            # Accumulate path
            if best_aid in planned:
                planned[best_aid] = planned[best_aid] + path[1:]
            else:
                planned[best_aid] = path

            # [FIX] Task goal arrival → L state + final_S(inf) 즉시 insert.
            # 이후 Phase 2/3 push 가 final_S 부터 이어지므로 LOADING/UNLOADING
            # dwell 이 항상 보존됨. arrival_idx 도 wrapper 에서 L state 위치로
            # 정확히 잡힘.
            is_task_goal = (best_goal == goals.get(best_aid))
            if is_task_goal:
                last_sid, last_t = planned[best_aid][-1]
                arr_node = self._node_of_state(last_sid)
                if arr_node is not None:
                    load_sid = f'L,{arr_node}'
                    if load_sid in self.graph.load_states_raw:
                        planned[best_aid].append((load_sid, last_t))
                        planned[best_aid].append((last_sid, last_t + self.dwell_time))

            # Update constraints
            new_cs = self._build_constraints(planned[best_aid], best_aid)
            all_constraints = [c for c in all_constraints if c['agent'] != best_aid]
            all_constraints.extend(new_cs)

            agent_nodes[best_aid] = best_goal
            if is_task_goal:
                remaining.discard(best_aid)

        result = PklPlanResult()
        for aid, path in planned.items():
            result.paths[aid] = path
            result.node_paths[aid] = self.path_to_nodes(path)
        return result

    def _find_alternate_goal(self, aid, cur_node, failed_goal, c_table,
                             t_start, timeout, agent_nodes, agent_goals):
        """Try alternate destinations (port + sidings) when SIPP fails."""
        occupied = set(agent_nodes.values())
        targeted = set(agent_goals.values())
        used = occupied | targeted | {failed_goal}
        for port in self._push_pool:
            if port in used:
                continue
            path = self._sipp_search(cur_node, port, c_table,
                                     start_time=t_start, timeout=timeout/2)
            if path is not None:
                return path, port
        return None

    def _find_empty_ports(self, agent_nodes, agent_goals):
        occupied = set(agent_nodes.values())
        targeted = set(agent_goals.values())
        used = occupied | targeted
        return [p for p in self._push_pool if p not in used]

    def assign_random_goals(self, agent_positions: Dict[int, str]) -> Dict[int, str]:
        goals = {}
        used = set()
        ports = list(self._port_nodes)
        random.shuffle(ports)
        for aid, cur in agent_positions.items():
            for p in ports:
                if p != cur and p not in used:
                    goals[aid] = p
                    used.add(p)
                    break
            else:
                for p in ports:
                    if p != cur:
                        goals[aid] = p
                        break
        return goals


class PklPlanResult:
    def __init__(self):
        self.paths:      Dict[int, List[Tuple[str, float]]] = {}
        self.node_paths: Dict[int, List[str]] = {}

    @property
    def makespan(self) -> float:
        if not self.paths:
            return 0.0
        return max(p[-1][1] for p in self.paths.values() if p)
