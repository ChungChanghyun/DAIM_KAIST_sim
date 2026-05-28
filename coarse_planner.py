"""Coarse-grained reservation planner for unidirectional layouts (e.g. songdo).

Plan time: shortest path *ignoring* inter-agent constraints (= fast Dijkstra
on state graph).

Execution time: segment-level token passing. AGV holds one segment at a
time. Before entering next segment, check if its corridor (= chain of
out-deg=1 nodes between two checkpoints) is free. If free, claim it; if
not, wait at current checkpoint.

Checkpoints = branching nodes (out-deg ≥ 2) ∪ ports ∪ sidings.
Segments = directed corridor sequence between two checkpoints.

Interface: drop-in replacement for `PklPrioritizedPlanner.plan()`.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pkl_loader import PklMapGraph
from pkl_prioritized_planner import PklPrioritizedPlanner, PklPlanResult


class CoarsePlanner:
    """Coarse-grained reservation planner. Reuses PklPrioritizedPlanner's SIPP
    for shortest-path computation, just with empty c_table.
    """

    def __init__(self, graph: PklMapGraph,
                 push_extras: Optional[List[str]] = None):
        self.graph = graph
        self._port_nodes: List[str] = list(set(graph.ports.values()))
        self._push_pool: List[str] = list(self._port_nodes)
        if push_extras:
            for n in push_extras:
                if n in graph.nodes and n not in self._push_pool:
                    self._push_pool.append(n)

        # Identify checkpoints — branching (out-deg≥2) + ports + sidings
        self.checkpoints = set()
        for nid in graph.nodes:
            if len(graph.adj.get(nid, [])) >= 2:
                self.checkpoints.add(nid)
        self.checkpoints.update(self._port_nodes)
        if push_extras:
            self.checkpoints.update(push_extras)

        # Build segments — directed corridor between two checkpoints
        # segments[seg_id] = {'from': cp_a, 'to': cp_b, 'nodes': [intermediate]}
        # seg_id = tuple(cp_a, cp_b)
        self.segments: Dict[Tuple[str, str], List[str]] = {}
        self.node_to_segment: Dict[str, Tuple[str, str]] = {}
        self._build_segments()

        # Push pool 확장: *충분히 긴 corridor* (interm len >= 3) 의 last-grey
        # 만. _extend_with_exit_coarse 의 push destination 과 일치.
        # plan validation 시 'NOT port/siding' warning 회피.
        for seg_id, interm in self.segments.items():
            if len(interm) >= 3 and interm[-1] not in self._push_pool:
                self._push_pool.append(interm[-1])

        # Reuse base SIPP for shortest-path computation
        self._base = PklPrioritizedPlanner(graph, push_extras=push_extras)

        # Dwell config — match base planner
        self.dwell_time = getattr(self._base, 'dwell_time', 3.0)
        self.halt_on_sipp_fail = False

    def _build_segments(self):
        """For each (checkpoint A, outgoing neighbor): follow corridor until
        next checkpoint. Record segment."""
        for cp in self.checkpoints:
            for next_nid in self.graph.adj.get(cp, []):
                interm: List[str] = []
                cur = next_nid
                visited = {cp}
                while cur not in self.checkpoints:
                    if cur in visited:
                        break
                    visited.add(cur)
                    interm.append(cur)
                    nxt = self.graph.adj.get(cur, [])
                    if len(nxt) != 1:
                        break  # branching mid-corridor (shouldn't happen
                               # since we already include all branching as cp)
                    cur = nxt[0]
                if cur in self.checkpoints:
                    seg_id = (cp, cur)
                    self.segments[seg_id] = interm
                    for n in interm:
                        self.node_to_segment[n] = seg_id

    def _find_all_first_non_cut(self, start_node: str,
                                  cut_nodes: set) -> List[str]:
        """start 에서 forward BFS, exit zone (port + cut) 빠져나오는 *모든*
        첫 grey node 반환. 여러 lane / branch 있는 경우 다 수집.

        start 가 이미 grey 면 [start] 반환.
        """
        exit_zone = cut_nodes | set(self._port_nodes)
        if start_node not in exit_zone:
            return [start_node]
        from collections import deque
        queue = deque([start_node])
        visited = {start_node}
        result = []
        while queue:
            n = queue.popleft()
            if n != start_node and n not in exit_zone:
                result.append(n)
                continue   # 첫 grey 도달 — 더 안 진행
            for s in self.graph.adj.get(n, []):
                if s not in visited:
                    visited.add(s)
                    queue.append(s)
        return result

    def plan(self, agent_positions: Dict[int, str],
             agent_goals: Dict[int, str],
             existing_constraints=None,
             start_times: Optional[Dict[int, float]] = None,
             timeout_per_agent: float = 10.0,
             disable_alternate_goal: bool = False,
             quiet_fail: bool = False,
             non_task_agents=None,
             port_exit_blocked_per_agent: Optional[Dict[int, set]] = None,
             cut_nodes: Optional[set] = None) -> PklPlanResult:
        """각 agent 의 shortest path 산출.

        기본: c_table={} (= 다른 AGV constraint 무시).
        Port-start auto-detect: agent.start ∈ self._port_nodes 이고
        port_exit_blocked_per_agent[aid] 가 nonempty + cut_nodes 도 주어지면
        2-stage SIPP. 첫 stage = port→첫 grey (constraint-aware, blocked 회피).
        둘째 stage = 첫 grey→goal (free). 다른 케이스는 기본 동작.
        """
        starts = dict(start_times or {})
        result = PklPlanResult()
        exit_blocked = port_exit_blocked_per_agent or {}
        cut_set = cut_nodes or set()

        for aid, goal in agent_goals.items():
            start = agent_positions.get(aid)
            if not start or not goal:
                continue
            t_start = starts.get(aid, 0.0)

            # Port-start exit-commit 모드 활성 조건:
            #   1) start ∈ port_nodes (= AGV 가 port 에서 출발)
            #   2) blocked 가 nonempty (= 회피할 cells 있음)
            #   3) cut_nodes 제공 (= exit zone 식별 가능)
            blocked = exit_blocked.get(aid, set()) - {start, goal}
            path = None
            if start in self._port_nodes and blocked and cut_set:
                exit_candidates = self._find_all_first_non_cut(start, cut_set)
                for exit_node in exit_candidates:
                    if not exit_node or exit_node == start or exit_node == goal:
                        continue
                    if exit_node in blocked:
                        continue
                    c_table_nodes = blocked - {exit_node}
                    c_table = {n: [(0.0, float('inf'))]
                               for n in c_table_nodes}
                    leg1 = self._base._sipp_search(
                        start, exit_node, c_table=c_table,
                        start_time=t_start, timeout=timeout_per_agent)
                    if not (leg1 and len(leg1) >= 2):
                        continue
                    leg1_end_sid, leg1_end_t = leg1[-1]
                    leg2 = self._base._sipp_search(
                        exit_node, goal, c_table={},
                        start_time=leg1_end_t,
                        timeout=timeout_per_agent,
                        start_sid=leg1_end_sid)
                    if leg2 and len(leg2) >= 2:
                        path = leg1 + leg2[1:]
                        break

            if path is None:
                # Fallback: 기존 free SIPP
                path = self._base._sipp_search(
                    start, goal,
                    c_table={},
                    start_time=t_start,
                    timeout=timeout_per_agent,
                )

            if path is None:
                if not quiet_fail:
                    print(f"[COARSE-FAIL] agent {aid}, {start} → {goal} "
                          f"(graph 자체 unreachable)")
                continue

            # Add L state + post-dwell S (LOADING/UNLOADING dwell)
            arr_sid, arr_t = path[-1]
            arr_node = self._base._node_of_state(arr_sid)
            if arr_node is not None:
                load_sid = f'L,{arr_node}'
                if load_sid in self.graph.load_states_raw:
                    path.append((load_sid, arr_t))
                    path.append((arr_sid, arr_t + self.dwell_time))

            # Validation: path 종점이 *port 또는 siding* 인지 검사.
            # Branching (out-deg >= 2) 또는 corridor 가 종점이면 atomic claim
            # 규칙 위반 - AGV 가 거기서 stop 했을 때 stuck 가능.
            final_node = self._base._node_of_state(path[-1][0])
            if final_node:
                is_port = final_node in self._port_nodes
                is_siding = final_node in self._push_pool and not is_port
                if not is_port and not is_siding:
                    out_deg = len(self.graph.adj.get(final_node, []))
                    print(f'[COARSE-WARN] agent {aid} path ends at {final_node} '
                          f'(out-deg={out_deg}, NOT port/siding). Goal was {goal}. '
                          f'AGV may stuck if no atomic forward claim possible.')

            result.paths[aid] = path
            result.node_paths[aid] = self._base.path_to_nodes(path)

        return result

    # Compat shim — PklPrioritizedPlanner interface 흉내
    def _get_state(self, state_id):
        return self._base._get_state(state_id)

    def _find_stop_state(self, node_id):
        return self._base._find_stop_state(node_id)

    def _build_constraints(self, path, agent_id):
        return self._base._build_constraints(path, agent_id)

    @property
    def _heuristic(self):
        return self._base._heuristic
