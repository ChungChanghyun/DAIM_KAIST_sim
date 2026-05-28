"""
elevator.py — 3D Shuttle용 엘리베이터 DES 모듈.

엘리베이터는 층간 화물 수직 이송을 담당하는 공유 자원입니다.
셔틀이 gate 노드에 화물을 내려놓으면, 엘리베이터가 해당 층으로 이동 →
화물 적재 → 목적 층 이동 → 화물 하역 → 목적 층 셔틀이 픽업.

DES 이벤트
──────────
  LIFT_MOVE_DONE   엘리베이터가 목표 층에 도착
  LIFT_XFER_DONE   화물 적재 또는 하역 완료

상태 전이
─────────
  IDLE         대기 중 (특정 층에 정지)
  MOVING       층간 이동 중
  LOADING      화물 적재 중 (셔틀 → 엘리베이터)
  UNLOADING    화물 하역 중 (엘리베이터 → 목적 층)

요청 처리 흐름
──────────────
  1) request(from_floor, to_floor, t)
     → 엘리베이터 idle이면 즉시 처리
     → busy이면 큐에 적재

  2) IDLE 상태에서 요청 수락:
     현재 층 == from_floor → 바로 LOADING
     현재 층 != from_floor → MOVING (from_floor로 이동)

  3) LIFT_MOVE_DONE:
     화물 없음(공차 이동) → LOADING 시작
     화물 있음(실차 이동) → UNLOADING 시작

  4) LIFT_XFER_DONE:
     LOADING 완료  → MOVING (to_floor로 이동)
     UNLOADING 완료 → IDLE + 콜백 + 다음 요청 처리

env_oht_des 연동
────────────────
  공유 힙에 이벤트를 삽입합니다.
  env.step() 루프에서:
      if lift_ctrl.handle_event(ev):
          continue

Usage
─────
  lift = Elevator(
      lift_id='Lift_1',
      floors={'1': 0.0, '2': 3000.0, '3': 6000.0},
      gate_nodes={'1': '00000365', '2': '00000031', '3': '00000011'},
      speed=1000.0,
      heap=env._heap,
  )

  # 화물 이송 요청
  lift.request(from_floor='1', to_floor='3', t=10.0,
               on_complete=lambda t: ...)
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque


# ── Lift DES event types ─────────────────────────────────────────────────────
LIFT_MOVE_DONE = 'LIFT_MOVE_DONE'
LIFT_XFER_DONE = 'LIFT_XFER_DONE'
LIFT_MOVE_TO_DONE = 'LIFT_MOVE_TO_DONE'   # MCS-driven single-leg move 완료

_LIFT_EVENT_TYPES = frozenset({LIFT_MOVE_DONE, LIFT_XFER_DONE, LIFT_MOVE_TO_DONE})

# 엘리베이터 전용 agent_id 범위 (주행 agent와 충돌 방지)
_LIFT_AGENT_BASE = -1000


# ── Lift states ──────────────────────────────────────────────────────────────
IDLE      = 'IDLE'
MOVING    = 'MOVING'
LOADING   = 'LOADING'
UNLOADING = 'UNLOADING'


# ── Event (env_oht_des.Event 호환) ───────────────────────────────────────────

class Event:
    __slots__ = ('t', 'kind', 'agent_id', 'token', 'data')

    def __init__(self, t: float, kind: str, agent_id: int,
                 token: int, data=None):
        self.t        = t
        self.kind     = kind
        self.agent_id = agent_id
        self.token    = token
        self.data     = data

    def __lt__(self, other):
        return self.t < other.t


# ── Transport Request ────────────────────────────────────────────────────────

@dataclass
class LiftRequest:
    """화물 수직 이송 요청."""
    from_floor:  str
    to_floor:    str
    t_requested: float
    on_complete: Optional[Callable] = None   # 하역 완료 시 콜백(t)
    on_pickup:   Optional[Callable] = None   # 적재 완료 시 콜백(t)
    priority:    int = 0
    req_id:      int = 0


# ── Lift Log Entry ───────────────────────────────────────────────────────────

@dataclass
class LiftLogEntry:
    time:      float
    lift_id:   str
    event:     str       # MOVE_START, MOVE_DONE, LOAD_START, LOAD_DONE,
                         # UNLOAD_START, UNLOAD_DONE, REQUEST, IDLE
    floor:     str = ''
    detail:    str = ''


# ── Elevator ─────────────────────────────────────────────────────────────────

class Elevator:
    """
    DES 기반 엘리베이터.

    Parameters
    ----------
    lift_id        : str   — 엘리베이터 식별자 (e.g. 'Lift_1')
    floors         : dict  — {floor_id: height_mm}  (e.g. {'1': 0, '2': 3000, '3': 6000})
    gate_nodes     : dict  — {floor_id: node_id}  각 층의 gate 노드
    speed          : float — 수직 이동 속도 (mm/s)
    heap           : list  — 공유 이벤트 힙 (env._heap)
    xfer_duration  : float — 적재/하역 소요 시간 (초, default 3.0)
    capacity       : int   — 동시 적재 가능 화물 수 (default 1)
    init_floor     : str   — 초기 위치 층 (default: floors의 첫 번째 키)
    agent_id_offset: int   — 이벤트 agent_id에 사용할 오프셋
                             (여러 엘리베이터 구분용, default 0)
    """

    def __init__(self, lift_id: str,
                 floors: Dict[str, float],
                 gate_nodes: Dict[str, str],
                 speed: float = 1000.0,
                 heap: list = None,
                 xfer_duration: float = 3.0,
                 capacity: int = 1,
                 init_floor: str = None,
                 agent_id_offset: int = 0):

        self.id           = lift_id
        self.floors       = dict(floors)        # floor_id -> height_mm
        self.gate_nodes   = dict(gate_nodes)    # floor_id -> node_id
        self.speed        = speed
        self._heap        = heap if heap is not None else []
        self.xfer_duration = xfer_duration
        self.capacity     = capacity

        self._agent_id    = _LIFT_AGENT_BASE - agent_id_offset
        self._token       = 0

        # 런타임 상태
        self.cur_floor    = init_floor or next(iter(floors))
        self.state        = IDLE
        self.cargo_count  = 0
        self._cur_req: Optional[LiftRequest] = None

        # 이동 보간용 (시각화)
        self.move_from_floor: Optional[str] = None
        self.move_to_floor:   Optional[str] = None
        self.move_start_t:    float = 0.0
        self.move_duration:   float = 0.0

        # 요청 큐
        self._queue: deque[LiftRequest] = deque()
        self._next_req_id = 0

        # 로그
        self._log: List[LiftLogEntry] = []

        # KPI 추적
        self.t_registered  = 0.0
        self.total_busy    = 0.0
        self._t_last_busy  = 0.0
        self.jobs_done     = 0
        self.total_moves   = 0
        self.total_distance = 0.0   # mm

    # ═════════════════════════════════════════════════════════════════════════
    #  공개 인터페이스
    # ═════════════════════════════════════════════════════════════════════════

    def request(self, from_floor: str, to_floor: str, t: float,
                on_complete: Callable = None,
                on_pickup: Callable = None,
                priority: int = 0) -> int:
        """
        화물 이송 요청.

        Parameters
        ----------
        from_floor   : 화물이 있는 층
        to_floor     : 목적 층
        t            : 요청 시각
        on_complete  : 하역 완료 콜백 — on_complete(t)
        on_pickup    : 적재 완료 콜백 — on_pickup(t)  (셔틀 해제 타이밍 등)
        priority     : 높을수록 우선

        Returns
        -------
        req_id : int
        """
        req = LiftRequest(
            from_floor=from_floor, to_floor=to_floor,
            t_requested=t,
            on_complete=on_complete, on_pickup=on_pickup,
            priority=priority, req_id=self._next_req_id,
        )
        self._next_req_id += 1
        self._record(t, 'REQUEST',
                     detail=f'{from_floor}→{to_floor} (req {req.req_id})')

        if self.state == IDLE and self.cargo_count < self.capacity:
            self._start_request(req, t)
        else:
            # 우선순위 기반 삽입
            inserted = False
            for i, q in enumerate(self._queue):
                if req.priority > q.priority:
                    self._queue.insert(i, req)
                    inserted = True
                    break
            if not inserted:
                self._queue.append(req)

        return req.req_id

    def gate_node(self, floor_id: str) -> str:
        """해당 층의 gate 노드 ID."""
        return self.gate_nodes[floor_id]

    # ── MCS-driven single-leg move ──────────────────────────────────────────
    def move_to(self, target_floor: str, t: float,
                on_arrive: Callable[[float], None]):
        """단일 leg 수직 이동 (load/unload 없이 이동만). MCS 가 dispatch 시 호출.

        target_floor 도달 시 on_arrive(t) 콜백 1회 fire.
        cur_floor 갱신, state=MOVING 유지 (busy). 도착 시 state는 IDLE 가 아닌
        그대로 두고 (MCS 의 LOADING/UNLOADING dwell 동안 차지 표시 유지),
        다음 move_to 호출 시 다시 갱신.
        """
        if target_floor not in self.floors:
            raise ValueError(f'unknown floor {target_floor} for {self.id}')

        if target_floor == self.cur_floor:
            # 즉시 도착 콜백 (이벤트 1tick 지연 없이)
            on_arrive(t)
            return

        dt = self.travel_time(self.cur_floor, target_floor)
        self.state = MOVING
        self.move_from_floor = self.cur_floor
        self.move_to_floor   = target_floor
        self.move_start_t    = t
        self.move_duration   = dt
        self.total_moves    += 1
        self.total_distance += abs(
            self.floors[target_floor] - self.floors[self.cur_floor])
        self._record(t, 'MOVE_START',
                     floor=self.cur_floor,
                     detail=f'(MCS)→{target_floor} ({dt:.1f}s)')
        # token 증가시켜 stale 이벤트 방지
        self._token += 1
        ev = Event(t + dt, LIFT_MOVE_TO_DONE, self._agent_id,
                   self._token, data={'target': target_floor,
                                       'on_arrive': on_arrive})
        heapq.heappush(self._heap, ev)

    def travel_time(self, from_floor: str, to_floor: str) -> float:
        """두 층 간 이동 소요 시간 (초)."""
        h1 = self.floors[from_floor]
        h2 = self.floors[to_floor]
        return abs(h2 - h1) / self.speed if self.speed > 0 else 0.0

    # ═════════════════════════════════════════════════════════════════════════
    #  DES 연동
    # ═════════════════════════════════════════════════════════════════════════

    def handle_event(self, ev) -> bool:
        """
        엘리베이터 이벤트이면 처리하고 True 반환.

        env.step() 루프에서:
            if lift.handle_event(ev):
                continue
        """
        if ev.kind not in _LIFT_EVENT_TYPES:
            return False
        if ev.agent_id != self._agent_id:
            return False
        if ev.token != self._token:
            return True   # stale

        if ev.kind == LIFT_MOVE_DONE:
            self._on_move_done(ev.t)
        elif ev.kind == LIFT_XFER_DONE:
            self._on_xfer_done(ev.t)
        elif ev.kind == LIFT_MOVE_TO_DONE:
            data = ev.data or {}
            target = data.get('target')
            cb = data.get('on_arrive')
            if target:
                self.cur_floor = target
            self.state = IDLE   # 일단 IDLE — MCS binding 으로 점유 추적
            self.move_from_floor = None
            self.move_to_floor   = None
            self._record(ev.t, 'MOVE_DONE', floor=self.cur_floor,
                         detail='(MCS)')
            if cb:
                cb(ev.t)
        return True

    # ═════════════════════════════════════════════════════════════════════════
    #  Internal — 요청 처리
    # ═════════════════════════════════════════════════════════════════════════

    def _start_request(self, req: LiftRequest, t: float):
        """요청 수락 → 공차 이동 또는 즉시 적재."""
        self._cur_req = req
        self.state = MOVING
        self._t_last_busy = t

        if self.cur_floor == req.from_floor:
            # 이미 해당 층 → 바로 적재 시작
            self._start_loading(t)
        else:
            # 공차 이동
            dt = self.travel_time(self.cur_floor, req.from_floor)
            self.move_from_floor = self.cur_floor
            self.move_to_floor   = req.from_floor
            self.move_start_t    = t
            self.move_duration   = dt
            self._record(t, 'MOVE_START',
                         floor=self.cur_floor,
                         detail=f'→{req.from_floor} ({dt:.1f}s)')
            self.total_moves += 1
            self.total_distance += abs(
                self.floors[req.from_floor] - self.floors[self.cur_floor])
            self._post(t + dt, LIFT_MOVE_DONE, data='to_pickup')

    def _start_loading(self, t: float):
        """적재 시작."""
        self.state = LOADING
        self.cur_floor = self._cur_req.from_floor
        self._record(t, 'LOAD_START', floor=self.cur_floor)
        self._post(t + self.xfer_duration, LIFT_XFER_DONE, data='loading')

    def _start_delivering(self, t: float):
        """적재 완료 → 목적 층으로 실차 이동."""
        req = self._cur_req
        self.state = MOVING
        dt = self.travel_time(self.cur_floor, req.to_floor)
        self.move_from_floor = self.cur_floor
        self.move_to_floor   = req.to_floor
        self.move_start_t    = t
        self.move_duration   = dt
        self._record(t, 'MOVE_START',
                     floor=self.cur_floor,
                     detail=f'→{req.to_floor} ({dt:.1f}s, loaded)')
        self.total_moves += 1
        self.total_distance += abs(
            self.floors[req.to_floor] - self.floors[self.cur_floor])
        self._post(t + dt, LIFT_MOVE_DONE, data='to_dest')

    def _start_unloading(self, t: float):
        """하역 시작."""
        self.state = UNLOADING
        self.cur_floor = self._cur_req.to_floor
        self._record(t, 'UNLOAD_START', floor=self.cur_floor)
        self._post(t + self.xfer_duration, LIFT_XFER_DONE, data='unloading')

    # ═════════════════════════════════════════════════════════════════════════
    #  Internal — 이벤트 핸들러
    # ═════════════════════════════════════════════════════════════════════════

    def _on_move_done(self, t: float):
        """이동 완료."""
        data = 'to_pickup'  # default
        # data 판별은 _post에서 저장한 값 활용
        # 실제로는 cargo 유무로 판별
        if self.cargo_count == 0:
            # 공차 이동 완료 → 적재 시작
            self.cur_floor = self._cur_req.from_floor
            self._record(t, 'MOVE_DONE', floor=self.cur_floor)
            self._start_loading(t)
        else:
            # 실차 이동 완료 → 하역 시작
            self.cur_floor = self._cur_req.to_floor
            self._record(t, 'MOVE_DONE', floor=self.cur_floor)
            self._start_unloading(t)

    def _on_xfer_done(self, t: float):
        """적재 또는 하역 완료."""
        if self.state == LOADING:
            # 적재 완료
            self.cargo_count += 1
            self._record(t, 'LOAD_DONE', floor=self.cur_floor)
            if self._cur_req.on_pickup:
                self._cur_req.on_pickup(t)
            # 목적 층으로 이동
            self._start_delivering(t)

        elif self.state == UNLOADING:
            # 하역 완료
            self.cargo_count -= 1
            self._record(t, 'UNLOAD_DONE', floor=self.cur_floor)
            self.total_busy += (t - self._t_last_busy)
            self.jobs_done += 1
            if self._cur_req.on_complete:
                self._cur_req.on_complete(t)
            self._cur_req = None
            self.state = IDLE
            self._record(t, 'IDLE', floor=self.cur_floor)
            # 다음 요청 처리
            self._process_next(t)

    def _process_next(self, t: float):
        """큐에 대기 중인 다음 요청 처리."""
        if self._queue and self.cargo_count < self.capacity:
            req = self._queue.popleft()
            self._start_request(req, t)

    # ═════════════════════════════════════════════════════════════════════════
    #  Internal — 이벤트 게시
    # ═════════════════════════════════════════════════════════════════════════

    def _post(self, t: float, kind: str, data=None):
        self._token += 1
        heapq.heappush(self._heap,
                       Event(t, kind, self._agent_id, self._token, data))

    def _record(self, t: float, event: str, floor: str = '',
                detail: str = ''):
        self._log.append(LiftLogEntry(
            time=t, lift_id=self.id, event=event,
            floor=floor or self.cur_floor, detail=detail,
        ))

    # ═════════════════════════════════════════════════════════════════════════
    #  Queries / KPI
    # ═════════════════════════════════════════════════════════════════════════

    def move_progress(self, t_now: float) -> float:
        """MOVING 상태일 때 0.0(출발)~1.0(도착) 진행도."""
        if self.state != MOVING or self.move_duration <= 0:
            return 0.0
        return min(1.0, max(0.0, (t_now - self.move_start_t) / self.move_duration))

    @property
    def event_log(self) -> List[LiftLogEntry]:
        return list(self._log)

    @property
    def is_idle(self) -> bool:
        return self.state == IDLE

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    def kpi(self, t_now: float) -> dict:
        """엘리베이터 KPI."""
        elapsed = t_now - self.t_registered
        busy = self.total_busy
        if self.state != IDLE:
            busy += (t_now - self._t_last_busy)
        return {
            'lift_id':        self.id,
            'state':          self.state,
            'cur_floor':      self.cur_floor,
            'queue_length':   len(self._queue),
            'elapsed':        elapsed,
            'busy_time':      busy,
            'idle_time':      elapsed - busy,
            'utilization':    busy / elapsed if elapsed > 0 else 0.0,
            'jobs_done':      self.jobs_done,
            'total_moves':    self.total_moves,
            'total_distance': self.total_distance,
            'avg_cycle':      busy / self.jobs_done if self.jobs_done > 0 else 0.0,
        }


# ── ElevatorController ───────────────────────────────────────────────────────

class ElevatorController:
    """
    복수 엘리베이터 통합 관리.

    Parameters
    ----------
    heap : list — 공유 이벤트 힙
    """

    def __init__(self, heap: list):
        self._heap  = heap
        self._lifts: Dict[str, Elevator] = {}

    def add_lift(self, lift: Elevator):
        self._lifts[lift.id] = lift

    def add_lift_from_map(self, lift_data: dict,
                          area_heights: Dict[str, float],
                          speed: float = 1000.0,
                          xfer_duration: float = 3.0,
                          capacity: int = 1):
        """
        KaistTB_map.json의 lift 항목에서 Elevator 생성.

        Parameters
        ----------
        lift_data     : lifts 배열의 한 항목
        area_heights  : {floor_id: height_mm}  (areas.viewShift.y 등에서 추출)
        """
        lift_id = lift_data['id']
        floors = {}
        gate_nodes = {}
        for fl in lift_data['floors']:
            fid = fl['id']
            floors[fid] = area_heights.get(fid, 0.0)
            # gate 노드: 첫 번째 entry/exit 노드 사용
            if fl.get('gates'):
                gate = fl['gates'][0]
                # entry/exit 동일한 경우 하나만
                nodes = gate.get('entryNodes', []) or gate.get('exitNodes', [])
                if nodes:
                    gate_nodes[fid] = nodes[0]

        offset = len(self._lifts)
        lift = Elevator(
            lift_id=lift_id,
            floors=floors,
            gate_nodes=gate_nodes,
            speed=speed,
            heap=self._heap,
            xfer_duration=xfer_duration,
            capacity=capacity,
            agent_id_offset=offset,
        )
        self._lifts[lift_id] = lift
        return lift

    def handle_event(self, ev) -> bool:
        """어느 엘리베이터의 이벤트인지 판별하여 위임."""
        if ev.kind not in _LIFT_EVENT_TYPES:
            return False
        for lift in self._lifts.values():
            if lift.handle_event(ev):
                return True
        return False

    def get_lift(self, lift_id: str) -> Optional[Elevator]:
        return self._lifts.get(lift_id)

    def request(self, lift_id: str, from_floor: str, to_floor: str,
                t: float, **kwargs) -> int:
        """특정 엘리베이터에 이송 요청."""
        return self._lifts[lift_id].request(
            from_floor, to_floor, t, **kwargs)

    def select_lift(self, from_floor: str, to_floor: str,
                    t: float) -> Optional[str]:
        """
        가장 빨리 서비스할 수 있는 엘리베이터 선택.

        기준: idle 우선 → 현재 층 == from_floor 우선 → 이동 시간 최소
        """
        best_id   = None
        best_cost = float('inf')
        for lift in self._lifts.values():
            if from_floor not in lift.floors or to_floor not in lift.floors:
                continue
            # idle이고 현재 층이면 즉시 가능
            if lift.is_idle and lift.cur_floor == from_floor:
                return lift.id
            # 비용: 대기 큐 * 예상 사이클 + 공차 이동 시간
            queue_cost = lift.queue_length * (
                lift.travel_time(from_floor, to_floor) + 2 * lift.xfer_duration)
            if not lift.is_idle:
                queue_cost += lift.travel_time(lift.cur_floor, from_floor)
            move_cost = lift.travel_time(lift.cur_floor, from_floor)
            cost = queue_cost + move_cost
            if cost < best_cost:
                best_cost = cost
                best_id   = lift.id
        return best_id

    def request_auto(self, from_floor: str, to_floor: str,
                     t: float, **kwargs) -> Optional[Tuple[str, int]]:
        """
        최적 엘리베이터 자동 선택 + 요청.

        Returns (lift_id, req_id) or None.
        """
        lid = self.select_lift(from_floor, to_floor, t)
        if lid is None:
            return None
        rid = self.request(lid, from_floor, to_floor, t, **kwargs)
        return lid, rid

    @property
    def lifts(self) -> Dict[str, Elevator]:
        return self._lifts

    def kpi(self, t_now: float) -> dict:
        """전체 엘리베이터 KPI."""
        details = []
        total_jobs = 0
        total_busy = 0.0
        total_elapsed = 0.0
        for lid in sorted(self._lifts):
            lk = self._lifts[lid].kpi(t_now)
            details.append(lk)
            total_jobs += lk['jobs_done']
            total_busy += lk['busy_time']
            total_elapsed += lk['elapsed']
        return {
            'lifts_total':     len(self._lifts),
            'total_jobs':      total_jobs,
            'avg_utilization': (total_busy / total_elapsed
                                if total_elapsed > 0 else 0.0),
            'lift_details':    details,
        }

    def print_kpi(self, t_now: float):
        k = self.kpi(t_now)
        lines = [
            '',
            '═══════════════════════════════════════════',
            f'  Elevator KPI  (t = {t_now:.1f}s)',
            '═══════════════════════════════════════════',
            f'  Lifts: {k["lifts_total"]}   '
            f'Jobs: {k["total_jobs"]}   '
            f'Avg util: {k["avg_utilization"]:.1%}',
            '',
        ]
        for ld in k['lift_details']:
            lines.append(
                f'  {ld["lift_id"]:<8s}  '
                f'{ld["state"]:<10s}  '
                f'F{ld["cur_floor"]}  '
                f'util {ld["utilization"]:>5.1%}  '
                f'jobs {ld["jobs_done"]:>3d}  '
                f'queue {ld["queue_length"]:>2d}  '
                f'moves {ld["total_moves"]:>3d}  '
                f'avg_cycle {ld["avg_cycle"]:>6.1f}s'
            )
        lines += ['═══════════════════════════════════════════', '']
        print('\n'.join(lines))
