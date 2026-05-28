"""
vis_mcs_unified.py — MCS 통합 시뮬레이터 (KaistTB map)

vis_combined.py 기반, MCS(Material Control System)를 연결하여
작업 생성 → 할당 → LOADING → 배송 → UNLOADING → 완료 사이클을 DES로 처리.

OHT: OHT_A 서브네트워크, OHTEnvironmentDES (세그먼트 큐 + 전방감지)
AGV: AMR_A PklMapGraph, PklPrioritizedPlanner + TAPGEnvironment (SIPP + TAPG DAG)

Controls
────────
  SPACE     : Start / Pause
  R         : Reset
  S         : Shuffle paths (both OHT & AGV)
  O / L     : Add / Remove OHT vehicle
  N / P     : Add / Remove AGV vehicle
  +/-       : Sim speed up / down
  Mouse drag: Pan
  Wheel     : Zoom
  Q / ESC   : Quit
"""
from __future__ import annotations
import sys, os, json, math, random, collections, csv
import pygame
import pygame_gui
from pygame_gui.elements import (UIWindow, UISelectionList, UIDropDownMenu,
                                 UIButton, UILabel, UITextBox)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 부모 KAIST 디렉토리도 path 에 추가: graph_des_v6 shim 이 ../graph_des_v6.py
# 를 importlib 으로 로드하는데, 그쪽이 vehicle_state 등 형제 모듈을 import 하므로
# 부모를 sys.path 에 두지 않으면 ModuleNotFoundError 발생.
sys.path.insert(0, os.path.dirname(_HERE))
from env_oht_v6_adapter import (OHTMap, OHTAgent, OHTEnvironmentDES,
                          IDLE, MOVING, FOLLOWING, BLOCKED, DONE)
from env_tapg import (TAPGAgent, TAPGEnvironment,
                      IDLE as AGV_IDLE, MOVING as AGV_MOVING,
                      WAITING as AGV_WAITING, ROTATING as AGV_ROTATING,
                      DONE as AGV_DONE)
from pkl_loader import PklMapGraph
from pkl_prioritized_planner import PklPrioritizedPlanner
from env_3ds import FloorGraph
from elevator import (Elevator, ElevatorController, LiftRequest,
                      IDLE as LIFT_IDLE, MOVING as LIFT_MOVING,
                      LOADING as LIFT_LOADING, UNLOADING as LIFT_UNLOADING)
from mcs_unified import (MCSEngine, VehicleJobState, LoadState, Recipe, RecipeStage,
                         LOAD_CREATED, TRY_ASSIGN, VEHICLE_ARRIVED, DWELL_DONE)

JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'Maps', 'KaistTB.map_latest.json')
AMR_PKL   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'KaistTB_AMR_A.pkl')
COLLISION_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'collision_log.txt')

# ── Colors ────────────────────────────────────────────────────────────────────
BG          = (18,  20,  28)
COL_SEG     = (40,  55,  70)
COL_SEG_ARR = (60,  80, 100)
COL_NODE    = (70,  85, 100)
COL_PORT    = (60, 220, 120)
COL_TEXT    = (200, 210, 220)
COL_DIM     = (80,  90, 100)
COL_WHITE   = (255, 255, 255)

COL_FOLLOWING = (255, 210,  50)
COL_BLOCKED   = (220,  60,  60)
COL_HEADLIGHT = (255, 255, 180)
COL_ZCU_FREE  = (60,  180,  60)
COL_ZCU_HELD  = (220, 100,  40)

# OHT palette (test_graph_v6.py 와 동일).
# MOVING (ACCEL/CRUISE/DECEL):
#   has job (TO_PICKUP/TO_DROP) → green / no job (via_push) → blue
# LOADING → purple. STOP → reason 별 분류.
OHT_COL_MOV_JOB        = ( 60, 200,  60)   # green
OHT_COL_MOV_PUSH       = ( 80, 140, 255)   # blue
OHT_COL_LOADING        = (200,  60, 200)   # purple
OHT_COL_STOP_FREE      = ( 90, 200, 230)   # cyan: idle free, pushable
OHT_COL_STOP_DEST      = ( 60, 220, 220)   # cyan-green: at dest
OHT_COL_STOP_LEADER    = (255,  90,  60)   # orange-red: blocked by leader
OHT_COL_STOP_ZCU       = (255,  60, 180)   # magenta-red: blocked by ZCU
OHT_COL_IDLE_GRAY      = (100, 100, 100)   # gray (= no path)

# area → 배경 노드/세그먼트 색상
AREA_COLORS = {
    '3DS_F1': (35, 60, 80),
    '3DS_F2': (35, 70, 55),
    '3DS_F3': (65, 50, 35),
    'OHT_A':  (60, 35, 70),
    'AMR_A':  (65, 65, 30),
    '':       (50, 55, 65),
}
AREA_SEG_COLORS = {
    '3DS_F1': (50, 80, 110),
    '3DS_F2': (50, 95,  70),
    '3DS_F3': (90, 70,  45),
    'OHT_A':  (85, 50, 100),
    'AMR_A':  (90, 90,  35),
    '':       (60, 70,  80),
}

# OHT agent colors (purple / blue tones)
OHT_COLORS = [
    (180,  80, 200), (140, 100, 255), (200, 120, 255),
    (120,  80, 220), (180, 140, 255),
]
# AGV agent color (통일)
AGV_COLOR = (100, 220, 120)
AGV_COLORS = [AGV_COLOR]  # 호환성 유지

# 3DS shuttle colors per floor
S3D_COLORS = {
    '3DS_F1': (80,  180, 255),
    '3DS_F2': (80,  230, 130),
    '3DS_F3': (255, 180,  80),
}
S3D_FLOOR_IDS = ['3DS_F1', '3DS_F2', '3DS_F3']

# ── Area layout offsets ──────────────────────────────────────────────────────
# 각 area를 겹치지 않게 2D에 펼치기 위한 (dx, dy) 오프셋 (mm 단위).
# 원본 좌표 범위:
#   AMR_A : x=[530,18152]  y=[740,14284]   w=17622 h=13544
#   OHT_A : x=[1870,17890] y=[890,3993]    w=16020 h=3103
#   3DS_*:  x=[1115,11115] y=[12721,14721] w=10000 h=2000 (3층 겹침)
#
# 레이아웃:  AMR_A(좌)  |  3DS(우, 세로 3층)  |  OHT_A(하단)
GAP = 1500  # area 간 간격 (mm)

# 레이아웃:
#   좌측 세로: AMR_A(위) → OHT_A(아래, 밀착)
#   우측 가로: 3DS_F1 | 3DS_F2 | 3DS_F3
#
# 원본 좌표:
#   AMR_A : x=[530,18152]  y=[740,14284]   w=17622 h=13544
#   OHT_A : x=[1870,17890] y=[890,3993]    w=16020 h=3103
#   3DS_*:  x=[1115,11115] y=[12721,14721] w=10000 h=2000
_3DS_W = 10000  # 3DS 한 층 원본 폭
_3DS_H = 2000   # 3DS 한 층 원본 높이
_3DS_SCALE = 1.5 # 3DS 표시 스케일 (실제 좌표 대비)

# 레이아웃 (위→아래):
#   3DS (아이소메트릭 적층: F3 위, F2 중간, F1 아래)
#   AMR_A
#   OHT_A
_AMR_TOP = 14284
_3DS_BOT = 12721
_3DS_BASE_Y = _AMR_TOP + GAP - _3DS_BOT
_3DS_BASE_X = -1115 + 530

# 아이소메트릭 층간 오프셋 (스케일 적용된 크기 기준)
_ISO_DX = 2000                       # 층당 x 이동
_ISO_DY = _3DS_H * _3DS_SCALE + GAP  # 층당 y 이동 (스케일된 높이 + 간격)

AREA_OFFSETS = {
    'AMR_A':  (0, 0),
    'OHT_A':  (0, 740 - GAP - 3993),
    '3DS_F1': (_3DS_BASE_X, _3DS_BASE_Y),
    '3DS_F2': (_3DS_BASE_X + _ISO_DX, _3DS_BASE_Y + _ISO_DY),
    '3DS_F3': (_3DS_BASE_X + 2*_ISO_DX, _3DS_BASE_Y + 2*_ISO_DY),
}
# area 미지정 노드 (포트 등)
_DEFAULT_OFFSET = (0, 0)

SIM_SPEEDS       = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SIM_SPEED_LABELS = ['0.1×', '0.25×', '0.5×', '1×', '2×', '4×', '8×', '16×']
DEFAULT_SPD_IDX  = 3

WIN_W, WIN_H = 1400, 900
SIDE_W       = 260
MAP_W        = WIN_W - SIDE_W
MAP_PAD      = 40
FPS          = 60
MAX_AGENTS   = 30   # ports (10) + strict-safe sidings (~19) 까지 수용


# ── Full-map background data ────────────────────────────────────────────────

def _area_offset(area: str):
    """Return (dx, dy) offset for the given area."""
    return AREA_OFFSETS.get(area, _DEFAULT_OFFSET)


def load_fromto_matrix(csv_path: str, port_ids: list) -> dict:
    """FromTo CSV → (src, dst) → λ (trips/sec) 딕셔너리.

    CSV 의 행/열 라벨 중 port_ids 에 있는 것만 채택. 대각선 0, 음수 0 으로 정규화.
    """
    table: dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        col_ports = header[1:]
        for row in rdr:
            if not row:
                continue
            src = row[0]
            for c, val in zip(col_ports, row[1:]):
                if src == c:
                    continue
                if src not in port_ids or c not in port_ids:
                    continue
                try:
                    lam = max(0.0, float(val))
                except ValueError:
                    lam = 0.0
                if lam > 0:
                    table[(src, c)] = lam
    return table


def _is_3ds_area(area: str) -> bool:
    return area.startswith('3DS_F')


# 3DS 좌표 중심 (스케일 기준점)
_3DS_CX = 1115 + _3DS_W / 2   # 원본 x 중심
_3DS_CY = 12721 + _3DS_H / 2  # 원본 y 중심


def _transform_3ds(x: float, y: float, area: str):
    """3DS 노드 좌표에 스케일 + offset 적용."""
    dx, dy = _area_offset(area)
    # 중심 기준 스케일
    sx = _3DS_CX + (x - _3DS_CX) * _3DS_SCALE + dx
    sy = _3DS_CY + (y - _3DS_CY) * _3DS_SCALE + dy
    return sx, sy


def load_full_map(json_path: str):
    """Load all nodes & segments for background rendering, applying area offsets."""
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    nodes = {}
    for n in d['nodes']:
        area = n.get('area', '')
        if _is_3ds_area(area):
            nx, ny = _transform_3ds(n['x'], n['y'], area)
            nodes[n['id']] = {**n, 'x': nx, 'y': ny}
        else:
            dx, dy = _area_offset(area)
            nodes[n['id']] = {**n, 'x': n['x'] + dx, 'y': n['y'] + dy}
    segments = d['segments']
    ports = {p['nodeId']: p for p in d.get('ports', [])}
    chargers = {c['nodeId'] for c in d.get('chargers', [])}
    xs = [n['x'] for n in nodes.values()]
    ys = [n['y'] for n in nodes.values()]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return nodes, segments, ports, chargers, bbox


# ── Camera ────────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, bbox, w, h, pad=MAP_PAD):
        x0, y0, x1, y1 = bbox
        mw, mh = x1 - x0, y1 - y0
        sx = (w - 2*pad) / mw if mw > 0 else 1
        sy = (h - 2*pad) / mh if mh > 0 else 1
        self.scale  = min(sx, sy)
        self.offset = [
            pad + (w  - 2*pad - mw*self.scale)/2 - x0*self.scale,
            pad + (h  - 2*pad - mh*self.scale)/2 + (y0+mh)*self.scale,
        ]
        self._drag_start  = None
        self._offset_start = None

    def to_screen(self, mx, my):
        return (int(mx * self.scale + self.offset[0]),
                int(-my * self.scale + self.offset[1]))

    def px(self, mm):
        return mm * self.scale

    def on_down(self, pos):
        self._drag_start   = pos
        self._offset_start = list(self.offset)

    def on_move(self, pos):
        if self._drag_start is None:
            return
        self.offset[0] = self._offset_start[0] + pos[0] - self._drag_start[0]
        self.offset[1] = self._offset_start[1] + pos[1] - self._drag_start[1]

    def on_up(self):
        self._drag_start = None

    def on_scroll(self, pos, up: bool):
        f = 1.15 if up else 1/1.15
        self.offset[0] = pos[0] + (self.offset[0] - pos[0]) * f
        self.offset[1] = pos[1] + (self.offset[1] - pos[1]) * f
        self.scale *= f


# ── Button ────────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, rect, label, toggle=False, base=(60, 70, 85)):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.toggle = toggle
        self.base   = base
        self.active = False
        self.hover  = False

    def update_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)

    def clicked(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surf, font):
        col = tuple(min(c+40, 255) for c in self.base) if self.hover else self.base
        if self.toggle and self.active:
            col = (min(col[0]+30, 255), min(col[1]+30, 255), col[2])
        pygame.draw.rect(surf, col, self.rect, border_radius=4)
        pygame.draw.rect(surf, COL_DIM, self.rect, 1, border_radius=4)
        lbl = font.render(self.label, True, COL_TEXT)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_arrow(surf, color, p1, p2, width=1, head=5):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    L = math.hypot(dx, dy)
    if L < 2:
        return
    ratio = max(0, (L - head*1.5)) / L
    tip   = (p1[0]+dx*ratio, p1[1]+dy*ratio)
    pygame.draw.line(surf, color, p1, (int(tip[0]), int(tip[1])), width)
    ux, uy = dx/L, dy/L
    px, py = -uy, ux
    h = head
    pts = [
        (int(p2[0]-ux*h*1.5+px*h*0.6), int(p2[1]-uy*h*1.5+py*h*0.6)),
        (int(p2[0]),                    int(p2[1])),
        (int(p2[0]-ux*h*1.5-px*h*0.6), int(p2[1]-uy*h*1.5-py*h*0.6)),
    ]
    pygame.draw.polygon(surf, color, pts)


def draw_dashed_line(surf, color, p1, p2, width=1, dash_len=8, gap_len=5):
    """Draw a dashed line from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    drawn = 0.0
    while drawn < dist:
        seg_end = min(drawn + dash_len, dist)
        x1 = int(p1[0] + ux * drawn)
        y1 = int(p1[1] + uy * drawn)
        x2 = int(p1[0] + ux * seg_end)
        y2 = int(p1[1] + uy * seg_end)
        pygame.draw.line(surf, color, (x1, y1), (x2, y2), width)
        drawn += dash_len + gap_len


def draw_rotated_rect(surf, color, cx, cy, length, width, angle_deg,
                      border=None, border_w=1):
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    hw, hl = width/2, length/2
    corners = [
        ( hl*cos_a - hw*sin_a,  hl*sin_a + hw*cos_a),
        (-hl*cos_a - hw*sin_a, -hl*sin_a + hw*cos_a),
        (-hl*cos_a + hw*sin_a, -hl*sin_a - hw*cos_a),
        ( hl*cos_a + hw*sin_a,  hl*sin_a - hw*cos_a),
    ]
    pts = [(int(cx+dx), int(cy+dy)) for dx, dy in corners]
    pygame.draw.polygon(surf, color, pts)
    if border:
        pygame.draw.polygon(surf, border, pts, border_w)


# ── AMR_A sub-network is loaded from pre-generated pkl ──────────────────────
# Run gen_amr_pkl.py to generate KaistTB_AMR_A.pkl before first use.


# ── Combined Simulator ───────────────────────────────────────────────────────

class MCSOHTBridge:
    """graph_des_v6 의 job_mgr 인터페이스를 MCSEngine 으로 라우팅.

    test_plan_micro / test_graph_v6 의 dispatch.py:JobManager 와 동일한
    on_arrive / on_load_done / on_unload_done flow 를 vis_mcs_unified 의
    OHT 에 적용. 효과:
    - LOAD/UNLOAD dwell 동안 graph_des_v6 의 v.state = LOADING 유지 (= STOP)
    - eager dispatch 안 함 → dwell 보장 (= purple 표시 중에 출발 X)
    - dwell 끝나면 on_load_done → MCS phase 전환 + 다음 leg dispatch

    Mode: OHT 만. AGV / 3DS / Lift 는 기존 MCS flow 그대로.
    """

    def __init__(self, sim, mcs_engine, oht_des, dwell_time: float = 5.0):
        self.sim = sim
        self.mcs = mcs_engine
        self.des = oht_des
        self.load_dwell = dwell_time
        self.unload_dwell = dwell_time

    def on_create_event(self, t):
        """v6 의 EV_JOB_CREATE 핸들. MCS 가 자체 schedule 하므로 no-op."""
        pass

    def on_arrive(self, t, v):
        """v6 가 v.dest_reached 시 호출. MCS phase 에 따라 LOAD/UNLOAD dwell 시작.

        주의: v6._on_arrive 가 job_mgr.on_arrive 호출 후 early return 하므로
        _notify_followers 가 누락된다. 모든 return path 에서 명시적 호출 필수.
        (이게 빠지면 follower 가 영원히 BLOCKED — 사용자가 관찰한 버그.)
        """
        from graph_des_v6 import EV_LOAD_DONE, EV_UNLOAD_DONE, LOADING
        vid = v.id
        b = self.mcs.bindings.get(vid)
        # b.load 없음 = push 도착 또는 비-MCS dispatch.
        if b is None or b.load is None:
            # v6 normal path 와 동일: follower wake
            self.des._notify_followers(t, v)
            return
        # MCS phase 별 분기
        if b.phase == VehicleJobState.TO_PICKUP:
            # src 도착 → LOAD dwell
            b.phase = VehicleJobState.LOADING
            b.token += 1
            b.load.state = LoadState.ON_VEHICLE
            b.load.t_pickup_arr = t
            # port 대기열에서 제거
            port_key = f'OHT:{b.load.src_port}'
            port = self.mcs.ports.get(port_key)
            if port and b.load in port.waiting_loads:
                port.waiting_loads.remove(b.load)
            # v6 LOADING state — dwell 동안 STOP 유지 + 색상 purple
            v.state = LOADING
            v.job_state = 'LOADING'
            self.des._post(t + self.load_dwell, EV_LOAD_DONE, v)
            # Dwell 시작 직후 followers 깨움 — leader 가 STOP 으로 정지.
            self.des._notify_followers(t, v)
        elif b.phase == VehicleJobState.TO_DELIVERY:
            # dst 도착 → UNLOAD dwell
            b.phase = VehicleJobState.UNLOADING
            b.token += 1
            b.load.state = LoadState.DELIVERED
            b.load.t_deliver_arr = t
            v.state = LOADING
            v.job_state = 'UNLOADING'
            self.des._post(t + self.unload_dwell, EV_UNLOAD_DONE, v)
            self.des._notify_followers(t, v)
        else:
            # Other phases (e.g. UNLOADING already, IDLE) — just wake followers
            self.des._notify_followers(t, v)

    def on_load_done(self, t, v):
        """LOAD dwell 종료. phase=TO_DELIVERY + dst 향한 dispatch."""
        from graph_des_v6 import STOP
        vid = v.id
        b = self.mcs.bindings.get(vid)
        if b is None or b.load is None:
            return
        if b.phase != VehicleJobState.LOADING:
            return
        b.phase = VehicleJobState.TO_DELIVERY
        b.token += 1
        b.load.t_pickup_end = t
        v.job_state = 'TO_DROP'
        v.state = STOP
        v.stop_reason = 'dest'
        # 이제 deliver leg dispatch
        dispatch = self.mcs._dispatch_cb.get('OHT')
        if dispatch:
            dispatch(vid, b.load.dst_port, t)

    def on_unload_done(self, t, v):
        """UNLOAD dwell 종료. MCS 에 완료 통보 + 차량 idle.

        MCS._on_dwell_done UNLOADING 분기를 OHT 용으로 inline replication.
        Recipe stage 전환은 OHT 의 경우 거의 없으나 호환 위해 포함.
        """
        from graph_des_v6 import STOP
        vid = v.id
        b = self.mcs.bindings.get(vid)
        if b is None:
            return
        if b.phase != VehicleJobState.UNLOADING:
            return
        load = b.load
        if load is None:
            b.phase = VehicleJobState.IDLE
            return
        load.t_deliver_end = t
        # KPI: vehicle idle 화
        try:
            self.mcs.kpi.mark_idle(vid, t)
        except Exception:
            pass
        # Recipe stage 진행 (있으면)
        recipe = (self.mcs.recipes.get(load.recipe_id)
                  if load.recipe_id else None)
        has_next_stage = (recipe is not None
                          and load.stage_idx + 1 < len(recipe.stages))
        if has_next_stage:
            # 다음 stage 의 src port 로 load 이동
            b.load = None
            b.phase = VehicleJobState.IDLE
            b.token += 1
            load.stage_idx += 1
            next_st = recipe.stages[load.stage_idx]
            load.system   = next_st.system
            load.src_port = next_st.src
            load.dst_port = next_st.dst
            load.state    = LoadState.WAITING
            load.vehicle_id = None
            load.t_assigned   = 0.0
            load.t_pickup_arr = 0.0
            load.t_pickup_end = 0.0
            load.t_deliver_arr = 0.0
            next_pk = f'{next_st.system}:{next_st.src}'
            next_port = self.mcs.ports.get(next_pk)
            if next_port is not None:
                next_port.waiting_loads.append(load)
        else:
            # Recipe 종료 또는 단일-시스템 load 완료
            load.state = LoadState.COMPLETED
            load.t_completed = t
            try:
                self.mcs.kpi.record_complete(load)
            except Exception:
                pass
            b.load = None
            b.phase = VehicleJobState.IDLE
            b.token += 1
            # WIP refill
            if recipe is not None and recipe.stages:
                st0 = recipe.stages[0]
                src_pk = f'{st0.system}:{st0.src}'
                src_port = self.mcs.ports.get(src_pk)
                if src_port is not None:
                    src_port.wip_count += 1
        v.job_state = 'IDLE'
        v.state = STOP
        v.stop_reason = 'dest'
        # 다음 task 배정 시도
        try:
            self.mcs._do_assign(t)
        except Exception:
            pass
        # follower wake (= 이 OHT 가 dest 에 정차 → 다른 OHT 가 지나갈 수 있음)
        try:
            self.des._notify_followers(t, v)
        except Exception:
            pass


class DeadlockDetected(RuntimeError):
    """모든 AGV 가 일정 시간 이상 MOVING/ROTATING 아닐 때 자동 raise."""
    pass


class CombinedSimulator:
    def __init__(self, oht_map: OHTMap, amr_graph: PklMapGraph,
                 n_oht: int = 5, n_agv: int = 3, n_s3d: int = 2,
                 conwip_agv: int = 0, use_sidings: bool = True,
                 max_sim_time: float = 0.0,
                 recipe_file: str = None,
                 recipe_conwip: int = 0,
                 recipe_rate: float = 0.0,
                 conwip_oht: int = 0,
                 lenient: bool = False,
                 sidings_override: 'list[str] | None' = None,
                 fromto_csv: str = None,
                 fromto_scale: float = 1.0,
                 planner_type: str = 'sipp',
                 coarse_debug: bool = False,
                 dwell_time: float = 3.0,
                 warmup_time: float = 0.0,
                 headless: bool = False,
                 profile_frames_ms: float = 0.0):
        self._lenient = lenient
        self._sidings_override = sidings_override
        self._fromto_csv = fromto_csv
        self._fromto_scale = fromto_scale
        self._planner_type = planner_type
        self._coarse_debug = coarse_debug
        self._dwell_time = dwell_time
        self._warmup_time = warmup_time
        self._warmup_done = (warmup_time <= 0.0)   # 0 이면 즉시 측정
        self._headless = headless
        self._profile_frames_ms = profile_frames_ms
        # Per-frame SIPP counters
        self._sipp_call_count = 0
        self._sipp_total_s = 0.0
        self._planner_plan_count = 0
        self._planner_plan_s = 0.0
        self._dispatch_count = 0
        # Per-frame _replan_done_agvs sub-section counters
        self._sub_prune_s = 0.0
        self._sub_recompute_s = 0.0
        self._sub_extend_s = 0.0
        self._sub_constraint_s = 0.0
        self._sub_truncate_s = 0.0
        self._sub_pickpark_s = 0.0
        self._sub_afterplan_s = 0.0
        self._sub_dispatch_s = 0.0
        self._replan_call_count = 0
        self._mcs_evt_count = 0
        self._mcs_evt_by_kind = {}   # kind -> (count, total_s)
        # after-plan sub-breakdown
        self._sub_leg2_s = 0.0
        self._sub_replan_hist_s = 0.0
        self._sub_batch_s = 0.0
        self._sub_findcycle_s = 0.0
        self._sub_snapshot_s = 0.0
        self.oht_map   = oht_map
        self.amr_graph = amr_graph
        self._n_oht  = min(n_oht, MAX_AGENTS)
        self._n_agv  = min(n_agv, MAX_AGENTS)
        self._n_s3d  = n_s3d
        self._conwip_agv = conwip_agv  # 0이면 OFF, >0이면 AGV 시스템 WIP target
        self._use_sidings = use_sidings  # Tier-A siding 사용 여부 (KPI 비교용)
        self._max_sim_time = max_sim_time  # >0 이면 자동 종료

        # Recipe 설정
        self._recipe_file: str = recipe_file
        self._recipe_conwip: int = recipe_conwip
        self._recipe_rate: float = recipe_rate
        self._conwip_oht: int = conwip_oht  # OHT 단독 CONWIP (검증용)

        # Load full map for background
        self.bg_nodes, self.bg_segments, self.bg_ports, self.bg_chargers, self.bg_bbox = \
            load_full_map(JSON_FILE)
        self._node_area = {n['id']: n.get('area', '') for n in self.bg_nodes.values()}

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption('OHT + AGV Combined Simulator -KaistTB')
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont('Consolas', 12)
        self.font_m = pygame.font.SysFont('Consolas', 14)
        self.font_b = pygame.font.SysFont('Consolas', 15, bold=True)

        self.sim_time = 0.0
        # --max-time 모드에선 자동 시작 (headless 자동 검증용) + 최고 속도로
        self.running  = (max_sim_time > 0)
        self.spd_idx  = (len(SIM_SPEEDS) - 1) if max_sim_time > 0 else DEFAULT_SPD_IDX

        self.cam = Camera(self.bg_bbox, MAP_W, WIN_H)
        self._build_buttons()
        # Sidebar scroll offset (= MOUSEWHEEL 로 조절. 사이드바 정보가 wh 넘을 때)
        self._sidebar_scroll: int = 0
        # OHT viz toggles (test_graph_v6 호환). L / T / C 키.
        self._show_oht_leaders: bool = False
        self._show_oht_dests:   bool = False
        self._show_oht_commit:  bool = False   # commit horizon (x_marker)
        # OHT verbose log (per-tick state/dispatch/push). 기본 OFF (= 너무 많음).
        # [OHT-SAFETY] (충돌 경고) 는 항상 출력.
        self._oht_verbose: bool = False

        # 디버그 표시 토글
        self._show_node_ids: bool = False
        self._show_sidings: bool = True   # X 키로 토글
        self._show_dep_arrows: bool = False   # A 키로 토글 (coarse 모드 dep)
        # CYCLE-PUSH 시 victim 의 escape route 표시 {aid: push_dest_node}
        self._cycle_push_dest: dict = {}

        # ── siding 후보 ──
        if self._sidings_override is not None:
            # 외부 JSON (--sidings) 으로 명시 — 송도 같이 layout-specific 한 경우.
            # Graph 에 실제 존재하는 노드만 valid.
            valid = [n for n in self._sidings_override if n in amr_graph.nodes]
            self._siding_tier_a = valid
            self._siding_tier_b = []
            print(f'[SIDING-OVERRIDE] {len(valid)}/{len(self._sidings_override)} '
                  f'valid sidings (graph 에 존재)')
        else:
            # KaistTB Tier-A 하드코딩 (probe_siding_candidates.py 결과)
            # Tier A: spur/pendant — inf-park 해도 graph 연결성 유지.
            self._siding_tier_a = [
                'na.198', 'na.197',                              # 우상단
                'na.59', 'na.60',                                # 좌상단
                'na.40', 'na.41', 'na.42',                       # 우하단
                'na.196',                                         # 좌하단
            ]
            self._siding_tier_b = [
                'na.7', 'na.8', 'na.37', 'na.39',
            ]
            # graph 에 없는 노드 제거 (layout 변경 시 invalid 자동 정리)
            self._siding_tier_a = [n for n in self._siding_tier_a if n in amr_graph.nodes]
            self._siding_tier_b = [n for n in self._siding_tier_b if n in amr_graph.nodes]
        self._siding_nodes = self._siding_tier_a + self._siding_tier_b

        # Branching nodes (out-deg ≥ 2) — viz 에 다른 색으로 표시.
        # corridor (out=1) = 회색, branching = 주황/노랑 등
        self._branching_nodes = set()
        for nid in amr_graph.nodes:
            if len(amr_graph.adj.get(nid, [])) >= 2:
                self._branching_nodes.add(nid)

        # Cut nodes — port 의 unique entry/exit chain.
        # 1) 기존 bidirectional walk: visited 아닌 이웃이 단일 인 동안.
        # 2) Chain 끝 직후의 *unique forward exit* 한 step 추가 (= bottleneck).
        #    예: Port5_g0_1 다음의 Port5_g0_0 — out-deg=2 지만 한쪽은
        #    chain 으로 돌아감. 막히면 port 영구 차단.
        self._cut_nodes = set()
        self._cut_to_port: dict = {}
        port_set = set(amr_graph.ports.values()) if amr_graph.ports else set()
        for p in port_set:
            visited = {p}
            current = p
            while True:
                succs = set(amr_graph.adj.get(current, []))
                preds = {u for u, ns in amr_graph.adj.items() if current in ns}
                neighbors = succs | preds
                next_set = neighbors - visited
                if len(next_set) != 1:
                    break
                nxt = next(iter(next_set))
                self._cut_nodes.add(nxt)
                self._cut_to_port[nxt] = p
                visited.add(nxt)
                current = nxt
            # Chain 끝 직후 한 step 추가: visited 의 forward succs 중 visited
            # 안 가는 게 정확히 1개면 추가. 그 노드도 막히면 port 차단됨.
            fwd_exits = [s for s in amr_graph.adj.get(current, [])
                          if s not in visited]
            if len(fwd_exits) == 1:
                bottleneck = fwd_exits[0]
                self._cut_nodes.add(bottleneck)
                self._cut_to_port[bottleneck] = p

        # Park pool: push 시 후보 위치 = ports ∪ Tier-A sidings.
        # Tier-A 만 포함 (어떤 AGV 조합이든 cut 발생 안 함).
        # 우선순위 없이 거리 기준으로 가장 가까운 free park 선택.
        # 초기화는 _init_mcs 이후 (agv_planner._port_nodes 가 준비된 시점) 수행.
        self._park_nodes: list = []

        # OHT push pool: port_nodes 만 사용 (10001 같은 U-turn dead-end 제외).
        # directed loop 에서 free push 후보가 부족할 수 있으나 운영 정책상
        # spur 진입은 금지.
        self._oht_park_nodes: set = set(oht_map.port_nodes)
        print(f'[OHT-PARK] {len(self._oht_park_nodes)} ports (push 후보)')

        # OHT agents (segment-queue DES). Layout 에 OHT 영역 없으면 stub.
        self._oht_done_notified: set = set()
        self._agv_done_notified: set = set()
        self._s3d_done_notified: set = set()
        self._oht_next_id = 0
        self.oht_agents: list[OHTAgent] = []
        if hasattr(oht_map, 'gmap'):
            self.oht_env = OHTEnvironmentDES(oht_map, cross_segment=True)
        else:
            # _EmptyOHTMap — n_oht=0 시나리오. 모든 OHT 호출을 no-op 처리.
            class _StubOHTEnv:
                agents: list = []
                _zcu_holders: dict = {}
                _zcu_waitlists: dict = {}
                des = type('_StubDes', (),
                           {'vehicles': {}, '_seg_occupants': {}})()
                def step(self, t): pass
                def reassign(self, *a, **k): pass
                def add_agent(self, *a, **k): pass
                def remove_agent(self, *a, **k): pass
            self.oht_env = _StubOHTEnv()
        self._init_oht_agents()

        # AGV agents (prioritized SIPP + TAPG execution)
        self._agv_next_id = 100   # offset to avoid ID collisions
        self.agv_agents: list[TAPGAgent] = []
        # L state (LOADING/UNLOADING dwell) 빌드 — port 별 1개씩.
        # SIPP plan 끝에 명시적 LOADING state 가 들어가도록.
        amr_graph.build_load_states(dwell_time=self._dwell_time)
        self.agv_env = TAPGEnvironment(amr_graph, accel=500.0, decel=500.0)
        # Coarse mode: TAPG 의 time-based cross-edge 비활성화 + live
        # occupancy + cut node admission rule 활성화
        self._configure_coarse_mode(amr_graph)
        # Tier-A sidings 를 planner 의 push 후보 풀로 주입.
        # _find_empty_ports / _find_alternate_goal 가 ports 외에 sidings 도
        # blocker push / alternate goal 후보로 사용한다.
        # (use_sidings=False 시 simulator-level park pool 도 sidings 미사용 →
        #  planner 에도 동일 정책 적용해 일관성 유지.)
        _push_extras = (self._siding_tier_a if self._use_sidings else None)
        if self._planner_type == 'coarse':
            from coarse_planner import CoarsePlanner
            from segment_lock import SegmentLockManager
            self.agv_planner = CoarsePlanner(amr_graph, push_extras=_push_extras)
            self._segment_lock = SegmentLockManager(
                self.agv_planner.segments,
                self.agv_planner.node_to_segment)
            print(f'[PLANNER] coarse: {len(self.agv_planner.checkpoints)} '
                  f'checkpoints, {len(self.agv_planner.segments)} segments')
            # Push 후보 빌드: *충분히 긴 corridor* (interm len >= 3) 의
            # last-grey 만. 짧은 segment 의 last-grey 는 port/cut node 직전
            # 이라 push location 으로 부적합 (= port 진입로 차단).
            push_cands = set()
            for seg_id, interm in self.agv_planner.segments.items():
                if len(interm) >= 3:
                    push_cands.add(interm[-1])
            self._coarse_push_candidates = list(push_cands)
            print(f'[COARSE-PUSH] {len(self._coarse_push_candidates)} candidate '
                  f'nodes (last-grey of segments with interm>=3)')
        else:
            self.agv_planner = PklPrioritizedPlanner(
                amr_graph, push_extras=_push_extras)
            self._segment_lock = None
        # --lenient: SIPP fail 시 halt 안 함 (skip + sim 계속).
        # 기본 (--lenient 없음): halt + dump (디버그 친화).
        self.agv_planner.halt_on_sipp_fail = not self._lenient

        # --profile-frames 일 때만 planner.plan / SIPP wrap (call/time 측정)
        if self._profile_frames_ms > 0:
            import time as _t
            _self = self
            _base = self.agv_planner._base if hasattr(self.agv_planner, '_base') else self.agv_planner
            _orig_plan = self.agv_planner.plan
            _orig_sipp = _base._sipp_search

            def _wrap_plan(*a, **kw):
                t0 = _t.perf_counter()
                try:
                    return _orig_plan(*a, **kw)
                finally:
                    _self._planner_plan_count += 1
                    _self._planner_plan_s += _t.perf_counter() - t0
            def _wrap_sipp(*a, **kw):
                t0 = _t.perf_counter()
                try:
                    return _orig_sipp(*a, **kw)
                finally:
                    _self._sipp_call_count += 1
                    _self._sipp_total_s += _t.perf_counter() - t0
            self.agv_planner.plan = _wrap_plan
            _base._sipp_search = _wrap_sipp
        self._agv_goals: dict = {}       # aid → current goal node
        self._agv_pushed: set = set()   # push(임시 목적지)된 AGV 추적
        self._oht_pushed: set = set()   # push(임시 목적지)된 OHT 추적
        self._agv_pending_replan: set = set()   # agents awaiting replan
        # init 시 multi-dispatch 를 단일 replan 으로 묶기 위한 플래그.
        # True 인 동안 _mcs_dispatch_agv 는 _replan_done_agvs 를 호출하지 않음.
        self._dispatch_defer_replan: bool = False
        # AGV 이동 이력 — retrieve / deliver / push 별 (aid, t, src, dst, load_id)
        # _save_movement_log() 로 CSV 저장.
        self._agv_movement_log: list = []
        # 실제 dwell 측정용 — phase 전이 추적
        self._agv_phase_prev: dict = {}    # aid → 직전 phase
        self._agv_dwell_log: list = []     # {aid, kind, t_start, t_end, duration, port}
        self._agv_dwell_open: dict = {}    # aid → (kind, t_start, port) 현재 진행 중
        # SIPP plan 시간 추적 — 슬로우 케이스 분석용
        self._plan_dur_log: list = []      # {t, dur, n_agents, cs, pending, goals, status}
        # Deadlock 자동 감지: 모든 AGV 가 MOVING/ROTATING 아니고 sim_time 이
        # threshold 이상 정체되면 halt + dump.
        self._deadlock_threshold: float = 300.0  # seconds without any movement
        self._last_active_sim_time: float = 0.0
        self._agv_arrival_idx: dict = {}  # aid → path_idx at which physical arrival occurs (before dwell)
        self._agv_arrived_notified: set = set()  # 도착 통보 완료된 AGV (DONE 전 arrival)
        self._plan_status = ''
        self._collision_log_f = open(COLLISION_LOG, 'w', encoding='utf-8')
        self._collision_pairs_logged: set = set()
        self._collision_count = 0
        self._plan_log_counter = 0
        # 로그 초기화
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        plan_log = os.path.join(log_dir, 'plan_log.txt')
        with open(plan_log, 'w', encoding='utf-8') as f:
            f.write(f'Plan Log started\n')
        self._init_agv_agents()

        # 3DS shuttles (one per floor, simple path-following)
        self._init_3ds()

        # ── MCS (Material Control System) ────────────────────────────────
        self._init_mcs()

        # OHT bridge: graph_des_v6 job_mgr ↔ MCSEngine. v6 의 LOAD/UNLOAD
        # 이벤트가 MCS phase 전환 + 다음 leg dispatch 로 연결됨. 효과:
        # dwell 동안 v.state=LOADING 유지 (= STOP 보장).
        try:
            self._oht_bridge = MCSOHTBridge(
                self, self.mcs, self.oht_env.des,
                dwell_time=self._dwell_time)
            self.oht_env.des.job_mgr = self._oht_bridge
        except Exception as e:
            print(f'[OHT-BRIDGE] init skipped: {e}')

        # ── pygame_gui: AGV jobs panel + KPI panel ───────────────────────
        self._init_agv_jobs_panel()
        self._init_kpi_panel()
        self._kpi_panel_last_t: float = -100.0

    # ── AGV Jobs UI Panel (pygame_gui) ────────────────────────────────────

    def _init_agv_jobs_panel(self):
        """floating window 로 AGV 작업 보기/추가/취소.
        J 키로 토글. 'Source'/'Destination' dropdown 은 AGV 포트만 노출.

        pygame_gui 의 UIManager / 위젯 생성이 python 글로벌 random 상태를
        소비하므로, 이후 시뮬레이션의 결정성 보존을 위해 init 전·후에
        random state 를 snapshot 한 뒤 복원한다.
        """
        _rng_snapshot = random.getstate()

        ww, wh = self.screen.get_size()
        self.ui_manager = pygame_gui.UIManager((ww, wh))

        # AGV 포트 목록 (dropdown 옵션)
        agv_ports = sorted(self.agv_planner._port_nodes)
        self._agv_port_options = agv_ports if agv_ports else ['(none)']

        self._agv_panel_visible = True
        panel_w, panel_h = 340, 520
        # 화면 좌상단에서 약간 안쪽
        self._agv_panel_rect = pygame.Rect(12, 60, panel_w, panel_h)

        self._build_agv_jobs_window()

        # selection list 항목 → load_id 매핑 (Cancel 처리용)
        self._agv_list_item_to_load: dict = {}
        # 마지막 갱신 시 load 스냅샷 (변경 감지)
        self._agv_jobs_signature = None

        # AGV Detail 패널 — 선택한 AGV 의 remaining path / claimable / load 정보
        self._selected_agv_id: int | None = None
        self._agv_detail_visible: bool = False
        self._build_agv_detail_window()

        # pygame_gui 가 소비한 RNG 상태 복원 — 패널 유무가 시뮬레이션
        # 결정성에 영향 미치지 않도록.
        random.setstate(_rng_snapshot)

    def _init_kpi_panel(self):
        """Live KPI side panel. 1s 주기 갱신. Resize 가능."""
        if getattr(self, '_headless', False):
            return   # headless 면 panel 안 만듦
        ww, wh = self.screen.get_size()
        rect = pygame.Rect(ww - 380, 620, 360, 280)
        self._kpi_window = UIWindow(
            rect=rect, manager=self.ui_manager,
            window_display_title='Live KPI',
            object_id='#kpi_window', resizable=True,
        )
        self._kpi_textbox = UITextBox(
            html_text='KPI loading...',
            relative_rect=pygame.Rect(4, 4, -8, -8),
            manager=self.ui_manager, container=self._kpi_window,
            anchors={'left': 'left', 'right': 'right',
                     'top': 'top', 'bottom': 'bottom'},
        )

    def _refresh_kpi_panel(self):
        """매 frame 호출되지만 1s 주기로만 update (= 비용 cap)."""
        if not hasattr(self, '_kpi_textbox'):
            return
        if abs(self.sim_time - self._kpi_panel_last_t) < 1.0:
            return
        self._kpi_panel_last_t = self.sim_time
        T = max(self.sim_time, 1e-9)
        warmup_t = getattr(self, '_kpi_start_t', 0.0)
        T_eff = max(T - warmup_t, 1e-9)
        n_ret = sum(1 for ev in self._agv_movement_log if ev['type'] == 'retrieve')
        n_del = sum(1 for ev in self._agv_movement_log if ev['type'] == 'deliver')
        n_push = sum(1 for ev in self._agv_movement_log if ev['type'] == 'push')
        completed_all = list(getattr(self.mcs.kpi, 'completed_loads', []))
        completed = [L for L in completed_all
                     if L.t_completed >= warmup_t and L.t_assigned > 0]
        # Total created = completed + active (bindings) + waiting (ports)
        all_created = list(completed_all)
        for vid, bd in self.mcs.bindings.items():
            if bd.load and bd.load not in all_created:
                all_created.append(bd.load)
        for port in self.mcs.ports.values():
            for L in port.waiting_loads:
                if L not in all_created:
                    all_created.append(L)
        all_warmup = [L for L in all_created if L.t_created >= warmup_t]
        if completed:
            avg_ret = sum(L.t_pickup_end - L.t_assigned for L in completed
                          if L.t_pickup_end >= L.t_assigned) / max(len(completed), 1)
            avg_del = sum(L.t_completed - L.t_pickup_end for L in completed
                          if L.t_completed >= L.t_pickup_end) / max(len(completed), 1)
            avg_cycle = sum(L.t_completed - L.t_created
                            for L in completed) / max(len(completed), 1)
        else:
            avg_ret = avg_del = avg_cycle = 0.0
        arrival_rate = len(all_warmup) / T_eff * 60.0
        throughput_rate = len(completed) / T_eff * 60.0
        warmup_status = 'measuring' if self._warmup_done else f'warmup ({warmup_t - T:.0f}s left)'
        # Cycle / push stats
        n_cycle_now = len(getattr(self, '_cycle_push_dest', {}))
        n_pushed_now = len(getattr(self, '_agv_pushed', set()))
        html = (
            f'<font color=#80c0ff>sim t = {T:.1f}s</font> '
            f'<font color=#a0a0a0>({warmup_status})</font><br>'
            f'<b>Throughput</b><br>'
            f'&nbsp;arrival: {arrival_rate:.2f} /min<br>'
            f'&nbsp;effective: {throughput_rate:.2f} /min '
            f'({len(completed)} done)<br><br>'
            f'<b>Movement counts</b><br>'
            f'&nbsp;retrieve / deliver / push = {n_ret} / {n_del} / {n_push}<br><br>'
            f'<b>Avg times (s)</b><br>'
            f'&nbsp;retrieve = {avg_ret:.1f}<br>'
            f'&nbsp;deliver  = {avg_del:.1f}<br>'
            f'&nbsp;cycle    = {avg_cycle:.1f}<br><br>'
            f'<b>Active</b><br>'
            f'&nbsp;pushed AGVs = {n_pushed_now}<br>'
            f'&nbsp;cycle-pushed = {n_cycle_now}'
        )
        try:
            self._kpi_textbox.set_text(html)
        except Exception:
            pass

    def _build_agv_detail_window(self):
        """AGV Detail UIWindow + UITextBox. 시작 시엔 hidden.
        Window resize 시 textbox 도 함께 늘어남 (anchors all sides)."""
        ww, wh = self.screen.get_size()
        rect = pygame.Rect(ww - 380, 60, 360, 540)
        self._agv_detail_window = UIWindow(
            rect=rect, manager=self.ui_manager,
            window_display_title='AGV Detail (K to toggle, [ ] to cycle)',
            object_id='#agv_detail_window', resizable=True,
        )
        # anchors: 4 방향 모두 anchor -> window 와 함께 resize
        self._agv_detail_textbox = UITextBox(
            html_text='Click an AGV on the map or press ] to select.',
            relative_rect=pygame.Rect(4, 4, -8, -8),
            manager=self.ui_manager, container=self._agv_detail_window,
            anchors={'left': 'left', 'right': 'right',
                     'top': 'top', 'bottom': 'bottom'},
        )
        self._agv_detail_window.hide()

    def _toggle_agv_detail_panel(self):
        self._agv_detail_visible = not self._agv_detail_visible
        if self._agv_detail_visible:
            self._agv_detail_window.show()
        else:
            self._agv_detail_window.hide()

    def _select_agv(self, aid: int | None):
        self._selected_agv_id = aid
        if aid is not None and not self._agv_detail_visible:
            self._toggle_agv_detail_panel()

    def _cycle_agv_selection(self, direction: int):
        """direction = +1 (next) or -1 (prev)."""
        if not self.agv_agents:
            return
        ids = sorted(a.id for a in self.agv_agents)
        if self._selected_agv_id is None or self._selected_agv_id not in ids:
            self._select_agv(ids[0] if direction > 0 else ids[-1])
            return
        i = ids.index(self._selected_agv_id)
        self._select_agv(ids[(i + direction) % len(ids)])

    def _pick_agv_at_screen(self, sx: int, sy: int, radius: int = 20) -> int | None:
        """화면 좌표 근처의 AGV id 반환 (없으면 None)."""
        best_aid, best_d = None, radius * radius
        for a in self.agv_agents:
            ax, ay = self.cam.to_screen(a.x, a.y)
            d = (ax - sx) ** 2 + (ay - sy) ** 2
            if d <= best_d:
                best_d = d
                best_aid = a.id
        return best_aid

    def _refresh_agv_detail_panel(self):
        """선택된 AGV 의 remaining path + load + claimable 정보를 textbox 에 렌더."""
        if not self._agv_detail_visible or self._selected_agv_id is None:
            return
        a = next((x for x in self.agv_agents if x.id == self._selected_agv_id), None)
        if a is None:
            self._agv_detail_textbox.set_text(
                f'A{self._selected_agv_id - 100} not found.')
            return
        # MCS load info
        b = self.mcs.bindings.get(a.id)
        load_info = '<b>No load (IDLE)</b>'
        if b and b.load:
            ld = b.load
            load_info = (
                f"<b>load=L{ld.load_id}</b> "
                f"{ld.src_port}&rarr;{ld.dst_port}<br>"
                f"phase={b.phase.value}  state={ld.state.value}")
        # Path summary
        n_total = len(a.raw_path)
        n_remain = n_total - a.path_idx
        claim_idx = getattr(a, 'claim_idx', a.path_idx)
        # Up to 25 remaining states (panel 크기 한계)
        end_idx = min(n_total, a.path_idx + 25)
        rows = []
        G = self.agv_env.G
        for i in range(a.path_idx, end_idx):
            sid, t = a.raw_path[i]
            nk = self.agv_env._nk(sid, a.id, t)
            in_G = nk in G
            cross_preds = []
            if in_G:
                for p in G.predecessors(nk):
                    if p[1] != a.id:
                        cross_preds.append(p[1])
            claim_mark = ('Y' if not in_G or not cross_preds else 'N')
            cp_str = ('-' if not cross_preds
                      else ','.join('A' + str(p - 100) for p in cross_preds[:3])
                           + ('+' if len(cross_preds) > 3 else ''))
            # state 종류별 색상. 단 *claimed range* [path_idx, claim_idx) 는 빨강.
            sid_short = sid[:18]
            is_claimed = (i < claim_idx)
            if is_claimed:
                color = '#ff4444'   # 빨강 — 현재 claim 영역
                tag = '★'
            else:
                color = ('#9cdcfe' if sid.startswith('M,')
                         else '#ce9178' if sid.startswith('R,')
                         else '#b5cea8')
                tag = ' '
            rows.append(
                f'<font color="{color}">{tag}{sid_short:<18}</font> '
                f't={t:>7.1f} {claim_mark} {cp_str}')
        more = '' if end_idx == n_total else f'<br><i>... {n_total - end_idx} more</i>'

        html = (
            f'<b>AGV A{a.id - 100}</b>  '
            f'state={a.state} idx={a.path_idx}/{n_total} '
            f'claim={claim_idx} ({n_remain} remain)<br>'
            f'<br>'
            f'{load_info}<br>'
            f'<br>'
            f'<b>Path (next {end_idx - a.path_idx}):</b> '
            f'<font color="#ff4444">★=claimed</font><br>'
            f'<font face="Consolas">'
            + '<br>'.join(rows)
            + '</font>'
            + more
        )
        self._agv_detail_textbox.set_text(html)

    def _build_agv_jobs_window(self):
        """UIWindow + 내부 위젯 구성. reset/toggle 에서도 재호출 안전."""
        self._agv_jobs_window = UIWindow(
            rect=self._agv_panel_rect,
            manager=self.ui_manager,
            window_display_title='AGV Jobs',
            object_id='#agv_jobs_window',
            resizable=True,
        )

        # 컨테이너 내부 좌표는 (0,0) 기준
        c = self._agv_jobs_window
        row = 0
        UILabel(relative_rect=pygame.Rect(8, row, 320, 22),
                text='Waiting / Assigned Loads',
                manager=self.ui_manager, container=c)
        row += 26
        # Loads list: window 가 커지면 함께 늘어남 (left/right/top anchor + bottom 도 anchor)
        self._agv_loads_list = UISelectionList(
            relative_rect=pygame.Rect(8, row, -16, 220),
            item_list=[],
            manager=self.ui_manager, container=c,
            allow_multi_select=False,
            anchors={'left': 'left', 'right': 'right',
                     'top': 'top'})
        row += 226

        UILabel(relative_rect=pygame.Rect(8, row, 80, 22),
                text='Source:', manager=self.ui_manager, container=c)
        self._agv_src_dd = UIDropDownMenu(
            options_list=self._agv_port_options,
            starting_option=self._agv_port_options[0],
            relative_rect=pygame.Rect(90, row, 220, 26),
            manager=self.ui_manager, container=c)
        row += 32

        UILabel(relative_rect=pygame.Rect(8, row, 80, 22),
                text='Dest:', manager=self.ui_manager, container=c)
        dst_default = (self._agv_port_options[1]
                       if len(self._agv_port_options) > 1
                       else self._agv_port_options[0])
        self._agv_dst_dd = UIDropDownMenu(
            options_list=self._agv_port_options,
            starting_option=dst_default,
            relative_rect=pygame.Rect(90, row, 220, 26),
            manager=self.ui_manager, container=c)
        row += 32

        self._agv_add_btn = UIButton(
            relative_rect=pygame.Rect(8, row, 145, 30),
            text='Add Load', manager=self.ui_manager, container=c)
        self._agv_cancel_btn = UIButton(
            relative_rect=pygame.Rect(166, row, 145, 30),
            text='Cancel Selected', manager=self.ui_manager, container=c)
        row += 36

        self._agv_status_label = UILabel(
            relative_rect=pygame.Rect(8, row, 304, 22),
            text='', manager=self.ui_manager, container=c)

    def _toggle_agv_jobs_panel(self):
        self._agv_panel_visible = not self._agv_panel_visible
        if self._agv_panel_visible:
            self._agv_jobs_window.show()
        else:
            self._agv_jobs_window.hide()

    def _refresh_agv_jobs_list(self):
        """waiting_loads + ASSIGNED 차량의 load 를 리스트에 반영.
        변경 없으면 skip (재정렬에 의한 selection 손실 방지)."""
        rows = []
        mapping = {}
        # WAITING
        for pk, port in self.mcs.ports.items():
            if port.system != 'AGV':
                continue
            for ld in port.waiting_loads:
                if ld.state == LoadState.WAITING:
                    txt = f'L{ld.load_id}  W  {ld.src_port}→{ld.dst_port}'
                    rows.append(txt)
                    mapping[txt] = ld.load_id
        # ASSIGNED / ON_VEHICLE / DELIVERED
        for vid, b in self.mcs.bindings.items():
            if b.system != 'AGV' or b.load is None:
                continue
            ld = b.load
            tag = {LoadState.ASSIGNED: 'A',
                   LoadState.ON_VEHICLE: 'V',
                   LoadState.DELIVERED: 'D'}.get(ld.state, '?')
            txt = (f'L{ld.load_id}  {tag} v{vid}  '
                   f'{ld.src_port}→{ld.dst_port}')
            rows.append(txt)
            mapping[txt] = ld.load_id

        sig = tuple(rows)
        if sig == self._agv_jobs_signature:
            return
        self._agv_jobs_signature = sig
        self._agv_list_item_to_load = mapping
        self._agv_loads_list.set_item_list(rows)

    def _panel_add_load(self):
        src = self._agv_src_dd.selected_option
        dst = self._agv_dst_dd.selected_option
        # pygame_gui 0.6.14: selected_option 은 tuple (text, object_id) 일 수 있음
        if isinstance(src, tuple): src = src[0]
        if isinstance(dst, tuple): dst = dst[0]
        if src == dst:
            self._agv_status_label.set_text('src=dst 불가')
            return
        port_key = f'AGV:{src}'
        port = self.mcs.ports.get(port_key)
        if port is None:
            self._agv_status_label.set_text(f'port 없음: {port_key}')
            return
        from mcs_unified import MCSEngine, Load
        MCSEngine._global_load_id += 1
        ld = Load(
            load_id=MCSEngine._global_load_id,
            src_port=src, dst_port=dst,
            system='AGV', t_created=self.sim_time,
        )
        port.waiting_loads.append(ld)
        self._agv_status_label.set_text(f'L{ld.load_id} 추가됨 ({src}→{dst})')
        # 즉시 할당 시도
        self.mcs._do_assign(self.sim_time)

    def _panel_cancel_selected(self):
        sel = self._agv_loads_list.get_single_selection()
        if not sel:
            self._agv_status_label.set_text('선택된 load 없음')
            return
        load_id = self._agv_list_item_to_load.get(sel)
        if load_id is None:
            return
        # WAITING 상태만 단순 제거 (ASSIGNED+ 는 차량 idle 복귀 필요)
        for port in self.mcs.ports.values():
            for ld in list(port.waiting_loads):
                if ld.load_id == load_id and ld.state == LoadState.WAITING:
                    port.waiting_loads.remove(ld)
                    self._agv_status_label.set_text(f'L{load_id} 취소됨')
                    return
        self._agv_status_label.set_text(
            f'L{load_id} 이미 차량에 할당 — 취소 미지원')

    # ── 3DS setup (SIPP + TAPG, same as AGV) ──────────────────────────────

    def _init_3ds(self):
        """Load 3DS pkl per floor → SIPP planner + TAPG env (AGV와 동일 구조)."""
        self._s3d_next_id = 200
        self.s3d_floor_data = {}
        self.s3d_agents = []

        for fid in S3D_FLOOR_IDS:
            pkl_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'KaistTB_{fid}.pkl')
            if not os.path.exists(pkl_path):
                print(f'[WARN] {pkl_path} not found -run gen_3ds_pkl.py first')
                continue

            # PklMapGraph (정밀 collision profile)
            graph = PklMapGraph(pkl_path)

            # Area offset + scale 적용
            for node in graph.nodes.values():
                node.x, node.y = _transform_3ds(node.x, node.y, fid)

            planner  = PklPrioritizedPlanner(graph)
            tapg_env = TAPGEnvironment(graph, accel=500.0, decel=500.0)

            # 셔틀 배치 — 포트에 DONE 상태로 대기, MCS가 할당할 때 출발
            port_nodes_list = list(planner._port_nodes)
            random.shuffle(port_nodes_list)
            n_shuttles = min(self._n_s3d, len(port_nodes_list))

            agents = []
            base_col = S3D_COLORS.get(fid, (150, 150, 150))
            for k in range(n_shuttles):
                aid = self._s3d_next_id
                self._s3d_next_id += 1
                nid = port_nodes_list[k]

                stop_sid = self._find_stop_state(planner, nid)
                if stop_sid is None:
                    stop_sid = f'S,{nid},0'
                raw_path = [(stop_sid, 0.0)]
                brightness = 1.0 if k == 0 else 0.7
                color = tuple(max(0, min(255, int(c * brightness)))
                              for c in base_col)
                a = TAPGAgent(aid, color, raw_path)
                agents.append(a)
                self.s3d_agents.append(a)

            if agents:
                tapg_env.setup(agents, t_start=0.0)

            self.s3d_floor_data[fid] = {
                'graph': graph,
                'planner': planner,
                'env': tapg_env,
                'agents': agents,
                'goals': {},
                'pending_replan': set(),
            }

        # ── Elevators ──
        self._init_elevators()

    # ── MCS setup ────────────────────────────────────────────────────────────

    def _init_mcs(self):
        """MCS 엔진 초기화 — OHT/AGV 포트 등록, 차량 등록, 생산 시작."""
        import heapq as _hq

        # MCS 자체 힙 (나중에 단일 힙 통합 시 공유 힙으로 교체)
        self._mcs_heap = []
        self._mcs_seq = [0]

        self.mcs = MCSEngine(
            heap=self._mcs_heap,
            seq_counter=self._mcs_seq,
            dwell_time=self._dwell_time,
            seed=42,
        )

        # MCS event count: 항상-on (벤치마크용) + profile-frames 시 시간도.
        self._mcs_total_events = 0
        _sim = self
        _prof_on = self._profile_frames_ms > 0
        import time as _t_evt
        _orig_he = self.mcs.handle_event
        def _wrapped_he(ev):
            _sim._mcs_total_events += 1
            if not _prof_on:
                return _orig_he(ev)
            t0 = _t_evt.perf_counter()
            try:
                return _orig_he(ev)
            finally:
                dt = _t_evt.perf_counter() - t0
                _sim._mcs_evt_count += 1
                c, s = _sim._mcs_evt_by_kind.get(ev.kind, (0, 0.0))
                _sim._mcs_evt_by_kind[ev.kind] = (c + 1, s + dt)
        self.mcs.handle_event = _wrapped_he

        # ── OHT 시스템 등록 ──
        # port_prod_rate = port 당 분당 Poisson rate. 0.3 = AGV (0.1) 의 3x.
        oht_port_nodes = list(self.oht_map.port_nodes)
        if oht_port_nodes:
            self.mcs.register_system(
                system='OHT',
                port_nodes=oht_port_nodes,
                on_dispatch=self._mcs_dispatch_oht,
                get_vehicle_node=self._mcs_get_oht_node,
                get_distance=self._mcs_get_oht_distance,
                port_prod_rate=0.3,
            )

        # ── AGV 시스템 등록 ──
        agv_port_nodes = list(self.agv_planner._port_nodes)

        # park pool 구성: ports + (옵션) Tier-A sidings
        # --no-siding 이면 ports 만 사용 (KPI 비교 baseline)
        if self._use_sidings:
            valid_sidings = [n for n in self._siding_tier_a
                             if n in self.amr_graph.nodes]
            self._park_nodes = list(set(agv_port_nodes + valid_sidings))
            print(f'[PARK POOL] {len(agv_port_nodes)} ports + '
                  f'{len(valid_sidings)} sidings = {len(self._park_nodes)} total')
        else:
            self._park_nodes = list(set(agv_port_nodes))
            print(f'[PARK POOL] {len(agv_port_nodes)} ports only '
                  f'(--no-siding 모드)')
        if agv_port_nodes:
            self.mcs.register_system(
                system='AGV',
                port_nodes=agv_port_nodes,
                on_dispatch=self._mcs_dispatch_agv,
                is_vehicle_free=self._mcs_is_agv_free,
                get_vehicle_node=self._mcs_get_agv_node,
                get_distance=None,
                port_prod_rate=0.1,
            )

        # ── 3DS 시스템 등록 (층별) ──
        # port_prod_rate = port 당 분당 Poisson rate. AGV 와 동일 0.1.
        for fid, fd in self.s3d_floor_data.items():
            s3d_port_nodes = list(fd['planner']._port_nodes)
            if s3d_port_nodes:
                self.mcs.register_system(
                    system=fid,
                    port_nodes=s3d_port_nodes,
                    on_dispatch=lambda vid, goal, t, _fid=fid: self._mcs_dispatch_3ds(_fid, vid, goal, t),
                    is_vehicle_free=lambda vid, _fid=fid: self._mcs_is_3ds_free(_fid, vid),
                    get_vehicle_node=lambda vid, _fid=fid: self._mcs_get_3ds_node(_fid, vid),
                    get_distance=None,
                    port_prod_rate=0.1,
                )

        # ── LIFT 시스템 등록 (엘리베이터를 MCS vehicle 로) ──
        # gate node → (floor, lift, system_for_floor) 매핑
        # 한 elevator 가 모든 층의 gate 를 가지므로 (lift, floor) 페어로 식별.
        self._lift_floor_by_gate: Dict[str, str] = {}      # gate → floor
        self._lift_by_gate: Dict[str, 'Elevator'] = {}     # gate → Elevator
        self._lift_by_vid: Dict[int, 'Elevator'] = {}      # vid → Elevator
        # floor id → 3DS system 이름 (예: '1' → '3DS_F1')
        floor_to_system = {'1': '3DS_F1', '2': '3DS_F2', '3': '3DS_F3'}
        # 같은 gate 노드가 어느 3DS 시스템에 속하는지 (gate 위치는 한 층 안에 있음)
        gate_to_3ds_system: Dict[str, str] = {}

        all_gate_nodes = set()
        for lift in self.lift_ctrl._lifts.values():
            for fid_l, gate in lift.gate_nodes.items():
                self._lift_floor_by_gate[gate] = fid_l
                self._lift_by_gate[gate] = lift
                all_gate_nodes.add(gate)
                sys_for_floor = floor_to_system.get(fid_l)
                if sys_for_floor:
                    gate_to_3ds_system[gate] = sys_for_floor

        if all_gate_nodes:
            self.mcs.register_system(
                system='LIFT',
                port_nodes=list(all_gate_nodes),
                on_dispatch=self._mcs_dispatch_lift,
                is_vehicle_free=self._mcs_is_lift_free,
                get_vehicle_node=self._mcs_get_lift_node,
                get_distance=None,
                port_prod_rate=0.0,
            )
            # 각 elevator 를 vehicle 로 등록 (vid = -1000, -1001, ...)
            for lift in self.lift_ctrl._lifts.values():
                vid = lift._agent_id
                self._lift_by_vid[vid] = lift
                self.mcs.register_vehicle(vid, 'LIFT')

            # ── 3DS 시스템에도 gate 노드를 port 로 추가 등록 ──
            # (셔틀이 gate 로 deliver 하거나 pickup 하려면 MCS 에서 port 로 인식 필요)
            # 그래프에 노드 자체는 이미 존재 → SIPP plan 가능.
            from mcs_unified import Port as _Port
            for gate, sys_name in gate_to_3ds_system.items():
                pk = f'{sys_name}:{gate}'
                if pk in self.mcs.ports:
                    continue
                self.mcs.ports[pk] = _Port(node_id=gate, system=sys_name,
                                            prod_rate=0.0, dest_ports=[])
            print(f'[LIFT] {len(all_gate_nodes)} gate ports registered '
                  f'({len(self.lift_ctrl._lifts)} lifts) + augmented 3DS ports')

        # ── 차량 등록 ──
        for a in self.oht_agents:
            self.mcs.register_vehicle(a.id, 'OHT')
        for a in self.agv_agents:
            self.mcs.register_vehicle(a.id, 'AGV')
        for fid, fd in self.s3d_floor_data.items():
            for a in fd['agents']:
                self.mcs.register_vehicle(a.id, fid)

        # ── FromTo CSV → 1-stage Recipe per OD pair (옵션) ──
        if self._fromto_csv:
            agv_port_nodes = list(self.agv_planner._port_nodes)
            self._register_fromto_recipes(self._fromto_csv, self._fromto_scale,
                                          agv_port_nodes)

        # ── Recipe 등록 (옵션) ──
        if self._recipe_file:
            self._load_recipes(self._recipe_file)

        # ── WIP 초기화: 모든 3DS shelf 를 wip=1 (full) ──
        # Lift gate node 는 augmented (transit 노드, 실제 shelf 아님) → exclude
        if self._recipe_file:
            self.mcs.init_wip(
                ['3DS_F1', '3DS_F2', '3DS_F3'],
                count=1,
                exclude_nodes=set(self._lift_floor_by_gate.keys()),
            )
            wip_total = sum(p.wip_count for p in self.mcs.ports.values())
            print(f'[WIP-INIT] 3DS shelves filled (wip=1 each), total wip={wip_total}')

        # ── 생산 시작 ──
        # 우선순위: FromTo Poisson > OHT 단독 검증 > recipe > AGV CONWIP > 기본
        if self._fromto_csv:
            rate_per_min = self._fromto_total_rate_per_min
            print(f'[FROMTO-POISSON] Σλ={rate_per_min:.3f} loads/min '
                  f'({len(self.mcs.recipes)} OD recipes)')
            self.agv_env.step(0.0)
            self._dispatch_defer_replan = True
            self.mcs.enable_recipe_poisson(rate_per_min, t=0.0)
            self._dispatch_defer_replan = False
            if self._agv_pending_replan:
                self._replan_done_agvs(0.0)
        elif self._conwip_oht > 0:
            # OHT push 로직 검증용 — 다른 시스템 prod_rate=0 으로 잠금,
            # OHT system 만 N 개 load CONWIP 으로 dispatch.
            print(f'[OHT-CONWIP] target={self._conwip_oht}  '
                  f'(검증 모드: recipe / AGV CONWIP 무시)')
            for _p in self.mcs.ports.values():
                _p.prod_rate = 0.0
            # OHT env 한 번 step → 차량 IDLE→DONE flush (is_vehicle_free 통과용)
            self.oht_env.step(0.0)
            self.mcs.enable_conwip('OHT', self._conwip_oht, t=0.0)
        elif self._recipe_conwip > 0 or self._recipe_rate > 0:
            # 모든 transport 환경의 TRY_ADVANCE flush → 차량들이 DONE 상태로 진입
            # (이 단계 없으면 _mcs_is_*_free 콜백이 False 반환 → _do_assign 이 dispatch 못함)
            self.agv_env.step(0.0)
            for _fid, _fd in self.s3d_floor_data.items():
                _fd['env'].step(0.0)
            self._dispatch_defer_replan = True
            if self._recipe_conwip > 0:
                print(f'[RECIPE-CONWIP] target={self._recipe_conwip}')
                self.mcs.enable_recipe_conwip(self._recipe_conwip, t=0.0)
            else:
                print(f'[RECIPE-POISSON] rate={self._recipe_rate} loads/min')
                self.mcs.enable_recipe_poisson(self._recipe_rate, t=0.0)
            self._dispatch_defer_replan = False
            if self._agv_pending_replan:
                self._replan_done_agvs(0.0)
        elif self._conwip_agv > 0:
            # CONWIP 모드: AGV prod_rate 강제 0, target 만큼 즉시 spawn + 1회 batched replan
            print(f'[CONWIP] AGV system: WIP target={self._conwip_agv}')
            # AGV 들이 IDLE→DONE 으로 전이되도록 env 를 한 번 step (TRY_ADVANCE flush).
            # 이 작업이 없으면 _do_assign 의 is_vehicle_free 체크가 모두 False 가 됨.
            self.agv_env.step(0.0)
            self._dispatch_defer_replan = True
            self.mcs.enable_conwip('AGV', self._conwip_agv, t=0.0)
            self._dispatch_defer_replan = False
            # 다른 시스템 (OHT/3DS) 은 prod_rate=0 이라 no-op
            self.mcs.start_production(t=0.0)
            # 모든 AGV dispatch 후 한 번에 path planning
            if self._agv_pending_replan:
                self._replan_done_agvs(0.0)
        else:
            self.mcs.start_production(t=0.0)

    def _register_fromto_recipes(self, csv_path: str, scale: float,
                                  port_nodes: list):
        """FromTo CSV → 1-stage Recipe per (src, dst). weight = λ_ij × scale.

        모든 OD pair 가 같은 src wip 풀을 공유하지 않도록, port wip_count 를 큰 값
        (= len(port_nodes)) 으로 채워서 항상 spawn 가능하게 함.
        """
        table = load_fromto_matrix(csv_path, port_nodes)
        if not table:
            raise RuntimeError(
                f'FromTo CSV {csv_path} 에 layout 포트({port_nodes}) 에 해당하는 항목 없음')
        self._fromto_table = table
        total_rate_per_sec = sum(table.values()) * scale
        self._fromto_total_rate_per_min = total_rate_per_sec * 60.0
        for (src, dst), lam in table.items():
            rid = f'fromto_{src}_{dst}'
            recipe = Recipe(
                id=rid,
                stages=[RecipeStage(system='AGV', src=src, dst=dst)],
                weight=lam * scale,
            )
            self.mcs.register_recipe(recipe)
        # 모든 src port wip 를 충분히 채워서 OD spawn 이 wip 부족으로 막히지 않게.
        for nid in port_nodes:
            pk = f'AGV:{nid}'
            port = self.mcs.ports.get(pk)
            if port is not None:
                port.wip_count = max(port.wip_count, len(port_nodes))

    def _load_recipes(self, path: str):
        """Recipe JSON 파일 로드 + 검증 + MCS 등록.

        검증:
          1) 각 stage 의 system 이 등록되어 있는가
          2) 인접 stage 의 boundary (prev.dst → next.src) 가 좌표상 일치
        """
        import json as _json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = _json.load(f)
        except FileNotFoundError:
            print(f'[RECIPE] file not found: {path}')
            return

        # 시스템별 노드 좌표 lookup
        sys_nodes = {'AGV': self.amr_graph.nodes,
                     'OHT': self.oht_map.nodes}
        for fid, fd in self.s3d_floor_data.items():
            sys_nodes[fid] = fd['graph'].nodes

        recipes = data.get('recipes', [])
        loaded = 0
        for r_def in recipes:
            rid = r_def.get('id', f'R{loaded}')
            stages_def = r_def.get('stages', [])
            if not stages_def:
                continue
            stages = [RecipeStage(s['system'], s['src'], s['dst'])
                      for s in stages_def]

            # boundary 좌표 검증 (warning only)
            # LIFT 는 gate node 들이 3DS 그래프에 있으므로 별도 lookup 필요 없음 — skip
            warn = []
            for i in range(len(stages) - 1):
                a, b = stages[i], stages[i + 1]
                if a.system == 'LIFT' or b.system == 'LIFT':
                    continue  # LIFT 의 좌표 검증은 elevator floor matching 으로 충분
                na = sys_nodes.get(a.system, {}).get(a.dst)
                nb = sys_nodes.get(b.system, {}).get(b.src)
                if na is None or nb is None:
                    warn.append(f'  stage{i}->stage{i+1}: node lookup 실패')
                    continue
                # _area_offset 의 영향이 있을 수 있어 상대 거리만 검사
                # (좌표가 동일한지는 raw 가 아닌 post-offset 기준 — area 다르면 다름)
                # 일단 raw distance 체크 후 큰 값이면 warn.
                d2 = (na.x - nb.x) ** 2 + (na.y - nb.y) ** 2
                if d2 > 100.0 ** 2:  # 100mm 이상 차이
                    warn.append(f'  stage{i}.dst({a.system}:{a.dst}@{na.x/1000:.2f},{na.y/1000:.2f}) → '
                                f'stage{i+1}.src({b.system}:{b.src}@{nb.x/1000:.2f},{nb.y/1000:.2f}): '
                                f'gap={d2**0.5:.1f}mm (offset 다른 시스템간 비교는 정상)')

            try:
                recipe = Recipe(id=rid, stages=stages,
                                weight=r_def.get('weight', 1.0))
                self.mcs.register_recipe(recipe)
                loaded += 1
                print(f'[RECIPE] {rid}: {len(stages)} stages '
                      f'({" -> ".join(f"{s.system}:{s.src}->{s.dst}" for s in stages)})')
                for w in warn:
                    print(f'  WARN: {w}')
            except ValueError as e:
                print(f'[RECIPE] {rid} 등록 실패: {e}')

        print(f'[RECIPE] {loaded}/{len(recipes)} loaded')

    # ── MCS 콜백: OHT ────────────────────────────────────────────────────────

    # push 안전 margin (mm). h_min 위에 추가 여유.
    _PUSH_SAFETY_MARGIN_MM = 500.0

    def _oht_safe_distance(self) -> float:
        """안전거리 (mm) = h_min + margin.

        v6 의 leader-follower envelope 가 brake_dist 를 자체적으로 보장하므로
        push 단계에서는 h_min 위에 margin 만 더한다.
        """
        h_min = getattr(self.oht_map, 'h_min', 1150.0)
        return h_min + self._PUSH_SAFETY_MARGIN_MM

    def _oht_forward_nodes_within(self, src_node: str,
                                   distance: float) -> list:
        """src_node 에서 directed forward 로 distance(mm) 이상이 누적될 때까지
        거치는 노드 list (src 자체는 포함 X). 한 segment 라도 distance 를
        초과하면 거기서 멈추므로 list 마지막 노드까지 가면 누적 ≥ distance.
        directed loop 끝 또는 segment 누락 시 조기 종료.

        Spur 노드 (10001 등) 는 walk 에서 제외 — 같은 노드의 non-spur 이웃이
        있으면 그걸로 우회, 모든 이웃이 spur 면 종료. 이 함수 결과는 push
        destination 산출에 쓰여서 spur 가 끼면 OHT 가 dead-end 로 push 됨.
        """
        spurs = getattr(self.oht_map.gmap, 'spur_nodes', set())
        nodes = []
        cur = src_node
        accumulated = 0.0
        while accumulated < distance:
            succ = self.oht_map.adj.get(cur, [])
            non_spur = [s for s in succ if s not in spurs]
            if not non_spur:
                break
            nxt = non_spur[0]
            seg = self.oht_map.segments.get((cur, nxt))
            if seg is None:
                break
            accumulated += seg.length
            cur = nxt
            nodes.append(cur)
        return nodes

    def _oht_compute_push_path(self, leader_node: str,
                                follower_path: list) -> list | None:
        """leader 가 follower 와 같은 방향으로 진행해 follower 통과를 허용
        하는 push 경로 산출.

        push_dst = follower goal 에서 forward 로 안전거리 (h_min+margin) 이상
        누적된 위치. leader 가 이미 그 위치/그 너머에 있으면 leader.cur_node
        기준으로 다시 안전거리만큼 더 앞으로.
        """
        if not follower_path:
            return None
        safe_dist = self._oht_safe_distance()

        # 1차 push_dst: follower goal 너머로 safe_dist 누적된 노드
        ahead = self._oht_forward_nodes_within(follower_path[-1], safe_dist)
        if ahead:
            push_dst = ahead[-1]
            pp = self.oht_map.bfs_path(leader_node, push_dst)
            if pp and len(pp) >= 2:
                return pp
        # 2차 push_dst: leader 기준 safe_dist 누적된 노드
        # (leader 가 이미 1차 push_dst 위치/그 너머인 경우)
        ahead = self._oht_forward_nodes_within(leader_node, safe_dist)
        if ahead:
            push_dst = ahead[-1]
            pp = self.oht_map.bfs_path(leader_node, push_dst)
            if pp and len(pp) >= 2:
                return pp
        return None

    def _oht_forward_dist(self, behind, ahead, max_dist: float) -> float:
        """behind 의 head 에서 ahead 의 head 까지 directed graph forward 거리(mm).

        Use *extrapolated* offsets (`seg_offset + _dist_traveled(t-t_ref)`),
        not raw seg_offset. Raw offset only updates on EV_SEG_END so a vehicle
        cruising mid-segment looks like it's at off=0, which produces head-to-
        head gaps off by the entire segment-traversal distance and triggers
        false-positive collision alerts. The engine's `des.gap` uses
        extrapolated offsets — match it here for consistency.

        Walking is along behind's path when possible, else adj[0]. If the
        extrapolated offset overshoots the current segment, advance virtually
        through subsequent segments before starting the search.
        """
        if behind is None or ahead is None:
            return float('inf')
        if behind.seg_from is None or ahead.seg_from is None:
            return float('inf')
        t = self.sim_time

        def _virt_advance(v):
            """Return (seg_from, seg_to, off) reflecting v's extrapolated
            position; if extrapolation overshoots seg.length, walk virtually
            through subsequent segments along v.path (engine SEG_CROSS_EPS
            slack — same as graph_des_v6.gap)."""
            off = v.seg_offset + v._dist_traveled(t - v.t_ref)
            pidx = v.path_idx
            path = list(getattr(v, 'path', []) or [])
            seg_lens = getattr(v, '_seg_lengths', None)
            from graph_des_v6 import SEG_CROSS_EPS
            while pidx < len(path) - 1:
                if seg_lens is not None and pidx < len(seg_lens):
                    sl = seg_lens[pidx]
                else:
                    seg = self.oht_map.segments.get((path[pidx], path[pidx+1]))
                    sl = seg.length if seg else 0.0
                if sl <= 0 or off < sl - SEG_CROSS_EPS:
                    break
                off -= sl
                pidx += 1
            seg_from = path[pidx] if pidx < len(path) else v.seg_from
            seg_to = path[pidx + 1] if pidx + 1 < len(path) else v.seg_to
            return seg_from, seg_to, max(0.0, off), pidx

        b_from, b_to, b_off, b_pidx = _virt_advance(behind)
        a_from, a_to, a_off, _ = _virt_advance(ahead)

        # Same segment after virtual advance
        if b_from == a_from and b_to == a_to:
            d = a_off - b_off
            return d if d >= 0 else float('inf')

        seg = self.oht_map.segments.get((b_from, b_to))
        if seg is None:
            return float('inf')
        accumulated = max(0.0, seg.length - b_off)
        cur = b_to
        visited = {(b_from, b_to)}
        path = list(getattr(behind, 'path', []) or [])
        pidx = b_pidx + 1 if b_pidx + 1 < len(path) else -1
        if pidx >= 0 and pidx < len(path) and path[pidx] != cur:
            pidx = -1
        while accumulated < max_dist:
            if pidx >= 0 and pidx + 1 < len(path):
                pidx += 1
                nxt = path[pidx]
            else:
                succs = self.oht_map.adj.get(cur, [])
                nxt = succs[0] if succs else None
                pidx = -1
            if nxt is None:
                return float('inf')
            seg_key = (cur, nxt)
            if seg_key in visited:
                return float('inf')
            visited.add(seg_key)
            if cur == a_from and nxt == a_to:
                return accumulated + a_off
            seg = self.oht_map.segments.get(seg_key)
            if seg is None:
                return float('inf')
            accumulated += seg.length
            cur = nxt
        return float('inf')

    def _oht_safety_check(self, t: float):
        """매 step 모든 OHT 쌍의 graph-forward 거리 측정. h_min 미만이면 위반.

        - 평행 track 상의 OHT 는 graph 상 멀어 자동으로 제외 (Euclidean 으로
          가깝게 측정되더라도 실제 충돌 불가능).
        - 위반은 한 쌍당 1회만 보고 (계속 가까이 있으면 spam 방지).
        - 위반 발생 시 v6 내부 state, leader 정보 dump.
        """
        if not getattr(self, 'oht_agents', None):
            return
        h_min = getattr(self.oht_map, 'h_min', 1150.0)
        vlen = getattr(self.oht_map, 'vehicle_length', 750.0)

        if not hasattr(self, '_oht_violation_pairs'):
            self._oht_violation_pairs: set = set()

        agents = self.oht_agents
        max_dist = h_min * 2.0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if a.vehicle is None or b.vehicle is None:
                    continue
                av = a.vehicle; bv = b.vehicle
                gap_ab = self._oht_forward_dist(bv, av, max_dist)  # b behind, a ahead
                gap_ba = self._oht_forward_dist(av, bv, max_dist)  # a behind, b ahead
                gap = min(gap_ab, gap_ba)
                key = tuple(sorted((a.id, b.id)))
                if gap < h_min:
                    if key in self._oht_violation_pairs:
                        continue
                    self._oht_violation_pairs.add(key)
                    severity = 'COLLISION' if gap < vlen else 'CLOSE'
                    print(f'\n[OHT-SAFETY-{severity}] t={t:.2f} gap={gap:.0f}mm '
                          f'(h_min={h_min:.0f}, vlen={vlen:.0f})')
                    print(f'  H{a.id}: ({a.x:.0f},{a.y:.0f}) state={a.state} '
                          f'pidx={av.path_idx}/{len(av.path)} '
                          f'seg={av.seg_from}->{av.seg_to} off={av.seg_offset:.0f} '
                          f'vel={av.vel:.1f} dest={av.dest_node} '
                          f'reached={av.dest_reached} '
                          f'leader={av.leader.id if av.leader else None} '
                          f'stop={av.stop_reason}')
                    print(f'  H{b.id}: ({b.x:.0f},{b.y:.0f}) state={b.state} '
                          f'pidx={bv.path_idx}/{len(bv.path)} '
                          f'seg={bv.seg_from}->{bv.seg_to} off={bv.seg_offset:.0f} '
                          f'vel={bv.vel:.1f} dest={bv.dest_node} '
                          f'reached={bv.dest_reached} '
                          f'leader={bv.leader.id if bv.leader else None} '
                          f'stop={bv.stop_reason}')
                    print(f'  H{a.id} path tail: {av.path[max(0, av.path_idx-1):av.path_idx+5]}')
                    print(f'  H{b.id} path tail: {bv.path[max(0, bv.path_idx-1):bv.path_idx+5]}')
                elif gap > h_min * 1.5:
                    self._oht_violation_pairs.discard(key)

    def _oht_idle_spread(self, t: float):
        """OHT 가 BLOCKED 인데 그를 막고 있는 leader 가 정차 중이면 leader 를
        follower 너머로 reassign. 매 step 호출 — DES 의 자연스러운 흐름.

        - cooldown 없음. BLOCKED 가 풀리면 후보에서 자동 제외, leader 가
          MOVING 이면 후보에서 자동 제외 → 매 step 호출이 무한 반복은 아님.
        - 같은 leader 에게 이미 동일 push 가 발행되어 진행 중이면 leader.state
          가 MOVING 이라 push_targets 에서 제외된다. 진짜로 BLOCKED 가 풀리지
          않는 케이스가 있다면 그건 reassign 자체가 작동하지 않는다는 뜻이고
          별도 조사 대상 (cooldown 으로 가리면 안 됨).
        """
        if not getattr(self, 'oht_agents', None):
            return

        # 진단 — 상태 변화가 있을 때만 한 줄 요약 출력
        states = tuple(sorted((a.id, a.state, a.cur_node) for a in self.oht_agents))
        prev = getattr(self, '_oht_state_snapshot', None)
        if states != prev:
            self._oht_state_snapshot = states
            if self._oht_verbose:
                parts = []
                for a in sorted(self.oht_agents, key=lambda x: x.id):
                    tag = f'H{a.id}:{a.state[:3]}@{a.cur_node}'
                    if a.state == BLOCKED and a.vehicle is not None:
                        tag += f'({a.vehicle.stop_reason})'
                    parts.append(tag)
                print(f'[OHT-STATE] t={t:.1f}  ' + ' '.join(parts))

        # BLOCKED OHT 후보 — follower 의 load 여부와 무관하게 모두 검사.
        # follower 가 작업 중 (load 보유) 이어도 그의 path 위에 idle leader 가
        # 막고 있으면 leader 를 비켜줘야 한다. follower 의 path 자체는 손대지
        # 않으므로 작업 흐름은 유지된다.
        blockers = []
        for a in self.oht_agents:
            if a.state != BLOCKED:
                continue
            v = a.vehicle
            if v is None or not v.path:
                continue
            # 잔여 path. follower 의 v.path 는 dispatch 시 이미 안전거리만큼
            # extend 되어 있으므로 추가 lookahead 불필요.
            rem = list(v.path[v.path_idx:])
            rem_set = set(rem)
            for other in self.oht_agents:
                if other.id == a.id:
                    continue
                if other.cur_node in rem_set and other.cur_node != a.cur_node:
                    blockers.append((a, other))
                    break

        if not blockers:
            return

        # 각 leader 를 follower path 끝 + N hop 너머로 reassign.
        # 동일 leader 가 여러 follower 에 등록되어도 1번만 처리.
        seen_leaders: set = set()
        for follower, leader in blockers:
            if leader.id in seen_leaders:
                continue
            seen_leaders.add(leader.id)
            b = self.mcs.bindings.get(leader.id)
            if b is not None and b.load is not None:
                continue
            if leader.state in (MOVING, FOLLOWING):
                continue
            # follower 의 잔여 path 끝까지 + N hop 너머로 leader 진행
            f_rem = follower.vehicle.path[follower.vehicle.path_idx:]
            push_path = self._oht_compute_push_path(leader.cur_node, f_rem)
            if push_path is None:
                continue
            # sticky 진단: 같은 leader 에게 같은 push_dst 가 반복 발행되면
            # reassign 이 깨우지 못하는 케이스. v6 내부 상태 함께 dump.
            counter = getattr(self, '_oht_spread_repeat_count', None)
            if counter is None:
                counter = {}
                self._oht_spread_repeat_count = counter
            key = (leader.id, push_path[-1])
            counter[key] = counter.get(key, 0) + 1
            if counter[key] >= 3:
                lv = leader.vehicle
                fv = follower.vehicle
                print(f'[OHT-STUCK] reassign 무시: H{leader.id} '
                      f'state={lv.state} vel={lv.vel:.2f} '
                      f'pidx={lv.path_idx}/{len(lv.path)} '
                      f'seg={lv.seg_from}->{lv.seg_to} off={lv.seg_offset:.1f} '
                      f'dest={lv.dest_node} reached={lv.dest_reached} '
                      f'leader={lv.leader.id if lv.leader else None} '
                      f'stop_reason={getattr(lv, "stop_reason", "?")} '
                      f'stop_dist={lv.stop_dist}')
                print(f'             follower H{follower.id} '
                      f'state={fv.state} vel={fv.vel:.2f} '
                      f'pidx={fv.path_idx}/{len(fv.path)} '
                      f'seg={fv.seg_from}->{fv.seg_to} off={fv.seg_offset:.1f} '
                      f'leader={fv.leader.id if fv.leader else None}')
                # ZCU lock 보유/대기 상태
                holders = {lid: (h.id if h else None)
                           for lid, h in self.oht_env._zcu_holders.items()}
                print(f'             ZCU holders={holders}')
                counter[key] = 0   # 리셋
            self.oht_env.reassign(leader, push_path, t)
            self._oht_done_notified.discard(leader.id)
            self._oht_pushed.add(leader.id)
            if self._oht_verbose:
                print(f'[OHT-SPREAD] H{leader.id} {leader.cur_node} → '
                      f'{push_path[-1]} (unblock H{follower.id})  t={t:.2f}')

    def _mcs_dispatch_oht(self, vid: int, goal_node: str, t: float):
        """MCS 가 OHT 차량에 이동 명령. follower 의 path 상에 정차해 있는
        IDLE/STOP OHT 는 leader 가 되어 follower 의 plan 을 막으므로 (v6 의
        leader-follower envelope 가 gap_d - h_min 까지만 brake 를 commit)
        path 밖 free port 로 비키는 reassign 을 먼저 발행한다 (= push).

        push 절차:
          1. follower path 산출 (BFS shortest)
          2. push 대상 = (load 없음) ∧ (state ∈ IDLE/BLOCKED/DONE) ∧
                          (잔여 path ∩ follower path ≠ ∅)
          3. 각 대상마다 used_nodes (follower path ∪ goal ∪ 다른 OHT 잔여 path)
             와 disjoint 한 free port 중 BFS-hop 가장 가까운 곳을 push_dst 로 선택
          4. push_path 전체가 follower path 와 disjoint 한지 검증 (겹치면 다음 후보)
          5. 모두 성공해야 follower reassign — 하나라도 push 실패 시 follower
             는 reassign 하지 않고 다음 dispatch cycle 에서 재시도 (leader 뒤
             강제 정차 회피)
        """
        agent = None
        for a in self.oht_agents:
            if a.id == vid:
                agent = a
                break
        if agent is None:
            return
        path = self.oht_map.bfs_path(agent.cur_node, goal_node)
        # path 를 goal 너머로 안전거리 (h_min + margin) mm 만큼 extend.
        # v6 의 _update_leader 가 path 위 차량만 leader 로 검출하므로, path
        # 끝 너머에 다른 OHT 가 (정차/진입/dynamic 으로) 있으면 인식 못 해
        # follower 가 goal 도달 시 안전거리 위반. path 를 미리 extend 해두면
        # v6 leader-follower envelope 가 자동 처리. dest_node 는 goal_node 로
        # 명시 → 도착 판정은 dest_reached + 어댑터 cur_node==dest_node fallback.
        if path and len(path) >= 2:
            extra = self._oht_forward_nodes_within(
                path[-1], self._oht_safe_distance())
            path = list(path) + extra
        # full-loop dispatch (origin 이 path 중간에 다시 등장) 표시
        loop_tag = ''
        if path and len(path) >= 2 and agent.cur_node in path[1:]:
            loop_tag = '  [LOOP-AROUND]'
        if self._oht_verbose:
            print(f'[MCS->OHT] vid={vid} {agent.cur_node} → {goal_node}  '
                  f'len={len(path) if path else 0}  state={agent.state}  '
                  f't={t:.2f}{loop_tag}')
        if not path:
            return
        if len(path) < 2:
            # src == cur_node — 차량이 이미 goal 에 있음. reassign 불가하지만
            # MCS 에 도착 통보를 해줘야 dwell→delivery 흐름이 진행된다.
            # 이 처리가 빠지면 vid 가 영원히 DONE 으로 정차하면서 leader 되고,
            # 다른 OHT 의 path 를 막아 BLOCKED 도미노가 발생한다.
            from mcs_unified import post_vehicle_arrived
            b = self.mcs.bindings.get(vid)
            if b is not None and b.load is not None:
                post_vehicle_arrived(self._mcs_heap, self._mcs_seq,
                                     vid, b.token, t)
                if self._oht_verbose:
                    print(f'[OHT-IMMEDIATE] vid={vid} 이미 {goal_node} 도달 — '
                          f'즉시 도착 통보')
            return

        path_set = set(path)   # path 에 이미 lookahead 포함됨

        # ── push 대상 선별 ──────────────────────────────────────────────
        # IDLE 차량 (load 미보유 ∧ 진행 중 아님) 만 push 대상.
        # 작업 중 OHT (b.load) 는 자기 dispatch path 가 따로 있으므로 손대지 않음.
        # MOVING/FOLLOWING 은 곧 path 비워줄 가능성 — 일단 손대지 않고 자연 진행.
        push_targets = []
        for other in self.oht_agents:
            if other.id == vid:
                continue
            b = self.mcs.bindings.get(other.id)
            if b is not None and b.load is not None:
                continue
            if other.state in (MOVING, FOLLOWING):
                continue
            v = other.vehicle
            if v is None or not v.path:
                continue
            rem = set(v.path[v.path_idx:])
            if not (rem & path_set):
                continue
            push_targets.append(other)

        # ── leader 들에 push_path reassign ─────────────────────────────
        # push_dst = follower goal 너머 N hop. leader 가 거기까지 진행하면
        # follower 가 goal 도달 시 안전거리 확보됨.
        # 가까운 leader 먼저 reassign 해야 v6 commit 순서 자연스러움.
        push_targets.sort(key=lambda o: len(
            self.oht_map.bfs_path(agent.cur_node, o.cur_node) or [None]*999))
        push_plans = []
        unpushed = []
        for other in push_targets:
            push_path = self._oht_compute_push_path(other.cur_node, path)
            if push_path is None:
                unpushed.append(other)
                continue
            push_plans.append((other, push_path))

        for other, push_path in push_plans:
            self.oht_env.reassign(other, push_path, t)
            self._oht_done_notified.discard(other.id)
            self._oht_pushed.add(other.id)
            if self._oht_verbose:
                print(f'[OHT-PUSH] H{other.id} {other.cur_node} → {push_path[-1]} '
                      f'(follow-ahead H{vid} → {goal_node})')

        if unpushed and self._oht_verbose:
            print(f'[OHT-PUSH-FAIL] H{vid} → {goal_node}  '
                  f'unpushed={[o.id for o in unpushed]} (BFS path 산출 실패)')

        # follower 의 path 는 cur → goal + 안전거리 lookahead 까지 extend 됨.
        # dest_node 는 goal_node 명시 (path[-1] 은 extend 끝점이라 다름).
        self.oht_env.reassign(agent, path, t)
        if agent.vehicle is not None:
            agent.vehicle.dest_node = goal_node
            agent.vehicle.dest_reached = False
        self._oht_done_notified.discard(vid)
        self._oht_pushed.discard(vid)  # 실제 MCS 작업 → push 상태 해제

    def _mcs_get_oht_node(self, vid: int) -> str | None:
        for a in self.oht_agents:
            if a.id == vid:
                return a.cur_node
        return None

    def _mcs_get_oht_distance(self, src: str, dst: str) -> float:
        path = self.oht_map.bfs_path(src, dst)
        if len(path) < 2:
            return float('inf')
        dist = 0.0
        for i in range(len(path) - 1):
            seg = self.oht_map.segments.get((path[i], path[i+1]))
            if seg:
                dist += seg.length
        return dist

    # ── MCS 콜백: AGV ────────────────────────────────────────────────────────

    # ── 테스트용: 작업을 받을 수 있는 AGV 제한 (None이면 전체 허용) ──
    _agv_work_allow = None  # None이면 전체 허용, {100}이면 A100만 할당

    def _mcs_is_agv_free(self, vid: int) -> bool:
        """MCS phase 가 IDLE 이면 free — TAPG state 무관.

        이전: TAPG state == DONE 만 free. → UNLOAD 후 siding extension 진행 중인
        AGV 가 phase=IDLE 인데도 dispatch 안 됨 → CONWIP 에서 waiting load 누적.
        이제: MCS 가 phase 만 보고 결정. 새 dispatch 시 wrapper replan 이 *현재
        위치에서* 새 path 짬 → 기존 siding-bound path overwrite.
        """
        if self._agv_work_allow is not None and vid not in self._agv_work_allow:
            return False
        b = self.mcs.bindings.get(vid)
        if b is None:
            return False
        # MCS-phase IDLE (no load assigned) 이면 free
        return b.phase == VehicleJobState.IDLE

    def _on_warmup_done(self):
        """Warmup 종료 시 KPI 데이터 초기화 (= 이후부터 진짜 측정)."""
        n_log = len(self._agv_movement_log)
        self._agv_movement_log.clear()
        self._kpi_start_t = self.sim_time
        self._kpi_completed_loads = []
        # Per-OD M state visits: {(od_tuple, phase, m_sid): count}
        self._kpi_move_visits = {}
        # Per S state blocked wait time: {s_sid: total_seconds}
        self._kpi_s_wait = {}
        # Tracking state for wait/move - reset
        self._agv_wait_start = {}    # aid -> (sid, t_start)
        self._agv_last_path_idx = {a.id: a.path_idx for a in self.agv_agents}
        print(f'\n[WARMUP DONE] t={self.sim_time:.1f}s. '
              f'KPI 측정 시작 (이전 {n_log} 이벤트 폐기).')

    def _update_kpi_tracking(self):
        """매 frame 호출. AGV state 변화 + path 진행 기반 KPI 갱신.
        Warmup 미완료 시 skip."""
        if not self._warmup_done:
            return
        from env_tapg import WAITING as _WAITING
        sim_t = self.sim_time
        if not hasattr(self, '_kpi_move_visits'):
            self._kpi_move_visits = {}
            self._kpi_s_wait = {}
            self._agv_wait_start = {}
            self._agv_last_path_idx = {a.id: a.path_idx for a in self.agv_agents}
        for a in self.agv_agents:
            # KPI #2: S state blocked wait time (excl. L state dwell)
            is_waiting_at_s = (a.state == _WAITING
                                and 0 <= a.path_idx < len(a.raw_path)
                                and a.raw_path[a.path_idx][0].startswith('S,'))
            in_record = a.id in self._agv_wait_start
            if is_waiting_at_s and not in_record:
                self._agv_wait_start[a.id] = (a.raw_path[a.path_idx][0], sim_t)
            elif in_record and not is_waiting_at_s:
                sid, t0 = self._agv_wait_start.pop(a.id)
                self._kpi_s_wait[sid] = self._kpi_s_wait.get(sid, 0.0) + (sim_t - t0)
            # KPI #1: Per-OD M state visits (path_idx 진행 시점)
            last_idx = self._agv_last_path_idx.get(a.id, a.path_idx)
            if a.path_idx > last_idx:
                b = self.mcs.bindings.get(a.id)
                if b and b.load:
                    od = (b.load.src_port, b.load.dst_port)
                    if b.phase in (VehicleJobState.TO_PICKUP, VehicleJobState.LOADING):
                        phase = 'retrieve'
                    elif b.phase in (VehicleJobState.TO_DELIVERY, VehicleJobState.UNLOADING):
                        phase = 'deliver'
                    else:
                        phase = None
                    if phase is not None:
                        for k in range(last_idx, min(a.path_idx, len(a.raw_path))):
                            sid = a.raw_path[k][0]
                            if sid.startswith('M,'):
                                key = (od, phase, sid)
                                self._kpi_move_visits[key] = (
                                    self._kpi_move_visits.get(key, 0) + 1)
            self._agv_last_path_idx[a.id] = a.path_idx

    def _log_movement(self, aid: int, kind: str, src: str, dst: str,
                      t: float, load_id: int | None = None):
        """이동 이벤트 기록. kind ∈ {'retrieve', 'deliver', 'push'}.
        push 의 경우 *attributed_phase* 자동 기록 (= MCS phase 시점).
        Warmup 미완료 시 기록 안 함."""
        if not self._warmup_done:
            return
        ev = {'t': t, 'aid': aid, 'type': kind,
              'src': src, 'dst': dst, 'load_id': load_id}
        # Push 이면 시점의 MCS phase 도 기록 (= retrieve/deliver 종속용)
        # TO_PICKUP, LOADING -> 'retrieve' (= pickup 과정의 일부)
        # TO_DELIVERY, UNLOADING -> 'deliver'
        if kind == 'push':
            b = self.mcs.bindings.get(aid)
            if b and b.phase in (VehicleJobState.TO_PICKUP,
                                  VehicleJobState.LOADING):
                ev['attr_phase'] = 'retrieve'
            elif b and b.phase in (VehicleJobState.TO_DELIVERY,
                                    VehicleJobState.UNLOADING):
                ev['attr_phase'] = 'deliver'
            else:
                ev['attr_phase'] = 'idle'   # IDLE 등 task 없는 경우
        self._agv_movement_log.append(ev)

    def _mcs_dispatch_agv(self, vid: int, goal_node: str, t: float):
        """MCS가 AGV 차량에 이동 명령 — SIPP replan."""
        b = self.mcs.bindings.get(vid)
        phase = b.phase.value if b else '?'
        load_id = b.load.load_id if (b and b.load) else None
        print(f'\n[MCS→AGV] dispatch A{vid} → {goal_node}  '
              f't={t:.2f}  phase={phase}  load={load_id}')
        agent = None
        for a in self.agv_agents:
            if a.id == vid:
                agent = a
                break
        if agent is None:
            return
        # 현재 위치 추출
        cur_node = self._mcs_get_agv_node(vid)
        if cur_node is None:
            return

        # 이미 목적지에 있음 → 이동 불필요, 즉시 도착 처리
        if cur_node == goal_node:
            from mcs_unified import post_vehicle_arrived
            b = self.mcs.bindings.get(vid)
            if b:
                post_vehicle_arrived(self._mcs_heap, self._mcs_seq,
                                     vid, b.token, t)
            return

        # Goal 기준 이동 분류 (phase 가 아닌 goal_node 비교 — MCS state machine
        # 의 eager dispatch 가 LOADING phase 에서 dst_port 로 firing 하므로 phase
        # 만 보면 deliver 가 누락됨).
        if b and b.load:
            if goal_node == b.load.src_port:
                self._log_movement(vid, 'retrieve', cur_node, goal_node, t, load_id)
            elif goal_node == b.load.dst_port:
                self._log_movement(vid, 'deliver', cur_node, goal_node, t, load_id)

        self._agv_goals[vid] = goal_node
        self._agv_pushed.discard(vid)  # 실제 MCS 작업 → push 상태 해제
        self._agv_pending_replan.add(vid)
        self._agv_done_notified.discard(vid)
        self._agv_arrived_notified.discard(vid)  # 새 경로 → arrival 재감지
        self._agv_arrival_idx.pop(vid, None)
        # 배치 모드: replan 호출 보류 (호출자가 batch 종료 후 일괄 replan).
        if self._dispatch_defer_replan:
            return
        self._replan_done_agvs(t)

    def _mcs_get_agv_node(self, vid: int) -> str | None:
        for a in self.agv_agents:
            if a.id == vid:
                idx = a.path_idx
                if idx < len(a.raw_path):
                    return a.raw_path[idx][0].split(',')[1]
                elif a.raw_path:
                    return a.raw_path[-1][0].split(',')[1]
        return None

    def _mcs_is_3ds_free(self, fid: str, vid: int) -> bool:
        """3DS 셔틀이 DES에서 DONE 상태인지 확인."""
        fd = self.s3d_floor_data.get(fid)
        if fd is None:
            return False
        for a in fd['agents']:
            if a.id == vid:
                return a.state == AGV_DONE
        return False

    # ── MCS 콜백: 3DS ────────────────────────────────────────────────────────

    def _mcs_dispatch_3ds(self, fid: str, vid: int, goal_node: str, t: float):
        """MCS가 3DS 셔틀에 이동 명령 — SIPP replan."""
        fd = self.s3d_floor_data.get(fid)
        if fd is None:
            return
        agent = None
        for a in fd['agents']:
            if a.id == vid:
                agent = a
                break
        if agent is None:
            return

        # 현재 위치 추출
        cur_node = self._mcs_get_3ds_node(fid, vid)
        if cur_node is None:
            return
        print(f'[MCS->{fid}] vid={vid} {cur_node} → {goal_node}  t={t:.2f}')

        # 이미 목적지에 있음 → 즉시 도착 처리
        if cur_node == goal_node:
            from mcs_unified import post_vehicle_arrived
            b = self.mcs.bindings.get(vid)
            if b:
                post_vehicle_arrived(self._mcs_heap, self._mcs_seq,
                                     vid, b.token, t)
            return

        planner = fd['planner']
        tapg_env = fd['env']

        # 다른 active 셔틀의 constraint 수집
        constraints = []
        for a in fd['agents']:
            if a.id == vid:
                continue
            idx = a.path_idx
            if idx >= len(a.raw_path):
                nid = a.raw_path[-1][0].split(',')[1]
                stop_sid = f'S,{nid},0'
                constraints.append({'agent': a.id, 'loc': stop_sid,
                                    'timestep': (t, float('inf'))})
                state = planner._get_state(stop_sid)
                if state:
                    for aff_id in state.affect_state:
                        constraints.append({'agent': a.id, 'loc': aff_id,
                                            'timestep': (t, float('inf'))})
            else:
                remaining = a.raw_path[idx:]
                cs = planner._build_constraints(remaining, a.id)
                constraints.extend(cs)

        # Recipe 모드면 alternate goal 비활성 (정확한 src 가 fixed)
        recipe_mode = (self._recipe_conwip > 0 or self._recipe_rate > 0)
        result = planner.plan(
            {vid: cur_node}, {vid: goal_node},
            existing_constraints=constraints,
            start_times={vid: t},
            disable_alternate_goal=recipe_mode,
            quiet_fail=recipe_mode,
        )

        if result and result.paths.get(vid):
            new_path = result.paths[vid]
            if len(new_path) >= 2:
                # dwell time 추가
                last_sid, last_t = new_path[-1]
                new_path.append((last_sid, last_t + self.mcs.dwell_time))
                fd['goals'][vid] = goal_node
                tapg_env.extend_agents_batch({vid: new_path}, t)
                self._s3d_done_notified.discard(vid)
                return

        # SIPP 실패 → MCS 작업 해제, 차량 IDLE 복귀
        b = self.mcs.bindings.get(vid)
        if b and b.load is not None:
            load = b.load
            # Load를 포트 대기큐에 되돌림
            load.state = LoadState.WAITING
            load.vehicle_id = None
            port_key = f'{fid}:{load.src_port}'
            port = self.mcs.ports.get(port_key)
            if port:
                port.waiting_loads.append(load)
            b.load = None
            b.phase = VehicleJobState.IDLE
            self.mcs.kpi.mark_idle(vid, t)

    def _mcs_get_3ds_node(self, fid: str, vid: int) -> str | None:
        fd = self.s3d_floor_data.get(fid)
        if fd is None:
            return None
        for a in fd['agents']:
            if a.id == vid:
                if a.raw_path:
                    idx = a.path_idx
                    if idx < len(a.raw_path):
                        return a.raw_path[idx][0].split(',')[1]
                    return a.raw_path[-1][0].split(',')[1]
        return None

    def _retreat_3ds_at_gates(self, t: float):
        """3DS 셔틀이 lift gate 노드에 idle 정착 시, 가까운 비-gate buffer 로 retreat.

        Gate 는 articulation point — 셔틀이 inf-park 하면 다른 셔틀의 SIPP 경로를
        막는다. deliver 완료 후 binding.load=None 이고 raw_path 끝 = gate 인 셔틀을
        찾아서 비-gate port 로 SIPP plan 후 path 확장.

        매 update tick 에서 호출되지만, retreat 후 raw_path 끝이 비-gate 가 되므로
        다음 tick 부터는 skip (idempotent).
        """
        if not self._lift_floor_by_gate:
            return
        gate_set = set(self._lift_floor_by_gate.keys())

        for fid, fd in self.s3d_floor_data.items():
            planner = fd['planner']
            tapg_env = fd['env']
            agents = fd['agents']
            port_set = set(planner._port_nodes)
            non_gate_ports = [p for p in port_set if p not in gate_set]
            if not non_gate_ports:
                continue

            # 다른 active 셔틀의 종착지 = 점유
            used = set()
            for a in agents:
                if a.raw_path:
                    used.add(a.raw_path[-1][0].split(',')[1])

            for a in agents:
                if a.state != AGV_DONE:
                    continue
                b = self.mcs.bindings.get(a.id)
                if b and b.load is not None:
                    continue   # 작업 중인 셔틀
                if not a.raw_path:
                    continue
                last_sid, last_t = a.raw_path[-1]
                cur = last_sid.split(',')[1]
                if cur not in gate_set:
                    continue   # gate 가 아니면 retreat 불필요

                # 가장 가까운 free non-gate port
                cur_node = fd['graph'].nodes.get(cur)
                if cur_node is None:
                    continue
                free_targets = [p for p in non_gate_ports
                                if p != cur and p not in used]
                if not free_targets:
                    continue
                free_targets.sort(key=lambda p: (
                    (fd['graph'].nodes[p].x - cur_node.x) ** 2 +
                    (fd['graph'].nodes[p].y - cur_node.y) ** 2
                ) if p in fd['graph'].nodes else float('inf'))
                target = free_targets[0]

                # 다른 셔틀의 inf 점유를 constraint 로
                constraints = []
                for other in agents:
                    if other.id == a.id or not other.raw_path:
                        continue
                    o_sid = other.raw_path[-1][0]
                    o_t = other.raw_path[-1][1]
                    if other.path_idx >= len(other.raw_path):
                        # DONE — inf block
                        constraints.append({
                            'agent': other.id, 'loc': o_sid,
                            'timestep': (last_t, float('inf')),
                        })
                    else:
                        # 진행 중 — 남은 path 전체를 constraint
                        remaining = other.raw_path[other.path_idx:]
                        cs = planner._build_constraints(remaining, other.id)
                        constraints.extend(cs)

                result = planner.plan(
                    {a.id: cur}, {a.id: target},
                    existing_constraints=constraints,
                    start_times={a.id: last_t},
                    disable_alternate_goal=True,
                    quiet_fail=True,
                )
                if not (result and result.paths.get(a.id)):
                    continue
                new_path = result.paths[a.id]
                if len(new_path) < 2:
                    continue

                fd['goals'][a.id] = target
                tapg_env.extend_agents_batch({a.id: new_path}, t)
                self._s3d_done_notified.discard(a.id)
                used.add(target)
                used.discard(cur)
                print(f'[3DS-RETREAT] {fid} vid={a.id} {cur} → {target}  t={t:.2f}')

    # ── MCS 콜백: LIFT (elevator) ────────────────────────────────────────────
    def _mcs_is_lift_free(self, vid: int) -> bool:
        """엘리베이터에 active load 가 없으면 free."""
        b = self.mcs.bindings.get(vid)
        return b is None or b.load is None

    def _mcs_get_lift_node(self, vid: int) -> 'str | None':
        """엘리베이터의 현재 위치 = cur_floor 의 gate node."""
        lift = self._lift_by_vid.get(vid)
        if lift is None:
            return None
        return lift.gate_nodes.get(lift.cur_floor)

    def _mcs_dispatch_lift(self, vid: int, goal_node: str, t: float):
        """MCS 가 엘리베이터에 이동 명령. goal_node 의 층으로 single-leg 이동.

        도착 시 VEHICLE_ARRIVED 이벤트를 MCS 힙에 게시 (token 으로 stale 보호).
        """
        lift = self._lift_by_vid.get(vid)
        if lift is None:
            print(f'[LIFT] dispatch failed: vid {vid} not found')
            return
        target_floor = self._lift_floor_by_gate.get(goal_node)
        if target_floor is None:
            print(f'[LIFT] dispatch failed: gate {goal_node} not in lift {lift.id}')
            return
        print(f'[LIFT->] {lift.id} (vid={vid}) {lift.cur_floor} → {target_floor} '
              f'gate={goal_node} t={t:.2f}')

        # MCS binding 의 token 캡처 — 도착 시점에 동일해야 유효
        bind_token = self.mcs.get_binding_token(vid)

        from mcs_unified import post_vehicle_arrived as _pva

        def _on_arrive(t_arr: float):
            _pva(self._mcs_heap, self._mcs_seq, vid, bind_token, t_arr)

        lift.move_to(target_floor, t, on_arrive=_on_arrive)

    def _init_elevators(self):
        """KaistTB 맵에서 엘리베이터 생성."""
        import json as _json
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            jdata = _json.load(f)

        # 독립 힙 (3DS TAPG 힙과 별도 -엘리베이터는 시간 기반 DES)
        self._lift_heap = []
        self.lift_ctrl = ElevatorController(self._lift_heap)

        # 층 높이 (areas.viewShift.y 또는 노드 z 좌표)
        area_heights = {'1': 0.0, '2': 3000.0, '3': 6000.0}

        for lift_data in jdata.get('lifts', []):
            self.lift_ctrl.add_lift_from_map(
                lift_data, area_heights,
                speed=1000.0,       # mm/s (수직)
                xfer_duration=3.0,  # 적재/하역 3초
                capacity=1,
            )

        # 주기적 자동 요청 타이머
        self._lift_next_request_t = 5.0   # 첫 요청 시각
        self._lift_request_interval = 15.0  # 요청 간격

        n_lifts = len(self.lift_ctrl.lifts)
        if n_lifts:
            print(f'  Elevators: {n_lifts} lifts loaded')

    def _step_elevators(self, sim_time: float):
        """엘리베이터 DES step + 주기적 자동 요청 (recipe 모드 OFF)."""
        # 힙 이벤트 처리
        import heapq as _hq
        while self._lift_heap and self._lift_heap[0].t <= sim_time:
            ev = _hq.heappop(self._lift_heap)
            self._lift_total_events = getattr(self, '_lift_total_events', 0) + 1
            self.lift_ctrl.handle_event(ev)

        # Recipe 모드 (LIFT 가 MCS 시스템) 면 random demo 요청 비활성화
        if self._recipe_conwip > 0 or self._recipe_rate > 0:
            return

        # 주기적 자동 요청 (데모용 — recipe 없을 때만)
        if sim_time >= self._lift_next_request_t:
            self._lift_next_request_t = sim_time + self._lift_request_interval
            floors = ['1', '2', '3']
            from_f = random.choice(floors)
            to_f = random.choice([f for f in floors if f != from_f])
            result = self.lift_ctrl.request_auto(from_f, to_f, sim_time)
            if result:
                lid, rid = result
                lift = self.lift_ctrl.get_lift(lid)

    def _replan_3ds_floor(self, fid: str, sim_time: float):
        """한 층의 DONE 셔틀을 순차 replan -한 대씩 계획 후 constraint 추가."""
        fd = self.s3d_floor_data[fid]
        planner = fd['planner']
        tapg_env = fd['env']
        agents = fd['agents']
        done_ids = list(fd['pending_replan'])

        if not done_ids:
            return

        # 기존 active agent의 constraint 수집
        base_constraints = []
        for a in agents:
            if a.id in done_ids:
                continue
            idx = a.path_idx
            if idx >= len(a.raw_path):
                # active지만 경로 끝 → 현재 위치를 영구 block
                nid = a.raw_path[-1][0].split(',')[1]
                stop_sid = f'S,{nid},0'
                base_constraints.append({
                    'agent': a.id, 'loc': stop_sid,
                    'timestep': (sim_time, float('inf')),
                })
                # affect state도 block
                state = planner._get_state(stop_sid)
                if state:
                    for aff_id in state.affect_state:
                        base_constraints.append({
                            'agent': a.id, 'loc': aff_id,
                            'timestep': (sim_time, float('inf')),
                        })
                continue
            remaining = a.raw_path[idx:]
            cs = planner._build_constraints(remaining, a.id)
            base_constraints.extend(cs)

        # DONE 셔틀들의 현재 위치도 constraint로 추가
        # (아직 replan 안 된 셔틀이 앉아있는 노드 보호)
        done_positions = {}
        for aid in done_ids:
            a = tapg_env.agents.get(aid)
            if a and a.raw_path:
                nid = a.raw_path[-1][0].split(',')[1]
                done_positions[aid] = nid

        # 점유/목표 노드 추적 (AGV와 동일 로직)
        occupied = set()
        for a in agents:
            if a.id not in done_ids and a.raw_path:
                occupied.add(a.raw_path[-1][0].split(',')[1])
        targeted = {g for aid, g in fd['goals'].items() if aid not in done_ids}
        used = occupied | targeted | set(done_positions.values())

        # 한 대씩 순차 replan
        all_constraints = list(base_constraints)

        # 아직 replan 안 된 DONE 셔틀의 위치를 임시 constraint로 추가
        pending_position_constraints = {}
        for aid in done_ids:
            nid = done_positions.get(aid)
            if nid is None:
                continue
            cs = []
            stop_sid = f'S,{nid},0'
            cs.append({'agent': aid, 'loc': stop_sid,
                       'timestep': (sim_time, float('inf'))})
            state = planner._get_state(stop_sid)
            if state:
                for aff_id in state.affect_state:
                    cs.append({'agent': aid, 'loc': aff_id,
                               'timestep': (sim_time, float('inf'))})
            pending_position_constraints[aid] = cs

        for aid in done_ids:
            a = tapg_env.agents.get(aid)
            if a is None:
                continue
            nid = done_positions.get(aid, '')
            used.add(nid)

            # 다른 DONE 셔틀의 위치를 constraint에 포함
            plan_constraints = list(all_constraints)
            for other_aid, cs in pending_position_constraints.items():
                if other_aid != aid:
                    plan_constraints.extend(cs)

            # 새 목표 선택
            ports = list(planner._port_nodes)
            random.shuffle(ports)
            goal = None
            for p in ports:
                if p != nid and p not in used:
                    goal = p
                    break
            if goal is None:
                for p in ports:
                    if p != nid:
                        goal = p
                        break
            if goal is None:
                continue

            result = planner.plan(
                {aid: nid}, {aid: goal},
                existing_constraints=plan_constraints,
                start_times={aid: sim_time},
            )

            if result and result.paths.get(aid):
                new_path = result.paths[aid]
                if len(new_path) >= 2:
                    fd['goals'][aid] = goal
                    used.add(goal)
                    tapg_env.extend_agents_batch({aid: new_path}, sim_time)
                    # 이 경로를 다음 셔틀의 constraint로 추가
                    cs = planner._build_constraints(new_path, aid)
                    all_constraints.extend(cs)
                    # 이 셔틀은 이제 움직이므로 위치 constraint 제거
                    pending_position_constraints.pop(aid, None)

        fd['pending_replan'].clear()

    # ── Agent setup ──────────────────────────────────────────────────────────

    def _make_oht_agent(self, aid: int, excluded: set) -> OHTAgent | None:
        """OHT 차량 생성 — 최소 길이(2 node) path 만 부여.

        Random walk 비활성화. Vehicle 은 path[0] 에서 path[1] 으로 한 번 이동
        후 정차 (auto-extend 는 vehicle.job sentinel 로 차단됨). 이후 MCS dispatch
        가 reassign 으로 실제 목적지 부여.

        Spur 노드(10001 등 U-turn dead-end)는 시작점에서 제외하고, 첫 step
        목적지로도 선택하지 않는다. 이렇게 안 하면 random.shuffle 결과로
        OHT 가 spur 에서 spawn 되거나 spur 로 진입한 채 정차해버림.
        """
        spurs = getattr(self.oht_map.gmap, 'spur_nodes', set())
        nodes = list(self.oht_map.nodes.keys())
        random.shuffle(nodes)
        for start in nodes:
            if start in excluded or start in spurs:
                continue
            nbrs = [n for n in self.oht_map.adj.get(start, [])
                    if n not in spurs]
            if not nbrs:
                continue
            # Vehicle 은 ≥2 노드 path 필요 — start 와 그 다음 인접 노드 한 개
            path = [start, nbrs[0]]
            color = OHT_COLORS[aid % len(OHT_COLORS)]
            return OHTAgent(aid, color, path,
                            self.oht_map.vehicle_length * (1000/1108))
        return None

    def _init_oht_agents(self):
        excluded = set()
        for _ in range(self._n_oht):
            a = self._make_oht_agent(self._oht_next_id, excluded)
            if a:
                excluded |= self.oht_map.nearby_nodes(a.node_path[0],
                                                       self.oht_map.h_min)
                self._oht_next_id += 1
                self.oht_agents.append(a)
                self.oht_env.add_agent(a, t_start=0.0)

    def _find_stop_state(self, planner, nid: str) -> str | None:
        """노드에서 유효한 S state ID 를 찾는다.
        먼저 표준 4-heading (0/90/180/270) 시도, 없으면 graph 내 모든 S state
        스캔해서 해당 nid 의 임의 heading 반환. 송도처럼 corridor 가 90도 단위
        아닌 layout 도 지원.
        """
        for angle in [0, 90, 180, 270]:
            sid = f'S,{nid},{angle}'
            if planner._get_state(sid):
                return sid
        # Fallback: scan all stop states for matching nid
        for sid in planner.graph.stop_states_raw:
            if sid.split(',')[1] == nid:
                return sid
        return None

    def _configure_coarse_mode(self, amr_graph):
        """Coarse mode flag + cut_nodes + checkpoints 를 agv_env 에 set.
        TAPGEnvironment 재생성 시점 (init / reset / replan) 마다 호출 필요."""
        if self._planner_type != 'coarse':
            return
        self.agv_env._coarse_mode = True
        self.agv_env._cut_nodes = self._cut_nodes
        self.agv_env._cut_to_port = self._cut_to_port
        _port_nids = set(amr_graph.ports.values()) if amr_graph.ports else set()
        _siding_nids = set(self._siding_tier_a)
        self.agv_env._checkpoints = (
            set(self._branching_nodes) |
            _port_nids |
            _siding_nids)
        self.agv_env._rest_places = _port_nids | _siding_nids
        self.agv_env._coarse_debug = getattr(self, '_coarse_debug', False)
        # Push 후보: *충분히 긴 corridor* (interm len >= 3) 의 last-grey 만.
        # 짧은 segment last-grey 는 port/cut 진입로라 push location 부적합.
        # planner 가 아직 안 만들어진 시점이면 later setup.
        if hasattr(self, 'agv_planner') and hasattr(self.agv_planner, 'segments'):
            push_cands = set()
            for seg_id, interm in self.agv_planner.segments.items():
                if len(interm) >= 3:
                    push_cands.add(interm[-1])
            self._coarse_push_candidates = list(push_cands)
            print(f'[COARSE-PUSH] {len(self._coarse_push_candidates)} candidate '
                  f'nodes (last-grey of each segment)')

    def _init_agv_agents(self):
        """AGV를 ports ∪ sidings ∪ push_candidates 위치에 DONE 상태로 배치.
        Farthest-first: collision 거리(vehicle_length) × 2 이상 떨어진 곳만 선택.
        부족 시 threshold 완화."""
        # 후보 풀: ports + sidings (graph 에 존재하는 노드만) +
        # coarse mode 의 push candidates (= segment last-grey).
        start_pool = list(self.agv_planner._port_nodes)
        # Tier-A 만 spawn 후보 (cut-safe 보장). Tier-B 는 polygon overlap 으로
        # 인접 corridor 의 unique transit 점 차단할 수 있음 → 제외.
        siding_avail = [s for s in self._siding_tier_a if s in self.amr_graph.nodes]
        start_pool.extend(siding_avail)
        # Coarse mode: push 후보 (segment last-grey) 도 spawn 가능 위치.
        push_cands = getattr(self, '_coarse_push_candidates', None)
        if push_cands:
            for n in push_cands:
                if n in self.amr_graph.nodes:
                    start_pool.append(n)
        start_pool = list(set(start_pool))
        random.shuffle(start_pool)
        self._agv_start_positions = {}
        n = min(self._n_agv, len(start_pool))

        # Farthest-first: vehicle_length × 2 이상 거리 보장 (spawn collision 방지)
        min_dist = self.amr_graph.vehicle_length * 2.0
        nodes_obj = self.amr_graph.nodes
        picked = []
        for nid in start_pool:
            if len(picked) >= n:
                break
            if nid not in nodes_obj:
                continue
            cand = nodes_obj[nid]
            too_close = any(
                math.hypot(cand.x - nodes_obj[p].x,
                           cand.y - nodes_obj[p].y) < min_dist
                for p in picked)
            if not too_close:
                picked.append(nid)
        # 부족 시 threshold 완화 단계적
        for relax in (1.0, 0.5, 0.0):
            if len(picked) >= n:
                break
            relax_min = self.amr_graph.vehicle_length * relax
            for nid in start_pool:
                if len(picked) >= n:
                    break
                if nid in picked or nid not in nodes_obj:
                    continue
                cand = nodes_obj[nid]
                too_close = any(
                    math.hypot(cand.x - nodes_obj[p].x,
                               cand.y - nodes_obj[p].y) < relax_min
                    for p in picked)
                if not too_close:
                    picked.append(nid)
        if len(picked) < n:
            print(f'[INIT] WARN: only {len(picked)}/{n} spawn slots — pool 부족')
            n = len(picked)
        print(f'[INIT] {n} AGVs spawning from pool of {len(start_pool)} '
              f'(ports={len(self.agv_planner._port_nodes)}, sidings={len(siding_avail)})')

        self.agv_agents = []
        for i in range(n):
            aid = self._agv_next_id
            nid = picked[i]
            self._agv_start_positions[aid] = nid
            self._agv_next_id += 1

            # 해당 노드의 실제 S state 사용
            stop_sid = self._find_stop_state(self.agv_planner, nid)
            if stop_sid is None:
                print(f'[INIT] WARN: no S state at spawn {nid} — agent A{aid} skipped')
                continue
            raw_path = [(stop_sid, 0.0)]
            color = AGV_COLORS[(aid - 100) % len(AGV_COLORS)]
            a = TAPGAgent(aid, color, raw_path)
            self.agv_agents.append(a)

        # TAPG 환경 구성 — 점유 정보 등록
        self.agv_env = TAPGEnvironment(self.amr_graph, accel=500.0, decel=500.0)
        self._configure_coarse_mode(self.amr_graph)
        if self.agv_agents:
            self.agv_env.setup(self.agv_agents, t_start=0.0)
        self._plan_status = f'{n} AGVs idle at ports'

    def _plan_agv_paths(self, t_start: float):
        """Run prioritized SIPP → create TAPGAgents → setup TAPGEnvironment."""
        if not self._agv_start_positions and not self.agv_agents:
            return

        # Current positions: from start_positions (init) or current agent positions
        positions = {}
        if self.agv_agents:
            for a in self.agv_agents:
                # Extract node from current state
                nid = a.raw_path[-1][0].split(',')[1] if a.state == AGV_DONE else \
                      a.raw_path[a.path_idx][0].split(',')[1]
                positions[a.id] = nid
        else:
            positions = dict(self._agv_start_positions)

        goals = self.agv_planner.assign_random_goals(positions)
        self._agv_goals = goals

        result = self.agv_planner.plan(positions, goals, start_times={
            aid: t_start for aid in positions
        })

        if result is None:
            self._plan_status = 'Plan FAILED'
            return

        # Create TAPGAgents from planned state-level paths
        self.agv_agents = []
        for i, aid in enumerate(sorted(result.paths)):
            raw_path = result.paths[aid]
            if len(raw_path) < 2:
                continue
            color = AGV_COLORS[(aid - 100) % len(AGV_COLORS)]
            a = TAPGAgent(aid, color, raw_path)
            self.agv_agents.append(a)

        # Setup TAPG environment
        self.agv_env = TAPGEnvironment(self.amr_graph, accel=500.0, decel=500.0)
        self._configure_coarse_mode(self.amr_graph)
        if self.agv_agents:
            self.agv_env.setup(self.agv_agents, t_start=t_start)

        n_planned = len(self.agv_agents)
        self._plan_status = f'Planned {n_planned} AGVs'
        self._save_plan_snapshot('initial_plan')

    def _save_plan_snapshot(self, label: str):
        """Planning 시점의 전체 AGV 경로 + TAPG + D-key dump를 단일 로그 파일에 추가."""
        import pickle
        self._plan_log_counter += 1
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

        # 1) 텍스트 로그: 단일 파일에 append
        log_path = os.path.join(log_dir, 'plan_log.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\n{"="*80}\n')
            f.write(f'[PLAN #{self._plan_log_counter}] {label} | t={self.sim_time:.2f}s\n')
            f.write(f'{"="*80}\n\n')
            for a in self.agv_agents:
                goal = self._agv_goals.get(a.id, '?')
                f.write(f'A{a.id-100} state={a.state} goal={goal} '
                        f'path_idx={a.path_idx}/{len(a.raw_path)} '
                        f'claim_idx={a.claim_idx} '
                        f'pos=({a.x:.0f},{a.y:.0f})\n')
                for idx, (sid, t) in enumerate(a.raw_path):
                    markers = []
                    if idx == a.path_idx: markers.append('CURRENT')
                    if a.path_idx <= idx < a.claim_idx: markers.append('CLAIMED')
                    marker = f' <<< {",".join(markers)}' if markers else ''
                    f.write(f'  [{idx:3d}] {sid:35s} t={t:.4f}{marker}\n')
                f.write('\n')

            # D-key dump: TAPG 상태 + 각 agent의 BLOCKED_BY
            G = self.agv_env.G
            f.write(f'TAPG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n\n')

            f.write('--- Agent dependency ---\n')
            for a in self.agv_agents:
                idx = a.path_idx
                if idx >= len(a.raw_path): continue
                # Find next M/R
                next_idx = idx
                while next_idx < len(a.raw_path) and a.raw_path[next_idx][0].startswith('S,'):
                    next_idx += 1
                if next_idx >= len(a.raw_path): continue
                ns, nt = a.raw_path[next_idx]
                nk = self.agv_env._nk(ns, a.id, nt)
                blockers = []
                if nk in G:
                    blockers = [f'A{p[1]-100}:{p[0][:20]}'
                                for p in G.predecessors(nk) if p[1] != a.id]
                if blockers:
                    f.write(f'  A{a.id-100} wants {ns} BLOCKED_BY={blockers}\n')
                else:
                    f.write(f'  A{a.id-100} wants {ns} (free)\n')
            f.write('\n')

        # 2) TAPG DAG pickle (최신 1개만 유지)
        try:
            tapg_data = {
                'G_nodes': list(self.agv_env.G.nodes(data=True)),
                'G_edges': list(self.agv_env.G.edges()),
                'agent_paths': {a.id: a.raw_path for a in self.agv_agents},
                'agent_states': {a.id: {
                    'state': a.state, 'path_idx': a.path_idx,
                    'claim_idx': a.claim_idx,
                    'x': a.x, 'y': a.y, 'v': a.v,
                    'goal': self._agv_goals.get(a.id),
                    '_tapg_node': a._tapg_node,
                } for a in self.agv_agents},
                'sim_time': self.sim_time,
                'label': label,
            }
            pkl_path = os.path.join(log_dir, 'latest_tapg.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(tapg_data, f)
        except Exception as e:
            print(f'[WARN] Failed to save TAPG: {e}')

    def _dump_agv_status(self):
        """D키: 현재 AGV 경로 + TAPG 상태를 콘솔과 파일(logs/agv_dump.txt) 양쪽에 출력."""
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        fpath = os.path.join(log_dir, 'agv_dump.txt')
        f = open(fpath, 'w', encoding='utf-8')

        def emit(s: str = ''):
            print(s)
            f.write(s + '\n')

        emit(f'\n{"="*80}')
        emit(f'[DUMP] t={self.sim_time:.2f}s  AGVs={len(self.agv_agents)}  '
             f'TAPG nodes={self.agv_env.G.number_of_nodes()} '
             f'edges={self.agv_env.G.number_of_edges()}')
        emit(f'{"="*80}')

        for a in self.agv_agents:
            goal = self._agv_goals.get(a.id, '?')
            b = self.mcs.bindings.get(a.id)
            phase = b.phase.value if b else '?'
            emit(f'\nA{a.id-100}: state={a.state} phase={phase} goal={goal} '
                 f'pos=({a.x:.0f},{a.y:.0f}) v={a.v:.0f} '
                 f'path_idx={a.path_idx}/{len(a.raw_path)} claim_idx={a.claim_idx}')

            # Coarse mode: 미도달 path[k] 중 *어느 state* 가 *어느 AGV* 에
            # 의해 blocked 인지 표시. _is_claimable_coarse 를 self-exclude 로
            # 호출해서 blocker 신원 추출.
            if getattr(self.agv_env, '_coarse_mode', False):
                blockers = []
                for k in range(a.path_idx, min(len(a.raw_path), a.path_idx + 20)):
                    sid, t = a.raw_path[k]
                    nk = self.agv_env._nk(sid, a.id, t)
                    orig = a.claim_idx
                    a.claim_idx = k
                    ok = self.agv_env._is_claimable_coarse(nk, a.id)
                    a.claim_idx = orig
                    if not ok:
                        blk = getattr(self.agv_env, '_last_block_info', '?')
                        blockers.append(f'    [{k}] {sid:35s} BLOCKED: {blk}')
                if blockers:
                    emit('  -- live blockers (forward path) --')
                    for line in blockers[:5]:
                        emit(line)

            # Remaining path (from current index)
            for i in range(a.path_idx, min(len(a.raw_path), a.path_idx + 15)):
                sid, t = a.raw_path[i]
                nk = self.agv_env._nk(sid, a.id, t)
                in_g = nk in self.agv_env.G
                cross_pred = []
                if in_g:
                    cross_pred = [f'A{p[1]-100}:{p[0][:20]}'
                                  for p in self.agv_env.G.predecessors(nk)
                                  if p[1] != a.id]
                marker = ' <<< CURRENT' if i == a.path_idx else ''
                blocked = f' BLOCKED_BY={cross_pred}' if cross_pred else ''
                dag = '' if in_g else ' [NOT_IN_DAG]'
                emit(f'  [{i:3d}] {sid:35s} t={t:.2f}{dag}{blocked}{marker}')

            if len(a.raw_path) > a.path_idx + 15:
                emit(f'  ... +{len(a.raw_path) - a.path_idx - 15} more states')

        # Coarse mode: live dependency graph (= blocker AGV summary)
        if getattr(self.agv_env, '_coarse_mode', False):
            from collections import defaultdict
            deps = defaultdict(set)   # blocked_agent -> {blocker_agents}
            for a in self.agv_agents:
                for k in range(a.path_idx, min(len(a.raw_path), a.path_idx + 30)):
                    sid, t = a.raw_path[k]
                    nk = self.agv_env._nk(sid, a.id, t)
                    orig = a.claim_idx
                    a.claim_idx = k
                    ok = self.agv_env._is_claimable_coarse(nk, a.id)
                    a.claim_idx = orig
                    if not ok:
                        blk_info = getattr(self.agv_env, '_last_block_info', '')
                        # parse blocker id from "V_X@..."
                        import re
                        m = re.search(r'by V(\d+)@', blk_info)
                        if m:
                            deps[a.id - 100].add(int(m.group(1)))
                        break   # first blocker only
            if deps:
                emit('\n--- Live dependency graph (coarse) ---')
                for aid in sorted(deps.keys()):
                    blockers = sorted(deps[aid])
                    emit(f'  A{aid} waits for: {", ".join(f"A{b}" for b in blockers)}')
                # Detect cycles
                def find_cycle():
                    for start in deps:
                        path = [start]
                        cur = start
                        while True:
                            blockers = deps.get(cur, set())
                            if not blockers:
                                break
                            nxt = next(iter(blockers))
                            if nxt in path:
                                return path[path.index(nxt):] + [nxt]
                            path.append(nxt)
                            cur = nxt
                            if len(path) > 20:
                                break
                    return None
                cyc = find_cycle()
                if cyc:
                    emit(f'  [DEADLOCK CYCLE] {" -> ".join(f"A{a}" for a in cyc)}')

        # TAPG cross edges summary
        G = self.agv_env.G
        cross_edges = [(u, v) for u, v in G.edges() if u[1] != v[1]]
        if cross_edges:
            emit(f'\n--- Cross-agent edges ({len(cross_edges)}) ---')
            for u, v in cross_edges[:50]:
                emit(f'  A{u[1]-100}:{u[0][:20]}(t={u[2]:.1f}) -> '
                     f'A{v[1]-100}:{v[0][:20]}(t={v[2]:.1f})')
            if len(cross_edges) > 50:
                emit(f'  ... +{len(cross_edges)-50} more')

        # wait_queues snapshot — 누가 누구를 기다리는지
        if self.agv_env.wait_queues:
            emit(f'\n--- wait_queues ({len(self.agv_env.wait_queues)} nodes) ---')
            for nk, waiters in list(self.agv_env.wait_queues.items())[:30]:
                if waiters:
                    emit(f'  {nk[0][:25]}(A{nk[1]-100} t={nk[2]:.1f}) '
                         f'← waiters=[{",".join(f"A{w-100}" for w in waiters)}]')

        # Cycle 검사
        try:
            import networkx as nx
            cyc = nx.find_cycle(G)
            emit(f'\n!!! CYCLE DETECTED in DAG ({len(cyc)} edges) !!!')
            for u, v, *_ in cyc:
                emit(f'  A{u[1]-100}:{u[0][:20]}(t={u[2]:.1f}) -> '
                     f'A{v[1]-100}:{v[0][:20]}(t={v[2]:.1f})')
        except nx.NetworkXNoCycle:
            emit(f'\n(no cycle in DAG)')
        except Exception as ex:
            emit(f'\n(cycle check error: {ex})')

        emit(f'{"="*80}\n')
        f.close()
        print(f'[DUMP] saved → {fpath}')
        self._save_plan_snapshot('manual_dump')

    def _dump_replan_history(self):
        """D키에서 호출: 각 AGV의 마지막 replan 이력을 파일로 저장."""
        if not hasattr(self, '_replan_history') or not self._replan_history:
            print('[REPLAN HISTORY] No replan history recorded.')
            return
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        fpath = os.path.join(log_dir, 'replan_history.txt')
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(f'{"="*70}\n')
            f.write(f'[REPLAN HISTORY] t={self.sim_time:.2f}s  {len(self._replan_history)} agents\n')
            f.write(f'{"="*70}\n\n')
            for aid in sorted(self._replan_history.keys()):
                rec = self._replan_history[aid]
                a = self.agv_env.agents.get(aid)
                cur_info = ''
                if a:
                    cur_info = (f' (now: {a.state} path_idx={a.path_idx}/{len(a.raw_path)}'
                                f' claim_idx={a.claim_idx})')
                f.write(f'A{aid-100} planned at t={rec["time"]:.2f}s{cur_info}\n')
                f.write(f'  done_ids={[d-100 for d in rec["done_ids"]]}\n')

                # Active agents snapshot at plan time
                f.write(f'  Active agents at plan time:\n')
                for snap in rec['active_snapshot']:
                    f.write(f'    A{snap["id"]-100} state={snap["state"]} '
                            f'cur={snap["cur_sid"]} '
                            f'path_idx={snap["path_idx"]}/{snap["total"]} '
                            f'claim_idx={snap["claim_idx"]}\n')

                # 전체 경로
                f.write(f'  Full path ({len(rec["path"])} steps):\n')
                for si in range(len(rec['path'])):
                    sid, t = rec['path'][si]
                    f.write(f'    [{si:3d}] {sid:<36s} t={t:.4f}\n')

                # 전체 constraints (이 경로가 지나는 노드 관련)
                f.write(f'  All relevant constraints ({len(rec["constraints"])}):\n')
                for c in sorted(rec['constraints'], key=lambda x: x['timestep'][0]):
                    f.write(f'    A{c["agent"]-100}: {c["loc"]}'
                            f' [{c["timestep"][0]:.4f}, {c["timestep"][1]:.4f}]\n')
                f.write(f'\n')
        print(f'[REPLAN HISTORY] Saved to {fpath}')

    def _check_agv_collisions(self):
        """AGV 간 물리적 근접 감지 → collision_log.txt에 상세 정보 기록."""
        if not self.agv_agents:
            return
        THRESHOLD = self.amr_graph.vehicle_length  # 780mm
        t = self.sim_time
        t_bucket = int(t * 2)  # 0.5초 단위 dedup

        active = [(a, a.x, a.y) for a in self.agv_agents]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ai, ax, ay = active[i]
                aj, bx, by = active[j]
                d = math.hypot(ax - bx, ay - by)
                if d < THRESHOLD:
                    pair_key = (min(ai.id, aj.id), max(ai.id, aj.id), t_bucket)
                    if pair_key in self._collision_pairs_logged:
                        continue
                    self._collision_pairs_logged.add(pair_key)
                    self._collision_count += 1

                    # 상세 정보 수집
                    def agent_info(a):
                        idx = a.path_idx
                        cur_sid = a.raw_path[idx][0] if idx < len(a.raw_path) else '?'
                        cur_t = a.raw_path[idx][1] if idx < len(a.raw_path) else 0
                        tapg_node = a._tapg_node
                        goal = self._agv_goals.get(a.id, '?')
                        return (f'A{a.id-100} state={a.state} '
                                f'pos=({a.x:.0f},{a.y:.0f}) v={a.v:.0f} '
                                f'cur_state={cur_sid} path_idx={idx}/{len(a.raw_path)} '
                                f'tapg={tapg_node} goal={goal}')

                    # TAPG dependency 확인
                    tapg_info = ''
                    G = self.agv_env.G
                    # 현재 실행중인 TAPG 노드 간 edge 확인
                    if ai._tapg_node and aj._tapg_node:
                        has_ij = G.has_edge(ai._tapg_node, aj._tapg_node) if (ai._tapg_node in G and aj._tapg_node in G) else False
                        has_ji = G.has_edge(aj._tapg_node, ai._tapg_node) if (ai._tapg_node in G and aj._tapg_node in G) else False
                        ij_done = ai._tapg_node not in G if ai._tapg_node else True
                        ji_done = aj._tapg_node not in G if aj._tapg_node else True
                        tapg_info = (f'  TAPG edge {ai.id}->{aj.id}={has_ij} '
                                     f'{aj.id}->{ai.id}={has_ji} '
                                     f'completed: {ai.id}={ij_done} {aj.id}={ji_done}')

                    # raw_path에서 현재 state의 TAPG 노드 존재 여부 확인
                    for label, a in [('ai', ai), ('aj', aj)]:
                        idx = a.path_idx
                        if idx < len(a.raw_path):
                            sid, ct = a.raw_path[idx]
                            nk = (sid, a.id, round(ct, 6))
                            in_g = nk in G
                            is_completed = nk not in G
                            preds = []
                            if in_g:
                                preds = [(p[0][:25], p[1], f'{p[2]:.2f}',
                                          p not in G)
                                         for p in G.predecessors(nk) if p[1] != a.id]
                            tapg_info += (f'\n  {label} nk={nk} in_G={in_g} '
                                          f'completed={is_completed} '
                                          f'cross_pred={preds}')

                    msg = (f'[COLLISION #{self._collision_count}] t={t:.2f}s dist={d:.0f}mm\n'
                           f'  {agent_info(ai)}\n'
                           f'  {agent_info(aj)}\n'
                           f'{tapg_info}\n')

                    self._collision_log_f.write(msg + '\n')
                    self._collision_log_f.flush()
                    print(msg)

                    # ── 첫 collision에서만 replan 이력 출력 + 자동 pause ──
                    if self._collision_count == 1 and hasattr(self, '_replan_history'):
                        for coll_aid in [ai.id, aj.id]:
                            rec = self._replan_history.get(coll_aid)
                            if rec is None:
                                continue
                            rpt = f'\n{"="*70}\n'
                            rpt += f'[REPLAN TRACE] A{coll_aid-100} planned at t={rec["time"]:.2f}s\n'
                            rpt += f'  done_ids={[d-100 for d in rec["done_ids"]]}\n'
                            rpt += f'  Active agents at replan time:\n'
                            for snap in rec['active_snapshot']:
                                rpt += (f'    A{snap["id"]-100} state={snap["state"]} '
                                        f'cur={snap["cur_sid"]} '
                                        f'path_idx={snap["path_idx"]}/{snap["total"]} '
                                        f'claim_idx={snap["claim_idx"]} '
                                        f'pos=({snap["pos"][0]:.0f},{snap["pos"][1]:.0f})\n')
                            # 충돌 노드 추출
                            coll_nodes = set()
                            for a_ in [ai, aj]:
                                idx_ = min(a_.path_idx, len(a_.raw_path) - 1)
                                sid_ = a_.raw_path[idx_][0]
                                p_ = sid_.split(',')
                                if len(p_) >= 2: coll_nodes.add(p_[1])
                                if len(p_) >= 3 and p_[0] == 'M': coll_nodes.add(p_[2])
                            rpt += f'  New path (collision-node states only):\n'
                            for si, (sid, t_) in enumerate(rec['path']):
                                p_ = sid.split(',')
                                sn = set()
                                if len(p_) >= 2: sn.add(p_[1])
                                if len(p_) >= 3 and p_[0] == 'M': sn.add(p_[2])
                                if sn & coll_nodes:
                                    rpt += f'    [{si:3d}] {sid:<36s} t={t_:.4f}\n'
                            rpt += f'  Constraints on collision nodes ({len(rec["constraints"])}):\n'
                            for c_ in sorted(rec['constraints'], key=lambda x: x['timestep'][0]):
                                p_ = c_['loc'].split(',')
                                cn = set()
                                if len(p_) >= 2: cn.add(p_[1])
                                if len(p_) >= 3 and p_[0] == 'M': cn.add(p_[2])
                                if cn & coll_nodes:
                                    rpt += (f'    A{c_["agent"]-100}: {c_["loc"]} '
                                            f'[{c_["timestep"][0]:.2f}, {c_["timestep"][1]:.2f}]\n')
                            rpt += '='*70
                            print(rpt)
                            self._collision_log_f.write(rpt + '\n')
                            self._collision_log_f.flush()
                        # 자동 pause
                        self.running = False
                        self._plan_status = 'PAUSED -collision detected'

    def _build_active_agent_constraints(self, excluded_aids: set,
                                         sim_time: float) -> list:
        """활성 AGV들의 현재 상태 + claimed + unclaimed 경로를 SIPP constraint로 변환.

        `_replan_done_agvs` 와 `_extend_with_exit` 에서 동일하게 쓰인다.

        excluded_aids : plan 대상 agent들 (constraint 제외)
        """
        constraints = []
        for a in self.agv_agents:
            if a.id in excluded_aids:
                continue
            idx = a.path_idx
            if idx >= len(a.raw_path):
                idx = len(a.raw_path) - 1

            # 현재 위치 물리적 점유
            cur_sid, cur_t = a.raw_path[idx]
            cur_state = self.agv_planner._get_state(cur_sid)
            if cur_state is not None:
                if idx + 1 < len(a.raw_path):
                    t_end_cur = max(a.raw_path[idx + 1][1], sim_time)
                else:
                    t_end_cur = float('inf')
                constraints.append({
                    'agent': a.id, 'loc': cur_sid,
                    'timestep': (sim_time, t_end_cur),
                })
                for aff_id in cur_state.affect_state:
                    aff = self.agv_planner._get_state(aff_id)
                    aff_cost = aff.cost if aff else 0.0
                    constraints.append({
                        'agent': a.id, 'loc': aff_id,
                        'timestep': (max(0.0, sim_time - aff_cost), t_end_cur),
                    })

            # Claimed 구간 (path_idx ~ claim_idx)
            claimed_path = a.raw_path[idx:a.claim_idx]
            if claimed_path:
                t_claim_start = sim_time
                for ci, (sid, t) in enumerate(claimed_path):
                    state = self.agv_planner._get_state(sid)
                    if state is None:
                        continue
                    if ci + 1 < len(claimed_path):
                        t_end = claimed_path[ci + 1][1]
                    else:
                        unclaimed_rest = a.raw_path[a.claim_idx:]
                        if unclaimed_rest:
                            t_end = unclaimed_rest[0][1]
                        else:
                            t_end = t + (state.cost if state.cost else 0)

                    constraints.append({
                        'agent': a.id, 'loc': sid,
                        'timestep': (t_claim_start, t_end),
                    })
                    for aff_id in state.affect_state:
                        aff = self.agv_planner._get_state(aff_id)
                        aff_cost = aff.cost if aff else 0.0
                        constraints.append({
                            'agent': a.id, 'loc': aff_id,
                            'timestep': (max(0.0, t_claim_start - aff_cost), t_end),
                        })

            # Unclaimed 구간
            unclaimed_path = a.raw_path[a.claim_idx:]
            if unclaimed_path:
                cs = self.agv_planner._build_constraints(unclaimed_path, a.id)
                constraints.extend(cs)

        return constraints

    def _compute_active_claims(self, exclude_aid: int) -> set:
        """다른 AGV 들이 *현재 점유* 또는 *예정 claim* 중인 node id set.

        Exit-phase constraint-aware planning 용. 포함:
          - 다른 AGV 의 현재 위치 (raw_path[path_idx])
          - claim 범위 [path_idx, claim_idx) 의 destination node
          - raw_path[-1] (= DONE/idle 영구 점유 위치)
        """
        blocked = set()
        for a in self.agv_agents:
            if a.id == exclude_aid:
                continue
            if not a.raw_path:
                continue
            # 1. raw_path[-1] (= 영구 점유 가능성)
            last_node = a.raw_path[-1][0].split(',')[1]
            blocked.add(last_node)
            # 2. 현재 위치 + claim 범위
            idx_start = min(a.path_idx, len(a.raw_path) - 1)
            idx_end = min(a.claim_idx, len(a.raw_path))
            for i in range(idx_start, max(idx_start + 1, idx_end)):
                sid = a.raw_path[i][0]
                parts = sid.split(',')
                if len(parts) >= 2:
                    blocked.add(parts[1])
                if parts[0] == 'M' and len(parts) >= 3:
                    blocked.add(parts[2])
        return blocked

    def _bfs_nearest_candidate(self, from_node: str, candidates,
                                used: set) -> 'str | None':
        """Forward BFS on directed graph - return first reachable candidate.

        Idle push 전용: from_node 에서 *forward* 로 진짜 가까운 candidate.
        _pick_nearest_free_park 의 reverse-Dijkstra heuristic 은
        cand→from_node 거리라 단방향 graph 에서 부정확.
        """
        if not candidates:
            return None
        cand_set = set(candidates) - {from_node} - set(used)
        if not cand_set:
            return None
        from collections import deque
        queue = deque([from_node])
        seen = {from_node}
        while queue:
            n = queue.popleft()
            if n in cand_set:
                return n
            for s in self.amr_graph.adj.get(n, []):
                if s not in seen:
                    seen.add(s)
                    queue.append(s)
        return None

    def _pick_nearest_free_park(self, from_node: str, used: set,
                                 candidates: list = None,
                                 anchor: 'str | None' = None) -> 'str | None':
        """가장 가까운 free park (port + tier-A siding) 선택.

        - 거리 metric: planner heuristic (graph 거리, reverse Dijkstra) — corridor
          우회 고려. SIPP 가 실제 도달 가능한 후보만 finite cost.
        - anchor: 거리 기준점. None 이면 from_node. anchor != from_node 인
          경우 = "anchor 근방으로 push" (예: AGV 가 곧 dst_port 로 UNLOAD 가는데
          그 근처 siding 으로 미리 push).
        - used 에 들어있지 않고, from_node 자신도 아닌 후보.
        """
        if candidates is None:
            candidates = self._park_nodes
        anchor_node = anchor if anchor is not None else from_node
        # planner heuristic — h_table[state_id] = state 에서 anchor 까지의 graph cost
        if (anchor_node not in self.amr_graph.nodes
                or anchor_node not in self.agv_planner.graph.nodes):
            # fallback: 첫 번째 free
            free = [p for p in candidates if p != from_node and p not in used]
            return free[0] if free else None
        h = self.agv_planner._heuristic(anchor_node)
        scored = []
        for p in candidates:
            if p == from_node or p in used:
                continue
            sid = self.agv_planner._find_stop_state(p)
            if sid is None:
                continue
            d = h.get(sid, float('inf'))
            if d == float('inf'):
                continue
            scored.append((d, p))
        if not scored:
            return None
        scored.sort()
        return scored[0][1]

    def _find_dep_cycles(self, dep_map: dict) -> list:
        """dep_map (waiter_aid -> blocker_aid) 의 cycle 추출.
        DFS 로 strongly connected component 중 size >= 2 인 것 반환.
        2-cycle (mutual block) 부터 N-cycle 까지 처리."""
        cycles = []
        visited = set()
        for start in list(dep_map.keys()):
            if start in visited:
                continue
            # start 부터 dep chain follow. 다시 chain 안 노드 만나면 cycle.
            chain = []
            chain_set = set()
            cur = start
            while cur is not None and cur not in chain_set:
                chain.append(cur)
                chain_set.add(cur)
                cur = dep_map.get(cur)
            visited.update(chain_set)
            if cur is not None and cur in chain_set:
                # cycle = cur ... cur (chain 의 cur idx 부터 끝까지)
                start_idx = chain.index(cur)
                cycle = chain[start_idx:]
                if len(cycle) >= 2:
                    cycles.append(cycle)
        return cycles

    def _resolve_cycle_replan(self, cycle: list, dep_map: dict,
                                block_node_of: dict, sim_time: float):
        """Cycle 해소.
          Small (2~3): 1 victim push.
          Large (4+): ceil(N/2) 명 다중 victim push - cycle 분할.
        """
        if not cycle:
            return
        # Pick victims (sorted by score, take ceil(N/2) for large cycles)
        ports_set = (set(self.amr_graph.ports.values())
                     if self.amr_graph.ports else set())
        env = self.agv_env
        def _victim_score(aid):
            a = env.agents.get(aid)
            if a is None or not a.raw_path:
                return (0, 0, 0)
            if a.path_idx >= len(a.raw_path):
                return (0, 0, 0)
            cur_sid = a.raw_path[a.path_idx][0]
            cur_node = cur_sid.split(',')[1] if ',' in cur_sid else cur_sid
            is_port = 1 if cur_node in ports_set else 0
            out_deg = len(self.amr_graph.adj.get(cur_node, []))
            remaining = len(a.raw_path) - a.path_idx
            return (is_port, out_deg, remaining)
        sorted_victims = sorted(cycle, key=_victim_score, reverse=True)
        n_victims = 1 if len(cycle) <= 3 else (len(cycle) + 1) // 2
        # Apply pushes sequentially - each push updates avoid set for next.
        for victim_aid in sorted_victims[:n_victims]:
            self._push_cycle_victim(victim_aid, cycle, sim_time, ports_set)
        return

    def _push_cycle_victim(self, victim_aid: int, cycle: list,
                            sim_time: float, ports_set: set):
        """Cycle 내 한 명의 victim 을 push (BFS escape + continuation stitching)."""
        env = self.agv_env
        victim = env.agents.get(victim_aid)
        if victim is None or not victim.raw_path:
            return
        # Re-push 시 *같은 destination 으로* 반복 금지 (= 무한 loop 방지).
        # 다른 destination 으로는 re-push 가능 (= cycle 해소 시도).
        # 최근 (마지막 30s 내) victim 이 보내진 destination 제외.
        if not hasattr(self, '_cycle_push_history'):
            self._cycle_push_history = {}   # aid -> list[(t, dest)]
        # 만료된 history 정리
        for aid in list(self._cycle_push_history.keys()):
            self._cycle_push_history[aid] = [
                (t, d) for (t, d) in self._cycle_push_history[aid]
                if sim_time - t < 30.0]
            if not self._cycle_push_history[aid]:
                del self._cycle_push_history[aid]
        recent_dests = {d for (t, d) in
                         self._cycle_push_history.get(victim_aid, [])}

        # 2. Avoid set:
        #    - 모든 *다른 AGV* 의 현재 위치 (= 거기 가면 거기 AGV 와 충돌)
        #    - Cycle 멤버의 다음 wanting (= 거기 가면 cycle 반복)
        avoid = set()
        for a in env.agents.values():
            if a.id == victim_aid:
                continue
            if not a.raw_path:
                continue
            if a.path_idx < len(a.raw_path):
                avoid.add(a.raw_path[a.path_idx][0].split(',')[1])
        # Cycle 멤버의 wanting
        for aid in cycle:
            if aid == victim_aid:
                continue
            a = env.agents.get(aid)
            if a is None or not a.raw_path:
                continue
            if a.claim_idx < len(a.raw_path):
                next_sid = a.raw_path[a.claim_idx][0]
                if ',' in next_sid:
                    parts = next_sid.split(',')
                    if parts[0] == 'M' and len(parts) >= 3:
                        avoid.add(parts[2])
                    else:
                        avoid.add(parts[1])

        # 3. BFS 로 victim 의 cur 에서 successor expand. 첫 *avoid 와 안 겹치는*
        #    + *valid stop node (= cut 아님)* 까지만 가는 짧은 path 가 push.
        #    그 이후 plan 은 push 완료 후 자동 replan (commit 0137e8c).
        cur_sid = victim.raw_path[victim.path_idx][0]
        cur_node = cur_sid.split(',')[1] if ',' in cur_sid else cur_sid
        from collections import deque
        cut_set = getattr(env, '_cut_nodes', set())
        # BFS expand. Escape = grey corridor (out-deg=1, non-cut, recent X).
        queue = deque([cur_node])
        prev = {cur_node: None}
        escape = None
        while queue:
            n = queue.popleft()
            if (n != cur_node and n not in avoid and n not in cut_set
                    and n not in recent_dests
                    and len(self.amr_graph.adj.get(n, [])) == 1):
                escape = n
                break
            for s in self.amr_graph.adj.get(n, []):
                if s not in prev:
                    prev[s] = n
                    queue.append(s)
        if escape is None:
            print(f'  [CYCLE-FAIL] V{victim_aid-100} no escape node from {cur_node} '
                  f'(cycle={[f"V{a-100}" for a in cycle]}, avoid={avoid})')
            return
        # SIPP plan cur -> escape (자유, c_table 없음)
        push_path = self.agv_planner._base._sipp_search(
            cur_node, escape, c_table={},
            start_time=victim.raw_path[victim.path_idx][1],
            timeout=2.0, start_sid=cur_sid)
        push_dest = escape
        if not push_path or len(push_path) < 2:
            print(f'  [CYCLE-FAIL] V{victim_aid-100} no SIPP path to escape={escape}')
            return

        # 4.5. Continuation plan: escape -> 원 MCS goal (또는 victim 의 push 이전
        # 원 destination). Push 도착 후 별도 replan 없이 *즉시 이어붙임*.
        b = self.mcs.bindings.get(victim_aid)
        cont_goal = None
        if b and b.load:
            if b.phase in (VehicleJobState.TO_PICKUP, VehicleJobState.LOADING):
                cont_goal = b.load.src_port
            elif b.phase in (VehicleJobState.TO_DELIVERY,
                              VehicleJobState.UNLOADING):
                cont_goal = b.load.dst_port
        cont_path = []
        if cont_goal and cont_goal != escape:
            # Escape state 의 sid 와 시간 - push_path 끝
            esc_sid, esc_t = push_path[-1]
            cont_raw = self.agv_planner._base._sipp_search(
                escape, cont_goal, c_table={},
                start_time=esc_t,
                timeout=3.0, start_sid=esc_sid)
            if cont_raw and len(cont_raw) >= 2:
                # L state + post-S 추가 (= LOAD/UNLOAD dwell)
                arr_sid, arr_t = cont_raw[-1]
                arr_node = self.agv_planner._base._node_of_state(arr_sid)
                if arr_node:
                    load_sid = f'L,{arr_node}'
                    if load_sid in self.agv_planner.graph.load_states_raw:
                        cont_raw.append((load_sid, arr_t))
                        cont_raw.append(
                            (arr_sid, arr_t + self.agv_planner.dwell_time))
                # 첫 state 는 push_path 끝과 중복 -> skip
                cont_path = cont_raw[1:]

        # 5. raw_path 교체: [..past..] + push_path + cont_path
        for k in range(victim.path_idx, len(victim.raw_path)):
            sid_k, t_k = victim.raw_path[k]
            nk_k = env._nk(sid_k, victim_aid, t_k)
            if env.G.has_node(nk_k):
                env.G.remove_node(nk_k)
        victim.raw_path = (victim.raw_path[:victim.path_idx]
                            + push_path + cont_path)
        victim.claim_idx = victim.path_idx
        victim._wanting = None
        # 새 노드 G 등록
        for k in range(victim.path_idx, len(victim.raw_path)):
            sid_k, t_k = victim.raw_path[k]
            nk_k = env._nk(sid_k, victim_aid, t_k)
            dur = (float('inf') if k == len(victim.raw_path) - 1
                   else victim.raw_path[k+1][1] - t_k)
            env.G.add_node(nk_k, agv_id=victim_aid,
                            start_time=t_k, duration=dur)
            if k > victim.path_idx:
                prev_sid, prev_t = victim.raw_path[k-1]
                prev_nk = env._nk(prev_sid, victim_aid, prev_t)
                env._add_edge(prev_nk, nk_k)
        # State reset. Continuation stitched 시 pushed 등록 불요 (= path 끝이
        # MCS goal). 미stitched 시 pushed 등록 (= 도착 후 별도 replan).
        from env_tapg import DONE as _DONE, WAITING as _WAITING, IDLE as _IDLE
        if victim.state in (_DONE, _WAITING):
            victim.state = _IDLE
            victim.v = 0.0
        # Cycle victim 의 다음 claim 우선순위 최상위 — FIFO 정렬에서 -1 이
        # 가장 앞. MOVING/ROTATING 진입 시 0 으로 자동 reset.
        victim._priority = -1
        victim._wait_start_t = None
        if not cont_path:
            # Continuation 실패 - pushed 등록해서 도착 후 replan 시도
            self._agv_pushed.add(victim_aid)
            self._agv_goals[victim_aid] = push_dest
        else:
            # Continuation OK - MCS goal 로 직접. pushed 등록 불요.
            self._agv_goals[victim_aid] = cont_goal
        self._agv_done_notified.discard(victim_aid)
        env._schedule(sim_time, 'TRY_ADVANCE', victim_aid)
        cycle_str = ','.join(f'V{a-100}' for a in cycle)
        cont_info = f', cont_len={len(cont_path)}' if cont_path else ' (no cont)'
        print(f'  [CYCLE-PUSH] V{victim_aid-100} -> {push_dest} -> {cont_goal} '
              f'(cycle=[{cycle_str}], push_len={len(push_path)}{cont_info})')
        # 시각화용: victim 의 push escape 위치 기록
        self._cycle_push_dest[victim_aid] = push_dest
        # Re-push 방지 history
        self._cycle_push_history.setdefault(victim_aid, []).append(
            (sim_time, push_dest))
        # Movement log - phase attribution (= retrieve/deliver 종속)
        self._log_movement(victim_aid, 'push', cur_node, push_dest, sim_time)

    def _extend_with_exit_coarse(self, done_ids: set, sim_time: float):
        """Coarse mode 의 reactive push.

        Trigger 흐름:
          1. AGV Y 가 port 로 진입 시도 (= cut admission). Claim 실패 (port 점유).
          2. Y 는 WAITING. Y 의 첫 blocker 가 *port 점유 X* (= path 끝이 S->L->S).
          3. X 를 nearest siding 으로 push (= raw_path 끝에 siding 까지 path 이어붙임).
          4. X 가 dwell 끝나고 claim 으로 자연스럽게 빠져나감.

        X 의 state 는 IDLE (post-L) 또는 WAITING (L dwell 중) 둘 다 가능.
        다른 case 의 WAITING (= 단순 claim 실패) 는 push 대상 아님.
        """
        import re
        env = self.agv_env
        targets = {}   # blocker_aid -> reason
        ports_set = (set(self.amr_graph.ports.values())
                     if self.amr_graph.ports else set())
        rest_set = getattr(env, '_rest_places', set())

        # Push 는 '다른 AGV 가 나 때문에 claim 실패' 인 경우에만 발동.
        # Dependency graph 도 동시 구축해서 cycle 감지에 사용.
        dep_map = {}   # waiter_aid -> blocker_aid
        block_node_of = {}   # waiter_aid -> blocking node
        for waiter in self.agv_agents:
            # Waiter = *실제 atomic claim 시도 + 실패* 한 AGV.
            # _try_claim_next 를 dry-run 해서 판단. MOVING/ROTATING 은
            # 이미 claim 한 segment 진행 중 -> skip.
            if waiter.state in (AGV_MOVING, AGV_ROTATING):
                continue
            if waiter.path_idx >= len(waiter.raw_path):
                continue
            saved_claim = waiter.claim_idx
            env._last_block_info = ''
            ok = env._try_claim_next(waiter)
            waiter.claim_idx = saved_claim   # dry-run -> 복원
            if ok:
                continue   # claim 가능 -> 막힘 없음
            info = env._last_block_info or ''
            m = re.search(r'by (?:DONE )?V(\d+)@', info)
            if not m:
                continue
            blocker_aid = int(m.group(1)) + 100
            blocker = env.agents.get(blocker_aid)
            if blocker is None or blocker.id == waiter.id:
                continue
            # Dep map *모든 continue 이전* 에 기록 (cycle 감지용).
            # 막힌 node 추출
            m2 = re.search(r'blocked by (?:DONE )?V\d+@([^\s\[]+)', info)
            if m2:
                block_sid = m2.group(1)
                block_node = (block_sid.split(',')[1] if ',' in block_sid
                              else block_sid)
                dep_map[waiter.id] = blocker.id
                block_node_of[waiter.id] = block_node
            else:
                continue
            # 이후는 push target 자격 검사 (= dep_map 와 분리).
            if blocker.id in targets:
                continue
            # Push 조건: blocker 가 정지 + 그 자리 영구 점유 예정
            #   (blocker.raw_path[-1] 의 node == 막힌 node)
            if blocker.state in (AGV_MOVING, AGV_ROTATING):
                continue
            if not blocker.raw_path:
                continue
            blocker_last_node = blocker.raw_path[-1][0].split(',')[1]
            if blocker_last_node != block_node:
                continue   # blocker 가 더 진행할 path 있음 (= 곧 비울 예정)
            targets[blocker_aid] = (f'blocking V{waiter.id-100}@'
                                     f'{block_node} (last={blocker_last_node})')

        # Cycle detection: dep_map 의 SCC (= 2+ AGV 가 서로 차단) 찾기.
        cycles = self._find_dep_cycles(dep_map)
        if cycles:
            for cycle in cycles:
                self._resolve_cycle_replan(cycle, dep_map, block_node_of,
                                            sim_time)

        if not targets:
            return

        # 2. Push 대상에 path extension 추가
        used = set()
        for aid, g in self._agv_goals.items():
            if g:
                used.add(g)
        for a in self.agv_agents:
            if a.raw_path:
                used.add(a.raw_path[-1][0].split(',')[1])

        ext_info = []
        for tid, reason in targets.items():
            t_agent = env.agents.get(tid)
            if t_agent is None or not t_agent.raw_path:
                continue
            last_sid, last_t = t_agent.raw_path[-1]
            arrival_node = last_sid.split(',')[1]
            b = self.mcs.bindings.get(tid)
            anchor = (b.load.dst_port
                      if (b and b.load and b.load.dst_port) else None)
            cands = getattr(self, '_coarse_push_candidates', None)
            if anchor is not None:
                # Loaded AGV: anchor (dst_port) 방향 forward dist 작은 후보
                exit_port = self._pick_nearest_free_park(arrival_node, used,
                                                          candidates=cands,
                                                          anchor=anchor)
            else:
                # Idle / no task: forward BFS from blocker - 진짜 가까운 후보.
                # _pick_nearest_free_park 의 anchor=from_node 는 reverse 거리
                # 사용 -> 단방향에서 부정확. BFS 가 정확.
                exit_port = self._bfs_nearest_candidate(
                    arrival_node, cands or [], used)
            if exit_port is None or exit_port == arrival_node:
                continue

            # Plan exit path
            exit_constraints = []
            result = self.agv_planner.plan(
                {tid: arrival_node}, {tid: exit_port},
                existing_constraints=exit_constraints,
                start_times={tid: last_t})
            if not (result and result.paths.get(tid)):
                continue
            exit_path = result.paths[tid]
            if len(exit_path) < 2:
                continue
            ext = exit_path[1:]
            new_lo = len(t_agent.raw_path)
            if new_lo > 0:
                prev_sid, prev_t = t_agent.raw_path[new_lo - 1]
                prev_nk = env._nk(prev_sid, tid, prev_t)
                if env.G.has_node(prev_nk):
                    env.G.nodes[prev_nk]['duration'] = ext[0][1] - prev_t
            t_agent.raw_path.extend(ext)
            new_hi = len(t_agent.raw_path)
            used.add(exit_port)
            # TAPG node + sequential edge
            for k in range(new_lo, new_hi):
                sid, t = t_agent.raw_path[k]
                nk = env._nk(sid, tid, t)
                duration = (float('inf') if k == new_hi - 1
                            else t_agent.raw_path[k + 1][1] - t)
                env.G.add_node(nk, agv_id=tid, start_time=t, duration=duration)
                if k > 0:
                    prev_sid, prev_t = t_agent.raw_path[k - 1]
                    prev_nk = env._nk(prev_sid, tid, prev_t)
                    env._add_edge(prev_nk, nk)
            ext_info.append((tid, new_lo, new_hi))
            self._agv_pushed.add(tid)
            self._agv_goals[tid] = exit_port
            # DONE -> IDLE 전환 (= append_agents_batch 와 동일). 안 그러면
            # _on_try_advance 가 state==DONE 보고 즉시 return -> 안 움직임.
            from env_tapg import DONE as _DONE, IDLE as _IDLE
            if t_agent.state == _DONE:
                t_agent.state = _IDLE
                t_agent.v = 0.0
                if t_agent.path_idx >= new_lo:
                    t_agent.path_idx = new_lo - 1
                env._schedule(sim_time, 'TRY_ADVANCE', tid)
            else:
                env._schedule(sim_time, 'TRY_ADVANCE', tid)
            self._log_movement(tid, 'push', arrival_node, exit_port, sim_time)
            print(f'  [REACTIVE-PUSH] V{tid-100} {arrival_node} -> {exit_port} '
                  f'({reason})')

    def _extend_with_exit(self, done_ids: set, mcs_goals: set,
                           sim_time: float):
        """ACTIVE AGV가 MCS goal과 겹치는 목적지로 향할 때,
        기존 경로 끝에 탈출 경로를 이어붙여서 inf 점유를 해소한다.

        Coarse mode: reactive push 만. WAITING 인 AGV 의 blocker 가 IDLE 일
        때만 그 blocker 를 push. 기존 SIPP preemptive push 비활성.

        대상 (SIPP):
          - ACTIVE pushed AGV → 빈 포트로 탈출
          - ACTIVE MCS 작업 AGV →
              TO_PICKUP: load.dst_port로 탈출 (다음 배달지 확정)
              TO_DELIVERY: 빈 포트로 탈출 (하역 후 idle)
        """
        # Coarse mode: reactive push 는 update() 의 1초 주기 호출에서 처리됨.
        # Dispatch 시점의 중복 호출 제거 - 매 호출 50ms 까지 spike 발생.
        if self._planner_type == 'coarse':
            return
        # park 후보: ports + Tier-A sidings (거리순 정렬은 _pick_nearest_free_park 가)
        park_pool = self._park_nodes

        # 현재 사용 중인 위치 수집 (다른 AGV가 향하거나 점유한 노드 제외)
        used = set(mcs_goals)
        for aid, g in self._agv_goals.items():
            if g:
                used.add(g)
        for a in self.agv_agents:
            if a.raw_path:
                used.add(a.raw_path[-1][0].split(',')[1])

        # 이번 호출에서 실제로 확장된 agent 기록 (aid, new_start_idx, new_end_idx)
        ext_info = []

        for a in self.agv_agents:
            if a.id in done_ids:
                continue
            if a.path_idx >= len(a.raw_path):
                continue  # DONE — blocker 감지에서 처리됨

            # 기존 경로 마지막 시간/위치 — 확장 여부 판단은 이 실제 종착지 기준
            # (self._agv_goals 는 이전 확장 반영 안됨 → 낡은 값일 수 있음)
            last_sid, last_t = a.raw_path[-1]
            arrival_node = last_sid.split(',')[1]

            # 이 AGV가 실제로 inf 점유할 port 가 mcs_goals 와 겹치는지 확인
            if arrival_node not in mcs_goals:
                continue

            # 탈출 목적지 결정
            exit_port = None
            b = self.mcs.bindings.get(a.id)
            # is_deliver_preplan: TO_PICKUP agent 의 path 를 자신의 dst_port 까지
            # 연장하는 경우. push 가 아닌 실제 delivery 의 pre-plan 이므로 끝에
            # L,dst + final_S 를 같이 추가해 UNLOAD dwell 을 명시한다.
            is_deliver_preplan = False

            # AGV 가 load 보유 중이면 dst_port 근방 anchor (다음 UNLOAD 후 backtrack 최소화)
            anchor_for_park = (b.load.dst_port
                               if (b and b.load and b.load.dst_port) else None)

            if a.id in self._agv_pushed:
                # Pushed AGV → 가장 가까운 free park
                reason = 'PUSHED'
                exit_port = self._pick_nearest_free_park(arrival_node, used,
                                                          anchor=anchor_for_park)
            elif b and b.load:
                if b.phase == VehicleJobState.TO_PICKUP:
                    # TO_PICKUP → dst_port 가 비어있으면 거기로(배달 replan 절약),
                    # 이미 다른 agent가 점유 중이면 가장 가까운 free park 로 fallback.
                    dst = b.load.dst_port
                    if dst and dst != arrival_node and dst not in used:
                        exit_port = dst
                        reason = f'TO_PICKUP→dst={exit_port}'
                        is_deliver_preplan = True
                    else:
                        reason = 'TO_PICKUP→dst-anchor free park (dst busy)'
                        exit_port = self._pick_nearest_free_park(arrival_node, used,
                                                                  anchor=anchor_for_park)
                elif b.phase in (VehicleJobState.TO_DELIVERY,
                                 VehicleJobState.UNLOADING):
                    # TO_DELIVERY/UNLOADING — dst 근방 free park 로 exit.
                    reason = f'{b.phase.value}→idle near dst'
                    exit_port = self._pick_nearest_free_park(arrival_node, used,
                                                              anchor=anchor_for_park)
                elif b.phase == VehicleJobState.LOADING:
                    if arrival_node == b.load.dst_port:
                        reason = 'LOADING(tail=dst)→idle'
                        exit_port = self._pick_nearest_free_park(arrival_node, used,
                                                                  anchor=anchor_for_park)
                    else:
                        continue  # tail=src, no extend
                else:
                    continue
            else:
                # IDLE agent executing residual raw_path extension —
                # 이전에 확장됐는데 그 사이 MCS 작업(배달)이 끝나서 load 가
                # 비워진 경우. PUSHED 와 동일하게 가까운 free park 로 재확장.
                reason = 'IDLE(residual)→nearest free park'
                exit_port = self._pick_nearest_free_park(arrival_node, used)

            if exit_port is None:
                print(f'  [EXIT] A{a.id}: no exit port from {arrival_node} ({reason})')
                continue
            if exit_port == arrival_node:
                continue  # 같은 곳으로 탈출은 무의미

            # 탈출 경로 SIPP 계획 — 활성 AGV들의 full 경로를 constraint로 사용.
            # done_ids 의 raw_path 는 곧 replan 될 예정이지만 그 사이 시간엔
            # 여전히 물리적 미래 점유를 나타냄 (extension 으로 추가된 미실행
            # 구간 포함). 이걸 constraint 에서 빼면 다른 agent extension 이
            # 그 위로 collision 을 만들 수 있음 → 자기 자신만 exclude.
            exit_constraints = self._build_active_agent_constraints(
                excluded_aids={a.id},
                sim_time=sim_time,
            )
            for other in self.agv_agents:
                if other.id == a.id or other.id in done_ids:
                    continue
                if other.path_idx >= len(other.raw_path):
                    o_sid = other.raw_path[-1][0]
                    exit_constraints.append({
                        'agent': other.id, 'loc': o_sid,
                        'timestep': (last_t, float('inf')),
                    })

            result = self.agv_planner.plan(
                {a.id: arrival_node}, {a.id: exit_port},
                existing_constraints=exit_constraints,
                start_times={a.id: last_t},
            )

            if result and result.paths.get(a.id):
                exit_path = result.paths[a.id]
                if len(exit_path) >= 2:
                    ext = exit_path[1:]  # 첫 step(출발지 중복) 제거
                    new_lo = len(a.raw_path)
                    # 직전 tail (이때까지의 inf-claim 종점) 의 duration 을
                    # finite 로 업데이트 — append_agents_batch 와 동일한
                    # 패턴. 안 그러면 G 노드에 stale inf duration 이 남아
                    # 디버깅 시 직관에 어긋남.
                    if new_lo > 0:
                        prev_sid, prev_t = a.raw_path[new_lo - 1]
                        prev_nk = self.agv_env._nk(prev_sid, a.id, prev_t)
                        if self.agv_env.G.has_node(prev_nk):
                            self.agv_env.G.nodes[prev_nk]['duration'] = \
                                ext[0][1] - prev_t
                    a.raw_path.extend(ext)
                    new_hi = len(a.raw_path)
                    used.add(exit_port)

                    # deliver pre-plan: TO_PICKUP agent 를 자기 dst 로 연장 시
                    # raw_path 끝에 L,dst + final_S 추가. 이렇게 해야 추후
                    # LOADING dispatch 가 또 같은 dst 로 SIPP plan 을 돌려
                    # detour 가 생기는 것을 막을 수 있고, 다른 agent SIPP 도
                    # 이 AGV 의 UNLOAD 3s 점유를 정확히 본다.
                    if is_deliver_preplan:
                        load_sid = f'L,{exit_port}'
                        if load_sid in self.agv_planner.graph.load_states_raw:
                            arr_sid, arr_t = a.raw_path[-1]
                            dwell = self.mcs.dwell_time
                            a.raw_path.append((load_sid, arr_t))
                            a.raw_path.append((arr_sid, arr_t + dwell))
                            new_hi = len(a.raw_path)
                            # arrival_idx 는 변경 안 함 — 지금은 TO_PICKUP phase 이고
                            # 현재 MCS goal (src port) 의 arrival_idx 가 유지돼야
                            # PICKUP arrival event 가 정상 fire 됨. 새 dst 의 L state
                            # 는 미래 LOADING dispatch 의 INLINE-DELIVERY 가 처리.

                    # TAPG DAG에 새 step 등록 (같은 agent sequential edge만)
                    for k in range(new_lo, new_hi):
                        sid, t = a.raw_path[k]
                        nk = self.agv_env._nk(sid, a.id, t)
                        duration = (float('inf') if k == new_hi - 1
                                    else a.raw_path[k + 1][1] - t)
                        self.agv_env.G.add_node(nk, agv_id=a.id, start_time=t,
                                                duration=duration)
                        if k > 0:
                            prev_sid, prev_t = a.raw_path[k - 1]
                            prev_nk = self.agv_env._nk(prev_sid, a.id, prev_t)
                            self.agv_env._add_edge(prev_nk, nk)

                    ext_info.append((a.id, new_lo, new_hi))

                    # is_deliver_preplan: push 가 아닌 본인 delivery 의 pre-plan
                    # → push 마커 추가 안 함 (color/visual 은 phase 우선이라 변화
                    # 없음, 다만 _agv_pushed 셋이 깨끗하게 유지됨).
                    if not is_deliver_preplan:
                        self._agv_pushed.add(a.id)
                    self._agv_goals[a.id] = exit_port
                    self._log_movement(a.id,
                                       'deliver' if is_deliver_preplan else 'push',
                                       arrival_node, exit_port, sim_time)
                else:
                    print(f'  [EXIT] A{a.id}: exit path too short ({reason})')
            else:
                print(f'  [EXIT] A{a.id}: exit FAILED {arrival_node} → {exit_port} ({reason})')

        # 모든 확장 완료 후 cross-agent TAPG edge 보강
        # (extend_agents_batch Phase 2 와 동일 로직을 확장 구간에 적용)
        if ext_info:
            self.agv_env.add_cross_edges_for_extensions(ext_info)

    def _prune_completed_states(self):
        """각 AGV 의 raw_path 에서 이미 지나간 state (0..path_idx-1) 을 잘라내고
        path_idx / claim_idx / _agv_arrival_idx 를 동일 오프셋만큼 시프트한다.

        DAG 노드는 `_complete_node` 에서 이미 제거되고, constraint 는
        `raw_path[claim_idx:]` 만 쓰므로 "과거 구간"은 Python list 에만 남아있는
        dead weight. Replan 시점에 잘라서 메모리/로그 가독성 확보.

        주의: agent.x/y, a._tapg_node 는 node key 기반이라 영향 없음.
        """
        total_removed = 0
        G = self.agv_env.G
        wait_queues = self.agv_env.wait_queues
        for a in self.agv_agents:
            if not a.raw_path:
                continue
            # 최소 한 state 는 남겨 "현재/종착 위치" 유지.
            # - 진행 중 (path_idx < len): shift=path_idx, 현재 state 가 idx=0 으로.
            # - DONE (path_idx == len): shift=len-1, 마지막 state 만 남고 path_idx=1.
            shift = min(a.path_idx, len(a.raw_path) - 1)
            if shift <= 0:
                continue

            # 0) G + wait_queues 정리 — 잘라낼 state 들의 TAPG node 제거.
            # _handle_arrival 의 cleanup loop 가 prune 후엔 range(0) 으로 비어버려
            # 이 노드들을 영원히 못 지움. 여기서 미리 정리.
            for i in range(shift):
                sid_i, t_i = a.raw_path[i]
                nk_i = self.agv_env._nk(sid_i, a.id, t_i)
                if G.has_node(nk_i):
                    # 대기자 깨우기
                    from env_tapg import WAITING, IDLE
                    for wid in wait_queues.pop(nk_i, []):
                        wa = self.agv_env.agents.get(wid)
                        if wa and wa.state == WAITING:
                            wa.state = IDLE
                            self.agv_env._schedule(self.sim_time + 1e-9,
                                                    'TRY_ADVANCE', wid)
                    G.remove_node(nk_i)

            # 1) raw_path / 인덱스 시프트
            a.raw_path = a.raw_path[shift:]
            a.path_idx -= shift
            a.claim_idx = max(0, a.claim_idx - shift)

            old_arr = self._agv_arrival_idx.get(a.id)
            if old_arr is not None:
                new_arr = old_arr - shift
                if new_arr < 0:
                    # 이미 arrival 이 fired 됐음 → 추적 불필요
                    self._agv_arrival_idx.pop(a.id)
                else:
                    self._agv_arrival_idx[a.id] = new_arr
            total_removed += shift
        if total_removed > 0:
            print(f'  [PRUNE] removed {total_removed} completed states '
                  f'across {len(self.agv_agents)} agents')

    def _dump_replan_state(self, sim_time: float, done_ids: set):
        """Replan FAIL 시 호출되는 verbose 상태 덤프."""
        print(f'  --- REPLAN FAIL DETAIL ---')
        n_state = {}
        for a in self.agv_agents:
            n_state[a.state] = n_state.get(a.state, 0) + 1
        n_wait_q = sum(len(q) for q in self.agv_env.wait_queues.values())
        print(f'  TAPG: ' + ' '.join(f'{k}={v}' for k, v in sorted(n_state.items())) +
              f' wait_nodes={len(self.agv_env.wait_queues)} waiters={n_wait_q}')
        for a in self.agv_agents:
            nid = a.raw_path[-1][0].split(',')[1] if a.raw_path else '?'
            goal = self._agv_goals.get(a.id)
            b = self.mcs.bindings.get(a.id)
            phase = b.phase.value if b else '?'
            load_id = b.load.load_id if (b and b.load) else None
            pushed = ' P' if a.id in self._agv_pushed else ''
            moving = 'A' if a.path_idx < len(a.raw_path) else 'D'
            waits = ''
            if a.state == AGV_WAITING and a.path_idx < len(a.raw_path):
                cur_sid, cur_t = a.raw_path[a.path_idx]
                nk = self.agv_env._nk(cur_sid, a.id, cur_t)
                if self.agv_env.G.has_node(nk):
                    cross = [f'A{p[1]}' for p in self.agv_env.G.predecessors(nk)
                             if p[1] != a.id]
                    if cross:
                        waits = f' waits={cross[:5]}'
            print(f'    A{a.id} pos={nid} goal={goal} ph={phase} load={load_id} '
                  f's={a.state} i={a.path_idx}/{len(a.raw_path)} '
                  f'c={a.claim_idx} {moving}{pushed}{waits}')

    def _replan_done_agvs(self, sim_time: float):
        """Incrementally replan finished AGVs. Active agents' TAPG is untouched."""
        import time as _pt
        _prof_on = self._profile_frames_ms > 0.0
        self._replan_call_count += 1 if _prof_on else 0
        done_ids = set(self._agv_pending_replan)

        # 이미 지나간 raw_path 구간 제거 — paths 가 unbounded 로 자라면
        # constraint extraction / cross-edge scan 이 path_length 에 비례해
        # 비싸지므로 매 replan 시점에 잘라준다. G + wait_queues 도 같이 정리해
        # 대기 chain orphan 발생 방지.
        _t0 = _pt.perf_counter() if _prof_on else 0.0
        self._prune_completed_states()
        if _prof_on: self._sub_prune_s += _pt.perf_counter() - _t0

        # 0) TAPG 시간 보정: SIPP planner 의 earliest-time constraint 갱신용.
        #    Coarse planner 는 c_table={} 라 안 읽음 → coarse mode 에선 skip.
        _t0 = _pt.perf_counter() if _prof_on else 0.0
        if self._planner_type != 'coarse':
            self.agv_env.recompute_earliest_schedule(current_time=sim_time)
        if _prof_on: self._sub_recompute_s += _pt.perf_counter() - _t0

        # 0.5) ACTIVE AGV 가 *re-dispatch 되는 agent 의 goal 또는 start* 로
        #      향하고 있으면 (= path end == 그 위치), 기존 경로 끝에 탈출
        #      경로를 이어붙여서 inf 점유를 해소.
        #      - goal 매칭: 후속 agent 가 그 위치 도착해야 함 → 선행 agent 비워주기
        #      - start 매칭: 후속 agent 가 그 위치에서 떠나는데, 선행 agent 가
        #        오면 충돌. 선행 agent path 를 extension 으로 끝낸 후 떠나게
        mcs_goals_pre = {self._agv_goals.get(aid) for aid in done_ids
                         if self._agv_goals.get(aid)}
        # done_ids agent 의 *현재 위치* (= start). raw_path[path_idx]
        mcs_starts = set()
        for aid in done_ids:
            a = self.agv_env.agents.get(aid)
            if a and a.raw_path:
                idx = min(a.path_idx, len(a.raw_path) - 1)
                cur_sid = a.raw_path[idx][0]
                mcs_starts.add(cur_sid.split(',')[1])
        relevant_nodes = mcs_goals_pre | mcs_starts
        _t0 = _pt.perf_counter() if _prof_on else 0.0
        if relevant_nodes:
            self._extend_with_exit(done_ids, relevant_nodes, sim_time)
        if _prof_on: self._sub_extend_s += _pt.perf_counter() - _t0

        # 0.7) Inline-skip: done_ids agent 의 raw_path[path_idx:] 에 이미
        #      L,mcs_goal 이 있으면 (TO_PICKUP→dst preplan 으로 _extend_with_exit
        #      가 사전에 추가한 경우) SIPP 와 끝-L append 를 건너뛴다. 안 그러면
        #      raw_path 가 dst 를 transit 으로 지나가는데 새 SIPP 가 끝에 또
        #      dst 를 붙여 detour 발생.
        agents_handled_inline = set()
        for aid in list(done_ids):
            if aid in self._agv_pushed:
                continue
            mcs_goal = self._agv_goals.get(aid)
            if mcs_goal is None:
                continue
            a = self.agv_env.agents.get(aid)
            if a is None or not a.raw_path:
                continue
            load_sid = f'L,{mcs_goal}'
            found_idx = -1
            for i in range(a.path_idx, len(a.raw_path)):
                if a.raw_path[i][0] == load_sid:
                    found_idx = i
                    break
            if found_idx >= 0:
                agents_handled_inline.add(aid)
                # arrival_S (L 직전) 의 idx 로 arrival_idx 설정
                self._agv_arrival_idx[aid] = found_idx - 1
                self._agv_arrived_notified.discard(aid)
        done_ids -= agents_handled_inline

        _t0 = _pt.perf_counter() if _prof_on else 0.0
        # 1) Build constraints from ALL active (non-done) agents' remaining paths.
        #    SIPP planner 의 c_table 에 들어가는 값.
        #    Coarse planner 는 plan() 에서 existing_constraints 를 받지만
        #    내부적으로 c_table={} 로 무시 (= shortest path). 따라서 coarse
        #    mode 에선 build 자체 skip. Runtime 충돌은 SegmentLockManager 가 처리.
        all_constraints = []
        _coarse_skip_constr = (self._planner_type == 'coarse')
        for a in self.agv_agents:
            if _coarse_skip_constr:
                break
            if a.id in done_ids:
                continue
            idx = a.path_idx
            if idx >= len(a.raw_path):
                # DONE/idle — 마지막 위치를 현재 위치로 취급
                idx = len(a.raw_path) - 1

            # 현재 위치 물리적 점유: 에이전트가 지금 있는 위치를 sim_time부터
            # 무조건 block (claimed/unclaimed 시간 갭 방지)
            cur_sid, cur_t = a.raw_path[idx]
            cur_state = self.agv_planner._get_state(cur_sid)
            if cur_state is not None:
                # 현재 상태가 M/R이면: 출발+도착 노드 모두 block
                # 현재 상태가 S이면: 해당 노드 block
                # t_end: 다음 상태 시작 or inf
                if idx + 1 < len(a.raw_path):
                    t_end_cur = max(a.raw_path[idx + 1][1], sim_time)
                else:
                    t_end_cur = float('inf')
                all_constraints.append({
                    'agent': a.id, 'loc': cur_sid,
                    'timestep': (sim_time, t_end_cur),
                })
                for aff_id in cur_state.affect_state:
                    aff = self.agv_planner._get_state(aff_id)
                    aff_cost = aff.cost if aff else 0.0
                    all_constraints.append({
                        'agent': a.id, 'loc': aff_id,
                        'timestep': (max(0.0, sim_time - aff_cost), t_end_cur),
                    })

            # Claimed 구간 (path_idx ~ claim_idx):
            # 첫 action 시작 시간 ~ 각 action 종료 시간으로 block
            claimed_path = a.raw_path[idx:a.claim_idx]
            if claimed_path:
                t_claim_start = sim_time  # 현재 시간부터 점유

                for ci, (sid, t) in enumerate(claimed_path):
                    state = self.agv_planner._get_state(sid)
                    if state is None:
                        continue
                    # 이 action의 종료 시간
                    if ci + 1 < len(claimed_path):
                        t_end = claimed_path[ci + 1][1]
                    else:
                        # 마지막 claimed step → unclaimed 구간 시작까지 연장
                        # (에이전트가 해당 노드에서 대기 중이므로 점유 지속)
                        unclaimed_rest = a.raw_path[a.claim_idx:]
                        if unclaimed_rest:
                            t_end = unclaimed_rest[0][1]
                        else:
                            # raw_path[-1] = idle inf-claim. 다음 dispatch 까지
                            # 영구 점유이므로 t_end = inf.
                            t_end = float('inf')

                    all_constraints.append({
                        'agent': a.id, 'loc': sid,
                        'timestep': (t_claim_start, t_end),
                    })
                    for aff_id in state.affect_state:
                        aff = self.agv_planner._get_state(aff_id)
                        aff_cost = aff.cost if aff else 0.0
                        all_constraints.append({
                            'agent': a.id, 'loc': aff_id,
                            'timestep': (max(0.0, t_claim_start - aff_cost), t_end),
                        })

            # Unclaimed 구간 (claim_idx ~): 시간 기반 constraint
            unclaimed_path = a.raw_path[a.claim_idx:]
            if unclaimed_path:
                cs = self.agv_planner._build_constraints(unclaimed_path, a.id)
                all_constraints.extend(cs)

        if _prof_on: self._sub_constraint_s += _pt.perf_counter() - _t0
        _t0 = _pt.perf_counter() if _prof_on else 0.0
        # 2) Current positions of done agents.
        #    Coarse mode: claim_idx 까지만 사용. 미claim 영역 (= push extension
        #    중 안 도달한 곳) 은 truncate. 새 plan 의 start = path[claim_idx-1]
        #    (= 현재 atomic claim 의 마지막 state).
        #    이유: push 도중 새 task 받으면 push destination 까지 안 가도 됨.
        #    Claim 영역만 완주 + 거기서 새 plan 시작.
        #
        #    SIPP mode: 기존 동작 (raw_path[-1] = inf-claim 종착).
        positions_to_plan = {}
        start_times_override = {}
        idle_aids = set()
        for a in self.agv_agents:
            if a.id in done_ids:
                if (self._planner_type == 'coarse'
                        and a.claim_idx > 0
                        and a.claim_idx < len(a.raw_path)):
                    # 미claim 영역 truncate + claim end 를 새 plan start 로.
                    # CRITICAL: truncate 가 mid-corridor S 에서 끝나면 AGV 가
                    # 그 자리에서 DONE 되어 (DONE-visibility 때문에) 다른 AGV
                    # 영구 차단. claim_idx 부터 *뒤로* 스캔해서 안전한 stop
                    # 지점 (= rest place OR push 후보 = segment last-grey) 에서
                    # 잘라야 안전.
                    cut_idx = a.claim_idx
                    rest_set = getattr(self.agv_env, '_rest_places', set())
                    push_set = set(getattr(self, '_coarse_push_candidates', ()) or ())
                    safe_set = rest_set | push_set
                    while cut_idx > a.path_idx + 1:
                        sid_check = a.raw_path[cut_idx - 1][0]
                        if sid_check.startswith('S,'):
                            node_check = sid_check.split(',')[1]
                            if node_check in safe_set:
                                break
                        cut_idx -= 1
                    if cut_idx <= a.path_idx + 1:
                        # 안전 지점 못 찾음 - truncate 안 함 (path 유지)
                        cut_idx = len(a.raw_path)
                    for i in range(cut_idx, len(a.raw_path)):
                        sid_i, t_i = a.raw_path[i]
                        nk_i = self.agv_env._nk(sid_i, a.id, t_i)
                        if self.agv_env.G.has_node(nk_i):
                            self.agv_env.G.remove_node(nk_i)
                    a.raw_path = a.raw_path[:cut_idx]
                    a.claim_idx = min(a.claim_idx, cut_idx)
                last_sid, last_t = a.raw_path[-1]
                positions_to_plan[a.id] = last_sid.split(',')[1]
                start_times_override[a.id] = last_t

        # 목적지에 앉아있는 idle AGV가 있으면 blocker로 포함
        mcs_goals = {self._agv_goals.get(aid) for aid in done_ids
                     if self._agv_goals.get(aid)}
        for a in self.agv_agents:
            if a.id in done_ids:
                continue
            if a.path_idx >= len(a.raw_path):
                last_sid, last_t = a.raw_path[-1]
                nid = last_sid.split(',')[1]
                if nid in mcs_goals:
                    positions_to_plan[a.id] = nid
                    start_times_override[a.id] = last_t
                    idle_aids.add(a.id)
                    done_ids.add(a.id)  # constraint 빌드에서 제외

        if _prof_on: self._sub_truncate_s += _pt.perf_counter() - _t0
        _t0 = _pt.perf_counter() if _prof_on else 0.0
        # 3) Assign new port goals (avoid other AGVs' destinations + positions)
        occupied = set()
        for a in self.agv_agents:
            if a.id not in done_ids and a.raw_path:
                occupied.add(a.raw_path[-1][0].split(',')[1])
        targeted = {g for aid, g in self._agv_goals.items() if aid not in done_ids}
        used = occupied | targeted | set(positions_to_plan.values())

        new_goals = {}
        push_failed_aids = []
        for aid, cur in positions_to_plan.items():
            # MCS가 목적지를 지정한 경우 그대로 사용 (push 임시 목적지 제외)
            mcs_goal = self._agv_goals.get(aid)
            if mcs_goal and mcs_goal != cur and aid not in self._agv_pushed:
                new_goals[aid] = mcs_goal
                used.add(mcs_goal)
                continue
            # idle/pushed AGV — 가장 가까운 free park (port + Tier-A siding)
            if aid in idle_aids or aid in self._agv_pushed:
                self._agv_goals.pop(aid, None)  # 이전 임시 목적지 클리어
                # AGV 가 load 보유 중이면 (= 다음에 UNLOAD 할 dst_port 있음)
                # dst_port 근방으로 push 해서 deliver 후 backtrack 최소화
                b = self.mcs.bindings.get(aid)
                anchor = b.load.dst_port if (b and b.load and b.load.dst_port) else None
                p = self._pick_nearest_free_park(cur, used, anchor=anchor)
                if p is not None:
                    new_goals[aid] = p
                    used.add(p)
                    self._log_movement(aid, 'push', cur, p, sim_time)
                else:
                    new_goals[aid] = cur
                    push_failed_aids.append(aid)
                continue
            # MCS 목적지 없음 → 가장 가까운 free park
            p = self._pick_nearest_free_park(cur, used)
            if p is not None:
                new_goals[aid] = p
                used.add(p)
            else:
                # 모든 park 점유 중 — 다른 AGV가 비킬 때까지 대기 의도로
                # 자리 유지 (replan 사이클에서 재시도)
                new_goals[aid] = cur

        if _prof_on: self._sub_pickpark_s += _pt.perf_counter() - _t0
        # 4) Plan only the done agents (active paths are constraints).
        #    start_time 은 각 agent의 raw_path[-1] 시각 (inf-claim 해제 시점).
        #    Phase 2/3 cycle-break push 는 모든 done_ids agent 대상 — task 든
        #    아니든. planner 의 L state insertion 이 dwell 을 path 안에 embedded
        #    형태로 보존하므로 push extension 이 추가돼도 dwell 깨지지 않음.
        import time as _time
        _plan_t0 = _time.perf_counter()
        # idle/pushed agent = non-task. Phase 2/3 push 후보 제한 — task agent 의
        # mid-dwell 깨지지 않게.
        non_task_set = (set(idle_aids) | set(self._agv_pushed)) & set(positions_to_plan.keys())
        # Port-exit commit: AGV 가 port 에서 출발할 때만 2-stage SIPP 적용
        # (= 첫 grey 까지 blocked-by-me 회피). coarse mode 전용 - SIPP planner
        # 는 이 kwargs 모름.
        plan_kwargs = dict(
            existing_constraints=all_constraints,
            start_times=start_times_override,
            non_task_agents=non_task_set,
        )
        if self._planner_type == 'coarse':
            port_exit_blocked = {}
            for aid in positions_to_plan:
                port_exit_blocked[aid] = self._compute_active_claims(aid)
            plan_kwargs['port_exit_blocked_per_agent'] = port_exit_blocked
            plan_kwargs['cut_nodes'] = getattr(self, '_cut_nodes', None)
        result = self.agv_planner.plan(
            positions_to_plan, new_goals,
            **plan_kwargs,
        )
        _plan_dt = _time.perf_counter() - _plan_t0
        _afterplan_t0 = _pt.perf_counter() if _prof_on else 0.0
        _stage_t0 = _afterplan_t0

        # Coarse mode: TO_PICKUP plan 끝에 *deliver leg* (src->dst) 자동 추가.
        # 'retrieve->deliver' 통합 path 산출 -> UNLOAD-bound AGV 는 이미 빠져
        # 나갈 plan 있음 -> push 불필요.
        if (self._planner_type == 'coarse'
                and result is not None and result.paths):
            for aid, path1 in list(result.paths.items()):
                b = self.mcs.bindings.get(aid)
                if not (b and b.load and b.phase == VehicleJobState.TO_PICKUP):
                    continue
                src = b.load.src_port
                dst = b.load.dst_port
                if not (src and dst and src != dst):
                    continue
                if not path1:
                    continue
                end_sid, end_t = path1[-1]
                end_node = end_sid.split(',')[1] if ',' in end_sid else end_sid
                if end_node != src:
                    continue
                # Leg 2: src(=port) → dst. Port-exit commit 적용 (2-stage SIPP).
                # blocked-by-me 회피 + 첫 grey 까지만 constraint, 이후는 free.
                leg2_blocked = self._compute_active_claims(aid) - {src, dst}
                leg2_raw = None
                if leg2_blocked and src in self.agv_planner._port_nodes:
                    exit_candidates = self.agv_planner._find_all_first_non_cut(
                        src, getattr(self, '_cut_nodes', set()))
                    for exit_node in exit_candidates:
                        if (not exit_node or exit_node == src
                                or exit_node == dst
                                or exit_node in leg2_blocked):
                            continue
                        c_table_nodes = leg2_blocked - {exit_node}
                        c_table = {n: [(0.0, float('inf'))]
                                   for n in c_table_nodes}
                        l2a = self.agv_planner._base._sipp_search(
                            src, exit_node, c_table=c_table,
                            start_time=end_t, timeout=5.0,
                            start_sid=end_sid)
                        if not (l2a and len(l2a) >= 2):
                            continue
                        l2a_end_sid, l2a_end_t = l2a[-1]
                        l2b = self.agv_planner._base._sipp_search(
                            exit_node, dst, c_table={},
                            start_time=l2a_end_t, timeout=10.0,
                            start_sid=l2a_end_sid)
                        if l2b and len(l2b) >= 2:
                            leg2_raw = l2a + l2b[1:]
                            break
                if not leg2_raw:
                    leg2_raw = self.agv_planner._base._sipp_search(
                        src, dst, c_table={}, start_time=end_t,
                        timeout=10.0, start_sid=end_sid)
                if not leg2_raw or len(leg2_raw) < 2:
                    continue
                # L,dst + post-dwell S 추가 (coarse_planner.plan() 동일 로직)
                arr_sid, arr_t = leg2_raw[-1]
                arr_node = self.agv_planner._base._node_of_state(arr_sid)
                if arr_node:
                    load_sid = f'L,{arr_node}'
                    if load_sid in self.agv_planner.graph.load_states_raw:
                        leg2_raw.append((load_sid, arr_t))
                        leg2_raw.append((arr_sid,
                                          arr_t + self.agv_planner.dwell_time))
                # 첫 state (= end_sid 와 동일) skip, 나머지 append
                result.paths[aid] = path1 + leg2_raw[1:]

        if _prof_on:
            _now = _pt.perf_counter()
            self._sub_leg2_s += _now - _stage_t0
            _stage_t0 = _now
        # plan 시간 추적
        n_agents = len(positions_to_plan)
        cs_total = len(all_constraints)
        plan_status = 'OK' if (result is not None and result.paths) else 'FAIL'
        plan_rec = {
            't': sim_time, 'dur': _plan_dt, 'n_agents': n_agents,
            'cs': cs_total, 'pending': sorted(done_ids),
            'goals': dict(new_goals), 'status': plan_status,
        }
        # path 길이 평균 (성공 시)
        if plan_status == 'OK':
            lengths = [len(p) for p in result.paths.values()]
            plan_rec['path_len_total'] = sum(lengths)
            plan_rec['path_len_max']   = max(lengths) if lengths else 0
        else:
            plan_rec['path_len_total'] = 0
            plan_rec['path_len_max']   = 0
        self._plan_dur_log.append(plan_rec)
        # 임계값 초과 시 즉시 경고 출력 (>=0.2s)
        if _plan_dt >= 0.20:
            print(f'  [SLOW PLAN] t={sim_time:.2f}s dur={_plan_dt:.3f}s '
                  f'agents={n_agents} cs={cs_total} '
                  f'path_total={plan_rec["path_len_total"]} '
                  f'max={plan_rec["path_len_max"]}')

        if result is None or not result.paths:
            self._agv_pending_replan.clear()
            # 진짜 FAIL (= pending 있는데 plan 못 만듦) 만 dump.
            # pending 없으면 no-op (= prune 만 한 사이클) - log 만 짧게.
            if done_ids:
                self._plan_status = 'Replan FAILED'
                print(f'[REPLAN] t={sim_time:.2f}s pending={sorted(done_ids)} '
                      f'goals={new_goals} cs={len(all_constraints)} '
                      f'dur={_plan_dt:.2f}s status=FAIL')
                self._dump_replan_state(sim_time, done_ids)
            return

        # ── DEBUG: replan 이력을 메모리에 저장 (collision 시 출력) ──
        if not hasattr(self, '_replan_history'):
            self._replan_history = {}  # agent_id → latest replan record
        for aid, new_path in result.paths.items():
            # 이 agent의 새 경로에서 지나는 노드 수집
            path_nodes = set()
            for sid, t in new_path:
                parts = sid.split(',')
                if len(parts) >= 2: path_nodes.add(parts[1])
                if len(parts) >= 3 and parts[0] == 'M': path_nodes.add(parts[2])

            # 활동 에이전트 스냅샷
            active_snapshot = []
            for a in self.agv_agents:
                if a.id in done_ids or a.path_idx >= len(a.raw_path):
                    continue
                cur_sid = a.raw_path[a.path_idx][0]
                active_snapshot.append({
                    'id': a.id, 'state': a.state, 'cur_sid': cur_sid,
                    'path_idx': a.path_idx, 'total': len(a.raw_path),
                    'claim_idx': a.claim_idx,
                    'pos': (a.x, a.y),
                })

            # constraint 중 이 경로가 지나는 노드에 해당하는 것만 필터
            relevant_constraints = []
            for c in all_constraints:
                loc = c['loc']
                parts = loc.split(',')
                c_nodes = set()
                if len(parts) >= 2: c_nodes.add(parts[1])
                if len(parts) >= 3 and parts[0] == 'M': c_nodes.add(parts[2])
                if c_nodes & path_nodes:
                    relevant_constraints.append(c)

            self._replan_history[aid] = {
                'time': sim_time,
                'done_ids': list(done_ids),
                'path': new_path,
                'path_nodes': path_nodes,
                'active_snapshot': active_snapshot,
                'constraints': relevant_constraints,
            }

        if _prof_on:
            _now = _pt.perf_counter()
            self._sub_replan_hist_s += _now - _stage_t0
            _stage_t0 = _now
        # 5) Planner 가 task goal 에 L,goal + final_S 를 이미 insert 함.
        #    Wrapper 는 path 안에서 L,mcs_goal 위치를 찾아 arrival_idx 만 설정.
        for aid, new_path in result.paths.items():
            if aid in idle_aids or aid in self._agv_pushed:
                continue  # push된 AGV는 arrival 통보 불필요
            if new_path and len(new_path) >= 2:
                # planner 가 삽입한 L,mcs_goal 위치 = arrival_S 의 다음 step
                mcs_goal = new_goals.get(aid)
                load_sid_target = f'L,{mcs_goal}' if mcs_goal else None
                arrival_idx_new = None
                if load_sid_target:
                    for i, (sid, _t) in enumerate(new_path):
                        if sid == load_sid_target:
                            arrival_idx_new = i - 1   # arrival_S 직전 step
                            break
                if arrival_idx_new is None:
                    # task goal 이 port 아닌 경우 (이론상 done_ids+non-push 분기엔
                    # 발생 안 함). fail-safe: arrival 통보 skip.
                    continue
                agent = self.agv_env.agents.get(aid)
                old_len = len(agent.raw_path) if agent and agent.raw_path else 0
                # 첫 state 중복 제거 후 append 되므로 -1
                self._agv_arrival_idx[aid] = old_len + arrival_idx_new - 1
                self._agv_arrived_notified.discard(aid)

        # 6) Update goals & extend TAPG -batch all new paths at once
        #    idle AGV를 pushed로 등록
        for aid, goal in new_goals.items():
            self._agv_goals[aid] = goal
            if aid in idle_aids:
                self._agv_pushed.add(aid)

        batch = {}
        for aid, new_path in result.paths.items():
            if not new_path or len(new_path) < 2:
                continue
            if self.agv_env.agents.get(aid) is None:
                continue
            batch[aid] = new_path

        if batch:
            # append semantics: 기존 raw_path 뒤에 이어붙임 (순간이동 없음).
            # new_path[0] 은 agent.raw_path[-1] 과 동일 state 여야 함.
            self.agv_env.append_agents_batch(batch, sim_time)
            # 이동하게 된 idle/pushed AGV의 done_notified 초기화
            for aid in batch:
                if aid in idle_aids or aid in self._agv_pushed:
                    self._agv_done_notified.discard(aid)

            # arrival_idx 노드에 is_dwell=True 표시 — recompute 가 duration 보존
            for aid in batch:
                if aid in idle_aids or aid in self._agv_pushed:
                    continue
                arr_idx = self._agv_arrival_idx.get(aid)
                if arr_idx is None:
                    continue
                agent = self.agv_env.agents.get(aid)
                if agent is None or arr_idx >= len(agent.raw_path):
                    continue
                sid, t = agent.raw_path[arr_idx]
                nk = self.agv_env._nk(sid, aid, t)
                if self.agv_env.G.has_node(nk):
                    self.agv_env.G.nodes[nk]['is_dwell'] = True

        if _prof_on:
            _now = _pt.perf_counter()
            self._sub_batch_s += _now - _stage_t0
            _stage_t0 = _now
        n_replanned = len(result.paths)
        self._plan_status = f'Replanned {n_replanned} AGV @ t={sim_time:.0f}s'
        self._agv_pending_replan.clear()
        # Debug-only: per-replan snapshot to logs/plan_log.txt + latest_tapg.pkl.
        # 30대 × raw_path text dump + pickle dump → 매 replan 12-27ms.
        if self._coarse_debug:
            self._save_plan_snapshot(f'replan_{n_replanned}agv')
        if _prof_on:
            _now = _pt.perf_counter()
            self._sub_snapshot_s += _now - _stage_t0
            _stage_t0 = _now

        # Cycle detection — debug only. TAPG G grow 에 따라 매 replan 13-34ms.
        # find_cycle 자체가 무거우므로 coarse-debug 일 때만.
        if not self._coarse_debug:
            if _prof_on:
                _now = _pt.perf_counter()
                self._sub_findcycle_s += _now - _stage_t0
                _stage_t0 = _now
        import networkx as nx
        try:
            if not self._coarse_debug:
                raise nx.NetworkXNoCycle()
            cycle_edges = nx.find_cycle(self.agv_env.G)
            cycles = list(nx.simple_cycles(self.agv_env.G))
            print(f'\n[CYCLE DETECTED] t={sim_time:.2f}s after replan, {len(cycles)} cycles')
            # find_cycle returns edges: [(u, v), ...]
            print(f'  Cycle edges (find_cycle):')
            for u, v, *_ in cycle_edges:
                u_aid = u[1] - 100 if isinstance(u[1], int) else u[1]
                v_aid = v[1] - 100 if isinstance(v[1], int) else v[1]
                print(f'    A{u_aid} ({u[0]}, t={u[2]:.2f}) → A{v_aid} ({v[0]}, t={v[2]:.2f})')
            # simple_cycles: 전체 cycle 노드 목록
            min_cyc = min(cycles, key=len)
            print(f'  Minimal simple_cycle ({len(min_cyc)} nodes):')
            for n in min_cyc:
                n_aid = n[1] - 100 if isinstance(n[1], int) else n[1]
                preds = list(self.agv_env.G.predecessors(n))
                succs = list(self.agv_env.G.successors(n))
                # self-loop 확인
                is_selfloop = n in succs
                print(f'    A{n_aid}: {n[0]} t={n[2]:.2f}  '
                      f'self_loop={is_selfloop}  '
                      f'in_deg={len(preds)}  out_deg={len(succs)}')
                # 이 노드의 모든 successor 출력
                for s in succs:
                    s_aid = s[1] - 100 if isinstance(s[1], int) else s[1]
                    print(f'      → A{s_aid}: {s[0]} t={s[2]:.2f}')
            self._save_plan_snapshot(f'CYCLE_DETECTED')
        except nx.NetworkXNoCycle:
            pass
        if _prof_on:
            _now = _pt.perf_counter()
            self._sub_findcycle_s += _now - _stage_t0
            _stage_t0 = _now

        # ── Compact 1-line replan summary ──
        n_state = {}
        for a in self.agv_agents:
            n_state[a.state] = n_state.get(a.state, 0) + 1
        tapg_brief = ' '.join(f'{k}={v}' for k, v in sorted(n_state.items()))
        push_note = f' push_fail={push_failed_aids}' if push_failed_aids else ''
        print(f'[REPLAN] t={sim_time:.2f}s pending={sorted(done_ids)} '
              f'goals={new_goals} cs={len(all_constraints)} '
              f'dur={_plan_dt:.2f}s tapg=[{tapg_brief}] status=OK{push_note}')
        if _prof_on: self._sub_afterplan_s += _pt.perf_counter() - _afterplan_t0

    # ── Buttons ──────────────────────────────────────────────────────────────

    def _build_buttons(self):
        sx, sw = MAP_W + 10, SIDE_W - 20
        y, H, G = 12, 28, 5

        self.btn_start = Button((sx, y, sw, H), '▶ Start / Pause'); y += H+G
        self.btn_reset = Button((sx, y, sw, H), '↺  Reset',
                                base=(80, 40, 40)); y += H+G+4

        bw = (sw+2)//4
        self.btns_spd = []
        for row in range(2):
            for col in range(4):
                idx = row*4 + col
                if idx >= len(SIM_SPEED_LABELS):
                    break
                b = Button((sx+col*bw, y, bw-2, H-4), SIM_SPEED_LABELS[idx],
                           toggle=True)
                b.active = (idx == self.spd_idx)
                self.btns_spd.append(b)
            y += H-4+2
        y += G+4

        # OHT controls
        half = (sw-G)//2
        self.btn_oht_del = Button((sx, y, 30, H-4), '-', base=(80,40,40))
        self.btn_oht_add = Button((sx+sw-30, y, 30, H-4), '+')
        self._oht_count_rect = pygame.Rect(sx+32, y, sw-64, H-4)
        y += H-4+G

        # AGV controls
        self.btn_agv_del = Button((sx, y, 30, H-4), '-', base=(80,40,40))
        self.btn_agv_add = Button((sx+sw-30, y, 30, H-4), '+')
        self._agv_count_rect = pygame.Rect(sx+32, y, sw-64, H-4)
        y += H-4+G

        self.btn_shuffle = Button((sx, y, sw, H-4), 'S: Shuffle All'); y += H-4+G

        self._info_y = y + 4
        self.all_btns = [self.btn_start, self.btn_reset, *self.btns_spd,
                         self.btn_oht_add, self.btn_oht_del,
                         self.btn_agv_add, self.btn_agv_del,
                         self.btn_shuffle]

    # ── Update ───────────────────────────────────────────────────────────────

    def _check_deadlock(self):
        """모든 AGV 가 MOVING/ROTATING 아니면 시간 경과 누적, threshold 초과시 raise.

        '진짜 deadlock' 만 잡기 위한 조건:
        - 어떤 AGV 라도 MOVING/ROTATING 이면 정상 → 타이머 reset
        - 진행 중인 작업이 하나도 없으면 (모두 phase=IDLE 이고 PUSH 아님)
          → 정상 idle (MCS 가 새 작업을 줄 때까지 대기) → 타이머 reset
        - 작업 진행 중인 AGV (TO_PICKUP/LOADING/TO_DELIVERY/UNLOADING) 또는
          push 된 AGV 가 있는데 아무도 안 움직이면 → 잠재적 deadlock
        """
        if not self.agv_agents:
            return
        any_active = any(a.state in (AGV_MOVING, AGV_ROTATING)
                         for a in self.agv_agents)
        if any_active:
            self._last_active_sim_time = self.sim_time
            return

        # 작업 진행 중인 AGV 가 있는지 확인
        has_pending_work = False
        for a in self.agv_agents:
            if a.id in self._agv_pushed:
                has_pending_work = True
                break
            b = self.mcs.bindings.get(a.id)
            if b and b.phase != VehicleJobState.IDLE:
                has_pending_work = True
                break
        if not has_pending_work:
            # 모든 AGV 정상 idle (MCS 의 새 task 대기 중) — deadlock 아님
            self._last_active_sim_time = self.sim_time
            return

        elapsed = self.sim_time - self._last_active_sim_time
        if elapsed >= self._deadlock_threshold:
            raise DeadlockDetected(
                f'no AGV moved for {elapsed:.1f}s '
                f'(threshold={self._deadlock_threshold:.1f}s)')

    def update(self, dt_real: float):
        # pygame_gui 는 sim pause 와 무관하게 매 프레임 update
        if hasattr(self, 'ui_manager'):
            self.ui_manager.update(dt_real)
            self._refresh_agv_jobs_list()
            self._refresh_agv_detail_panel()
            self._refresh_kpi_panel()

        if not self.running:
            return
        # Frame profiler — section 별 wall time 측정
        _prof = self._profile_frames_ms > 0.0
        if _prof:
            import time as _time
            _sec = {}
            _t_last = _time.perf_counter()
            # Reset per-frame planner/SIPP counters
            self._sipp_call_count = 0
            self._sipp_total_s = 0.0
            self._planner_plan_count = 0
            self._planner_plan_s = 0.0
            self._sub_prune_s = 0.0
            self._sub_recompute_s = 0.0
            self._sub_extend_s = 0.0
            self._sub_constraint_s = 0.0
            self._sub_truncate_s = 0.0
            self._sub_pickpark_s = 0.0
            self._sub_afterplan_s = 0.0
            self._sub_dispatch_s = 0.0
            self._replan_call_count = 0
            self._mcs_evt_count = 0
            self._mcs_evt_by_kind = {}
            self._sub_leg2_s = 0.0
            self._sub_replan_hist_s = 0.0
            self._sub_batch_s = 0.0
            self._sub_findcycle_s = 0.0
            self._sub_snapshot_s = 0.0
            def _mark(name):
                nonlocal _t_last
                t = _time.perf_counter()
                _sec[name] = _sec.get(name, 0.0) + (t - _t_last)
                _t_last = t
        else:
            def _mark(name): pass
        if self._headless:
            # Headless: fixed step (= max speed, no wall-time coupling)
            dt_sim = 1.0 / 30.0   # 30Hz sim tick = ~0.033s per iter
        else:
            dt_sim = dt_real * SIM_SPEEDS[self.spd_idx]
        self.sim_time += dt_sim

        # Warmup 종료 시점 도달 -> KPI 측정 시작 (= log clear)
        if not self._warmup_done and self.sim_time >= self._warmup_time:
            self._warmup_done = True
            self._on_warmup_done()
        # KPI tracking (매 frame)
        self._update_kpi_tracking()

        # max-time 자동 종료 (KPI 비교용)
        if self._max_sim_time > 0 and self.sim_time >= self._max_sim_time:
            print(f'\n[AUTO-STOP] sim_time {self.sim_time:.2f}s >= '
                  f'max_time {self._max_sim_time:.2f}s')
            self._print_event_stats()
            self._print_movement_summary()
            pygame.quit()
            import sys
            sys.exit(0)

        _mark('pre')
        # step all DES engines
        self.oht_env.step(self.sim_time)
        _mark('oht.step')
        self.agv_env.step(self.sim_time)
        _mark('agv.step')
        # Coarse mode: reactive push - event-driven. 새로 WAITING 으로 전이한
        # AGV 가 있을 때만 trigger (= 1초 polling 제거). [[no-polling]]
        if self._planner_type == 'coarse':
            cur_waiting = frozenset(a.id for a in self.agv_agents
                                    if a.state == AGV_WAITING)
            prev_waiting = getattr(self, '_prev_waiting_set', frozenset())
            new_waiters = cur_waiting - prev_waiting
            self._prev_waiting_set = cur_waiting
            if new_waiters:
                self._extend_with_exit_coarse(set(), self.sim_time)
        _mark('reactive_push')
        self._check_deadlock()
        _mark('deadlock_check')
        # Step all 3DS TAPG environments + elevators
        for fid, fd in self.s3d_floor_data.items():
            fd['env'].step(self.sim_time)
        self._step_elevators(self.sim_time)
        _mark('s3d_elev')

        # OHT idle-spread: dispatch 외에서도 BLOCKED leader-follower 관계 해소
        self._oht_idle_spread(self.sim_time)

        # OHT 안전거리 검증 — 매 step 모든 차량 쌍 거리 측정
        self._oht_safety_check(self.sim_time)
        _mark('oht_spread_safety')

        # ── MCS: DONE 감지 → push 정리 ──
        # OHT 의 LOAD/UNLOAD 흐름은 MCSOHTBridge (graph_des_v6.job_mgr) 가
        # on_arrive 에서 처리. 여기서는 push 도착 정리만.
        from mcs_unified import post_vehicle_arrived
        for a in self.oht_agents:
            if a.state == DONE:
                self._oht_done_notified.add(a.id)
                self._oht_pushed.discard(a.id)

        # AGV: 물리적 도착 감지 (dwell 진입 시점) → MCS VEHICLE_ARRIVED
        for a in self.agv_agents:
            if a.id in self._agv_arrived_notified:
                continue
            arr_idx = self._agv_arrival_idx.get(a.id)
            if arr_idx is not None and a.path_idx >= arr_idx:
                b = self.mcs.bindings.get(a.id)
                if b and b.load is not None:
                    post_vehicle_arrived(self._mcs_heap, self._mcs_seq,
                                        a.id, b.token, self.sim_time)
                self._agv_arrived_notified.add(a.id)

        # AGV DONE → pushed 처리 / idle 감지 (MCS 통보는 위에서 이미 완료)
        for a in self.agv_agents:
            if a.state == AGV_DONE and a.id not in self._agv_done_notified:
                # push된 AGV가 임시 목적지 도착 → goal 클리어, MCS 통보 안 함
                if a.id in self._agv_pushed:
                    self._agv_pushed.discard(a.id)   # push 완료
                    self._cycle_push_dest.pop(a.id, None)   # 시각화 클리어
                    # MCS task 아직 활성 (TO_PICKUP/TO_DELIVERY) 이면
                    # *원래 MCS goal* (src/dst port) 로 _agv_goals 복원 +
                    # pending_replan 에 추가. 안 하면 다음 replan 이 다시
                    # push destination 으로 plan -> 영구 push-cycle.
                    b = self.mcs.bindings.get(a.id)
                    if (b and b.load
                            and b.phase == VehicleJobState.TO_PICKUP):
                        self._agv_goals[a.id] = b.load.src_port
                        self._agv_pending_replan.add(a.id)
                    elif (b and b.load
                            and b.phase == VehicleJobState.TO_DELIVERY):
                        self._agv_goals[a.id] = b.load.dst_port
                        self._agv_pending_replan.add(a.id)
                    else:
                        self._agv_goals.pop(a.id, None)
                    self._agv_done_notified.add(a.id)
                    continue
                # arrival_idx가 설정된 AGV는 위에서 이미 통보됨 → 중복 방지
                if a.id in self._agv_arrived_notified:
                    self._agv_done_notified.add(a.id)
                    continue
                # arrival_idx 없는 AGV (초기 배치 등) → MCS goal 위치 확인 후 통보.
                # AGV 가 실제로 MCS expected port (src/dst) 에 있을 때만 fire.
                # 그 외 (SIPP fail 로 plan 누락된 채 DONE 됨) 는 다음 tick 재시도.
                b = self.mcs.bindings.get(a.id)
                if b and b.load is not None and a.raw_path:
                    cur_node = a.raw_path[-1][0].split(',')[1]
                    if b.phase == VehicleJobState.TO_PICKUP:
                        expected = b.load.src_port
                    elif b.phase == VehicleJobState.TO_DELIVERY:
                        expected = b.load.dst_port
                    else:
                        expected = None
                    if expected is not None and cur_node == expected:
                        post_vehicle_arrived(self._mcs_heap, self._mcs_seq,
                                            a.id, b.token, self.sim_time)
                        self._agv_done_notified.add(a.id)
                    # cur_node != expected: don't notify, don't mark done_notified
                    # → 다음 tick 에 plan 성공하면 정상 처리
                else:
                    self._agv_done_notified.add(a.id)

        # 3DS DONE → MCS VEHICLE_ARRIVED
        for fid, fd in self.s3d_floor_data.items():
            for a in fd['agents']:
                if a.state == AGV_DONE and a.id not in self._s3d_done_notified:
                    b = self.mcs.bindings.get(a.id)
                    if b and b.load is not None:
                        post_vehicle_arrived(self._mcs_heap, self._mcs_seq,
                                            a.id, b.token, self.sim_time)
                    self._s3d_done_notified.add(a.id)

        _mark('done_notify')
        # ── MCS step ──
        self.mcs.handle_all(self.sim_time)
        _mark('mcs.handle_all')

        # ── 3DS shuttle retreat ──
        # Lift gate 노드는 articulation point — 셔틀이 거기 inf-park 하면
        # 다른 셔틀의 경로를 막아 SIPP fail. deliver 후 (free + 위치=gate) 이면
        # 가까운 비-gate buffer 로 자동 retreat.
        self._retreat_3ds_at_gates(self.sim_time)

        # ── Dwell 측정: phase 전이 감지 ──
        # LOADING/UNLOADING 진입 시 t_start 기록, 이탈 시 duration 계산
        for a in self.agv_agents:
            b = self.mcs.bindings.get(a.id)
            cur = b.phase if b else None
            prev = self._agv_phase_prev.get(a.id)
            if cur != prev:
                if cur in (VehicleJobState.LOADING, VehicleJobState.UNLOADING):
                    # 진입
                    cur_node = self._mcs_get_agv_node(a.id) or '?'
                    self._agv_dwell_open[a.id] = (cur.value, self.sim_time, cur_node)
                if prev in (VehicleJobState.LOADING, VehicleJobState.UNLOADING):
                    # 이탈 → duration 기록
                    open_rec = self._agv_dwell_open.pop(a.id, None)
                    if open_rec:
                        kind, t_start, port = open_rec
                        self._agv_dwell_log.append({
                            'aid': a.id, 'kind': kind,
                            't_start': t_start, 't_end': self.sim_time,
                            'duration': self.sim_time - t_start,
                            'port': port,
                        })
                self._agv_phase_prev[a.id] = cur

        _mark('dwell_track')
        # AGV collision detection
        self._check_agv_collisions()
        _mark('agv_collision')

        if _prof:
            total_ms = sum(_sec.values()) * 1000.0
            if total_ms >= self._profile_frames_ms:
                top = sorted(_sec.items(), key=lambda kv: -kv[1])[:6]
                parts = ' '.join(f'{k}={v*1000:.0f}' for k, v in top
                                 if v * 1000 >= 1.0)
                extra = (f' :: replan={self._replan_call_count} '
                         f'plan={self._planner_plan_s*1000:.0f} '
                         f'sipp={self._sipp_total_s*1000:.0f}'
                         f'/{self._sipp_call_count}')
                sub = (f' | sub: extend={self._sub_extend_s*1000:.0f} '
                       f'park={self._sub_pickpark_s*1000:.0f} '
                       f'after={self._sub_afterplan_s*1000:.0f} '
                       f'(leg2={self._sub_leg2_s*1000:.0f} '
                       f'hist={self._sub_replan_hist_s*1000:.0f} '
                       f'batch={self._sub_batch_s*1000:.0f} '
                       f'snap={self._sub_snapshot_s*1000:.0f} '
                       f'cyc={self._sub_findcycle_s*1000:.0f})')
                # MCS event breakdown
                evs = sorted(self._mcs_evt_by_kind.items(),
                             key=lambda kv: -kv[1][1])[:5]
                ev_str = ' '.join(f'{k}={c}/{s*1000:.0f}'
                                  for k, (c, s) in evs if s * 1000 >= 0.5)
                sub += f' | evt[{self._mcs_evt_count}]: {ev_str}'
                print(f'[SLOW-FRAME t={self.sim_time:.1f}] '
                      f'total={total_ms:.0f}ms | {parts}{extra}{sub}')

    # ── Input ────────────────────────────────────────────────────────────────

    def handle_events(self):
        if self._headless:
            return   # 키/마우스 입력 무시
        ww, wh = self.screen.get_size()
        mpos = pygame.mouse.get_pos()
        for b in self.all_btns:
            b.update_hover(mpos)

        for ev in pygame.event.get():
            # pygame_gui 이벤트 처리 (위젯 입력). True 반환 = UI 가 소비.
            ui_consumed = self.ui_manager.process_events(ev)
            # UI 가 소비한 마우스 클릭 / 드래그는 우리 핸들러로 전달 X
            # (= window 드래그 시 배경 카메라 같이 움직이는 문제 fix)
            if ui_consumed and ev.type in (pygame.MOUSEBUTTONDOWN,
                                            pygame.MOUSEBUTTONUP,
                                            pygame.MOUSEMOTION,
                                            pygame.MOUSEWHEEL):
                continue

            # AGV jobs panel 버튼 핸들링
            if ev.type == pygame_gui.UI_BUTTON_PRESSED:
                if ev.ui_element == self._agv_add_btn:
                    self._panel_add_load()
                elif ev.ui_element == self._agv_cancel_btn:
                    self._panel_cancel_selected()

            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif k == pygame.K_SPACE:
                    self.running = not self.running
                elif k == pygame.K_r:
                    self._reset()
                elif k == pygame.K_s:
                    self._shuffle()
                elif k == pygame.K_d:
                    self._dump_agv_status()
                    self._dump_replan_history()
                elif k == pygame.K_m:
                    self._print_movement_summary()
                elif k == pygame.K_o:
                    self._add_oht()
                elif k == pygame.K_l:
                    # OHT leader chain toggle (test_graph_v6 호환)
                    self._show_oht_leaders = not self._show_oht_leaders
                    print(f'[OHT-LEADERS] {"ON" if self._show_oht_leaders else "OFF"}')
                elif k == pygame.K_t:
                    # OHT destination markers toggle (orange=push, green=dest)
                    self._show_oht_dests = not self._show_oht_dests
                    print(f'[OHT-DESTS] {"ON" if self._show_oht_dests else "OFF"}')
                elif k == pygame.K_c:
                    # OHT commit horizon (x_marker = trajectory commit 지점)
                    self._show_oht_commit = not self._show_oht_commit
                    print(f'[OHT-COMMIT] {"ON" if self._show_oht_commit else "OFF"}')
                elif k == pygame.K_n:
                    self._add_agv()
                elif k == pygame.K_p:
                    self._del_agv()
                elif k == pygame.K_j:
                    self._toggle_agv_jobs_panel()
                elif k == pygame.K_v:
                    self._toggle_agv_detail_panel()
                elif k == pygame.K_a:
                    self._show_dep_arrows = not self._show_dep_arrows
                    print(f'[DEP-ARROWS] {"ON" if self._show_dep_arrows else "OFF"}')
                elif k == pygame.K_f:
                    # Zoom-to-fit (camera reset)
                    self.cam = Camera(self.bg_bbox, MAP_W, self.screen.get_height())
                elif k == pygame.K_LEFTBRACKET:
                    self._cycle_agv_selection(-1)
                elif k == pygame.K_RIGHTBRACKET:
                    self._cycle_agv_selection(+1)
                elif k == pygame.K_BACKSPACE:
                    self._select_agv(None)
                    if self._agv_detail_visible:
                        self._toggle_agv_detail_panel()
                elif k in (pygame.K_EQUALS, pygame.K_PLUS):
                    self._set_spd(min(self.spd_idx+1, len(SIM_SPEEDS)-1))
                elif k == pygame.K_MINUS:
                    self._set_spd(max(self.spd_idx-1, 0))
                elif k == pygame.K_k:
                    self.cam = Camera(self.bg_bbox, ww-SIDE_W, wh)
                elif k == pygame.K_i:
                    self._show_node_ids = not self._show_node_ids
                elif k == pygame.K_x:
                    self._show_sidings = not self._show_sidings
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    pos = ev.pos
                    if self.btn_start.clicked(pos):
                        self.running = not self.running
                    elif self.btn_reset.clicked(pos):
                        self._reset()
                    elif self.btn_shuffle.clicked(pos):
                        self._shuffle()
                    elif self.btn_oht_add.clicked(pos):
                        self._add_oht()
                    elif self.btn_oht_del.clicked(pos):
                        self._del_oht()
                    elif self.btn_agv_add.clicked(pos):
                        self._add_agv()
                    elif self.btn_agv_del.clicked(pos):
                        self._del_agv()
                    else:
                        for i, b in enumerate(self.btns_spd):
                            if b.clicked(pos):
                                self._set_spd(i)
                    if ev.pos[0] < MAP_W:
                        # AGV 클릭 픽 우선 — 가까운 AGV 있으면 선택, 없으면 카메라 드래그
                        picked = self._pick_agv_at_screen(ev.pos[0], ev.pos[1])
                        if picked is not None:
                            self._select_agv(picked)
                        else:
                            self.cam.on_down(ev.pos)
                elif ev.button == 4:
                    if ev.pos[0] >= MAP_W:
                        # Sidebar 영역: 위로 scroll (= 위쪽 정보 보임)
                        self._sidebar_scroll = max(
                            0, self._sidebar_scroll - 30)
                    else:
                        self.cam.on_scroll(ev.pos, up=True)
                elif ev.button == 5:
                    if ev.pos[0] >= MAP_W:
                        # Sidebar 영역: 아래로 scroll (= 아래쪽 정보 보임)
                        self._sidebar_scroll += 30
                    else:
                        self.cam.on_scroll(ev.pos, up=False)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    self.cam.on_up()
            elif ev.type == pygame.MOUSEMOTION:
                self.cam.on_move(ev.pos)
            elif ev.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(ev.size, pygame.RESIZABLE)
                self._on_resize(ev.size)

    def _on_resize(self, size):
        """Window resize 시 layout 재계산:
        - global MAP_W = ww - SIDE_W
        - Camera 재구성 (= 더 넓은 map 영역 활용)
        - 우측 UIWindow (KPI / detail / jobs) 를 새 우측 가장자리로 set_position
        - native 버튼들 (_build_buttons) 도 새 위치로 재구성
        - KPI window 높이를 wh-padding 으로 확장 (= 하단 정보 다 보이게)
        """
        global MAP_W
        ww, wh = size
        MAP_W = max(200, ww - SIDE_W)
        self.cam = Camera(self.bg_bbox, MAP_W, wh)
        if hasattr(self, 'ui_manager'):
            self.ui_manager.set_window_resolution(size)
        # Native buttons: MAP_W 갱신 후 재배치 — sidebar 의 sx 가 MAP_W 기반
        if hasattr(self, 'all_btns'):
            self._build_buttons()
        # KPI window: 우측 + wh 에 맞춰 늘림 (= 잘림 방지).
        # AGV Detail / Jobs window 는 floating — 사용자가 직접 드래그하는
        # 창이라 auto-reposition 안 함.
        if hasattr(self, '_kpi_window'):
            kpi_w, kpi_h = 380, max(280, wh - 80)
            self._kpi_window.set_position((max(0, ww - kpi_w), 60))
            try:
                self._kpi_window.set_dimensions((kpi_w, kpi_h))
            except Exception:
                pass

    def _reset(self):
        self.sim_time = 0.0
        self.running  = False

        # reset OHT
        self._oht_done_notified = set()
        self._oht_pushed = set()
        self._agv_done_notified = set()
        self._s3d_done_notified = set()
        self._agv_arrived_notified = set()
        self._agv_arrival_idx = {}
        self._oht_next_id = 0
        self.oht_agents = []
        for seg in self.oht_map.segments.values():
            seg.queue.clear()
        self.oht_env = OHTEnvironmentDES(self.oht_map, cross_segment=True)
        self._init_oht_agents()

        # reset AGV
        self._agv_next_id = 100
        self.agv_agents = []
        self._agv_start_positions = {}
        self.agv_env = TAPGEnvironment(self.amr_graph, accel=500.0, decel=500.0)
        self._configure_coarse_mode(self.amr_graph)
        self._agv_goals = {}
        self._agv_pushed = set()
        self._agv_pending_replan = set()
        self._plan_status = ''
        self._init_agv_agents()

        # reset 3DS (SIPP+TAPG per floor) + elevators
        self.s3d_agents = []
        self._init_3ds()  # _init_elevators() 포함

        # reset MCS
        self._init_mcs()

    def _shuffle(self):
        # OHT
        nodes = list(self.oht_map.nodes.keys())
        random.shuffle(nodes)
        excluded = set()
        for a in self.oht_agents:
            for start in nodes:
                if start not in excluded:
                    path = self.oht_map.bfs_path(start)
                    if len(path) > 1:
                        excluded |= self.oht_map.nearby_nodes(start, self.oht_map.h_min)
                        self.oht_env.reassign(a, path, self.sim_time)
                        break
        # AGV -replan with new port destinations
        self._plan_agv_paths(self.sim_time)
        # 3DS -reinit (re-plan all floors)
        self.s3d_agents = []
        self._init_3ds()

    def _add_oht(self):
        if len(self.oht_agents) >= MAX_AGENTS:
            return
        excluded = set()
        for a in self.oht_agents:
            excluded |= self.oht_map.nearby_nodes(a.cur_node, self.oht_map.h_min)
        a = self._make_oht_agent(self._oht_next_id, excluded)
        if a:
            self._oht_next_id += 1
            self.oht_agents.append(a)
            self.oht_env.add_agent(a, t_start=self.sim_time)
            self.mcs.register_vehicle(a.id, 'OHT')
        self._n_oht = len(self.oht_agents)

    def _del_oht(self):
        if not self.oht_agents:
            return
        a = self.oht_agents.pop()
        self.oht_env.remove_agent(a.id)
        self.mcs.unregister_vehicle(a.id)
        self._n_oht = len(self.oht_agents)

    def _add_agv(self):
        if len(self.agv_agents) >= MAX_AGENTS:
            return
        # Record current positions for replanning
        occupied = set()
        for a in self.agv_agents:
            nid = a.raw_path[-1][0].split(',')[1] if a.state == AGV_DONE else \
                  a.raw_path[a.path_idx][0].split(',')[1]
            occupied.add(nid)
        nodes = list(self.amr_graph.nodes.keys())
        random.shuffle(nodes)
        for start in nodes:
            if start not in occupied:
                aid = self._agv_next_id
                self._agv_start_positions[aid] = start
                self._agv_next_id += 1
                break
        self._n_agv += 1
        self._plan_agv_paths(self.sim_time)

    def _del_agv(self):
        if not self.agv_agents:
            return
        a = self.agv_agents.pop()
        self.agv_env.remove_agent(a.id)
        self._agv_goals.pop(a.id, None)
        self._agv_pushed.discard(a.id)
        self.mcs.unregister_vehicle(a.id)
        self._n_agv = len(self.agv_agents)

    def _set_spd(self, idx: int):
        self.spd_idx = idx
        for i, b in enumerate(self.btns_spd):
            b.active = (i == idx)

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self):
        if self._headless:
            return   # 장기 sim 가속 - render skip
        ww, wh = self.screen.get_size()
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, (24, 26, 36),
                         pygame.Rect(MAP_W, 0, SIDE_W, wh))

        self._draw_background(ww, wh)
        self._draw_oht_network(ww, wh)
        self._draw_agv_network(ww, wh)
        self._draw_selected_agv_path(ww, wh)
        self._draw_agents(ww, wh)
        if self._show_dep_arrows:
            self._draw_dep_arrows(ww, wh)
        if self._show_oht_leaders:
            self._draw_oht_leaders(ww, wh)
        if self._show_oht_dests:
            self._draw_oht_dests(ww, wh)
        if self._show_oht_commit:
            self._draw_oht_commit(ww, wh)
        self._draw_sidebar(wh)
        if hasattr(self, 'ui_manager'):
            self.ui_manager.draw_ui(self.screen)
        pygame.display.flip()

    def _draw_cycle_push_routes(self, ww, wh):
        """CYCLE-PUSH 받은 victim AGV 의 escape 경로를 굵은 실선으로 표시.
        AGV 현재 raw_path 의 [path_idx..end] 를 따라 그림. 도착 후 자동 클리어."""
        if not self._cycle_push_dest:
            return
        nodes = self.amr_graph.nodes
        color = (255, 100, 0)   # 주황
        width = max(3, int(self.cam.scale * 0.0015))
        for aid, dest_node in list(self._cycle_push_dest.items()):
            a = next((x for x in self.agv_agents if x.id == aid), None)
            if a is None or not a.raw_path:
                continue
            # raw_path 의 미실행 구간을 node sequence 로 변환
            pts = []
            seen = set()
            for sid, _ in a.raw_path[a.path_idx:]:
                if ',' not in sid:
                    continue
                n = sid.split(',')[1]
                if n in seen:
                    continue
                seen.add(n)
                if n in nodes:
                    pts.append((nodes[n].x, nodes[n].y))
            if len(pts) < 2:
                continue
            # 화면 좌표로 변환 + 굵은 line
            spts = [self.cam.to_screen(x, y) for x, y in pts]
            pygame.draw.lines(self.screen, color, False, spts, width)
            # Dest 노드에 작은 원 마커
            if dest_node in nodes:
                d = nodes[dest_node]
                dx, dy = self.cam.to_screen(d.x, d.y)
                pygame.draw.circle(self.screen, color, (dx, dy),
                                    max(6, int(self.cam.scale * 0.005)), 2)

    def _draw_dep_arrows(self, ww, wh):
        """Coarse dependency 시각화. _try_claim_next dry-run 은 비싸므로
        sim_time 동일하면 cache 재사용 (= 같은 sim tick 안의 frame 들은 1회)."""
        import re
        from env_tapg import MOVING as _MOVING, ROTATING as _ROTATING
        env = getattr(self, 'agv_env', None)
        if env is None or not getattr(env, '_coarse_mode', False):
            return
        agent_by_aid = {a.id: a for a in self.agv_agents}
        # Cache: 0.2s 간격으로만 recompute (frame rate 와 무관하게 비용 cap)
        last_t = getattr(self, '_dep_arrow_cache_t', -1.0)
        cache = getattr(self, '_dep_arrow_cache_pairs', None)
        if cache is not None and abs(self.sim_time - last_t) < 0.2:
            pairs = cache
        else:
            pairs = []
            for a in self.agv_agents:
                if a.state in (_MOVING, _ROTATING):
                    continue
                if not a.raw_path:
                    continue
                if a.path_idx >= len(a.raw_path):
                    continue
                saved_claim = a.claim_idx
                env._last_block_info = ''
                ok = env._try_claim_next(a)
                a.claim_idx = saved_claim
                if ok:
                    continue
                info = env._last_block_info or ''
                m = re.search(r'by (?:DONE )?V(\d+)@', info)
                if not m:
                    continue
                blocker_aid = int(m.group(1)) + 100
                if blocker_aid in agent_by_aid and blocker_aid != a.id:
                    pairs.append((a.id, blocker_aid))
            self._dep_arrow_cache_t = self.sim_time
            self._dep_arrow_cache_pairs = pairs
        # Draw arrows
        for bid_blocked, bid_blocker in pairs:
            ag_b = agent_by_aid[bid_blocked]
            ag_k = agent_by_aid[bid_blocker]
            p1 = self.cam.to_screen(ag_b.x, ag_b.y)
            p2 = self.cam.to_screen(ag_k.x, ag_k.y)
            if (p1[0] < 0 or p1[0] > MAP_W or p2[0] < 0 or p2[0] > MAP_W):
                continue
            draw_arrow(self.screen, (255, 60, 60), p1, p2, width=2, head=10)

    def _draw_oht_leaders(self, ww, wh):
        """L 키 토글. 모든 OHT 의 leader chain (test_graph_v6 호환).
        OHT → leader 위치로 노란 선 + 작은 동그라미."""
        LEAD_COL = (255, 255, 100)
        for a in self.oht_agents:
            v = getattr(a, 'vehicle', None)
            if v is None or v.leader is None:
                continue
            p1 = self.cam.to_screen(a.x, a.y)
            p2 = self.cam.to_screen(v.leader.x, v.leader.y)
            if (p1[0] < 0 or p1[0] > MAP_W) and (p2[0] < 0 or p2[0] > MAP_W):
                continue
            pygame.draw.line(self.screen, LEAD_COL, p1, p2, 2)
            pygame.draw.circle(self.screen, LEAD_COL,
                               (int(p2[0]), int(p2[1])), 6, 2)

    def _draw_oht_dests(self, ww, wh):
        """T 키 토글. 모든 OHT 의 destination marker (test_graph_v6 호환).
        orange = push (= 다른 차 위해 비키는 임시 dest), green = 본인 목적지.

        Push 판단: vis_mcs 가 직접 추적하는 _oht_pushed set 사용 (v.via_push
        는 oht_env.reassign 이 set 안 함).
        """
        PUSH_COL = (255, 140, 0)
        DEST_COL = (0, 220, 120)
        nodes = self.oht_map.nodes
        for a in self.oht_agents:
            v = getattr(a, 'vehicle', None)
            if v is None or not v.dest_node or v.dest_reached:
                continue
            dn = nodes.get(v.dest_node)
            if dn is None:
                continue
            p1 = self.cam.to_screen(a.x, a.y)
            p2 = self.cam.to_screen(dn.x, dn.y)
            if (p1[0] < 0 or p1[0] > MAP_W) and (p2[0] < 0 or p2[0] > MAP_W):
                continue
            is_push = (a.id in self._oht_pushed) or getattr(v, 'via_push', False)
            col = PUSH_COL if is_push else DEST_COL
            pygame.draw.line(self.screen, col, p1, p2, 1)
            pygame.draw.circle(self.screen, col,
                               (int(p2[0]), int(p2[1])), 6, 2)

    def _draw_oht_commit(self, ww, wh):
        """C 키 토글. 각 OHT 의 commit horizon (= trajectory/lock commit 지점,
        v.x_marker). 현재 위치 → commit 지점 까지 굵은 선 + X 마커.

        commit 범위 = '실제 도달 속도 기준 brake distance' (graph_des_v6 의
        commit_horizon_dist). dest marker (T) 와 달리 *이 만큼만 trajectory 가
        commit/lock 됨* 을 보여줌.
        """
        from graph_des_v5 import _interp_path as _ip
        COMMIT_COL = (255, 255, 255)   # white
        gmap = self.oht_map.gmap
        for a in self.oht_agents:
            v = getattr(a, 'vehicle', None)
            if v is None or not v.path:
                continue
            pidx = getattr(v, 'x_marker_pidx', 0)
            off = getattr(v, 'x_marker_offset', 0.0)
            if pidx >= len(v.path) - 1:
                # commit 지점이 path 끝 — node 위치 사용
                node = gmap.nodes.get(v.path[min(pidx, len(v.path)-1)])
                if node is None:
                    continue
                mx, my = node.x, node.y
            else:
                seg = gmap.segment_between(v.path[pidx], v.path[pidx + 1])
                if seg and seg.path_points:
                    mx, my, _th = _ip(seg.path_points, max(0.0, off))
                else:
                    node = gmap.nodes.get(v.path[pidx])
                    if node is None:
                        continue
                    mx, my = node.x, node.y
            p1 = self.cam.to_screen(a.x, a.y)
            p2 = self.cam.to_screen(mx, my)
            if (p1[0] < 0 or p1[0] > MAP_W) and (p2[0] < 0 or p2[0] > MAP_W):
                continue
            # 현재 위치 → commit 지점 굵은 선
            pygame.draw.line(self.screen, COMMIT_COL, p1, p2, 3)
            # commit 지점 X 마커
            x, y = int(p2[0]), int(p2[1])
            pygame.draw.line(self.screen, COMMIT_COL, (x-6, y-6), (x+6, y+6), 2)
            pygame.draw.line(self.screen, COMMIT_COL, (x-6, y+6), (x+6, y-6), 2)

    def _draw_selected_agv_path(self, ww, wh):
        """선택된 AGV 의 remaining raw_path 를 오버레이로 그림.
        S = 녹색 dot, M = 파란 선, R = 노란 호 (간이 표시)."""
        aid = self._selected_agv_id
        if aid is None: return
        a = next((x for x in self.agv_agents if x.id == aid), None)
        if a is None: return
        path = a.raw_path
        if not path: return

        # node_id → world coord lookup
        def node_xy(nid):
            n = self.amr_graph.nodes.get(nid)
            return (n.x, n.y) if n is not None else None

        COL_S = (120, 240, 120)
        COL_M = (90, 170, 255)
        COL_R = (255, 200, 80)
        COL_HIGHLIGHT = (255, 80, 80)  # 현재 위치 강조
        # Claim 된 영역 (= path[path_idx, claim_idx)) 은 빨강 highlight.
        COL_CLAIM_S = (255, 80, 80)
        COL_CLAIM_M = (255, 80, 80)
        COL_CLAIM_R = (255, 80, 80)

        # 굵기 화면 줌에 비례
        w_line = max(2, int(self.cam.scale * 0.0015))
        r_dot = max(3, int(self.cam.scale * 0.0025))

        claim_idx = getattr(a, 'claim_idx', a.path_idx)

        # 진행 방향 강조 표시: 처음 ~ 끝
        for i in range(a.path_idx, len(path)):
            sid, t = path[i]
            parts = sid.split(',')
            kind = parts[0]
            is_claimed = (i < claim_idx)
            if kind == 'M' and len(parts) >= 3:
                p1 = node_xy(parts[1]); p2 = node_xy(parts[2])
                if p1 and p2:
                    s1 = self.cam.to_screen(*p1)
                    s2 = self.cam.to_screen(*p2)
                    line_col = COL_CLAIM_M if is_claimed else COL_M
                    line_w = w_line + 2 if is_claimed else w_line
                    pygame.draw.line(self.screen, line_col, s1, s2, line_w)
                    draw_arrow(self.screen, line_col, s1, s2,
                               width=line_w, head=max(5, r_dot * 2))
            elif kind in ('S', 'R') and len(parts) >= 2:
                p = node_xy(parts[1])
                if p:
                    sx, sy = self.cam.to_screen(*p)
                    if is_claimed:
                        col = COL_CLAIM_S if kind == 'S' else COL_CLAIM_R
                        radius = r_dot + 2
                    else:
                        col = COL_S if kind == 'S' else COL_R
                        radius = r_dot
                    pygame.draw.circle(self.screen, col, (sx, sy), radius, 0)

        # 현재 위치 (path_idx 노드) 강조
        if a.path_idx < len(path):
            sid, _ = path[a.path_idx]
            parts = sid.split(',')
            if parts[0] in ('S', 'R', 'M') and len(parts) >= 2:
                nid = parts[1]
                p = node_xy(nid)
                if p:
                    sx, sy = self.cam.to_screen(*p)
                    pygame.draw.circle(self.screen, COL_HIGHLIGHT,
                                       (sx, sy), r_dot + 3, 2)

    def _draw_background(self, ww, wh):
        """Draw 3DS floor segments/nodes as dim background."""
        seg_w  = max(1, int(self.cam.scale * 0.0004))
        node_r = max(2, int(self.cam.scale * 0.002))

        for seg in self.bg_segments:
            n1 = self.bg_nodes.get(seg['startNodeId'])
            n2 = self.bg_nodes.get(seg['endNodeId'])
            if not n1 or not n2:
                continue
            a1 = n1.get('area', '')
            a2 = n2.get('area', '')
            # skip OHT_A and AMR_A -drawn separately with detail
            if a1 in ('OHT_A', 'AMR_A') or a2 in ('OHT_A', 'AMR_A'):
                continue
            p1 = self.cam.to_screen(n1['x'], n1['y'])
            p2 = self.cam.to_screen(n2['x'], n2['y'])
            if max(p1[0], p2[0]) < 0 or min(p1[0], p2[0]) > ww:
                continue
            if max(p1[1], p2[1]) < 0 or min(p1[1], p2[1]) > wh:
                continue
            col = AREA_SEG_COLORS.get(a1, COL_SEG)
            pygame.draw.line(self.screen, col, p1, p2, seg_w)

        for nid, node in self.bg_nodes.items():
            area = node.get('area', '')
            if area in ('OHT_A', 'AMR_A'):
                continue
            sx, sy = self.cam.to_screen(node['x'], node['y'])
            if sx < -10 or sx > ww+10 or sy < -10 or sy > wh+10:
                continue
            col = AREA_COLORS.get(area, COL_NODE)
            pygame.draw.circle(self.screen, col, (sx, sy), node_r)
            if self._show_node_ids:
                short = nid.split('.')[-1] if '.' in nid else nid
                lbl = self.font_s.render(short, True, (180, 180, 180))
                self.screen.blit(lbl, (sx + node_r + 1, sy - 6))

        # Area labels
        area_centers = {}
        for nid, node in self.bg_nodes.items():
            area = node.get('area', '')
            if not area:
                continue
            area_centers.setdefault(area, {'xs': [], 'ys': []})
            area_centers[area]['xs'].append(node['x'])
            area_centers[area]['ys'].append(node['y'])
        for area, coords in area_centers.items():
            cx = sum(coords['xs']) / len(coords['xs'])
            cy = max(coords['ys']) + 500
            lx, ly = self.cam.to_screen(cx, cy)
            col = AREA_SEG_COLORS.get(area, COL_DIM)
            lbl = self.font_b.render(area, True, col)
            self.screen.blit(lbl, lbl.get_rect(center=(lx, ly)))

        # ── Elevator shafts: 층간 gate 노드를 연결하는 선 ──
        self._draw_elevator_shafts(ww, wh)

    def _draw_elevator_shafts(self, ww, wh):
        """Draw elevator shafts connecting 3DS floor gate nodes."""
        import json as _json
        if not hasattr(self, '_lift_gate_cache'):
            # 캐시: lift별 [(floor_id, gate_node_id)] -offset 적용된 좌표
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                jdata = _json.load(f)
            self._lift_gate_cache = []
            for lift in jdata.get('lifts', []):
                shaft = []
                for fl in lift.get('floors', []):
                    floor_id = '3DS_F' + fl['id']
                    for g in fl.get('gates', []):
                        for nid in g.get('entryNodes', []):
                            node = self.bg_nodes.get(nid)
                            if node:
                                shaft.append((floor_id, nid,
                                              node['x'], node['y']))
                if shaft:
                    self._lift_gate_cache.append({
                        'id': lift['id'], 'floors': shaft})

        shaft_w = max(2, int(self.cam.scale * 0.001))
        gate_r  = max(4, int(self.cam.scale * 0.006))
        col_shaft = (180, 100, 100)
        col_gate  = (255, 100, 100)

        for lift in self._lift_gate_cache:
            floors = lift['floors']
            # 층간 샤프트 선
            screen_pts = []
            for fid, nid, x, y in floors:
                sp = self.cam.to_screen(x, y)
                screen_pts.append(sp)

            if len(screen_pts) >= 2:
                # 점선 스타일 샤프트
                for i in range(len(screen_pts) - 1):
                    p1, p2 = screen_pts[i], screen_pts[i + 1]
                    pygame.draw.line(self.screen, col_shaft, p1, p2, shaft_w)

                # 샤프트 양 옆에 레일 (입체감)
                for i in range(len(screen_pts) - 1):
                    p1, p2 = screen_pts[i], screen_pts[i + 1]
                    off = max(3, int(self.cam.scale * 0.004))
                    pygame.draw.line(self.screen, (col_shaft[0]//2, col_shaft[1]//2, col_shaft[2]//2),
                                     (p1[0]-off, p1[1]), (p2[0]-off, p2[1]), 1)
                    pygame.draw.line(self.screen, (col_shaft[0]//2, col_shaft[1]//2, col_shaft[2]//2),
                                     (p1[0]+off, p1[1]), (p2[0]+off, p2[1]), 1)

            # gate 노드 마커
            for fid, nid, x, y in floors:
                sp = self.cam.to_screen(x, y)
                pygame.draw.circle(self.screen, col_gate, sp, gate_r)
                pygame.draw.circle(self.screen, COL_WHITE, sp, gate_r, 1)

            # Lift 현재 위치 표시 (MOVING 시 보간)
            lift_id = lift['id']
            if hasattr(self, 'lift_ctrl'):
                elev = self.lift_ctrl.get_lift(lift_id)
                if elev and len(screen_pts) >= 2:
                    floor_idx = {'1': 0, '2': 1, '3': 2}
                    box_r = max(6, int(self.cam.scale * 0.008))

                    if elev.state == LIFT_MOVING and elev.move_from_floor and elev.move_to_floor:
                        # 보간: 출발층↔도착층 사이
                        fi = floor_idx.get(elev.move_from_floor, 0)
                        ti = floor_idx.get(elev.move_to_floor, 0)
                        if fi < len(screen_pts) and ti < len(screen_pts):
                            prog = elev.move_progress(self.sim_time)
                            fx, fy = screen_pts[fi]
                            tx, ty = screen_pts[ti]
                            ex = int(fx + (tx - fx) * prog)
                            ey = int(fy + (ty - fy) * prog)
                        else:
                            ci = floor_idx.get(elev.cur_floor, 0)
                            ex, ey = screen_pts[min(ci, len(screen_pts)-1)]
                        box_col = (255, 200, 60)
                    else:
                        ci = floor_idx.get(elev.cur_floor, 0)
                        ex, ey = screen_pts[min(ci, len(screen_pts)-1)]
                        if elev.state in (LIFT_LOADING, LIFT_UNLOADING):
                            box_col = (60, 200, 255)
                        else:
                            box_col = (180, 180, 180)

                    # 엘리베이터 박스
                    pygame.draw.rect(
                        self.screen, box_col,
                        (ex - box_r, ey - box_r, box_r*2, box_r*2),
                        border_radius=3)
                    pygame.draw.rect(
                        self.screen, COL_WHITE,
                        (ex - box_r, ey - box_r, box_r*2, box_r*2),
                        1, border_radius=3)

                    # 화물 표시
                    if elev.cargo_count > 0:
                        pygame.draw.circle(
                            self.screen, (255, 100, 100),
                            (ex, ey), max(3, box_r // 2))

                    # 라벨
                    state_short = elev.state[:4]
                    lbl = self.font_s.render(
                        f'{lift_id} F{elev.cur_floor} {state_short}',
                        True, box_col)
                    self.screen.blit(lbl, (ex + box_r + 3, ey - 6))

    def _draw_oht_network(self, ww, wh):
        """Draw OHT sub-network with polyline segments and ZCU nodes."""
        omap = self.oht_map
        seg_col, arr_col = (85, 50, 100), (110, 70, 140)
        seg_w  = max(1, int(self.cam.scale * 0.0006))
        h_size = max(4, int(self.cam.scale * 0.002))
        node_r = max(3, int(self.cam.scale * 0.004))

        for seg in omap.segments.values():
            pts = seg.path_points
            if not pts:
                continue
            spts = [self.cam.to_screen(x, y) for x, y in pts]
            xs = [p[0] for p in spts]; ys = [p[1] for p in spts]
            if min(xs) > ww or max(xs) < 0 or min(ys) > wh or max(ys) < 0:
                continue
            if len(spts) >= 2:
                pygame.draw.lines(self.screen, seg_col, False, spts, seg_w)
            mid_idx = max(1, min(int(len(spts) * 0.6), len(spts) - 1))
            draw_arrow(self.screen, arr_col,
                       spts[mid_idx - 1], spts[mid_idx],
                       width=seg_w, head=h_size)

        for nid, node in omap.nodes.items():
            sx, sy = self.cam.to_screen(node.x, node.y)
            if sx < -10 or sx > ww+10 or sy < -10 or sy > wh+10:
                continue
            if nid in omap.zcu_node_ids:
                zone_held = any(
                    z for z in omap.zcu_zones
                    if nid in {t for _, t in z.entry_segs | z.exit_segs}
                    and self.oht_env._zcu_holders.get(z.id) is not None
                )
                zcu_col = COL_ZCU_HELD if zone_held else COL_ZCU_FREE
                r = node_r + 2
                diamond = [(sx, sy-r), (sx+r, sy), (sx, sy+r), (sx-r, sy)]
                pygame.draw.polygon(self.screen, zcu_col, diamond)
                pygame.draw.polygon(self.screen, COL_WHITE, diamond, 1)
            else:
                is_port = nid in omap.port_nodes
                col = COL_PORT if is_port else COL_NODE
                pygame.draw.circle(self.screen, col, (sx, sy), node_r)
                pygame.draw.circle(self.screen, COL_WHITE, (sx, sy), node_r, 1)
                if is_port:
                    n_wait = self._port_waiting_count('OHT_A', nid)
                    if n_wait > 0:
                        wt = self.font_s.render(f'[{n_wait}]', True,
                              (255, 180, 60) if n_wait < 3 else (255, 80, 80))
                        self.screen.blit(wt, (sx + node_r + 2, sy - 6))
            # Node ID label (I 키 토글, AGV 와 공통)
            if self._show_node_ids:
                lbl = self.font_s.render(nid, True, (180, 180, 200))
                self.screen.blit(lbl, (sx + node_r + 2, sy + 2))

    def _draw_agv_network(self, ww, wh):
        """Draw AGV (AMR_A) sub-network with edges and nodes."""
        graph = self.amr_graph
        seg_col, arr_col = (90, 90, 35), (120, 120, 50)
        seg_w  = max(1, int(self.cam.scale * 0.0006))
        h_size = max(4, int(self.cam.scale * 0.002))
        node_r = max(3, int(self.cam.scale * 0.004))

        for (fn, tn), edge in graph.edges.items():
            n1, n2 = graph.nodes[fn], graph.nodes[tn]
            p1 = self.cam.to_screen(n1.x, n1.y)
            p2 = self.cam.to_screen(n2.x, n2.y)
            if max(p1[0], p2[0]) < 0 or min(p1[0], p2[0]) > ww:
                continue
            if max(p1[1], p2[1]) < 0 or min(p1[1], p2[1]) > wh:
                continue
            draw_arrow(self.screen, arr_col, p1, p2,
                       width=seg_w, head=h_size)

        port_nids = set(graph.ports.values()) if graph.ports else set()
        port_r = max(5, int(self.cam.scale * 0.008))
        for nid, node in graph.nodes.items():
            sx, sy = self.cam.to_screen(node.x, node.y)
            if sx < -10 or sx > ww+10 or sy < -10 or sy > wh+10:
                continue
            if nid in port_nids:
                # Port: larger, brighter, square shape with label
                r = port_r
                pygame.draw.rect(self.screen, COL_PORT,
                                 (sx - r, sy - r, r*2, r*2), border_radius=2)
                pygame.draw.rect(self.screen, COL_WHITE,
                                 (sx - r, sy - r, r*2, r*2), 1, border_radius=2)
                # Label + 대기 Load 수
                n_wait = self._port_waiting_count('AMR_A', nid)
                name = nid.replace('na.', '')
                if n_wait > 0:
                    txt = f'{name} [{n_wait}]'
                    col_txt = (255, 180, 60) if n_wait < 3 else (255, 80, 80)
                else:
                    txt = name
                    col_txt = COL_PORT
                lbl = self.font_s.render(txt, True, col_txt)
                self.screen.blit(lbl, (sx + r + 2, sy - 6))
            else:
                # Siding 후보면 다이아몬드 + 라벨 (tier 별 색상)
                if self._show_sidings and nid in self._siding_nodes:
                    s_r = max(5, int(self.cam.scale * 0.007))
                    if nid in self._siding_tier_a:
                        siding_col = (255, 220, 50)   # 노랑 — 항상 안전
                        tag = 'A'
                    else:
                        siding_col = (255, 140, 40)   # 주황 — 조건부
                        tag = 'B'
                    pts = [(sx, sy - s_r), (sx + s_r, sy),
                           (sx, sy + s_r), (sx - s_r, sy)]
                    pygame.draw.polygon(self.screen, siding_col, pts)
                    pygame.draw.polygon(self.screen, (255, 255, 255), pts, 1)
                    name = nid.replace('na.', '')
                    lbl = self.font_s.render(f'{name}·{tag}', True, siding_col)
                    self.screen.blit(lbl, (sx + s_r + 2, sy - 6))
                else:
                    # Cut node (port 의 entry/exit) > Branching (분기) > Corridor (회색)
                    if nid in self._cut_nodes:
                        c_r = max(5, int(self.cam.scale * 0.007))
                        c_col = (255, 80, 200)   # 분홍 — port 종속 cut
                        pygame.draw.rect(self.screen, c_col,
                                         (sx - c_r, sy - c_r, c_r*2, c_r*2),
                                         border_radius=2)
                        pygame.draw.rect(self.screen, (255, 255, 255),
                                         (sx - c_r, sy - c_r, c_r*2, c_r*2),
                                         1, border_radius=2)
                        if self._show_node_ids:
                            p = self._cut_to_port.get(nid, '?')
                            lbl = self.font_s.render(f'cut→{p}', True, c_col)
                            self.screen.blit(lbl, (sx + c_r + 2, sy - 6))
                    elif nid in self._branching_nodes:
                        b_r = max(4, int(self.cam.scale * 0.006))
                        b_col = (255, 170, 0)   # 주황
                        pts = [(sx, sy - b_r), (sx + b_r, sy),
                               (sx, sy + b_r), (sx - b_r, sy)]
                        pygame.draw.polygon(self.screen, b_col, pts)
                        pygame.draw.polygon(self.screen, (255, 255, 255), pts, 1)
                    else:
                        pygame.draw.circle(self.screen, COL_NODE, (sx, sy), node_r)
                        pygame.draw.circle(self.screen, COL_WHITE, (sx, sy), node_r, 1)
                    if self._show_node_ids:
                        name = nid.replace('na.', '')
                        lbl = self.font_s.render(name, True, (180, 180, 180))
                        self.screen.blit(lbl, (sx + node_r + 1, sy - 6))

    def _draw_agents(self, ww, wh):
        """Draw all OHT + AGV + 3DS agents."""
        # Goal lines (dashed) -먼저 그려서 차량 아래에
        self._draw_goal_lines(ww, wh)

        # 3DS shuttles (draw first -background layer)
        self._draw_3ds_agents(ww, wh)

        # OHT vehicles (longer, narrower -overhead rail)
        oht_len = max(self.oht_map.vehicle_length * self.cam.scale, 10.0)
        oht_wid = max(self.oht_map.vehicle_width  * self.cam.scale,  5.0)
        # X marker / leader chain (graph_des_v6 결과 표시) — 차량 아래에
        self._draw_oht_markers(ww, wh)
        self._draw_oht_agents(oht_len, oht_wid, ww, wh)

        # AGV vehicles (squarish -ground robot)
        agv_len = max(self.amr_graph.vehicle_length * self.cam.scale, 8.0)
        agv_wid = max(self.amr_graph.vehicle_width  * self.cam.scale, 8.0)
        self._draw_agv_agents(agv_len, agv_wid, ww, wh)

    def _draw_goal_lines(self, ww, wh):
        """Draw dashed lines from each vehicle to its goal + diamond marker."""
        r = max(5, int(self.cam.scale * 0.006))
        dim = 0.4   # 점선 밝기 비율

        # ── OHT: 경로 끝 노드가 목적지
        for a in self.oht_agents:
            if a.state == DONE or len(a.node_path) < 2:
                continue
            goal_nid = a.node_path[-1]
            gn = self.oht_map.nodes.get(goal_nid)
            if not gn:
                continue
            ax, ay = self.cam.to_screen(a.x, a.y)
            gx, gy = self.cam.to_screen(gn.x, gn.y)
            col = tuple(max(0, int(c * dim)) for c in a.color)
            draw_dashed_line(self.screen, col, (ax, ay), (gx, gy), 1, 6, 4)
            diamond = [(gx, gy-r), (gx+r, gy), (gx, gy+r), (gx-r, gy)]
            pygame.draw.polygon(self.screen, a.color, diamond, 2)

        # ── AGV (phase/push 상태 색상으로 라인 + 다이아몬드)
        # 점선 목적지 = raw_path 의 다음 의미있는 stop. 색상 align:
        #   retrieve → 다음 L state 의 node (= src port)
        #   deliver  → 다음 L state 의 node (= dst port)
        #   push     → raw_path[-1] node (= exit_port)
        # _agv_goals 는 push extension 으로 stale 갱신될 수 있어 사용 안 함.
        for a in self.agv_agents:
            if a.state == AGV_DONE or not a.raw_path:
                continue
            # path_idx 이후 첫 L state 찾기 (= 다음 retrieve/deliver dwell port)
            goal_node = None
            for i in range(a.path_idx, len(a.raw_path)):
                sid, _t = a.raw_path[i]
                if sid.startswith('L,'):
                    goal_node = sid.split(',')[1]
                    break
            # L state 없음 → push extension 또는 idle. raw_path 끝 node.
            if goal_node is None:
                last_sid, _ = a.raw_path[-1]
                parts = last_sid.split(',')
                goal_node = parts[1] if len(parts) >= 2 else None
            if not goal_node:
                continue
            node = self.amr_graph.nodes.get(goal_node)
            if not node:
                continue
            ax, ay = self.cam.to_screen(a.x, a.y)
            gx, gy = self.cam.to_screen(node.x, node.y)
            base = self._agv_phase_color(a)
            col = tuple(max(0, int(c * dim)) for c in base)
            draw_dashed_line(self.screen, col, (ax, ay), (gx, gy), 1, 6, 4)
            diamond = [(gx, gy-r), (gx+r, gy), (gx, gy+r), (gx-r, gy)]
            pygame.draw.polygon(self.screen, base, diamond, 2)

        # ── 3DS
        for fid, fd in self.s3d_floor_data.items():
            graph = fd['graph']
            for a in fd['agents']:
                goal = fd['goals'].get(a.id)
                if not goal or a.state == AGV_DONE:
                    continue
                node = graph.nodes.get(goal)
                if not node:
                    continue
                ax, ay = self.cam.to_screen(a.x, a.y)
                gx, gy = self.cam.to_screen(node.x, node.y)
                col = tuple(max(0, int(c * dim)) for c in a.color)
                draw_dashed_line(self.screen, col, (ax, ay), (gx, gy), 1, 6, 4)
                diamond = [(gx, gy-r), (gx+r, gy), (gx, gy+r), (gx-r, gy)]
                pygame.draw.polygon(self.screen, a.color, diamond, 2)

    # ── MCS 시각화 헬퍼 ────────────────────────────────────────────────────

    _AREA_TO_MCS_SYSTEM = {
        'OHT_A': 'OHT', 'AMR_A': 'AGV',
        '3DS_F1': '3DS_F1', '3DS_F2': '3DS_F2', '3DS_F3': '3DS_F3',
    }

    COL_LOAD = (255, 230, 0)   # 화물 표시 (test_graph_v6 호환, 노란 사각형)

    def _draw_load_marker(self, vid: int, sx: int, sy: int, v_len: float):
        """화물 보유 시 차량 내부에 빨간 네모 표시."""
        b = self.mcs.bindings.get(vid)
        if b is None:
            return
        # 화물을 싣고 있는 상태: LOADING 완료 후 ~ UNLOADING 완료 전
        if b.phase not in (VehicleJobState.TO_DELIVERY, VehicleJobState.UNLOADING):
            return
        # 차량 중앙에 노란 사각형 (test_graph_v6 호환)
        r = max(4, int(v_len * 0.25))
        rect = pygame.Rect(sx - r, sy - r, r * 2, r * 2)
        pygame.draw.rect(self.screen, self.COL_LOAD, rect)
        pygame.draw.rect(self.screen, COL_WHITE, rect, 1)

    def _port_waiting_count(self, area: str, node_id: str) -> int:
        """포트의 대기 Load 수 조회."""
        sys = self._AREA_TO_MCS_SYSTEM.get(area, '')
        if not sys:
            return 0
        port_key = f'{sys}:{node_id}'
        port = self.mcs.ports.get(port_key)
        if port is None:
            return 0
        return sum(1 for l in port.waiting_loads
                   if l.state == LoadState.WAITING)

    def _oht_phase_color(self, a):
        """OHT 의 phase / push 상태 기반 base color (주행 중인 경우만 적용).

        - PUSH 진행 중 (다른 차 경로 비키기 reassign)  : 보라 (180, 120, 230)
        - 본인 목적지로 주행 (MCS dispatch / job)       : 초록 (100, 220, 120)

        Push 도착 시 (state=DONE) 즉시 phase 색상 — 위 _oht_done_notified
        루프에서 _oht_pushed 에서 자동 제거된다.
        """
        if a.id in self._oht_pushed:
            return (180, 120, 230)
        return (100, 220, 120)

    def _draw_oht_agents(self, v_len, v_wid, ww, wh):
        """Draw OHT agents.

        주행 중 (MOVING/FOLLOWING) : push vs 본인 목적지 phase 색상
        그 외 (IDLE/BLOCKED/DONE)   : base color (어둡게)
        """
        for a in self.oht_agents:
            sx, sy = self.cam.to_screen(a.x, a.y)
            if sx < -50 or sx > ww+50 or sy < -50 or sy > wh+50:
                continue
            scr_ang = math.degrees(-a.theta)

            # test_graph_v6 와 동일한 로직 + MCS 모드 dwell 보정.
            # 우선순위: STOP > LOADING (v6 자체 또는 MCS dwell) > MOVING.
            from graph_des_v6 import (ACCEL as _ACCEL, CRUISE as _CRUISE,
                                       DECEL as _DECEL, STOP as _STOP,
                                       LOADING as _LOADING)
            v = getattr(a, 'vehicle', None)
            b = self.mcs.bindings.get(a.id)
            mcs_dwelling = (b is not None and b.phase in
                            (VehicleJobState.LOADING, VehicleJobState.UNLOADING))
            if v is None:
                fill = OHT_COL_IDLE_GRAY
                border = COL_DIM
            elif v.state == _STOP and not mcs_dwelling:
                reason = getattr(v, 'stop_reason', None)
                fill = {
                    None:     OHT_COL_STOP_FREE,
                    'dest':   OHT_COL_STOP_DEST,
                    'leader': OHT_COL_STOP_LEADER,
                    'zcu':    OHT_COL_STOP_ZCU,
                }.get(reason, OHT_COL_STOP_LEADER)
                border = COL_DIM
            elif v.state == _LOADING or mcs_dwelling:
                fill = OHT_COL_LOADING
                border = COL_WHITE
            else:   # ACCEL/CRUISE/DECEL/IDLE
                has_job = (v.job is not None and
                           getattr(v, 'job_state', None) in
                           ('TO_PICKUP', 'TO_DROP'))
                # MCS mode: job_state attribute 없으면 b.phase 로 판단
                if not has_job and b is not None and b.phase in (
                        VehicleJobState.TO_PICKUP, VehicleJobState.TO_DELIVERY):
                    has_job = True
                fill = OHT_COL_MOV_JOB if has_job else OHT_COL_MOV_PUSH
                border = COL_WHITE

            draw_rotated_rect(self.screen, fill, sx, sy,
                              v_len, v_wid, scr_ang, border=border, border_w=2)

            # headlight
            rad = math.radians(scr_ang)
            hx = sx + math.cos(rad) * v_len / 2
            hy = sy + math.sin(rad) * v_len / 2
            pygame.draw.circle(self.screen, COL_HEADLIGHT,
                               (int(hx), int(hy)), max(2, int(v_wid * 0.2)))

            lbl = self.font_s.render(f'H{a.id}', True, COL_WHITE)
            self.screen.blit(lbl, lbl.get_rect(center=(sx, sy)))

            # MCS Load 상태 마커
            self._draw_load_marker(a.id, sx, sy, v_len)

    def _agv_phase_color(self, a):
        """AGV 의 현재 주행 의도 기반 base color.

        색상은 "현재 무슨 임무를 수행 중인가" 를 나타냄. dwell (LOADING/UNLOADING)
        은 직전 임무의 일부 (LOADING = retrieve 의 마지막 단계, UNLOADING =
        deliver 의 마지막 단계) 로 같은 색상 유지.

        우선순위: 명시적 임무 phase > push > IDLE.

        - retrieve (TO_PICKUP / LOADING)   : 초록 (100, 220, 120)
        - deliver (TO_DELIVERY / UNLOADING) : 진파랑 (50, 100, 220)
        - push (phase=IDLE 또는 None 일 때만): 보라 (180, 120, 230)
        - IDLE                              : 하늘 (80, 180, 255)
        """
        b = self.mcs.bindings.get(a.id)
        phase = b.phase if b else None
        # 명시적 임무 phase 우선 — push 마킹과 무관
        if phase in (VehicleJobState.TO_PICKUP, VehicleJobState.LOADING):
            return (100, 220, 120)  # retrieve
        if phase in (VehicleJobState.TO_DELIVERY, VehicleJobState.UNLOADING):
            return (50, 100, 220)   # deliver
        # phase = IDLE 또는 binding 없음 → push 여부에 따라 분기
        if a.id in self._agv_pushed and a.state != AGV_DONE:
            return (180, 120, 230)  # push
        return (80, 180, 255)       # IDLE

    def _draw_oht_markers(self, ww, wh):
        """OHT vehicle 의 X marker (정지 예정점) + leader chain 화살표 표시.

        graph_des_v6 가 매 replan 시 결정한 v.x_marker_pidx / x_marker_offset 를
        세그먼트 path_points 에 보간해서 X 마크. v.leader 가 있으면 노란 선.
        """
        from graph_des_v5 import _interp_path as _ip
        for agent in self.oht_agents:
            v = agent.vehicle
            if v is None:
                continue

            # X marker
            if v.x_marker_pidx >= 0 and v.x_marker_pidx < len(v.path) - 1:
                seg = v.gmap.segment_between(
                    v.path[v.x_marker_pidx], v.path[v.x_marker_pidx + 1])
                if seg and seg.path_points:
                    mx, my, _ = _ip(seg.path_points, max(0, v.x_marker_offset))
                    ssx, ssy = self.cam.to_screen(mx, my)
                    if 0 <= ssx <= ww and 0 <= ssy <= wh:
                        xr = max(3, int(self.cam.scale * 0.003))
                        col = agent.color
                        pygame.draw.line(self.screen, col,
                                         (ssx-xr, ssy-xr), (ssx+xr, ssy+xr), 2)
                        pygame.draw.line(self.screen, col,
                                         (ssx-xr, ssy+xr), (ssx+xr, ssy-xr), 2)
                        # 차량과 X 잇는 선
                        vsx, vsy = self.cam.to_screen(v.x, v.y)
                        pygame.draw.line(self.screen, col, (vsx, vsy),
                                         (ssx, ssy), 1)

            # Leader chain (노란 선, 짧은 화살)
            if v.leader is not None:
                lx, ly = self.cam.to_screen(v.leader.x, v.leader.y)
                vx, vy = self.cam.to_screen(v.x, v.y)
                pygame.draw.line(self.screen, (255, 255, 100),
                                 (vx, vy), (lx, ly), 1)
                pygame.draw.circle(self.screen, (255, 255, 100),
                                   (lx, ly),
                                   max(3, int(self.cam.scale * 0.0015)), 1)

    def _draw_agv_agents(self, v_len, v_wid, ww, wh):
        """Draw AGV agents."""
        for a in self.agv_agents:
            sx, sy = self.cam.to_screen(a.x, a.y)
            if sx < -50 or sx > ww+50 or sy < -50 or sy > wh+50:
                continue
            scr_ang = math.degrees(-a.theta)

            base = self._agv_phase_color(a)

            if a.state == AGV_MOVING:
                vf = min(a.v / 1500.0, 1.0)
                fill = tuple(min(c + int(40*vf), 255) for c in base)
                border = COL_WHITE
            else:
                fill, border = base, COL_WHITE

            draw_rotated_rect(self.screen, fill, sx, sy,
                              v_len, v_wid, scr_ang, border=border, border_w=2)

            # headlight
            rad = math.radians(scr_ang)
            hx = sx + math.cos(rad) * v_len / 2
            hy = sy + math.sin(rad) * v_len / 2
            pygame.draw.circle(self.screen, COL_HEADLIGHT,
                               (int(hx), int(hy)), max(2, int(v_wid * 0.2)))

            display_id = a.id - 100
            lbl = self.font_s.render(f'A{display_id}', True, COL_WHITE)
            self.screen.blit(lbl, lbl.get_rect(center=(sx, sy)))

            # MCS Load 상태 마커
            self._draw_load_marker(a.id, sx, sy, v_len)

    def _draw_3ds_agents(self, ww, wh):
        """Draw 3DS shuttle agents (TAPG-based, same rendering as AGV)."""
        v_size = max(16, int(self.cam.scale * 850 * 0.001 * _3DS_SCALE * 4))  # 850mm × scale × 4
        COL_S3D_WAITING  = (255, 165, 0)
        COL_S3D_ROTATING = (255, 220, 60)
        for a in self.s3d_agents:
            sx, sy = self.cam.to_screen(a.x, a.y)
            if sx < -50 or sx > ww+50 or sy < -50 or sy > wh+50:
                continue
            scr_ang = math.degrees(-a.theta)

            if a.state == AGV_DONE:
                fill = tuple(c // 4 for c in a.color)
                border = COL_DIM
            elif a.state == AGV_WAITING:
                fill, border = COL_S3D_WAITING, COL_WHITE
            elif a.state == AGV_ROTATING:
                fill, border = COL_S3D_ROTATING, COL_WHITE
            elif a.state == AGV_MOVING:
                vf = min(a.v / 1500.0, 1.0)
                fill = tuple(min(c + int(40*vf), 255) for c in a.color)
                border = COL_WHITE
            else:  # idle
                fill = tuple(max(c - 50, 0) for c in a.color)
                border = COL_DIM

            draw_rotated_rect(self.screen, fill, sx, sy,
                              v_size, v_size, scr_ang,
                              border=border, border_w=2)

            # headlight
            rad = math.radians(scr_ang)
            hx = sx + math.cos(rad) * v_size / 2
            hy = sy + math.sin(rad) * v_size / 2
            pygame.draw.circle(self.screen, COL_HEADLIGHT,
                               (int(hx), int(hy)), max(2, int(v_size * 0.15)))

            if a.state == AGV_WAITING:
                pygame.draw.circle(self.screen, COL_S3D_WAITING,
                                   (sx, sy), int(v_size/2)+4, 2)

            display_id = a.id - 200
            lbl = self.font_s.render(f'S{display_id}', True, COL_WHITE)
            self.screen.blit(lbl, lbl.get_rect(center=(sx, sy)))

            # MCS Load 상태 마커
            self._draw_load_marker(a.id, sx, sy, v_size)

    def _draw_sidebar(self, wh):
        for b in self.all_btns:
            b.draw(self.screen, self.font_s)

        sx = MAP_W + 10
        sw = SIDE_W - 20

        # OHT count display
        r = self._oht_count_rect
        pygame.draw.rect(self.screen, (40, 25, 50), r, border_radius=3)
        lbl = self.font_s.render(f'OHT: {len(self.oht_agents)}', True,
                                 OHT_COLORS[0])
        self.screen.blit(lbl, lbl.get_rect(center=r.center))

        # AGV count display
        r = self._agv_count_rect
        pygame.draw.rect(self.screen, (40, 40, 20), r, border_radius=3)
        lbl = self.font_s.render(f'AGV: {len(self.agv_agents)}', True,
                                 AGV_COLORS[0])
        self.screen.blit(lbl, lbl.get_rect(center=r.center))

        # Scroll: _info_y 이하 영역만 scroll offset 적용. _info_y 위쪽 (버튼/
        # OHT 카운트/AGV 카운트) 은 고정. 정보가 wh 넘으면 wheel 로 내림.
        scroll_top = self._info_y
        y = scroll_top - self._sidebar_scroll
        # clip rect: scroll 영역 밖으로 텍스트 안 새도록
        old_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(MAP_W, scroll_top,
                                          SIDE_W, wh - scroll_top))

        def line(txt, col=COL_TEXT, f=None):
            nonlocal y
            fnt = f or self.font_m
            self.screen.blit(fnt.render(txt, True, col), (sx, y))
            y += fnt.get_linesize() + 2

        line('── Simulation ──', f=self.font_b)
        line(f'Time  : {self.sim_time:8.2f} s')
        line(f'Speed : {SIM_SPEED_LABELS[self.spd_idx]}')
        state = '▶ Running' if self.running else '|| Paused'
        line(state, col=(100,220,100) if self.running else (160,160,160))
        y += 6

        # OHT stats - count only
        if self.oht_agents:
            line('── OHT ──', f=self.font_b, col=OHT_COLORS[0])
            n_mov = sum(1 for a in self.oht_agents if a.state == MOVING)
            n_fol = sum(1 for a in self.oht_agents if a.state == FOLLOWING)
            n_blk = sum(1 for a in self.oht_agents if a.state == BLOCKED)
            line(f'  Mov:{n_mov} Fol:{n_fol} Blk:{n_blk} (total {len(self.oht_agents)})',
                 f=self.font_s)
            y += 4

        # AGV stats — 카운트만 표시 (per-AGV detail 은 V 키로 panel 사용)
        COL_AGV_WAITING_S = (255, 165, 0)
        line('── AGV (SIPP+TAPG) ──', f=self.font_b, col=AGV_COLORS[0])
        if self._plan_status:
            line(f'  {self._plan_status}', f=self.font_s, col=COL_DIM)
        n_mov = sum(1 for a in self.agv_agents if a.state == AGV_MOVING)
        n_rot = sum(1 for a in self.agv_agents if a.state == AGV_ROTATING)
        n_wai = sum(1 for a in self.agv_agents if a.state == AGV_WAITING)
        n_idl = sum(1 for a in self.agv_agents if a.state == AGV_IDLE)
        n_dne = sum(1 for a in self.agv_agents if a.state == AGV_DONE)
        line(f'  Mov:{n_mov} Rot:{n_rot} Wait:{n_wai}', f=self.font_s)
        line(f'  Idle:{n_idl} Done:{n_dne}', f=self.font_s)

        # Done = 물리적으로 할당 가능 (state==DONE, phase==IDLE)
        # Stuck = 작업 배정(phase in TO_PICKUP/TO_DELIVERY)되었는데
        #         실제로 움직이지 않음 (state != MOVING/ROTATING)
        n_done_phys = 0
        n_stuck = 0
        for a in self.agv_agents:
            b = self.mcs.bindings.get(a.id)
            if b is None:
                continue
            if (a.state == AGV_DONE
                    and b.phase == VehicleJobState.IDLE):
                n_done_phys += 1
            if (b.phase in (VehicleJobState.TO_PICKUP,
                            VehicleJobState.TO_DELIVERY)
                    and a.state not in (AGV_MOVING, AGV_ROTATING)):
                n_stuck += 1
        line(f'  Idle_done:{n_done_phys} Stuck:{n_stuck}',
             f=self.font_s, col=AGV_COLORS[0])
        # Pushed/cycle-pushed (= cycle resolution 적용 중인 AGV)
        n_pushed = len(getattr(self, '_agv_pushed', set()))
        n_cycle = len(getattr(self, '_cycle_push_dest', {}))
        if n_pushed or n_cycle:
            line(f'  Pushed:{n_pushed} CyclePush:{n_cycle}',
                 f=self.font_s, col=COL_AGV_WAITING_S)
        y += 6

        # Port wait queue (= top 5 most loaded)
        if self.mcs.ports:
            port_loads = [(pid, len(p.waiting_loads))
                          for pid, p in self.mcs.ports.items()]
            port_loads.sort(key=lambda x: -x[1])
            n_total_wait = sum(c for _, c in port_loads)
            line(f'── Port wait ({n_total_wait}) ──', f=self.font_b,
                 col=COL_TEXT)
            for pid, cnt in port_loads[:6]:
                if cnt == 0:
                    continue
                col = COL_TEXT if cnt < 3 else (255, 180, 60) if cnt < 6 else (255, 80, 80)
                line(f'  {pid:<14s} {cnt:3d}', f=self.font_s, col=col)
            y += 6

        # 3DS stats - count only
        if self.s3d_agents:
            line('── 3DS Shuttle ──', f=self.font_b, col=S3D_COLORS['3DS_F1'])
            n_mov_s = sum(1 for a in self.s3d_agents if a.state == AGV_MOVING)
            n_wai_s = sum(1 for a in self.s3d_agents if a.state == AGV_WAITING)
            line(f'  Mov:{n_mov_s} Wait:{n_wai_s} (total {len(self.s3d_agents)})',
                 f=self.font_s)
            y += 6

        # Elevators
        if hasattr(self, 'lift_ctrl') and self.lift_ctrl.lifts:
            COL_LIFT_MOV  = (255, 200, 60)
            COL_LIFT_XFER = (60, 200, 255)
            COL_LIFT_IDLE = (180, 180, 180)
            line('── Elevator ──', f=self.font_b, col=COL_LIFT_XFER)
            for lid in sorted(self.lift_ctrl.lifts):
                elev = self.lift_ctrl.get_lift(lid)
                if elev.state == LIFT_MOVING:
                    scol = COL_LIFT_MOV
                elif elev.state in (LIFT_LOADING, LIFT_UNLOADING):
                    scol = COL_LIFT_XFER
                else:
                    scol = COL_LIFT_IDLE
                cargo = ' [C]' if elev.cargo_count > 0 else ''
                queue = f' q={elev.queue_length}' if elev.queue_length > 0 else ''
                line(f'  {lid} F{elev.cur_floor} {elev.state:<9s}{cargo}{queue}',
                     f=self.font_s, col=scol)
            y += 6

        # MCS stats — per-system KPI
        line('── MCS (per-system) ──', f=self.font_b, col=(60, 220, 220))
        sys_colors = {
            'OHT':    OHT_COLORS[0],
            'AGV':    AGV_COLORS[0],
            '3DS_F1': S3D_COLORS['3DS_F1'],
            '3DS_F2': S3D_COLORS['3DS_F2'],
            '3DS_F3': S3D_COLORS['3DS_F3'],
        }
        sys_order = ['OHT', 'AGV', '3DS_F1', '3DS_F2', '3DS_F3']
        registered = {p.system for p in self.mcs.ports.values()}
        for sys_name in sys_order:
            if sys_name not in registered:
                continue
            col = sys_colors.get(sys_name, COL_TEXT)
            s = self.mcs.stats_summary(self.sim_time, system=sys_name)
            line(f'[{sys_name}] W:{s["waiting"]} Act:{s["active"]} '
                 f'Done:{s["completed"]}',
                 f=self.font_s, col=col)
            line(f'  Thru:{s["throughput"]:.1f}/m '
                 f'Cyc:{s["avg_cycle"]:.1f}s '
                 f'Wait:{s["avg_wait"]:.1f}s '
                 f'Util:{s["utilization"]:.0%}',
                 f=self.font_s, col=col)
        y += 6

        # ZCU
        if self.oht_map.zcu_zones:
            line('── ZCU ──', f=self.font_b)
            for z in self.oht_map.zcu_zones:
                holder = self.oht_env._zcu_holders.get(z.id)
                n_wait = len(self.oht_env._zcu_waitlists.get(z.id, []))
                name = z.id.replace('ZCU_', '')
                if holder is not None:
                    txt = f'{name}: H{holder.id:02d}'
                    if n_wait:
                        txt += f' ({n_wait}w)'
                    line(txt, col=COL_ZCU_HELD, f=self.font_s)
                else:
                    line(f'{name}: free', col=COL_ZCU_FREE, f=self.font_s)
            y += 4

        # Map info
        line('── Map ──', f=self.font_b)
        line(f'OHT nodes: {len(self.oht_map.nodes)}  segs: {len(self.oht_map.segments)}',
             col=COL_DIM, f=self.font_s)
        line(f'AGV: {len(self.amr_graph.nodes)}n {len(self.amr_graph.edges)}e '
             f'{len(self.amr_graph.move_states_raw)}s',
             col=COL_DIM, f=self.font_s)
        y += 4

        line('── Keys ──', f=self.font_b)
        for hint in ['SPACE - start/pause', 'R - reset', 'S - shuffle',
                     '+/- - sim speed', 'O/L - OHT +/-', 'N/P - AGV +/-',
                     'I - node IDs', 'X - siding nodes',
                     'drag - pan', 'wheel - zoom', 'Q - quit']:
            line(hint, col=COL_DIM, f=self.font_s)

        # area legend
        y += 6
        line('── Areas ──', f=self.font_b)
        for area, col in [('OHT_A', OHT_COLORS[0]), ('AMR_A', AGV_COLORS[0]),
                          ('3DS_F1', AREA_COLORS['3DS_F1']),
                          ('3DS_F2', AREA_COLORS['3DS_F2']),
                          ('3DS_F3', AREA_COLORS['3DS_F3'])]:
            pygame.draw.circle(self.screen, col, (sx+6, y+6), 5)
            self.screen.blit(self.font_s.render(area, True, COL_DIM),
                             (sx+15, y))
            y += 16

        # Scroll bound: 마지막 line 의 y 가 화면 위로 안 올라가게 cap
        content_height = (y + self._sidebar_scroll) - scroll_top
        visible_height = wh - scroll_top
        max_scroll = max(0, content_height - visible_height + 20)
        if self._sidebar_scroll > max_scroll:
            self._sidebar_scroll = max_scroll
        # clip 복원
        self.screen.set_clip(old_clip)

    # ── Run ──────────────────────────────────────────────────────────────────

    def _dump_state_on_halt(self, reason: str):
        """SIPP fail 등 치명적 상황에서 시뮬레이션 상태 전체를 덤프."""
        from pkl_prioritized_planner import SippFailure  # local import for clarity
        print(f'\n{"="*70}')
        print(f'[HALT] {reason}')
        print(f'  sim_time = {self.sim_time:.2f}s')
        print(f'{"="*70}')

        # TAPG 상태 분포
        n_state = {}
        for a in self.agv_agents:
            n_state[a.state] = n_state.get(a.state, 0) + 1
        n_wait_q = sum(len(q) for q in self.agv_env.wait_queues.values())
        print(f'  TAPG summary: ' + ' '.join(
            f'{k}={v}' for k, v in sorted(n_state.items())) +
            f'  wait_queue_nodes={len(self.agv_env.wait_queues)}'
            f' waiters={n_wait_q}')

        print(f'  AGV full status:')
        for a in self.agv_agents:
            nid = a.raw_path[-1][0].split(',')[1] if a.raw_path else '?'
            cur_sid = a.raw_path[a.path_idx][0] if a.path_idx < len(a.raw_path) else '(past end)'
            goal = self._agv_goals.get(a.id)
            b = self.mcs.bindings.get(a.id)
            phase = b.phase.value if b else '?'
            load_id = b.load.load_id if (b and b.load) else None
            load_dst = b.load.dst_port if (b and b.load) else None
            pushed = 'PUSHED' if a.id in self._agv_pushed else ''
            arrival = self._agv_arrival_idx.get(a.id)
            # WAITING 상태면 cross-pred 힌트
            waits = ''
            if a.state == AGV_WAITING and a.path_idx < len(a.raw_path):
                cur_t = a.raw_path[a.path_idx][1]
                nk = self.agv_env._nk(cur_sid, a.id, cur_t)
                if self.agv_env.G.has_node(nk):
                    cross = [f'A{p[1]}' for p in self.agv_env.G.predecessors(nk)
                             if p[1] != a.id]
                    if cross:
                        waits = f' waits={cross[:5]}'
            print(f'    A{a.id}: end={nid} cur_sid={cur_sid}  goal={goal}  '
                  f'phase={phase}  load={load_id}  load_dst={load_dst}  '
                  f'state={a.state}  idx={a.path_idx}/{len(a.raw_path)}  '
                  f'claim={a.claim_idx}  arrival_idx={arrival}  {pushed}{waits}')

        # DAG 크기
        print(f'  DAG: {self.agv_env.G.number_of_nodes()} nodes, '
              f'{self.agv_env.G.number_of_edges()} edges')
        print(f'{"="*70}\n')

    def _print_event_stats(self):
        """벤치마크용: wall-clock vs sim-time (= 배속) + 시스템별 DES 이벤트 수.
        한 줄씩 '[EVTSTAT] key=value' 형식 (= benchmark 파서가 grep)."""
        import time as _wt
        wall = _wt.perf_counter() - getattr(self, '_wall_start', _wt.perf_counter())
        wall = max(wall, 1e-9)
        sim_t = self.sim_time
        speedup = sim_t / wall   # realtime 배속

        agv_evt = getattr(self.agv_env, 'event_count', 0)
        oht_evt = getattr(getattr(self.oht_env, 'des', None), 'event_count', 0)
        s3d_evt = sum(getattr(fd['env'], 'event_count', 0)
                      for fd in self.s3d_floor_data.values())
        mcs_evt = getattr(self, '_mcs_total_events', 0)
        lift_evt = getattr(self, '_lift_total_events', 0)
        total_evt = agv_evt + oht_evt + s3d_evt + mcs_evt + lift_evt

        print(f'\n{"="*70}')
        print('[EVENT STATS]')
        print(f'[EVTSTAT] sim_time_s={sim_t:.1f}')
        print(f'[EVTSTAT] wall_s={wall:.1f}')
        print(f'[EVTSTAT] speedup_x={speedup:.1f}')
        print(f'[EVTSTAT] evt_agv={agv_evt}')
        print(f'[EVTSTAT] evt_oht={oht_evt}')
        print(f'[EVTSTAT] evt_3ds={s3d_evt}')
        print(f'[EVTSTAT] evt_mcs={mcs_evt}')
        print(f'[EVTSTAT] evt_lift={lift_evt}')
        print(f'[EVTSTAT] evt_total={total_evt}')
        print(f'[EVTSTAT] evt_per_simsec={total_evt/max(sim_t,1e-9):.2f}')
        print(f'{"="*70}')

    def _print_movement_summary(self):
        """AGV별 retrieve/deliver/push 이력 + 카운트 + KPI 요약을 콘솔에 출력."""
        # 0) 모드 헤더
        mode = 'PORTS+SIDINGS' if self._use_sidings else 'PORTS-ONLY (baseline)'
        print(f'\n{"="*70}')
        print(f'[MODE] park pool = {mode}, '
              f'AGV={self._n_agv}, CONWIP={self._conwip_agv}')

        # 1) 전체 이벤트 시간순 출력
        print(f'\n{"="*70}')
        print(f'[MOVEMENT EVENTS] sim_time={self.sim_time:.2f}s '
              f'total={len(self._agv_movement_log)}')
        print(f'  {"t":>9}  {"aid":>4}  {"type":>8}  {"src":>10} → {"dst":<10}  load')
        for ev in self._agv_movement_log:
            lid = '' if ev['load_id'] is None else ev['load_id']
            print(f"  {ev['t']:>9.2f}  A{ev['aid']:<3}  {ev['type']:>8}  "
                  f"{ev['src']:>10} → {ev['dst']:<10}  {lid}")
        # 2) AGV별 카운트
        counts: dict = {}
        for ev in self._agv_movement_log:
            d = counts.setdefault(ev['aid'], {'retrieve': 0, 'deliver': 0, 'push': 0})
            d[ev['type']] = d.get(ev['type'], 0) + 1
        print(f'\n[MOVEMENT SUMMARY]')
        print(f'  {"AID":>6}  {"retrieve":>9} {"deliver":>9} {"push":>9}  total')
        for aid in sorted(counts):
            c = counts[aid]
            tot = c['retrieve'] + c['deliver'] + c['push']
            print(f'  A{aid:<5} {c["retrieve"]:>9} {c["deliver"]:>9} {c["push"]:>9}  {tot}')

        # 3a) SIPP plan 시간 통계
        if self._plan_dur_log:
            durs = [r['dur'] for r in self._plan_dur_log]
            durs_sorted = sorted(durs)
            n = len(durs)
            p50 = durs_sorted[n // 2]
            p95 = durs_sorted[min(n - 1, int(n * 0.95))]
            p99 = durs_sorted[min(n - 1, int(n * 0.99))]
            mx = max(durs)
            avg = sum(durs) / n
            print(f'\n[PLAN TIME] n={n} avg={avg:.3f}s p50={p50:.3f}s '
                  f'p95={p95:.3f}s p99={p99:.3f}s max={mx:.3f}s')
            slow = sorted([r for r in self._plan_dur_log if r['dur'] >= 0.10],
                          key=lambda r: -r['dur'])[:10]
            if slow:
                print(f'  -- top {len(slow)} slow plans (≥0.10s) --')
                print(f"  {'t':>9}  {'dur':>7}  {'ag':>3} {'cs':>5} "
                      f"{'pathT':>5} {'pathM':>5}  pending → goals")
                for r in slow:
                    print(f"  {r['t']:>9.2f}  {r['dur']:>7.3f}  "
                          f"{r['n_agents']:>3} {r['cs']:>5} "
                          f"{r['path_len_total']:>5} {r['path_len_max']:>5}  "
                          f"{r['pending']} → {r['goals']}")

        # 3) Dwell 시간 검증 — 측정값 vs 설정값
        cfg = self.mcs.dwell_time
        print(f'\n[DWELL VERIFICATION] configured={cfg:.2f}s '
              f'measured_count={len(self._agv_dwell_log)}'
              f' (open={len(self._agv_dwell_open)})')
        if self._agv_dwell_log:
            for kind in ('loading', 'unloading'):
                durs = [d['duration'] for d in self._agv_dwell_log
                        if d['kind'] == kind]
                if durs:
                    mn, mx = min(durs), max(durs)
                    avg = sum(durs) / len(durs)
                    n_low  = sum(1 for d in durs if d < cfg - 0.01)
                    n_high = sum(1 for d in durs if d > cfg + 0.01)
                    flag = '' if (n_low == 0 and n_high == 0) else \
                           f'  ⚠ {n_low} below, {n_high} above cfg'
                    print(f'  {kind:>10s}: n={len(durs):3d}  '
                          f'min={mn:.3f}s  max={mx:.3f}s  avg={avg:.3f}s{flag}')
            # 자세한 이벤트 (앞부분 8건만, 너무 많을 때)
            print(f'  -- recent dwell events (last 10) --')
            for d in self._agv_dwell_log[-10:]:
                delta = d['duration'] - cfg
                mark = '✓' if abs(delta) < 0.01 else f'Δ{delta:+.3f}s'
                print(f"    A{d['aid']} {d['kind']:>9s} @ {d['port']:<10s} "
                      f"t=[{d['t_start']:.2f}, {d['t_end']:.2f}]  "
                      f"dur={d['duration']:.3f}s  {mark}")

        # 4) KPI 비교 블록 — 두 모드 (port-only vs port+siding) 직접 비교용
        print(f'\n[KPI SUMMARY] mode={mode}')
        T = max(self.sim_time, 1e-9)
        n_retrieve = sum(1 for ev in self._agv_movement_log if ev['type'] == 'retrieve')
        n_deliver  = sum(1 for ev in self._agv_movement_log if ev['type'] == 'deliver')
        n_push     = sum(1 for ev in self._agv_movement_log if ev['type'] == 'push')
        n_total    = n_retrieve + n_deliver + n_push
        print(f'  sim_time             = {T:.2f}s ({T/60:.2f} min)')
        print(f'  deliveries (완료)    = {n_deliver}')
        print(f'  throughput           = {n_deliver / T * 60:.2f} loads/min')
        print(f'  retrieve / deliver / push = {n_retrieve} / {n_deliver} / {n_push}')
        push_ratio = (n_push / n_total * 100) if n_total else 0.0
        print(f'  push ratio           = {push_ratio:.1f}%  '
              f'(push / total movements)')
        # Push 의 phase attribution
        n_push_ret = sum(1 for ev in self._agv_movement_log
                         if ev['type'] == 'push' and ev.get('attr_phase') == 'retrieve')
        n_push_del = sum(1 for ev in self._agv_movement_log
                         if ev['type'] == 'push' and ev.get('attr_phase') == 'deliver')
        n_push_idle = sum(1 for ev in self._agv_movement_log
                          if ev['type'] == 'push' and ev.get('attr_phase') == 'idle')
        if n_push > 0:
            print(f'  push attribution     = retrieve:{n_push_ret} / '
                  f'deliver:{n_push_del} / idle:{n_push_idle}')

        # 평균 push 거리 (Euclidean, mm)
        push_dists = []
        for ev in self._agv_movement_log:
            if ev['type'] != 'push': continue
            s = self.amr_graph.nodes.get(ev['src'])
            d = self.amr_graph.nodes.get(ev['dst'])
            if s and d:
                push_dists.append(((s.x - d.x) ** 2 + (s.y - d.y) ** 2) ** 0.5)
        if push_dists:
            avg_pd = sum(push_dists) / len(push_dists)
            print(f'  avg push distance    = {avg_pd/1000:.2f}m  '
                  f'(min={min(push_dists)/1000:.2f}m, max={max(push_dists)/1000:.2f}m)')
        else:
            print(f'  avg push distance    = N/A (no pushes)')

        # 평균 deliver 거리 (참고용)
        del_dists = []
        for ev in self._agv_movement_log:
            if ev['type'] != 'deliver': continue
            s = self.amr_graph.nodes.get(ev['src'])
            d = self.amr_graph.nodes.get(ev['dst'])
            if s and d:
                del_dists.append(((s.x - d.x) ** 2 + (s.y - d.y) ** 2) ** 0.5)
        if del_dists:
            avg_dd = sum(del_dists) / len(del_dists)
            print(f'  avg deliver distance = {avg_dd/1000:.2f}m')

        # Replan / SIPP 통계 (이미 PLAN TIME 출력했지만 KPI 섹션에도 한 줄 요약)
        n_replan = len(self._plan_dur_log)
        n_fail = sum(1 for r in self._plan_dur_log if r['status'] != 'OK')
        if n_replan:
            avg_dur = sum(r['dur'] for r in self._plan_dur_log) / n_replan
            print(f'  replan count / fail  = {n_replan} / {n_fail}  '
                  f'(avg dur {avg_dur*1000:.1f}ms)')

        # Push 분포: port vs siding
        siding_set = set(self._siding_tier_a)
        push_to_port    = sum(1 for ev in self._agv_movement_log
                              if ev['type'] == 'push' and ev['dst'] not in siding_set)
        push_to_siding  = sum(1 for ev in self._agv_movement_log
                              if ev['type'] == 'push' and ev['dst'] in siding_set)
        if n_push > 0:
            print(f'  push → port / siding = {push_to_port} / {push_to_siding}  '
                  f'(siding share {push_to_siding/n_push*100:.1f}%)')

        # ── 작업별 retrieve/deliver 시간 + 시간당 처리량/생성량 ──────────
        warmup_t = getattr(self, '_kpi_start_t', 0.0)
        completed_all = list(getattr(self.mcs.kpi, 'completed_loads', []))
        completed = [L for L in completed_all
                     if L.t_completed >= warmup_t and L.t_assigned > 0]
        if completed:
            ret_times = [L.t_pickup_end - L.t_assigned for L in completed
                         if L.t_pickup_end >= L.t_assigned]
            del_times = [L.t_completed - L.t_pickup_end for L in completed
                         if L.t_completed >= L.t_pickup_end]
            if ret_times:
                print(f'  retrieve time (avg)  = {sum(ret_times)/len(ret_times):.1f}s '
                      f'(min={min(ret_times):.1f}, max={max(ret_times):.1f})')
            if del_times:
                print(f'  deliver time (avg)   = {sum(del_times)/len(del_times):.1f}s '
                      f'(min={min(del_times):.1f}, max={max(del_times):.1f})')
            cycle_times = [L.t_completed - L.t_created for L in completed]
            if cycle_times:
                print(f'  cycle time (avg)     = {sum(cycle_times)/len(cycle_times):.1f}s '
                      f'(create→complete)')

        # 시간당 처리량/생성량
        # Arrival rate = *모든* 생성된 load 빈도 (= waiting + active + completed)
        # Throughput  = 완료된 load 빈도.
        # 둘은 다른 metric — arrival > throughput 이 정상 (= 미완료 잔여).
        all_loads_created = []
        # 1) 완료된 loads
        all_loads_created.extend(completed_all)
        # 2) 현재 active loads (= MCS bindings 안)
        for vid, bd in self.mcs.bindings.items():
            if bd.load and bd.load not in all_loads_created:
                all_loads_created.append(bd.load)
        # 3) 대기 중인 loads (= ports 의 waiting_loads)
        for port in self.mcs.ports.values():
            for L in port.waiting_loads:
                if L not in all_loads_created:
                    all_loads_created.append(L)
        created_times = [L.t_created for L in all_loads_created
                         if L.t_created >= warmup_t]
        completed_times = [L.t_completed for L in completed]
        if T > 60:
            T_eff = T - warmup_t
            arrival_rate = len(created_times) / T_eff * 60.0
            throughput_rate = len(completed_times) / T_eff * 60.0
            print(f'  arrival rate         = {arrival_rate:.2f} loads/min '
                  f'({len(created_times)} created in {T_eff/60:.1f}min)')
            print(f'  effective throughput = {throughput_rate:.2f} loads/min '
                  f'({len(completed_times)} delivered)')

        # ── KPI #1: per-OD M state visits ─────────────────────────────
        move_visits = getattr(self, '_kpi_move_visits', {})
        if move_visits:
            # OD 별 phase 별 총 visit 수
            od_sum = {}   # (od, phase) -> total visit count
            for (od, ph, sid), cnt in move_visits.items():
                od_sum[(od, ph)] = od_sum.get((od, ph), 0) + cnt
            print(f'  per-OD move visits   = {len(od_sum)} (od, phase) groups, '
                  f'{len(move_visits)} unique (od, phase, M-sid)')
            # 파일로 저장 (= dump 가 자세함)
            import os
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'logs')
            os.makedirs(log_dir, exist_ok=True)
            fpath = os.path.join(log_dir, 'kpi_od_visits.txt')
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f'# OD move state visits (warmup_t={warmup_t:.1f}s)\n')
                f.write(f'# src dst phase M_state count\n')
                for (od, ph, sid), cnt in sorted(move_visits.items()):
                    f.write(f'{od[0]} {od[1]} {ph} {sid} {cnt}\n')
            print(f'  ↳ saved → {fpath}')

        # ── KPI #2: per-S blocked wait time ───────────────────────────
        # 진행 중 wait 도 누적
        for aid, (sid, t0) in self._agv_wait_start.items():
            self._kpi_s_wait[sid] = self._kpi_s_wait.get(sid, 0.0) + (T - t0)
        if self._kpi_s_wait:
            total_wait = sum(self._kpi_s_wait.values())
            top = sorted(self._kpi_s_wait.items(), key=lambda x: -x[1])[:5]
            print(f'  total S-wait time    = {total_wait:.1f}s '
                  f'({len(self._kpi_s_wait)} S states blocked)')
            print(f'  top 5 blocked S:')
            for sid, t in top:
                print(f'    {sid:40s} {t:.1f}s')
            # 파일 저장
            import os
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'logs')
            os.makedirs(log_dir, exist_ok=True)
            fpath = os.path.join(log_dir, 'kpi_s_wait.txt')
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f'# Blocked wait time per S state (warmup_t={warmup_t:.1f}s)\n')
                f.write(f'# S_state wait_seconds\n')
                for sid, t in sorted(self._kpi_s_wait.items(),
                                      key=lambda x: -x[1]):
                    f.write(f'{sid} {t:.2f}\n')
            print(f'  ↳ saved → {fpath}')

        print(f'{"="*70}\n')

    def run(self):
        from pkl_prioritized_planner import SippFailure
        import time as _wt
        self._wall_start = _wt.perf_counter()
        try:
            while True:
                try:
                    if self._headless:
                        # Headless: no clock tick (= no idle wait), fixed dt
                        dt = 1.0 / FPS
                    else:
                        dt = self.clock.tick(FPS) / 1000.0
                    self.handle_events()
                    self.update(dt)
                    self.render()
                except SippFailure as e:
                    if self._lenient:
                        # 실패한 dispatch 만 취소 — 다른 AGV 들은 계속 진행.
                        # 영향: 그 load 는 다음 _do_assign tick 에서 재시도.
                        print(f'[LENIENT] SippFailure ignored: {e}')
                        continue
                    self._dump_state_on_halt(f'SIPP FAIL: {e}')
                    import traceback
                    traceback.print_exc()
                    self._dump_agv_status()
                    self._dump_replan_history()
                    self._print_movement_summary()
                    pygame.quit()
                    import sys
                    sys.exit(1)
                except DeadlockDetected as e:
                    self._dump_state_on_halt(f'DEADLOCK: {e}')
                    self._dump_agv_status()
                    self._dump_replan_history()
                    self._print_event_stats()
                    self._print_movement_summary()
                    pygame.quit()
                    import sys
                    sys.exit(2)
        except (KeyboardInterrupt, SystemExit):
            self._print_movement_summary()
            raise


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--oht', type=int, default=0, help='Number of OHT vehicles')
    ap.add_argument('--agv', type=int, default=7, help='Number of AGV vehicles')
    ap.add_argument('--s3d', type=int, default=0, help='Number of 3DS shuttles per floor')
    ap.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    ap.add_argument('--conwip', type=int, default=0,
                    help='CONWIP target for AGV system (0=off, default Poisson)')
    ap.add_argument('--no-siding', action='store_true',
                    help='Disable Tier-A sidings; use ports only for push (KPI baseline)')
    ap.add_argument('--max-time', type=float, default=0.0,
                    help='Auto-stop sim and dump KPIs at this sim_time (0=run forever)')
    ap.add_argument('--recipe-file', type=str, default=None,
                    help='Recipe JSON 파일 경로 (다중-시스템 chain 작업)')
    ap.add_argument('--recipe-conwip', type=int, default=0,
                    help='Recipe 단위 CONWIP target (0=off)')
    ap.add_argument('--recipe-rate', type=float, default=0.0,
                    help='Recipe Poisson 도착률 (loads/min, 0=off)')
    ap.add_argument('--oht-conwip', type=int, default=0,
                    help='OHT 단독 CONWIP target (push 로직 검증용; recipe 와 배타)')
    ap.add_argument('--lenient', action='store_true',
                    help='SIPP fail 시 halt 안 함 — dispatch 만 취소하고 다음 tick 에 재시도')
    ap.add_argument('--json', type=str, default=None,
                    help=f'Layout JSON 경로 (default: {JSON_FILE})')
    ap.add_argument('--pkl', type=str, default=None,
                    help=f'AMR collision pkl 경로 (default: {AMR_PKL}). '
                         '파일 없으면 gen_songdo_pkl.py 로 자동 생성.')
    ap.add_argument('--sidings', type=str, default=None,
                    help='Siding 노드 list JSON 파일. 지정 시 외부 정의된 '
                         'cut-safe siding set 사용 (compute pendant heuristic 대체).')
    ap.add_argument('--fromto', type=str, default=None,
                    help='FromTo CSV 경로 (지정 시 OD-Poisson 모드, recipe 와 배타).')
    ap.add_argument('--fromto-scale', type=float, default=1.0,
                    help='FromTo λ 전체 스케일 (1.0=원본). 처리율 sweep 에 사용.')
    ap.add_argument('--planner', type=str, default='sipp',
                    choices=['sipp', 'coarse'],
                    help='Planner type. sipp = constraint-aware SIPP (default). '
                         'coarse = unconstrained shortest + segment-level lock '
                         '(unidirectional layouts e.g. songdo).')
    ap.add_argument('--coarse-debug', action='store_true',
                    help='Coarse mode: enable [CLAIM/CUT/REVOKE/BUG/OVERLAP] log. '
                         'Slows sim significantly. Default off.')
    ap.add_argument('--dwell', type=float, default=3.0,
                    help='LOAD/UNLOAD dwell time (sec). Default 3.0. '
                         'Songdo layout: 10.0.')
    ap.add_argument('--warmup', type=float, default=0.0,
                    help='Warmup period (sec). KPI 측정은 warmup 후부터 시작. '
                         'Default 0 (= 전체 sim 측정).')
    ap.add_argument('--headless', action='store_true',
                    help='Render 완전 skip + clock tick 무시 -> 장기 sim 가속.')
    ap.add_argument('--profile-frames', type=float, default=0.0,
                    help='Per-frame section profiler. 값 N>0 -> N ms 초과 frame log '
                         '(예: 50). 0=off. 끊김 원인 진단용.')
    args = ap.parse_args()

    # --profile-frames 모드: 모든 일반 print 억제, SLOW-FRAME 만 console.
    if args.profile_frames > 0:
        import sys as _sys, os as _os
        os.makedirs('logs', exist_ok=True)
        _real_stdout = _sys.stdout
        _log_fh = open('logs/sim_console.log', 'w', encoding='utf-8',
                       buffering=1)
        class _ProfileStdout:
            """일반 print → file. [SLOW-FRAME … 만 console."""
            def write(self, s):
                _log_fh.write(s)
                if 'SLOW-FRAME' in s:
                    _real_stdout.write(s)
                    _real_stdout.flush()
            def flush(self):
                _log_fh.flush(); _real_stdout.flush()
        _sys.stdout = _ProfileStdout()

    # Layout / pkl override
    if args.json:
        JSON_FILE = args.json
        print(f'[LAYOUT] JSON: {JSON_FILE}')
    if args.pkl:
        AMR_PKL = args.pkl
    else:
        # JSON 만 override 됐으면 pkl 도 같은 dir 의 동명 .pkl 로 추론
        if args.json:
            AMR_PKL = os.path.splitext(args.json)[0] + '.pkl'
    if not os.path.exists(AMR_PKL):
        print(f'[PKL] {AMR_PKL} 없음 — gen_songdo_pkl.py 로 자동 생성')
        import subprocess
        gen_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'gen_songdo_pkl.py')
        r = subprocess.run(
            ['python', gen_script, '--json', JSON_FILE, '--out', AMR_PKL],
            capture_output=True, text=True)
        if r.returncode != 0:
            print('[PKL-GEN] STDOUT:', r.stdout[-500:])
            print('[PKL-GEN] STDERR:', r.stderr[-500:])
            sys.exit(1)
        print(f'[PKL-GEN] 생성 완료: {AMR_PKL}')

    # Seed 설정 및 저장
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 999999)
    random.seed(seed)
    print(f'Random seed: {seed}  (reproduce with --seed {seed})')

    print('Loading maps...')
    # OHTMap — JSON 에 OHT_A area 없으면 stub wrapper. n_oht=0 시 사용 안 함.
    try:
        oht_map = OHTMap(JSON_FILE, area='OHT_A')
        print(f'  OHT: {len(oht_map.nodes)} nodes, {len(oht_map.segments)} segments')
    except Exception as e:
        print(f'  OHT: 로딩 실패 (layout 에 OHT 영역 없음): {e}')
        # Stub — 빈 collection 들로 OHT-iter 코드 무영향 통과시킴
        class _EmptyOHTMap:
            nodes = {}
            segments = {}
            adj = {}
            port_nodes = set()
            zcu_zones = []
            zcu_node_ids = set()
            bbox = (0, 0, 0, 0)
            vehicle_length = 2000.0
            vehicle_width = 1300.0
            h_min = 2200.0
            accel = 500.0
            decel = 500.0
            max_brake = 1.0
            area = 'OHT_A'
            def bfs_path(self, *a, **k): return None
            def nearby_nodes(self, *a, **k): return set()
        oht_map = _EmptyOHTMap()

    amr_graph = PklMapGraph(AMR_PKL)
    print(f'  AGV: {len(amr_graph.nodes)} nodes, {len(amr_graph.edges)} edges')
    print(f'  AGV states: {len(amr_graph.move_states_raw)} move+rot, '
          f'{len(amr_graph.stop_states_raw)} stop')

    # Apply area offsets — KaistTB 만 area-multi, 송도 같은 single-area 는 무영향
    dx_oht, dy_oht = _area_offset('OHT_A')
    for node in oht_map.nodes.values():
        node.x += dx_oht
        node.y += dy_oht
    for seg in oht_map.segments.values():
        if seg.path_points:
            seg.path_points = [(px + dx_oht, py + dy_oht)
                               for px, py in seg.path_points]

    dx_amr, dy_amr = _area_offset('AMR_A')
    for node in amr_graph.nodes.values():
        node.x += dx_amr
        node.y += dy_amr

    # Sidings override 로드 (--sidings JSON path)
    sidings_override = None
    if args.sidings:
        with open(args.sidings, 'r', encoding='utf-8') as _f:
            sidings_override = json.load(_f)
        print(f'[SIDING-OVERRIDE] {args.sidings} → {len(sidings_override)} nodes')

    print(f'Starting combined simulator: {args.oht} OHT + {args.agv} AGV + {args.s3d} 3DS/floor')
    sim = CombinedSimulator(oht_map, amr_graph,
                            n_oht=args.oht, n_agv=args.agv, n_s3d=args.s3d,
                            conwip_agv=args.conwip,
                            use_sidings=not args.no_siding,
                            max_sim_time=args.max_time,
                            recipe_file=args.recipe_file,
                            recipe_conwip=args.recipe_conwip,
                            recipe_rate=args.recipe_rate,
                            conwip_oht=args.oht_conwip,
                            lenient=args.lenient,
                            sidings_override=sidings_override,
                            fromto_csv=args.fromto,
                            fromto_scale=args.fromto_scale,
                            planner_type=args.planner,
                            coarse_debug=args.coarse_debug,
                            dwell_time=args.dwell,
                            warmup_time=args.warmup,
                            headless=args.headless,
                            profile_frames_ms=args.profile_frames)
    sim.run()
