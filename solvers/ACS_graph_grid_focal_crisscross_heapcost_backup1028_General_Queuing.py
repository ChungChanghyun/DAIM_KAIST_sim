import numpy as np
import openpyxl
from enum import Enum
from queue import PriorityQueue
import time
import math
import random
import heapq
import copy
import heapq
import json
import pickle
from bisect import bisect
from collections import Counter
from itertools import chain
from typing import Dict, Iterable, Tuple, Any, Optional
from collections import deque, defaultdict
import networkx as nx
class LoadStatus(Enum):
    Empty = 1
    Full = 2


class Direction(Enum):
    Right = 0
    Left = 1
    Down = 2
    Up = 3
    Unknown = 9


class PointType(Enum):
    Unused = 0
    Normal = 1
    Queue = 2
    Junction = 3


class Astar_tuple:
    def __init__(self, tuple, priority, count):
        self.priority = priority
        self.tuple = tuple
        self.count = count

    def __lt__(self, other):
        return self.priority < other.priority



class SIPP_tuple:
    def __init__(self, state, cost = float('inf'),time = float('inf'),interval = [0,float('inf')]):
        self.state = state
        self.cost = cost
        self.interval = interval
        self.time = time
    def __lt__(self, other):
        return self.cost < other.cost

class State:
    def __init__(self,type,input1,input2=None,input3=None): #type이 stop이면 input 1 = node, input 2 = angle, drive input 1,2는 각각 prev next node,
        if type == 'Stop':
            self.id = 'S,'+str(input1)+','+str(input2)
            self.cost = 0
        if type == 'Move':
            self.id = 'M,'+str(input1)+','+str(input2)
            self.cost = 1
        if type == 'Rotate':
            self.id = 'R,'+str(input1)+','+str(input2)+','+str(input3)
            Angle = abs(int(input2)-int(input3))
            if Angle == 270:
                Angle = Angle - 180
            self.cost = Angle/90
        self.offset = []
        self.sr = 0
        self.sc = 0
        self.interval_list = [(0,float('inf'))]
        self.type = None
        self.next_state = []
        self.affect_state = [] #물리적인 경로가 중복되는 list
        self.rsv_veh_list = []
        self.rsv_time_table = []

    def split_interval(self, t, next_t):
        """
        Function to generate safe-intervals
        """
        if t != next_t:
            temp_list = []
            for interval in self.interval_list:
                if next_t == float('inf'):
                    if t <= interval[0]:
                        continue
                        # self.interval_list.remove(interval)
                    elif t >= interval[1]:
                        # continue
                        temp_list.append(interval)
                    else:
                        temp_list.append((interval[0],t))
                        # self.interval_list.remove(interval)
                        # self.interval_list.append((interval[0], t))
                else:
                    if t <= interval[0]:
                        if interval[0] < next_t:
                            # self.interval_list.remove(interval)
                            if interval[1] >= next_t:
                                temp_list.append((next_t, interval[1]))
                        else:
                            temp_list.append(interval)
                    elif next_t >= interval[1]:
                        if interval[1] >= t:
                            temp_list.append((t,interval[1]))
                        else:
                            temp_list.append(interval)
                    elif t >interval[0] and next_t <interval[1]:
                        # self.interval_list.remove(interval)
                        temp_list.append((interval[0],t))
                        temp_list.append((next_t,interval[1]))
                    elif t < interval[0] and next_t > interval[1]:
                        continue
                        # self.interval_list.remove(interval)
            self.interval_list = temp_list
            self.interval_list.sort()


class Point:
    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.type = PointType.Unused
        self.rsv_veh_list = []
        self.dir_accessible_empty = {Direction.Left: 0, Direction.Right: 0, Direction.Down: 0,
                                     Direction.Up: 0}  # 0: 사용 가능, 1: 사용 불가
        self.dir_accessible_full = {Direction.Left: 0, Direction.Right: 0, Direction.Down: 0, Direction.Up: 0}
        self.dir_usable_empty = {Direction.Left: 0, Direction.Right: 0, Direction.Down: 0, Direction.Up: 0}
        self.dir_usable_full = {Direction.Left: 0, Direction.Right: 0, Direction.Down: 0, Direction.Up: 0}
        self.prev_point = None
        self.fixed_route = False

    def block_link(self, dir):
        # self.dir_usable_empty[dir] = 1
        self.dir_usable_full[dir] = 1

    def release_link(self, dir):
        self.dir_usable_empty[dir] = self.dir_accessible_empty[dir]
        self.dir_usable_full[dir] = self.dir_accessible_full[dir]

    def get_free_link(self, dir):
        # self.dir_usable_empty[dir] = 0
        self.dir_usable_full[dir] = 0

    @property
    def cp_name(self):
        return f'cp_{self.r + 1}_{self.c + 1}'


def get_idx_from_cp(cp_name: str):
    _, r, c = cp_name.split('_')
    r = int(r) - 1
    c = int(c) - 1
    return r, c


def reverse_direction(direction):
    if direction == Direction.Right:
        return Direction.Left
    elif direction == Direction.Left:
        return Direction.Right
    elif direction == Direction.Down:
        return Direction.Up
    elif direction == Direction.Up:
        return Direction.Down
    else:
        return Direction.Unknown


class AGV:
    def __init__(self, veh_id):
        self.id = veh_id  # Vehicle id
        self.blocking_veh_id = None
        self.planned_path = []
        self.fixed_path = []  # 확정 경로 (list)
        self.cur_state = None
        self.dst_state_list = []
        self.prev_state = None
        self.cur_loc = None  # 현재 위치 (string)
        self.dst_loc = None  # 목적지 (string)
        self.waypoint = None  # waypoint(string)
        self.direction = Direction.Unknown
        self.fixed_path_state = []
        self._load_status = 'nload'  # shelf를 들고 있는지 여부 {'load', 'nload'}
        # Python에서 현재는 쓰지 않음
        self.hold = 0  # 확정 경로는 있는데 현재 갈 수 없는 상황일 때
        self.is_return = 0  # 반환여부  (*) For what?
        self.return_loc = None  # 반환지점(처음 가져간 지점) (*) For what?
        self.cur_loc_AutoMod = None
        self.dst_loc_AutoMod = None
        self.return_loc_AutoMod = None
        self.pushed = False
        self.blocked = False

    @property
    def empty(self):
        if self._load_status == 'nload':
            return True
        else:
            return False

    @property
    def full(self):
        if self._load_status == 'load':
            return True
        else:
            return False

    @property
    def arrived(self):
        if self.cur_loc == self.dst_loc:
            return True
        else:
            return False


class ACS_focal:
    def __init__(self, filename,pickle = None,fromto_rate = 0.05):
        self.direction_mode = 'BI'
        self.fromto_rate = fromto_rate
        self.lam = {}
        self.n_dir = 4  # 상하좌우: 4개 1:우 2:좌 3:하 4:상

        if filename[0] == 'u':
            self.direction_mode = 'UNI'
        self.points = {}
        self.Move_state = {}
        self.Stop_state = {}
        self.Rotate_state = {}
        self.link_list = {}  # link list는 from,to node들을 key로(from,to)로 조회했을 시 해당 경로가 available한 지 여부: available = 0
        self.mode = 0  # 0은 일반 Grid 형태, 1은 Warehouse
        if filename[0] == 'w':
            self.mode = 1
        # self.reset()  # reset vehicle positio
        self.angle = [0,180,270,90]
        self.row_col_dif = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # direction 표기를 위해서 쉽게 (우,좌,하,상)
        self.adj_array = []  # adjacency array는 node가 있을 경우 해당 value, 없을 경우 -1
        self.direction_list = ['r', 'l', 'd', 'u']
        self.AGVs = {}
        self.push_location =[(3,0),(3,4)]
        self.push_status = {(3,0):0,(3,4):0}
        self.loading_time = 5
        self.unloading_time = 5
        self.focal_bound = 6
        self.goal_candidates = []
        self.weight = 4
        self.pickle_path = pickle
        self.goal_candidates = self.load_od_pool_from_pickle(pickle)

        self.start_goal_list = []


        #1013 수정 이전 버전
        self.load_collision_profile(pickle)

        start_list, goal_list, num_agents = self.importmap_file(filename)
        n_agv = num_agents
        # for agv_id in range(1, n_agv + 1):
        #     self.AGVs[agv_id] = AGV(agv_id)
        # crit_freq, crit_affect_count = self.compute_critical_and_affect_counts(pickle)
        #
        # # (선택) always_critical 제외하고 싶으면:
        # # for a in data.get("always_critical", []): crit_affect_count.pop(a, None)
        #
        # # 2) 사전계산형 highway heuristic 테이블 생성
        # self.build_reverse_state_graph_weighted(highways=crit_affect_count, w=self.focal_bound)
        # #   ↳ 여기서 revW/divisor까지 준비됨
        #
        # # 2) 목표들에 대해 h 테이블 사전계산
        # self.compute_all_highway_heuristic(self.goal_candidates)
        # self.set_highway_by(mode="crit_affect",w=self.focal_bound)
        # For random_experiment based exp
    #     for agv_id in range(1, num_agents + 1):
    #         self.AGVs[agv_id] = AGV(agv_id)
    #         # print("Only in init ###########################")
    #         self.AGVs[agv_id].cur_loc = self.rc_to_index(start_list[agv_id - 1][0], start_list[agv_id - 1][1])
    #         state_key = self.rc_to_stop_state(start_list[agv_id - 1][0], start_list[agv_id - 1][1],
    #                                           start_list[agv_id - 1][2])
    #         self.AGVs[agv_id].planned_path.append((state_key, float('inf')))
    #         self.AGVs[agv_id].fixed_path_state.append((state_key, float('inf')))
    #         self.AGVs[agv_id].cur_state = self.Stop_state[state_key]
    #         self.AGVs[agv_id].dst_state_list.append((self.rc_to_stop_state(goal_list[agv_id - 1][0],
    #                                                                        goal_list[agv_id - 1][1],
    #                                                                        goal_list[agv_id - 1][2]), 0))
    #         # self.Stop_state[state_key].rsv_veh_list.append(agv_idx)
    #         self.Stop_state[state_key].rsv_time_table.append((float('inf'), agv_id, state_key))
    #         self.Stop_state[state_key].interval_list = []
    #         for states in self.Stop_state[state_key].affect_state:
    #             #     self.Stop_state[states].rsv_veh_list.append(agv_idx)
    #             temp_state = self.convert_id_to_state(states)
    #             temp_state.rsv_time_table.append((float('inf'), agv_id, state_key))
    #             # temp_state.interval_list = []
    #             # self.Stop_state[states].split_interval(0, float('inf'))
    # print("Line 324")
    def load_od_pool_from_pickle(self,pickle_path: str):
        """
        pickle 안의 od_pairs와 Stop_state 정보를 이용해,
        od_pairs에 등장하는 모든 node 번호에 해당하는
        Stop_state의 key("S_<node>_<angle>")를 리스트로 반환한다.
        """
        with open(pickle_path, "rb") as f:
            od_data = pickle.load(f)
            od_list = od_data["od_pairs"]
            stop_states = od_data["Stop_state"]

        # 1️⃣ od_pairs에 등장한 모든 node 번호를 수집
        used_nodes = set()
        for s, g in od_list:
            used_nodes.add(s)
            used_nodes.add(g)

        # 2️⃣ Stop_state key에서 node 번호 추출해 매핑
        node_to_key = {}
        for key in stop_states.keys():  # 예: "S_240_90"
            try:
                node_idx = key.split(",")[1]
                node_to_key[node_idx] = key
            except (IndexError, ValueError):
                continue

        # 3️⃣ od_list에 등장한 노드만 남기기
        matched_keys = [node_to_key[n] for n in used_nodes if n in node_to_key]

        print(f"[INFO] Found {len(matched_keys)} Stop_state keys used in OD pairs.")
        return matched_keys

    def compute_critical_and_affect_counts(self,pickle_path, dedup_per_od=True):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        def flatten_crit(x):
            out = []
            if isinstance(x, dict):
                for lst in x.values():
                    out.extend(set(lst) if dedup_per_od else list(lst))
            elif isinstance(x, list):
                out.extend(x)
            return out

        crit_all = []
        crit_all += flatten_crit(data.get("critical_nodes"))
        crit_all += flatten_crit(data.get("critical"))  # (o,d)별 critical 있으면 포함
        crit_all += data.get("always_critical", []) or []  # 원하면 제외해도 됨

        critical_freq = Counter(crit_all)  # critical 자체 빈도
        affect_map = data.get("collision_profile", {})  # c_state -> [affect states]

        crit_affect_count = Counter()
        for c, k in critical_freq.items():
            crit_affect_count[c] += k
            for s in affect_map.get(c, []) or []:
                crit_affect_count[s] += k
        return dict(critical_freq), dict(crit_affect_count)

    def normalize_to_lambda(self,counts: dict, w: float):
        if not counts:
            return {}
        cmax = max(counts.values())
        if cmax <= 0: return {}
        return {s: 1.0 + (w - 1.0) * (cnt / cmax) for s, cnt in counts.items()}

    def load_collision_profile(self, filepath="collision_profile.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.collision_profile = data["collision_profile"]
        self.move_regions = data["move_regions"]
        self.stop_regions = data["stop_regions"]
        self.Stop_state = data["Stop_state"]
        self.Move_state = data["Move_state"]
        self.critical_nodes = data["critical_nodes"]
        print(f"Collision profile loaded from {filepath}")


    def importmap_file(self, filename, first = True):  # map file을 input으로 받아 데이터 변환
        with open(filename, "r") as f:
            scn = json.load(f)

        layout_path = scn["layout_path"]
        agv_list = scn["agv_list"]

        # 2) LayoutManager 로드

        # 3) start / goal state IDs
        start_list = [agv["start"] for agv in agv_list]
        goal_list = [agv["goal"] for agv in agv_list]

        num_agents = len(agv_list)
        print(num_agents,"NUMAGENTS")

        self.AGVs = {}
        for agv_id in range(1, num_agents + 1):
            self.AGVs[agv_id] = AGV(agv_id)

        for agv_id in range(1, num_agents + 1):
            self.AGVs[agv_id] = AGV(agv_id)
            # print("Only in init ###########################")
            state_key = start_list[agv_id-1]
            self.AGVs[agv_id].planned_path.append((state_key, float('inf')))
            self.AGVs[agv_id].fixed_path_state.append((state_key, float('inf')))
            self.AGVs[agv_id].cur_state = self.Stop_state[state_key].id
            self.AGVs[agv_id].dst_state_list.append((goal_list[agv_id-1],0))
            # self.Stop_state[state_key].rsv_veh_list.append(agv_idx)
            self.Stop_state[state_key].rsv_time_table.append((float('inf'), agv_id, state_key))
            self.Stop_state[state_key].interval_list = []
            for states in self.Stop_state[state_key].affect_state:
                #     self.Stop_state[states].rsv_veh_list.append(agv_idx)
                temp_state = self.convert_id_to_state(states)
                temp_state.rsv_time_table.append((float('inf'), agv_id, state_key))
                # temp_state.interval_list = []
                # self.Stop_state[states].split_interval(0, float('inf'))
            # print("here")

        return start_list, goal_list, num_agents


    def rc_to_index(self, r, c):
        return r * self.nc + c

    def index_to_rc(self, idx):  # idx --> (r, c)
        return idx // self.nc, idx % self.nc

    def rc_to_stop_state(self,r,c, dir): #initialize할 때만 새용
        index = self.rc_to_index(r,c)
        name = 'S,' + str(index) + ',' + str(self.angle[dir])
        return name

    def index_to_distance(self, point1, point2):
        r1, c1 = self.index_to_rc(point1)
        r2, c2 = self.index_to_rc(point2)
        return np.abs(r1 - r2) + np.abs(c1 - c2)

    def get_move_direction(self, prev_cp, next_cp):
        next_r, next_c = get_idx_from_cp(next_cp)
        prev_r, prev_c = get_idx_from_cp(prev_cp)

        if prev_r == next_r and prev_c + 1 == next_c:
            return Direction.Right
        elif prev_r == next_r and prev_c - 1 == next_c:
            return Direction.Left
        elif prev_r + 1 == next_r and prev_c == next_c:
            return Direction.Down
        elif prev_r - 1 == next_r and prev_c == next_c:
            return Direction.Up
        else:
            print('Error in get_move_direction')
            return Direction.Unknown

    def get_point_from_cp(self, cp_name: str):
        # 1029 message change
        # _, r, c = cp_name.split('_')
        _, r, c = cp_name.split('_')
        r = int(r) - 1
        c = int(c) - 1
        return self.points[r, c]

    def get_rc_from_cp(self, cp_name: str):
        _, r, c = cp_name.split('_')
        r = int(r) - 1
        c = int(c) - 1
        return r,c

    def convert_id_to_state(self,id):
        if id[0] == 'S':
            state = self.Stop_state[id]
        else:
            state= self.Move_state[id]
        return state

    def convert_dsloc_to_state(self,location):
        r,c = self.get_rc_from_cp(location)
        idx = self.rc_to_index(r,c)
        if r == 0:
            dst_state_key = 'S,'+str(idx)+',90'
            dst_state1 = self.Stop_state[dst_state_key]
            r = r+1
            idx2 = self.rc_to_index(r,c)
            dst_state2 = self.Stop_state['S,'+str(idx2)+',270']
        elif r == 6:
            dst_state_key = 'S,'+str(idx)+',270'
            dst_state1 = self.Stop_state[dst_state_key]
            r = r-1
            idx2 = self.rc_to_index(r,c)
            dst_state2 = self.Stop_state['S,'+str(idx2)+',90']
        return dst_state1, dst_state2

    def merge_intervals(self,intervals):
        # Sort intervals based on the start point
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        return merged

    def find_non_overlapping_intervals(self,a, b, intervals):
        # Add bounds to the intervals list
        merged_intervals = self.merge_intervals(intervals)

        # Initialize the list of non-overlapping intervals
        non_overlapping_intervals = []
        # Start with the lower bound
        current_start = a

        for interval in merged_intervals:
            if interval[0] > current_start:
                # If there's a gap before the current interval starts, add it
                non_overlapping_intervals.append((current_start, interval[0]))
            current_start = max(current_start, interval[1])

        # After processing all intervals, check if we need to add the final interval up to b
        if current_start < b:
            non_overlapping_intervals.append((current_start, b))

        return non_overlapping_intervals

    def get_successors(self, state, curt, interval, goal,constraint_table):  # input은 state 자체가 되어야함(name x)
        successors = []
        # if constraint_table != {}:
        #     print("Here1")
        #0125 temp erase
        # if goal.interval_list == []:
        #     return successors
        # else:
        #     goal_time = goal.interval_list[-1][1]

        for neighbor in state.next_state:
            if neighbor in constraint_table:
                interval_list = []
                for i in range(len(constraint_table[neighbor])):
                    interval_list.append(constraint_table[neighbor][i]['timestep'])
                interval_real = self.find_non_overlapping_intervals(interval[0], interval[1], interval_list)
            else:
                interval_real = [(curt
                                  ,float('inf'))]

        # for neighbor in state.next_state:
        #     if neighbor in constraint_table:
        #         interval_real = []
        #         interval_list = []
        #
        #         for i in range(len(constraint_table[neighbor])):
        #             interval_list.append(constraint_table[neighbor][i]['timestep'])
        #         interval_list.sort()
        #         #building safe interval
        #         for i in range(len(interval_list)):
        #             if i == 0:
        #                 # print("Here",interval_list,interval_list[i])
        #                 if interval_list[i][0] == 0:
        #                     pass
        #                 elif curt == interval_list[i][0]:
        #                     pass
        #                 else:
        #                     interval_real.append((curt,interval_list[i][0]))
        #             if i == len(interval_list)-1:
        #                 if interval_list[i][1] == float('inf'):
        #                     pass
        #                 else:
        #                     interval_real.append((interval_list[i][1],float('inf')))
        #             else:
        #                 if interval_list[i][1] == interval_list[i+1][0]:
        #                     continue
        #                 elif interval_list[i][1] > interval_list[i+1][0]:
        #                     print("Wrong interval listcheck", interval_list[i], interval_list[i + 1])
        #                     continue
        #                 interval_real.append((interval_list[i][1],interval_list[i+1][0]))
        #     else:
        #         interval_real = [(0,float('inf'))]

            start_t = curt + state.cost
            # if start_t >= goal_time:
            #     continue
            end_t = interval[1]
            if neighbor == goal.id:
                end_t = float('inf')  # +13은 loading, unloading시간 + 회전 및 이동. 구간별 분류는 나중에 추가 예정 디버깅
            for i in interval_real:
                # print("CHECK",state.id,i,start_t,end_t)
                if state.id[0]  == 'S':
                    # if i[0] > end_t or i[1] < start_t or end_t > i[1]:
                    # print("Support",i,end_t,start_t)
                    if i[0] > end_t or i[1] <= start_t :
                        # print("1")
                        # print(i, start_t, end_t, i[0] > end_t or i[1] < start_t or end_t > i[1] or i[0] > start_t)
                        if neighbor == goal.id:
                            print("continue",start_t,end_t,neighbor,i)
                        continue
                else:
                    if i[0] > end_t or i[1] <= start_t:
                    # if i[0] > end_t or i[1] < start_t or end_t > i[1] or start_t < i[0]:

                        # print(i, start_t, end_t, i[0] > end_t or i[1] < start_t or end_t > i[1] or i[0] > start_t)
                        # print("2")
                        if neighbor == goal.id:
                            print("continue",start_t,end_t,neighbor,i)
                        continue

                time = max(start_t, i[0])
                #CCH 왜
                # if i[1] != float('inf'):
                    # print("Check",time,end_t,i)
                s = SIPP_tuple(neighbor, float('inf'), time, i)
                successors.append(s)
        return successors


    def compute_plan_prioritized(self, v_id, start, goal_state, constraints, cur_time=0, timeout_seconds=5.0):
        search_start_time = time.time()  # 1. 타임아웃을 위한 시작 시간 기록

        constraint_table = self.get_constraint_table(v_id, constraints)
        goal = self.convert_id_to_state(goal_state)
        start_state = start
        goal_state_id = goal.id

        if start_state in constraint_table:
            print("Constraint in compute plan prioritized", constraint_table[start_state])
            interval_s = (cur_time, constraint_table[start_state][0]['timestep'][0])
            # interval_s = (cur_time, float('inf'))
        else:
            interval_s = (cur_time, float('inf'))

        s_start = SIPP_tuple(start_state, 0, float(cur_time), interval_s)
        s_start.g = cur_time
        # goal_id = goal_state[0]
        h_hwy = self.highway_heuristic_table[goal_state]
        s_start.h = h_hwy.get(start_state, float('inf'))
        s_start.f = s_start.g + s_start.h

        open_list = [(s_start.f, s_start)]
        heapq.heapify(open_list)

        came_from = {start_state: None}
        cost_so_far = {start_state: s_start.g}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current.state == goal_state_id and not self.is_goal_constrained(goal_state_id, current.time,
                                                                               constraint_table):
                return came_from, cost_so_far, current.f, current

            cur_state = self.convert_id_to_state(current.state)
            successors = self.get_successors(cur_state, current.time, current.interval, goal, constraint_table)

            for next in successors:
                g_val = next.time
                h_val = h_hwy.get(next.state, float('inf'))
                f_val = g_val + h_val

                if next.state not in cost_so_far or g_val < cost_so_far[next.state]:
                    cost_so_far[next.state] = g_val
                    came_from[next.state] = current.state

                    next_node = SIPP_tuple(next.state, g_val, next.time, next.interval)
                    next_node.g = g_val
                    next_node.h = h_val
                    next_node.f = f_val

                    heapq.heappush(open_list, (f_val, next_node))

        return None, None, float('inf'), None


    def get_constraint_table(self, agent, constraints):
        ##############################
        # Task 1.2/1.3: Return a table that constains the list of constraints of
        #               the given agent for each time step. The table can be used
        #               for a more efficient constraint violation check in the
        #               is_constrained function.
        c_table = dict()
        for cagent in constraints:
            # we need to consider only the constraints for the given agent
            # 4.1 Supporting positive constraints
            # if (not 'positive' in c.keys()):
            #     c['positive'] = False
            if cagent != agent:
                for c in constraints[cagent]:
                    state = c['loc']
                    if state not in c_table:
                        c_table[state] = [c]
                    else:
                        c_table[state].append(c)
        return c_table


    def build_constraint_table(self,agent,constraints):
        ##############################
        # Task 1.2/1.3: Return a table that constains the list of constraints of
        #               the given agent for each time step. The table can be used
        #               for a more efficient constraint violation check in the
        #               is_constrained function.
        c_table = dict()
        for c in constraints:
            # we need to consider only the constraints for the given agent
            # 4.1 Supporting positive constraints
            # if (not 'positive' in c.keys()):
            #     c['positive'] = False
            if c['agent'] == agent:
                state = c['loc']
                if state not in c_table:
                    c_table[state] = [c]
                else:
                    c_table[state].append(c)
        return c_table

    def is_goal_constrained(self, goal_loc, timestep, constraint_table):
        """
        checks if there's a constraint on the goal in the future.
        goal_loc            - goal location
        timestep            - current timestep
        constraint_table    - generated constraint table for current agent
        """

        for c in constraint_table:
            if c == goal_loc:
                # print("iscgoalcons", goal_loc, constraint_table, constraint_table[c], timestep)
                for j in range(len(constraint_table[c])):
                    if constraint_table[c][j]['timestep'][1] > timestep:
                        return True
        return False

    def state_cost(self, sid):
        obj = None
        S = getattr(self, "Stop_state", None)
        if S and sid in S:
            obj = S[sid]
        else:
            M = getattr(self, "Move_state", None)
            if M and sid in M: obj = M[sid]
        if obj is None: return 0.0
        val = (obj.get("cost") if isinstance(obj, dict) else getattr(obj, "cost", None))
        return float(val) if isinstance(val, (int, float)) else 0.0

    ########### 1013 highway generation
    def _forward_state_graph(self):
        """u -> list(v) (state id 기준, 기존 next_state 집계)"""
        G = {}
        all_states = {}
        all_states.update(self.Stop_state)
        all_states.update(self.Move_state)
        for sid, sobj in all_states.items():
            G[sid] = list(getattr(sobj, "next_state", []))
        return G


    def _brandes_edge_bc_weighted_subset(self, G, sources, targets, rates_map):
        """
        Weighted Brandes edge-betweenness for subset pairs:
          sum_{(s,t)} λ_{st} * (# of s→t shortest paths through edge)
        - G: dict[u] -> list of v (forward adjacency, unweighted)
        - sources: iterable of source nodes
        - targets: iterable of target nodes
        - rates_map: dict[(s,t)] = lambda_st (float>0)
        반환: dict[(u,v)] = weighted edge-betweenness
        """
        from collections import deque, defaultdict

        EBC = defaultdict(float)
        V = list(G.keys())
        Tset = set(targets)

        for s in sources:
            # 1) BFS from s (unweighted shortest paths)
            S = []  # stack
            P = {v: [] for v in V}  # predecessors
            sigma = dict((v, 0.0) for v in V)
            dist = dict((v, -1) for v in V)

            sigma[s] = 1.0
            dist[s] = 0
            Q = deque([s])

            while Q:
                v = Q.popleft()
                S.append(v)
                for w in G.get(v, []):
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        Q.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            # 2) dependency accumulation (edge) with weighted target reward
            delta = dict((v, 0.0) for v in V)

            # 타겟 가중치: base_s[w] = sum_t λ_{st} * 1{w==t}
            # (s 고정 상태에서 해당 타겟만 보상)
            base_s = defaultdict(float)
            # rates_map 키는 (s,t)
            # t가 그래프에 있고 BFS로 도달 가능한 경우에만 유효
            for t in Tset:
                lam = rates_map.get((s, t), 0.0)
                if lam > 0.0 and dist.get(t, -1) >= 0:
                    base_s[t] += lam

            while S:
                w = S.pop()
                # weighted terminal reward: base_s[w]
                # edge (v->w)의 기여: (sigma[v]/sigma[w]) * (base_s[w] + delta[w])
                for v in P[w]:
                    if sigma[w] > 0:
                        c = (sigma[v] / sigma[w]) * (base_s[w] + delta[w])
                        EBC[(v, w)] += c
                        delta[v] += c
            # 다음 source로

        return dict(EBC)

    def _brandes_edge_bc_subset(self, G, sources, targets):
        """
        Brandes edge-betweenness(무가중·최단경로 수 기준)를
        (s in sources, t in targets, s!=t) 쌍으로만 합산.
        반환: dict[(u,v)] = score
        """
        from collections import deque, defaultdict
        EBC = defaultdict(float)
        V = list(G.keys())
        Tset = set(targets)

        for s in sources:
            # 1) BFS (unweighted shortest paths)
            S = []                         # stack
            P = {v: [] for v in V}         # predecessors
            sigma = dict((v, 0.0) for v in V)
            d     = dict((v, -1)   for v in V)

            sigma[s] = 1.0
            d[s] = 0
            Q = deque([s])

            while Q:
                v = Q.popleft()
                S.append(v)
                for w in G.get(v, []):
                    if d[w] < 0:
                        d[w] = d[v] + 1
                        Q.append(w)
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            # 2) dependency accumulation (edge)
            delta = dict((v, 0.0) for v in V)
            # 타겟 제한: targets만 종단으로 간주(없으면 전체)
            is_target = (len(Tset) > 0)

            while S:
                w = S.pop()
                # 타겟이면 1을 소스에 환류(노드 betweenness 유사),
                # edge betweenness는 아래 e누적에서 처리됨.
                w_dependency = 0.0
                if w in Tset and w != s:
                    w_dependency = 1.0
                for v in P[w]:
                    if sigma[w] > 0:
                        c = (sigma[v] / sigma[w]) * (w_dependency + delta[w])
                        # edge (v->w) 의 기여
                        EBC[(v, w)] += c
                        delta[v] += c

        return dict(EBC)


    def compute_port_od_edge_betweenness(self, OD_list=None, OD_rates=None,
                                         sources=None, targets=None, assign='move'):
        """
        Adaptive (subset) edge-betweenness with OD rates:
          - OD_rates: dict[(src_sid, dst_sid)] = lambda_st (우선)
          - OD_list  : [(src_sid, dst_sid), ...] 가 주어지고 OD_rates 없으면 모두 λ=1
          - sources/targets를 직접 주면 그걸 우선 사용하되,
            rates_map은 (s,t) 쌍만 고려(없는 쌍은 λ=0 취급).
        assign:
          'move' : edge (u->v)의 점수를 v가 Move_state일 때만 v에 귀속
          'head' : edge (u->v)의 점수를 v state에(Stop/Move 상관없이) 귀속
        반환:
          base_scores: dict[state_id] = betweenness-derived score (OD 가중 포함)
        """
        # 0) 그래프
        G = self._forward_state_graph()

        # 1) rates_map 준비
        rates_map = {}
        if OD_rates and len(OD_rates) > 0:
            # 복사 방지용 새 dict (float 변환)
            for (s, t), lam in OD_rates.items():
                try:
                    lamf = float(lam)
                except:
                    lamf = 0.0
                if lamf > 0:
                    rates_map[(s, t)] = lamf
        elif OD_list:
            # λ=1로 균등 가중
            for (s, t) in OD_list:
                rates_map[(s, t)] = rates_map.get((s, t), 0.0) + 1.0
        else:
            # 둘 다 없으면 기존 goal_candidates fallback (균등)
            # 여기서는 goal_candidates에서 (s,t) 쌍을 구성할 근거가 없으니 반환을 비우거나,
            # 혹은 기존 전체쌍을 만들려면 별도의 로직이 필요.
            # 안전하게 빈 결과 반환
            return {}

        # 2) sources/targets 자동 추출 (미지정 시)
        if sources is None:
            sources = sorted({s for (s, _t) in rates_map.keys()})
        if targets is None:
            targets = sorted({t for (_s, t) in rates_map.keys()})

        if not sources or not targets:
            return {}

        # 3) subset weighted edge betweenness 계산
        ebc = self._brandes_edge_bc_weighted_subset(G, sources, targets, rates_map)

        # 4) edge score -> state score 귀속
        base_scores = {}
        for (u, v), val in ebc.items():
            if not v:
                continue
            if assign == 'move':
                # v가 Move_state일 때만
                if v[0] == 'M' or v[0] == 'R':
                    base_scores[v] = base_scores.get(v, 0.0) + val
            else:
                base_scores[v] = base_scores.get(v, 0.0) + val

        return base_scores
    # --------------------------
    # (B) Cut-State Frequency on Stop-graph (Tarjan + 분리쌍 수)
    # --------------------------
    def _build_undirected_stop_graph(self):
        """
        Stop 상태만으로 무방향 그래프 구성:
        Stop s에서 Move m으로, 다시 Stop t로 가는 전이를 s~t 무방향 간선으로 본다.
        """
        from collections import defaultdict
        UG = defaultdict(set)
        for sid, sobj in self.Stop_state.items():
            for nid in getattr(sobj, "next_state", []):
                if nid and nid[0] == 'M':
                    mobj = self.Move_state.get(nid)
                    if not mobj: continue
                    for nid2 in getattr(mobj, "next_state", []):
                        if nid2 and nid2[0] == 'S':
                            UG[sid].add(nid2)
                            UG[nid2].add(sid)
        return UG

    def _articulation_impact_portpairs(self, UG, ports):
        """
        관절점 v 제거 시, 포트 집합이 여러 컴포넌트로 쪼개짐.
        각 컴포넌트의 포트 개수를 p1, p2, ... 라 하면
        v의 pair-분리 수 = sum_{i<j} (p_i * p_j).
        반환: dict[stop_state]=count
        """
        import sys
        sys.setrecursionlimit(10**7)
        ports = set(ports)
        # Tarjan articulation points + 컴포넌트 분해
        time = 0
        disc = {}
        low  = {}
        parent = {}
        visited = set()
        ap = set()
        comps_per_ap = {}  # v -> [component_nodes sets]  (v 빼고 이웃 분기별 모음)

        def dfs(u):
            nonlocal time
            visited.add(u)
            disc[u] = low[u] = time; time += 1
            child_cnt = 0
            # 각 자식 분기별 노드 수집
            child_comps = []

            for v in UG.get(u, []):
                if v not in visited:
                    parent[v] = u
                    child_cnt += 1
                    comp_nodes = set()
                    sub = dfs_collect(v, comp_nodes, stop_at=u)
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    child_comps.append(sub)

                    # root ap
                    if parent.get(u) is None and child_cnt > 1:
                        ap.add(u)
                        comps_per_ap.setdefault(u, []).extend(child_comps)
                        child_comps = []
                    # non-root ap
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                        comps_per_ap.setdefault(u, []).append(sub)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
            return

        def dfs_collect(x, bag, stop_at):
            """ stop_at(=부모 후보) 쪽으로 역행하지 않고 분기 컴포넌트를 수집 """
            bag.add(x)
            for y in UG.get(x, []):
                if y == stop_at:
                    continue
                if y not in bag:
                    dfs_collect(y, bag, stop_at)
            return bag

        # 실행
        for s in UG.keys():
            if s not in visited:
                parent[s] = None
                dfs(s)

        # 각 관절점 u에 대해 포트 분리쌍 수 계산
        impact = {}
        for u in ap:
            # u를 중심으로 갈라진 분기들의 포트 개수
            counts = []
            for comp in comps_per_ap.get(u, []):
                counts.append(sum(1 for p in ports if p in comp))
            # u 자신을 경유하지 않는 나머지 포트(=u 포함된 중심 컴포넌트)의 포트 수도 한 바구니로
            total_ports = sum(1 for p in ports if p in UG)
            counted = sum(counts)
            rest = max(0, total_ports - counted - (1 if u in ports else 0))
            if rest > 0:
                counts.append(rest)

            # sum_{i<j} p_i * p_j
            s = 0
            for i in range(len(counts)):
                for j in range(i+1, len(counts)):
                    s += counts[i] * counts[j]
            impact[u] = float(s)
        return impact

    def compute_cut_state_frequency(self, ports=None, affected=False):
        """
        Stop-graph에서 관절점(articulation) 영향도를 '빈도'처럼 환산.
        ports 기본값: goal_candidates(Stop id들) 가정.
        affected=True면 affect_state에 동일 가중을 전파(Stop/Move 모두).
        반환: dict[state_id] = score
        """
        if ports is None:
            ports = [gid for gid in self.goal_candidates if gid and gid[0] == 'S']

        UG = self._build_undirected_stop_graph()
        impact = self._articulation_impact_portpairs(UG, ports)  # Stop-state 기준

        if not affected:
            return impact

        # affected version: stop의 점수를 해당 stop의 affect_state 전체에 전파(+자기 자신)
        out = dict(impact)
        for s, val in impact.items():
            sobj = self.Stop_state.get(s)
            if not sobj: continue
            for a in getattr(sobj, "affect_state", []):
                out[a] = out.get(a, 0.0) + val
        return out

    # --------------------------
    # (C) Highway 설정 스위처
    # --------------------------

    # === Helper: decide if a Move_state follows the crisscross field ===

    def _crisscross_preferred_move(self, move_obj):
        rs, cs = self.index_to_rc(int(move_obj.start_id.split(',')[1]))
        re, ce = self.index_to_rc(int(move_obj.end_id.split(',')[1]))

        dr = re - rs
        dc = ce - cs

        row_rule = (rs % 2 == 1 and dc == 1) or (rs % 2 == 0 and dc == -1)
        col_rule = (cs % 2 == 1 and dr == -1) or (cs % 2 == 0 and dr == 1)
        return bool(row_rule or col_rule)

    # === Helper: build highway edge list for crisscross (edges where target is preferred move) ===
    def _build_crisscross_highway_edges(self):
        # ensure reverse graph is available to expand to (pred, v) pairs
        rev = getattr(self, 'reverse_state_graph', None)
        if rev is None:
            rev = self.build_reverse_state_graph()
        # collect preferred Move_state ids
        preferred_moves = set()
        for mid, mobj in getattr(self, 'Move_state', {}).items():
            if self._crisscross_preferred_move(mobj):
                preferred_moves.add(mid)
        # convert to edge tuples (pred -> v) for all predecessors of v
        highways = []
        for v in preferred_moves:
            for u in rev.get(v, []):
                highways.append((u, v))
        return highways

    # --------------------------
    # (C) Highway 설정 스위처 (extended)
    # --------------------------
    def _build_plain_focal_heuristic_table(self, goal_candidates):
        """Build self.highway_heuristic_table using Dijkstra on reverse graph (no highway).
        Dist(u) = min sum of per-state costs along forward path from u to goal.
        Uses cost of the *next* state (v) for edge (u->v)."""
        # ensure reverse graph exists
        if not hasattr(self, 'reverse_state_graph') or self.reverse_state_graph is None:
            try:
                self.build_reverse_state_graph()
            except Exception:
                return
        self.highway_heuristic_table = {}

        for goal_id in goal_candidates:
            dist = {goal_id: 0.0}
            pq = [(0.0, goal_id)]
            while pq:
                d, v = heapq.heappop(pq)
                if d > dist.get(v, float('inf')):
                    continue
                for u in self.reverse_state_graph.get(v, []):
                    nd = d + self.state_cost(v)  # entering v's cost
                    if nd < dist.get(u, float('inf')):
                        dist[u] = nd
                        heapq.heappush(pq, (nd, u))
            self.highway_heuristic_table[goal_id] = dist

    def _heading(self, sid):
        # 필요시 네 클래스 필드명에 맞게 수정
        if isinstance(sid, str) and sid.startswith("M,"):
            sobj = self.Move_state.get(sid)
            return getattr(sobj, "heading", None)
        if isinstance(sid, str) and sid.startswith("S,"):
            try:
                return int(sid.split(",")[2])
            except Exception:
                return None
        return None  # 회전/기타는 heading 없음

    def _service_time(self, sid):
        # 타깃 state i의 점유시간(=processing time). Move의 cost를 사용.
        sobj = self.Move_state.get(sid) if isinstance(sid, str) and sid.startswith("M,") else None
        return float(getattr(sobj, "cost", 0.0) or 0.0)

    def compute_weights_from_arrival_profile(
            self,
            arrival_profile,
            *,
            beta=0.7,
            Wmax=2.5,
            rho_star=0.8,
            rho_cap=0.98,
            detour_ratio=None,
            eta=0.10,
            eps=1e-9
    ):
        detour_ratio = detour_ratio or {}
        weights, debug = {}, {}

        for sid, prof in (arrival_profile or {}).items():
            lam = float(prof.get('lambda', 0.0) or 0.0)
            m1 = float(prof.get('m1', 0.0) or 0.0)
            m2 = float(prof.get('m2', 0.0) or 0.0)

            if lam <= 0.0 or m1 <= 0.0 or m2 <= 0.0:
                weights[sid] = 1.0
                debug[sid] = {"lambda": lam, "m1": m1, "m2": m2, "rho": 0.0, "Wq": 0.0}
                continue

            rho = min(lam * m1, rho_cap)
            if rho >= 1.0 - 1e-12:
                Wq = float("inf")
            else:
                Wq = (lam * m2) / (2.0 * max(1.0 - rho, eps))

            # 기본 weight
            if rho < rho_star:
                w = 1.0
            else:
                w = 1.0 + beta * (Wq / (m1 + eps))

            # detour gate (옵션)
            if detour_ratio.get(sid, 0.0) > eta:
                w = 1.0

            # clip
            w = float(min(max(w, 1.0), Wmax))

            weights[sid] = w
            debug[sid] = {"lambda": lam, "m1": m1, "m2": m2, "rho": rho, "Wq": Wq}

        return weights, debug

    def _directional_affect_propagation(self, base_scores, affect_map):
        """
        base_scores: dict[state_id] = 중요도(critical freq, betweenness 등)
        affect_map : dict[critical_state] -> [affect_state,...]
                     여기서는 collision_profile 같은 걸 그대로 쓸 수 있음.

        규칙:
        - critical state c 자기 자신은 base_scores[c]만큼 penalty에 포함
        - affect state s는 방향 계수만큼 가중해서 누적
        - 방향 계수:
            * c 또는 s가 'S,...'면 전파 스킵 (Stop은 최종 대기지점일 뿐이라면)
            * c 또는 s가 'R,...' or heading 없음 → coeff = 1
            * 둘 다 Move이고 heading 있음 → coeff = 1 - cos(Δθ)
        """

        penalty = Counter()

        for c_state, k in base_scores.items():
            # 기반 점수 k를 가진 핵심 state
            penalty[c_state] += k

            # heading 추출 helper
            def _get_heading(sid):
                if sid[0] == "M":
                    sobj = self.Move_state.get(sid)
                    return getattr(sobj, "heading", None)
                elif sid[0] == "S":
                    # Stop은 여기서는 아예 propagate 안 할 거라 굳이 안 씀
                    try:
                        return int(sid.split(",")[2])
                    except Exception:
                        return None
                else:
                    # e.g. 'R,...' or 기타
                    return None

            c_head = _get_heading(c_state)

            for s_state in affect_map.get(c_state, []) or []:

                # Stop state로는 전파하지 않는 네 최신 규칙 반영
                if c_state[0] == "S" or s_state[0] == "S":
                    continue

                s_head = _get_heading(s_state)

                # 계수 계산
                if (c_state[0] == "R") or (s_state[0] == "R"):
                    coeff = 0
                else:
                    if c_head is None or s_head is None:
                        # heading이 없는 상태도 회전 계열 취급 = 1
                        coeff = 1.0
                    else:
                        dtheta = abs(c_head - s_head) % 360
                        if dtheta > 180:
                            dtheta = 360 - dtheta
                        coeff = 1.0 - math.cos(math.radians(dtheta))
                penalty[s_state] += k * coeff

        return dict(penalty)

    def _directional_affect_propagation_with_demand(self, base_scores, affect_map, *,
                                                    stop_propagate=False,
                                                    cv_default=0.0):
        """
        base_scores: dict[state_id] = k  (여기서 k는 λ*BC 등 '수요 강도'로 해석)
        affect_map : dict[critical_state] -> [affect_state,...]
        stop_propagate=False: Stop으로는 전파 안 함(네 규칙대로 기본 끔)
        cv_default: 서비스시간 분산 근사용 (deterministic이면 0.0)

        반환:
          penalty: dict[state_id] -> float (이전과 동일한 합산 결과)
          arrival_profile: dict[state_id] -> {
              'lambda': float,            # 도착률 총합 (전파 포함)
              'mix': [                    # 소스별 수요 믹스(큐잉용)
                  {'src': c_state,
                   'rate': lam_contrib,   # 해당 소스에서 온 도착률 기여
                   'm1': m1,              # 서비스시간 평균
                   'm2': m2},             # 서비스시간 2차 모멘트
              ],
              'm1': mbar,                 # rate 가중 평균 서비스시간
              'm2': m2bar                 # rate 가중 2차 모멘트
          }
        """
        penalty = Counter()
        # 수요 믹스 임시버퍼: state -> list of (src, rate, m1, m2)
        mix_buffer = defaultdict(list)

        for c_state, k in (base_scores or {}).items():
            # 핵심 state 자체 가산 (기존 penalty 유지)
            penalty[c_state] += k

            c_is_stop = (c_state[0] == "S")
            c_head = self._get_heading_from_sid(c_state)

            # 자기 자신도 '수요'로 포함 (자기 state를 사용하는 트래픽)
            # 서비스시간은 대상(state)의 것으로 잡아야 하므로 아래 루프에서 처리
            # (자기 자신이 Move인 경우 affect_map에 스스로 포함되어 있지 않아도 넣고 싶다면 여기에 별도로 추가)

            for s_state in (affect_map.get(c_state, []) or []):
                s_is_stop = (s_state[0] == "S")
                if (c_is_stop or s_is_stop) and (not stop_propagate):
                    # Stop 전파 금지 규칙
                    continue

                # 방향 계수 계산
                if c_state[0] == "R" or s_state[0] == "R":
                    coeff = 0
                else:
                    # heading 없는 경우도 회전 계열 취급
                    s_head = self._get_heading_from_sid(s_state)
                    if c_head is None or s_head is None:
                        coeff = 1.0
                    else:
                        dtheta = abs(c_head - s_head) % 360
                        if dtheta > 180:
                            dtheta = 360 - dtheta
                        coeff = 1.0 - math.cos(math.radians(dtheta))

                coeff = 1
                if c_state[0] == "R" or s_state[0] == "R":
                    coeff = 0
                # 도착률 기여
                lam_contrib = k * coeff
                if lam_contrib <= 0:
                    continue

                # 대상 state의 서비스시간(=processing time)
                m1 = self._get_service_time_from_sid(s_state)
                if m1 <= 0:
                    # 서비스시간이 0인 state는 큐잉 대상 X (단, penalty 합산엔 반영돼 있음)
                    continue
                # 두 번째 모멘트 (deterministic이면 cv_default=0 권장)
                m2 = m1 * m1 * (1.0 + cv_default * cv_default)

                # penalty는 기존대로 누적
                penalty[s_state] += lam_contrib

                # 큐잉용 믹스에는 (src, rate, m1, m2)로 기록
                mix_buffer[s_state].append({
                    'src': c_state,
                    'rate': lam_contrib,
                    'm1': m1,
                    'm2': m2
                })

        # state별 합산 λ, m̄, m̄² 계산
        arrival_profile = {}
        for s_state, mix_list in mix_buffer.items():
            lam_total = sum(x['rate'] for x in mix_list)
            if lam_total <= 0:
                continue
            mbar = sum((x['rate'] / lam_total) * x['m1'] for x in mix_list)
            m2bar = sum((x['rate'] / lam_total) * x['m2'] for x in mix_list)
            arrival_profile[s_state] = {
                'lambda': lam_total,
                'mix': mix_list,
                'm1': mbar,
                'm2': m2bar
            }

        return dict(penalty), arrival_profile
    def _get_heading_from_sid(self, sid):
        """네 기존 helper를 밖으로 뺀 버전: state id에서 heading 추출."""
        if sid[0] == "M":
            sobj = self.Move_state.get(sid)
            return getattr(sobj, "heading", None)
        elif sid[0] == "S":
            try:
                return int(sid.split(",")[2])
            except Exception:
                return None
        else:
            return None  # 'R,...' 등


    def _get_service_time_from_sid(self, sid):
        """
        Move state의 processing time(=cost)을 반환.
        회전/정지 등 케이스는 네 클래스 구조에 맞춰 필요시 분기.
        """
        if sid[0] == "M" or sid[0] =="R":
            sobj = self.Move_state.get(sid)
            if sobj is not None and hasattr(sobj, "cost"):
                return float(sobj.cost)
        # 기본값(안전빵): 0 -> 나중 집계에서 무시됨
        return 0.0

    def compute_queue_based_weights(
            # 필수 입력
            self,
            server_of_state: Dict[Any, Any],  # state_id -> server_id  (충돌영역/자원 묶음)
            lambda_i: Dict[Any, float],  # state_id -> 도착률 λ_i
            m1_i: Dict[Any, float],  # state_id -> E[S_i] (평균 서비스시간)
            m2_i: Optional[Dict[Any, float]] = None,  # state_id -> E[S_i^2] (없으면 cv로 근사)
            # 선택 입력
            detour_ratio: Optional[Dict[Any, float]] = None,  # state_id -> 우회 비용 증가율 r_i (게이팅용)
            # 하이퍼파라미터 / 안정화 옵션
            beta: float = 0.7,  # 민감도 (W_q/m 대비)
            Wmax: float = 2.5,  # 최종 상한
            rho_star: float = 0.80,  # ρ-gate 시작점 (이하에서는 w≈1 유지)
            rho_cap: float = 0.98,  # 발산 방지용 상한
            eta: float = 0.10,  # detour gate 임계치 (r_i>eta면 w=1 강제)
            cv_default: float = 0.75,  # m2 근사용 c_v
            eps: float = 1e-9  # 0나눗셈 방지
    ) -> Tuple[Dict[Any, float], Dict[Any, dict]]:
        """
        다종 M/G/1 혼합 기반 weight 계산.
        1) 서버(자원)별로 {λ_i, m1_i, m2_i}를 집계 → 혼합모멘트 m̄, m̄^(2), ρ, W_q
        2) 서버별 공통 W_q를 상태별 m_i로 정규화하여 w_i 산출 (+게이팅/클립)

        반환:
          weights: dict[state_id] = w_i
          server_stats: dict[server_id] = {
             "lambda": λ, "mbar": m̄, "m2bar": m̄^(2), "rho": ρ, "Wq": W_q,
             "states": [state_ids...]
          }
        """
        # --- 0) 기본 세팅
        detour_ratio = detour_ratio or {}
        m2_i = m2_i or {}

        # --- 1) 서버별 묶기
        states_by_server: Dict[Any, list] = {}
        for s, srv in server_of_state.items():
            states_by_server.setdefault(srv, []).append(s)

        # --- 2) 서버별 혼합모멘트/ρ/Wq 계산
        server_stats: Dict[Any, dict] = {}
        for srv, states in states_by_server.items():
            # 유효 상태만 필터(모든 모수 존재 + λ_i>0, m1_i>0)
            valid = []
            lam_total = 0.0
            for i in states:
                lam = float(lambda_i.get(i, 0.0) or 0.0)
                m1 = float(m1_i.get(i, 0.0) or 0.0)
                if lam > 0.0 and m1 > 0.0:
                    valid.append(i)
                    lam_total += lam

            if lam_total <= 0.0 or not valid:
                # 트래픽이 없으면 Wq=0으로 기록
                server_stats[srv] = {
                    "lambda": 0.0, "mbar": 0.0, "m2bar": 0.0,
                    "rho": 0.0, "Wq": 0.0, "states": states
                }
                continue

            # 클래스 비율 p_i, 혼합 모멘트 m̄, m̄^(2)
            mbar = 0.0
            m2bar = 0.0
            for i in valid:
                lam = float(lambda_i[i])
                p_i = lam / lam_total
                m1 = float(m1_i[i])
                # m2 없으면 근사
                if i in m2_i and m2_i[i] is not None:
                    m2 = float(m2_i[i])
                else:
                    m2 = (m1 ** 2) * (1.0 + (cv_default ** 2))
                mbar += p_i * m1
                m2bar += p_i * m2

            rho = min(lam_total * mbar, rho_cap)
            if rho >= 1.0 - 1e-12:
                Wq = float('inf')  # 이론상 발산
            else:
                Wq = (lam_total * m2bar) / (2.0 * max(1.0 - rho, eps))

            server_stats[srv] = {
                "lambda": lam_total, "mbar": mbar, "m2bar": m2bar,
                "rho": rho, "Wq": Wq, "states": states
            }

        # --- 3) 상태별 weight 산출
        weights: Dict[Any, float] = {}
        for srv, info in server_stats.items():
            Wq_srv = info["Wq"]
            rho = info["rho"]
            for i in info["states"]:
                m1 = float(m1_i.get(i, 0.0) or 0.0)
                # 기본값: 데이터 없으면 w=1
                w = 1.0
                if m1 > 0.0 and math.isfinite(Wq_srv):
                    # ρ-gate
                    if rho < rho_star:
                        w = 1.0
                    else:
                        w = 1.0 + beta * (Wq_srv / (m1 + eps))
                    # detour gate
                    if detour_ratio.get(i, 0.0) > eta:
                        w = 1.0
                    # 상하한 clip
                    w = float(min(max(w, 1.0), Wmax))
                weights[i] = w

        return weights, server_stats

    def set_highway_by(self, mode="crit_affect", w=None):
        """
        mode ∈ {
          'plain'            : (NEW) highway 없이 순수 Focal Search
          'crit'             : critical only
          'crit_affect'      : critical + affect 전파
          'port_bc'          : Port-OD edge betweenness(도착 Move-state에 귀속)
          'cut'              : cut-state frequency(Stop-only)
          'cut_affect'       : cut-state freq + affect 전파
          'port_bc_affect'   : port_bc + affect 전파 (신규)
          'crisscross_soft'  : 행/열 교차 방향장 기반 soft highway (off-highway penalty = w)
          'crisscross_strict': 동일하되 매우 큰 penalty로 사실상 금지(엄격)
        }
        w : 오프하이웨이 penalty 또는 인플레이트 상한(기존 dict-λ 모드).
        """
        # default w

        OD_list = [(agv.cur_state, agv.dst_state_list[0][0]) for agv in self.AGVs.values()]
        OD_rates = {(src, dst): self.fromto_rate for (src, dst) in OD_list}

        if w is None:
            w = float(getattr(self, 'focal_bound', 2.0))

        if mode != "crisscross_strict":
            self.focal_bound = self.focal_bound / w
        # --- (NEW) Plain Focal: highway 비활성화 ---
        if mode in ("plain", "none", "off"):
            self._build_plain_focal_heuristic_table(self.goal_candidates)
        counts = None
        if mode == "crit":
            crit, _ = self.compute_critical_and_affect_counts(getattr(self, "pickle_path", None))
            counts = crit
        elif mode == "crit_affect":
            _, crit_aff = self.compute_critical_and_affect_counts(getattr(self, "pickle_path", None))
            counts = crit_aff
        elif mode == "port_bc":
            counts = self.compute_port_od_edge_betweenness(OD_list=OD_list, assign='move')
        elif mode == "port_bc_affect":
            # base: move-wise port-OD betweenness
            counts = self.compute_port_od_edge_betweenness(OD_list=OD_list, assign='move') or {}
            counts = self._directional_affect_propagation(counts, self.collision_profile)

        elif mode == "cut":
            counts = self.compute_cut_state_frequency(affected=False)
        elif mode == "cut_affect":
            counts = self.compute_cut_state_frequency(affected=True)
        elif mode in ("crisscross_soft", "crisscross_strict"):
            # edge-list highway: on-highway edges get 1.0, off-highway get 'off_highway_penalty'
            edges = self._build_crisscross_highway_edges()
            off_p = (w if mode == "crisscross_soft" else max(w, 1e6))
            self.build_reverse_state_graph_weighted(highways=edges, off_highway_penalty=off_p)
            self.compute_all_highway_heuristic(self.goal_candidates)
            return {"edges": len(edges), "off_penalty": off_p}
        elif mode == "optimal":
            self.focal_bound = 1
            self._build_plain_focal_heuristic_table(self.goal_candidates)
        elif mode == 'Queuing_bc':
            counts = self.compute_port_od_edge_betweenness(OD_rates=OD_rates, assign='move')
            penalty, arrival_profile = self._directional_affect_propagation_with_demand(
                base_scores=counts,
                affect_map=self.collision_profile,
                stop_propagate=False,  # Stop으로는 전파 끔
                cv_default=0.0  # 결정론적이면 0, 불확실성 반영하려면 0.5~1.0
            )

            # 2) 큐잉 입력으로 변환
            lambda_i = {sid: prof['lambda'] for sid, prof in arrival_profile.items()}
            m1_i = {sid: prof['m1'] for sid, prof in arrival_profile.items()}
            m2_i = {sid: prof['m2'] for sid, prof in arrival_profile.items()}

            # (선택) server_of_state, detour_ratio 등은 너의 기존 맵에서 가져옴
            weights, server_stats = self.compute_weights_from_arrival_profile(
                arrival_profile=arrival_profile,
                beta=1, Wmax=w, rho_star=0, rho_cap=0.98, eta=0.1
            )
            print(weights, server_stats)
            self.build_reverse_state_graph_weighted(highways=weights, w=w)
            self.compute_all_highway_heuristic(self.goal_candidates)
            return weights
        else:
            counts = {}

        # dict (state→count) 모드 → λ 정규화 후 적용
        lam = self.normalize_to_lambda(counts, w=w) if counts else {}
        self.build_reverse_state_graph_weighted(highways=lam, w=w)
        self.compute_all_highway_heuristic(self.goal_candidates)
        return counts
        ########### 1013 highway generation
    def build_reverse_state_graph(self):
        from collections import defaultdict
        self.reverse_state_graph = defaultdict(set)
        all_states = {**self.Stop_state, **self.Move_state}

        for state_id, state_obj in all_states.items():
            for next_id in state_obj.next_state:
                self.reverse_state_graph[next_id].add(state_id)
        return self.reverse_state_graph

    def build_reverse_state_graph_weighted(self, highways=None, w=None, off_highway_penalty=None):
        """
        highways:
          - dict[state->count] : critical+affect count (λ 정규화 사용)
          - list[(u,v)]        : 하이웨이 간선 목록(온=1, 오프=penalty)
          - None               : 가중 없이 state_cost만 사용
        결과:
          self.revW[v] = [(u, w_prime(u->v)), ...]  # 휴리스틱용 가상 가중치
          self.revW_divisor = (dict모드: w, list모드: off_pen, 기본: 1.0)
        """
        # 0) 기존 역그래프 얻기 (네가 이미 쓰는 함수 그대로 호출)
        rev = self.build_reverse_state_graph()  # dict: v -> [preds...]

        # 1) 모드 결정
        w = float(w if w is not None else getattr(self, "focal_bound", 1.3))
        off_p = float(off_highway_penalty if off_highway_penalty is not None
                      else getattr(self, "off_highway_penalty", 2.0))

        if isinstance(highways, dict):  # ── count → λ(s) 모드
            counts = highways
            cmax = max(counts.values()) if counts else 0
            lam = {sid: 1.0 + (w - 1.0) * (cnt / cmax) for sid, cnt in counts.items()} if cmax > 0 else {}
            self.lam = lam

            def edge_cost_prime(u, v):
                return self.state_cost(v) * float(lam.get(v, 1.0))

            divisor = w
        elif isinstance(highways, (list, set, tuple)):  # ── 간선 리스트 모드
            hwy = set(map(tuple, highways))

            def edge_cost_prime(u, v):
                return self.state_cost(v) * (1.0 if (u, v) in hwy else off_p)

            divisor = off_p
        else:  # ── 기본(가중치 없음)
            def edge_cost_prime(u, v):
                return self.state_cost(v)

            divisor = 1.0

        # 2) 역가중 그래프 구성
        revW = {}
        for v, preds in (rev.items() if hasattr(rev, "items") else []):
            lst = []
            for u in preds:
                lst.append((u, edge_cost_prime(u, v)))
            revW[v] = lst

        self.revW = revW
        self.revW_divisor = divisor

        return revW

    # def compute_highway_heuristic(self, goal_id, highway_edges, w):
    #     import heapq
    #     h_hwy = {goal_id: 0}
    #     frontier = [(0, goal_id)]
    #     visited = set()
    #
    #     while frontier:
    #         cost, node = heapq.heappop(frontier)
    #         if node in visited:
    #             continue
    #         visited.add(node)
    #
    #         for predecessor in self.reverse_state_graph.get(node, []):
    #             edge = (predecessor, node)
    #             weight = 1 if edge in highway_edges else w
    #             new_cost = cost + weight
    #             if predecessor not in h_hwy or new_cost < h_hwy[predecessor]:
    #                 h_hwy[predecessor] = new_cost
    #                 heapq.heappush(frontier, (new_cost, predecessor))
    #
    #     return h_hwy

    # Automatically compute highway heuristics for a list of goal candidates

    # def compute_all_highway_heuristics(self, goal_candidates, highway_edges, w):
    #     self.highway_heuristic_table = {}
    #     self.build_reverse_state_graph()
    #     for goal_state in goal_candidates:
    #         goal_id = goal_state[0]  # Assumes goal_state is a tuple like (node_id, ...)
    #         h_hwy = self.compute_highway_heuristic(goal_id, highway_edges, w)
    #         self.highway_heuristic_table[goal_id] = h_hwy

    def compute_all_highway_heuristic(self, goal_ids):
        """
        build_reverse_state_graph_weighted()가 만들어둔
        self.revW, self.revW_divisor만 사용해 h-table 생성.
        """
        R = getattr(self, "revW", None)
        div = float(getattr(self, "revW_divisor", 1.0))
        assert R is not None, "call build_reverse_state_graph_weighted() first"

        self.highway_heuristic_table = {}
        for g in goal_ids:
            dist = {g: 0.0}
            pq = [(0.0, g)]
            while pq:
                d, x = heapq.heappop(pq)
                if d != dist[x]: continue
                for (p, wprime) in R.get(x, []):  # (pred, 가상가중)
                    nd = d + wprime
                    if nd < dist.get(p, np.inf):
                        dist[p] = nd
                        heapq.heappush(pq, (nd, p))
            # 허용성 유지용 스케일
            #1014 div 빼버림
            # self.highway_heuristic_table[g] = {sid: dist[sid] / div for sid in dist}
            self.highway_heuristic_table[g] = {sid: dist[sid] for sid in dist}
        return self.highway_heuristic_table

    def compute_plan(self, v_id, start, goal_state, constraints):
        cur_time = 0
        constraint_table = self.build_constraint_table(v_id, constraints)
        goal = self.convert_id_to_state(goal_state[0])
        start_state = start.id
        goal_state_id = goal.id

        if start_state in constraint_table:
            interval_s = (cur_time, constraint_table[start_state][0]['timestep'][0])
        else:
            interval_s = (0, float('inf'))

        s_start = SIPP_tuple(start_state, 0, float(cur_time), interval_s)
        s_start.g = cur_time
        goal_id = goal_state[0]
        h_hwy = self.highway_heuristic_table[goal_id]
        s_start.h = h_hwy.get(start_state, float('inf'))
        s_start.f = s_start.g + s_start.h

        open_list = [(s_start.f, s_start)]
        heapq.heapify(open_list)

        came_from = {start_state: None}
        cost_so_far = {start_state: s_start.g}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current.state == goal_state_id and not self.is_goal_constrained(goal_state_id, current.time,
                                                                               constraint_table):
                return came_from, cost_so_far, current.f

            cur_state = self.convert_id_to_state(current.state)
            successors = self.get_successors(cur_state, current.time, current.interval, goal, constraint_table)

            for next in successors:
                g_val = next.time
                h_val = h_hwy.get(next.state, float('inf'))
                f_val = g_val + h_val

                if next.state not in cost_so_far or g_val < cost_so_far[next.state]:
                    cost_so_far[next.state] = g_val
                    came_from[next.state] = current.state

                    next_node = SIPP_tuple(next.state, g_val, next.time, next.interval)
                    next_node.g = g_val
                    next_node.h = h_val
                    next_node.f = f_val

                    heapq.heappush(open_list, (f_val, next_node))

        return None, None, float('inf')

    def reconstruct_path(self, came_from, start_state, goal_state, cost_so_far):
        current = goal_state
        cost = cost_so_far[current]
        path = [(current,cost)]
        start = start_state
        while current != start:
            current = came_from[current]
            cost = cost_so_far[current]
            path.append((current,cost))

        path.reverse()
        return path

    def heuristic(self, start,goal):
        r1,c1 = start.sr, start.sc
        r2,c2 = goal.sr, goal.sc

        return abs(r1 - r2) + abs(c1 - c2)


    def detect_collision(self,pathA, pathB):
        pathA_idx = 0
        pathB_idx = 0
        while pathA_idx < len(pathA)- 1 or pathB_idx < len(pathB)-1 :
            if pathA[pathA_idx][0][0] == 'S':
                pathA_state = self.Stop_state[pathA[pathA_idx][0]]
            else:
                pathA_state = self.Move_state[pathA[pathA_idx][0]]

            if pathB[pathB_idx][0][0] == 'S':
                pathB_state = self.Stop_state[pathB[pathB_idx][0]]
            else:
                pathB_state = self.Move_state[pathB[pathB_idx][0]]

            pathA_start_time = pathA[pathA_idx][1]

            if pathA_idx == len(pathA)-1:
                pathA_end_time = float('inf')
            else:
                pathA_end_time = pathA[pathA_idx+1][1]
            pathB_start_time = pathB[pathB_idx][1]

            if pathB_idx == len(pathB)-1:
                pathB_end_time = float('inf')
            else:
                pathB_end_time = pathB[pathB_idx+1][1]


            #Zero stop action skip
            if pathA_start_time == pathA_end_time:
                pathA_idx +=1
                continue
            if pathB_start_time == pathB_end_time:
                pathB_idx +=1
                continue

            if pathB_state.id in pathA_state.affect_state:
                #0129
                # con = [pathA[pathA_idx][0],pathB[pathB_idx][0]],(pathA_start_time,pathB_end_time),(pathB_start_time,pathA_end_time)
                con = [pathA[pathA_idx][0],pathB[pathB_idx][0]],(pathA_start_time,pathA_end_time),(pathB_start_time,pathB_end_time)
                return con

            if pathA_idx == len(pathA) - 1:
                pathB_idx += 1
            elif pathB_idx == len(pathB) -1:
                pathA_idx += 1
            elif pathB_end_time > pathA_end_time:
                pathA_idx +=1
            elif pathA_end_time > pathB_end_time:
                pathB_idx += 1
            elif pathA_end_time == pathB_end_time:
                pathA_idx +=1
                pathB_idx +=1
        return None

    def standard_splitting(self,collision):
        ##############################
        # Task 3.2: Return a list of (two) constraints to resolve the given collision
        #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
        #                            specified timestep, and the second constraint prevents the second agent to be at the
        #                            specified location at the specified timestep.
        #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
        #                          specified timestep, and the second constraint prevents the second agent to traverse the
        #                          specified edge at the specified timestep
        # in this case, we can ignore final as all the paths are normalized
        time1s, time1e = collision['timestep1'][0], collision['timestep1'][1]
        time2s, time2e = collision['timestep2'][0], collision['timestep2'][1]


        #추후 move 끼리의 time offset 추가 필요 CCH
        if collision['loc1'][0] == 'S':
            collision_interval1 = (time2s,time2e)
        else:
            collision_interval1 = (time1s, time2e)


        if collision['loc2'][0] == 'S':
            collision_interval2 = (time1s,time1e)
        else:
            collision_interval2 = (time2s, time1e)

        constraints = []

        #0125
        # if collision['type'] == 'vertex':
        constraints.append({
            'agent': collision['a1'],
            'loc': collision['loc1'],
            'timestep': collision_interval1,
            'final': False
        })
        constraints.append({
            'agent': collision['a2'],
            'loc': collision['loc2'],
            'timestep': collision_interval2,
            'final': False
        })
        # elif collision['type'] == 'edge':
        #     constraints.append({
        #         'agent': collision['a1'],
        #         'loc': collision['loc'],
        #         'timestep': collision['timestep'],
        #         'final': False
        #     })
        #     constraints.append({
        #         'agent': collision['a2'],
        #         # revesred returns an iterator. In python list == iterator returns false, not an error: nasty bug
        #         'loc': list(reversed(collision['loc'])),
        #         'timestep': collision['timestep'],
        #         'final': False
        #     })
        return constraints


    def detect_collisions(self,paths):
        # paths = []
        # for i in range(len(self.AGVs)):
        #     paths.append(self.AGVs[i+1].planned_path)

        collisions = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                coll_data = self.detect_collision(paths[i], paths[j])

                # if coll_data is not None (collision detected)
                if coll_data:
                    collisions.append({
                        'a1': i+1,
                        'a2': j+1,
                        'loc1': coll_data[0][0],  # vertex or edge
                        'loc2': coll_data[0][1],  # vertex or edge
                        'timestep1': coll_data[1],  # timestep
                        'timestep2':coll_data[2]
                    })
        # if collisions != []:
        #     for i in range(len(collisions)):
        #         if collisions[i]['timestep1'][1] - collisions[i]['timestep1'][0] > 2 or collisions[i]['timestep2'][1] - collisions[i]['timestep2'][0] > 2:
        #             print(paths[0])
        #             print(paths[1])
        #             print("Collision",collisions)
        return collisions


    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=False, max_time=float('inf')):
        import heapq, time, random, copy
        self.max_time = max_time
        self.start_time = time.time()
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.open_heap = []  # (LB, gen_id, node)
        self.focal_heap = []  # (prio, gen_id, node)
        self.node_alive = {}  # lazy deletion용 플래그
        self.cont = 0

        # --- root node 생성 ---
        root = {
            'constraints': [],
            'paths': [],
            'lb_values': [],
            'cost': 0.0,
            'LB': 0.0,
            'collisions': []
        }

        for i in range(len(self.AGVs)):
            agent = i + 1
            came_from, cost, lb = self.compute_plan(
                agent,
                self.AGVs[agent].cur_state,
                self.AGVs[agent].dst_state_list[0],
                root['constraints']
            )
            if came_from is None:
                return None, None
                raise BaseException("No solution for agent {}".format(agent))
            path = self.reconstruct_path(
                came_from,
                self.AGVs[agent].cur_state,
                self.AGVs[agent].dst_state_list[0][0],
                cost
            )
            root['paths'].append(path)
            root['lb_values'].append(lb)

        root['cost'] = float(sum(self.compute_total_cost(root['paths'])))
        root['collisions'] = self.detect_collisions(root['paths'])
        root['LB'] = float(sum(root['lb_values']))

        heapq.heappush(self.open_heap, (root['LB'], self.num_of_generated, root))
        self.node_alive[self.num_of_generated] = True

        # 첫 FOCAL 구성
        min_LB = root['LB']
        heapq.heappush(
            self.focal_heap,
            (len(root['collisions']), self.num_of_generated, root)
        )
        self.num_of_generated += 1

        # --- 메인 루프 ---
        while self.focal_heap and (time.time() - self.start_time < self.max_time):
            # OPEN의 현재 최소 LB (O(1))
            while self.open_heap and not self.node_alive[self.open_heap[0][1]]:
                heapq.heappop(self.open_heap)
            if not self.open_heap:
                break
            min_LB = self.open_heap[0][0]

            # FOCAL에서 노드 pop
            while self.focal_heap:
                _, gid, node = heapq.heappop(self.focal_heap)
                if not self.node_alive.get(gid, False):
                    continue  # lazy deletion skip
                if node['LB'] > self.focal_bound * min_LB:
                    continue  # FOCAL eligibility loss
                break
            else:
                # focal_heap empty
                # rebuild focal from open_heap (band refresh)
                self.focal_heap.clear()
                for LB, gid, n in self.open_heap:
                    if LB <= self.focal_bound * min_LB and self.node_alive.get(gid, False):
                        prio = (len(n['collisions']), n['cost'])
                        heapq.heappush(self.focal_heap, (prio, gid, n))
                continue

            # 이제 유효 노드 확장
            self.node_alive[gid] = False  # mark popped
            self.num_of_expanded += 1

            # 충돌 없으면 해 반환
            if not node['collisions']:
                return node['paths'], time.time() - self.start_time

            # --- conflict 분할 ---
            collision = random.choice(node['collisions'])
            constraints = (
                disjoint_splitting(collision)
                if disjoint else self.standard_splitting(collision)
            )

            for c in constraints:
                new_node = {
                    'constraints': node['constraints'] + [c],
                    'paths': copy.deepcopy(node['paths']),
                    'lb_values': node['lb_values'][:],
                    'collisions': [],
                    'cost': 0.0,
                    'LB': 0.0
                }
                agent = c['agent']
                came_from, cost, lb = self.compute_plan(
                    agent,
                    self.AGVs[agent].cur_state,
                    self.AGVs[agent].dst_state_list[0],
                    new_node['constraints']
                )
                if came_from is None:
                    continue

                path = self.reconstruct_path(
                    came_from,
                    self.AGVs[agent].cur_state,
                    self.AGVs[agent].dst_state_list[0][0],
                    cost
                )
                new_node['paths'][agent - 1] = path
                new_node['lb_values'][agent - 1] = lb
                new_node['cost'] = float(sum(self.compute_total_cost(new_node['paths'])))
                new_node['collisions'] = self.detect_collisions(new_node['paths'])
                new_node['LB'] = float(sum(new_node['lb_values']))

                entry = (new_node['LB'], self.num_of_generated, new_node)
                heapq.heappush(self.open_heap, entry)
                self.node_alive[self.num_of_generated] = True

                # focal eligibility check
                if new_node['LB'] <= self.focal_bound * min_LB:
                    prio = (len(new_node['collisions']), new_node['cost'])
                    heapq.heappush(self.focal_heap, (prio, self.num_of_generated, new_node))

                self.num_of_generated += 1

        return None, None

    def extend_rsv_path(self, vehicle, simtime):
        timea = time.time()
        print("etxreserve")
        if len(vehicle.planned_path) == 1 and len(vehicle.dst_state_list) > 0:
            state = self.convert_id_to_state(vehicle.planned_path[0][0])
            state.rsv_time_table.sort()
            if state.rsv_time_table[-1][0] != float('inf'):
                print("Rsvt1",state.rsv_time_table,state.id)
            state.rsv_time_table.pop()


            if len(state.rsv_time_table) == 0:
                state.interval_list = [(simtime, float('inf'))]
            # elif state.rsv_time_table[-1][0] != float('inf'):
            #     state.interval_list.append((simtime,float('inf')))


            for aff_states in state.affect_state:
                self.convert_id_to_state(aff_states).rsv_time_table.sort()
                if self.convert_id_to_state(aff_states).rsv_time_table[-1][0] != float('inf'):
                    print("Rsvt2",aff_states, self.convert_id_to_state(aff_states).rsv_time_table , state.id)
                self.convert_id_to_state(aff_states).rsv_time_table.pop()
                if len(self.convert_id_to_state(aff_states).rsv_time_table) == 0:
                    self.convert_id_to_state(aff_states).interval_list = [(simtime, float('inf'))]
                # elif self.convert_id_to_state(aff_states).rsv_time_table[-1][0] != float('inf'):
                #     self.convert_id_to_state(aff_states).interval_list.append((simtime,float('inf')))
            self.compute_path_from_list(vehicle,simtime)


        timeb = time.time()
        print("Timea-b",timeb-timea)
        temp_list = []
        count =0
        for i in range(len(vehicle.planned_path)):
            #0516 조건 맞는지 확인
            if i == 0:
                temp_list.append(vehicle.planned_path[i])
                continue

            cur_state = vehicle.planned_path[i]
            state = self.convert_id_to_state(vehicle.planned_path[i][0])
            state.rsv_time_table.sort()
            if vehicle.planned_path[i][1] <= simtime and state.rsv_time_table[0][1] == vehicle.id:  #0511 debug point
            # if state.rsv_time_table[0][1] == vehicle.id:  #0511 debug point
                temp_list.append(vehicle.planned_path[i])
            else:
                print("Early arrival", state.id,state.rsv_time_table)
                break

            # if cur_state.rsv_time_table[0][0] <= time:
            #     temp_list.append(vehicle.planned_path[i])
            # else:
            #     break
        vehicle.fixed_path_state = temp_list
        vehicle.fixed_path = self.convert_state_path_to_cp(vehicle.fixed_path_state)
        timec = time.time()

    def convert_state_path_to_cp(self,path):
        prev_angle = None
        cp_prev = None
        temp_list_convert = []
        for i in range(len(path)):
            if path[i][0][0] == 'S':
                stopc, pos, angle = path[i][0].split(',')
                r = self.Stop_state[path[i][0]].sr
                c = self.Stop_state[path[i][0]].sc
                cp = self.cp_name(r,c)
                if cp_prev != cp:
                    temp_list_convert.append(cp)
                elif cp_prev == cp and i == len(path)-1:
                    turn = 'T,'+str(prev_angle)+','+str(angle)
                    temp_list_convert.append(turn)
                prev_angle = angle
                cp_prev = cp
        return temp_list_convert



    def cp_name(self,r,c):
        return f'cp_{r + 1}_{c + 1}'

    # def get_push_location(self,state):
    #     state_id = state.id
    #     if state_id[0] != 'S':
    #         print("Wrong at push loc@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #     else:
    #         _,idx,angle = state_id.split(',')
    #         r,c = self.index_to_rc(int(idx))
    #         if r == 1:
    #             pr = r+2
    #             angle = str(270)
    #         elif r == 5:
    #             pr = r-2
    #             angle = str(90)
    #         else:
    #             print("Once pushed")
    #             return None
    #
    #         if c<=2:
    #             c=0
    #         else:
    #             c=4
    #
    #         if (pr, c) in self.push_location:
    #             push_state_id = 'S,'+str(self.rc_to_index(pr,c))+','+angle
    #             if self.Stop_state[push_state_id].rsv_time_table == []:
    #                 return push_state_id
    #             else:
    #                 push_state_id = None
    #
    #         if (pr, 4-c) in self.push_location:
    #             push_state_id = 'S,'+str(self.rc_to_index(pr,4-c))+','+angle
    #             if self.Stop_state[push_state_id].rsv_time_table == []:
    #                 return push_state_id
    #             else:
    #                 push_state_id = None
    #         else:
    #             print("No available pushloc---------------------------------------------------------------------")
    #             return None
    #
    #     return push_state_id



    def build_constraint_table_prioritized(self, agent, pathA):
        pathA_idx = 0
        const = []

        if len(pathA) == 1:
            if pathA[pathA_idx][0][0] == 'S':
                pathA_state = self.Stop_state[pathA[pathA_idx][0]]
            else:
                pathA_state = self.Move_state[pathA[pathA_idx][0]]
            pathA_start_time = pathA[0][1]
            pathA_end_time = float('inf')

            for states in pathA_state.affect_state:
                cost_state = self.convert_id_to_state(states).cost
                if cost_state == 0:
                    collision_interval = (pathA_start_time, pathA_end_time)
                else:  # negative value?
                    collision_interval = (pathA_start_time - cost_state, pathA_end_time)
                const.append({
                    'agent': agent,
                    'loc': states,
                    'timestep': collision_interval,
                    'final': False
                })

        while pathA_idx <= len(pathA) - 1:
            # print("Heredebug",pathA,pathA_idx,pathA[0])
            if pathA[pathA_idx][0][0] == 'S':
                pathA_state = self.Stop_state[pathA[pathA_idx][0]]
            else:
                pathA_state = self.Move_state[pathA[pathA_idx][0]]

            pathA_start_time = pathA[pathA_idx][1]

            if pathA_idx == len(pathA) - 1:
                pathA_end_time = float('inf')
            else:
                pathA_end_time = pathA[pathA_idx + 1][1]
            # Zero stop action skip
            if pathA_start_time == pathA_end_time:
                pathA_idx += 1
                continue

            # collisions.append({
            #     'a1': i + 1,
            #     'a2': j + 1,
            #     'loc1': coll_data[0][0],  # vertex or edge
            #     'loc2': coll_data[0][1],  # vertex or edge
            #     'timestep1': coll_data[1],  # timestep
            #     'timestep2': coll_data[2]
            # })

            for states in pathA_state.affect_state:
                cost_state = self.convert_id_to_state(states).cost
                if cost_state == 0:
                    collision_interval = (pathA_start_time, pathA_end_time)
                else:  # negative value?
                    collision_interval = (pathA_start_time - cost_state, pathA_end_time)
                const.append({
                    'agent': agent,
                    'loc': states,
                    'timestep': collision_interval,
                    'final': False
                })
            pathA_idx += 1
        return const



    def find_empty_terminals(self):
        """
        Find empty terminals by comparing goal_list and AGV dst_state_list[0] node IDs.
        Return a copy of available terminals.
        """
        available_terminals = copy.deepcopy(self.goal_candidates)

        occupied_nodes = set()
        for agv in self.AGVs.values():
            if hasattr(agv, 'dst_state_list') and agv.dst_state_list:
                node_id = agv.dst_state_list[0][0].split(',')[1]  # Extract node id
                occupied_nodes.add(node_id)
            if hasattr(agv, 'cur_state') and agv.cur_state:
                node_id = agv.cur_state.split(',')[1]  # Extract node id
                occupied_nodes.add(node_id)

        filtered_terminals = []
        for terminal in available_terminals:
            terminal_node_id = terminal.split(',')[1]  # Extract node id
            if terminal_node_id not in occupied_nodes:
                filtered_terminals.append(terminal)

        return filtered_terminals


    def find_solution_prioritized(self, max_time=100):
        """Dependency-based planning until every AGV has visited all its goals.
           Order: Z(unvisited→goal) → move finished blockers to empty terminal → break cycles.
        """
        n_agents = len(self.AGVs)

        # --- [NEW] 각 AGV가 dst_state_list에서 몇 번째 goal을 향하고 있는지 인덱스 관리 ---
        goal_idx = {}  # agv_id -> 현재 목표의 인덱스 (0부터 시작)

        for agv_id, agv in self.AGVs.items():
            if not agv.dst_state_list:
                # 할당된 goal이 없으면 바로 완료 취급
                goal_idx[agv_id] = 0
            else:
                print("DSTSTATELIST", agv.id, agv.dst_state_list)
                # 일단 첫 번째 goal부터 시작
                goal_idx[agv_id] = 0
                cur = agv.cur_state if agv.cur_state is not None else None
                # 만약 이미 첫 goal에 서 있다면 1로 올려도 되지만,
                # 보수적으로는 여기서는 0으로 두고 아래 visited_once 계산에서 처리해도 됨.
                # 필요하면 아래와 같이:
                # if cur == agv.dst_state_list[0][0]:
                #     goal_idx[agv_id] = 1

        # --- [CHANGED] visited_once: 이제 "해당 AGV가 자신의 모든 goal을 다 방문했는지" ---
        visited_once = set()
        for agv_id, agv in self.AGVs.items():
            if not agv.dst_state_list:
                visited_once.add(agv_id)
            else:
                cur = agv.cur_state if agv.cur_state is not None else None
                # dst_state_list의 마지막 goal까지 이미 도착해 있는 경우 완료 처리
                last_goal = agv.dst_state_list[-1][0]
                if cur == last_goal:
                    visited_once.add(agv_id)
                    goal_idx[agv_id] = len(agv.dst_state_list)

        planned_paths = {i + 1: [] for i in range(n_agents)}  # 누적 경로 보관
        constraints = {}

        # for i in range(len(self.AGVs)):
        #     #     aid = i + 1
        #     #     constraints[aid] = self.build_constraint_table_prioritized(aid, self.AGVs[aid].planned_path)
        #
        for aid, agv in self.AGVs.items():
            if getattr(agv, "planned_path", None):  # 혹은 planned_path 쓰는 구조면 그걸로
                constraints[aid] = self.build_constraint_table_prioritized(aid, agv.planned_path)
            else:
                constraints[aid] = []
        # 초기화
        self.max_time = max_time
        self.start_time = time.time()
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.open_list = []
        self.cont = 0

        while len(visited_once) < n_agents:
            G_dep = self.build_dependency_graph()  # 목표=다른 AGV 현재 위치이면 (i→j)

            if len(G_dep.nodes) == 0:
                break

            # 1) Z 집합(출차수 0) 중 아직 모든 goal을 다 방문하지 않은 AGV
            Z = [a for a in G_dep.nodes if G_dep.out_degree(a) == 0 and a not in visited_once]

            best_agv, best_target, best_dist = None, None, float('inf')

            # --- [CHANGED] Z ∩ 미완료 AGV들 중에서 "현재 goal_idx의 목표"로 보내기 ---
            for agv_id in Z:
                agv = self.AGVs[agv_id]
                if not agv.dst_state_list:
                    continue

                # 이 AGV가 아직 남은 goal이 있는지 확인
                if goal_idx[agv_id] >= len(agv.dst_state_list):
                    continue  # 이미 모든 goal 소화

                # 현재 타겟 goal (순서적으로)
                goal = agv.dst_state_list[goal_idx[agv_id]][0]
                curr = agv.cur_state if agv.cur_state else None
                if curr is None:
                    continue

                dist = self.highway_heuristic_table[goal].get(curr, float('inf'))
                if dist < best_dist:
                    best_dist = dist
                    best_agv = agv_id
                    best_target = goal  # 자신의 현재 목표 (k-th goal)

            # 2) 없으면, indegree>0 인 '방해자' 중 이미 모든 goal을 다 방문한 AGV를 빈 터미널로 이동
            if best_agv is None:
                empty_terms = self.find_empty_terminals()  # state_id 리스트
                if empty_terms:
                    blockers = [a for a in G_dep.nodes if G_dep.in_degree(a) > 0 and a in visited_once]
                    for agv_id in blockers:
                        curr = self.AGVs[agv_id].cur_state if self.AGVs[agv_id].cur_state else None
                        if curr is None:
                            continue
                        for t in empty_terms:
                            dist = self.highway_heuristic_table[t].get(curr, float('inf'))
                            if dist < best_dist:
                                best_dist = dist
                                best_agv = agv_id
                                best_target = t  # 빈 터미널

            # 3) 그래도 없으면, 사이클 깨기: (가능하면 이미 모든 goal을 방문한) 에이전트를 빈 터미널로
            if best_agv is None:
                empty_terms = self.find_empty_terminals()
                cycles = self.detect_cycles(G_dep)
                cyc_agents = [a for cyc in cycles for a in cyc]
                # 모든 goal을 끝낸 AGV 우선, 없으면 아무나
                prefer = [a for a in cyc_agents if a in visited_once] or cyc_agents
                for agv_id in prefer:
                    curr = self.AGVs[agv_id].cur_state if self.AGVs[agv_id].cur_state else None
                    if curr is None:
                        continue
                    for t in empty_terms:
                        dist = self.highway_heuristic_table[t].get(curr, float('inf'))
                        if dist < best_dist:
                            best_dist = dist
                            best_agv = agv_id
                            best_target = t
                if best_agv is None:
                    print("No movable choice (no Z, no blocker to move, no cycle-break).")
                    break

            # --- 경로 계획 실행 ---
            cur_time = planned_paths[best_agv][-1][1] if planned_paths[best_agv] else 0

            if self.AGVs[best_agv].cur_state is None or best_target is None:
                print(f"[WARN] AGV {best_agv}: cur_state or target missing; skipping.")
                break

            came_from, cost, lb, end_tuple = self.compute_plan_prioritized(
                best_agv,
                self.AGVs[best_agv].cur_state,
                best_target,
                constraints,
                cur_time
            )
            if came_from is None:
                print(f"[FAIL] No path for AGV {best_agv} to {best_target}.")
                break

            plan_path = self.reconstruct_path(
                came_from,
                self.AGVs[best_agv].cur_state,
                end_tuple.state,
                cost
            )

            # 경로 누적
            if not planned_paths[best_agv]:
                planned_paths[best_agv] = plan_path
            else:
                planned_paths[best_agv] = planned_paths[best_agv] + plan_path[1:]

            # 제약 갱신
            constraints[best_agv] = self.build_constraint_table_prioritized(
                best_agv,
                planned_paths[best_agv]
            )

            # 상태 갱신: 실제 위치를 목표로 이동했다고 간주
            self.AGVs[best_agv].cur_state = best_target

            # --- [NEW] goal 달성 시 goal_idx 업데이트 및 완료 여부 체크 ---
            agv = self.AGVs[best_agv]
            if agv.dst_state_list:
                print("DSTSTATELIST", agv.dst_state_list)
                # 현재 타겟(goal_idx)이 best_target이면 한 스텝 진전
                if goal_idx.get(best_agv, 0) < len(agv.dst_state_list) and \
                        best_target == agv.dst_state_list[goal_idx[best_agv]][0]:
                    goal_idx[best_agv] += 1

                # 모든 goal을 방문했으면 완료 집합에 추가
                if goal_idx[best_agv] >= len(agv.dst_state_list):
                    visited_once.add(best_agv)

        # 결과 정리
        all_path = []
        for i in range(n_agents):
            agv_id = i + 1
            if not planned_paths[agv_id]:
                cur_id = self.AGVs[agv_id].cur_state if self.AGVs[agv_id].cur_state else 'S,0,0'
                planned_paths[agv_id] = [(cur_id, 0)]
            self.AGVs[agv_id].path = planned_paths[agv_id]
            all_path.append(planned_paths[agv_id])

        elapsed = time.time() - self.start_time
        print("ALl_path_prioritized", all_path)
        return all_path, elapsed

    def build_dependency_graph(self, goal_idx=None):
        G = nx.DiGraph()
        # 1) 모든 AGV를 노드로 추가 (cur==goal이어도 추가!)
        for agv_id, agv in self.AGVs.items():
            if agv.cur_state is None or not agv.dst_state_list:
                continue
            G.add_node(agv_id)

        # 2) 간선: i의 goal == j의 현재면 (i -> j)
        for i, agv_i in self.AGVs.items():
            if agv_i.cur_state is None or not agv_i.dst_state_list:
                continue
            if goal_idx:
                if goal_idx[i] >= len(agv_i.dst_state_list):
                    goal_idx_agv = -1
                else:
                    goal_idx_agv = goal_idx[i]
            else:
                goal_idx_agv = 0
            goal_i = agv_i.dst_state_list[goal_idx_agv][0]
            for j, agv_j in self.AGVs.items():
                if i == j or agv_j.cur_state is None:
                    continue
                curr_j = agv_j.cur_state

                # if curr_j == goal_i:

                if curr_j == goal_i or curr_j in self.convert_id_to_state(goal_i).affect_state:

                    G.add_edge(i, j)

                num1 = int(agv_i.cur_state.split(',')[1])
                num2 = int(goal_i.split(',')[1])

                # 2. (숫자1, 숫자2) 형태의 튜플 키 생성
                search_key = (num1, num2)
                tempbblockinglist = []
                if search_key in self.critical_nodes:
                    for state in self.critical_nodes[search_key]:
                        if state in self.convert_id_to_state(curr_j).affect_state:
                            G.add_edge(i,j)
                            break





        # 노드가 하나도 없을 때 예외처리(선택)
        if len(G.nodes) == 0:
            for agv_id, agv in self.AGVs.items():
                if agv.cur_state is not None:
                    G.add_node(agv_id)
        return G

    def detect_cycles(self, G):
        """
        Detect cycles in the dependency graph.
        """
        cycles = list(nx.simple_cycles(G))
        return cycles

    def find_nearest_empty_terminal(self, agv_id, empty_terminals, heuristic_table):
        """
        Find the nearest empty terminal for a given AGV.
        """
        min_dist = float('inf')
        best_terminal = None
        for terminal in empty_terminals:
            dist = self.highway_heuristic_table.get(self.AGVs[agv_id].cur_state, {}).get(terminal, float('inf'))
            # print("Terminalddd",dist,terminal)
            if dist < min_dist:
                min_dist = dist
                best_terminal = terminal
        return best_terminal, min_dist

    def select_agv_to_break_cycle(self, cycle_agents, empty_terminals, heuristic_table):
        """
        Select the best AGV in the cycle to move to an empty terminal to break the cycle.
        """
        best_agv = None
        best_terminal = None
        best_distance = float('inf')

        for agv_id in cycle_agents:
            terminal, dist = self.find_nearest_empty_terminal(agv_id, empty_terminals, heuristic_table)
            if dist < best_distance:
                best_distance = dist
                best_agv = agv_id
                best_terminal = terminal

        return best_agv, best_terminal, best_distance

    def compute_path_from_list(self,vehicle,simtime):
        plan_path = []
        initial_state = self.convert_id_to_state(vehicle.planned_path[-1][0])
        # for i in range(len(vehicle.dst_state_list)): #장기 plan을 짤 경우 변경 필요
        for i in range(1):
            dst_state = vehicle.dst_state_list[i]
            if vehicle.dst_state_list[i][1] == 1:
                simtime = simtime + self.loading_time
            came_from, cost = self.compute_plan(vehicle.id,initial_state,dst_state, simtime)
            if came_from != None:
                plan_path = self.reconstruct_path(came_from, initial_state,dst_state[0], cost)
            vehicle.planned_path = plan_path

    def compute_total_cost(self, List = None):
        if List == None:
            cost = []
            for i in range(len(self.AGVs)):
                cost.append(self.AGVs[i+1].planned_path[-1][1])
        else:
            cost = []
            for i in range(len(List)):
                cost.append(List[i][-1][1])
        return cost

    def idle_procedure(self, vehicle):
        self.block_point(vehicle.cur_loc)

    def convert_route_for_AutoMod(self, vehicle):
        Move_convert = {Direction.Left: ['l', 'r'], Direction.Right: ['r', 'l'], Direction.Down: ['d', 'u'],
                        Direction.Up: ['u', 'd']}
        AutoMod_list = []
        # print("Vehicle fixedpath", vehicle.fixed_path)
        # print("Vehicle plannedpath", vehicle.planned_path)
        detour = False
        if len(vehicle.fixed_path) <= 1:
            AutoMod_list.append(vehicle.cur_loc_AutoMod)
        else:
            prev_point = vehicle.cur_loc
            next_point = vehicle.fixed_path[0]
            #
            # if len(vehicle.fixed_path) >= 2:
            #     if vehicle.fixed_path[1] == vehicle.cur_loc:
            #         detour = True
            if prev_point != next_point:
                print("prev,next",prev_point,next_point)
                move_dir = self.get_move_direction(prev_point, next_point)
                prev_point_AutoMod = prev_point + '_' + Move_convert[move_dir][0]
                next_point_AutoMod = next_point + '_' + Move_convert[move_dir][1]
                if prev_point_AutoMod != vehicle.cur_loc_AutoMod:
                    AutoMod_list.append(prev_point_AutoMod)
                AutoMod_list.append(next_point_AutoMod)

            for i in range(len(vehicle.fixed_path) - 1):
                prev_point = vehicle.fixed_path[i]
                next_point = vehicle.fixed_path[i + 1]
                if next_point[0] == 'T' and i == len(vehicle.fixed_path)-2:
                    AutoMod_list.append(vehicle.cur_loc_AutoMod)
                    _,angle_prev,angle = next_point.split(',')
                    next_point_AutoMod = prev_point + '_' + self.direction_list[self.angle.index(int(angle))]
                    if vehicle.cur_loc_AutoMod != next_point_AutoMod:
                        AutoMod_list.append(next_point_AutoMod)
                    #05016 933
                    # else:
                    #     for j in range(2):
                    #         # cur_state = self.Move_state[vehicle.planned_path[0][0]]
                    #         cur_state = self.convert_id_to_state(vehicle.planned_path[0][0])
                    #         cur_state.rsv_time_table.sort()
                    #
                    #         # print("Rsvt9", cur_state.rsv_time_table, cur_state.id)
                    #         cur_state.rsv_time_table.pop(0)
                    #         if len(cur_state.rsv_time_table) == 0:
                    #             cur_state.interval_list = [(0, float('inf'))]
                    #         # else:
                    #             # print("impossible in rsvt9")
                    #
                    #         for aff_states in cur_state.affect_state:
                    #             self.convert_id_to_state(aff_states).rsv_time_table.sort()
                    #             # print("Rsvt10",self.convert_id_to_state(aff_states).rsv_time_table, cur_state.id)
                    #             self.convert_id_to_state(aff_states).rsv_time_table.pop(0)
                    #             if len(self.convert_id_to_state(aff_states).rsv_time_table) == 0:
                    #                 self.convert_id_to_state(aff_states).interval_list = [(0, float('inf'))]
                    #             # else:
                    #                 # print("impossible in rsvt10")
                    #
                    #         vehicle.planned_path.pop(0)
                    return AutoMod_list
                # elif next_point[0] == 'T' and len(vehicle.fixed_path) > 2:
                #     print("S1231omethings wrong at convert route for automod")
                #     _, _, angle = next_point.split(',')
                #     next_point_AutoMod = prev_point + '_' + self.direction_list[self.angle.index(int(angle))]
                #     AutoMod_list.append(next_point_AutoMod)
                #     return AutoMod_list
                else:  ##0517 아래 else문 밖으로 빼고 위에 Automod list 아래까지주석처리 해제
                    move_dir = self.get_move_direction(prev_point, next_point)
                    prev_point_AutoMod = prev_point + '_' + Move_convert[move_dir][0]
                    next_point_AutoMod = next_point + '_' + Move_convert[move_dir][1]
                # if prev_point_AutoMod != vehicle.cur_loc_AutoMod:
                #     AutoMod_list.append(prev_point_AutoMod)
                if AutoMod_list != [] and AutoMod_list[-1] == prev_point_AutoMod:
                    AutoMod_list.append(next_point_AutoMod)
                    continue
                else:
                    AutoMod_list.append(prev_point_AutoMod)
                    AutoMod_list.append(next_point_AutoMod)

        if vehicle.dst_loc_AutoMod != None:
            if AutoMod_list == []:
                if vehicle.cur_loc_AutoMod[:-2] == vehicle.dst_loc_AutoMod[:-2]:
                    AutoMod_list.append(vehicle.dst_loc_AutoMod)

            elif AutoMod_list[-1][:-2] == vehicle.dst_loc_AutoMod[:-2] and AutoMod_list[-1] != vehicle.dst_loc_AutoMod:
                AutoMod_list.append(vehicle.dst_loc_AutoMod)

        if AutoMod_list == []:
            AutoMod_list.append(vehicle.cur_loc_AutoMod)

        if len(AutoMod_list) >= 2:
            if AutoMod_list[0] == AutoMod_list[1]:
                print("Detour!", AutoMod_list, vehicle.cur_loc)
                AutoMod_list.pop(0)

        return AutoMod_list

if __name__ == '__main__':
    A = ACS('img/output_map.txt')
    for i in range(len(A.AGVs)):
        A.compute_path_from_list(A.AGVs[i+1],0)
    cost = A.compute_total_cost()
    print(sum(cost))
