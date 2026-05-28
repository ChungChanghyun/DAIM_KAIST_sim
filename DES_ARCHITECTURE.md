# 통합 시뮬레이터 DES 아키텍처 (C 재구현용 명세)

이 문서는 `vis_mcs_unified.py` 를 정점으로 하는 KAIST 통합 물류 시뮬레이터의
이산사건(DES) 구조를 **다른 언어(C 등)에서 정확히 재구현**할 수 있도록
정리한 것이다. 4개의 독립 이동 엔진(OHT / AGV / 3DS / Elevator)을 단일
MCS(Material Control System) 경계에서 통합한다.

> 핵심 사실 4가지 (먼저 읽을 것)
> 1. **단일 우선순위 큐 + `(t, seq)` FIFO 정렬**이 모든 인과를 강제한다.
> 2. **`token` 정수**가 모든 stale-event 방어의 보편 메커니즘 (phase 전이마다 +1, 처리 시 비교).
> 3. **MCS 는 시스템 비의존적** — 각 이동 엔진과 5개 callback 으로만 대화.
> 4. **이동 모델이 2종**: OHT 는 연속 물리(leader-following + ZCU lock), AGV/3DS 는 offline pkl 충돌기하 + SIPP + TAPG precedence. MCS 경계에서만 합쳐진다.

---

## 0. 전체 구성

```
                    ┌─────────────────────────────────────┐
                    │  MCS (mcs_unified.py)                │
                    │  단일 heap, Load 수명주기, Port 생산  │
                    │  register_system(5 callbacks)        │
                    └───┬─────────┬─────────┬──────────┬───┘
       on_dispatch / is_free / get_node / get_distance  │
            ┌───────────┘         │         │          └────────────┐
            ▼                     ▼         ▼                       ▼
   ┌──────────────┐   ┌──────────────────┐  ┌──────────────┐  ┌──────────┐
   │ OHT          │   │ AGV              │  │ 3DS ×3 floor │  │ Elevator │
   │ graph_des_v6 │   │ env_tapg (coarse)│  │ env_tapg     │  │ elevator │
   │ 연속 물리     │   │ SIPP+TAPG        │  │ SIPP+TAPG    │  │ 수직 DES  │
   │ ZCU lock      │   │ segment lock     │  │              │  │          │
   └──────────────┘   └──────────────────┘  └──────────────┘  └──────────┘
            │ 모두 PklMapGraph 또는 GraphMap 위에서 동작
            ▼
   layout JSON (Maps/KaistTB.map_latest.json)
     ├─ OHT: GraphMap(area_filter='OHT_A') 직접 로드 (pkl 불필요)
     └─ AGV/3DS: gen_*.py → .pkl (충돌 기하 + S/M/R state) → PklMapGraph
```

이벤트 루프의 기본형(모든 엔진 공통):
```
step(t_now):
    while heap and heap[0].t <= t_now:
        ev = heap.pop_min()
        dispatch(ev)              # 핸들러 호출 → 상태 전이 + 새 이벤트 post
    update_positions(t_now)       # (env_tapg 만) 위치 적분 / 도달 감지
```

---

## 1. 공통 이벤트 인프라

### 1.1 Event 객체

모든 엔진이 동일 형태의 이벤트를 쓴다.

| 필드 | 의미 |
|--|--|
| `t` (float) | 발화 시각 (sim seconds) |
| `seq` (int) | 생성 순번 (monotonic, 전역 카운터) |
| `kind` (str) | 이벤트 종류 |
| `vid` / `agent_id` | 대상 차량 id (시스템 이벤트는 sentinel, 예: -1) |
| `data` (dict/tuple) | payload (lock_id, bnd_node 등) |

**정렬(`__lt__`)**: `(t, seq)` 만 비교. `kind/vid/data` 는 비교 제외.
→ **같은 t 면 먼저 생성된(seq 작은) 이벤트가 먼저**. C 에서는 min-heap 에
`(t, seq)` 를 키로 쓰면 된다.

### 1.2 `_post` 의 우선순위 offset (graph_des_v6 한정, 중요)

OHT 엔진에서 `EV_REPLAN` 은 post 시 `t += 1e-9` (`_REPLAN_PRIORITY_DELTA`).
→ 같은 t 의 다른 이벤트(SEG_END, BOUNDARY 등)가 **항상 REPLAN 보다 먼저**
처리되게 보장. C 재구현 시 이 offset 을 그대로 둘 것 (생략하면 tie 순서가
바뀌어 동작이 달라짐).

### 1.3 이벤트 취소 없음 — stale 자가검증

**어떤 엔진도 heap 에서 이벤트를 취소하지 않는다.** 차량이 re-plan 되면
이전 plan 이 post 한 이벤트가 여전히 heap 에 남는다. 각 핸들러가 발화 시점에
*아직 유효한지* 스스로 검사한다:
- `token` 불일치 → drop (MCS).
- `EV_ZCU_EXIT`: `v.seg_from ∈ _lock_exit_nodes[lock_id]` 아니면 drop.
- `EV_TIMEOUT`: `(state, waiting_at_zcu, reroute_count)` 토큰 검사.
- `EV_BOUNDARY`: `_relevant_zones` 비면 stale recovery.

→ C 에서는 이벤트 취소 자료구조를 만들지 말고 **핸들러별 stale 검사**를
복제할 것.

### 1.4 부동소수 비교 (verbatim 복제 필수)

`t_nxt > t_cur + 1e-9`, `seg_offset >= seg_len - SEG_CROSS_EPS`,
`dist >= edge_length - 1e-6` 등의 epsilon 비교는 **대수적으로 동등한 형태로
바꾸지 말 것** (`a - b > c` ≠ `a > b + c` in float). 어떤 이벤트가 post 되는지
가 달라진다. 모든 키 lookup 은 `round(t, 6)` 으로 시각을 양자화한다 — 동일하게.

---

## 2. OHT 엔진 — `graph_des_v6.py` (연속 물리)

OHT 는 **연속 사다리꼴 속도 프로파일** + **leader-following** + **ZCU zone
lock** 의 완전 이벤트 기반 엔진이다. 도달 시각을 *미리 계산해서* SEG_END
이벤트로 예약한다 (env_tapg 의 위치적분 방식과 대조).

### 2.1 차량 상태기계

상태: `IDLE / ACCEL / CRUISE / DECEL / STOP / LOADING`
(LOADING 은 선언돼있으나 엔진 자체는 거의 안 씀 — dwell 은 `job_state`
+ 외부 JobManager/Bridge 로 관리. 통합 sim 에서는 MCSOHTBridge 가 dwell 동안
`v.state = LOADING` 으로 set 해서 정지+시각화).

운동학 파라미터(기본): `v_max=3600`, `a_max=500`, `d_max=500` (mm, mm/s²).
`h_min = vehicle_length + 200` (= 안전 차간거리). `vehicle_length` 는
layout 의 vehicleModels OHT dimension (KaistTB: 1108mm).

상태 전이 driver:
- `_go(t,v,target_v)`: vel < target-50 → ACCEL(+a_max); vel > target+10 → DECEL(-d_max); else CRUISE(0).
- `_on_phase_done`: committed acc 부호로 ACCEL/DECEL/CRUISE.
- `_brake_to_stop`: DECEL 시작.
- `_on_stopped` / arrival / teleport / plan_boundary≤0 → STOP.

### 2.2 이벤트 종류 (12개)

| 이벤트 | 언제 post | 핸들러 | 동작 요약 |
|--|--|--|--|
| `EV_START` | `start_all()` t=0, leader-first 순 | `_replan` | 최초 plan |
| `EV_REPLAN` | plan 변화 모든 지점 (+1e-9 offset) | `_replan` | 핵심 재계획 |
| `EV_SEG_END` | `_schedule_plan_events` (committed seg 마다), `_brake_to_stop` | `_on_seg_end` | 세그먼트 경계 통과 = occupancy/exit/leader 갱신. **재계획 안 함** (단 leader 변경 시 예외) |
| `EV_PHASE_DONE` | acc 전이 시각마다 | `_on_phase_done` | acc 동기화 (committed_traj 에서) |
| `EV_STOPPED` | v=0 도달 시각 | `_on_stopped` | STOP 확정 + follower wake + diverge lock 해제 |
| `EV_BOUNDARY` | `stop_reason=='zcu'` 일 때 brake-start 시각 | `_on_boundary` | ZCU lock 획득 시도 (payload: bnd_node) |
| `EV_ZCU_GRANT` | `_zone_release` 가 다음 waiter 에게 | `_on_zcu_grant` | lock 양도받음 → replan |
| `EV_ZCU_EXIT` | zone exit 도달 시각 | `_on_zcu_exit` | lock 해제 (payload: lock_id) |
| `EV_TIMEOUT` | `_zone_wait` 가 15s 후 | `_on_timeout` | ZCU-stuck 우회 재경로 |
| `EV_JOB_CREATE` | JobManager (vid=-1) | `job_mgr.on_create_event` | job 생성 (OHT-only test) |
| `EV_LOAD_DONE` | dwell 타이머 | `_on_load_done` | 적재 완료 → 다음 leg |
| `EV_UNLOAD_DONE` | dwell 타이머 | `_on_unload_done` | 하역 완료 → idle |

### 2.3 `_replan` (엔진의 두뇌)

발화: EV_START / EV_REPLAN / leader 변경된 SEG_END 등.
순서:
1. **commit-alive 조기탈출**: `commit_end_t > t+1e-6` 면 skip + retry REPLAN 을
   `commit_end_t + 1e-9` 에 dedup post (`_retry_replan_at`, 1e-6 tol).
2. `advance_position(t)` (= 위치/속도 적분 갱신).
3. 도착 판정 (at_path_end / at_dest / dest_reached) → STOP + `job_mgr.on_arrive`
   (job 있으면) 또는 `_notify_followers`.
4. `target_v = min(v_max, segment_speed)`.
5. `_find_first_boundary` (다음 미소유 ZCU 경계).
6. `_update_leader` + `_try_push`.
7. caps 계산: `leader_free = gap_d + leader_committed_remaining - h_min`,
   `dest_dist`, `path_end_dist`.
8. teleport-leader (leader_free≤0) → STOP(stop_reason='leader').
9. **commit horizon 계산** (§2.6).
10. `plan_boundary = min(commit_dist, leader_free, leader_traj_end_x, forward_stop)`.
11. `stop_reason` 결정 (zcu/leader/dest) + X marker pin.
12. 미소유 경계에서 정지 중이면 lock 획득 시도; 거부 시 `_zone_wait`.
13. `plan_boundary≤0` → STOP.
14. else `_go` + `_schedule_plan_events`.

### 2.4 ZCU lock 프로토콜

자료구조: `_zone_lock[lock_id]→holder`, `_zone_waiters[lock_id]→[Vehicle]` (FIFO),
`_boundary_to_zones`, `_exit_to_zones`, `_lock_exit_nodes`, `_seg_to_zone`.
`lock_id = f"{node_id}_{kind}"`, kind ∈ {merge, diverge}.

- **Merge zone**: boundary = merge 노드의 선행 노드들, exit = merge 노드 자체.
- **Diverge zone**: boundary = diverge 노드, exit = 각 후행 노드.

획득 (`_on_boundary`):
- `_relevant_zones` → `_try_acquire_all_zones` (atomic all-or-nothing, 실패 시 롤백).
- 거부 → `_zone_wait` (FIFO 등록 + EV_TIMEOUT post).
- 획득 → `passed_zcu.add` + replan.

해제:
- `EV_ZCU_EXIT` (주 경로, exit 노드 검증).
- `_update_occupancy` safety net (crossed set 에 exit 있으면).
- `_on_stopped` diverge-only (merge 는 절대 정지로 해제 안 함).
- `_assign_destination` stale-lock 해제 (새 path 가 exit 안 거치면).

양도 (`_zone_release`): waiter FIFO pop + **path-aware skip** (현재 path 가
exit 안 거치는 stale waiter 건너뜀) → 직접 양도 (`_zone_lock=next_v`) + EV_ZCU_GRANT.

위반 감지 (`_check_zcu_entry`): Type1 NO_LOCK (lock 없이 zone 진입),
Type2 SIMULTANEOUS (zone 안 2대+).

### 2.5 Leader-follower

- `_update_leader(v,t)`: 전방 최근접 차량 탐색. same-segment(`_seg_occupants`) →
  corridor → forward walk(`leader_walk_cap = brake_dist_v_max + h_min`) →
  path-end peek. `v.leader / leader_dist / forward_stop_cap` set.
- `gap(follower,t)`: `(경로거리, leader_vel)`. `leader_dist` 캐시 재사용.
- `_notify_followers(t,leader)`: leader 의 모든 follower 에게 EV_REPLAN.
  **mutual-link skip** (상호 leader cycle → ping-pong 방지),
  **per-follower-per-t dedup** (`last_notify_post_t`).
  = **유일한 follower wake 메커니즘 (polling 없음)**. leader 가 안 움직이면
  follower 영원히 BLOCKED → leader 의 모든 상태변화 지점에서 notify 필수.

### 2.6 trajectory commit 모델

| 필드 | 의미 |
|--|--|
| `committed_traj` | `[(t, dist_abs, vel, acc)]` phase 시작점. `state_at` 가 읽음 |
| `committed_segs` | `[(t_enter, t_exit, seg_key, plan_dist)]` 점유 세그먼트 |
| `commit_end_idx` | 물리적 commit 끝 path index. dispatch/push/reroute 가 `path[path_idx:commit_end_idx+1]` 보존해야 |
| `commit_horizon_dist` | commit/lock 스케줄 범위 = `min(eff_brake_dist, bnd_dist)` |
| `commit_end_t` | 마지막 `_schedule_plan_events` 가 post 한 이벤트 시각 상한 |

**commit horizon 거리** (= 얼마나 멀리 trajectory/lock 을 commit 하나):
```
brake_dist_v_max = v_max² / (2·d_max)          # 안전 invariant 용
eff_v_max = max(commit 범위 안 segment speed)   # 실제 도달 속도
eff_brake_dist = eff_v_max² / (2·d_max)
commit_horizon_dist = min(eff_brake_dist, 다음_ZCU경계_거리)
```
→ KAIST 처럼 segment speed(1000) << v_max(3600) 면 commit 이 실제 속도 기준
(~1m) 으로 축소. `physical_reach = brake_dist_v_max + h_min` 은 leader-visibility
안전 invariant 라 v_max 기준 유지.

**Phase 2.1 truncation**: transient stop (zcu/leader/horizon) 시
committed_traj 를 brake-start 시각에서 자름 (감속/정지 꼬리는 commit 안 함).
follower 는 `_leader_traj_from_now` 에서 worst-case 감속 꼬리를 합성해 추론.

### 2.7 메인 루프

`step(t_now)` → `run_until(t_now)` (heap pop while t≤t_now, vid=-1 은
job_mgr 라우팅, else `_dispatch`) → `query_positions(t_now)` (render only,
상태 변경 X).

---

### 2.8 주행 Trajectory 상세 — committed_traj 와 follower 설계

이 절은 OHT 의 *연속 주행 궤적* 이 어떻게 표현되고, 후행 OHT 가 선행 OHT 의
궤적을 참조해 자신의 궤적을 설계하는지를 C 재구현 수준으로 설명한다.
관련 파일: `vehicle_state.py` (`state_at`), `velocity_profile.py`
(`compute_velocity_profile`), `graph_des_v6.py` (`_leader_traj_from_now`,
`_leader_traj_end_x`, `_commit_state`, `_schedule_plan_events`).

#### 2.8.1 committed_traj — 무엇인가

차량의 **확정된 운동 궤적**을 phase 단위로 저장한 리스트:
```
committed_traj = [(t_i, dist_i, vel_i, acc_i), ...]   # t 오름차순
```
각 entry = **상수 가속도 phase 의 시작점**. phase `i` 는 `t_i` ~ `t_{i+1}` 까지
지속하며 그 동안:
```
vel(τ)  = clamp0( vel_i + acc_i · (τ − t_i) )          # ≥ 0 으로 clamp
dist(τ) = dist_i + vel_i·(τ−t_i) + ½·acc_i·(τ−t_i)²    # 절대 누적거리
```
- `dist_i` = **절대 누적거리** (path 시작부터, mm). segment offset 아님.
- 마지막 entry 의 phase 는 *앞으로 연장* (cruise 면 등속, decel 이면 v=0 도달 후
  그 자리 정지).

병행 자료: `committed_segs = [(t_enter, t_exit, seg_key, plan_dist), ...]`
= "어느 시각에 어느 segment 위에 있나" + 그 segment 시작점의 절대거리.

> 핵심 성질 (= "committed" 의 의미): 한 번 기록되면 `commit_end_t` 까지
> **소급 변경 불가**. replan 은 그 끝에서 *연장만* 한다 (`_get_plan_start`
> Phase 1). → 다른 차량이 이 궤적을 읽고 신뢰할 수 있는 근거.

#### 2.8.2 state_at(v, t) — 임의 시각의 상태 조회 (순수 함수)

```
VehicleState(t, dist, vel, acc, seg_key, seg_offset, path_idx) = state_at(v, t)
```
- **committed_traj / committed_segs / path 만 읽음.** `v.t_ref/vel/acc`
  (런타임 캐시 = 마지막 발화 이벤트의 단일 phase) 와 *무관*.
- binary search 로 (a) `t` 포함 phase, (b) `t` 포함 segment 를 찾음.
- decel phase 면 `dt` 를 v=0 도달 시점으로 clamp (이후 정지 유지).
- **side-effect 없음 + dispatch 순서 무관** → cross-vehicle 조회(gap, leader
  lookahead, follower 계획) + 미래시각 조회(commit-end-start replan) 에 사용.

> C 재구현: 이 함수가 "차량의 진실"이다. 런타임 `vel/acc` 는 보조 캐시일 뿐,
> 모든 cross-vehicle 판단은 `state_at` 로 한다.

#### 2.8.3 자기 궤적 생성 — compute_velocity_profile

`_schedule_plan_events` (replan tail) 에서:
```
traj_rel, c_segs_rel = compute_velocity_profile(
    seg_lengths, seg_speeds, seg_keys,    # plan_start 부터의 path 슬라이스
    seg_offset = seg_offset_start,
    v0 = v0_start,                        # plan_start 속도
    plan_boundary,                        # = min(commit_dist, leader_free, leader_traj_end_x, forward_stop)
    v_max, a_max, d_max,
    t_now = t_start,
    leader_traj, leader_dist_offset, h_min)  # ← 선행차 참조 (아래)
```
- 출력 `traj_rel` = 상대 궤적 → `base_dist = abs_dist_start` 더해 절대화 →
  `committed_traj` 에 append (Phase 2.1 이면 brake-start 에서 truncate).
- 프로파일은 **plan_boundary 에서 정지 가능**하도록 사다리꼴(accel/cruise/decel)
  로 생성 (= look-ahead braking, §8 참조).

#### 2.8.4 후행 OHT 가 선행 궤적을 참조하는 방법 (★ 핵심)

후행차(follower)의 `compute_velocity_profile` 호출이 받는 3개 인자:

1. **leader_traj** = `_leader_traj_from_now(leader, t_start)`:
   - leader.committed_traj 를 `t_start` 이후로 슬라이스 (첫 entry 는 `t_start`
     시점 상태로 보간).
   - **worst-case 감속 꼬리 합성**: Phase 2.1 로 leader 의 commit 이 brake-start
     (등속 vel>0) 에서 끝나므로, follower 는 "leader 가 거기서 d_max 로 정지할
     것"이라 가정하고 `(last_t + v/d_max, d_stop, 0, 0)` entry 를 덧붙인다.
     → leader 가 실제로 cruise 연장하면 *더 빨라질 뿐*이라 gap 은 안전 쪽으로만
     벌어짐 (보수적).

2. **leader_dist_offset** = `gap_d` = `gap(v, t)` (현재 follower→leader 경로거리).

3. **h_min** = 안전 차간거리.

내부 `_compute_follower_trajectory` 의 핵심 로직:
```
각 위치 x 에서:
   effective_speed(x) = min( seg_speed(x),  leader_cap(x) )
   leader_cap(x) = "leader 가 (x + h_min) 지점에 도달했을 때의 leader 속도"
                 = _leader_state_at_dist(leader_traj, d_L).vel
                   where d_L = x + h_min − leader_dist_offset + x_L0
```
즉 **follower 의 위치 x 에서 낼 수 있는 최대속도 = 그보다 h_min 앞선 지점을
지나는 leader 의 속도**. → follower 가 leader 의 *과거 궤적 envelope* 안쪽에
머물도록 cap → h_min 위반 불가.
- 추가: 시작점 brake-feasibility waypoint (`x = leader_pos − h_min`, v=v_L_now)
  로 *현재* 근접 상황에서도 즉시 멈출 수 있게 보장.
- leader_traj 의 각 phase 경계 + segment 경계마다 구간 분할 → 각 구간 보수적
  cap → backward/forward pass 로 최종 사다리꼴 궤적.

#### 2.8.5 plan_boundary 의 leader cap 항

`_replan` 에서 follower 의 plan_boundary 를 제한하는 두 leader 항:
- `leader_free = gap_d + _leader_committed_remaining(leader) − h_min`
  = leader 의 committed 잔여거리까지 고려한 "정지 없이 갈 수 있는 거리".
- `leader_traj_end_x` = leader committed 궤적이 끝나는 x (그 너머는 follower 가
  계획 못 함 — leader 의 미래가 아직 미확정).

`leader_free ≤ 0` → 즉시 STOP(stop_reason='leader'). 아니면 위 effective_speed
cap 으로 부드럽게 따라감.

#### 2.8.6 전체 흐름 (follow 한 사이클)

```
1. leader 가 전진 → _commit_state / _schedule_plan_events → committed_traj 연장
2. leader 의 _notify_followers(t) → follower 에게 EV_REPLAN post
3. follower _replan:
     gap_d = gap(v, t)
     leader_traj = _leader_traj_from_now(leader, t_start)   # + worst-case stop tail
     plan_boundary = min(..., leader_free, leader_traj_end_x)
     compute_velocity_profile(..., leader_traj, gap_d, h_min)
        → effective_speed(x) = min(seg_speed, leader_cap(x))
     committed_traj 연장 + 이벤트 예약
4. leader 가 또 전진 → 2 로 (실제 brake 전에 재계획 → 끊김 없는 cruise)
5. leader 가 진짜 정지하면 follower 의 decel-to-stop 이 실현됨
```

#### 2.8.7 C 재구현 핵심

- committed_traj entry = `(t, abs_dist, vel, acc)`, phase 상수가속. `state_at` 는
  binary search + phase 내 운동방정식 + decel v=0 clamp.
- follower cap = "x+h_min 앞 leader 속도" (leader_traj 의 거리→속도 역조회).
- leader_traj 에 **worst-case 정지 꼬리** 합성 (Phase 2.1 truncation 의 짝).
- commit 불변 + 연장-only → cross-vehicle 일관성 (dispatch 순서 무관). 이게
  없으면 leader 궤적이 follower 계산 중 바뀌어 race.

---

## 3. AGV / 3DS 엔진 — `env_tapg.py` (위치적분 + TAPG)

env_tapg 는 **위치기반 DES** 다. 이벤트 큐는 "전진 시도" 결정만 발화하고,
실제 운동과 도달 감지는 매 tick `_update_positions` 적분으로 한다.
2개 모드:
- **SIPP 모드** (3DS): TAPG DAG 에 cross-agent 시간 간선 → 타이밍 무관 충돌회피.
- **coarse 모드** (AGV, `_coarse_mode=True`): cross 간선 없음. live-occupancy claim.

### 3.1 상태기계

상태: `idle / moving / rotating / waiting / done`.
파라미터: env 생성 시 `accel=500, decel=500` (mm/s²), `ROTATION_TIME_90=1.0s`,
`ANGULAR_SPEED = π/2 rad/s`.

### 3.2 이벤트 종류 (단 2개)

| 이벤트 | 언제 | 핸들러 | 동작 |
|--|--|--|--|
| `TRY_ADVANCE` | 매 도달 후, 노드 완료 후, wakeup, periodic | `_on_try_advance` | 핵심 전진 결정 |
| `LOAD_DONE` | L state 진입 시 `+cost` 에 | `_on_load_done` | dwell 완료 → 전진 |

**도달 이벤트는 없다** — `_update_positions` 가 거리/각도 임계로 감지.

### 3.3 raw_path / 상태 ID 포맷

`raw_path = [(state_id, cbs_cost), ...]`. state_id:
- `S,node[,heading]` — 정지. S-walk 로 즉시 통과.
- `M,from,to` — 이동. 실행 중 {from,to} 점유, idle 시 {from} 만.
- `R,node,from_deg,to_deg` — 제자리 회전 (최단방향).
- `L,node` — LOADING/UNLOADING dwell. cost = dwell time.

`path_idx` = 현재 상태 index. `claim_idx` = 원자 예약 범위 `[path_idx, claim_idx)`
의 exclusive 끝 (= 멈출 수 없는 구간). `_wanting` = 다음 진입 희망 노드 (cycle 감지용).

### 3.4 `_on_try_advance` 로직

1. MOVING/ROTATING/DONE 이면 return.
2. **S-walk**: 현재 S 면 좌표 snap + 다음 M/R 검사 (`claim_idx` 안이면 통과,
   밖이면 `_is_claimable` AND `_try_claim_next`). 실패 시 break.
3. path 끝 → DONE.
4. L state → WAITING + LOAD_DONE post.
5. S 차단 → WAITING + `wait_queues` 등록 (`_wait_start_t` set).
6. claim 범위 안 → 무조건 실행 (M→`_start_move`, R→`_start_rotate`).
7. claim 범위 밖 → `_is_claimable` / `_try_claim_next` 검사, 실패 시 WAITING.

### 3.5 coarse 모드 (AGV) claim

- `_is_claimable_coarse(nk, aid)`: 다른 agent 의 `[path_idx, claim_idx)` 점유 노드와
  node-level overlap 있으면 거부. release-timing 보정 (path_idx 가 M/R 이면 직전 S 포함).
  DONE agent 는 `path[-1]` 점유.
- `_try_claim_next`: `claim_idx` 부터 다음 M/R 까지 walk →
  - cut node 면 `_cut_admission_end_idx` (port 까지 OR cut zone 통과해 첫 grey 까지).
  - 일반이면 `_coarse_segment_end` (= 다음 *non-cut* rest place 까지 atomic).
  - `_would_claim_create_cycle` 검사 (dep chain 따라가 자기 회귀 시 거부).
- **cut node** = port 진입 단방향 chain. 거기 멈추면 port 차단 → atomic 으로
  통과 강제.

### 3.6 WAITING wake 순서

`_complete_node(nk, t)`: `wait_queues[nk]` waiter pop + 노드 G 제거. coarse 면
**모든 WAITING agent 를 `(_priority, _wait_start_t)` 정렬**해 깨움:
- `_priority = -1` (cycle-push victim) 최우선.
- 동일하면 `_wait_start_t` 오름차순 = **먼저 기다린 게 먼저** (FIFO).
- 깨울 때 `TRY_ADVANCE` 를 `t + 1e-9` 에 post.

`_priority` 는 MOVING/ROTATING 진입 시 0 으로 reset.

### 3.7 위치 적분 (`_update_positions`)

MOVING: 사다리꼴 프로파일. `d_brake = v²/(2·decel)`. brake zone 진입 시
다음 M 가 claimable 이면 감속 안 하고 chaining (무정지 통과). 도달은
`dist_traveled >= edge_length - 1e-6`.
ROTATING: `angle_traversed += ANGULAR_SPEED·dt`, 도달 `>= angle_total`.
`_handle_arrival`: path_idx++, state=IDLE, 완료 노드 waiter wake,
chaining 가능하면 `_try_chain_move`.

### 3.8 동적 replan API

- `extend_agents_batch`: DONE agent path 교체 (DAG 정리 + 재구축).
- `append_agents_batch`: 기존 raw_path 뒤 이어붙임 (path_idx 유지).
- `recompute_earliest_schedule`: topological earliest-start 전파 (SIPP 시간 보정).

### 3.9 C 재구현 노트

- coarse 모드는 DAG 가 sequential 간선만 → per-agent index 범위 + node→owner 검사로
  표현 가능 (graph 불필요).
- 모든 노드 키 `round(t,6)`.
- 도달은 위치/각도 임계 기반 — 매 tick 적분 필수 (도달시각 예약 X).

---

## 4. MCS — `mcs_unified.py` (시스템 통합)

단일 heap. 모든 subsystem 이벤트 + MCS 이벤트가 한 heap. `system` 태그로 라우팅.

### 4.1 이벤트 종류 (5개)

| 이벤트 | 문자열 | post 시점 | 핸들러 |
|--|--|--|--|
| `LOAD_CREATED` | `MCS_LOAD_CREATED` | `_schedule_port_production` (Poisson `t+expovariate(rate/60)`) | `_on_load_created` |
| `TRY_ASSIGN` | `MCS_TRY_ASSIGN` | idle 발생 시 | `_do_assign` |
| `VEHICLE_ARRIVED` | `MCS_VEHICLE_ARRIVED` | **transport 층**이 `post_vehicle_arrived` | `_on_vehicle_arrived` |
| `DWELL_DONE` | `MCS_DWELL_DONE` | `_on_vehicle_arrived` 가 `t+dwell` | `_on_dwell_done` |
| (recipe) | `MCS_RECIPE_ARRIVAL` | `_schedule_recipe_arrival` | inline |

### 4.2 phase / load 상태기계

`VehicleJobState`: `IDLE → TO_PICKUP →(arrive src)→ LOADING →(dwell)→
TO_DELIVERY →(arrive dst)→ UNLOADING →(dwell)→ IDLE`. 전이마다 `token += 1`.

`LoadState`: `WAITING →(assign)→ ASSIGNED →(arrive src)→ ON_VEHICLE →(arrive dst)→
DELIVERED →(dwell)→ COMPLETED`.

### 4.3 token staleness

차량 binding `(vehicle_id, system, load, phase, token)`. phase 전이마다 token++.
transport 가 `VEHICLE_ARRIVED` / `DWELL_DONE` post 시 event 의 token 과
`b.token` 비교 → 불일치면 drop (재dispatch 로 무효화된 이전 leg).

### 4.4 register_system (5 callback)

```
register_system(system, port_nodes,
    on_dispatch(vid, goal_node, t),      # 이동 명령 (transport 가 경로 계획)
    is_vehicle_free(vid) -> bool,        # 진짜 idle 인지 (assign gate)
    get_vehicle_node(vid) -> node,       # 현재 노드 (matching)
    get_distance(src, dst) -> float,     # 경로 거리 (_select_nearest)
    port_prod_rate=loads/min)            # Poisson 생성률
```
통합 sim 의 prod_rate: OHT 0.3, AGV 0.1, 3DS 0.1 (per port/min).

### 4.5 `_do_assign`

system 별 idle 차량 그룹핑 (`is_vehicle_free` false 제외) → 대기 load oldest-first
정렬 → 각 load 에 `_select_nearest` (= `get_distance` 최소) 차량 배정 →
phase=TO_PICKUP, token++, `on_dispatch(vid, src, t)`. **cross-system 매칭 없음**.

### 4.6 Recipe (다단계 cross-system)

`RecipeStage(system, src, dst)` 의 list. `_on_dwell_done` UNLOADING 분기에서
다음 stage 있으면 차량 free + load 를 다음 stage src port 로 이동 (WAITING) +
`_do_assign`. 마지막 stage 면 완료 + WIP refill (closed cycle). 인접 stage 전이점은
두 시스템 graph 가 공유하는 노드 (LIFT 는 floor 매칭으로 검증).

---

## 5. Elevator — `elevator.py` (수직 DES)

`Elevator`, `ElevatorController`, `LiftRequest`. 층간 공유자원.
이벤트: `LIFT_MOVE_DONE / LIFT_XFER_DONE / LIFT_MOVE_TO_DONE`.
상태: `IDLE/MOVING/LOADING/UNLOADING`. lift agent_id 음수 (base -1000).
MCS API: `move_to(floor, t, on_arrive)` (단일 leg). gate 노드는 3DS port 로도
등록 (셔틀이 gate 에 deliver/pickup).

기본 sim 은 **15초마다 랜덤 데모 요청** 생성 (`_step_elevators` 의
`_lift_request_interval`). recipe 모드면 비활성 → 실제 LIFT system MCS dispatch.

---

## 6. OHT 통합 브릿지 — `MCSOHTBridge` (vis_mcs_unified.py)

graph_des_v6 의 `job_mgr` 인터페이스 ↔ MCSEngine. OHT 만 적용.
test_plan_micro 의 `dispatch.JobManager` 와 동일 flow:
- `on_arrive(t,v)`: MCS phase 따라 LOAD/UNLOAD dwell 시작 — `v.state=LOADING` +
  `EV_LOAD_DONE/EV_UNLOAD_DONE` post. **모든 return path 에서 `_notify_followers`
  호출 필수** (v6 의 `_on_arrive` 가 job_mgr 호출 후 early-return 하므로).
- `on_load_done`: phase=TO_DELIVERY + dst dispatch.
- `on_unload_done`: phase=IDLE + KPI + `_do_assign`.

OHT path swap (`oht_env.reassign`) 은 v6 의 `_assign_destination` 직접 호출 →
commit prefix 보존 + stale lock release + passed_zcu cleanup 자동 (=
dispatch.py `_reroute` 와 정합).

---

## 7. 파일 카탈로그

### 7.1 위치 규칙

| 위치 | 파일 |
|--|--|
| **parent `KAIST/` (canonical 엔진)** | graph_des_v5.py, graph_des_v6.py, vehicle_state.py, velocity_profile.py, dispatch.py, test_plan_micro.py, test_graph_v6.py |
| **testbed `TeamKoreaPhysicalAI_Testbed/`** | mcs_unified.py, vis_mcs_unified.py, env_oht_v6_adapter.py, env_tapg.py, env_3ds.py, elevator.py, pkl_loader.py, pkl_prioritized_planner.py, coarse_planner.py, segment_lock.py, gen_songdo_pkl.py, Maps/, *.pkl |

> ⚠️ testbed 의 `graph_des_v5.py` / `graph_des_v6.py` 는 **1KB re-export shim**.
> importlib 로 parent 의 canonical 을 로드해 re-export. 엔진 수정은 **parent**
> 파일을 고칠 것. (testbed 가 parent 를 sys.path 에 추가)

### 7.2 역할별

**Core DES 엔진**
- `graph_des_v6.py` (parent) — OHT 연속물리 엔진 (§2). `GraphDESv6` + `Vehicle`.
- `graph_des_v5.py` (parent) — JSON map 층. `GraphMap(json, area_filter)`,
  `MapNode/MapSegment/ZCUZone`, `_interp_path`, `random_safe_path`.
- `vehicle_state.py` (parent) — `state_at(v,t)` 스냅샷.
- `velocity_profile.py` (parent) — `compute_velocity_profile` 사다리꼴 타이밍.
- `env_tapg.py` (testbed) — AGV/3DS 위치적분+TAPG 엔진 (§3).

**Planner**
- `pkl_loader.py` — `PklMapGraph(pkl)`. node=centroid, edge max_speed=dist/cost,
  `_LoadState` (L,node dwell), `build_load_states`.
- `pkl_prioritized_planner.py` — `PklPrioritizedPlanner` + SIPP. `[(state_id,t)]` 출력.
- `coarse_planner.py` — `CoarsePlanner`. shortest path + segment token (unidirectional).
- `segment_lock.py` — `SegmentLockManager` (1-AGV-per-corridor + deadlock cycle 감지).

**MCS / dispatch**
- `mcs_unified.py` — 통합 단일-heap MCS (§4).
- `dispatch.py` (parent) — **OHT-only** `JobManager` (test 용. job_state `TO_DROP` 사용 — 구버전).

**Environment 어댑터**
- `env_oht_v6_adapter.py` — `GraphDESv6` 를 legacy OHT 인터페이스로 wrap.
  `OHTMap/OHTAgent/OHTEnvironmentDES`. state 매핑 IDLE/MOVING/FOLLOWING/BLOCKED/DONE.
- `env_3ds.py` — `FloorGraph/build_floor_graph` (통합 sim 은 이것만 import).
- `elevator.py` — 수직 DES (§5).

**시각화 / harness**
- `vis_mcs_unified.py` — 통합 pygame 시뮬레이터 (= 시스템 entry point).
- `test_graph_v6.py` (parent) — OHT-only pygame test/vis. `dispatch.JobManager` 부착.
- `test_plan_micro.py` (parent) — OHT micro 시나리오 단위테스트 (synthetic map).

**Asset 생성**
- `gen_songdo_pkl.py` (testbed) — **가독성 좋은 canonical 생성기**. S/M/R state +
  Shapely footprint + STRtree 충돌 프로파일(`affect_state`) → pkl.
- `archive/gen_amr_pkl.py`, `archive/gen_3ds_pkl.py` — KaistTB AMR_A / 3DS_F* 용.

### 7.3 데이터 파이프라인

```
Maps/KaistTB.map_latest.json  (xmsmap-v4: nodes, segments[speed], ports, vehicleModels, lifts, areas)
   │
   ├─ OHT: GraphMap(area_filter='OHT_A')  ──직접 로드, pkl 불필요──▶ graph_des_v6
   │
   └─ AGV/3DS: gen_*.py
         ├ S,node,heading / M,from,to (cost=dist/speed) / R,node,d1,d2 state 생성
         ├ Shapely footprint (Stop=차량 사각, Move=swept, Rotate=회전 union)
         ├ STRtree 로 footprint 교차 → 각 state 의 affect_state (= TAPG precedence)
         └ save_pkl: {Stop_state, Move_state, Rotate_state, regions, collision_profile, od_pairs}
              │
              ▼
         PklMapGraph(pkl)  ──▶ PklPrioritizedPlanner (SIPP) / CoarsePlanner ──▶ env_tapg
```

pkl 내용 (PklMapGraph 소비): node 위치 = stop_region centroid, edge =
`M,*` 키 (max_speed = dist/m_cost), affect_state = 충돌 state 목록 (TAPG 가
precedence 강제), ports = od_pairs 목적지. Rotate cost 는 최단회전/90 으로 정규화.

---

## 8. 모션 / 충돌회피 모델 비교 (Automod vs OHT vs AGV)

세 모델 모두 **"안전하게 정지 가능한 거리까지만 주행"** 이라는 *동일 원칙*
(= Automod 의 `decelerate_ok`). 차이는 *어떻게 그 보장을 구현* 하는가에 있다.

### 8.1 축별 비교표

| 축 | Automod `decelerate_ok` | **OHT** (graph_des_v6) | **AGV coarse** (env_tapg) | **AGV SIPP** (env_tapg) |
|--|--|--|--|--|
| 충돌회피 단위 | per-node 예약 | ZCU zone lock + leader gap | segment(corridor) atomic claim | state(node) precedence |
| look-ahead 범위 | 다음 node (+감속거리) | `plan_boundary` = min(leader_gap−h_min, 다음ZCU, dest, brake reach) | 다음 rest place(checkpoint) | 전체 path (시간창) |
| 체크 트리거 | **매 step polling** | **event-driven** (committed_traj + EV_REPLAN) | event-driven (TRY_ADVANCE) + node 마다 leader 재평가 | DAG cross-edge (precompute) |
| 정지 보장 | 매 step 재계산 | velocity profile 에 baked-in | claim 성공 구간만 진행 | safe-interval |
| leader 추적 | 매 step 거리 확인 | leader 이동 → `_notify_followers` → replan | `_on_seg_end` 매 crossing 재평가 | cross-edge 로 대체 |
| 시간 개념 | 실시간 polling | 연속물리 (committed_traj 시각) | 위치적분, 시간 무관 | SIPP 계획시각 + TAPG |
| 재계획 시점 | 매 step | 조건변화 이벤트 | claim 실패/wake | dispatch 시 일괄 |

### 8.2 "멈출 수 있는 데까지만" 구현 차이

```
Automod:    [매 step] "다음 칸 비었나? 아니면 지금 감속하면 멈출 수 있나?"
              → 반응형 polling. 단순하나 매 tick 비용.

OHT:        [replan 1회] velocity profile 이 plan_boundary 에서 정지하도록 생성 (committed).
              leader 가 이동하면 EV_REPLAN 으로 plan_boundary 연장 → 실제 brake 전에 재계획.
              → polling 없음. profile 자체가 "정지 가능" 보장. leader-following = 부드러운 줄서기.

AGV coarse: [claim] 현재 → 다음 rest place 를 atomic 점유 가능할 때만 진행.
              + _on_seg_end 가 매 node 마다 leader 재평가 (= Automod 정신 직역).
              → corridor 통째 예약 단위.

AGV SIPP:   [계획시] 다른 AGV 의 시간창(c_table) 피해 충돌없는 path 산출
              + TAPG cross-edge 가 실행시 precedence 강제 (drift 나도 순서 보장 — 이론상).
```

### 8.3 lock/commit 단위의 근본 차이 (★ 가장 중요)

| 시스템 | 일반 corridor | 교차점(merge/diverge) |
|--|--|--|
| **OHT** | **lock 없음** — leader-following gap 만 | **ZCU zone lock** |
| **AGV coarse** | segment lock (corridor 통째) | cut node admission rule |
| **AGV SIPP** | state affect_state precedence | 동일 (state 단위) |

→ **OHT 만 "교차점만 lock, 직선은 줄서기(gap)"**. AGV 는 모든 구간 lock.
- OHT 는 레일 위(추월 불가, 단방향 루프) → 충돌점이 merge/diverge 교차점뿐.
- AGV 는 2D 자유주행 → 모든 cell 충돌 가능 → per-node/segment lock 필요.

`_find_first_boundary` 가 찾는 commit checkpoint = `_boundary_nodes` (= **ZCU 만**).
일반 node 는 commit/lock 대상이 아니며 그냥 통과한다 (충돌회피는 leader gap 이 담당).

### 8.4 디버깅 함의 (충돌 원인 위치)

| 시스템 | 충돌 시 의심 지점 |
|--|--|
| **OHT** | 직선 충돌 → leader-following (gap/notify) 버그. 교차 충돌 → ZCU lock 버그. (두 메커니즘 분리) |
| **AGV SIPP** | cross-edge 누락 또는 c_table 시간 drift. **live-occupancy 백스톱 없음** → edge 하나 빠지면 충돌 |
| **AGV coarse** | `_is_claimable_coarse` 검사 누락. 단 live-occupancy 검사라 **백스톱 있음** (SIPP 보다 견고) |

---

## 9. C 재구현 체크리스트

1. **단일 min-heap** `(t, seq)` 키. 시스템별 분리 안 해도 됨 (MCS 가 통합).
2. **이벤트 취소 금지** — 핸들러별 stale 검사 (token / exit-node / state token).
3. **token** 모든 binding 에 정수. phase 전이마다 ++.
4. **부동소수 epsilon verbatim** + `round(t,6)` 키 양자화.
5. OHT: 도달시각 **예약** (committed_traj → SEG_END/PHASE_DONE/STOPPED/BOUNDARY).
6. AGV/3DS: 도달 **위치적분 감지** (매 tick), 이벤트는 TRY_ADVANCE/LOAD_DONE 만.
7. ZCU lock: merge/diverge 별 boundary/exit, FIFO waiter + path-aware grant skip.
8. leader notify: mutual-link skip + per-t dedup. polling 없음.
9. commit horizon = `min(eff_brake_dist, bnd_dist)`, physical_reach 는 v_max 기준.
10. coarse claim: atomic [현재 → 다음 rest place], cut node 통과 강제, cycle 거부.
11. WAITING wake: `(_priority, _wait_start_t)` 정렬 (cycle victim 우선 + FIFO).
12. pkl: offline 충돌기하 (affect_state) → SIPP/TAPG precedence.
