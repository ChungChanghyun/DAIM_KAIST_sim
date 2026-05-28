# KAIST 통합 물류 시뮬레이터

OHT + AGV + 3DS(층별 셔틀) + Elevator 를 단일 MCS 로 통합한 이산사건
시뮬레이터. KaistTB factory layout.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 기본 (AGV 7대)
python vis_mcs_unified.py --agv 7

# OHT + AGV + 3DS 통합
python vis_mcs_unified.py --agv 7 --oht 6 --s3d 2

# 헤드리스 (장기 평가, GUI off)
python vis_mcs_unified.py --agv 7 --oht 6 --s3d 2 --headless --max-time 43200
```

## 주요 CLI 옵션

| 옵션 | 설명 |
|--|--|
| `--agv N` | AGV 수 (default 7) |
| `--oht N` | OHT 수 |
| `--s3d N` | 층별 3DS 셔틀 수 (3층 → 총 3N) |
| `--seed N` | 랜덤 시드 |
| `--headless` | render skip + fixed-step (장기 sim 가속) |
| `--max-time SEC` | sim 자동 종료 시간 |
| `--lenient` | SIPP fail 시 plan 만 취소 (halt 안 함) |
| `--planner` | `sipp` (default, KAIST) / `coarse` |

## GUI 키

| 키 | 기능 |
|--|--|
| Space | pause/resume |
| F | zoom-to-fit |
| I | node id 표시 |
| L / T / C | OHT leader chain / dest marker / commit horizon |
| D | AGV dump |
| 마우스 휠 (사이드바) | 정보 스크롤 |

## 구조

- DES 이벤트 아키텍처 + C 재구현 명세 → `DES_ARCHITECTURE.md`
- 4개 엔진(OHT/AGV/3DS/Elevator)이 단일 MCS 경계에서 통합.

## 파일

| 그룹 | 파일 |
|--|--|
| Entry | vis_mcs_unified.py |
| DES 엔진 | graph_des_v6.py (OHT), env_tapg.py (AGV/3DS), graph_des_v5.py (map) |
| 보조 | vehicle_state.py, velocity_profile.py |
| Planner | pkl_loader.py, pkl_prioritized_planner.py, coarse_planner.py, segment_lock.py |
| MCS | mcs_unified.py |
| 어댑터 | env_oht_v6_adapter.py, env_3ds.py, elevator.py |
| Solver | solvers/ (pkl 언피클링용) |
| Data | Maps/KaistTB.map_latest.json, KaistTB_*.pkl |
