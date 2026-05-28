"""Segment-level lock manager for coarse-grained execution.

Each segment (corridor between two checkpoints) can be held by at most one
AGV at a time. AGV at checkpoint requesting next segment:
- If free → claim, advance into segment.
- If held → wait at checkpoint.

Deadlock detection: cycle in wait graph → backoff one AGV (push to siding).
"""
from __future__ import annotations
from typing import Dict, Tuple, Set, Optional
from collections import defaultdict


class SegmentLockManager:
    def __init__(self, segments: Dict[Tuple[str, str], list],
                 node_to_segment: Dict[str, Tuple[str, str]]):
        self.segments = segments        # seg_id → [intermediate nodes]
        self.node_to_segment = node_to_segment   # any corridor node → seg_id

        # Lock state
        self.held: Dict[Tuple[str, str], int] = {}        # seg_id → vid
        self.waiting: Dict[int, Tuple[str, str]] = {}      # vid → seg_id waiting on

    def try_claim(self, vid: int, seg_id: Tuple[str, str]) -> bool:
        """vid 가 seg_id segment 진입 시도. True if claimed, False if blocked."""
        if seg_id in self.held:
            if self.held[seg_id] == vid:
                return True   # 이미 자기가 holding
            # held by other AGV → wait
            self.waiting[vid] = seg_id
            return False
        self.held[seg_id] = vid
        if vid in self.waiting:
            del self.waiting[vid]
        return True

    def release(self, vid: int, seg_id: Tuple[str, str]):
        """vid 가 seg_id 떠남."""
        if self.held.get(seg_id) == vid:
            del self.held[seg_id]
        if vid in self.waiting and self.waiting[vid] == seg_id:
            del self.waiting[vid]

    def release_all(self, vid: int):
        """vid 가 holding 한 모든 segment release (e.g., re-plan 시)."""
        to_release = [s for s, holder in self.held.items() if holder == vid]
        for s in to_release:
            del self.held[s]
        if vid in self.waiting:
            del self.waiting[vid]

    def detect_deadlock_cycle(self) -> Optional[list]:
        """Wait graph 의 cycle 검출. cycle 있으면 vid list 반환."""
        # vid → vid (waiting on holder)
        wait_edge: Dict[int, int] = {}
        for vid, seg in self.waiting.items():
            holder = self.held.get(seg)
            if holder is not None and holder != vid:
                wait_edge[vid] = holder
        # DFS cycle detection
        for start in wait_edge:
            visited = []
            cur = start
            while cur in wait_edge and cur not in visited:
                visited.append(cur)
                cur = wait_edge[cur]
            if cur in visited:
                # Found cycle starting from cur
                idx = visited.index(cur)
                return visited[idx:]
        return None

    def status(self) -> str:
        held_str = ', '.join(f'{a}-{b}:V{v}' for (a, b), v in self.held.items())
        wait_str = ', '.join(f'V{v}→{a}-{b}' for v, (a, b) in self.waiting.items())
        return f'held={{{held_str}}} waiting={{{wait_str}}}'
