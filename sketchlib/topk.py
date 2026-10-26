"""从草图里取 top-k：候选表 + 懒更新最小堆，以及精确基线。"""

import heapq
import itertools
import sys

from .cms import CountMinSketch, dict_bytes


class CmsTopK:
    """Count-Min Sketch 加一个大小为 capacity 的候选堆。

    草图本身不记得见过哪些键，所以必须另存候选表；容量决定了这项
    "隐形"内存，它和 width*depth 一样进 bytes_used 的账。
    """

    def __init__(self, width=1024, depth=5, capacity=256, seed=0, mode="min"):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.sketch = CountMinSketch(width, depth, seed)
        self.capacity = capacity
        self.mode = mode
        self.candidates = {}
        self._heap = []
        self._tie = itertools.count()

    def add(self, key, count=1):
        """更新草图后维护候选表：已入选则刷新，未满则进表，否则换掉堆顶。"""
        if count == 0:
            return
        self.sketch.add(key, count)
        est = self.sketch.estimate_all(key)[self.mode]
        cand = self.candidates
        if key in cand:
            cand[key] = est
            h = self._heap
            h.append((est, next(self._tie), key))
            if len(h) > 4 * self.capacity + 16:
                self._repair()
            return
        if len(cand) < self.capacity:
            cand[key] = est
            heapq.heappush(self._heap, (est, next(self._tie), key))
            return
        top = self._peek_valid()
        if top is not None and est > top[0]:
            heapq.heappop(self._heap)
            del cand[top[2]]
            cand[key] = est
            heapq.heappush(self._heap, (est, next(self._tie), key))

    def _peek_valid(self):
        """丢掉堆里过期的 (估计, 键)，只留与候选表一致的堆顶。"""
        h = self._heap
        cand = self.candidates
        while h:
            e, _, k = h[0]
            if cand.get(k) == e:
                return h[0]
            heapq.heappop(h)
        return None

    def _repair(self):
        self._heap = [(e, next(self._tie), k) for k, e in self.candidates.items()]
        heapq.heapify(self._heap)

    def top_k(self, k):
        """按当前草图估计（而非入表时的旧估计）降序返回前 k 项。"""
        fresh = [(self.sketch.estimate_all(key)[self.mode], key)
                 for key in self.candidates]
        fresh.sort(key=lambda kv: (-kv[0], kv[1]))
        return [(key, est) for est, key in fresh[:k]]

    def estimate(self, key):
        return self.sketch.estimate_all(key)[self.mode]

    def bytes_used(self):
        return (self.sketch.bytes_used() + dict_bytes(self.candidates)
                + sys.getsizeof(self._heap))

    def __repr__(self):
        return "CmsTopK(width=%d, depth=%d, capacity=%d, candidates=%d)" % (
            self.sketch.width, self.sketch.depth, self.capacity,
            len(self.candidates))


def top_k_from_counts(counts, k):
    """精确基线：普通 dict 排序，平局按键名保证可复现。"""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:k]
