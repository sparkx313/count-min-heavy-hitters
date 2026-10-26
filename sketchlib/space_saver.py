"""Space-Saver（加权版 Misra-Gries）：固定容量的重键候选表。"""

import heapq
import sys

from .cms import dict_bytes


class SpaceSaver:
    """最多监控 capacity 个键；新键取代最小计数者并继承其计数。

    插入-only 的保证：对任意键 `query(k) >= 真实计数`，且被监控键的高估
    不超过 `total // capacity`（`total` 为已见权重和）；未被监控的键直接报 0，
    所以它对"稀有键"会低估 —— 草图的误差预算是花在重键上的。
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.estimates = {}
        self.errors = {}
        self.total = 0
        self._heap = []

    def _repair_heap(self):
        """懒堆过长时按当前估计重建，摊还 O(log capacity) 的替换成本。"""
        self._heap = [(e, k) for k, e in self.estimates.items()]
        heapq.heapify(self._heap)

    def _push(self, key):
        h = self._heap
        # 必须 heappush：list.append 不维护堆序，替换最小键就会挑错受害者
        heapq.heappush(h, (self.estimates[key], key))
        if len(h) > 4 * self.capacity + 16:
            self._repair_heap()

    def _pop_valid_min(self):
        est = self.estimates
        h = self._heap
        while h:
            e, k = heapq.heappop(h)
            if est.get(k) == e:
                return e, k
        return None

    def add(self, key, count=1):
        """count 为负请改用 remove_event；这里只做正向记账。"""
        if count == 0:
            return
        if count < 0:
            self.remove_event(key, -count)
            return
        est = self.estimates
        hit = est.get(key)
        if hit is not None:
            est[key] = hit + count
            self.total += count
            self._push(key)
            return
        if len(est) < self.capacity:
            est[key] = count
            self.errors[key] = 0
            self.total += count
            self._push(key)
            return
        # 替换：继承被淘汰者的计数，把它当作自己的误差预算
        top = self._pop_valid_min()
        if top is None:
            est[key] = count
            self.errors[key] = 0
            self.total += count
            self._push(key)
            return
        min_est, victim = top
        del est[victim]
        del self.errors[victim]
        est[key] = min_est + count
        self.errors[key] = min_est
        self.total += count
        self._push(key)

    def query(self, key):
        """未监控的键返回 0，而不是返回一个下界。"""
        return self.estimates.get(key, 0)

    def remove_event(self, key, count=1):
        """"删除一个事件"：只对表里的键有效，未监控键无从减起。"""
        if count == 0:
            return False
        est = self.estimates
        hit = est.get(key)
        if hit is None:
            return False
        new = hit - count
        if new <= 0:
            del est[key]
            del self.errors[key]
        else:
            est[key] = new
            self._push(key)
        self.total -= count
        return True

    def subtract_estimate(self, key, scale=1.0):
        """"减去报告值"：扣掉的是含继承误差的估计，通常比真实计数多。"""
        reported = self.query(key)
        if reported == 0:
            return 0
        return self.remove_event(key, scale * reported) and scale * reported

    def drop(self, key):
        """整键出表：槽位回收，但它继承来的误差留在别人身上。"""
        if key not in self.estimates:
            return False
        del self.estimates[key]
        del self.errors[key]
        return True

    def top_k(self, k):
        """按估计降序返回 (key, estimate) 前 k 项。"""
        items = sorted(self.estimates.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:k]

    def keys(self):
        return set(self.estimates)

    def over_estimate_bound(self):
        """插入-only 时任意键的高估上界。"""
        return self.total // self.capacity

    def error_of(self, key):
        return self.errors.get(key, 0)

    def __len__(self):
        return len(self.estimates)

    def bytes_used(self):
        """表体 + 键值对象，与精确基线同一口径。"""
        # errors 的键与 estimates 的键是同一批对象，只计一次
        used = dict_bytes(self.estimates)
        used += sys.getsizeof(self.errors)
        used += sum(sys.getsizeof(v) for v in self.errors.values())
        return used + sys.getsizeof(self._heap)

    def __repr__(self):
        return "SpaceSaver(capacity=%d, monitored=%d, total=%d)" % (
            self.capacity, len(self.estimates), self.total)
