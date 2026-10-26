"""Count-Min Sketch：d 行 w 列的计数草图，以及它的几种查询修正。"""

import hashlib
import random
import sys
from array import array

_MASK64 = (1 << 64) - 1

# 估计模式：min 是教科书定义，mean/median/count_mean 是降低高估的修正
MODES = ("min", "mean", "median", "count_mean")


def dict_bytes(mapping):
    """记账口径：dict 表体（含槽位数组）加上每个键、每个值对象本身。"""
    return sys.getsizeof(mapping) + sum(
        sys.getsizeof(k) + sys.getsizeof(v) for k, v in mapping.items())


class CountMinSketch:
    """只靠"多次插入"成立的频率草图。

    保证（全部更新非负时）：`query(..., "min") >= 真实计数`，
    且超出量的期望不超过 `e * n / w`，其中 `e` 是权重和、`n` 是插入次数、
    `w` 是宽度；失败概率受 `2 ** -d` 约束（与宽度无关）。
    允许写入负数，但那时上面的下界立刻失效 —— 见 delete_* 方法与基准第 6 节。
    """

    def __init__(self, width=1024, depth=5, seed=0):
        if width < 2:
            raise ValueError("width must be >= 2 (count-mean divides by w-1)")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.width = width
        self.depth = depth
        self.seed = seed
        # 有符号 64 位数组：允许负数，所以删除操作不会绕回大正数
        self.rows = [array("q", [0]) * width for _ in range(depth)]
        # 每行所有单元格之和，供 count-mean 估计"别的键"造成的噪声
        self.row_totals = array("q", [0]) * depth
        self.updates = 0
        self.total_weight = 0
        rnd = random.Random(seed)
        # 每行一个奇数乘数：同一摘要映射到不同列，避免行与行同步碰撞
        self._mult = [rnd.getrandbits(64) | 1 for _ in range(depth)]
        self._salt = rnd.getrandbits(64)

    def _cols(self, key):
        """键 -> 每行的列号；纯函数，所以删除能沿原路返回。"""
        raw = key if isinstance(key, bytes) else str(key).encode("utf-8")
        digest = hashlib.blake2b(raw, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little")
        width = self.width
        cols = []
        for i in range(self.depth):
            x = (h1 + i * h2 + self._salt) & _MASK64
            # 乘积的高 32 位混合充分；低 32 位仍由 x 的低位主导，不能用
            y = ((x * self._mult[i]) & _MASK64) >> 32
            cols.append(y % width)
        return cols

    def add(self, key, count=1):
        """一次更新写 d 个单元格；count 为负即"删除事件"。"""
        if count == 0:
            return
        rows = self.rows
        row_totals = self.row_totals
        for i, col in enumerate(self._cols(key)):
            rows[i][col] += count
            row_totals[i] += count
        self.updates += 1
        self.total_weight += count

    def values(self, key):
        """该键在每一行的原始单元格值，供测试和多种估计复用。"""
        return [self.rows[i][c] for i, c in enumerate(self._cols(key))]

    def estimate_all(self, key):
        """一次哈希、四种估计，避免基准里重复算摘要。"""
        vals = self.values(key)
        d = self.depth
        w = self.width
        vmin = min(vals)
        vmean = sum(vals) / d
        svals = sorted(vals)
        vmedian = svals[d // 2] if d % 2 else (svals[d // 2 - 1] + svals[d // 2]) / 2.0
        noise = 0.0
        for i, v in enumerate(vals):
            # 该行除本列外的平均负载，就是这一列里"别人"带来的期望噪声
            noise += (self.row_totals[i] - v) / (w - 1)
        return {
            "min": vmin,
            "mean": vmean,
            "median": vmedian,
            "count_mean": vmin - noise / d,
        }

    def query(self, key, mode="min"):
        if mode not in MODES:
            raise ValueError("unknown mode %r" % (mode,))
        if mode == "min":
            return min(self.values(key))
        return self.estimate_all(key)[mode]

    def delete_event(self, key, count=1, clamp=False):
        """"删除一个事件"：沿原路把单元格减回去，别的键不受影响。"""
        if not clamp:
            self.add(key, -count)
            return True
        # clamp 只夹到 0：删除于是不再是插入的逆运算，之后重插会被多计一次
        for i, col in enumerate(self._cols(key)):
            row = self.rows[i]
            after = row[col] - count
            if after < 0:
                after = 0
            self.row_totals[i] += after - row[col]
            row[col] = after
        self.updates += 1
        return True

    def delete_by_subtracting_estimate(self, key, mode="min", scale=1.0):
        """"减去报告的估计"：多扣的那部分是同格子里别人的计数。"""
        amount = int(round(scale * self.query(key, mode)))
        if amount == 0:
            return 0
        for i, col in enumerate(self._cols(key)):
            self.rows[i][col] -= amount
            self.row_totals[i] -= amount
        self.updates += 1
        self.total_weight -= amount
        return amount

    def cells(self):
        return self.width * self.depth

    def bytes_used(self):
        """Python 里的真实占用：数组对象头 + 8 字节的单元格 + 容器指针。"""
        n = self.width
        body = sum(sys.getsizeof(row) - (64 + 8 * n) for row in self.rows)
        heads = len(self.rows) * (64 + 8 * n)
        return (body + heads + sys.getsizeof(self.rows)
                + sys.getsizeof(self.row_totals) + sys.getsizeof(self._mult)
                + sys.getsizeof(self._salt))

    def compact_bytes(self):
        """理论下限：只有单元格本身需要的字节数。"""
        return 8 * self.cells()

    def __repr__(self):
        return "CountMinSketch(width=%d, depth=%d, seed=%d)" % (
            self.width, self.depth, self.seed)
