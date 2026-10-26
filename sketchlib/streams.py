"""确定性数据源：同一 (seed, skew, universe) 永远得到同一条流。"""

import random


def ip_key(rank):
    """把 Zipf 秩映射成看起来像源 IP 的键，避免测试依赖 Python 哈希顺序。"""
    if not 1 <= rank <= 0xFFFFFF:
        raise ValueError("rank must be in 1..16777215 to stay inside one /8")
    return "10.%d.%d.%d" % (rank // 65536, (rank // 256) % 256, rank % 256)


def zipf_weights(n, skew):
    """秩 1..n 的未归一化幂律权重。"""
    return [float(rank) ** (-skew) for rank in range(1, n + 1)]


def zipf_stream(count, skew=1.1, universe=1000, seed=0):
    """返回长度为 count 的键列表；两次同参数调用逐元素相同。"""
    if count < 0:
        raise ValueError("count must be >= 0")
    if universe < 1:
        raise ValueError("universe must be >= 1")
    rnd = random.Random(seed)
    ranks = rnd.choices(range(1, universe + 1),
                        weights=zipf_weights(universe, skew), k=count)
    return [ip_key(r) for r in ranks]


def uniform_stream(count, universe=1000, seed=0):
    """无偏流：所有键等概率，用来检验"没有重键时会发生什么"。"""
    rnd = random.Random(seed)
    return [ip_key(rnd.randrange(1, universe + 1)) for _ in range(count)]


def exact_counts(keys):
    """精确基线：普通 dict 计数，作为所有误差的参照。"""
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts
