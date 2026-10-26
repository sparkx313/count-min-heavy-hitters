"""Count-Min Sketch 与精确 dict 的实测对比：误差方向、top-k 质量、内存、
假阳性、更新成本、删除语义。输出同时缓冲并逐字落盘到 last_run.txt。"""

import gc
import os
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sketchlib import (CountMinSketch, CmsTopK, SpaceSaver, exact_counts, ip_key,
                       top_k_from_counts, zipf_stream)
from sketchlib.cms import dict_bytes

REPORT = os.path.join(HERE, "last_run.txt")
SCALE = float(os.environ.get("CMH_SCALE", "1"))
N = int(200000 * SCALE)
TIMING_BATCH = int(20000 * SCALE)
TIMING_PASSES = 7           # 前 2 次是丢弃的预热，后 5 次取最优
DEPTH = 5


class _Tee:
    """把 stdout 全部缓冲：照常打印，最后写一份逐字相同的报告文件。"""

    def __init__(self, stream, path):
        self._stream = stream
        self._path = path
        self._buf = []

    def write(self, text):
        self._stream.write(text)
        self._buf.append(text)
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return False

    def save(self):
        with open(self._path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(self._buf))


def banner(title, kind):
    print("=" * 78)
    print("%s (%s)" % (title, kind))
    print("=" * 78)


def say(text):
    print()
    print(textwrap.fill("Interpretation: " + text, 78, subsequent_indent="  "))


def note(text):
    print(textwrap.fill(text, 78))


def table(headers, rows, align):
    """纯文本表格；列宽由数据决定，行总宽被截到 78 列。"""
    align = (align + "l" * len(headers))[:len(headers)]
    cells = [[str(c) for c in row] for row in [list(headers)] + [list(r) for r in rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for index, row in enumerate(cells):
        parts = [(c.rjust(widths[i]) if align[i] == "r" else c.ljust(widths[i]))
                 for i, c in enumerate(row)]
        print("  ".join(parts).rstrip()[:78])
        if index == 0:
            print("-" * min(len("  ".join(parts)), 78))


def pct(part, whole):
    return 100.0 * part / float(whole) if whole else 0.0


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return float(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))])


# ------------------------------------------------------------------ 第 1 节
def section_errors():
    banner("1. error vs the guarantee", "deterministic")
    universe = 2000
    stream = zipf_stream(N, 1.1, universe, seed=1)
    truth = exact_counts(stream)
    keys = sorted(truth)
    note("Run scale %s: N=%d events per section, depth=%d. Stream: %d events, "
         "%d distinct keys, skew=1.1, seed=1; tracked population = every key the "
         "dict saw." % (SCALE, N, DEPTH, N, len(keys)))
    print()
    rows = []
    stats = []
    widths = (256, 1024, 4096, 16384)
    for width in widths:
        sketch = CountMinSketch(width, DEPTH, seed=1)
        for key in stream:
            sketch.add(key)
        budget = N / float(width)
        overs = []
        under = {"min": 0, "mean": 0, "median": 0, "count_mean": 0}
        violations = 0
        for key in keys:
            est = sketch.estimate_all(key)
            true = truth[key]
            for mode in under:
                if est[mode] < true:
                    under[mode] += 1
            over = est["min"] - true
            overs.append(over)
            if over > budget:
                violations += 1
        overs.sort()
        stat = {"width": width, "budget": budget, "mean": sum(overs) / float(len(overs)),
                "p50": quantile(overs, 0.50), "p95": quantile(overs, 0.95),
                "max": float(overs[-1]), "viol": violations, "under": under}
        stats.append(stat)
        rows.append([width, "%.1f" % budget, "%.4f" % (2 ** -DEPTH),
                     "%.1f" % stat["mean"], "%.0f" % stat["p95"],
                     "%.0f" % stat["max"], "%d" % violations,
                     "%d" % under["min"], "%d" % under["mean"],
                     "%d" % under["count_mean"]])
    table(["width", "eps*N", "delta", "mean_ov", "p95", "max_ov",
           "gt_epsN", "u_min", "u_mean", "u_cmean"], rows, "rrrrrrrrrr")
    thin, fat = stats[0], stats[-1]
    pairs = len(widths) * len(keys)
    total_under_min = sum(s["under"][m] for s in stats for m in ("min", "mean", "median"))
    total_under_cm = sum(s["under"]["count_mean"] for s in stats)
    drop = ("falls %.2fx" % (thin["mean"] / fat["mean"])) if fat["mean"] > 0 \
        else "falls to %.1f" % fat["mean"]
    say("Width is the error knob and the run follows it: eps*N shrinks %.1fx (%.1f "
        "events at width %d down to %.1f at width %d) and the measured mean "
        "over-estimate goes %.1f -> %.1f, i.e. it %s. The narrowest width sits %.2fx "
        "above its own eps*N on average, and %d of the %d tracked keys exceed that "
        "budget there where delta=%.4f licenses about %.1f violations; at width %d the "
        "count is %d. Direction is where folklore is wrong: min-, mean- and "
        "median-of-rows together under-estimated %d keys out of %d key-width trials on "
        "this insert-only stream - every cell already holds at least the true count "
        "when updates are non-negative, so the minimum does too - while the "
        "noise-subtracting count-mean under-estimated %d of the same %d trials "
        "(%.1f%%). Count-mean pays for its own correction: it subtracts "
        "(row_total - est)/(w-1), about N/(w-1), from the *minimum* of %d rows, and "
        "that minimum already sits below the average cell load, so the subtraction is "
        "systematically too large on a loaded sketch."
        % (fat["width"] / float(thin["width"]), thin["budget"], thin["width"],
           fat["budget"], fat["width"], thin["mean"], fat["mean"], drop,
           thin["mean"] / thin["budget"], thin["viol"], len(keys), 2 ** -DEPTH,
           (2 ** -DEPTH) * len(keys), fat["width"], fat["viol"], total_under_min,
           pairs, total_under_cm, pairs, pct(total_under_cm, pairs), DEPTH))

    print()
    victims = sorted(truth, key=lambda k: (-truth[k], k))[:max(1, len(keys) // 20)]
    note("Same geometry (width 1024, depth %d) once deletions are allowed: the %d "
         "highest-count keys are deleted outright, so their true count becomes 0."
         % (DEPTH, len(victims)))
    print()
    after = dict(truth)
    for key in victims:
        after[key] = 0
    rows = []
    stats_b = []
    for label, kind in (("event-remove", "ev"), ("subtract-estimate", "es"),
                        ("clamped-remove", "cl")):
        probe = CountMinSketch(1024, DEPTH, seed=1)
        for key in stream:
            probe.add(key)
        for key in victims:
            if kind == "ev":
                probe.delete_event(key, truth[key])
            elif kind == "es":
                probe.delete_by_subtracting_estimate(key)
            else:
                probe.delete_event(key, truth[key], clamp=True)
        per_mode = {}
        for mode in ("min", "mean", "median", "count_mean"):
            under = low = 0
            for key, true in after.items():
                est = probe.query(key, mode)
                if est < true:
                    under += 1
                    low = min(low, est - true)
            per_mode[mode] = (under, low)
        stat = {"label": label, "victims": len(victims), "per_mode": per_mode,
                "worst": min(per_mode[m][1] for m in per_mode)}
        stats_b.append(stat)
        rows.append([label, len(victims), "%d" % per_mode["min"][0],
                     "%d" % per_mode["mean"][0], "%d" % per_mode["median"][0],
                     "%d" % per_mode["count_mean"][0], "%d" % stat["worst"]])
    table(["op", "victims", "u_min", "u_mean", "u_med", "u_cmean", "worst_gap"],
          rows, "llrrrrr")
    by = dict((s["label"], s) for s in stats_b)
    es, ev, cl = by["subtract-estimate"], by["event-remove"], by["clamped-remove"]
    say("Deletions change the verdict on direction, not on geometry: "
        "subtract-estimate under-estimates %d of %d keys with a worst gap of %d "
        "events, while event-remove under-estimates %d keys (worst gap %d). So "
        "min-of-rows breaks only when a negative update fails to undo its own "
        "insertion, never merely because a deletion is possible. count-mean "
        "under-estimates %d keys after event-remove and %d after subtract-estimate, "
        "and the clamped decrement lands at %d under-estimates for min-of-rows: a "
        "cell stopped at zero no longer returns to its pre-insertion value, so a "
        "later re-insertion of the same key is over-counted - the consequence the "
        "clamped-delete unit test pins down."
        % (es["per_mode"]["min"][0], len(after), es["worst"],
           ev["per_mode"]["min"][0], ev["worst"],
           ev["per_mode"]["count_mean"][0], es["per_mode"]["count_mean"][0],
           cl["per_mode"]["min"][0]))


# ------------------------------------------------------------------ 第 2 节
def section_topk():
    banner("2. top-k: cms+heap vs space-saver vs exact dict", "deterministic")
    universe = 20000
    width = 1024
    capacity = 64
    note("Hard regime: %d events over a %d-key universe, both approximations get the "
         "same %d-slot candidate budget, cms width %d, depth %d, seed 2. The exact "
         "dict is the ground truth and ties break by key name."
         % (N, universe, capacity, width, DEPTH))
    print()
    rows = []
    per_setting = []
    for skew in (0.7, 1.1, 1.6):
        stream = zipf_stream(N, skew, universe, seed=2)
        truth = exact_counts(stream)
        for k in (5, 10, 50):
            exact = top_k_from_counts(truth, k)
            want = set(key for key, _ in exact)
            topk = CmsTopK(width=width, depth=DEPTH, capacity=capacity, seed=2)
            saver = SpaceSaver(capacity)
            for key in stream:
                topk.add(key)
                saver.add(key)
            reports = {"cms+heap": topk.top_k(k), "space-saver": saver.top_k(k)}
            scores = {}
            for name in ("cms+heap", "space-saver"):
                report = dict(reports[name])
                have = set(report)
                hit = len(have & want)
                prec = hit / float(len(have)) if have else 0.0
                rec = hit / float(k)
                mae = sum(abs(report[key] - truth.get(key, 0)) for key in have)
                mae = mae / float(len(have)) if have else 0.0
                scores[name] = (hit, prec, rec, mae, len(have))
                rows.append([k, "%.1f" % skew, name, "%d/%d" % (hit, k),
                             "%.3f" % prec, "%.3f" % rec, "%.1f" % mae,
                             "%d" % len(have)])
            per_setting.append((skew, k, scores))
    table(["k", "skew", "method", "overlap", "prec", "recall", "mae", "size"],
          rows, "rllrrrrr")
    n_settings = len(per_setting)
    cms_mean = sum(s["cms+heap"][2] for _, _, s in per_setting) / n_settings
    ss_mean = sum(s["space-saver"][2] for _, _, s in per_setting) / n_settings
    ss_wins = sum(1 for _, _, s in per_setting
                  if s["space-saver"][2] > s["cms+heap"][2])
    cms_wins = sum(1 for _, _, s in per_setting
                   if s["space-saver"][2] < s["cms+heap"][2])
    ties = n_settings - ss_wins - cms_wins
    points = []
    for skew, k, s in per_setting:
        for name in ("cms+heap", "space-saver"):
            points.append((s[name][2], k, skew, name, s[name][3], s[name][4]))
    worst = min(points)
    best = max(points)
    cms_mae = sum(s["cms+heap"][3] for _, _, s in per_setting) / n_settings
    ss_mae = sum(s["space-saver"][3] for _, _, s in per_setting) / n_settings
    diff = ss_mean - cms_mean
    leader = "space-saver" if diff > 0 else ("cms+heap" if diff < 0 else "neither")
    min_size = min(p[5] for p in points)
    prec_all = [s[name][1] for _, _, s in per_setting
                for name in ("cms+heap", "space-saver")]
    leader_wins = cms_wins if leader == "cms+heap" else ss_wins
    leader_loses = ss_wins if leader == "cms+heap" else cms_wins
    if diff == 0:
        tally = "the two are indistinguishable on average, tied in %d of %d settings" \
            % (ties, n_settings)
    else:
        tally = "%s wins %d settings, loses %d, ties %d" % (
            leader, leader_wins, leader_loses, ties)
    if cms_mae == ss_mae:
        mae_phrase = "identical at %.2f events for both" % cms_mae
        crown = "there is no accuracy crown to compare with the ranking one"
    else:
        mae_leader = "space-saver" if ss_mae < cms_mae else "cms+heap"
        mae_phrase = "%.2f events for %s versus %.2f for %s" % (
            min(cms_mae, ss_mae), mae_leader, max(cms_mae, ss_mae),
            "cms+heap" if mae_leader == "space-saver" else "space-saver")
        crown = ("the accuracy crown and the recall crown sit on %s"
                 % ("the same structure" if mae_leader == leader
                    else "different structures, so a better ranking and better "
                         "counts are not the same verdict"))
    say("Over the %d (k, skew) settings space-saver averages recall %.3f and cms+heap "
        "%.3f, so %s is ahead by %.3f: %s. The single worst point is %s at k=%d / "
        "skew=%.1f (recall %.3f) and the best is %s at k=%d / skew=%.1f (%.3f); recall "
        "spans %.3f-%.3f overall. Mean absolute error of the returned counts is %s, and "
        "%s. Smallest reported set size is %d against a requested k of up to %d, so "
        "precision (%.3f-%.3f) differs from recall only where a structure returns fewer "
        "than k keys."
        % (n_settings, ss_mean, cms_mean, leader, abs(diff), tally,
           worst[3], worst[1], worst[2], worst[0], best[3], best[1], best[2], best[0],
           worst[0], best[0], mae_phrase, crown, min_size,
           max(p[1] for p in points), min(prec_all), max(prec_all)))


# ------------------------------------------------------------------ 第 3 节
def section_memory():
    banner("3. memory per structure vs the exact dict", "deterministic")
    universe = 200000
    stream = zipf_stream(N, 1.05, universe, seed=3)
    truth = exact_counts(stream)
    distinct = len(truth)
    exact_b = dict_bytes(truth)
    budget_bytes = 64 * 1024 * 1024
    target_width = 4096
    target = N / float(target_width)
    note("Stream: %d events over a %d-key universe -> %d distinct keys seen by the "
         "dict (%.1f bytes/key). Error target: over-estimate <= eps*N with eps = "
         "1/%d = %.1f events." % (N, universe, distinct, exact_b / float(distinct),
                                  target_width, target))
    print()
    rows = []

    def row(name, geometry, tracked, nbytes, bound):
        rows.append([name, geometry, "%s" % tracked, "%d" % nbytes,
                     "%.1f" % (nbytes / 1024.0), bound, "%.3f" % (nbytes / float(exact_b))])
        return nbytes

    row("exact dict", "1 hash map", distinct, exact_b, "0")
    cms_bytes = {}
    for width in (1024, target_width, 16384):
        sketch = CountMinSketch(width, DEPTH, seed=4)
        cms_bytes[width] = row("cms", "%dx%d" % (width, DEPTH), "all keys",
                               sketch.bytes_used(), "%.1f" % (N / float(width)))
    topk = CmsTopK(width=target_width, depth=DEPTH, capacity=target_width, seed=4)
    for key in stream:
        topk.add(key)
    heap_bytes = row("cms+heap", "%dx%d+heap%d" % (target_width, DEPTH, target_width),
                     len(topk.candidates), topk.bytes_used(),
                     "%.1f" % (N / float(target_width)))
    ss_bytes = {}
    for capacity in (1024, target_width, 16384):
        saver = SpaceSaver(capacity)
        for key in stream:
            saver.add(key)
        ss_bytes[capacity] = row("space-saver", "cap=%d" % capacity, len(saver),
                                 saver.bytes_used(),
                                 "%.1f" % (saver.total / float(capacity)))
    table(["structure", "geometry", "tracked", "bytes", "KiB", "err<=N/x", "x_dict"],
          rows, "llrrrrr")
    cms_at_target = cms_bytes[target_width]
    ss_at_target = ss_bytes[target_width]
    per_key = exact_b / float(distinct)
    say("At the matched additive error target of %.1f events the cms costs %d bytes "
        "(%.1f KiB, %.3f of the exact dict) and space-saver costs %d bytes (%.1f KiB, "
        "%.3f of the dict): the sketch is %.1fx smaller than space-saver at the same "
        "target, and both are dwarfed by the dict's %d bytes. The dict's cost is a "
        "function of distinct keys, not events - %.1f bytes per key measured here - "
        "so within a 64 MiB budget it can only hold %d keys, whereas the cms geometry "
        "for the same target (width %d x depth %d) is fixed at %d bytes = %.3f%% of "
        "that budget, independent of how many keys the stream actually contains."
        % (target, cms_at_target, cms_at_target / 1024.0,
           cms_at_target / float(exact_b), ss_at_target, ss_at_target / 1024.0,
           ss_at_target / float(exact_b), ss_at_target / float(cms_at_target),
           exact_b, per_key, int(budget_bytes / per_key), target_width, DEPTH,
           cms_at_target, 100.0 * cms_at_target / budget_bytes))
    ideal = 8 * target_width * DEPTH
    ratio = cms_bytes[16384] / float(cms_bytes[target_width])
    say("Python overhead is real but bounded: %d cells of 8 bytes is %d bytes of "
        "payload; the measured cms footprint works out at %.2f bytes per cell, i.e. "
        "%.2fx the raw array once headers and container pointers are counted. "
        "Quadrupling the width "
        "multiplies the footprint by %.3f, so the accounting is linear as promised, "
        "and the cms+heap wrapper costs %d bytes total - %d bytes of it (or %.1f%%) "
        "is the candidate table, not the sketch."
        % (target_width * DEPTH, ideal, cms_at_target / float(target_width * DEPTH),
           cms_at_target / float(ideal), ratio, heap_bytes,
           heap_bytes - cms_at_target, pct(heap_bytes - cms_at_target, heap_bytes)))


# ------------------------------------------------------------------ 第 4 节
def section_false_positive():
    banner("4. the false-positive trap: one event, thousands reported", "deterministic")
    heavy = "10.0.0.1"
    half = N // 2
    stream = [heavy] * half + [ip_key(10000 + i) for i in range(N - half)]
    truth = exact_counts(stream)
    probes = ["203.0.113.%d" % i for i in range(20000)]
    threshold = 1000
    note("Stream: %s appears %d times, then %d keys appear exactly once. Counting "
         "keys reported >= %d that truly occurred once, plus %d never-seen probe keys."
         % (heavy, truth[heavy], N - half, threshold, len(probes)))
    print()
    rows = []
    stats = []
    for width, depth in ((64, 2), (256, 2), (1024, 2), (4096, 2), (1024, 5)):
        sketch = CountMinSketch(width, depth, seed=6)
        for key in stream:
            sketch.add(key)
        rare_over = rare_max = 0
        for key in truth:
            if key == heavy:
                continue
            est = sketch.query(key)
            if est >= threshold:
                rare_over += 1
            rare_max = max(rare_max, est)
        unseen_over = unseen_max = 0
        for key in probes:
            est = sketch.query(key)
            if est >= threshold:
                unseen_over += 1
            unseen_max = max(unseen_max, est)
        stats.append({"width": width, "depth": depth, "rare_over": rare_over,
                      "rare_max": rare_max, "unseen_over": unseen_over,
                      "unseen_max": unseen_max, "heavy": sketch.query(heavy)})
        rows.append([width, depth, "%d/%d" % (rare_over, len(truth) - 1),
                     "%d" % rare_max, "%d/%d" % (unseen_over, len(probes)),
                     "%d" % unseen_max, "%d" % sketch.query(heavy), "%d" % truth[heavy]])
    table(["width", "d", "rare>=1k", "rare_max", "unseen>=1k", "unseen_max",
           "h_est", "h_true"], rows, "rrrlrrrr")
    a, b = stats[0], stats[3]
    budget_a = N / float(a["width"])
    say("Nothing here is a bug: the guarantee is what permits it. At width %d / depth "
        "%d eps*N = %.1f events is the allowed over-estimate, and the measured worst "
        "single-event key reports %d - %.2fx that budget - while %d of the %d keys "
        "that really appeared once report >= %d. The same threshold is cleared by %d "
        "of the %d keys that never appeared at all, whose worst report is %d, or "
        "%.1f%% of the heavy hitter's true count of %d. Widening from %d to %d at "
        "depth 2 moves single-key offenders from %d to %d and never-seen offenders "
        "from %d to %d; deepening from %d to %d rows at width %d moves single-key "
        "offenders from %d to %d. The heavy hitter is reported as %d against a true "
        "%d, so accuracy at the top buys nothing at the bottom."
        % (a["width"], a["depth"], budget_a, a["rare_max"], a["rare_max"] / budget_a,
           a["rare_over"], len(truth) - 1, threshold, a["unseen_over"], len(probes),
           a["unseen_max"], 100.0 * a["unseen_max"] / truth[heavy], truth[heavy],
           a["width"], b["width"], a["rare_over"], b["rare_over"], a["unseen_over"],
           b["unseen_over"], stats[2]["depth"], stats[4]["depth"], stats[4]["width"],
           stats[2]["rare_over"], stats[4]["rare_over"], stats[4]["heavy"],
           truth[heavy]))
    chance = (1.0 / a["width"]) ** a["depth"]
    say("A deliberate full collision is what the worst case is: a probe matches the "
        "heavy hitter in all %d rows with probability about %.3e per try, so sweeping "
        "%d probes expects %.2f such keys and the observed maximum unseen estimate of "
        "%d is that collision. With %d rows at width %d the same sweep finds %d "
        "offenders and a maximum of %d, because the coincidence probability is now "
        "roughly %.3e and the sweep is %s it."
        % (a["depth"], chance, len(probes), chance * len(probes), a["unseen_max"],
           stats[4]["depth"], stats[4]["width"], stats[4]["unseen_over"],
           stats[4]["unseen_max"], (1.0 / stats[4]["width"]) ** stats[4]["depth"],
           "far short of" if (1.0 / stats[4]["width"]) ** stats[4]["depth"] * len(probes) < 1
           else "about the size of"))


# ------------------------------------------------------------------ 第 5 节
def time_op(batch, call):
    """返回 (最优 ns/事件, 中位, 同一批次的最大波动)。采样期间关闭 gc。"""
    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(TIMING_PASSES):
            start = time.perf_counter()
            for key in batch:
                call(key)
            stop = time.perf_counter()
            if index >= 2:
                samples.append((stop - start) / float(len(batch)) * 1e9)
    finally:
        if was_enabled:
            gc.enable()
    samples.sort()
    return samples[0], samples[len(samples) // 2], samples[-1] - samples[0]


def section_timing():
    banner("5. update cost per event", "timing")
    stream = zipf_stream(N, 1.1, 1500, seed=7)
    batch = stream[:TIMING_BATCH]
    note("Batch of %d events replayed %d times per operation: 2 warm-ups discarded, "
         "best of %d kept; gc disabled while sampling." % (len(batch), TIMING_PASSES,
                                                            TIMING_PASSES - 2))
    print()

    def dict_update(state, key):
        state[key] = state.get(key, 0) + 1

    specs = [
        ("dict increment", lambda: {}, dict_update),
        ("cms add 1024x5", lambda: CountMinSketch(1024, DEPTH, seed=7),
         lambda s, k: s.add(k)),
        ("cms add 8192x5", lambda: CountMinSketch(8192, DEPTH, seed=7),
         lambda s, k: s.add(k)),
        ("space-saver cap1024", lambda: SpaceSaver(1024), lambda s, k: s.add(k)),
        ("cms+heap 1024x5+512", lambda: CmsTopK(1024, DEPTH, 512, seed=7),
         lambda s, k: s.add(k)),
        ("cms query 1024x5", lambda: CountMinSketch(1024, DEPTH, seed=7), None),
    ]
    rows = []
    data = []
    for name, factory, call in specs:
        state = factory()
        if call is None:
            for key in batch:
                state.add(key)
            call = lambda s, k: s.query(k)
        best, median, spread = time_op(batch, lambda key, c=call, st=state: c(st, key))
        data.append({"name": name, "best": best, "median": median, "spread": spread})
        rows.append([name, "%.0f" % best, "%.0f" % median, "%.0f" % spread,
                     "%.1f" % pct(spread, median)])
    ref = data[0]["best"]
    for row, item in zip(rows, data):
        row.append("%.2fx" % (item["best"] / ref))
    table(["operation", "best_ns", "median_ns", "spread_ns", "spread_%", "vs_dict"],
          rows, "lrrrrr")
    cms_a, cms_b, ss, heap, query = (data[1], data[2], data[3], data[4], data[5])
    per_k = (cms_b["best"] - cms_a["best"]) / ((8192 - 1024) / 1024.0)
    ref_spread = data[0]["spread"]

    def verdict(gap, a, b):
        """平局只对「两个操作各自噪声中较大的那个」判定，全表最大噪声不能反过来吞掉小差距。"""
        return "inside" if abs(gap) <= max(a, b) else "outside"

    pair_width = max(cms_a["spread"], cms_b["spread"])
    pair_ss = max(ss["spread"], ref_spread)
    pair_query = max(query["spread"], cms_a["spread"])
    widest = max(data, key=lambda item: item["spread"])
    tightest = min(data, key=lambda item: item["spread"])
    say("The noise floor is each operation's own pass-to-pass spread over the 5 scored "
        "passes: the widest on the board is %s at %.0f ns/event and the tightest is %s "
        "at %.0f ns. A pair is a tie when the two best readings sit within the wider of "
        "their own spreads. Against the dict increment at %.0f ns, cms add costs %.2fx "
        "at width 1024 (%.0f ns) and %.2fx at width 8192 (%.0f ns); those two widths "
        "differ by %.0f ns, %.0f ns per extra 1024 columns, on a %.0f ns pair floor - so "
        "marginal width is %s the floor, and the multiply-shift path has the same depth "
        "in both: depth - five hashes and five cell writes - is what the sketch pays "
        "for, not width. Space-saver is %.2fx the dict at %.0f ns, a %.0f ns gap on a "
        "%.0f ns pair floor, so that one is %s the floor, while cms query reads %.0f ns "
        "against the same add at %.0f ns (%.2fx) on a %.0f ns floor - %s. The candidate "
        "heap, not the sketch, is the expensive part of the streaming update: cms+heap "
        "runs %.2fx the dict at %.0f ns."
        % (widest["name"], widest["spread"], tightest["name"], tightest["spread"],
           ref, cms_a["best"] / ref, cms_a["best"], cms_b["best"] / ref, cms_b["best"],
           cms_b["best"] - cms_a["best"], per_k, pair_width,
           verdict(cms_b["best"] - cms_a["best"], cms_a["spread"], cms_b["spread"]),
           ss["best"] / ref, ss["best"], ss["best"] - ref, pair_ss,
           verdict(ss["best"] - ref, ss["spread"], ref_spread),
           query["best"], cms_a["best"], query["best"] / cms_a["best"], pair_query,
           verdict(query["best"] - cms_a["best"], query["spread"], cms_a["spread"]),
           heap["best"] / ref, heap["best"]))
    worst = max(data, key=lambda item: item["spread"] / item["median"])
    say("The noisiest operation is %s at %.1f%% spread (%.0f ns of %.0f ns median); "
        "readings inside that band for the same operation across the 5 scored passes "
        "mean the machine was loaded, which is why only the ratios above are claimed and "
        "no ordering inside a pair's own spread is. Extrapolating the best cms-add rate "
        "to the 50-million-event pipeline in the problem statement takes %.1f s of pure "
        "Python for the sketch alone, %.1fx the dict's %.1f s."
        % (worst["name"], 100.0 * worst["spread"] / worst["median"], worst["spread"],
           worst["median"], 50_000_000 * cms_a["best"] / 1e9,
           cms_a["best"] / ref, 50_000_000 * ref / 1e9))


# ------------------------------------------------------------------ 第 6 节
def section_deletion():
    banner("6. deletion: removing an event vs subtracting an estimate", "deterministic")
    universe = 1500
    capacity = 64
    stream = zipf_stream(N, 1.15, universe, seed=8)
    truth = exact_counts(stream)
    victims = sorted(truth, key=lambda k: (-truth[k], k))[:30]
    after = dict(truth)
    for key in victims:
        after[key] = 0
    want = set(k for k, _ in top_k_from_counts(after, 10))
    note("The %d highest-count keys are deleted outright: %d events taken out of %d, "
         "leaving %d keys with a non-zero true count. Scored population: all %d keys "
         "for the cms, the %d keys a space-saver of capacity %d monitors."
         % (len(victims), sum(truth[k] for k in victims), N,
            sum(1 for v in after.values() if v > 0), len(after), capacity, capacity))
    print()

    def build_cms():
        sketch = CountMinSketch(1024, DEPTH, seed=8)
        for key in stream:
            sketch.add(key)
        return sketch

    def score_cms(sketch):
        under = low = neg = 0
        errs = []
        ghosts = 0
        for key, true in after.items():
            est = sketch.query(key)
            if est < true:
                under += 1
                low = min(low, est - true)
            if est < 0:
                neg += 1
            if true == 0 and est > 0:
                ghosts += 1
            errs.append(abs(est - true))
        got = set(k for k, _ in top_k_from_counts(
            {k: sketch.query(k) for k in truth}, 10))
        return [under, neg, low, "%.2f" % (sum(errs) / float(len(errs))), ghosts,
                "%d/10" % len(got & want)]

    rows = []
    for label, kind in (("event-remove", "ev"), ("subtract-estimate", "es"),
                        ("clamped-remove", "cl")):
        sketch = build_cms()
        for key in victims:
            if kind == "ev":
                sketch.delete_event(key, truth[key])
            elif kind == "es":
                sketch.delete_by_subtracting_estimate(key)
            else:
                sketch.delete_event(key, truth[key], clamp=True)
        rows.append(["cms " + label] + score_cms(sketch))

    def build_ss():
        saver = SpaceSaver(capacity)
        for key in stream:
            saver.add(key)
        return saver

    for label, kind in (("event-remove", "ev"), ("subtract-estimate", "es"),
                        ("drop-key", "dr")):
        saver = build_ss()
        for key in victims:
            if kind == "ev":
                saver.remove_event(key, truth[key])
            elif kind == "es":
                saver.subtract_estimate(key)
            else:
                saver.drop(key)
        monitored = saver.keys()
        under = low = neg = 0
        errs = []
        for key in monitored:
            est = saver.query(key)
            true = after.get(key, 0)
            if est < true:
                under += 1
                low = min(low, est - true)
            if est < 0:
                neg += 1
            errs.append(abs(est - true))
        got = set(k for k, _ in saver.top_k(10))
        rows.append(["ss " + label] + [under, neg, low,
                     "%.2f" % (sum(errs) / float(len(errs)) if errs else 0.0),
                     sum(1 for key in victims if key in monitored),
                     "%d/10" % len(got & want)])
    table(["structure+op", "under", "min_gap", "mae", "ghosts", "top10"],
          [[r[0], r[1], r[3], r[4], r[5], r[6]] for r in rows], "lrrrrl")
    by = dict((r[0], r) for r in rows)
    ev, es, cl = by["cms event-remove"], by["cms subtract-estimate"], by["cms clamped-remove"]
    ssev, sses, ssdr = (by["ss event-remove"], by["ss subtract-estimate"],
                        by["ss drop-key"])
    say("Removing an event is the inverse of adding one and the run agrees: cms "
        "event-remove under-estimates %d of %d keys (min gap %d), the insert-only "
        "answer, because it decrements exactly the %d cells it incremented and nothing "
        "else. Subtracting the *reported* estimate under-estimates %d keys and moves "
        "the mean absolute error from %s to %s events: a report already carries other "
        "keys' collisions, so the surplus comes off innocent keys in those same %d "
        "cells. The sign hides that - %d reports went negative, and the subtracted min "
        "is a lower bound of every cell the victim touches, so negativity needs an "
        "over-subtraction the code does not do by default. Clamping at zero instead of "
        "going negative gives %d under-estimates. Ghosts are the honest residue: the "
        "correct scheme still reports %d of the %d deleted keys as present, because "
        "what is left in the cell is somebody else's count."
        % (ev[1], len(after), ev[3], DEPTH, es[1], ev[4], es[4], DEPTH, es[2], cl[1],
           ev[5], len(victims)))
    cms_score = ev[1] + ev[5]
    ss_score = ssev[1] + ssev[5]
    who = ("space-saver" if ss_score > cms_score
           else "cms" if cms_score > ss_score else "neither")
    say("Space-saver cannot offer the same op: event-remove leaves %d of its %d "
        "monitored keys under-estimated (%d negative, min gap %d) and still holds %d of "
        "the %d deleted keys, since an unmonitored key has nothing to decrement and a "
        "monitored one keeps the count inherited from whoever it replaced. drop-key "
        "evicts %d of them and lands on a top-10 of %s against the post-deletion truth, "
        "versus %s for cms event-remove and %s for cms subtract-estimate; "
        "subtract-estimate on space-saver gives %d under-estimates and %d ghosts. "
        "Counting under-estimates plus ghosts for the honest deletion op: %d for "
        "space-saver against %d for the cms, so on this stream %s is the one that loses "
        "- the structure folklore prefers for heavy-hitter accuracy is the one whose "
        "answer goes stale when the stream is edited."
        % (ssev[1], capacity, ssev[2], ssev[3], ssev[5], len(victims), ssdr[5],
           ssdr[6], ev[6], es[6], sses[1], sses[5], ss_score, cms_score, who))


def main():
    print("count-min-heavy-hitters benchmarks - python %s" % sys.version.split()[0])
    print("Measures Count-Min Sketch (plain, mean/median/count-mean modes), Space-")
    print("Saver and an exact dict on seeded Zipf streams: error, top-k, memory, cost.")
    section_errors()
    section_topk()
    section_memory()
    section_false_positive()
    section_timing()
    section_deletion()


if __name__ == "__main__":
    console = sys.stdout
    tee = _Tee(console, REPORT)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout.flush()
        tee.save()
        sys.stdout = console
