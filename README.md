# count-min-heavy-hitters

A stdlib-only Python toolkit for one question: which source IPs dominate a log stream
when you cannot afford a counter per IP. It implements a Count-Min Sketch (min-of-rows
plus mean/median/count-mean modes), a Space-Saver heavy-hitter table, a CMS+heap top-k
wrapper and an exact `dict` baseline, and measures all of them against that baseline on
seeded Zipf streams. Four findings from `benchmarks/last_run.txt` (200,000 events a
section):

1. **Plain CMS never under-estimated**: 0 under-estimates in 8,000 key-width trials on
   the insert-only stream, count-mean under-estimated 7,999. Under-estimation appeared
   only where a delete was not the inverse of an insert: subtracting a reported estimate
   broke 150 keys (worst gap -126), removing the same events along the hash path broke 0.
2. **CMS+heap beat Space-Saver on top-k recall** at a matched 64-slot budget (mean 0.989
   vs 0.671 over 9 settings, Space-Saver's floor 0.140) - yet Space-Saver's *counts* were
   exact (mae 0.0) where the CMS was off by 1.3-1.8. Ranking and counting disagree.
3. **Memory**: at the matched additive target of 48.8 events the sketch costs 164,636
   bytes = 0.037 of the dict's 4,394,588, but a 4096-slot candidate heap takes the
   wrapper to 796,658 bytes, 79.3% heap.
4. **The false-positive trap is the guarantee, not a bug**: at width 64 / depth 2 all
   100,000 single-event keys and all 20,000 never-seen probes reported >= 1000, worst
   101,531 against the heavy hitter's true 100,000.

## What it does / does not do

Does: fixed-memory frequency estimation, top-k heavy hitters over a replayable stream,
error accounting that says which estimator under-estimates and when, and the two deletion
operations people conflate kept apart.

Does not: it is not a live-pipeline structure - pure Python spends ~5.0-5.5 us per sketch
update, so 50M events take ~273 s (measured). It gives no "this key appeared once"
answer, treats negative counts as a documented footgun, and has no concurrency,
persistence or decay window.

## Install

```bash
git clone https://github.com/sparkx313/count-min-heavy-hitters.git
cd count-min-heavy-hitters
python -u -m unittest discover -s tests -p 'test_*.py' -t tests   # 80 cases, no deps
```

## Quick start

Sketch modes on the heaviest key of a 50,000-event stream (true count 9,012):

```python
from sketchlib import CountMinSketch, exact_counts, top_k_from_counts, zipf_stream

stream = zipf_stream(50000, skew=1.1, universe=1000, seed=1)
truth = exact_counts(stream)
cms = CountMinSketch(width=1024, depth=5, seed=1)
for key in stream:
    cms.add(key)
head = top_k_from_counts(truth, 1)[0]
print("head", head, "cms", cms.query(head[0]))
print("gap min/max", min(cms.query(k) - c for k, c in truth.items()),
      max(cms.query(k) - c for k, c in truth.items()))
print("modes", cms.estimate_all(head[0]))
```

    head ('10.0.0.1', 9012) cms 9017
    gap min/max 0 29
    modes {'min': 9017, 'mean': 9060.4, 'median': 9034, 'count_mean': 8976.980840664712}

min-of-rows is over by 5 and never under (gap minimum 0 over all 1,000 keys);
`count_mean` returns 8,976.98, i.e. under the truth.

Top-k with the same 64-slot budget, then the two deletion ops on that heaviest key:

```python
from sketchlib import (CountMinSketch, CmsTopK, SpaceSaver, exact_counts,
                       top_k_from_counts, zipf_stream)

stream = zipf_stream(50000, skew=1.1, universe=1000, seed=1)
truth = exact_counts(stream)
victim = top_k_from_counts(truth, 1)[0][0]
saver = SpaceSaver(capacity=64)
topk = CmsTopK(1024, 5, 64, seed=1)
for key in stream:
    saver.add(key)
    topk.add(key)
print("ss top3", saver.top_k(3))
print("heap top3", topk.top_k(3))
a = CountMinSketch(1024, 5, seed=1)
b = CountMinSketch(1024, 5, seed=1)
for key in stream:
    a.add(key)
    b.add(key)
a.delete_event(victim, truth[victim])
b.delete_by_subtracting_estimate(victim)
print("event-remove:", a.query(victim), "under-estimates:",
      sum(1 for k, c in truth.items() if k != victim and a.query(k) < c))
print("subtract-estimate:", b.query(victim), "under-estimates:",
      sum(1 for k, c in truth.items() if k != victim and b.query(k) < c))
```

    ss top3 [('10.0.0.1', 9012), ('10.0.0.2', 4210), ('10.0.0.3', 2607)]
    heap top3 [('10.0.0.1', 9017), ('10.0.0.2', 4210), ('10.0.0.3', 2607)]
    event-remove: 5 under-estimates: 0
    subtract-estimate: 0 under-estimates: 1

Event-remove leaves 5 events of *other* keys in the victim's cells (correct: they are
really there) and under-estimates nobody. Subtract-estimate erases 9,017 from every row,
so one innocent colliding key loses count that was never its own.

## Measured results

Quoted line-for-line from `benchmarks/last_run.txt`. Deterministic tables replay
identically; only the timing table moves.

### 1. Error versus the guarantee

```text
width  eps*N   delta  mean_ov  p95  max_ov  gt_epsN  u_min  u_mean  u_cmean
---------------------------------------------------------------------------
  256  781.2  0.0312    122.2  251    1038        1      0       0     1999
 1024  195.3  0.0312      8.9   36     135        0      0       0     2000
 4096   48.8  0.0312      0.1    0      22        0      0       0     2000
16384   12.2  0.0312      0.0    0       0        0      0       0     2000
```

`u_min`/`u_mean` are 0 at every width: min-, mean- and median-of-rows never
under-estimate the 2,000 tracked keys. `u_cmean` is 1,999-2,000: count-mean subtracts a
row's average other-key load from the *minimum* of 5 rows, which already sits below that
average. Mean over-estimate tracks `eps*N` (122.2 vs a 781.2 budget at width 256; 0.1 vs
48.8 at 4096), and exactly 1 key exceeds the budget at width 256, where delta = 0.0312
licenses about 62.

```text
op                 victims  u_min  u_mean  u_med  u_cmean  worst_gap
--------------------------------------------------------------------
event-remove       100          0       0      0     1983        -54
subtract-estimate  100        150       1      0     1981       -126
clamped-remove     100          0       0      0     1983        -54
```

Same sketch, the 100 highest-count keys deleted: `u_min` goes 0 -> 150 (worst gap -126).
min-of-rows breaks only when a decrement is not the inverse of its own increment.

### 2. Top-k: CMS+heap versus Space-Saver

```text
 k  skew  method       overlap   prec  recall     mae  size
-----------------------------------------------------------
 5  0.7   cms+heap         5/5  1.000   1.000    99.2     5
 5  0.7   space-saver      2/5  0.400   0.400  2190.6     5
10  0.7   cms+heap       10/10  1.000   1.000   109.0    10
10  0.7   space-saver     3/10  0.300   0.300  2340.4    10
50  0.7   cms+heap       46/50  0.920   0.920   117.2    50
50  0.7   space-saver     7/50  0.140   0.140  2917.7    50
 5  1.1   cms+heap         5/5  1.000   1.000    25.4     5
 5  1.1   space-saver      5/5  1.000   1.000     0.0     5
10  1.1   cms+heap       10/10  1.000   1.000    30.3    10
10  1.1   space-saver    10/10  1.000   1.000     1.8    10
50  1.1   cms+heap       49/50  0.980   0.980    37.1    50
50  1.1   space-saver    20/50  0.400   0.400  1519.7    50
 5  1.6   cms+heap         5/5  1.000   1.000     1.8     5
 5  1.6   space-saver      5/5  1.000   1.000     0.0     5
10  1.6   cms+heap       10/10  1.000   1.000     1.3    10
10  1.6   space-saver    10/10  1.000   1.000     0.0    10
50  1.6   cms+heap       50/50  1.000   1.000     1.7    50
50  1.6   space-saver    40/50  0.800   0.800   154.8    50
```

At flat skew (0.7) Space-Saver names 2/5, 3/10, 7/50 of the true top-k against
CMS+heap's 5/5, 10/10, 46/50; at skew 1.6 it is 40/50 vs 50/50 and its counts are exact
(`mae` 0.0 in four rows where CMS+heap is 1.3-1.8). Precision equals recall because both
return k keys (`size` == `k`). Contradiction, stated plainly: the structure usually
recommended for heavy hitters is worse at naming them and better at counting them.

### 3. Memory versus the exact dict

```text
structure    geometry          tracked    bytes     KiB  err<=N/x  x_dict
-------------------------------------------------------------------------
exact dict   1 hash map          39062  4394588  4291.6         0   1.000
cms          1024x5           all keys    41756    40.8     195.3   0.010
cms          4096x5           all keys   164636   160.8      48.8   0.037
cms          16384x5          all keys   656156   640.8      12.2   0.149
cms+heap     4096x5+heap4096      4096   796658   778.0      48.8   0.181
space-saver  cap=1024             1024   236232   230.7     195.3   0.054
space-saver  cap=4096             4096   983597   960.5      48.8   0.224
space-saver  cap=16384           16384  3971605  3878.5      12.2   0.904
```

39,062 distinct keys cost the dict 4,394,588 bytes (112.5/key). Matched to the same
additive bound of 48.8 events (`eps*N` at width 4096, `N/C` at capacity 4096) the sketch
is 164,636 bytes: 6.0x smaller than Space-Saver's 983,597 and 0.037 of the dict. The
sketch's footprint ignores the key count, the dict's does not - that is the whole
argument. Two caveats in the same rows: the candidate heap adds 632,022 bytes (79.3% of
the wrapper is heap, not sketch), and Space-Saver scales with capacity (3,971,605 bytes
at cap=16384, 0.904 of the dict). The measured footprint works out at 8.04 bytes per
cell - 1.00x the raw 8-byte array once headers and container pointers are counted - and
quadrupling the width multiplied the footprint by 3.985.

### 4. The false-positive trap

```text
width  d       rare>=1k  rare_max   unseen>=1k  unseen_max   h_est  h_true
--------------------------------------------------------------------------
   64  2  100000/100000  101531    20000/20000      101531  101531  100000
  256  2       0/100000  448           0/20000         444  100393  100000
 1024  2       0/100000  127           0/20000         125  100097  100000
 4096  2       0/100000  40            0/20000          39  100025  100000
 1024  5       0/100000  110           0/20000         106  100091  100000
```

One key repeated 100,000 times plus 100,000 keys appearing exactly once; offenders are
keys reported >= 1000. At width 64 / depth 2 every real single-event key and all 20,000
never-seen probes clear it, worst 101,531: that probe hit the heavy hitter's cell in both
rows, so a key with zero events reports 101.5% of the heavy hitter's own count. Width 256
drops offenders to 0 and the worst report to 448 while `h_est` stays 100,393 against a
true 100,000 - accurate at the top, worthless at the bottom, and `eps*N` = 3,125 at width
64 is what licenses it.

### 5. Update cost per event

```text
operation            best_ns  median_ns  spread_ns  spread_%  vs_dict
---------------------------------------------------------------------
dict increment           191        199         20      10.2    1.00x
cms add 1024x5          5451       5496        447       8.1   28.57x
cms add 8192x5          5025       5050        459       9.1   26.34x
space-saver cap1024      792        808         27       3.4    4.15x
cms+heap 1024x5+512    11983      12458        984       7.9   62.80x
cms query 1024x5        4612       4743        259       5.5   24.17x
```

The noise floor is each operation's own pass-to-pass spread, and a pair is judged against
the wider of the two: the widest band here is cms+heap at 984 ns/event, the tightest the
dict at 20 ns. Width 1024 (5,451 ns) against 8192 (5,025 ns) differs by 426 ns on a 459 ns
pair floor - a tie, so extra width buys nothing measurable, and depth (5 hashes, 5 cell
writes) is what the sketch pays for. Outside their floors: Space-Saver 4.15x the dict
(792 ns, a 601 ns gap on a 27 ns band) and cms+heap 62.80x (11,983 ns) - the candidate
heap, not the sketch, is the expensive part of the streaming update. Query at 4,612 ns is
0.85x a plain add of the same geometry on a 447 ns band, so reading really is cheaper than
writing. At 5,451 ns/event, 50M events cost ~273 s of pure Python.

### 6. Deletion semantics

```text
structure+op           under  min_gap      mae  ghosts  top10
-------------------------------------------------------------
cms event-remove           0        0     4.14       6  10/10
cms subtract-estimate     22      -61     4.26       0  10/10
cms clamped-remove         0        0     4.14       6  10/10
ss event-remove            0        0  1315.71      12  2/10
ss subtract-estimate       0        0  1569.25       0  2/10
ss drop-key                0        0  1569.25       0  2/10
```

The 30 heaviest keys are deleted outright. CMS event-remove keeps the invariant (under 0,
min_gap 0) and still gets 10/10 of the post-deletion top-10; subtract-estimate pushes 22
keys under, min_gap -61. Space-Saver shows 0 under-estimates only because its scored
population is the 64 keys it chose to monitor - the same rows show it holding 12 of the
30 deleted keys as ghosts and landing 2/10, and `drop-key` clears the ghosts while
leaving the ranking at 2/10. Ghosts for the honest op: 6 for the CMS against 12 for
Space-Saver, so the structure folklore prefers is the one whose answer goes stale when
the stream is edited.

## How it decides

* `eps = 1/w`: with non-negative updates `query_min(key) >= true`, and the expected
  over-estimate is at most `eps * N`.
* `delta = 2 ** -d`: `P(over > eps*N) <= delta`, so `d = ceil(log2(1/delta))` rows;
  d=5 gives 0.0312, d=2 gives 0.25, which is why section 4's depth-2 sketch is where
  the trap is reachable.
* Geometry: for an additive error `t` on `N` events take `w = N / t`,
  `d = ceil(log2(1/delta))`, memory `8 * w * d` bytes of cells.
* `count_mean = min - mean_i((row_total_i - value_i) / (w - 1))` removes a row's
  expected collision load; it only pays when `w` is large relative to the distinct-key
  count, otherwise it subtracts noise the minimum already removed and under-estimates.
* Space-Saver keeps `C` counters with `count - error <= true <= count`, so its
  over-estimate is bounded by `N / C` (match the CMS by `C = N / t`). Unmonitored keys
  report 0: it trades the tail for the head, the mirror image of the CMS guarantee.
* A CMS cannot enumerate keys, so its top-k needs a candidate table; the wrapper
  re-reads the sketch before ranking, so its answers cost sketch + table.

## Limitations

* ~5.4-5.7 us per sketch update in CPython; nothing vectorised. `blake2b`-based hashing
  is most of that budget.
* Memory comes from `sys.getsizeof` under CPython; a 32-bit cell array would halve the
  absolute bytes (signed 64-bit cells are chosen so deletes cannot wrap).
* Decrementing along the hash path is exact; subtracting a report or clamping at zero
  silently corrupts colliding keys. Space-Saver cannot delete an unmonitored key at all.
* Zipf/power-law streams only; an adversary that controls key names can force collisions.
* Top-k ties are broken by key name, so precision hides ties instead of reporting them.

## API

| Name | Signature | Notes |
| --- | --- | --- |
| `CountMinSketch` | `(width=1024, depth=5, seed=0)` | array-backed d x w sketch |
| `.add` / `.query` / `.estimate_all` | `(key, count=1)` / `(key, mode="min")` | d cell writes; `min`/`mean`/`median`/`count_mean`, all four modes in one hash pass |
| `.delete_event` | `(key, count=1, clamp=False)` | exact inverse of `add`; `clamp` is not |
| `.delete_by_subtracting_estimate` | `(key, mode="min", scale=1.0)` | removes the report: hurts innocent collisions |
| `.cells` / `.bytes_used` / `.compact_bytes` | `()` | geometry, measured footprint, payload |
| `SpaceSaver` | `(capacity)` | weighted Misra-Gries / Stream-Summary |
| `.add` `.remove_event` `.subtract_estimate` `.drop` | `(key, count=1)` | insert, event delete, delete the report, evict |
| `.query` `.top_k` `.keys` `.over_estimate_bound` | various | 0 if unmonitored; ranked list; `total // capacity` |
| `CmsTopK` | `(width, depth, capacity, seed, mode)` | sketch + bounded candidate heap |
| `zipf_stream` / `uniform_stream` | `(count, skew, universe, seed)` | deterministic key lists |
| `exact_counts` / `top_k_from_counts` | `(keys)` / `(counts, k)` | dict baseline; exact top-k, ties by name |
| `ip_key` / `zipf_weights` / `MODES` | helpers | rank -> key, rank weights, mode names |

## Directory tree

```tree
count-min-heavy-hitters/
|-- .gitattributes
|-- .gitignore
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- benchmarks/
|   |-- last_run.txt
|   `-- run_benchmarks.py
|-- sketchlib/
|   |-- __init__.py
|   |-- cms.py
|   |-- space_saver.py
|   |-- streams.py
|   `-- topk.py
`-- tests/
    `-- test_sketch.py
```

## Reproduce

```bash
cd count-min-heavy-hitters
python -u benchmarks/run_benchmarks.py    # ~65 s, rewrites benchmarks/last_run.txt
```

The script tees stdout into `benchmarks/last_run.txt`, so the tables above are that text.
Seeded sections replay exactly; the timing table does not (that run's widest per-operation band
was 984 ns/event, 7.9% of its own median). `CMH_SCALE=0.05` runs it at 5% of the events.

## License

MIT. See `LICENSE`. Copyright (c) 2026 sparkx313.
