"""结构不变量的单元测试：误差方向、宽度/深度记账、堆行为、删除语义、确定性。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sketchlib import (CountMinSketch, SpaceSaver, CmsTopK, exact_counts,
                       ip_key, top_k_from_counts, uniform_stream, zipf_stream,
                       zipf_weights)


def same_col_partner(sketch, key):
    """在浅草图里暴力找一个与 key 同行冲突的伙伴键，用于构造碰撞。"""
    cols = sketch._cols(key)
    probe = 0
    while True:
        probe += 1
        cand = "probe-%d" % probe
        if sketch._cols(cand) == cols and cand != key:
            return cand


class CountMinBasics(unittest.TestCase):
    def test_empty_sketch_queries_zero(self):
        s = CountMinSketch(width=64, depth=4)
        self.assertEqual(s.query("never-seen"), 0)

    def test_empty_sketch_all_modes_zero(self):
        s = CountMinSketch(64, 4)
        self.assertEqual(s.estimate_all("x"),
                         {"min": 0, "mean": 0.0, "median": 0.0, "count_mean": 0.0})

    def test_add_single_key_min(self):
        s = CountMinSketch(128, 5)
        s.add("a", 7)
        self.assertEqual(s.query("a"), 7)

    def test_min_never_underestimates_on_uniform_stream(self):
        s = CountMinSketch(64, 5, seed=3)
        stream = uniform_stream(20000, universe=500, seed=4)
        for k in stream:
            s.add(k)
        for key, true in exact_counts(stream).items():
            self.assertGreaterEqual(s.query(key, "min"), true)

    def test_mean_and_median_never_underestimate(self):
        s = CountMinSketch(64, 5, seed=3)
        stream = uniform_stream(8000, universe=400, seed=9)
        for k in stream:
            s.add(k)
        for key, true in exact_counts(stream).items():
            est = s.estimate_all(key)
            self.assertGreaterEqual(est["mean"], true)
            self.assertGreaterEqual(est["median"], true)

    def test_mean_is_at_least_min(self):
        s = CountMinSketch(32, 5, seed=1)
        for k in zipf_stream(4000, skew=1.0, universe=200, seed=2):
            s.add(k)
        for key in ["10.0.0.1", "10.0.0.2", "10.0.0.7"]:
            est = s.estimate_all(key)
            self.assertGreaterEqual(est["mean"], est["min"])
            self.assertGreaterEqual(est["median"], est["min"])

    def test_count_mean_can_underestimate(self):
        s = CountMinSketch(128, 5, seed=7)
        stream = uniform_stream(60000, universe=128, seed=8)
        for k in stream:
            s.add(k)
        counts = exact_counts(stream)
        found = [key for key, true in counts.items()
                 if s.query(key, "count_mean") < true]
        self.assertTrue(found, "noise-subtracting estimator should bite somewhere")

    def test_count_mean_leq_min(self):
        s = CountMinSketch(64, 4, seed=2)
        for k in zipf_stream(3000, 1.1, 300, seed=5):
            s.add(k)
        est = s.estimate_all("10.0.0.3")
        self.assertLessEqual(est["count_mean"], est["min"])

    def test_width_depth_accounting(self):
        s = CountMinSketch(width=37, depth=6)
        self.assertEqual(s.cells(), 37 * 6)
        self.assertEqual(len(s.rows), 6)
        self.assertTrue(all(len(r) == 37 for r in s.rows))
        self.assertEqual(len(s.row_totals), 6)

    def test_compact_bytes_formula(self):
        s = CountMinSketch(256, 5)
        self.assertEqual(s.compact_bytes(), 256 * 5 * 8)

    def test_bytes_used_grows_with_width(self):
        small = CountMinSketch(256, 5).bytes_used()
        big = CountMinSketch(4096, 5).bytes_used()
        self.assertGreater(big, small)
        self.assertGreaterEqual(big, 4096 * 5 * 8)

    def test_row_totals_match_cell_sums(self):
        s = CountMinSketch(50, 4, seed=11)
        for k, c in [("a", 3), ("b", 9), ("c", 1)]:
            s.add(k, c)
        for i in range(4):
            self.assertEqual(s.row_totals[i], sum(s.rows[i]))

    def test_total_weight_tracks_updates(self):
        s = CountMinSketch(64, 3)
        s.add("a", 5)
        s.add("b", 2)
        s.add("a", -1)
        self.assertEqual(s.total_weight, 6)
        self.assertEqual(s.updates, 3)

    def test_same_seed_same_paths(self):
        a = CountMinSketch(512, 5, seed=42)
        b = CountMinSketch(512, 5, seed=42)
        self.assertEqual(a._cols("10.0.0.9"), b._cols("10.0.0.9"))

    def test_different_seed_uses_different_paths(self):
        a = CountMinSketch(512, 5, seed=1)
        b = CountMinSketch(512, 5, seed=2)
        self.assertNotEqual(a._cols("10.0.0.9"), b._cols("10.0.0.9"))

    def test_replay_of_seeded_stream_is_identical(self):
        s1 = CountMinSketch(64, 4, seed=0)
        s2 = CountMinSketch(64, 4, seed=0)
        for k in zipf_stream(1500, 1.05, 250, seed=17):
            s1.add(k)
        for k in zipf_stream(1500, 1.05, 250, seed=17):
            s2.add(k)
        self.assertEqual(list(s1.rows[0]), list(s2.rows[0]))

    def test_query_matches_estimate_all(self):
        s = CountMinSketch(64, 5, seed=4)
        for k in uniform_stream(2000, 100, seed=6):
            s.add(k)
        allv = s.estimate_all("10.0.0.5")
        for mode in ("mean", "median", "count_mean"):
            self.assertAlmostEqual(s.query("10.0.0.5", mode), allv[mode])

    def test_unknown_mode_raises(self):
        s = CountMinSketch(16, 2)
        with self.assertRaises(ValueError):
            s.query("a", "mode-not-here")

    def test_rejects_small_width(self):
        with self.assertRaises(ValueError):
            CountMinSketch(width=1, depth=3)

    def test_rejects_bad_depth(self):
        with self.assertRaises(ValueError):
            CountMinSketch(width=32, depth=0)

    def test_rejects_bad_capacity(self):
        with self.assertRaises(ValueError):
            CmsTopK(width=32, depth=2, capacity=0)

    def test_zero_count_is_noop(self):
        s = CountMinSketch(32, 3)
        s.add("a", 0)
        self.assertEqual((s.updates, s.query("a")), (0, 0))

    def test_huge_key_count_fits(self):
        s = CountMinSketch(8, 2)
        s.add("big", 10 ** 12)
        self.assertEqual(s.query("big"), 10 ** 12)

    def test_guarantee_holds_statistically(self):
        n, w = 40000, 256
        s = CountMinSketch(w, 5, seed=21)
        stream = uniform_stream(n, universe=1000, seed=22)
        for k in stream:
            s.add(k)
        eps_n = n / float(w)
        worst = max(s.query(key, "min") - true
                    for key, true in exact_counts(stream).items())
        self.assertLessEqual(worst, 6 * eps_n)

    def test_repr_mentions_geometry(self):
        self.assertIn("width=16", repr(CountMinSketch(16, 2)))

    def test_delete_event_is_exact_inverse(self):
        s = CountMinSketch(128, 5, seed=3)
        s.add("x", 10)
        s.add("y", 4)
        s.add("x", -10)
        self.assertEqual(s.query("x"), 0)
        self.assertEqual(s.query("y"), 4)
        self.assertEqual(s.total_weight, 4)

    def test_delete_event_keeps_informal_invariant(self):
        """沿原路删除后，其余键仍不低于真值：这正是它比"减估计"安全的原因。"""
        s = CountMinSketch(64, 4, seed=8)
        stream = uniform_stream(4000, universe=200, seed=8)
        for k in stream:
            s.add(k)
        victim = "10.0.0.17"
        s.add(victim, -7)
        counts = exact_counts(stream)
        counts[victim] = counts[victim] - 7
        for key, true in counts.items():
            self.assertGreaterEqual(s.query(key), true)


class SpaceSaverBehaviour(unittest.TestCase):
    def test_empty_query_zero(self):
        self.assertEqual(SpaceSaver(4).query("nope"), 0)

    def test_single_key_stream(self):
        ss = SpaceSaver(1)
        for _ in range(500):
            ss.add("only")
        self.assertEqual(ss.query("only"), 500)
        self.assertEqual(len(ss), 1)

    def test_monitored_never_underestimates(self):
        ss = SpaceSaver(32)
        stream = zipf_stream(6000, 1.1, 400, seed=5)
        for k in stream:
            ss.add(k)
        counts = exact_counts(stream)
        for key in ss.keys():
            self.assertGreaterEqual(ss.query(key), counts[key])

    def test_over_estimate_bound_holds(self):
        ss = SpaceSaver(16)
        stream = zipf_stream(5000, 1.2, 300, seed=13)
        for k in stream:
            ss.add(k)
        counts = exact_counts(stream)
        bound = ss.over_estimate_bound()
        self.assertGreaterEqual(bound, 1)
        for key in ss.keys():
            self.assertLessEqual(ss.query(key) - counts[key], bound)

    def test_capacity_one_keeps_a_heavy_key(self):
        ss = SpaceSaver(1)
        stream = zipf_stream(2000, 1.5, 50, seed=3)
        for k in stream:
            ss.add(k)
        top = max(exact_counts(stream).items(), key=lambda kv: kv[1])[0]
        self.assertIn(top, ss.keys())

    def test_len_bounded_by_capacity(self):
        ss = SpaceSaver(10)
        for k in uniform_stream(3000, universe=500, seed=2):
            ss.add(k)
        self.assertLessEqual(len(ss), 10)

    def test_top_k_sorted_descending(self):
        ss = SpaceSaver(20)
        for k in zipf_stream(3000, 1.1, 100, seed=1):
            ss.add(k)
        vals = [v for _, v in ss.top_k(10)]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_top_k_truncates(self):
        ss = SpaceSaver(30)
        for k in uniform_stream(900, universe=300, seed=7):
            ss.add(k)
        self.assertEqual(len(ss.top_k(3)), 3)

    def test_top_k_more_than_monitored(self):
        ss = SpaceSaver(5)
        ss.add("a", 3)
        self.assertEqual(ss.top_k(50), [("a", 3)])

    def test_deterministic_replay(self):
        a, b = SpaceSaver(12), SpaceSaver(12)
        for k in zipf_stream(1200, 1.0, 90, seed=4):
            a.add(k)
            b.add(k)
        self.assertEqual(a.top_k(12), b.top_k(12))

    def test_error_inherited_on_replacement(self):
        ss = SpaceSaver(2)
        ss.add("a", 5)
        ss.add("b", 4)
        ss.add("c", 1)
        self.assertEqual(ss.query("c"), 5)
        self.assertEqual(ss.error_of("c"), 4)

    def test_remove_event_drops_at_zero(self):
        ss = SpaceSaver(4)
        ss.add("a", 3)
        self.assertTrue(ss.remove_event("a", 3))
        self.assertEqual(ss.query("a"), 0)
        self.assertNotIn("a", ss.keys())

    def test_remove_event_on_unmonitored_is_noop(self):
        ss = SpaceSaver(1)
        ss.add("a", 5)
        self.assertFalse(ss.remove_event("ghost", 2))
        self.assertEqual(ss.total, 5)

    def test_remove_event_decreases_total(self):
        ss = SpaceSaver(4)
        ss.add("a", 9)
        ss.remove_event("a", 4)
        self.assertEqual((ss.query("a"), ss.total), (5, 5))

    def test_add_negative_routes_to_removal(self):
        ss = SpaceSaver(4)
        ss.add("a", 6)
        ss.add("a", -2)
        self.assertEqual(ss.query("a"), 4)

    def test_subtract_estimate_removes_inherited_error(self):
        ss = SpaceSaver(2)
        ss.add("a", 5)
        ss.add("b", 4)
        ss.add("c", 1)
        # c 报告 5，但真值只有 1：减去报告值会把它彻底抹掉
        ss.subtract_estimate("c")
        self.assertEqual(ss.query("c"), 0)

    def test_drop_frees_slot(self):
        ss = SpaceSaver(2)
        ss.add("a", 5)
        ss.add("b", 4)
        self.assertTrue(ss.drop("a"))
        ss.add("c", 1)
        self.assertIn("c", ss.keys())
        self.assertEqual(len(ss), 2)

    def test_clamped_delete_is_not_invertible(self):
        """夹到 0 的删除不是逆运算：再插回去会多出不该存在的计数。"""
        clamped = CountMinSketch(16, 2, seed=1)
        plain = CountMinSketch(16, 2, seed=1)
        clamped.add("x", 2)
        plain.add("x", 2)
        clamped.delete_event("x", 5, clamp=True)
        plain.delete_event("x", 5)
        for sketch in (clamped, plain):
            sketch.add("x", 5)
        self.assertEqual(clamped.query("x"), 5)
        self.assertEqual(plain.query("x"), 2)
        self.assertNotEqual(clamped.query("x"), plain.query("x"))

    def test_clamped_delete_keeps_row_totals_consistent(self):
        s = CountMinSketch(8, 1, seed=0)
        s.add("a", 2)
        s.delete_event("a", 5, clamp=True)
        self.assertEqual(s.row_totals[0], sum(s.rows[0]))
        self.assertEqual(s.query("a"), 0)

    def test_drop_unknown_key(self):
        self.assertFalse(SpaceSaver(2).drop("nothing"))

    def test_big_weights(self):
        ss = SpaceSaver(8)
        ss.add("hot", 10 ** 9)
        ss.add("cold", 1)
        self.assertEqual(ss.query("hot"), 10 ** 9)

    def test_bytes_used_positive(self):
        ss = SpaceSaver(16)
        for k in zipf_stream(500, 1.1, 60, seed=1):
            ss.add(k)
        self.assertGreater(ss.bytes_used(), 0)

    def test_repr_counts_monitored(self):
        ss = SpaceSaver(3)
        ss.add("a", 2)
        self.assertIn("monitored=1", repr(ss))


class TopKStructure(unittest.TestCase):
    def test_finds_the_heaviest_key(self):
        top = CmsTopK(width=512, depth=5, capacity=32, seed=1)
        stream = zipf_stream(4000, 1.2, 200, seed=2)
        for k in stream:
            top.add(k)
        exact = top_k_from_counts(exact_counts(stream), 1)[0][0]
        self.assertEqual(top.top_k(1)[0][0], exact)

    def test_candidates_bounded_by_capacity(self):
        top = CmsTopK(width=128, depth=3, capacity=16, seed=0)
        for k in uniform_stream(4000, universe=800, seed=3):
            top.add(k)
        self.assertLessEqual(len(top.candidates), 16)

    def test_empty_stream_returns_nothing(self):
        self.assertEqual(CmsTopK(64, 3, 8).top_k(10), [])

    def test_single_key(self):
        top = CmsTopK(64, 3, 8)
        for _ in range(25):
            top.add("solo")
        self.assertEqual(top.top_k(3), [("solo", 25)])

    def test_k_larger_than_candidate_set(self):
        top = CmsTopK(256, 4, 64, seed=5)
        for k in ["a", "b", "c"]:
            top.add(k, 3)
        self.assertEqual(len(top.top_k(20)), 3)

    def test_full_coverage_matches_exact(self):
        """候选容量覆盖整个键空间时，top-k 必须与精确基线完全一致。"""
        universe = 60
        top = CmsTopK(width=1024, depth=5, capacity=universe, seed=9)
        stream = zipf_stream(3000, 1.1, universe, seed=9)
        for k in stream:
            top.add(k)
        counts = exact_counts(stream)
        self.assertEqual([k for k, _ in top.top_k(10)],
                         [k for k, _ in top_k_from_counts(counts, 10)])

    def test_mean_mode_still_tracks_top(self):
        top = CmsTopK(width=256, depth=5, capacity=32, seed=2, mode="mean")
        stream = zipf_stream(2500, 1.3, 120, seed=4)
        for k in stream:
            top.add(k)
        self.assertEqual(top.mode, "mean")
        self.assertGreaterEqual(top.top_k(1)[0][1], 1)

    def test_estimate_uses_selected_mode(self):
        top = CmsTopK(width=128, depth=4, capacity=8, seed=1, mode="count_mean")
        for k in uniform_stream(900, universe=64, seed=1):
            top.add(k)
        self.assertAlmostEqual(top.estimate("10.0.0.2"),
                               top.sketch.query("10.0.0.2", "count_mean"))

    def test_zero_count_noop_in_topk(self):
        top = CmsTopK(64, 3, 8)
        top.add("a", 0)
        self.assertEqual(top.candidates, {})

    def test_bytes_used_covers_sketch(self):
        top = CmsTopK(width=512, depth=5, capacity=32, seed=0)
        self.assertGreaterEqual(top.bytes_used(), top.sketch.bytes_used())

    def test_lazy_heap_prunes_stale_entries(self):
        top = CmsTopK(width=64, depth=3, capacity=4, seed=1)
        for k in uniform_stream(5000, universe=400, seed=2):
            top.add(k)
        self.assertLessEqual(len(top._heap), 4 * top.capacity + 16)

    def test_repr(self):
        self.assertIn("capacity=8", repr(CmsTopK(64, 3, 8)))


class ExactBaselineHelpers(unittest.TestCase):
    def test_top_k_from_counts_order(self):
        counts = {"a": 5, "b": 9, "c": 1}
        self.assertEqual(top_k_from_counts(counts, 2), [("b", 9), ("a", 5)])

    def test_top_k_from_counts_ties_by_key(self):
        counts = {"z": 4, "a": 4}
        self.assertEqual(top_k_from_counts(counts, 2), [("a", 4), ("z", 4)])

    def test_exact_counts_sum(self):
        stream = zipf_stream(700, 1.1, 50, seed=3)
        self.assertEqual(sum(exact_counts(stream).values()), 700)

    def test_empty_stream_shape(self):
        self.assertEqual(zipf_stream(0, 1.0, 10, seed=0), [])
        self.assertEqual(exact_counts([]), {})

    def test_zipf_stream_length_and_determinism(self):
        a = zipf_stream(500, 1.05, 100, seed=77)
        b = zipf_stream(500, 1.05, 100, seed=77)
        self.assertEqual(len(a), 500)
        self.assertEqual(a, b)

    def test_zipf_stream_is_seed_sensitive(self):
        self.assertNotEqual(zipf_stream(200, 1.0, 50, seed=1),
                            zipf_stream(200, 1.0, 50, seed=2))

    def test_higher_skew_means_heavier_head(self):
        soft = exact_counts(zipf_stream(4000, 0.7, 300, seed=11))
        hard = exact_counts(zipf_stream(4000, 1.6, 300, seed=11))
        self.assertGreater(max(hard.values()), max(soft.values()))

    def test_zipf_weights_shape(self):
        w = zipf_weights(4, 1.0)
        self.assertEqual(w, [1.0, 0.5, 1 / 3.0, 0.25])

    def test_uniform_stream_universe_respected(self):
        stream = uniform_stream(400, universe=5, seed=1)
        self.assertEqual(len(stream), 400)
        self.assertEqual(len(exact_counts(stream)), 5)

    def test_ip_key_rejects_negative_rank(self):
        with self.assertRaises(ValueError):
            ip_key(-1)

    def test_ip_key_format(self):
        self.assertEqual(ip_key(1), "10.0.0.1")

    def test_streams_reject_bad_args(self):
        with self.assertRaises(ValueError):
            zipf_stream(10, 1.0, universe=0, seed=0)
        with self.assertRaises(ValueError):
            zipf_stream(-1, 1.0, universe=5, seed=0)


class DeletionSemantics(unittest.TestCase):
    def test_subtract_estimate_never_goes_negative(self):
        """减去 min 不会为负：min 是该键每一行的下界，所以伤害藏在别人的计数里。"""
        s = CountMinSketch(4, 1, seed=0)
        s.add("heavy", 100)
        s.delete_by_subtracting_estimate("heavy")
        self.assertEqual(s.query("heavy"), 0)
        self.assertGreaterEqual(s.query("heavy"), -1)

    def test_over_subtracting_does_go_negative(self):
        s = CountMinSketch(4, 1, seed=0)
        s.add("heavy", 100)
        s.delete_by_subtracting_estimate("heavy", scale=2.0)
        self.assertEqual(s.query("heavy"), -100)

    def test_subtract_estimate_breaks_colliding_keys(self):
        s = CountMinSketch(8, 1, seed=0)
        partner = same_col_partner(s, "heavy")
        s.add("heavy", 1000)
        s.add(partner, 1)
        s.delete_by_subtracting_estimate("heavy")
        self.assertLess(s.query(partner), 1)

    def test_delete_event_does_not_break_colliding_keys(self):
        s = CountMinSketch(8, 1, seed=0)
        partner = same_col_partner(s, "heavy")
        s.add("heavy", 1000)
        s.add(partner, 1)
        s.delete_event("heavy", 1000)
        self.assertEqual(s.query(partner), 1)

    def test_delete_event_zeroes_only_victim(self):
        s = CountMinSketch(64, 4, seed=1)
        s.add("a", 30)
        s.add("b", 12)
        s.delete_event("a", 30)
        self.assertEqual((s.query("a"), s.query("b")), (0, 12))

    def test_delete_event_clamps_row_values(self):
        s = CountMinSketch(16, 2, seed=1)
        s.add("x", 2)
        self.assertTrue(s.delete_event("x", 5, clamp=False))
        self.assertEqual(s.query("x"), -3)


if __name__ == "__main__":
    unittest.main()
