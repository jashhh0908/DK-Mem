"""Tests for FCR/MCR using tiny hand-built gold + runs. Run: python -m unittest discover -s tests -t ."""
import unittest

from evaluator.loaders import JsonlResult
from evaluator.metrics import evaluate_run, fcr, mcr, pair_details, rate, seed_summary
from evaluator.validate import build_gold_index, validate_run
from tests.test_validate import MANIFEST, SCHEMA_DIR, Base, entry, gold_rec, rec


class RateTests(unittest.TestCase):
    def test_zero_denominator_is_na_not_zero(self):
        c = rate(0, 0)
        self.assertIsNone(c["rate"])
        self.assertEqual(c["display"], "n/a")
        self.assertFalse(c["indicative"])

    def test_indicative_flag_boundary(self):
        self.assertTrue(rate(1, 19)["indicative"])
        self.assertFalse(rate(1, 20)["indicative"])
        self.assertEqual(rate(0, 5)["rate"], 0.0)


class MetricTests(Base):
    """Gold: 3 same (kinship/name_variant/evidentiality-legacy) + 3 different."""

    def setUp(self):
        super().setUp()
        self.g = [gold_rec(1, "same", "name_variant"), gold_rec(2, "same", "name_variant"),
                  gold_rec(3, "same", "evidentiality"), gold_rec(4, "different", "kinship"),
                  gold_rec(5, "different", "kinship"), gold_rec(6, "different", "classifier_measure")]
        self.gold = build_gold_index(JsonlResult(records=self.g))

    def evaluate(self, records, name="run"):
        rep = validate_run(self.run_dir(records, name=name), self.gold, SCHEMA_DIR)
        return evaluate_run(rep, self.gold), rep

    def test_hand_computed_overall_and_per_class(self):
        g = self.g
        res, _ = self.evaluate([
            rec(g[0], decision="merge"),            # same, merged        -> hit
            rec(g[1], decision="no_merge"),         # same, not merged    -> MCR miss
            # g[2] (same, evidentiality): no records at all -> no_entry   -> MCR miss
            rec(g[3], decision="merge"),            # different, merged   -> FCR hit
            rec(g[4], decision="underdetermined_link"),  # not merged
            rec(g[5], decision="supersede"),        # different, merged (supersede counts)
        ])
        o = res["overall"]
        self.assertEqual((o["fcr"]["numerator"], o["fcr"]["denominator"]), (2, 3))
        self.assertEqual((o["mcr"]["numerator"], o["mcr"]["denominator"]), (2, 3))
        pc = res["per_class"]
        self.assertEqual((pc["kinship"]["fcr"]["numerator"], pc["kinship"]["fcr"]["denominator"]), (1, 2))
        self.assertEqual(pc["kinship"]["mcr"]["display"], "n/a")
        self.assertEqual((pc["name_variant"]["mcr"]["numerator"], pc["name_variant"]["mcr"]["denominator"]), (1, 2))
        self.assertEqual(pc["name_variant"]["fcr"]["display"], "n/a")
        self.assertEqual(pc["evidentiality"]["mcr"]["rate"], 1.0)
        self.assertEqual(pc["classifier_measure"]["fcr"]["rate"], 1.0)
        self.assertEqual(pc["evidentiality"]["scope"], "legacy")
        self.assertEqual(pc["kinship"]["scope"], "in_scope")
        # honorific_register has no gold pairs in this fixture -> n/a both ways, not absent
        self.assertEqual(pc["honorific_register"]["fcr"]["display"], "n/a")

    def test_unmerged_different_pairs_count_in_fcr_denominator(self):
        g = self.g
        res, _ = self.evaluate([rec(g[3], decision="merge")])   # others: no entry
        f = res["overall"]["fcr"]
        self.assertEqual((f["numerator"], f["denominator"]), (1, 3))

    def test_no_entry_and_not_compared_are_distinct_but_unmerged(self):
        g = self.g
        same_a = rec(g[1], ea=entry(g[1], "a"), eb=entry(g[1], "a", idx=1), rid="s1")
        same_b = rec(g[1], ea=entry(g[1], "b"), eb=entry(g[1], "b", idx=1), rid="s2")
        res, _ = self.evaluate([same_a, same_b])    # g[1]: entries on both sides, never compared
        rows = {r["eval_pair_id"]: r for r in res["_rows"]}
        self.assertEqual(rows[g[1]["eval_pair_id"]]["outcome"], "not_compared")
        self.assertEqual(rows[g[0]["eval_pair_id"]]["outcome"], "no_entry")
        self.assertEqual(res["no_entry_diagnostics"]["count"], 5)
        self.assertEqual(res["overall"]["mcr"]["numerator"], 3)       # all three same pairs unmerged
        self.assertEqual(res["overall"]["fcr"]["numerator"], 0)
        by_rel = res["overall"]["outcomes"]["same"]
        self.assertEqual((by_rel["not_compared"], by_rel["no_entry"]), (1, 2))

    def test_underdetermined_and_no_merge_are_not_merged(self):
        g = self.g
        res, _ = self.evaluate([rec(g[0], decision="underdetermined_link"), rec(g[3], decision="no_merge")])
        rows = {r["eval_pair_id"]: r for r in res["_rows"]}
        self.assertEqual(rows[g[0]["eval_pair_id"]]["outcome"], "compared_not_merged")
        self.assertFalse(rows[g[3]["eval_pair_id"]]["merged"])

    def test_multi_entry_any_cross_merge_means_merged(self):
        g = self.g[0]
        res, _ = self.evaluate([rec(g, rid="1", decision="no_merge"),
                                rec(g, eb=entry(g, "b", idx=1), rid="2", decision="merge")])
        row = next(r for r in res["_rows"] if r["eval_pair_id"] == g["eval_pair_id"])
        self.assertTrue(row["merged"])

    def test_same_utterance_merge_does_not_count(self):
        g = self.g[3]
        res, _ = self.evaluate([rec(g, ea=entry(g, "a"), eb=entry(g, "a", idx=1), decision="merge", rid="x")])
        row = next(r for r in res["_rows"] if r["eval_pair_id"] == g["eval_pair_id"])
        self.assertFalse(row["merged"])
        self.assertEqual(row["outcome"], "no_entry")   # utt_b has no entries

    def test_cross_episode_merge_does_not_count(self):
        g0, g3 = self.g[0], self.g[3]
        res, _ = self.evaluate([rec(g0, eb=entry(g3, "b"), decision="merge", rid="x")])
        self.assertEqual(res["overall"]["fcr"]["numerator"], 0)
        self.assertEqual(res["overall"]["mcr"]["numerator"], 3)

    def test_all_gold_pairs_present_in_rows_and_denominators_fixed(self):
        res, rep = self.evaluate([])
        self.assertEqual(len(res["_rows"]), 6)
        self.assertEqual(res["overall"]["fcr"]["denominator"], 3)
        self.assertEqual(res["overall"]["mcr"]["denominator"], 3)
        self.assertEqual((res["overall"]["fcr"]["rate"], res["overall"]["mcr"]["rate"]), (0.0, 1.0))

    def test_scope_subsets_partition_the_gold(self):
        g = self.g
        res, _ = self.evaluate([rec(g[3], decision="merge"), rec(g[5], decision="merge")])
        s = res["scope_subsets"]
        self.assertEqual(s["in_scope"]["fcr"]["denominator"] + s["legacy"]["fcr"]["denominator"], 3)
        self.assertEqual(s["in_scope"]["fcr"]["numerator"], 1)
        self.assertEqual(s["legacy"]["fcr"]["numerator"], 1)

    def test_seed_summary(self):
        g = self.g
        r1, _ = self.evaluate([rec(g[3], decision="merge")], name="a")
        r2, _ = self.evaluate([], name="b")
        s = seed_summary([r1, r2])
        self.assertEqual(s["fcr"]["max"], 1 / 3)
        self.assertEqual(s["fcr"]["min"], 0.0)


if __name__ == "__main__":
    unittest.main()
