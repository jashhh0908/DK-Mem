"""Tests for the offline tau sweep (evaluator/sweep.py) on tiny hand-built runs."""
import unittest

from evaluator.loaders import JsonlResult
from evaluator.metrics import evaluate_run
from evaluator.sweep import BANNER, check_recorded_decisions, default_taus, derived_merge, sweep_run
from evaluator.validate import build_gold_index, validate_run
from tests.test_validate import SCHEMA_DIR, Base, entry, gold_rec, rec


class DerivedMergeRule(unittest.TestCase):
    def test_strict_greater_than_not_greater_equal(self):
        r = {"similarity_score": 0.85, "compatibility": "compatible"}
        self.assertFalse(derived_merge(r, 0.85))      # contract: strictly greater
        self.assertTrue(derived_merge(r, 0.84))

    def test_incompatible_and_underdetermined_never_merge(self):
        for c in ("incompatible", "underdetermined"):
            self.assertFalse(derived_merge({"similarity_score": 1.0, "compatibility": c}, 0.0))

    def test_no_gating_uses_similarity_only(self):
        self.assertTrue(derived_merge({"similarity_score": 0.9, "compatibility": None}, 0.5))
        self.assertFalse(derived_merge({"similarity_score": 0.4, "compatibility": None}, 0.5))


class SweepRuns(Base):
    def setUp(self):
        super().setUp()
        # 2 same, 2 different, 1 same left without any entries
        self.g = [gold_rec(1, "same", "name_variant"), gold_rec(2, "same", "name_variant"),
                  gold_rec(3, "different", "kinship"), gold_rec(4, "different", "kinship"),
                  gold_rec(5, "same", "name_variant")]
        self.gold = build_gold_index(JsonlResult(records=self.g))

    def report(self, records, name="run"):
        return validate_run(self.run_dir(records, name=name), self.gold, SCHEMA_DIR)

    def records(self, **over):
        g = self.g
        return [rec(g[0], similarity_score=0.95, decision="merge"),
                rec(g[1], similarity_score=0.60, decision="no_merge"),
                rec(g[2], similarity_score=0.90, compatibility="compatible", decision="merge"),
                rec(g[3], similarity_score=0.99, compatibility="incompatible", decision="no_merge")]

    def point(self, run, tau):
        return next(p for p in run["points"] if p["tau"] == tau)

    def test_hand_computed_points(self):
        run = sweep_run(self.report(self.records()), self.gold, [0.5, 0.85, 0.97])
        p = self.point(run, 0.5)     # both same merged, kinship-compatible merged, incompatible blocked
        self.assertEqual((p["fcr"]["numerator"], p["fcr"]["denominator"]), (1, 2))
        self.assertEqual((p["mcr"]["numerator"], p["mcr"]["denominator"]), (1, 3))   # only the no-entry pair
        p = self.point(run, 0.85)
        self.assertEqual((p["fcr"]["numerator"], p["mcr"]["numerator"]), (1, 2))     # 0.60 same pair now missed
        p = self.point(run, 0.97)
        self.assertEqual((p["fcr"]["numerator"], p["mcr"]["numerator"]), (0, 3))

    def test_no_entry_pair_is_constant_across_tau_and_floors_mcr(self):
        run = sweep_run(self.report(self.records()), self.gold, default_taus())
        self.assertEqual({p["constant_mcr_misses"] for p in run["points"]}, {1})
        self.assertTrue(all(p["mcr"]["numerator"] >= 1 for p in run["points"]))
        self.assertTrue(all(p["n_pairs_compared"] == 4 for p in run["points"]))

    def test_fcr_never_increases_and_mcr_never_decreases_with_tau(self):
        run = sweep_run(self.report(self.records()), self.gold, default_taus())
        f = [p["fcr"]["numerator"] for p in run["points"]]
        m = [p["mcr"]["numerator"] for p in run["points"]]
        self.assertEqual(f, sorted(f, reverse=True))
        self.assertEqual(m, sorted(m))

    def test_agrees_with_phase3_metrics_at_recorded_tau(self):
        rep = self.report(self.records())
        run = sweep_run(rep, self.gold, [0.85])
        res = evaluate_run(rep, self.gold)["overall"]
        p = self.point(run, 0.85)
        self.assertEqual((p["fcr"], p["mcr"]), (res["fcr"], res["mcr"]))
        self.assertEqual(run["recorded_decision_check"]["severity"], "OK")

    def test_metadata_and_banner(self):
        run = sweep_run(self.report(self.records()), self.gold, [0.85])
        self.assertEqual((run["run_id"], run["strategy"], run["seed"]), ("r1", "dk-mem-lexicon", 0))
        self.assertEqual(run["banner"], BANNER)
        p = self.point(run, 0.85)
        self.assertEqual((p["n_pairs_compared"], p["n_pairs_merged"]), (4, 2))

    def test_disagreement_is_reported_not_fixed_and_ties_are_separated(self):
        g = self.g
        recs = self.records()
        recs[0] = rec(g[0], similarity_score=0.85, decision="merge")                       # gate '>=' merged a tie
        recs[1] = rec(g[1], similarity_score=0.95, decision="no_merge", compatibility="compatible")   # non-tie mismatch
        chk = check_recorded_decisions(self.report(recs, ), self.gold)
        self.assertEqual(chk["severity"], "WARN")
        self.assertEqual((chk["disagree"], chk["disagree_at_tie_sim_equals_tau"], chk["disagree_not_at_tie"]), (2, 1, 1))
        self.assertEqual(chk["agree"] + chk["disagree"], chk["records_checked"])
        # the swept result follows the contract rule, not the recorded decision
        run = sweep_run(self.report(recs, "r2"), self.gold, [0.85])
        self.assertEqual(self.point(run, 0.85)["n_pairs_merged"], 2)   # g[1]@0.95 and g[2]@0.90 merge; the 0.85 tie does not


if __name__ == "__main__":
    unittest.main()
