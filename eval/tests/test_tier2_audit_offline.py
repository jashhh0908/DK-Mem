"""Offline Tier-2 audit / D sweep: synthetic semantics tests + a read-only check on the stored full-run artifacts when present."""
import json
import tempfile
import unittest
from pathlib import Path

from tier2 import audit_offline as A
from tier2 import generate_tier2 as G

RUNS = G.EVAL / "tier2_full_runs"
HAVE = RUNS.exists() and (RUNS / "MANIFEST.json").exists()
HAVE_KEY = (G.EVAL / "teamB_answer_key.jsonl").is_file()   # the evaluator-side Tier-1 key is local-only and never committed


def st(rel, cls, sim, compat=None):
    return {"id": "x", "relation": rel, "class": cls, "language": "hi", "distance": 1, "record": {"similarity_score": sim, "compatibility": compat}}


class SweepSemantics(unittest.TestCase):
    def test_strict_greater_than(self):
        states = [st("different", "k", 0.9), st("same", "n", 0.9)]
        c = A.counts(A.at_tau(states, 0.9))                       # score == tau -> NOT merged
        self.assertEqual((c["fcr_n"], c["mcr_n"]), (0, 1))
        c = A.counts(A.at_tau(states, 0.8999))
        self.assertEqual((c["fcr_n"], c["mcr_n"]), (1, 0))

    def test_gate_only_removes_merges_and_blocks_both_categories(self):
        base = [st("different", "k", 0.8), st("different", "k", 0.8), st("same", "n", 0.7)]
        gated = [st("different", "k", 0.8, "incompatible"), st("different", "k", 0.8, "underdetermined"), st("same", "n", 0.7, "compatible")]
        for tau in (0.0, 0.5, 0.75, 0.9):
            o, g = A.counts(A.at_tau(base, tau)), A.counts(A.at_tau(gated, tau))
            self.assertLessEqual(g["fcr_n"], o["fcr_n"])
            self.assertGreaterEqual(g["mcr_n"], o["mcr_n"])
        self.assertEqual(A.counts(A.at_tau(gated, 0.5))["fcr_n"], 0)

    def test_empty_relation_class_does_not_divide_by_zero(self):
        c = A.counts(A.at_tau([st("different", "kinship", 0.8)], 0.5))
        self.assertIsNone(c["mcr"])

    def test_frontier_runs_on_synthetic_states(self):
        states = [st("different", "k", 0.9), st("different", "k", 0.6), st("same", "n", 0.7), st("same", "n", 0.5)]
        run = A.sweep_run(states, "r", "off", [0.4, 0.65, 0.95], 20, {})
        self.assertEqual([(p["fcr_n"], p["mcr_n"]) for p in run["pareto_exact"]], [(0, 2), (1, 1), (2, 0)])
        self.assertEqual(len(run["points"]), 3)


@unittest.skipUnless(HAVE and HAVE_KEY, "full-run artifacts or the local-only Tier-1 key (eval/teamB_answer_key.jsonl) not present")
class StoredRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = A.build(RUNS, n_boot=50)

    def test_all_audit_checks_pass(self):
        for n, c in self.rep["audit"]["configs"].items():
            self.assertTrue(c["all_audit_checks_pass"], (n, [k for k, v in c.items() if v is False]))
        self.assertEqual(self.rep["audit"]["raw_artifacts_byte_identical_to_download_manifest"]["mismatches"], [])
        self.assertTrue(self.rep["audit"]["raw_similarity_identical_across_all_four_runs"])

    def test_c_is_degenerate(self):
        for d in self.rep["audit"]["c_judge_behaviour"].values():
            self.assertEqual((d["judge_calls"], d["pairs_judged_same"], d["merge_decisions_in_pairwise_records"]), (3105, 0, 0))
        self.assertIn("DEGENERATE", self.rep["audit"]["c_judge_classification"])

    def test_false_merges_and_gate(self):
        fm = self.rep["false_merges_tau_0.90"]
        self.assertEqual(len(fm), 5)
        self.assertTrue(all(f["gate_verdict"] == "compatible" and f["language"] == "de" and f["raw_cosine"] > 0.9 for f in fm))
        s = self.rep["gate"]["incompatible_summary"]
        self.assertEqual((s["incompatible_total"], s["effective_vetoes_at_0.90"]), (37, 0))

    def test_sweep_matches_recorded_run_at_tau_090(self):
        S = self.rep["sweep"]
        for key in ("off", "lexicon"):
            p = {q["tau"]: q for q in S[key]["points"]}[0.9]
            self.assertEqual((p["fcr"]["n"], p["mcr"]["n"]), (5, 27))

    def test_refuses_overwrite_and_writes_nothing_into_runs(self):
        before = sorted(p.name for p in RUNS.rglob("*"))
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "rep"
            self.assertEqual(A.main(["--runs-dir", str(RUNS), "--out", str(out)]), 0)
            with self.assertRaises(SystemExit):
                A.main(["--runs-dir", str(RUNS), "--out", str(out)])
        self.assertEqual(before, sorted(p.name for p in RUNS.rglob("*")))


if __name__ == "__main__":
    unittest.main()
