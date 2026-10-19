"""eval/tier2/verify_entity_inputs.py: the evaluator-side re-verification of the runs behind the hash-pinned entity report.

The verifier must (1) pass the stored runs, (2) FAIL a run whose embedder revision or judge log does not match its manifest, (3) catch a
report that analysed other runs or recorded another contract outcome, and (4) never overwrite its output. It does not make
``entity_report.py`` itself refuse bad runs; these tests only establish what the verifier detects.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).parent.parent
EVAL = REPO / "eval"
for p in (str(REPO), str(EVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tier2 import verify_entity_inputs as V  # noqa: E402

ENTITY = EVAL / "tier2_entity_runs"
SMOKE_RUNS = ENTITY / "cd7b_smoke_v10_20261014" / "runs"
SMOKE_EVAL = REPO / "validation" / "tier2_entity" / "kernel_cd7b_smoke_v10.evaluator"
FULL_REPORT = ENTITY / "TIER2_FULL_entity_report_v1.json"
FULL = {"v1.0": (ENTITY / "cd7b_full_v10_20261014" / "runs", REPO / "validation" / "tier2_entity" / "kernel_cd7b_full_v10.evaluator"),
        "mirror": (ENTITY / "cd7b_full_mirror_20261014" / "runs", REPO / "validation" / "tier2_entity" / "kernel_cd7b_full_mirror.evaluator")}


class TestCrossCheck(unittest.TestCase):
    def report(self, runs):
        return {"arms": {"v1.0": {"runs": {rid: {"contract_ok": ok} for rid, ok in runs.items()}}}}

    def test_consistent_report_has_no_problems(self):
        self.assertEqual(V.cross_check(self.report({"r1": True, "r2": True}), {"v1.0": {"r1": {"ok": True}, "r2": {"ok": True}}}), [])

    def test_different_contract_outcome_is_reported(self):
        problems = V.cross_check(self.report({"r1": True}), {"v1.0": {"r1": {"ok": False}}})
        self.assertTrue(problems and "contract_ok" in problems[0])

    def test_report_that_analysed_other_runs_is_reported(self):
        self.assertTrue(V.cross_check(self.report({"r1": True, "extra": True}), {"v1.0": {"r1": {"ok": True}}}))
        self.assertTrue(V.cross_check(self.report({"r1": True}), {"v1.0": {"r1": {"ok": True}, "r2": {"ok": True}}}))

    def test_arm_missing_from_the_report_is_reported(self):
        self.assertTrue(V.cross_check({"arms": {}}, {"mirror": {"r": {"ok": True}}}))


@unittest.skipUnless(SMOKE_RUNS.is_dir() and (SMOKE_EVAL / "qa_id_map.json").exists(), "stored smoke runs or evaluator inputs absent (gitignored)")
class TestCliOnSmokeRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.runs = self.tmp / "runs"
        shutil.copytree(SMOKE_RUNS, self.runs)

    def run_cli(self, name="out.json", report=None):
        args = ["--runs", str(self.runs), "--arm", "v1.0", "--evaluator", str(SMOKE_EVAL), "--out", str(self.tmp / name)]
        if report:
            args += ["--report", str(report)]
        rc = V.main(args)
        return rc, json.loads((self.tmp / name).read_text(encoding="utf-8"))

    def edit_manifest(self, run, **edits):
        f = self.runs / run / "run_manifest.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        m.update(edits)
        f.write_text(json.dumps(m), encoding="utf-8")

    def test_unmodified_runs_pass(self):
        rc, out = self.run_cli()
        self.assertEqual(rc, 0, out["failed_runs"])
        self.assertTrue(out["ok"])
        self.assertEqual({v["embedder"] for v in out["arms"]["v1.0"].values()}, {"ok"})
        self.assertIn("verified", {(v["judge_log"] or {}).get("status") for v in out["arms"]["v1.0"].values()})

    def test_a_wrong_embedder_revision_makes_the_verification_fail(self):
        run = "t2cd7b-smoke-v10-dk-mem-lexicon-D_seed0"
        self.edit_manifest(run, similarity_model_revision="0" * 40)
        rc, out = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertFalse(out["ok"])
        self.assertTrue(any(run in f and "embedder_revision_wrong_revision" in f for f in out["failed_runs"]), out["failed_runs"])

    def test_a_judge_log_that_does_not_match_the_manifest_makes_the_verification_fail(self):
        run = "t2cd7b-smoke-v10-dk-mem-lexicon-C_seed0"
        log = self.runs / f"{run}.judge_log.jsonl"
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows[0]["raw_output"], rows[0]["status"], rows[0]["proposal"] = '{"decision": "different"}', "ok", "keep_both"
        log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        rc, out = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertTrue(any(run in f and "judge_log_row_status_mismatch" in f for f in out["failed_runs"]), out["failed_runs"])

    def test_the_output_is_never_overwritten(self):
        self.run_cli("first.json")
        with self.assertRaises(SystemExit):
            self.run_cli("first.json")


@unittest.skipUnless(FULL_REPORT.exists() and all(r.is_dir() and (e / "qa_id_map.json").exists() for r, e in FULL.values()),
                     "the stored full entity report, runs or evaluator inputs are absent (gitignored)")
class TestStoredEntityReport(unittest.TestCase):
    def test_the_runs_behind_the_stored_report_verify_and_match_it(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        args = []
        for arm, (runs, ev) in FULL.items():
            args += ["--runs", str(runs), "--arm", arm, "--evaluator", str(ev)]
        rc = V.main([*args, "--report", str(FULL_REPORT), "--out", str(tmp / "v.json")])
        out = json.loads((tmp / "v.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 0, (out["failed_runs"], out["report_cross_check"]))
        self.assertEqual({arm: len(r) for arm, r in out["arms"].items()}, {"v1.0": 4, "mirror": 4})
        for arm, runs in out["arms"].items():
            for rid, v in runs.items():
                self.assertEqual(v["embedder"], "ok", rid)
                if v["pipeline_config"] == "C":
                    self.assertEqual(v["judge_log"]["status"], "verified", rid)
        self.assertEqual(out["report_cross_check"], [])


if __name__ == "__main__":
    unittest.main()
