"""eval/tier2/entity_report_verified.py: the entity report produced only from independently verified runs, without touching pinned files.

Established here (not assumed): valid runs give a report byte-identical to the pinned code alone; a run whose embedder revision or judge log
contradicts its manifest stops generation (strict) or is excluded by the pinned report's own contract rule (--allow-excluded); the pinned
contract is always restored; outputs are never overwritten. The pinned entry point ``tier2.entity_report`` remains directly callable and
is NOT protected by this module; one test states that explicitly.
"""

import hashlib
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

from tier2 import contract, entity_report  # noqa: E402
from tier2 import entity_report_verified as EV  # noqa: E402

TIER2 = EVAL / "tier2"
ENTITY = EVAL / "tier2_entity_runs"
SMOKE_RUNS = ENTITY / "cd7b_smoke_v10_20261014" / "runs"
SMOKE_EVAL = REPO / "validation" / "tier2_entity" / "kernel_cd7b_smoke_v10.evaluator"
FULL_REPORT = ENTITY / "TIER2_FULL_entity_report_v1.json"
FULL = {"v1.0": (ENTITY / "cd7b_full_v10_20261014" / "runs", REPO / "validation" / "tier2_entity" / "kernel_cd7b_full_v10.evaluator"),
        "mirror": (ENTITY / "cd7b_full_mirror_20261014" / "runs", REPO / "validation" / "tier2_entity" / "kernel_cd7b_full_mirror.evaluator")}
D_RUN = "t2cd7b-smoke-v10-dk-mem-lexicon-D_seed0"
C_RUN = "t2cd7b-smoke-v10-dk-mem-lexicon-C_seed0"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TestPinnedFilesAreUntouched(unittest.TestCase):
    def test_the_four_analysis_plan_pins_still_match(self):
        fz = json.loads((TIER2 / "ANALYSIS_PLAN_FREEZE_v1.json").read_text(encoding="utf-8"))
        for rel, want in fz["files_sha256"].items():
            self.assertEqual(sha(TIER2 / rel), want, rel)


@unittest.skipUnless(SMOKE_RUNS.is_dir() and (SMOKE_EVAL / "qa_id_map.json").exists(), "stored smoke runs or evaluator inputs absent (gitignored)")
class TestOnSmokeRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.runs = self.tmp / "runs"
        shutil.copytree(SMOKE_RUNS, self.runs)

    def args(self, out, *extra):
        return ["--runs", str(self.runs), "--arm", "v1.0", "--evaluator", str(SMOKE_EVAL), "--out", str(self.tmp / out), *extra]

    def tamper_embedder(self):
        f = self.runs / D_RUN / "run_manifest.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        m["similarity_model_revision"] = "0" * 40
        f.write_text(json.dumps(m), encoding="utf-8")

    def tamper_judge_log(self):
        f = self.runs / f"{C_RUN}.judge_log.jsonl"
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows[0]["raw_output"], rows[0]["status"], rows[0]["proposal"] = '{"decision": "different"}', "ok", "keep_both"
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def test_valid_runs_give_the_same_bytes_as_the_pinned_code_and_a_passing_sidecar(self):
        self.assertEqual(EV.main(self.args("v.json")), 0)
        self.assertEqual(entity_report.main(["--runs", str(self.runs), "--arm", "v1.0", "--evaluator", str(SMOKE_EVAL), "--out", str(self.tmp / "p.json")]), 0)
        self.assertEqual((self.tmp / "v.json").read_bytes(), (self.tmp / "p.json").read_bytes())
        side = json.loads((self.tmp / "v.json.verification.json").read_text(encoding="utf-8"))
        self.assertTrue(side["ok"])
        self.assertEqual(side["report_sha256"], sha(self.tmp / "v.json"))
        self.assertEqual(side["entity_report_py_sha256"], sha(TIER2 / "entity_report.py"))
        self.assertEqual((side["failed_runs"], side["report_cross_check"]), ([], []))

    def test_a_wrong_embedder_revision_stops_generation_and_writes_nothing(self):
        self.tamper_embedder()
        self.assertEqual(EV.main(self.args("v.json")), 1)
        self.assertFalse((self.tmp / "v.json").exists())
        self.assertFalse((self.tmp / "v.json.verification.json").exists())

    def test_a_judge_log_that_contradicts_its_manifest_stops_generation_and_writes_nothing(self):
        self.tamper_judge_log()
        self.assertEqual(EV.main(self.args("v.json")), 1)
        self.assertFalse((self.tmp / "v.json").exists())

    def test_allow_excluded_makes_the_pinned_report_itself_exclude_the_bad_run(self):
        self.tamper_embedder()
        self.assertEqual(EV.main(self.args("v.json", "--allow-excluded")), 1)       # still a failing exit status
        report = json.loads((self.tmp / "v.json").read_text(encoding="utf-8"))
        run = report["arms"]["v1.0"]["runs"][D_RUN]
        self.assertFalse(run["contract_ok"])
        self.assertIn("embedder_revision_wrong_revision", {e["code"] for e in run["contract_errors"]})
        self.assertIn("excluded", run)                                              # the pinned code withheld every metric for it
        self.assertNotIn("pairs", run)
        self.assertIn(D_RUN, report["arms"]["v1.0"]["failures"])
        good = report["arms"]["v1.0"]["runs"][C_RUN]
        self.assertTrue(good["contract_ok"])                                        # the other runs are unaffected
        side = json.loads((self.tmp / "v.json.verification.json").read_text(encoding="utf-8"))
        self.assertFalse(side["ok"])
        self.assertEqual(side["report_cross_check"], [])                            # the report and the verification agree on the exclusion

    def test_the_pinned_contract_is_restored_after_success_and_after_failure(self):
        original = contract.validate_run
        EV.main(self.args("ok.json"))
        self.assertIs(contract.validate_run, original)
        self.tamper_embedder()
        EV.main(self.args("bad.json", "--allow-excluded"))
        self.assertIs(contract.validate_run, original)

    def test_outputs_are_never_overwritten(self):
        EV.main(self.args("v.json"))
        with self.assertRaises(SystemExit):
            EV.main(self.args("v.json"))
        (self.tmp / "w.json.verification.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            EV.main(self.args("w.json"))

    def test_the_pinned_entry_point_alone_does_not_notice_a_wrong_embedder_revision(self):
        # documents the residual limitation: calling tier2.entity_report directly bypasses the verification
        self.tamper_embedder()
        self.assertEqual(entity_report.main(["--runs", str(self.runs), "--arm", "v1.0", "--evaluator", str(SMOKE_EVAL),
                                             "--out", str(self.tmp / "direct.json")]), 0)
        report = json.loads((self.tmp / "direct.json").read_text(encoding="utf-8"))
        self.assertTrue(report["arms"]["v1.0"]["runs"][D_RUN]["contract_ok"])


@unittest.skipUnless(FULL_REPORT.exists() and all(r.is_dir() and (e / "qa_id_map.json").exists() for r, e in FULL.values()),
                     "the stored full entity report, runs or evaluator inputs are absent (gitignored)")
class TestReproducesTheStoredReport(unittest.TestCase):
    def test_the_verified_entry_point_regenerates_the_stored_report_byte_for_byte(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        args = []
        for arm, (runs, ev) in FULL.items():
            args += ["--runs", str(runs), "--arm", arm, "--evaluator", str(ev)]
        self.assertEqual(EV.main([*args, "--out", str(tmp / "r.json")]), 0)
        self.assertEqual((tmp / "r.json").read_bytes(), FULL_REPORT.read_bytes())
        side = json.loads((tmp / "r.json.verification.json").read_text(encoding="utf-8"))
        self.assertTrue(side["ok"])
        self.assertEqual({a: len(r) for a, r in side["verification"].items()}, {"v1.0": 4, "mirror": 4})


if __name__ == "__main__":
    unittest.main()
