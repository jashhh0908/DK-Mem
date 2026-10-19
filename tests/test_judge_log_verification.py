"""Independent verification of Config C judge logs against their manifests (eval/tier2/judge_provenance.verify_log and
validate_run_verified). A log's own status / proposal / counts are never trusted: each row's prompt identity is compared with the
manifest and every raw_output is re-parsed with the strict parser of the declared judge family.

Synthetic cases always run; cases on the stored artifacts skip with a reason when the (gitignored) runs are absent.

Run: python -m pytest tests/test_judge_log_verification.py
"""

import glob
import json
import os
import re
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

from tier2 import judge_provenance as P  # noqa: E402
from tier2.frozen_judge import (  # noqa: E402
    ENTITY_JUDGE_V2_PROMPT_ID, ENTITY_JUDGE_V2_PROMPT_SHA256, FROZEN_JUDGE_PROMPT_ID, FROZEN_JUDGE_PROMPT_SHA256,
    LEGACY_HARNESS_JUDGE_PROMPT_SHA256,
)

TIER2 = EVAL / "tier2"
ENTITY_RUNS = EVAL / "tier2_entity_runs"
FULL_RUNS = EVAL / "tier2_full_runs"
SMOKE_RUNS = EVAL / "tier2_smoke_runs"


def manifest(sha=FROZEN_JUDGE_PROMPT_SHA256, pid=FROZEN_JUDGE_PROMPT_ID, failures=0):
    return {"judge_prompt_id": pid, "judge_prompt_sha256": sha, "judge_parse_failures": failures,
            "generation_run_info": {"generation_params": {"max_new_tokens": 128}}}


def good_row(decision="keep_both", *, pid=FROZEN_JUDGE_PROMPT_ID, sha=FROZEN_JUDGE_PROMPT_SHA256):
    return {"raw_output": json.dumps({"decision": decision, "reason": "x"}), "prompt_id": pid, "prompt_sha256": sha,
            "status": "ok", "proposal": decision}


class TestVerifyLogSynthetic(unittest.TestCase):
    def test_parser_per_family_is_explicit(self):
        self.assertIs(P.strict_parser(P.FAMILY_TIER1), P.strict_parser(P.FAMILY_IDENTITY))
        self.assertEqual(P.strict_parser(P.FAMILY_TIER1).__name__, "parse_judge_output")
        self.assertIsNone(P.strict_parser(P.FAMILY_LEGACY))          # its parser is not in this repository: never guessed
        self.assertIsNone(P.strict_parser(None))

    def test_current_logs_verify(self):
        v = P.verify_log(manifest(), [good_row("merge"), good_row("keep_both")])
        self.assertEqual((v["status"], v["n_reparsed_ok"], v["n_identity_verified"], v["problems"]), ("verified", 2, 2, {}))

    def test_identity_judge_with_either_key_order_verifies(self):
        m = manifest(ENTITY_JUDGE_V2_PROMPT_SHA256, ENTITY_JUDGE_V2_PROMPT_ID)
        row = {"raw_output": '{"reason": "A is about X; B is about X; same entity.", "decision": "merge"}',
               "prompt_id": ENTITY_JUDGE_V2_PROMPT_ID, "prompt_sha256": ENTITY_JUDGE_V2_PROMPT_SHA256, "status": "ok", "proposal": "merge"}
        self.assertEqual(P.verify_log(m, [row])["status"], "verified")

    def test_mislabelled_vocabulary_under_a_valid_frozen_manifest_fails(self):
        # the invalidated external judge's answers, logged as "ok" by rows that carry the frozen prompt's identity
        rows = [{"raw_output": '{"decision": "different"}', "prompt_id": FROZEN_JUDGE_PROMPT_ID, "prompt_sha256": FROZEN_JUDGE_PROMPT_SHA256,
                 "status": "ok", "proposal": "keep_both"} for _ in range(5)]
        rows += [{**rows[0], "raw_output": '{"decision": "same"}', "proposal": "merge"} for _ in range(3)]
        v = P.verify_log(manifest(), rows)
        self.assertEqual(v["status"], "failed")
        self.assertEqual(v["problems"], {"row_status_mismatch": 8})
        self.assertEqual((v["n_reparsed_ok"], v["n_reparsed_rejected"]), (0, 8))
        text = P.describe(manifest(), rows)["classification"]
        self.assertIn("NOT VERIFIED", text)
        self.assertNotIn("VERIFIED: every row", text)
        self.assertIsNone(re.search(r"(\d+)/\1 raw outputs accepted", text))     # never "8/8 parsed" when 0/8 re-parse

    def test_row_identity_inconsistent_with_the_manifest(self):
        wrong_sha = {**good_row(), "prompt_sha256": "0" * 64}
        wrong_id = {**good_row(), "prompt_id": "merge_judge_entity_t2_v2"}
        v = P.verify_log(manifest(), [good_row(), wrong_sha, wrong_id])
        self.assertEqual(v["status"], "failed")
        self.assertEqual(v["problems"], {"row_prompt_sha_mismatch": 1, "row_prompt_id_mismatch": 1})
        self.assertEqual(v["first_bad_row"], {"row_prompt_sha_mismatch": 1, "row_prompt_id_mismatch": 2})

    def test_strict_parse_disagreeing_with_logged_status_or_proposal(self):
        cases = {
            "row_proposal_mismatch": {**good_row("merge"), "proposal": "keep_both"},
            "row_status_mismatch": {**good_row("merge"), "status": "parse_error", "proposal": None},
        }
        invalid_but_ok = {**good_row(), "raw_output": "keep_both", "status": "ok"}                          # not JSON, logged ok
        invalid_with_proposal = {**good_row(), "raw_output": "keep_both", "status": "parse_error"}          # rejected, but a proposal kept
        for code, row in cases.items():
            self.assertIn(code, P.verify_log(manifest(), [row])["problems"], code)
        self.assertIn("row_status_mismatch", P.verify_log(manifest(), [invalid_but_ok])["problems"])
        self.assertIn("row_proposal_mismatch", P.verify_log(manifest(), [invalid_with_proposal])["problems"])
        # a correctly logged failure is consistent
        ok_failure = {"raw_output": "not json", "prompt_id": FROZEN_JUDGE_PROMPT_ID, "prompt_sha256": FROZEN_JUDGE_PROMPT_SHA256,
                      "status": "parse_error", "proposal": None}
        v = P.verify_log(manifest(failures=1), [ok_failure])
        self.assertEqual((v["status"], v["n_reparsed_rejected"]), ("verified", 1))

    def test_current_manifest_without_row_hashes_is_reported_not_downgraded(self):
        rows = [{k: v for k, v in good_row().items() if k not in ("prompt_id", "prompt_sha256")}]
        v = P.verify_log(manifest(), rows)
        self.assertEqual(v["status"], "failed")                       # NOT legacy_unverifiable
        self.assertEqual(v["problems"], {"row_identity_missing": 1})

    def test_declared_legacy_run_is_unverifiable_never_verified(self):
        rows = [{"raw_output": '{"decision": "different"}', "status": "ok", "proposal": "keep_both"} for _ in range(4)]
        m = manifest(LEGACY_HARNESS_JUDGE_PROMPT_SHA256)
        v = P.verify_log(m, rows)
        self.assertEqual(v["status"], "legacy_unverifiable")
        self.assertIsNone(v["n_reparsed_ok"])                         # no parser: nothing is claimed re-parsed
        text = P.describe(m, rows)["classification"]
        self.assertIn("LEGACY_UNVERIFIABLE", text)
        self.assertNotIn("VERIFIED: every row", text)

    def test_declared_legacy_run_with_current_row_hashes_is_an_inconsistency(self):
        rows = [good_row()]                                           # rows claim the frozen prompt, manifest says historical
        v = P.verify_log(manifest(LEGACY_HARNESS_JUDGE_PROMPT_SHA256), rows)
        self.assertEqual(v["status"], "failed")
        self.assertEqual(v["problems"], {"row_prompt_sha_mismatch": 1})

    def test_unrecognised_prompt_is_not_verified(self):
        self.assertEqual(P.verify_log(manifest("a" * 64), [good_row()])["status"], "unrecognised")


def _current_logs():
    out = []
    for f in glob.glob(str(EVAL / "tier2_*runs" / "**" / "*.judge_log.jsonl"), recursive=True):
        if "/src/" in f.replace("\\", "/"):
            continue
        run = os.path.basename(f)[: -len(".judge_log.jsonl")]
        out.append((run, Path(f), Path(f).parent / run / "run_manifest.json"))
    return sorted(out)


def _load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


V2_FULL = "v2_full_20261017"          # the v2 full run (6 judge logs); counted separately so neither set can switch the other's check off
CURRENT = [(r, f, m) for r, f, m in _current_logs() if not r.startswith("t2full") and V2_FULL not in str(f)]
V2_FULL_LOGS = [(r, f, m) for r, f, m in _current_logs() if V2_FULL in str(f)]
LEGACY = [(r, f, m) for r, f, m in _current_logs() if r.startswith("t2full")]


class _VerifiesLogs:
    def check(self, logs):
        for run, log, man in logs:
            with self.subTest(run=run):
                v = P.verify_log(json.loads(man.read_text(encoding="utf-8")), _load(log))
                self.assertEqual(v["status"], "verified", v["problems"])
                self.assertEqual(v["n_reparsed_ok"], v["n_rows"])
                self.assertEqual(v["n_identity_verified"], v["n_rows"])
                self.assertEqual(v["n_logged_agree_with_reparse"], v["n_rows"])
                self.assertEqual(v["problems"], {})


@unittest.skipUnless(len(CURRENT) == 10, "the 10 stored current judge logs are not all present (gitignored run outputs)")
class TestStoredCurrentLogs(_VerifiesLogs, unittest.TestCase):
    def test_every_current_log_verifies_independently(self):
        self.check(CURRENT)


@unittest.skipUnless(len(V2_FULL_LOGS) == 6, "the 6 v2 full-run judge logs are not all present (gitignored run outputs)")
class TestStoredV2FullLogs(_VerifiesLogs, unittest.TestCase):
    def test_every_v2_full_run_log_verifies_independently(self):
        self.check(V2_FULL_LOGS)


@unittest.skipUnless(len(LEGACY) == 2 and (FULL_RUNS / "t2_passthrough.jsonl").exists(), "the two historical t2full runs are not present")
class TestStoredLegacyLogs(unittest.TestCase):
    def test_legacy_runs_remain_loadable_with_an_explicit_warning(self):
        for run, log, man in LEGACY:
            with self.subTest(run=run):
                rep = P.validate_run_verified(man.parent, TIER2 / "data", passthrough_file=FULL_RUNS / "t2_passthrough.jsonl",
                                              frozen_file=TIER2 / "FROZEN_v1.0.json")
                self.assertTrue(rep["ok"], rep["errors"][:3])
                codes = [w["code"] for w in rep["warnings"]]
                self.assertIn("legacy_unverifiable", codes)
                self.assertIn("historical_harness_judge", codes)
                self.assertEqual(rep["judge_log_verification"]["status"], "legacy_unverifiable")
                self.assertNotEqual(rep["judge_log_verification"]["status"], "verified")


SMOKE_RUN = ENTITY_RUNS / "cd7b_smoke_v10_20261014" / "runs" / "t2cd7b-smoke-v10-store-surface-only-C_seed0"
SMOKE_EVAL = REPO / "validation" / "tier2_entity" / "kernel_cd7b_smoke_v10.evaluator"


@unittest.skipUnless(SMOKE_RUN.is_dir() and (SMOKE_EVAL / "qa_id_map.json").exists(), "stored smoke run or its evaluator inputs are not present")
class TestWrapperOnStoredRunCopy(unittest.TestCase):
    def copy(self):
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, ignore_errors=True)
        for p in SMOKE_RUN.parent.iterdir():
            if p.name == SMOKE_RUN.name or p.name.startswith(SMOKE_RUN.name + "."):
                (shutil.copytree if p.is_dir() else shutil.copy)(p, t / p.name)
        return t

    def run_wrapper(self, t):
        return P.validate_run_verified(t / SMOKE_RUN.name, TIER2 / "data", passthrough_file=SMOKE_EVAL / "t2_passthrough_used.jsonl",
                                       qa_id_map=SMOKE_EVAL / "qa_id_map.json", frozen_file=TIER2 / "FROZEN_v1.0.json")

    def test_unmodified_copy_validates_and_is_verified(self):
        rep = self.run_wrapper(self.copy())
        self.assertTrue(rep["ok"], rep["errors"][:3])
        self.assertEqual(rep["judge_log_verification"]["status"], "verified")

    def test_a_log_edited_to_a_wrong_vocabulary_is_rejected_even_with_a_valid_manifest(self):
        t = self.copy()
        f = t / f"{SMOKE_RUN.name}.judge_log.jsonl"
        rows = _load(f)
        for r in rows[:3]:
            r["raw_output"], r["status"], r["proposal"] = '{"decision": "different"}', "ok", "keep_both"
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        rep = self.run_wrapper(t)
        self.assertFalse(rep["ok"])
        self.assertIn("judge_log_row_status_mismatch", {e["code"] for e in rep["errors"]})

    def test_a_row_naming_another_prompt_is_rejected(self):
        t = self.copy()
        f = t / f"{SMOKE_RUN.name}.judge_log.jsonl"
        rows = _load(f)
        rows[0]["prompt_sha256"] = "0" * 64
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        rep = self.run_wrapper(t)
        self.assertIn("judge_log_row_prompt_sha_mismatch", {e["code"] for e in rep["errors"]})

    def test_a_manifest_parse_failure_count_that_the_re_parse_does_not_support_is_rejected(self):
        t = self.copy()
        f = t / SMOKE_RUN.name / "run_manifest.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        m["judge_parse_failures"] = 7
        f.write_text(json.dumps(m), encoding="utf-8")
        rep = self.run_wrapper(t)
        self.assertIn("judge_log_parse_failure_count", {e["code"] for e in rep["errors"]})


if __name__ == "__main__":
    unittest.main()
