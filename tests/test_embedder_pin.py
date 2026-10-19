"""Embedder-revision pin: a manifest whose recorded bge-m3 revision differs from the registered one must be REJECTED, not just reported.

``eval/tier2/embedder_pin.py`` holds the registered revision and the check; ``judge_provenance.validate_run_verified`` (used by
``score_v2.score_run``) turns a failed check into a contract error, which excludes the run from scoring. The vendored
``harness.similarity.is_real_bge_m3_run`` is deliberately untouched (byte-pinned); one test documents why that is not enough.

Synthetic cases always run; the stored-run cases skip with a reason when the gitignored run outputs are absent.
"""

import glob
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).parent.parent
EVAL = REPO / "eval"
for p in (str(REPO), str(EVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tier2 import embedder_pin as E  # noqa: E402
from tier2 import judge_provenance as P  # noqa: E402
from tier2 import score_v2  # noqa: E402

PIN = "5617a9f61b028005a4858fdac845db406aefb181"
OTHER = "0" * 40
TIER2 = EVAL / "tier2"
ENTITY_RUNS = EVAL / "tier2_entity_runs"


def manifest(recorded=PIN, requested=PIN, model="BAAI/bge-m3", placeholder=False):
    m = {"similarity_model_id": model, "similarity_is_placeholder": placeholder}
    if recorded is not None:
        m["similarity_model_revision"] = recorded
    if requested is not None:
        m["similarity_requested_revision"] = requested
    return m


class TestVerifyManifestEmbedder(unittest.TestCase):
    def test_the_registered_revision_is_the_documented_one(self):
        self.assertEqual(E.BGE_M3_REVISION, PIN)

    def test_matching_manifest_is_ok(self):
        r = E.verify_manifest_embedder(manifest())
        self.assertEqual((r["status"], r["recorded"], r["expected"]), (E.OK, PIN, PIN))

    def test_a_different_resolved_revision_is_rejected(self):
        r = E.verify_manifest_embedder(manifest(recorded=OTHER))
        self.assertEqual(r["status"], E.WRONG_REVISION)
        self.assertEqual((r["recorded"], r["expected"]), (OTHER, PIN))

    def test_a_request_for_another_revision_is_rejected_even_if_it_resolved_to_the_pin(self):
        self.assertEqual(E.verify_manifest_embedder(manifest(requested=OTHER))["status"], E.WRONG_REVISION)

    def test_an_unrecorded_revision_is_rejected(self):
        self.assertEqual(E.verify_manifest_embedder(manifest(recorded=None))["status"], E.MISSING)
        self.assertEqual(E.verify_manifest_embedder(manifest(recorded=""))["status"], E.MISSING)

    def test_another_model_is_rejected(self):
        self.assertEqual(E.verify_manifest_embedder(manifest(model="BAAI/bge-m3-other"))["status"], E.WRONG_MODEL)

    def test_a_declared_placeholder_has_no_embedder(self):
        self.assertEqual(E.verify_manifest_embedder(manifest(recorded=None, requested=None, placeholder=True))["status"], E.NOT_APPLICABLE)

    def test_a_custom_expected_value_is_honoured(self):
        self.assertEqual(E.verify_manifest_embedder(manifest(recorded=OTHER, requested=OTHER), expected=OTHER)["status"], E.OK)

    def test_the_vendored_check_alone_would_accept_the_wrong_revision(self):
        from harness.similarity import is_real_bge_m3_run
        wrong = {"pipeline_config": "D", "similarity_backend": "bge-m3", "similarity_model_id": "BAAI/bge-m3",
                 "similarity_metric": None, "similarity_is_placeholder": False, "similarity_range": [-1.0, 1.0],
                 "similarity_model_revision": OTHER}
        from harness.similarity import BGE_M3_METRIC_NAME
        wrong["similarity_metric"] = BGE_M3_METRIC_NAME
        self.assertTrue(is_real_bge_m3_run(wrong))                                    # documents the gap (vendored code is not edited)
        self.assertEqual(E.verify_manifest_embedder(wrong)["status"], E.WRONG_REVISION)


class TestLauncherDefaultsMatchThePin(unittest.TestCase):
    FILES = ["eval/tier2/run_tier2.py", "eval/tier2/judge_validation/memory_probe.py", "validation/tier2_entity/build_kernels.py",
             "validation/tier2_smoke/build_kernel.py", "validation/baselines_smoke/build_kernel.py", "dkmem/baselines/cli.py", "dkmem/store/cli.py"]
    HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
    KEY = re.compile(r"similarity-revision|embedding-revision|SIMILARITY_REVISION|EMB_REV")

    def test_every_launcher_embedder_revision_literal_equals_the_registered_pin(self):
        seen = 0
        for rel in self.FILES:
            f = REPO / rel
            if not f.exists():
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if self.KEY.search(line):
                    for h in self.HEX40.findall(line):
                        seen += 1
                        self.assertEqual(h, PIN, f"{rel}:{n} embedder revision differs from embedder_pin.BGE_M3_REVISION")
        self.assertGreaterEqual(seen, 5)          # the scan found the launcher literals (guards against a silently empty scan)


V2_FULL = "v2_full_20261017"          # the v2 full run (8 runs); counted separately so neither set can switch the other's check off
_ALL_MANIFESTS = [f for f in glob.glob(str(EVAL / "tier2*runs" / "**" / "run_manifest.json"), recursive=True) if "/src/" not in f.replace("\\", "/")]
MANIFESTS = [f for f in _ALL_MANIFESTS if V2_FULL not in f]
V2_FULL_MANIFESTS = [f for f in _ALL_MANIFESTS if V2_FULL in f]


@unittest.skipUnless(len(MANIFESTS) == 28, "the 28 stored Tier-2 run manifests are not all present (gitignored run outputs)")
class TestStoredManifests(unittest.TestCase):
    def test_all_28_recorded_embedder_revisions_are_the_registered_one(self):
        for f in MANIFESTS:
            with self.subTest(manifest=f[-70:]):
                r = E.verify_manifest_embedder(json.loads(Path(f).read_text(encoding="utf-8")))
                self.assertEqual(r["status"], E.OK, r)


@unittest.skipUnless(len(V2_FULL_MANIFESTS) == 8, "the 8 v2 full-run manifests are not all present (gitignored run outputs)")
class TestStoredV2FullManifests(unittest.TestCase):
    def test_all_8_v2_full_run_embedder_revisions_are_the_registered_one(self):
        for f in V2_FULL_MANIFESTS:
            with self.subTest(manifest=f[-70:]):
                r = E.verify_manifest_embedder(json.loads(Path(f).read_text(encoding="utf-8")))
                self.assertEqual(r["status"], E.OK, r)


SMOKE_C = ENTITY_RUNS / "cd7b_smoke_v10_20261014" / "runs" / "t2cd7b-smoke-v10-dk-mem-lexicon-C_seed0"
SMOKE_D = ENTITY_RUNS / "cd7b_smoke_v10_20261014" / "runs" / "t2cd7b-smoke-v10-dk-mem-lexicon-D_seed0"
EVALUATOR = REPO / "validation" / "tier2_entity" / "kernel_cd7b_smoke_v10.evaluator"


@unittest.skipUnless(SMOKE_C.is_dir() and SMOKE_D.is_dir() and (EVALUATOR / "qa_id_map.json").exists(), "stored smoke runs or evaluator inputs absent")
class TestWrapperAndScorerRejectAWrongRevision(unittest.TestCase):
    def copy(self, src, **manifest_edits):
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, ignore_errors=True)
        for p in src.parent.iterdir():
            if p.name == src.name or p.name.startswith(src.name + "."):
                (shutil.copytree if p.is_dir() else shutil.copy)(p, t / p.name)
        f = t / src.name / "run_manifest.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        for k, v in manifest_edits.items():
            if v is None:
                m.pop(k, None)
            else:
                m[k] = v
        f.write_text(json.dumps(m), encoding="utf-8")
        return t / src.name

    def verified(self, rd):
        return P.validate_run_verified(rd, TIER2 / "data", passthrough_file=EVALUATOR / "t2_passthrough_used.jsonl",
                                       qa_id_map=EVALUATOR / "qa_id_map.json", frozen_file=TIER2 / "FROZEN_v1.0.json")

    def test_unmodified_runs_pass_for_both_configs(self):
        for src in (SMOKE_C, SMOKE_D):
            rep = self.verified(self.copy(src))
            self.assertTrue(rep["ok"], (src.name, rep["errors"][:3]))
            self.assertEqual(rep["embedder_verification"]["status"], E.OK)

    def test_wrong_revision_is_a_contract_error_for_config_c_and_d(self):
        for src in (SMOKE_C, SMOKE_D):
            rep = self.verified(self.copy(src, similarity_model_revision=OTHER))
            self.assertFalse(rep["ok"], src.name)
            self.assertIn("embedder_revision_wrong_revision", {e["code"] for e in rep["errors"]})
            self.assertEqual(rep["embedder_verification"]["recorded"], OTHER)

    def test_wrong_request_and_missing_revision_are_errors(self):
        rep = self.verified(self.copy(SMOKE_D, similarity_requested_revision=OTHER))
        self.assertIn("embedder_revision_wrong_revision", {e["code"] for e in rep["errors"]})
        rep = self.verified(self.copy(SMOKE_D, similarity_model_revision=None))
        self.assertIn("embedder_revision_missing", {e["code"] for e in rep["errors"]})

    def test_the_scorer_excludes_a_run_with_a_wrong_revision_before_any_metric(self):
        rd = self.copy(SMOKE_D, similarity_model_revision=OTHER)
        res = score_v2.score_run(rd, run_data_dir=TIER2 / "data", arm_frozen=TIER2 / "FROZEN_v1.0.json", evaluator=EVALUATOR,
                                 turns={}, pairs=[], injected=set(), lexicon=None, data_dir=TIER2 / "data", conv_ids=[])
        self.assertFalse(res["provenance_contract_ok"])
        self.assertIn("provenance contract failed", res["excluded"])
        self.assertIn("embedder_revision_wrong_revision", {e["code"] for e in res["provenance_errors"]})
        self.assertEqual(res["embedder_verification"]["status"], E.WRONG_REVISION)
        self.assertNotIn("overall", res)                                              # no metric was computed

    def test_the_scorer_calls_the_verified_wrapper_not_the_bare_contract(self):
        failing = {"ok": False, "errors": [{"code": "x", "message": "y"}], "warnings": [], "judge_log_verification": None,
                   "embedder_verification": {"status": "stub"}}
        rd = self.copy(SMOKE_D)
        with mock.patch.object(score_v2.judge_provenance, "validate_run_verified", return_value=failing) as w, \
                mock.patch.object(score_v2.contract, "validate_run", side_effect=AssertionError("bare contract must not be used")):
            res = score_v2.score_run(rd, run_data_dir=TIER2 / "data", arm_frozen=TIER2 / "FROZEN_v1.0.json", evaluator=EVALUATOR,
                                     turns={}, pairs=[], injected=set(), lexicon=None, data_dir=TIER2 / "data", conv_ids=[])
        w.assert_called_once()
        self.assertIn("excluded", res)


if __name__ == "__main__":
    unittest.main()
