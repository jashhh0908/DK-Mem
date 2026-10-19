"""Tier-2 v2.1 repair package (data/tier2_v2_1): structure, separation and generator validation. Skips when the package or Team B's delivery
is absent (both are gitignored, local-only data); runs when they are present."""
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "data" / "tier2_v2_1"
DELIVERY = REPO / "data" / "team_b_delivery"
LEX = [REPO / "dkmem/memory/distinction_features.json", DELIVERY / "member3_lexicon_proposal/proposed_distinction_features.json"]
HAVE = (PKG / "MANIFEST_v2_1.json").is_file() and (PKG / "heldout_sealed/build").is_dir() and (PKG / "build_dev").is_dir() and LEX[1].is_file()


def _validate(data_dir, specs, splits):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "eval")]), "PYTHONUTF8": "1"}
    rel = lambda p: Path(p).relative_to(REPO).as_posix()          # the manifest records source paths as given: use the build's repo-relative form
    r = subprocess.run([sys.executable, "-m", "tier2.generate_tier2_v2", "validate", "--data-dir", rel(data_dir), "--specs", rel(specs), "--splits", rel(splits),
                        "--distractors", rel(PKG / "banks/distractors_v2_1.json"), "--lexicon", *map(rel, LEX)], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    return r


@unittest.skipUnless(HAVE, "data/tier2_v2_1 package or Team B's lexicon proposal is not present (local-only data)")
class V21Package(unittest.TestCase):
    def test_dev_build_validates(self):
        r = _validate(PKG / "build_dev", PKG / "dev/pair_specs_v2_1.json", PKG / "dev/splits_v2_1.jsonl")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_heldout_build_validates(self):
        r = _validate(PKG / "heldout_sealed/build", PKG / "heldout_sealed/pair_specs_heldout_v1.json", PKG / "heldout_sealed/splits_heldout_v1.jsonl")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_heldout_is_disjoint_from_dev_in_ids_and_texts(self):
        dev = json.loads((PKG / "dev/pair_specs_v2_1.json").read_text(encoding="utf-8"))["specs"]
        ho = json.loads((PKG / "heldout_sealed/pair_specs_heldout_v1.json").read_text(encoding="utf-8"))["specs"]
        self.assertFalse({s["spec_id"] for s in dev} & {s["spec_id"] for s in ho})
        dev_texts = {t for s in dev for t in (s["text_a"], s["text_b"])}
        self.assertFalse(dev_texts & {t for s in ho for t in (s["text_a"], s["text_b"])})

    def test_no_dev_row_is_called_test_and_exposed_rows_are_marked(self):
        rows = [json.loads(l) for l in (PKG / "dev/splits_v2_1.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["split"] for r in rows}, {"dev"})
        self.assertEqual(sum(r["origin_split"] == "team_b_test_EXPOSED" for r in rows), 30)

    def test_heldout_integrity_audit_does_not_certify_independence(self):
        sys.path[:0] = [str(REPO / "eval")]
        from tier2 import heldout_integrity_v2_1 as H
        rep = H.audit(PKG)
        self.assertEqual(rep["n_heldout"], 24)
        self.assertGreater(rep["n_template_derived_from_exposed"], 0)       # the set must never be reported as independent of the exposed labels
        self.assertEqual(rep["verdict"], "NOT_INDEPENDENT")

    def test_manifest_hashes_match_files(self):
        man = json.loads((PKG / "MANIFEST_v2_1.json").read_text(encoding="utf-8"))["files_sha256"]
        for rel, h in man.items():
            self.assertEqual(hashlib.sha256((PKG / rel).read_bytes()).hexdigest(), h, rel)


if __name__ == "__main__":
    unittest.main()
