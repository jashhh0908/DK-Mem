"""The v2 benchmark ARM: assembly and freeze tooling (tier2.freeze_v2), the v2 arm of the kernel builder, and payload self-containment.

These tests build a THROWAWAY arm from the local v2.1 package under ``eval/tier2/_selftest_<id>/`` (git-ignored, removed afterwards) and freeze THAT
copy with the tooling's own confirmation flag. They never create ``eval/tier2/v2/FROZEN_v2.0.json``: the real freeze is a one-off, deliberate step.

Skipped with a reason when the (git-ignored) v2.1 package or the Tier-2 data is absent.
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).parent.parent
EVAL = REPO / "eval"
TIER2 = EVAL / "tier2"
PKG = REPO / "data" / "tier2_v2_1"
SPECS, SPLITS, BANK = PKG / "dev" / "pair_specs_v2_1.json", PKG / "dev" / "splits_v2_1.jsonl", PKG / "banks" / "distractors_v2_1.json"
LEXICON = REPO / "dkmem" / "memory" / "distinction_features.json"
HAVE = all(p.is_file() for p in (SPECS, SPLITS, BANK, LEXICON)) and (TIER2 / "data" / "tier2_conversations.jsonl").is_file()

for p in (str(REPO), str(EVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_build_kernels():
    spec = importlib.util.spec_from_file_location("_bk_v2_test", REPO / "validation" / "tier2_entity" / "build_kernels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_package(out: Path, lexicon: Path = LEXICON):
    from tier2 import generate_tier2_v2 as V
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = V.main(["build", "--specs", str(SPECS), "--splits", str(SPLITS), "--distractors", str(BANK), "--lexicon", str(lexicon), "--out", str(out)])
    assert rc == 0, buf.getvalue()[-500:]


@unittest.skipUnless(HAVE, "the v2.1 package (data/tier2_v2_1) or the Tier-2 data is not present (git-ignored, local only)")
class TestFreezeTooling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tier2 import freeze_v2 as F
        cls.F = F
        cls.tmp = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        cls.build = cls.tmp / "build"
        build_package(cls.build)
        cls.roots = []

    @classmethod
    def new_root(cls):
        root = TIER2 / f"_selftest_{uuid.uuid4().hex[:8]}"
        cls.roots.append(root)
        cls.addClassCleanup(shutil.rmtree, root, ignore_errors=True)
        return root

    def assemble(self, root=None, **kw):
        root = root or self.new_root()
        with redirect_stdout(io.StringIO()):
            asm = self.F.assemble(self.build, SPECS, [SPLITS], [BANK], LEXICON, root, **kw)
        return root, asm

    def test_assemble_lays_out_the_arm_and_records_a_passing_validation(self):
        root, asm = self.assemble()
        self.assertTrue(asm["validation"]["ok"])
        for rel in [*(f"data/{f}" for f in self.F.DATA_FILES), "banks/pair_specs_v2.json", "banks/splits.jsonl", "banks/distractors/distractors_v2_1.json",
                    "lexicon/distinction_features.json", "ASSEMBLY.json"]:
            self.assertTrue((root / rel).is_file(), rel)
        self.assertFalse((root / "FROZEN_v2.0.json").exists())                      # assembling never freezes
        self.assertEqual(asm["sources_sha256"]["lexicon_lf"], self.F.sha(root / "lexicon" / "distinction_features.json"))
        self.assertEqual(asm["benchmark_manifest_counts"]["injected_pairs"], 126)

    def test_an_arm_root_is_never_overwritten(self):
        root, _ = self.assemble()
        with self.assertRaisesRegex(SystemExit, "not empty"):
            self.assemble(root)

    def test_a_build_that_does_not_validate_against_the_given_lexicon_is_refused_and_nothing_is_written(self):
        other = self.tmp / "other_lexicon.json"
        other.write_text(LEXICON.read_text(encoding="utf-8").replace("Pilot-reviewed.", "Pilot-reviewed (edited).", 1), encoding="utf-8")
        root = self.new_root()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "does not validate"):
            self.F.assemble(self.build, SPECS, [SPLITS], [BANK], other, root)
        self.assertFalse(root.exists())

    def test_a_directory_that_is_not_a_build_output_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "not a generate_tier2_v2 build output"):
            self.F.assemble(self.tmp, SPECS, [SPLITS], [BANK], LEXICON, self.new_root())

    def test_the_lexicon_is_stored_with_lf_line_endings(self):
        crlf = self.tmp / "crlf_lexicon.json"
        crlf.write_bytes(LEXICON.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        build = self.tmp / "build_crlf"
        build_package(build, crlf)
        root = self.new_root()
        with redirect_stdout(io.StringIO()):
            asm = self.F.assemble(build, SPECS, [SPLITS], [BANK], crlf, root)
        stored = (root / "lexicon" / "distinction_features.json").read_bytes()
        self.assertNotIn(b"\r", stored)
        self.assertNotEqual(asm["sources_sha256"]["lexicon_raw"], asm["sources_sha256"]["lexicon_lf"])

    def test_freeze_needs_the_explicit_confirmation_and_writes_once(self):
        root, _ = self.assemble()
        with self.assertRaisesRegex(SystemExit, "--confirm-freeze-v2"):
            self.F.freeze(root, confirm=False)
        self.assertFalse((root / "FROZEN_v2.0.json").exists())
        path = self.F.freeze(root, confirm=True)
        fz = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((fz["status"], fz["version"]), ("FROZEN", "2.0"))
        self.assertEqual(set(fz["sha256"]), set(self.F.pin_table_keys(root)))
        self.assertIn("lexicon/distinction_features.json", fz["sha256"])
        self.assertEqual(self.F.verify(root), [])
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.F.freeze(root, confirm=True)

    def test_a_frozen_arm_is_accepted_by_the_runners_own_verification(self):
        import tier2.run_tier2 as run
        root, _ = self.assemble()
        self.F.freeze(root, confirm=True)
        res = run.verify_frozen(root / "data", root / "FROZEN_v2.0.json")
        self.assertIn("data/tier2_conversations.jsonl", res["verified_files"])
        self.assertIn("lexicon/distinction_features.json", res["verified_files"])

    def test_every_kind_of_tampering_is_detected(self):
        root, _ = self.assemble()
        self.F.freeze(root, confirm=True)
        for rel in ("data/tier2_conversations.jsonl", "data/tier2_gold.jsonl", "lexicon/distinction_features.json", "banks/pair_specs_v2.json"):
            f = root / rel
            original = f.read_bytes()
            f.write_bytes(original + b" ")
            self.assertIn(f"{rel}: changed", self.F.verify(root), rel)
            f.write_bytes(original)
            f.unlink()
            self.assertIn(f"{rel}: missing", self.F.verify(root), rel)
            f.write_bytes(original)
        self.assertEqual(self.F.verify(root), [])

    def test_an_unvalidated_assembly_cannot_be_frozen(self):
        root, asm = self.assemble(validate=False)
        self.assertIsNone(asm["validation"]["ok"])
        with self.assertRaisesRegex(SystemExit, "not validated"):
            self.F.freeze(root, confirm=True)
        self.assertFalse((root / "FROZEN_v2.0.json").exists())

    def test_a_lexicon_changed_after_assembly_cannot_be_frozen(self):
        root, _ = self.assemble()
        f = root / "lexicon" / "distinction_features.json"
        f.write_bytes(f.read_bytes() + b" ")
        with self.assertRaisesRegex(SystemExit, "differs from the lexicon that was validated"):
            self.F.freeze(root, confirm=True)

    def test_the_real_arm_once_frozen_still_verifies(self):
        # the tests above only freeze throwaway _selftest_ roots; the real arm is frozen once, by the lead (2026-10-17), and must stay intact
        real = TIER2 / "v2"
        if not (real / "FROZEN_v2.0.json").exists():
            self.skipTest("eval/tier2/v2 is not frozen")
        self.assertEqual(self.F.verify(real), [])


@unittest.skipUnless(HAVE, "the v2.1 package (data/tier2_v2_1) or the Tier-2 data is not present (git-ignored, local only)")
class TestKernelV2Arm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tier2 import freeze_v2 as F
        cls.BK = load_build_kernels()
        cls.saved_arm = dict(cls.BK.ARMS["v2"])
        cls.tmp = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        build = cls.tmp / "build"
        build_package(build)
        cls.root = TIER2 / f"_selftest_{uuid.uuid4().hex[:8]}"
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        cls.root_rel = "eval/tier2/" + cls.root.name
        with redirect_stdout(io.StringIO()):
            F.assemble(build, SPECS, [SPLITS], [BANK], LEXICON, cls.root)
        cls.F = F

    def ensure_frozen(self):
        if not (self.root / "FROZEN_v2.0.json").exists():
            self.F.freeze(self.root, confirm=True)

    def setUp(self):
        self.BK.ARMS["v2"] = dict(self.saved_arm)         # _build re-registers the arm from the spec; keep the module default intact

    def tearDown(self):
        self.BK.ARMS["v2"] = dict(self.saved_arm)

    def spec(self, **kw):
        return {"kind": "tier2", "arm": "v2", "arm_root": self.root_rel, "run_prefix": "t2v2", "smoke": True, **kw}

    def write(self, spec):
        out = self.tmp / f"kernel_{uuid.uuid4().hex[:6]}"
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = self.BK.write(spec, "user", "slug", str(out))
        return rc, out, err.getvalue()

    def test_an_unfrozen_arm_is_never_built(self):
        other = TIER2 / f"_selftest_{uuid.uuid4().hex[:8]}"
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        with redirect_stdout(io.StringIO()):
            self.F.assemble(self.tmp / "build", SPECS, [SPLITS], [BANK], LEXICON, other)
        rc, out, err = self.write(self.spec(arm_root="eval/tier2/" + other.name))
        self.assertEqual(rc, 1)
        self.assertIn("not frozen", err)
        self.assertFalse(out.exists())

    def test_frozen_arm_builds_a_label_free_payload_with_the_pinned_lexicon(self):
        self.ensure_frozen()
        rc, out, err = self.write(self.spec())
        self.assertEqual(rc, 0, err)
        ev = json.loads((Path(str(out) + ".evaluator") / "payload_manifest.json").read_text(encoding="utf-8"))
        paths = {f["path"] for f in ev["files"]}
        r = self.root_rel
        for needed in (f"{r}/data/tier2_conversations.jsonl", f"{r}/data/tier2_qa_questions.jsonl", f"{r}/FROZEN_v2.0.json", f"{r}/lexicon/distinction_features.json",
                       "eval/tier2/entry_extraction.py", "eval/tier2/run_tier2.py", "inputs/t2_passthrough.jsonl"):
            self.assertIn(needed, paths, needed)
        for forbidden in ("gold", "qa_reference", "injection_metadata", "pair_specs", "distractors", "splits", "ASSEMBLY"):
            self.assertFalse([p for p in paths if forbidden in p.lower()], forbidden)
        args = ev["run_args"]
        self.assertEqual(args[args.index("--lexicon") + 1], f"/kaggle/working/src/{r}/lexicon/distinction_features.json")
        self.assertEqual(args[args.index("--configs") + 1], self.BK.V2_CONFIGS)
        self.assertEqual(args[args.index("--judge-prompt") + 1], "merge_judge_entity_t2_v2")
        self.assertIn("--judge-shard", args)

    def test_the_v2_smoke_selection_covers_every_language(self):
        self.ensure_frozen()
        rc, out, err = self.write(self.spec())
        self.assertEqual(rc, 0, err)
        ev = json.loads((Path(str(out) + ".evaluator") / "payload_manifest.json").read_text(encoding="utf-8"))
        conv_text = {}
        for c in (json.loads(l) for l in (self.root / "data" / "tier2_conversations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()):
            conv_text[c["conversation_id"]] = {t["language"] for s in c["sessions"] for t in s["turns"] if t["speaker"] == "user"}
        langs = sorted(next(iter(conv_text[c])) for c in ev["conversations"])
        self.assertEqual(langs, ["gu", "hi", "hi", "mr"])

    def test_a_built_kernel_is_fresh_and_a_changed_lexicon_makes_it_stale_or_unbuildable(self):
        self.ensure_frozen()
        rc, out, err = self.write(self.spec())
        self.assertEqual(rc, 0, err)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.BK.check(str(out)), 0)
        lex = self.root / "lexicon" / "distinction_features.json"
        original = lex.read_bytes()
        try:
            lex.write_bytes(original + b" ")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(self.BK.check(str(out)), 1)                 # the pin no longer matches
            rc, _, err = self.write(self.spec())
            self.assertEqual(rc, 1)
            self.assertIn("no longer matches its frozen pin", err)
        finally:
            lex.write_bytes(original)

    def test_any_valid_config_list_builds_and_invalid_ones_are_refused(self):
        self.ensure_frozen()
        rc, out, err = self.write(self.spec(configs="A:off,A:lexicon"))
        self.assertEqual(rc, 0, err)
        ev = json.loads((Path(str(out) + ".evaluator") / "payload_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(ev["run_args"][ev["run_args"].index("--configs") + 1], "A:off,A:lexicon")
        for bad in ("E:off", "A:off,A:off", "", "A:lexicon+llm", "full"):
            rc, _, err = self.write(self.spec(configs=bad))
            self.assertEqual(rc, 1, bad)

    def test_the_v1_arms_keep_exactly_their_two_config_sets(self):
        for arm in ("v1.0", "mirror"):
            self.BK.check_configs(arm, self.BK.ALL_CONFIGS)
            self.BK.check_configs(arm, self.BK.D_CONFIGS)
            with self.assertRaises(self.BK.BuildError):
                self.BK.check_configs(arm, "A:off")
        self.BK.check_configs("v2", self.BK.V2_CONFIGS)

    def test_the_arm_root_must_live_under_eval(self):
        for bad in ("data/tier2_v2_1", "../elsewhere", "eval/../x"):
            with self.assertRaises(self.BK.BuildError):
                self.BK.v2_arm(bad)


@unittest.skipUnless((TIER2 / "data" / "tier2_conversations.jsonl").is_file(), "the v1.0 Tier-2 data is not present (git-ignored, local only)")
class TestPayloadIsSelfContained(unittest.TestCase):
    """The runner imports ``entry_extraction`` at import time; a payload without it would crash on Kaggle. Import the runner from a tree that holds ONLY
    the payload's files."""

    def test_the_v1_payload_imports_the_runner_on_its_own(self):
        BK = load_build_kernels()
        files, _ = BK.t2_collect("v1.0")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for rel, text in files.items():
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8", newline="\n")
        code = "import tier2.run_tier2 as r, tier2.entry_extraction as e; print(r.__file__); print(e.__file__)"
        proc = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=tmp / "eval", capture_output=True, text=True, timeout=120,
                              env={**os.environ, "PYTHONPATH": str(tmp / "eval") + os.pathsep + str(tmp)})
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        for line in proc.stdout.strip().splitlines():
            self.assertTrue(Path(line).resolve().is_relative_to(tmp.resolve()), line)      # imported from the payload tree, not from the checkout


class TestScoreV2ReplaysThePinnedLexicon(unittest.TestCase):
    """score_v2 replays the gate offline: by default it must apply the lexicon the arm pins (the one its gated runs used), not the v1 file."""

    def setUp(self):
        from tier2 import score_v2
        self.S = score_v2

    def test_an_arm_that_pins_a_lexicon_is_replayed_with_that_lexicon(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "lexicon").mkdir()
        shutil.copyfile(LEXICON, root / "lexicon" / "distinction_features.json")
        path, is_pin = self.S.replay_lexicon(root / "FROZEN_v2.0.json")
        self.assertEqual(path, root / "lexicon" / "distinction_features.json")
        self.assertTrue(is_pin)

    def test_an_arm_without_a_pinned_lexicon_falls_back_to_v1(self):
        from harness import _dkmem
        path, is_pin = self.S.replay_lexicon(TIER2 / "FROZEN_v1.0.json")
        self.assertEqual(path, Path(_dkmem.LEXICON_PATH))
        self.assertFalse(is_pin)

    def test_an_explicit_other_lexicon_is_reported_as_not_the_pin(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "lexicon").mkdir()
        (root / "lexicon" / "distinction_features.json").write_text('{"x": 1}\n', encoding="utf-8")
        path, is_pin = self.S.replay_lexicon(root / "FROZEN_v2.0.json", LEXICON)
        self.assertEqual(path, LEXICON)
        self.assertFalse(is_pin)

    @unittest.skipUnless((TIER2 / "v2" / "FROZEN_v2.0.json").is_file(), "eval/tier2/v2 is not frozen")
    def test_the_real_v2_arm_replays_its_frozen_lexicon(self):
        path, is_pin = self.S.replay_lexicon(TIER2 / "v2" / "FROZEN_v2.0.json")
        self.assertEqual(path, TIER2 / "v2" / "lexicon" / "distinction_features.json")
        self.assertTrue(is_pin)


if __name__ == "__main__":
    unittest.main()
