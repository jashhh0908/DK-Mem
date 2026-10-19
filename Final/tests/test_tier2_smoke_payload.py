"""Protection of the Tier-2 evaluator-only files, and the exact Kaggle smoke-test payload.

Three Tier-2 files carry what Team B's key carries, or ids that join back to it, and the system under test must never
receive them: the gold (relation labels), the QA references (answer aliases; their ``qa_type`` also tells same from
different) and the injection metadata (which turns are injected, source Tier-1 pair ids). The pair specs hold relation
labels too. This test pins

* the ``.gitignore`` rules and that nothing evaluator-only is tracked;
* the payload definition of ``validation/tier2_smoke/build_kernel.py`` (exactly the declared files, refusing anything
  evaluator-only, gold-like or renamed);
* that the smoke runner executes from the unpacked notebook payload alone, with a Python audit hook recording every file
  it tries to open.

No protected file is opened here: payload checks use the frozen sha256 list and field NAMES, and the end-to-end run uses
fake models. Skipped where ``eval/`` or ``validation/`` is absent (the Kaggle payload embeds ``dkmem/`` and ``tests/``).
"""

import base64
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).parent.parent
BUILDER = REPO / "validation" / "tier2_smoke" / "build_kernel.py"
EVAL = REPO / "eval"

PROTECTED_NAMES = ("tier2_gold.jsonl", "tier2_qa_reference.jsonl", "tier2_injection_metadata.jsonl", "pair_specs.json",
                   "pilot_selection.json", "teamB_answer_key.jsonl")
# label-derived: ignored by git AND absent from the payload. (banks/distractors.json is hand-authored text, not label-derived:
# committable, but still kept out of the payload because it would reveal which turns are distractors.)
EVALUATOR_ONLY_PATHS = ("eval/tier2/data/tier2_gold.jsonl", "eval/tier2/data/tier2_qa_reference.jsonl",
                        "eval/tier2/data/tier2_injection_metadata.jsonl", "eval/tier2/banks/pair_specs.json",
                        "eval/tier2/pilot/pilot_selection.json",
                        "eval/tier2_full_runs/t2full-dk-mem-lexicon-C_seed0.tier2_contract_report.json",
                        "eval/tier2_full_runs/t2_passthrough.jsonl")     # historical rule separates injected from distractor turns
LABEL_FREE_PATHS = ("eval/tier2/data/tier2_conversations.jsonl", "eval/tier2/data/tier2_qa_questions.jsonl")

B = None  # the builder module


def setUpModule():
    global B
    if not (BUILDER.exists() and (EVAL / "tier2" / "run_tier2.py").exists()):
        raise unittest.SkipTest("eval/ and validation/tier2_smoke are not present in this checkout")
    spec = importlib.util.spec_from_file_location("_tier2_smoke_build_kernel", BUILDER)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None


class TestGitProtection(unittest.TestCase):
    def setUp(self):
        if not (REPO / ".git").exists():
            self.skipTest("not a git checkout (e.g. the Kaggle copy)")

    def test_evaluator_only_paths_are_ignored(self):
        for rel in EVALUATOR_ONLY_PATHS:
            r = _git("check-ignore", "-q", rel)
            if r is None:
                self.skipTest("git is not available")
            self.assertEqual(r.returncode, 0, f"{rel} is not ignored")

    def test_evaluator_only_names_are_ignored_at_any_depth(self):
        for name in PROTECTED_NAMES:
            r = _git("check-ignore", "-q", f"some/other/place/{name}")
            self.assertEqual(r.returncode, 0, name)

    def test_label_free_inputs_are_not_ignored(self):
        for rel in LABEL_FREE_PATHS:
            r = _git("check-ignore", "-q", rel)
            self.assertEqual(r.returncode, 1, f"{rel} is ignored but is a payload input")

    def test_nothing_evaluator_only_is_tracked(self):
        r = _git("ls-files")
        out = r.stdout.splitlines()
        offenders = [f for f in out if any(n in f.lower() for n in ("tier2_gold", "tier2_qa_reference", "tier2_injection_metadata", "pair_specs",
                                                                     "pilot_selection.json", "answer_key", "tier2_contract_report"))]
        self.assertEqual(offenders, [])

    def test_a_dry_run_add_stages_no_evaluator_only_file(self):
        r = _git("add", "-n", "eval/")
        staged = [l for l in r.stdout.splitlines()]
        self.assertFalse([l for l in staged if any(n in l for n in ("tier2_gold", "tier2_qa_reference", "tier2_injection_metadata",
                                                                      "pair_specs", "pilot_selection.json", "tier2_contract_report.json"))])
        # the gitignore file itself is the source of truth
        text = (REPO / ".gitignore").read_text(encoding="utf-8")
        for pat in ("tier2_gold*.jsonl", "tier2_qa_reference*.jsonl", "tier2_injection_metadata*.jsonl", "pair_specs*.json",
                    "pilot_selection*.json", "*tier2_contract_report*.json", "*answer_key*.jsonl"):
            self.assertIn(pat, [l.strip() for l in text.splitlines()])


class TestPayloadDefinition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = B.collect_files()

    def test_the_real_payload_passes_every_check(self):
        B.check_payload(self.files)

    def test_non_dkmem_part_is_exactly_the_declared_list(self):
        self.assertEqual({n for n in self.files if not n.startswith("dkmem/")}, set(B.ALLOWED_NON_DKMEM))

    def test_no_evaluator_only_or_key_material_is_present_by_name(self):
        low = [n.lower() for n in self.files]
        for part in B.FORBIDDEN_NAME_PARTS:
            self.assertFalse([n for n in low if part in n], part)
        self.assertFalse([n for n in self.files if n.startswith(("dkmem/data/", "dkmem/eval/", "eval/tests/", "eval/results/",
                                                                 "eval/tier2/banks/", "eval/tier2/pilot/"))])

    def test_no_payload_file_is_byte_identical_to_a_frozen_evaluator_only_file(self):
        protected = B.protected_hashes(self.files)
        self.assertGreaterEqual(len(protected), 5)
        self.assertFalse({hashlib.sha256(t.encode("utf-8")).hexdigest() for t in self.files.values()} & protected)

    def test_declared_inputs_are_the_frozen_label_free_files(self):
        fz = json.loads(self.files["eval/tier2/FROZEN_v1.0.json"])["sha256"]
        for rel in B.DATA_FILES:
            self.assertEqual(hashlib.sha256(self.files[rel].encode("utf-8")).hexdigest(), fz[rel.removeprefix("eval/tier2/")])
        # field names only
        conv = json.loads(self.files[B.CONVERSATIONS].splitlines()[0])
        self.assertEqual(set(conv), B.CONVERSATION_KEYS)
        self.assertEqual(set(json.loads(self.files[B.QUESTIONS].splitlines()[0])), B.QUESTION_KEYS)
        self.assertEqual(set(json.loads(self.files[B.PASSTHROUGH].splitlines()[0])), B.PASSTHROUGH_KEYS)

    def test_system_side_code_names_no_evaluator_only_file(self):
        names = ("tier2_gold", "qa_reference", "injection_metadata", "pair_specs", "teamB", "opaque_entity_id", "source_tier1")
        for rel in B.SYSTEM_FILES:
            if rel.endswith(".py"):
                text = self.files[rel]
                self.assertFalse([n for n in names if n in text], (rel, [n for n in names if n in text]))

    def test_selection_is_label_free_and_deterministic(self):
        sel = B.select_conversations(self.files)
        self.assertEqual(len(sel), 3)
        self.assertEqual(sel, B.select_conversations(self.files))
        langs = []
        for c in map(json.loads, self.files[B.CONVERSATIONS].splitlines()):
            if c["conversation_id"] in sel:
                langs.append({t["language"] for s in c["sessions"] for t in s["turns"] if t["speaker"] == "user"}.pop())
        self.assertEqual(sorted(langs), ["de", "hi", "hi"])

    def refused(self, mutate):
        files = dict(self.files)
        mutate(files)
        with self.assertRaises(B.BuildError):
            B.check_payload(files)

    def test_refuses_evaluator_only_files_and_key(self):
        for name in ("eval/tier2/data/tier2_gold.jsonl", "eval/tier2/data/tier2_qa_reference.jsonl", "eval/tier2/data/tier2_injection_metadata.jsonl",
                     "eval/tier2/banks/pair_specs.json", "eval/tier2/banks/distractors.json", "eval/teamB_answer_key.jsonl",
                     "eval/tier2/generate_tier2.py", "eval/tier2/contract.py", "dkmem/data/tier1/eval_input/anything.jsonl", "dkmem/eval/cli.py"):
            with self.subTest(name=name):
                self.refused(lambda f, n=name: f.__setitem__(n, "{}\n"))

    def test_refuses_a_renamed_copy_of_a_protected_file(self):
        fz = json.loads(self.files["eval/tier2/FROZEN_v1.0.json"])["sha256"]
        # a file whose bytes hash to a frozen evaluator-only value cannot be built here without opening the protected file, so
        # simulate it by pinning one payload file's hash into the protected list
        files = dict(self.files)
        victim = "eval/tier2/judge_host.py"
        fake = json.loads(files["eval/tier2/FROZEN_v1.0.json"])
        fake["sha256"]["data/tier2_gold.jsonl"] = hashlib.sha256(files[victim].encode("utf-8")).hexdigest()
        files["eval/tier2/FROZEN_v1.0.json"] = json.dumps(fake)
        with self.assertRaises(B.BuildError):
            B.check_payload(files)
        self.assertIn("data/tier2_gold.jsonl", fz)

    def test_refuses_gold_like_content_and_relation_labels(self):
        made_up = "diff" + "erent"
        self.refused(lambda f: f.__setitem__("eval/tier2/judge_host.py", f["eval/tier2/judge_host.py"] + f'\nX = {{"relation": "{made_up}"}}\n'))
        self.refused(lambda f: f.__setitem__("dkmem/memory/judge.py", f["dkmem/memory/judge.py"] + "\n# " + "t1_" + "0042" + "\n"))
        self.refused(lambda f: f.__setitem__("eval/tier2/run_tier2.py", f["eval/tier2/run_tier2.py"] + "\n# injected_turn_ids\n"))

    def test_refuses_crlf_and_wrong_input_hash_and_bad_structure(self):
        self.refused(lambda f: f.__setitem__("eval/tier2/judge_host.py", f["eval/tier2/judge_host.py"].replace("\n", "\r\n")))
        self.refused(lambda f: f.__setitem__(B.CONVERSATIONS, f[B.CONVERSATIONS] + "\n"))
        first = json.loads(self.files[B.PASSTHROUGH].splitlines()[0])
        extra = json.dumps({**first, "relation_hint": 1})
        self.refused(lambda f: f.__setitem__(B.PASSTHROUGH, extra + "\n" + "\n".join(f[B.PASSTHROUGH].splitlines()[1:]) + "\n"))
        q0 = json.loads(self.files[B.QUESTIONS].splitlines()[0])
        # a question row with an extra field changes the frozen hash AND the field layout: both are refused
        bad_q = json.dumps({**q0, "qa_type": "x"}) + "\n" + "\n".join(self.files[B.QUESTIONS].splitlines()[1:]) + "\n"
        self.refused(lambda f: f.__setitem__(B.QUESTIONS, bad_q))

    def test_refuses_a_missing_required_file(self):
        files = dict(self.files)
        del files["eval/tier2/judge_host.py"]
        with self.assertRaises(B.BuildError):
            B.check_payload(files)


class TestNotebookArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = B.collect_files()
        cls.convs = B.select_conversations(cls.files)
        cls.run_args = B.build_run_args(cls.convs, 0, 0.9, "Qwen/Qwen2.5-3B-Instruct", "r" * 40, "s" * 40)
        cls.nb, cls.payload = B.build_notebook(cls.files, cls.run_args, cls.convs)

    def test_payload_round_trips_and_notebook_is_small(self):
        B.verify_payload(self.payload, self.files)
        self.assertLess(len(json.dumps(self.nb).encode("utf-8")), B.MAX_NOTEBOOK_BYTES)

    def test_notebook_embeds_exactly_the_payload_and_scans_for_forbidden_names(self):
        src = "".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code")
        m = re.search(r'PAYLOAD = "([A-Za-z0-9+/=]+)"', src)
        self.assertIsNotNone(m)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1)))) as z:
            self.assertEqual(sorted(z.namelist()), sorted(self.files))
        self.assertIn("FORBIDDEN = ", src)
        for part in B.FORBIDDEN_NAME_PARTS:
            self.assertIn(part, src)

    def test_run_command_uses_the_frozen_settings_and_a_new_prefix(self):
        a = self.run_args
        self.assertEqual(a[a.index("--configs") + 1], "C:off,C:lexicon,D:off,D:lexicon")
        self.assertEqual(a[a.index("--run-prefix") + 1], "t2frozen-smoke")
        self.assertEqual(a[a.index("--out") + 1], "/kaggle/working/runs")
        self.assertEqual(a[a.index("--conversations") + 1].count(","), 2)
        self.assertEqual(a[a.index("--data-dir") + 1], "/kaggle/working/src/eval/tier2/data")


_DRIVER = textwrap.dedent('''
    import json, os, sys
    root, out, result = sys.argv[1], sys.argv[2], sys.argv[3]
    convs, passthrough = sys.argv[4], sys.argv[5]
    sys.path[:0] = [os.path.join(root, "eval"), root]
    opened = set()
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = args[0]
            opened.add(os.fsdecode(p) if not isinstance(p, int) else str(p))
    sys.addaudithook(hook)
    from unittest import mock
    import tier2.run_tier2 as run
    import harness.similarity as sim
    class FakeEmbedder:
        def embed(self, texts):
            out = []
            for t in texts:
                v = [0.0] * 16
                for i, ch in enumerate(t):
                    v[i % 16] += (ord(ch) % 17) + 1
                out.append(v)
            return out
        def describe(self):
            return {"similarity_backend": "bge-m3", "similarity_model_id": "BAAI/bge-m3", "similarity_model_revision": "fake",
                    "similarity_requested_revision": None, "similarity_dtype": "float16", "similarity_device": "cuda:0",
                    "similarity_max_seq_length": 512, "similarity_embed_batch_size": 32, "similarity_is_placeholder": False}
    class FakeTok:
        name_or_path = "fake/tokenizer"
        eos_token = "<eos>"
        pad_token = "<pad>"
        padding_side = "left"
        chat_template = "fake-template"
        def __len__(self):
            return 1000
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return " ".join(m["content"] for m in messages)
    import types
    class FakeGen:
        backbone = "fake"
        tokenizer = FakeTok()
        config = types.SimpleNamespace(use_chat_template=True)
        def generate(self, prompts, params):
            outs = []
            for m in prompts:
                if m[0]["content"].startswith("You answer questions"):
                    outs.append("unknown")
                else:
                    a, b = m[-1]["content"].split("\\nMemory B: ", 1)
                    same = a[len("Memory A: "):] == b
                    outs.append(json.dumps({"decision": "merge" if same else "keep_both", "reason": "Fake judge."}))
            return outs
        def run_info(self, params):
            return {"backbone": "fake", "generation_params": {"max_new_tokens": params.max_new_tokens}}
    with mock.patch.object(run, "build_models", lambda args: (sim.build_metric(lambda: FakeEmbedder()), FakeGen())):
        rc = run.main(["--passthrough", passthrough, "--data-dir", os.path.join(root, "eval", "tier2", "data"), "--conversations", convs,
                       "--out", out, "--run-prefix", "t2frozen-smoke", "--tau", "0.9"])
    json.dump({"rc": rc, "opened": sorted(opened)}, open(result, "w"))
''')


class TestSmokeRunsFromThePayloadAlone(unittest.TestCase):
    """The unpacked payload (a tree without any evaluator-only file) runs the real runner end to end; an audit hook lists every
    path the process tried to open."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        files = B.collect_files()
        cls.convs = B.select_conversations(files)
        nb, payload = B.build_notebook(files, B.build_run_args(cls.convs, 0, 0.9, "m", "r" * 40, "s" * 40), cls.convs)
        cls.tree = root / "src"
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            z.extractall(cls.tree)
        cls.out = root / "runs"
        cls.result = root / "result.json"
        cls.proc = subprocess.run(
            [sys.executable, "-I", "-c", _DRIVER, str(cls.tree), str(cls.out), str(cls.result), ",".join(cls.convs), str(cls.tree / B.PASSTHROUGH)],
            cwd=root, capture_output=True, text=True, timeout=300)
        cls.res = json.loads(cls.result.read_text(encoding="utf-8")) if cls.result.exists() else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_succeeds(self):
        self.assertIsNotNone(self.res, self.proc.stderr[-2000:])
        self.assertEqual(self.res["rc"], 0, self.proc.stderr[-2000:])

    def test_the_unpacked_tree_has_no_evaluator_only_file(self):
        names = {p.name for p in self.tree.rglob("*") if p.is_file()}
        for n in PROTECTED_NAMES + ("distractors.json", "generate_tier2.py", "validate_tier2.py", "contract.py"):
            self.assertNotIn(n, names)

    def test_no_protected_path_was_even_opened(self):
        bad = [p for p in self.res["opened"] if any(n in p.replace("\\", "/").lower() for n in (
            "tier2_gold", "qa_reference", "injection_metadata", "pair_specs", "answer_key", "distractors", "pilot_selection", "generate_tier2",
            "validate_tier2"))]
        self.assertEqual(bad, [])

    def test_four_runs_with_the_frozen_judge_and_recorded_absence(self):
        runs = sorted(p.name for p in self.out.iterdir() if p.is_dir())
        self.assertEqual(runs, ["t2frozen-smoke-dk-mem-lexicon-C_seed0", "t2frozen-smoke-dk-mem-lexicon-D_seed0",
                                "t2frozen-smoke-embedding-threshold-D_seed0", "t2frozen-smoke-store-surface-only-C_seed0"])
        for r in runs:
            m = json.loads((self.out / r / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["dataset_tier"], "tier2_injected_sessions")
            t2 = m["tier2"]
            self.assertEqual(sorted(t2["conversations"]), sorted(self.convs))
            self.assertEqual(sorted(t2["frozen_files_verified"]),
                             ["data/tier2_conversations.jsonl", "data/tier2_qa_questions.jsonl"])
            self.assertIn("data/tier2_gold.jsonl", t2["frozen_files_not_in_payload"])
            self.assertIn("banks/pair_specs.json", t2["frozen_files_not_in_payload"])
            rows = [json.loads(l) for l in (self.out / r / "pairwise_eval.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3 * 45)
            if r.endswith("C_seed0"):
                self.assertEqual({x["threshold"] for x in rows}, {None})
                self.assertEqual(m["judge_prompt_sha256"], "3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff")
                self.assertEqual(m["generation_run_info"]["generation_params"]["max_new_tokens"], 128)


class TestFrozenVerificationIsPayloadAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        for rel, text in B.collect_files().items():          # the payload tree: no evaluator-only file in it
            p = cls.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="")
        cls.data = cls.root / "eval" / "tier2" / "data"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def verify(self):
        # a clean interpreter, so the payload copy (not this repository's eval/) is the module under test
        code = textwrap.dedent(f'''
            import json, sys
            sys.path[:0] = [r"{self.root / 'eval'}", r"{self.root}"]
            from pathlib import Path
            import tier2.run_tier2 as run
            try:
                r = run.verify_frozen(Path(r"{self.data}"))
                print(json.dumps({{"ok": True, "verified": r["verified_files"], "absent": r["not_in_payload_evaluator_only"]}}))
            except RuntimeError as e:
                print(json.dumps({{"ok": False, "error": str(e)}}))
        ''')
        p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        return json.loads(p.stdout.strip().splitlines()[-1])

    def test_system_files_verified_and_evaluator_files_reported_absent(self):
        r = self.verify()
        self.assertTrue(r["ok"], r)
        for rel in ("data/tier2_conversations.jsonl", "data/tier2_qa_questions.jsonl"):
            self.assertIn(rel, r["verified"])
        for rel in ("data/tier2_gold.jsonl", "data/tier2_qa_reference.jsonl", "data/tier2_injection_metadata.jsonl",
                    "banks/pair_specs.json", "banks/distractors.json", "generate_tier2.py"):
            self.assertIn(rel, r["absent"])

    def test_tampering_or_missing_system_input_is_an_error(self):
        conv = self.data / "tier2_conversations.jsonl"
        original = conv.read_bytes()
        try:
            conv.write_bytes(original + b"\n")
            self.assertFalse(self.verify()["ok"])
            conv.unlink()
            r = self.verify()
            self.assertFalse(r["ok"])
            self.assertIn("missing", r["error"])
        finally:
            conv.write_bytes(original)
        self.assertTrue(self.verify()["ok"])


class TestBlindPassthroughAndFrozenProjection(unittest.TestCase):
    """The historical passthrough gives distractor turns a hash of their turn id and injected turns their Tier-1 ids, so the
    rule identifies the injected turns (690 recomputable ids, the other 138 are exactly the injected turns). The payload's
    passthrough follows ONE rule for every turn; the payload's FROZEN file is projected to what the runner reads."""

    @classmethod
    def setUpClass(cls):
        cls.files = B.collect_files()

    def rows(self):
        return [json.loads(l) for l in self.files[B.PASSTHROUGH].splitlines()]

    def test_every_turn_follows_the_same_rule(self):
        rows = self.rows()
        self.assertEqual(len(rows), 69 * 12)
        self.assertTrue(all(r["gold_entity_id"] == B.uniform_id(r["turn_id"]) for r in rows))
        self.assertEqual(len({r["gold_entity_id"] for r in rows}), len(rows))

    def test_no_id_follows_the_historical_distractor_rule(self):
        self.assertEqual(sum(r["gold_entity_id"] == B.legacy_distractor_id(r["turn_id"]) for r in self.rows()), 0)

    def test_the_payload_passthrough_is_not_the_historical_file(self):
        hist = REPO / "eval" / "tier2_full_runs" / "t2_passthrough.jsonl"
        if not hist.exists():
            self.skipTest("historical passthrough not present")
        self.assertNotEqual(hashlib.sha256(hist.read_bytes()).hexdigest(), hashlib.sha256(self.files[B.PASSTHROUGH].encode()).hexdigest())

    def test_a_mixed_rule_passthrough_is_refused(self):
        rows = self.rows()
        for r in rows[:5]:
            r["gold_entity_id"] = B.legacy_distractor_id(r["turn_id"])
        files = dict(self.files)
        files[B.PASSTHROUGH] = "".join(json.dumps(r) + "\n" for r in rows)
        with self.assertRaises(B.BuildError):
            B.check_payload(files)

    def test_the_historical_passthrough_would_be_refused(self):
        hist = REPO / "eval" / "tier2_full_runs" / "t2_passthrough.jsonl"
        if not hist.exists():
            self.skipTest("historical passthrough not present")
        files = dict(self.files)
        files[B.PASSTHROUGH] = hist.read_text(encoding="utf-8")
        with self.assertRaises(B.BuildError):
            B.check_payload(files)

    def test_frozen_file_is_the_projection(self):
        payload = json.loads(self.files[B.FROZEN_FILE])
        self.assertEqual(sorted(payload), sorted(B.FROZEN_PROJECTION_KEYS))
        self.assertEqual(payload["status"], "FROZEN")
        self.assertNotRegex(self.files[B.FROZEN_FILE], r"t2[cpq]_\d+")
        repo_full = json.loads((EVAL / "tier2" / "FROZEN_v1.0.json").read_text(encoding="utf-8"))
        pins = dict(payload["sha256"])
        view = payload["blinded_views"][B.QUESTIONS_PIN_KEY]
        # every pin is the frozen file's own, except the questions file: its pin is the blinded view's, and the view records
        # the frozen hash it derives from
        self.assertEqual(pins.pop(B.QUESTIONS_PIN_KEY), view["view_sha256"])
        frozen_pins = dict(repo_full["sha256"])
        self.assertEqual(frozen_pins.pop(B.QUESTIONS_PIN_KEY), view["derived_from_frozen_sha256"])
        self.assertEqual(pins, frozen_pins)
        self.assertEqual(list(payload["blinded_views"]), [B.QUESTIONS_PIN_KEY])
        self.assertEqual(payload["version"], repo_full["version"])

    def test_the_full_frozen_file_is_refused(self):
        files = dict(self.files)
        files[B.FROZEN_FILE] = (EVAL / "tier2" / "FROZEN_v1.0.json").read_text(encoding="utf-8")
        with self.assertRaises(B.BuildError):
            B.check_payload(files)

    def test_qa_id_is_only_echoed_by_the_runner(self):
        # the QA file's ids and per-conversation counts separate same from different pairs; the runner must not read them
        src = self.files["eval/tier2/run_tier2.py"]
        uses = [l.strip() for l in src.splitlines() if "qa_id" in l]
        self.assertEqual(len(uses), 1, uses)
        self.assertIn('"qa_id": q["qa_id"]', uses[0])


class TestBlindedQuestions(unittest.TestCase):
    """The frozen question ids end in _a/_b (two questions: a different pair) or _ab (one combined question: a same pair)."""

    @classmethod
    def setUpClass(cls):
        cls.files = B.collect_files()
        cls.original = (EVAL / "tier2" / "data" / "tier2_qa_questions.jsonl").read_text(encoding="utf-8").replace("\r\n", "\n")

    def rows(self, text):
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def test_the_view_changes_only_the_ids(self):
        orig, view = self.rows(self.original), self.rows(self.files[B.QUESTIONS])
        self.assertEqual(len(orig), len(view))
        for o, v in zip(orig, view):                       # same order, same fields, same values
            self.assertEqual({k: x for k, x in o.items() if k != "qa_id"}, {k: x for k, x in v.items() if k != "qa_id"})
            self.assertEqual(v["qa_id"], B.opaque_qa_id(o["qa_id"]))

    def test_no_id_in_the_payload_carries_the_pair_suffix(self):
        ids = [r["qa_id"] for r in self.rows(self.files[B.QUESTIONS])]
        self.assertTrue(all(re.fullmatch(r"qa_[0-9a-f]{12}", i) for i in ids))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertFalse(re.search(r"t\d+q_\d+_(?:a|b|ab)\b", "\n".join(self.files.values())))

    def test_the_map_inverts_the_view(self):
        qmap = B.qa_id_map()
        self.assertEqual(len(qmap), 111)
        self.assertEqual(sorted(qmap.values()), sorted(r["qa_id"] for r in self.rows(self.original)))
        self.assertEqual(set(qmap), {r["qa_id"] for r in self.rows(self.files[B.QUESTIONS])})

    def test_the_map_is_not_in_the_payload(self):
        self.assertNotIn(B.QA_ID_MAP_FILE, [n.split("/")[-1] for n in self.files])
        for text in self.files.values():
            self.assertNotIn(next(iter(B.qa_id_map().values())), text)       # no frozen qa id anywhere in the payload

    def test_the_unblinded_questions_file_is_refused(self):
        files = dict(self.files)
        files[B.QUESTIONS] = self.original
        with self.assertRaises(B.BuildError):
            B.check_payload(files)

    def test_a_view_that_does_not_record_its_derivation_is_refused(self):
        files = dict(self.files)
        proj = json.loads(files[B.FROZEN_FILE])
        proj["blinded_views"] = {}
        files[B.FROZEN_FILE] = json.dumps(proj)
        with self.assertRaises(B.BuildError):
            B.check_payload(files)


class TestKernelDirectoryFreshness(unittest.TestCase):
    """A built kernel directory is checked against the working tree before it is pushed (``--check``) and is git-ignored."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "kernel_test"
        cls.rc = B.main(["--user", "someone", "--slug", "t2-test", "--out", str(cls.dir)])
        cls.ev = Path(str(cls.dir) + ".evaluator")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self):
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, ignore_errors=True)
        shutil.copytree(self.dir, t / "kernel_x")
        shutil.copytree(self.ev, t / "kernel_x.evaluator")
        return t / "kernel_x"

    @staticmethod
    def ev_of(k):
        return Path(str(k) + ".evaluator")

    @staticmethod
    def _tamper_notebook(d):
        nb = json.loads((d / "dkmem_tier2_smoke.ipynb").read_text(encoding="utf-8"))
        cell = nb["cells"][1]
        cell["source"] = [line.replace('PAYLOAD_SHA256 = "', 'PAYLOAD_SHA256 = "0', 1) for line in cell["source"]]
        (d / "dkmem_tier2_smoke.ipynb").write_text(json.dumps(nb), encoding="utf-8")

    def test_build_wrote_every_artifact(self):
        self.assertEqual(self.rc, 0)
        # the pushed directory holds only the notebook and its metadata; the evaluator-side artifacts sit beside it
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["dkmem_tier2_smoke.ipynb", "kernel-metadata.json"])
        self.assertEqual(sorted(p.name for p in self.ev.iterdir()), ["payload_manifest.json", "qa_id_map.json", "t2_passthrough_used.jsonl"])

    def test_a_fresh_directory_passes(self):
        self.assertEqual(B.main(["--check", str(self.dir)]), 0)

    def test_the_manifest_hashes_describe_the_written_files(self):
        m = json.loads((self.ev / "payload_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["n_files"], 71)      # 70 + eval/tier2/entry_extraction.py: run_tier2 imports it, so every payload with the runner must carry it
        self.assertEqual(hashlib.sha256((self.ev / "t2_passthrough_used.jsonl").read_bytes()).hexdigest(), m["passthrough_sha256"])
        self.assertEqual(hashlib.sha256((self.ev / "qa_id_map.json").read_bytes()).hexdigest(), m["qa_id_map_sha256"])
        nb = (self.dir / "dkmem_tier2_smoke.ipynb").read_text(encoding="utf-8")
        self.assertNotIn(next(iter(json.loads((self.ev / "qa_id_map.json").read_text(encoding="utf-8")).values())), nb)   # frozen ids never reach the notebook

    def test_a_stale_or_tampered_directory_fails(self):
        for mutate in (
            lambda d: (self.ev_of(d) / "payload_manifest.json").write_text(
                json.dumps({**json.loads((self.ev_of(d) / "payload_manifest.json").read_text(encoding="utf-8")), "tree_sha256": "0" * 64}), encoding="utf-8"),
            lambda d: (self.ev_of(d) / "t2_passthrough_used.jsonl").write_bytes((self.ev_of(d) / "t2_passthrough_used.jsonl").read_bytes() + b"x"),
            lambda d: (self.ev_of(d) / "qa_id_map.json").write_text("{}", encoding="utf-8"),
            self._tamper_notebook,
            lambda d: (self.ev_of(d) / "qa_id_map.json").unlink(),
            lambda d: (d / "qa_id_map.json").write_text("{}", encoding="utf-8"),               # an evaluator file in the pushed dir
            lambda d: (d / "extra_notes.txt").write_text("x", encoding="utf-8"),
        ):
            k = self.copy()
            mutate(k)
            self.assertEqual(B.main(["--check", str(k)]), 1)

    def test_a_non_empty_output_directory_is_refused(self):
        self.assertEqual(B.main(["--user", "someone", "--slug", "t2-test", "--out", str(self.dir)]), 1)

    @unittest.skipUnless((REPO / ".git").exists(), "not a git checkout")
    def test_generated_kernel_directories_are_ignored_and_the_builder_is_not(self):
        for path in ("validation/tier2_smoke/kernel_3b_smoke/dkmem_tier2_smoke.ipynb",
                     "validation/tier2_smoke/kernel_3b_smoke.evaluator/qa_id_map.json"):
            self.assertEqual(_git("check-ignore", "-q", path).returncode, 0, path)
        r = _git("check-ignore", "-q", "validation/tier2_smoke/build_kernel.py")
        self.assertEqual(r.returncode, 1)


class TestNotebookCellsExecute(unittest.TestCase):
    """Run the notebook's own cells locally (nvidia-smi and pip stubbed) in clean interpreters, so a cell error is found
    before Kaggle, not on it."""

    @classmethod
    def setUpClass(cls):
        cls.files = B.collect_files()
        cls.convs = B.select_conversations(cls.files)
        cls.nb, cls.payload = B.build_notebook(cls.files, B.build_run_args(cls.convs, 0, 0.9, "m", "r" * 40, "s" * 40), cls.convs)
        cls.cells = ["".join(c["source"]) for c in cls.nb["cells"] if c["cell_type"] == "code"]

    def run_script(self, text, cwd):
        script = Path(cwd) / "cell_runner.py"
        script.write_text(text, encoding="utf-8")
        return subprocess.run([sys.executable, "-I", str(script)], cwd=cwd, capture_output=True, text=True, timeout=300)

    def test_notebook_has_four_code_cells_and_all_compile(self):
        self.assertEqual(len(self.cells), 4)
        for i, src in enumerate(self.cells):
            compile(src, f"cell{i}", "exec")

    def test_unpack_and_forbidden_scan_cells_run_and_leave_a_clean_tree(self):
        with tempfile.TemporaryDirectory() as d:
            src = d.replace("\\", "/") + "/src"
            unpack = (self.cells[0].replace('subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()', '""')
                      .replace('"/kaggle/working/src"', json.dumps(src))
                      .replace('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)', "pass"))
            self.assertNotIn("nvidia-smi", unpack)
            p = self.run_script(unpack + "\n" + self.cells[1] + "\nprint('CELLS_OK')\n", d)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("CELLS_OK", p.stdout)
            self.assertIn("payload tree clean", p.stdout)
            names = {q.name for q in Path(src).rglob("*") if q.is_file()}
            self.assertFalse(names & set(PROTECTED_NAMES))

    def test_scan_cell_fails_if_an_evaluator_only_file_is_present(self):
        with tempfile.TemporaryDirectory() as d:
            src = d.replace("\\", "/") + "/src"
            unpack = (self.cells[0].replace('subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()', '""')
                      .replace('"/kaggle/working/src"', json.dumps(src))
                      .replace('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)', "pass"))
            plant = "\nopen(os.path.join(SRC, 'eval', 'tier2', 'data', 'tier2_gold.jsonl'), 'w').write('{}')\n"
            p = self.run_script(unpack + plant + self.cells[1], d)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("evaluator-only or key material", p.stderr)

    def test_summary_cell_accepts_a_real_pipeline_run_and_a_constant_judge_is_only_an_observation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tree = root / "src"
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(self.payload))) as z:
                z.extractall(tree)
            result = root / "result.json"
            run = subprocess.run([sys.executable, "-I", "-c", _DRIVER, str(tree), str(root / "runs"), str(result), ",".join(self.convs),
                                  str(tree / B.PASSTHROUGH)], cwd=root, capture_output=True, text=True, timeout=300)
            self.assertEqual(json.loads(result.read_text(encoding="utf-8"))["rc"], 0, run.stderr[-1500:])
            cell = self.cells[3].replace("/kaggle/working", str(root).replace("\\", "/"))
            prelude = "import os, json\nGPU_LIST = ''\nARGS = []\nCONVERSATIONS = []\nTREE_SHA256 = ''\n"
            p = self.run_script(prelude + cell + "\nprint('SUMMARY_OK')\n", d)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:] + p.stdout[-1500:])
            summary = json.loads((root / "smoke_summary.json").read_text(encoding="utf-8"))
            used = (root / "t2_passthrough_used.jsonl").read_bytes()
            payload_pt = (tree / B.PASSTHROUGH).read_bytes()
        self.assertTrue(all(summary["acceptance"].values()), summary["acceptance"])
        self.assertEqual(sorted(summary["observations"]), ["judge_not_constant", "judge_output_tokens", "judge_proposals", "pairwise_decisions"])
        self.assertEqual(sorted(summary["acceptance"]), sorted([
            "four_runs", "frozen_prompt_sha", "budget_128", "no_parse_failures", "tokenization_recorded", "budget_not_binding",
            "c_thresholds_null", "tier2_manifests", "evaluator_files_absent", "passthrough_is_the_payload_file", "qa_ids_opaque",
            "questions_read_after_consolidation"]))
        # the exact passthrough the runs were given is copied next to the summary, and its hash is the manifests' hash
        self.assertEqual(hashlib.sha256(used).hexdigest(), summary["passthrough_sha256"])
        self.assertEqual(used, payload_pt)
        for e in summary["runs"].values():
            self.assertEqual(e["passthrough_sha256"], summary["passthrough_sha256"])
            self.assertTrue(e["qa"]["ids_opaque"])
            self.assertEqual(e["tokenizer"]["name_or_path"], "fake/tokenizer")
            if "judge" in e:
                tok = e["judge"]["tokens"]
                self.assertTrue(tok["measured"])
                self.assertEqual(tok["prompt"]["n"], e["judge"]["calls"])
                self.assertEqual(tok["at_budget"], 0)
                self.assertIn("proposals", e["judge"]["decision_distribution"])
        for rid, e in summary["runs"].items():
            if "judge" in e:
                self.assertEqual(e["judge"]["parse_failures"], 0)


if __name__ == "__main__":
    unittest.main()
