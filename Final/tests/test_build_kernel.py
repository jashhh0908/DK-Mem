"""Tests for validation/abcd_first_run/build_kernel.py: the zip packaging of the Kaggle notebook, its integrity and
size checks, and that the command it runs is the frozen one.

The builder is imported and driven in-process (``main(argv)``) into temporary directories; nothing is pushed and no
GPU or network is used. The whole file is skipped where ``validation/`` is absent (the Kaggle copy embeds only
``dkmem/`` and ``tests/``). Strings that look like gold records are assembled at run time so this file itself never
contains one (tests/test_key_protection.py scans every embedded file).

Run: python -m unittest discover -s tests
"""

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).parent.parent
BUILDER = REPO / "validation" / "abcd_first_run" / "build_kernel.py"

MODEL_REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EMB_REV = "5617a9f61b028005a4858fdac845db406aefb181"
GUARD_ARGV = ["--user", "someone", "--slug", "dk-mem-abcd-3b-guard", "--model-id", "Qwen/Qwen2.5-3B-Instruct",
              "--revision", MODEL_REV, "--embedding-revision", EMB_REV, "--seed", "0", "--tau-d", "0.85",
              "--limit", "10", "--configs", "A", "B", "C", "D", "--modes", "off", "lexicon", "lexicon+llm"]
# What the frozen builder produced for this command (checked against the notebook that ran on Kaggle).
GUARD_CLI_ARGS = ["--configs", "A", "B", "C", "D", "--modes", "off", "lexicon", "lexicon+llm", "--model-id",
                  "Qwen/Qwen2.5-3B-Instruct", "--seed", "0", "--out", "/kaggle/working/runs", "--tau-d", "0.85",
                  "--revision", MODEL_REV, "--limit", "10", "--embedding-revision", EMB_REV]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_kernel_abcd_under_test", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(bk, argv, out):
    """``(exit code, stdout, stderr)`` of ``bk.main(argv + --out out)``; argparse errors become exit code 2."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = bk.main([*argv, "--out", str(out)])
        except SystemExit as e:
            code = e.code
    return code, stdout.getvalue(), stderr.getvalue()


def notebook(out):
    return json.loads((Path(out) / "dkmem_abcd_run.ipynb").read_text(encoding="utf-8"))


def cell_source(nb, i):
    return "".join(nb["cells"][i]["source"])


def payload_of(nb):
    return re.search(r'^PAYLOAD = "([A-Za-z0-9+/=]+)"', cell_source(nb, 1), re.M).group(1)


def working_tree_files(bk):
    return bk.collect_files()


@unittest.skipUnless(BUILDER.is_file(), "validation/ is not present (the Kaggle copy)")
class TestBuiltNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bk = load_builder()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "guard"
        cls.code, cls.stdout, cls.stderr = run_main(cls.bk, GUARD_ARGV, cls.out)
        cls.nb = notebook(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_succeeds_and_reports_sizes(self):
        self.assertEqual((self.code, self.stderr), (0, ""))
        self.assertRegex(self.stdout, r"payload [\d,]+ B, notebook [\d,]+ B \(limit 1,000,000\) \| tree sha256 [0-9a-f]{16}")

    def test_the_command_is_the_frozen_a_to_d_command(self):
        ns = {}
        exec(re.search(r"^ARGS = .*$", cell_source(self.nb, 3), re.M).group(0), ns)
        self.assertEqual(ns["ARGS"], GUARD_CLI_ARGS)
        self.assertIn('"-m", "dkmem.pipeline.cli"', cell_source(self.nb, 3))

    def test_metadata_requests_a_private_gpu_internet_notebook(self):
        meta = json.loads((self.out / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (meta["id"], meta["is_private"], meta["enable_gpu"], meta["enable_internet"], meta["machine_shape"],
             meta["code_file"]),
            ("someone/dk-mem-abcd-3b-guard", True, True, True, "NvidiaTeslaT4", "dkmem_abcd_run.ipynb"),
        )
        self.assertEqual(len(self.nb["cells"]), 5)

    def test_notebook_is_far_below_the_size_kaggle_rejected(self):
        size = (self.out / "dkmem_abcd_run.ipynb").stat().st_size
        self.assertLessEqual(size, self.bk.MAX_NOTEBOOK_BYTES)
        self.assertLess(size, 1_566_345)  # the plain-text notebook Kaggle answered with HTTP 400
        plain = sum(len(t.encode("utf-8")) for t in working_tree_files(self.bk).values())
        self.assertLess(size, plain / 2, "the zip payload should be much smaller than the plain text it carries")

    def test_payload_is_a_zip_of_exactly_the_working_tree(self):
        files = working_tree_files(self.bk)
        payload = payload_of(self.nb)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload, validate=True))) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(sorted(z.namelist()), sorted(files))
            for name in z.namelist():
                self.assertEqual(z.read(name).decode("utf-8"), files[name], name)
        src = cell_source(self.nb, 1)
        self.assertEqual(re.search(r'^PAYLOAD_SHA256 = "([0-9a-f]{64})"', src, re.M).group(1),
                         hashlib.sha256(payload.encode("ascii")).hexdigest())
        manifest = json.loads(json.loads(re.search(r"^MANIFEST = json\.loads\((.*)\)  #", src, re.M).group(1)))
        self.assertEqual(manifest, {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in files.items()})

    def test_only_dkmem_and_tests_are_embedded_and_no_key_material(self):
        names = sorted(working_tree_files(self.bk))
        self.assertEqual({n.split("/")[0] for n in names}, {"dkmem", "tests"})
        self.assertEqual([n for n in names if "answer_key" in n.lower()], [])
        self.assertIn("dkmem/pipeline/runner.py", names)
        self.assertIn("dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl", names)

    def test_the_same_tree_and_arguments_give_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, _ = run_main(self.bk, GUARD_ARGV, Path(tmp) / "again")
            self.assertEqual(code, 0)
            self.assertEqual((Path(tmp) / "again" / "dkmem_abcd_run.ipynb").read_bytes(),
                             (self.out / "dkmem_abcd_run.ipynb").read_bytes())


@unittest.skipUnless(BUILDER.is_file(), "validation/ is not present (the Kaggle copy)")
class TestUnpackCellVerifiesTheIntegrity(unittest.TestCase):
    """Execute the notebook's first cell (Kaggle-only calls stubbed) and tamper with what it carries."""

    @classmethod
    def setUpClass(cls):
        cls.bk = load_builder()
        cls.tmp = tempfile.TemporaryDirectory()
        code, _, err = run_main(cls.bk, GUARD_ARGV, Path(cls.tmp.name) / "nb")
        assert code == 0, err
        cls.source = cell_source(notebook(Path(cls.tmp.name) / "nb"), 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cell(self, source):
        with tempfile.TemporaryDirectory() as dest:
            code = source.replace('SRC = "/kaggle/working/src"', f"SRC = {str(Path(dest) / 'src')!r}")
            code = re.sub(r"^GPU_LIST = subprocess\.run\(.*\)$", 'GPU_LIST = ""', code, flags=re.M)
            ns: dict = {}
            with contextlib.redirect_stdout(io.StringIO()) as out:
                exec(code, ns)
            written = sorted(p.relative_to(Path(dest) / "src").as_posix() for p in (Path(dest) / "src").rglob("*") if p.is_file())
            return ns, out.getvalue(), written

    def test_clean_payload_unpacks_and_is_reported_verified(self):
        ns, out, written = self.run_cell(self.source)
        self.assertEqual(written, sorted(ns["FILES"]))
        self.assertIn("payload verified, tree sha256", out)
        self.assertIn("dkmem/pipeline/cli.py", written)

    def test_a_changed_payload_is_caught_before_anything_is_written(self):
        payload = re.search(r'^PAYLOAD = "([A-Za-z0-9+/=]+)"', self.source, re.M).group(1)
        mid = len(payload) // 2
        flipped = payload[:mid] + ("A" if payload[mid] != "A" else "B") + payload[mid + 1:]
        with self.assertRaises(AssertionError) as cm:
            self.run_cell(self.source.replace(payload, flipped))
        self.assertIn("payload changed in transit", str(cm.exception))

    def test_a_corrupt_zip_with_a_matching_checksum_is_still_caught(self):
        payload = re.search(r'^PAYLOAD = "([A-Za-z0-9+/=]+)"', self.source, re.M).group(1)
        raw = bytearray(base64.b64decode(payload))
        raw[len(raw) // 3] ^= 0xFF  # damage a member's compressed bytes
        bad = base64.b64encode(bytes(raw)).decode("ascii")
        source = self.source.replace(payload, bad).replace(
            re.search(r'^PAYLOAD_SHA256 = "([0-9a-f]{64})"', self.source, re.M).group(1),
            hashlib.sha256(bad.encode("ascii")).hexdigest(),
        )
        with self.assertRaises(Exception) as cm:  # CRC failure, bad zlib stream or a manifest mismatch
            self.run_cell(source)
        self.assertNotIsInstance(cm.exception, KeyboardInterrupt)

    def test_a_file_that_differs_from_the_manifest_is_caught(self):
        m = re.search(r"^MANIFEST = json\.loads\((.*)\)  #", self.source, re.M)
        manifest = json.loads(json.loads(m.group(1)))
        name = next(iter(sorted(manifest)))
        manifest[name] = "0" * 64
        tampered = self.source.replace(m.group(1), json.dumps(json.dumps(manifest, sort_keys=True)))
        with self.assertRaises(AssertionError) as cm:
            self.run_cell(tampered)
        self.assertIn("differs from the manifest", str(cm.exception))


@unittest.skipUnless(BUILDER.is_file(), "validation/ is not present (the Kaggle copy)")
class TestRefusals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bk = load_builder()
        cls.files = cls.bk.collect_files()

    def mutated(self, **edits):
        files = dict(self.files)
        for name, text in edits.items():
            files[name.replace("__", "/")] = text
        return files

    def test_the_working_tree_passes_the_payload_checks(self):
        self.bk.check_payload(self.files)

    def test_an_answer_key_file_is_refused(self):
        files = dict(self.files)
        files["dkmem/data/tier1/eval_input/team" + "B_answer" + "_key.jsonl"] = "{}\n"
        with self.assertRaises(self.bk.BuildError) as cm:
            self.bk.check_payload(files)
        self.assertIn("answer-key", str(cm.exception))

    def test_a_gold_record_or_original_pair_id_is_refused(self):
        record = '{"eval_pair_id": "ep_' + "0123456789ab" + '", "relation": "same"}\n'
        for leak in (record, "pair " + "t1_" + "0007" + "\n"):
            with self.subTest(leak=leak[:12]):
                files = dict(self.files)
                files["tests/test_extra.py"] = "# notes\n" + leak
                with self.assertRaises(self.bk.BuildError) as cm:
                    self.bk.check_payload(files)
                self.assertIn("blinding", str(cm.exception))

    def test_files_outside_dkmem_and_tests_are_refused(self):
        for name in ("validation/x.py", "../dkmem/x.py", "results/run.json"):
            files = dict(self.files)
            files[name] = "x = 1\n"
            with self.subTest(name=name), self.assertRaises(self.bk.BuildError):
                self.bk.check_payload(files)

    def test_carriage_returns_are_refused(self):
        files = dict(self.files)
        files["dkmem/config.py"] = files["dkmem/config.py"] + "\r\n"
        with self.assertRaises(self.bk.BuildError) as cm:
            self.bk.check_payload(files)
        self.assertIn("carriage", str(cm.exception))

    def test_a_missing_required_file_is_refused(self):
        files = {k: v for k, v in self.files.items() if k != "dkmem/pipeline/runner.py"}
        with self.assertRaises(self.bk.BuildError) as cm:
            self.bk.check_payload(files)
        self.assertIn("pipeline/runner.py", str(cm.exception))

    def test_an_input_that_does_not_match_the_pinned_hash_is_refused(self):
        files = dict(self.files)
        name = "dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl"
        files[name] = files[name].replace("hi", "ho", 1)
        with self.assertRaises(self.bk.BuildError) as cm:
            self.bk.check_payload(files)
        self.assertIn("pinned", str(cm.exception))

    def test_a_payload_that_does_not_decode_to_its_files_is_refused(self):
        payload = self.bk.make_payload(self.files)
        self.bk.verify_payload(payload, self.files)
        other = dict(self.files)
        other["dkmem/config.py"] = other["dkmem/config.py"] + "# changed\n"
        with self.assertRaises(self.bk.BuildError):
            self.bk.verify_payload(payload, other)
        with self.assertRaises(self.bk.BuildError):
            self.bk.verify_payload("not-base64!", self.files)
        with self.assertRaises(self.bk.BuildError):
            self.bk.verify_payload(base64.b64encode(b"not a zip").decode("ascii"), self.files)

    def test_main_refuses_an_oversized_notebook_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = run_main(self.bk, [*GUARD_ARGV, "--max-notebook-bytes", "1000"], Path(tmp) / "x")
            self.assertEqual(code, 1)
            self.assertIn("limit", err)
            self.assertFalse((Path(tmp) / "x").exists())

    def test_main_refuses_when_the_payload_check_fails(self):
        bad = dict(self.files)
        bad["tests/test_extra.py"] = "# " + "t1_" + "0042" + "\n"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.bk, "collect_files", return_value=bad):
            code, _, err = run_main(self.bk, GUARD_ARGV, Path(tmp) / "x")
            self.assertEqual(code, 1)
            self.assertIn("refusing to write", err)
            self.assertFalse((Path(tmp) / "x").exists())

    def test_argument_errors_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            no_tau = [a for a in GUARD_ARGV]
            i = no_tau.index("--tau-d")
            del no_tau[i:i + 2]
            self.assertEqual(run_main(self.bk, no_tau, Path(tmp) / "a")[0], 2)
            self.assertEqual(run_main(self.bk, ["--user", "u", "--slug", "s", "--baseline", "mem0", "--tau-d", "0.5"],
                                      Path(tmp) / "b")[0], 2)

    def test_baseline_command_is_the_baselines_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_main(self.bk, ["--user", "u", "--slug", "s", "--baseline", "flat", "--seed", "0",
                                              "--embedding-revision", EMB_REV], Path(tmp) / "f")
            self.assertEqual(code, 0)
            ns = {}
            exec(re.search(r"^ARGS = .*$", cell_source(notebook(Path(tmp) / "f"), 3), re.M).group(0), ns)
            self.assertEqual(ns["ARGS"], ["flat", "--seed", "0", "--out", "/kaggle/working/runs",
                                          "--embedding-revision", EMB_REV])
            self.assertIn("python -m dkmem.baselines.cli", out)


if __name__ == "__main__":
    unittest.main()
