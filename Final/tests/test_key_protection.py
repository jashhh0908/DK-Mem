"""Protection of Team B's gold answer key (the tier1_eval_contract.md blinding rule).

Everything here works without the key: it checks the *places* the key could leak to, not its content.

* git: the key is ignored (``.gitignore``) and, where a git checkout exists, not tracked;
* the Kaggle notebook builders embed the contents of ``dkmem/`` and ``tests/`` (text, json and jsonl files) and nothing
  else, so no file in those two trees may hold gold labels -- a pair record carrying both a real ``ep_`` id and a
  ``relation``, or an original ``t1_NNNN`` pair id;
* the system code never imports the evaluation package (the key is read only by ``dkmem.eval``, given explicitly);
* no run output or report under ``results/``, ``runs/``, ``outputs/`` or ``validation/`` (JSON or JSONL) carries
  gold labels either.

Run: python -m unittest discover -s tests
"""

import ast
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).parent.parent
EMBEDDED_TREES = ("dkmem", "tests")  # what validation/*/build_kernel.py embeds
EMBEDDED_SUFFIXES = (".py", ".json", ".jsonl", ".md", ".txt")
OUTPUT_TREES = ("results", "runs", "outputs", "validation")  # where run outputs, scores and reports are written
SYSTEM_PACKAGES = ("backends", "baselines", "memory", "pipeline", "store", "tier1")

REAL_EP_ID = re.compile(r"ep_[0-9a-f]{12}")
RELATION = re.compile(r"\"(?:gold_)?relation\"\s*:\s*\"(?:same|different)\"")
ORIGINAL_PAIR_ID = re.compile(r"\bt1_\d{4}\b")


def is_source_checkout(repo: Path = REPO) -> bool:
    """True for the Team A source repository (a git checkout, or a copy that still has ``validation/``), False for
    the Kaggle copy, which embeds only ``dkmem/`` and ``tests/``."""
    return (repo / ".git").exists() or (repo / "validation").is_dir()


def gitignore_status(repo: Path = REPO) -> tuple[str, str]:
    """``("ok" | "fail" | "skip", why)``: does ``.gitignore`` keep Team B's key out of git?

    The check may be skipped only where there is no source repository at all (the Kaggle payload has no
    ``.gitignore``). In a source checkout a missing or incomplete ``.gitignore`` is a failure: skipping there would
    silently remove the protection this test exists for.
    """
    gi = repo / ".gitignore"
    if not gi.is_file():
        if is_source_checkout(repo):
            return "fail", ".gitignore is missing from a source checkout: the answer key would no longer be ignored"
        return "skip", "no .gitignore and not a source checkout (the Kaggle copy embeds only dkmem/ and tests/)"
    lines = {l.strip() for l in gi.read_text(encoding="utf-8").splitlines()}
    missing = [p for p in ("teamB_answer_key.jsonl", "*answer_key*.jsonl") if p not in lines]
    if missing:
        return "fail", f".gitignore lacks {missing}"
    return "ok", ""


def embedded_files():
    for top in EMBEDDED_TREES:
        for path in (REPO / top).rglob("*"):
            if path.is_file() and path.suffix in EMBEDDED_SUFFIXES and "__pycache__" not in path.parts:
                yield path


class TestKeyIsNotCommitted(unittest.TestCase):
    def test_gitignore_covers_the_key_wherever_it_is_placed(self):
        status, why = gitignore_status()
        if status == "skip":
            self.skipTest(why)
        self.assertEqual(status, "ok", why)

    @unittest.skipUnless((REPO / ".git").exists(), "not a git checkout (e.g. the Kaggle copy)")
    def test_no_tracked_file_is_an_answer_key(self):
        try:
            out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git is not available")
        self.assertEqual([f for f in out.splitlines() if "answer_key" in f.lower()], [])

    @unittest.skipUnless((REPO / ".git").exists(), "not a git checkout (e.g. the Kaggle copy)")
    def test_git_reports_the_key_path_as_ignored(self):
        try:
            r = subprocess.run(["git", "check-ignore", "-q", "teamB_answer_key.jsonl"], cwd=REPO, timeout=30)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git is not available")
        self.assertEqual(r.returncode, 0)


class TestGitignoreRule(unittest.TestCase):
    """The rule behind the check above, on synthetic layouts (so the Kaggle case is tested without Kaggle)."""

    GOOD = "# comment\nteamB_answer_key.jsonl\n*answer_key*.jsonl\n"

    def layout(self, gitignore=None, git=False, validation=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        if gitignore is not None:
            (root / ".gitignore").write_text(gitignore, encoding="utf-8")
        if git:
            (root / ".git").mkdir()
        if validation:
            (root / "validation").mkdir()
        return root

    def test_complete_gitignore_passes(self):
        self.assertEqual(gitignore_status(self.layout(self.GOOD, git=True)), ("ok", ""))
        self.assertEqual(gitignore_status(self.layout(self.GOOD)), ("ok", ""))

    def test_missing_gitignore_is_a_failure_in_a_source_checkout(self):
        for kwargs in ({"git": True}, {"validation": True}, {"git": True, "validation": True}):
            with self.subTest(layout=kwargs):
                self.assertEqual(gitignore_status(self.layout(None, **kwargs))[0], "fail")

    def test_missing_gitignore_is_skipped_only_in_the_kaggle_layout(self):
        status, why = gitignore_status(self.layout(None))
        self.assertEqual(status, "skip")
        self.assertIn("Kaggle", why)

    def test_incomplete_gitignore_fails_everywhere(self):
        for text in ("", "teamB_answer_key.jsonl\n", "*answer_key*.jsonl\n",
                     "# teamB_answer_key.jsonl\n*answer_key*.jsonl\n"):
            for kwargs in ({}, {"git": True}):
                with self.subTest(text=text, layout=kwargs):
                    self.assertEqual(gitignore_status(self.layout(text, **kwargs))[0], "fail")

    def test_this_repository_is_recognised_as_a_source_checkout_where_it_should_be(self):
        if (REPO / "validation").is_dir() or (REPO / ".git").exists():
            self.assertTrue(is_source_checkout())
        else:
            self.assertFalse(is_source_checkout())


class TestNothingEmbeddedHoldsGoldLabels(unittest.TestCase):
    def test_no_file_named_like_a_key_is_in_an_embedded_tree(self):
        self.assertEqual([str(p.relative_to(REPO)) for p in embedded_files() if "answer_key" in p.name.lower()], [])

    def test_no_embedded_file_holds_a_gold_record_or_an_original_pair_id(self):
        offenders = []
        for p in embedded_files():
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if (REAL_EP_ID.search(line) and RELATION.search(line)) or ORIGINAL_PAIR_ID.search(line):
                    offenders.append(f"{p.relative_to(REPO)}:{n}")
        self.assertEqual(offenders, [])

    def test_the_builders_embed_only_dkmem_and_tests(self):
        builders = list((REPO / "validation").glob("*/build_kernel.py"))
        if is_source_checkout():
            self.assertTrue(builders, "no validation/*/build_kernel.py found: this guard would pass vacuously")
        for builder in builders:
            src = builder.read_text(encoding="utf-8")
            if "for top in" in src:
                self.assertIn('for top in ("dkmem", "tests")', src, builder)

    def test_run_outputs_carry_no_gold_labels(self):
        # every place run outputs and reports land (gitignored or committed), JSON and JSONL alike
        offenders = []
        for top in OUTPUT_TREES:
            for p in (REPO / top).rglob("*"):
                if not p.is_file() or p.suffix not in (".json", ".jsonl"):
                    continue
                if "src" in p.relative_to(REPO / top).parts:  # embedded source copies
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                if (RELATION.search(text) and REAL_EP_ID.search(text)) or ORIGINAL_PAIR_ID.search(text):
                    offenders.append(str(p.relative_to(REPO)))
        self.assertEqual(offenders, [])


class TestSystemCodeNeverReadsTheKey(unittest.TestCase):
    def test_system_packages_do_not_import_the_evaluation_package(self):
        offenders = []
        for pkg in SYSTEM_PACKAGES:
            for path in (REPO / "dkmem" / pkg).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    if any(n == "dkmem.eval" or n.startswith("dkmem.eval.") for n in names):
                        offenders.append(f"{path.relative_to(REPO)}: {names}")
        self.assertEqual(offenders, [])

    def test_the_evaluation_package_is_the_only_reader_of_a_key(self):
        readers = []
        for path in (REPO / "dkmem").rglob("*.py"):
            rel = path.relative_to(REPO / "dkmem").parts
            if rel[0] == "eval":
                continue
            if "answer_key" in path.read_text(encoding="utf-8").lower() and "tier1/io.py" != "/".join(rel):
                readers.append("/".join(rel))
        self.assertEqual(readers, [])  # tier1/io.py only names the file in a docstring


if __name__ == "__main__":
    unittest.main()
