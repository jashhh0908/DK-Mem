"""The paper pipeline's own tooling: numbers reproduce the verified report, review summaries count every verdict, the paper checker catches
what "done" forbids. Stored-report cases skip when the (gitignored) report is absent, i.e. for teammates without the artifacts."""

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).parent.parent


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NUM = load("paper/tools/make_numbers.py", "make_numbers")
REV = load("paper/tools/summarize_reviews.py", "summarize_reviews")
CHK = load("tools/check_paper.py", "check_paper")


class TestWilson(unittest.TestCase):
    def test_known_values(self):
        lo, hi = NUM.wilson(15, 42)
        self.assertAlmostEqual(lo, 0.2299, places=3)
        self.assertAlmostEqual(hi, 0.5083, places=3)          # the stored report's interval for 15/42 is [0.2299, 0.5083]
        self.assertEqual(NUM.wilson(0, 27)[0], 0.0)


@unittest.skipUnless(NUM.REPORT.exists(), "verified entity report not present (held by the lead)")
class TestNumbersFromTheStoredReport(unittest.TestCase):
    def setUp(self):
        self.nums = NUM.build(json.loads(NUM.REPORT.read_bytes()), set())

    def test_hindi_and_german_partition_the_pairs_exactly(self):
        for arm in self.nums["arms"].values():
            for e in arm["runs"].values():
                self.assertEqual(e["by_language_n"], {"de": 6, "hi": 63})
                self.assertEqual(e["hindi"]["FCR"]["n"] + 6, e["all"]["FCR"]["n"])
                self.assertEqual(e["hindi"]["MCR"]["n"], e["all"]["MCR"]["n"])

    def test_the_headline_numbers(self):
        r = self.nums["arms"]["Main"]["runs"]
        self.assertEqual((r["Coff"]["hindi"]["FCR"]["k"], r["Clex"]["hindi"]["FCR"]["k"]), (9, 1))
        self.assertEqual((r["Coff"]["hindi"]["MCR"]["k"], r["Clex"]["hindi"]["MCR"]["k"]), (0, 5))

    def test_committed_generated_files_are_current(self):
        self.assertEqual(NUM.main(["--check"]), 0)

    def test_the_committed_exclusion_list_is_the_default_for_writing_and_checking(self):
        # --check and a plain run read the same list, so following the documented workflow never leaves the merge gate stale
        if not NUM.EXCLUSIONS.exists():
            self.skipTest("no paper/review/exclusions.txt")
        self.assertEqual(NUM.main(["--check", "--exclude", str(NUM.EXCLUSIONS)]), 0)
        excluded = {l.split("#")[0].strip() for l in NUM.EXCLUSIONS.read_text(encoding="utf-8").splitlines()} - {""}
        if excluded:
            self.assertEqual(NUM.main(["--check", "--no-exclude"]), 1)      # the committed files carry the sensitivity view

    def test_exclusions_add_a_sensitivity_view_and_never_change_the_primary_numbers(self):
        ex = NUM.build(json.loads(NUM.REPORT.read_bytes()), {"t2p_0002"})
        for arm in ("Main", "Mirror"):
            for tag, e in ex["arms"][arm]["runs"].items():
                self.assertEqual(e["all"], self.nums["arms"][arm]["runs"][tag]["all"])
                self.assertIn("sensitivity_excluding_review_flags", e)

    def test_a_tampered_report_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        bad = tmp / "r.json"
        bad.write_bytes(NUM.REPORT.read_bytes() + b" ")
        with self.assertRaises(SystemExit):
            NUM.main(["--report", str(bad), "--check"])


FIG = load("paper/tools/make_figures.py", "make_figures")
T1 = load("paper/tools/make_tier1_numbers.py", "make_tier1_numbers")


@unittest.skipUnless(NUM.REPORT.exists(), "verified entity report not present (held by the lead)")
class TestFigureNumbers(unittest.TestCase):
    def setUp(self):
        self.c = FIG.cosines(json.loads(NUM.REPORT.read_bytes()))

    def test_every_pair_is_inside_the_drawn_range_and_counts_match(self):
        for arm, d in self.c.items():
            self.assertEqual((len(d["different"]), len(d["same"])), (42, 27))
            self.assertTrue(all(0.40 <= v <= 1.0 for vals in d.values() for v in vals), arm)    # the first draft silently cut off values above 0.90

    def test_the_main_arm_orders_the_relations_the_wrong_way_and_the_mirror_reverses_it(self):
        import statistics
        m, r = self.c["Main"], self.c["Mirror"]
        self.assertGreater(statistics.median(m["different"]), statistics.median(m["same"]))
        self.assertLess(statistics.median(r["different"]), statistics.median(r["same"]))

    def test_above_tau_counts_match_the_stored_false_merges(self):
        self.assertEqual(sum(v > FIG.TAU for v in self.c["Main"]["different"]), 5)       # the 5 false merges of D at tau 0.90
        self.assertEqual(sum(v > FIG.TAU for v in self.c["Main"]["same"]), 0)

    def test_committed_figure_numbers_are_current(self):
        self.assertEqual(FIG.main(["--check"]), 0)


@unittest.skipUnless((T1.SCORED / "score.json").exists(), "Tier-1 key-holder scoring not present (held by the key holder)")
class TestTier1Numbers(unittest.TestCase):
    def setUp(self):
        self.n = T1.build(json.loads((T1.SCORED / "score.json").read_bytes()), json.loads((T1.SCORED / "ablation.json").read_bytes()))

    def test_six_groups_three_modes_and_the_same_63_source_pairs_as_tier2(self):
        self.assertEqual(sorted(self.n["groups"]), ["A", "B", "C", "DEighty", "DEightyFive", "DNinety"])
        for g in self.n["groups"].values():
            self.assertEqual(sorted(g["modes"]), ["lexicon", "lexicon+llm", "off"])
            for e in g["modes"].values():
                self.assertEqual((e["in_scope_hindi"]["FCR"]["n"], e["in_scope_hindi"]["MCR"]["n"]), (36, 27))
                self.assertEqual((e["all"]["FCR"]["n"], e["all"]["MCR"]["n"]), (98, 75))

    def test_the_gate_never_increases_false_merges(self):
        for g in self.n["groups"].values():
            off, on = g["modes"]["off"], g["modes"]["lexicon"]
            for view in ("in_scope_hindi", "all"):
                self.assertLessEqual(on[view]["FCR"]["k"], off[view]["FCR"]["k"])     # veto-only: a gate can only remove merges
                self.assertGreaterEqual(on[view]["MCR"]["k"], off[view]["MCR"]["k"])

    def test_committed_tier1_files_are_current(self):
        self.assertEqual(T1.main(["--check"]), 0)


V2 = load("paper/tools/make_v2_numbers.py", "make_v2_numbers")


@unittest.skipUnless(all((V2.RUN_DIR / n).exists() for n in V2.PINS), "v2 score reports not present (held by the lead)")
class TestV2Numbers(unittest.TestCase):
    def setUp(self):
        self.p, self.k = V2.load_pinned()
        self.n = V2.build(self.p, self.k)

    def test_eight_runs_and_the_126_pairs_split_by_language(self):
        self.assertEqual(sorted(self.n["runs"]), sorted(c + m for c in "ABCD" for m in ("Off", "Lex")))
        self.assertEqual((self.n["pairs"]["different"], self.n["pairs"]["same"]), (68, 58))
        bl = self.n["pairs"]["by_language"]
        self.assertEqual(sum(v["different"] for v in bl.values()), 68)
        self.assertEqual(sum(v["same"] for v in bl.values()), 58)

    def test_the_gate_never_increases_false_merges(self):
        for c in "ABCD":
            off, on = self.n["runs"][c + "Off"], self.n["runs"][c + "Lex"]
            self.assertLessEqual(on["FCR"]["k"], off["FCR"]["k"])                       # veto-only
            self.assertGreaterEqual(on["MCR"]["k"], off["MCR"]["k"])
            self.assertLessEqual(on["kinship_only"]["FCR"]["k"], off["FCR"]["k"])
            self.assertEqual(self.n["contrasts"][f"{c}|different"]["discordant_y_only"], 0)

    def test_the_headline_numbers(self):
        r = self.n["runs"]
        self.assertEqual((r["COff"]["FCR"]["k"], r["CLex"]["FCR"]["k"], r["COff"]["MCR"]["k"], r["CLex"]["MCR"]["k"]), (18, 7, 5, 20))
        self.assertEqual((r["CLex"]["kinship_only"]["FCR"]["k"], r["CLex"]["kinship_only"]["MCR"]["k"]), (7, 11))
        self.assertEqual(r["CLex"]["by_class"]["kinship"]["FCR"]["k"], 0)

    def test_the_outputs_carry_no_per_pair_labels(self):
        text = "".join(V2.render(self.n).values())
        self.assertNotIn("t2p_", text)
        self.assertNotIn("injected_pair", text)

    def test_a_replay_with_another_lexicon_is_refused(self):
        bad = json.loads(json.dumps(self.p))
        bad["inputs"]["lexicon_is_arm_pin"] = False
        with self.assertRaises(AssertionError):
            V2.build(bad, self.k)

    def test_a_tampered_report_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for n in V2.PINS:
            shutil.copyfile(V2.RUN_DIR / n, tmp / n)
        first = next(iter(V2.PINS))
        (tmp / first).write_bytes((tmp / first).read_bytes() + b" ")
        with self.assertRaises(SystemExit):
            V2.load_pinned(tmp)

    def test_committed_v2_files_are_current(self):
        self.assertEqual(V2.main(["--check"]), 0)


class TestReviewSummary(unittest.TestCase):
    def test_every_row_is_counted_and_only_wrong_or_ambiguous_hindi_pairs_are_excluded(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        review, out = tmp / "review", tmp / "generated"
        review.mkdir()
        rows = [("t2p_1", "correct"), ("t2p_2", "wrong"), ("t2p_3", "ambiguous"), ("t2p_4", ""), ("t2p_5", "Correct")]
        with open(review / "hi_pairs_review.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["injected_pair_id", "verdict(correct/wrong/ambiguous)", "comment"])
            w.writerows([(i, v, "") for i, v in rows])
        old = REV.REVIEW, REV.OUT
        REV.REVIEW, REV.OUT = review, out
        try:
            REV.main()
        finally:
            REV.REVIEW, REV.OUT = old
        s = json.loads((out / "review_summary.json").read_text(encoding="utf-8"))["hi_pairs_review.csv"]
        self.assertEqual((s["rows"], s["Correct"], s["Wrong"], s["Ambiguous"], s["Unreviewed"]), (5, 2, 1, 1, 1))
        excl = [l for l in (review / "exclusions.txt").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
        self.assertEqual(excl, ["t2p_2", "t2p_3"])                        # a blank verdict is unreviewed, never "correct", never excluded


class TestPaperChecker(unittest.TestCase):
    def make_paper(self, results_body):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "sections").mkdir()
        (tmp / "generated").mkdir()
        (tmp / "main.tex").write_text("\\newcommand{\\todo}[1]{}\n\\input{sections/results}\n", encoding="utf-8")
        (tmp / "generated" / "numbers.tex").write_text("% GENERATED\n\\newcommand{\\MainCoffFCR}{15/42}\n", encoding="utf-8")
        (tmp / "refs.bib").write_text("@misc{known,\n title={x}}\n", encoding="utf-8")
        (tmp / "sections" / "results.tex").write_text(results_body, encoding="utf-8")
        return tmp

    def run_check(self, paper):
        old = CHK.PAPER
        CHK.PAPER = paper
        try:
            return CHK.main([])
        finally:
            CHK.PAPER = old

    def test_a_clean_paper_passes(self):
        self.assertEqual(self.run_check(self.make_paper("FCR was \\MainCoffFCR{} \\citep{known}.\n")), 0)

    def test_todo_unknown_cite_undefined_macro_and_hand_typed_numbers_fail(self):
        for body in ("\\todo{x}\n", "\\citep{missing}\n", "\\MainCoffFCRtypo{}\n", "FCR was 15/42.\n"):
            self.assertEqual(self.run_check(self.make_paper(body)), 1, body)

    def test_the_real_paper_has_no_undefined_macros_or_unknown_citations(self):
        old = CHK.PAPER
        problems = []
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CHK.main(["--draft"])
        problems = [l for l in buf.getvalue().splitlines() if "undefined macro" in l or "unknown key" in l or "hand-typed" in l
                    or "not included" in l]
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
