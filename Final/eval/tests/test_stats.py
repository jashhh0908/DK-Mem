"""Tests for bootstrap CIs, Cohen's kappa, per-language / class x language tables and seed reporting."""
import unittest

from evaluator import stats
from evaluator.contract_report import groupings, run_tables, signature, strategy_seed_stats
from evaluator.stats import bootstrap_ci, cohens_kappa, mean_sd, seed_for


def row(i, relation, merged, cls="kinship", lang="hi", status="compared"):
    return {"eval_pair_id": f"p{i}", "relation": relation, "merged": merged, "distinction_class": cls,
            "language": lang, "structural_status": status, "outcome": "merged" if merged else "compared_not_merged"}


def rows_example():
    # 10 different (4 merged), 6 same (2 not merged)
    return ([row(i, "different", i < 4) for i in range(10)] +
            [row(10 + i, "same", i >= 2, cls="name_variant") for i in range(6)])


class Bootstrap(unittest.TestCase):
    def test_reproducible_and_label_dependent(self):
        rows = rows_example()
        a = bootstrap_ci(rows, "fcr", label="x", n_boot=500)
        b = bootstrap_ci(rows, "fcr", label="x", n_boot=500)
        self.assertEqual(a, b)
        c = bootstrap_ci(rows, "fcr", label="y", n_boot=500)
        self.assertNotEqual(a["seed"], c["seed"])
        self.assertEqual(seed_for("x", "fcr"), seed_for("x", "fcr"))
        self.assertNotEqual(seed_for("x", "fcr"), seed_for("x", "mcr"))
        self.assertNotEqual(bootstrap_ci(rows, "fcr", label="x", n_boot=500, base_seed=1)["seed"], a["seed"])

    def test_resamples_whole_gold_pairs_within_the_relation(self):
        rows = rows_example()
        originals = {id(r) for r in rows}
        seen = []
        real = stats._METRICS["fcr"]

        def spy(sample):
            seen.append(list(sample))
            return real[0](sample)

        stats._METRICS["fcr"] = (spy, "different")
        try:
            bootstrap_ci(rows, "fcr", label="spy", n_boot=50)
        finally:
            stats._METRICS["fcr"] = real
        self.assertGreaterEqual(len(seen), 50)                      # replicates (plus the point estimate call is not via spy)
        for sample in seen[-50:]:
            self.assertEqual(len(sample), 10)                       # same count as the gold-different pairs
            self.assertTrue(all(id(r) in originals for r in sample))   # whole original pair rows, never new objects
            self.assertTrue(all(r["relation"] == "different" for r in sample))   # relation preserved
        self.assertTrue(any(len({r["eval_pair_id"] for r in s}) < 10 for s in seen[-50:]))   # with replacement

    def test_ci_bounds_and_coverage_of_point(self):
        ci = bootstrap_ci(rows_example(), "fcr", label="b", n_boot=2000)
        self.assertEqual(ci["point"], 0.4)
        self.assertTrue(0 <= ci["lo"] <= ci["point"] <= ci["hi"] <= 1)
        self.assertEqual((ci["n"], ci["n_boot"], ci["confidence"]), (10, 2000, 0.95))

    def test_degenerate_cell_gives_point_interval(self):
        rows = [row(i, "different", False) for i in range(8)]
        ci = bootstrap_ci(rows, "fcr", label="z", n_boot=200)
        self.assertEqual((ci["point"], ci["lo"], ci["hi"]), (0.0, 0.0, 0.0))

    def test_n_zero_is_na(self):
        ci = bootstrap_ci([row(0, "different", True)], "mcr", label="n0", n_boot=100)
        self.assertEqual((ci["point"], ci["lo"], ci["hi"], ci["n"], ci["n_boot"]), (None, None, None, 0, 0))

    def test_mcr_uses_same_pairs_only(self):
        ci = bootstrap_ci(rows_example(), "mcr", label="m", n_boot=300)
        self.assertEqual((ci["n"], ci["point"]), (6, 2 / 6))


class Kappa(unittest.TestCase):
    @staticmethod
    def make(tp, fn, fp, tn):
        return ([row(i, "same", True) for i in range(tp)] + [row(100 + i, "same", False) for i in range(fn)] +
                [row(200 + i, "different", True) for i in range(fp)] + [row(300 + i, "different", False) for i in range(tn)])

    def test_hand_computed(self):
        k = cohens_kappa(self.make(20, 5, 10, 15))
        self.assertEqual(k["n"], 50)
        self.assertAlmostEqual(k["observed_agreement"], 0.7)
        self.assertAlmostEqual(k["expected_agreement"], 0.5)
        self.assertAlmostEqual(k["kappa"], 0.4)

    def test_perfect_and_chance_level(self):
        self.assertAlmostEqual(cohens_kappa(self.make(10, 0, 0, 10))["kappa"], 1.0)
        self.assertAlmostEqual(cohens_kappa(self.make(5, 5, 5, 5))["kappa"], 0.0)
        self.assertLess(cohens_kappa(self.make(0, 10, 10, 0))["kappa"], 0)     # systematic disagreement

    def test_undefined_when_both_raters_constant(self):
        k = cohens_kappa([row(i, "different", False) for i in range(5)])   # e.g. a kinship-only class
        self.assertIsNone(k["kappa"])
        self.assertIn("kappa is 0/0", k["undefined_reason"])
        self.assertEqual(k["observed_agreement"], 1.0)

    def test_empty(self):
        k = cohens_kappa([])
        self.assertEqual((k["n"], k["kappa"], k["undefined_reason"]), (0, None, "no pairs"))

    def test_no_entry_and_not_compared_count_as_not_merged_without_dropping_pairs(self):
        rows = [row(0, "same", True), row(1, "same", False, status="no_entry_both"),
                row(2, "same", False, status="not_compared"), row(3, "different", False)]
        k = cohens_kappa(rows)
        self.assertEqual(k["n"], 4)
        self.assertEqual(k["confusion"]["gold_same_not_merged"], 2)


class MeanSd(unittest.TestCase):
    def test_identical_values_give_zero_sd(self):
        self.assertEqual(mean_sd([0.5, 0.5, 0.5]), {"n_seeds": 3, "mean": 0.5, "sd": 0.0})

    def test_sample_sd_single_value_and_na(self):
        self.assertAlmostEqual(mean_sd([0.0, 1.0])["sd"], 0.7071067811865476)
        self.assertIsNone(mean_sd([0.3])["sd"])
        self.assertEqual(mean_sd([None, None]), {"n_seeds": 0, "mean": None, "sd": None})
        self.assertEqual(mean_sd([None, 0.2, 0.4])["n_seeds"], 2)


class Tables(unittest.TestCase):
    def rows(self):
        return ([row(0, "different", True, "kinship", "hi"), row(1, "different", False, "kinship", "hi"),
                 row(2, "same", False, "name_variant", "hi"), row(3, "same", True, "name_variant", "hi"),
                 row(4, "different", False, "honorific_register", "de"), row(5, "same", True, "honorific_register", "ko")])

    def result(self, seed, rows):
        return {"run_id": f"r{seed}", "strategy": "s", "seed": seed, "backbone": None, "validation_ok": True, "_rows": rows}

    def test_per_language_and_class_x_language_tables(self):
        t = run_tables(self.result(0, self.rows()), 100, {})["tables"]
        self.assertEqual(set(t["per_language"]), {"de", "hi", "ko"})
        hi = t["per_language"]["hi"]
        self.assertEqual((hi["fcr"]["numerator"], hi["fcr"]["denominator"]), (1, 2))
        self.assertEqual((hi["mcr"]["numerator"], hi["mcr"]["denominator"]), (1, 2))
        self.assertEqual(t["per_language"]["de"]["mcr"]["rate"], None)          # n=0 -> n/a, not 0
        self.assertEqual(t["per_language"]["de"]["mcr"]["display"], "n/a")
        self.assertTrue(t["per_language"]["hi"]["fcr"]["indicative"])           # n<20 flag carried through
        self.assertEqual(set(t["class_x_language"]), {"kinship|hi", "name_variant|hi", "honorific_register|de", "honorific_register|ko"})
        self.assertEqual(t["class_x_language"]["kinship|hi"]["mcr"]["denominator"], 0)
        # cells partition the gold pairs
        self.assertEqual(sum(c["n_pairs"] for c in t["class_x_language"].values()), 6)
        self.assertEqual(sum(c["n_pairs"] for c in t["per_class"].values()), 6)
        # CI present only where a denominator exists
        self.assertIsNone(t["per_language"]["de"]["mcr"]["ci"]["lo"])
        self.assertIsNotNone(t["per_language"]["hi"]["fcr"]["ci"]["lo"])

    def test_groupings_keep_all_rows(self):
        g = groupings(self.rows())
        self.assertEqual(len(g["overall"]["all"]), 6)

    def test_zero_variance_seed_reporting_is_explicit(self):
        results = [self.result(s, self.rows()) for s in (0, 1, 2)]
        runs = [run_tables(r, 100, {}) for r in results]
        identical = len({str(signature(r)) for r in results}) == 1
        self.assertTrue(identical)
        st = strategy_seed_stats(runs, identical)
        o = st["cells"]["overall"]["all"]["fcr"]
        self.assertEqual((o["sd"], o["n_seeds"]), (0.0, 3))
        self.assertTrue(st["seeds_decision_identical"] and st["min_seeds_met"])
        self.assertIn("NOT empirical seed variance", st["note"])
        self.assertIsNone(st["cells"]["per_language"]["de"]["mcr"]["mean"])   # n/a cell stays n/a

    def test_differing_seeds_give_empirical_sd_and_fewer_than_3_seeds_flagged(self):
        a, b = self.rows(), self.rows()
        b[0] = row(0, "different", False, "kinship", "hi")
        results = [self.result(0, a), self.result(1, b)]
        runs = [run_tables(r, 100, {}) for r in results]
        st = strategy_seed_stats(runs, len({str(signature(r)) for r in results}) == 1)
        self.assertFalse(st["seeds_decision_identical"])
        self.assertGreater(st["cells"]["overall"]["all"]["fcr"]["sd"], 0)
        self.assertFalse(st["min_seeds_met"])
        self.assertEqual(st["note"], "Seeds differ; SD is empirical.")


if __name__ == "__main__":
    unittest.main()
