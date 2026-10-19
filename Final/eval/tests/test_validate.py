"""Focused tests for the Tier 1 validation layer, using tiny in-memory fixtures.

Run from eval/:  python -m unittest discover -s tests -t .
"""
import json
import tempfile
import unittest
from pathlib import Path

from evaluator import jsonschema_lite
from evaluator.loaders import JsonlResult, load_jsonl
from evaluator.validate import (
    ERROR, INFO, WARN, build_gold_index, check_seed_group, validate_run,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "dkmem"   # adapted: dkmem sits next to eval/ in this repo


def gold_rec(n: int, relation="same", cls="kinship", lang="hi") -> dict:
    p = f"ep_{n:012d}"
    return {"eval_pair_id": p, "pair_id": f"t1_{n:04d}", "relation": relation, "distinction_class": cls,
            "language": lang, "utterance_a_id": f"{p}_utt_a", "utterance_b_id": f"{p}_utt_b",
            "opaque_entity_id_a": f"ent_{n:08d}a", "opaque_entity_id_b": f"ent_{n:08d}b"}


def entry(g: dict, side: str, idx=0, **over) -> dict:
    e = {"entry_id": f"{g[f'utterance_{side}_id']}_e{idx}", "source_utterance_id": g[f"utterance_{side}_id"],
         "gold_entity_id": g[f"opaque_entity_id_{side}"], "language": g["language"], "gloss": "x",
         "surface": "x", "distinction": None}
    e.update(over)
    return e


def rec(g: dict, ea=None, eb=None, rid=None, decision="no_merge", **over) -> dict:
    r = {"record_id": rid or f"{g['eval_pair_id']}::r", "run_id": "r1", "strategy": "dk-mem-lexicon",
         "threshold": 0.85, "similarity_score": 0.5, "compatibility": "compatible", "decision": decision,
         "superseded_entry_id": None, "predicted_entity_id": None,
         "entry_a": ea or entry(g, "a"), "entry_b": eb or entry(g, "b")}
    if decision in ("merge", "supersede"):
        r["predicted_entity_id"] = r["entry_a"]["entry_id"]
    r.update(over)
    return r


MANIFEST = {"run_id": "r1", "strategy": "dk-mem-lexicon", "dataset_tier": "tier1_minimal_pairs",
            "seed": 0, "backbone": None, "created_at": "2026-09-30T18:04:15+00:00"}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.g = [gold_rec(1, "same"), gold_rec(2, "different")]
        self.gold = build_gold_index(JsonlResult(records=self.g))

    def run_dir(self, records, manifest=MANIFEST, name="run", raw_lines=None) -> Path:
        d = Path(self.tmp.name) / name
        d.mkdir()
        (d / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        lines = raw_lines if raw_lines is not None else [json.dumps(r, ensure_ascii=False) for r in records]
        (d / "pairwise_eval.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return d

    def validate(self, records, **kw):
        return validate_run(self.run_dir(records, **kw), self.gold, SCHEMA_DIR)

    @staticmethod
    def codes(report, severity=None):
        return {i.code for i in report.issues if severity is None or i.severity == severity}


class GoldTests(unittest.TestCase):
    def test_clean_gold(self):
        idx = build_gold_index(JsonlResult(records=[gold_rec(1), gold_rec(2, "different")]))
        self.assertEqual(idx.errors, [])
        self.assertEqual(len(idx.utt), 4)
        self.assertEqual(idx.distributions["relation"], {"different": 1, "same": 1})

    def test_duplicates_and_bad_fields(self):
        a, b = gold_rec(1), gold_rec(2)
        b["pair_id"] = a["pair_id"]
        bad = gold_rec(3); bad["relation"] = "maybe"
        dup = dict(a)
        missing = gold_rec(4); del missing["language"]
        idx = build_gold_index(JsonlResult(records=[a, b, bad, dup, missing], malformed=[(9, "invalid JSON")]))
        codes = {i.code for i in idx.errors}
        self.assertTrue({"gold_duplicate_pair_id", "gold_bad_relation", "gold_duplicate_eval_pair_id",
                         "gold_missing_field", "gold_malformed_line"} <= codes)


class RunTests(Base):
    def test_clean_run_has_no_errors_or_warnings(self):
        r = self.validate([rec(self.g[0], decision="merge"), rec(self.g[1])])
        self.assertTrue(r.ok)
        self.assertEqual(r.warnings, [])
        self.assertEqual(r.counts["pair_status"], {"compared": 2})

    def test_missing_pair_is_a_normal_outcome_not_a_validation_problem(self):
        r = self.validate([rec(self.g[0])])   # pair 2: no entries at all
        self.assertTrue(r.ok)
        self.assertEqual(r.pair_status[self.g[1]["eval_pair_id"]], "no_entry_both")
        self.assertEqual(self.codes(r, INFO), {"pair_no_entry"})

    def test_not_compared_vs_no_entry_one_side(self):
        a1 = entry(self.g[1], "a")
        # only utt_a of pair 2 produced an entry, compared with itself -> same-utterance record
        same = rec(self.g[1], ea=a1, eb=entry(self.g[1], "a", idx=1), rid="s")
        r = self.validate([rec(self.g[0]), same])
        self.assertEqual(r.pair_status[self.g[1]["eval_pair_id"]], "no_entry_b")
        self.assertIn("same_utterance_record", self.codes(r, WARN))
        # both sides have entries via two same-utterance records but no cross record
        same_b = rec(self.g[1], ea=entry(self.g[1], "b"), eb=entry(self.g[1], "b", idx=1), rid="s2")
        r = self.validate([rec(self.g[0]), same, same_b], name="run2")
        self.assertEqual(r.pair_status[self.g[1]["eval_pair_id"]], "not_compared")

    def test_unknown_utterance_ignored_with_warning(self):
        bad = rec(self.g[0], ea=entry(self.g[0], "a", source_utterance_id="zzz"), rid="u")
        r = self.validate([rec(self.g[0]), rec(self.g[1]), bad])
        self.assertIn("unknown_utterance_id", self.codes(r, WARN))
        self.assertEqual(r.counts["ignored_records"], {"unknown_utterance": 1})
        self.assertEqual(len(r.pair_records[self.g[0]["eval_pair_id"]]), 1)

    def test_cross_episode_ignored_with_warning(self):
        x = rec(self.g[0], eb=entry(self.g[1], "b"), rid="x")
        r = self.validate([x])
        self.assertIn("cross_episode_record", self.codes(r, WARN))
        self.assertEqual(r.pair_records, {})

    def test_reversed_orientation_is_still_aligned(self):
        g = self.g[0]
        r = self.validate([rec(g, ea=entry(g, "b"), eb=entry(g, "a"))])
        self.assertEqual(r.pair_status[g["eval_pair_id"]], "compared")

    def test_duplicate_record_id_and_duplicate_comparison(self):
        a, b = rec(self.g[0], rid="same"), rec(self.g[1], rid="same")
        r = self.validate([a, b, rec(self.g[0], rid="again")])
        self.assertIn("duplicate_record_id", self.codes(r, ERROR))
        self.assertIn("duplicate_comparison", self.codes(r, WARN))

    def test_gold_entity_and_language_mismatch(self):
        g = self.g[0]
        r = self.validate([rec(g, ea=entry(g, "a", gold_entity_id="ent_wrong", language="de"))])
        self.assertIn("gold_entity_id_mismatch", self.codes(r, ERROR))
        self.assertIn("language_mismatch", self.codes(r, WARN))

    def test_entry_id_reused_across_utterances(self):
        g0, g1 = self.g
        shared = entry(g0, "a")["entry_id"]
        r = self.validate([rec(g0), rec(g1, ea=entry(g1, "a", entry_id=shared), rid="y")])
        self.assertIn("entry_id_reused_across_utterances", self.codes(r, ERROR))

    def test_malformed_line_and_schema_violation(self):
        good = json.dumps(rec(self.g[0]))
        broken = rec(self.g[1]); broken["similarity_score"] = 2.0; broken["decision"] = "maybe"
        r = self.validate(None, raw_lines=[good, "{not json", json.dumps(broken), "[1,2]"])
        self.assertIn("pairwise_malformed_line", self.codes(r, ERROR))
        self.assertIn("schema_violation", self.codes(r, ERROR))
        self.assertEqual(next(i for i in r.issues if i.code == "pairwise_malformed_line").count, 2)

    def test_merge_requires_predicted_entity_id(self):
        bad = rec(self.g[0], decision="merge"); bad["predicted_entity_id"] = None
        r = self.validate([bad])
        self.assertIn("schema_violation", self.codes(r, ERROR))

    def test_legacy_distinction_class_is_warn_unknown_class_is_error(self):
        g = self.g[0]
        d = {"class": "evidentiality", "value": "mis", "script": None, "extraction_method": "lexicon"}
        r = self.validate([rec(g, ea=entry(g, "a", distinction=d))])
        self.assertTrue(r.ok)
        self.assertEqual(self.codes(r, WARN), {"legacy_distinction_class_outside_schema_enum"})
        # an unknown class (not one of the 4 known legacy classes) stays a hard error
        d2 = dict(d, **{"class": "made_up"})
        r = self.validate([rec(g, ea=entry(g, "a", distinction=d2))], name="r2")
        self.assertIn("schema_violation", self.codes(r, ERROR))

    def test_record_run_id_and_strategy_must_match_manifest(self):
        r = self.validate([rec(self.g[0], run_id="other", strategy="mem0")])
        self.assertTrue({"record_run_id_mismatch", "record_strategy_mismatch"} <= self.codes(r, ERROR))

    def test_compatibility_consistency_warnings(self):
        r = self.validate([rec(self.g[0], compatibility=None), rec(self.g[1])])
        self.assertIn("compatibility_mixed_null", self.codes(r, WARN))
        r = self.validate([rec(self.g[0], compatibility=None), rec(self.g[1], compatibility=None)], name="r2")
        self.assertIn("compatibility_all_null_for_dkmem", self.codes(r, WARN))

    def test_supersede_pointer_warning(self):
        r = self.validate([rec(self.g[0], decision="supersede", superseded_entry_id="nope")])
        self.assertIn("supersede_id_not_an_entry", self.codes(r, WARN))

    def test_multi_entry_utterance_gives_multiple_cross_records_info(self):
        g = self.g[0]
        r = self.validate([rec(g, rid="1"), rec(g, eb=entry(g, "b", idx=1), rid="2")])
        self.assertIn("pair_multiple_cross_records", self.codes(r, INFO))
        self.assertNotIn("duplicate_comparison", self.codes(r))


class ManifestTests(Base):
    def test_manifest_problems(self):
        bad = dict(MANIFEST, dataset_tier="tier2_injected_sessions", dkmem_mode="lexicon+llm", seed="0")
        r = self.validate([rec(self.g[0])], manifest=bad)
        self.assertTrue({"manifest_wrong_dataset_tier", "manifest_dkmem_mode_mismatch",
                         "manifest_schema_violation"} <= self.codes(r, ERROR))

    def test_backbone_consistency_warnings(self):
        r = self.validate([rec(self.g[0])], manifest=dict(MANIFEST, backbone="qwen"))
        self.assertIn("manifest_lexicon_strategy_with_backbone", self.codes(r, WARN))
        m = dict(MANIFEST, strategy="dk-mem-lexicon-llm")
        r = self.validate([rec(self.g[0], strategy="dk-mem-lexicon-llm")], manifest=m, name="r2")
        self.assertIn("manifest_llm_strategy_without_backbone", self.codes(r, WARN))

    def test_extra_files_in_run_dir_warn(self):
        d = self.run_dir([rec(self.g[0])])
        (d / "notes.txt").write_text("x")
        r = validate_run(d, self.gold, SCHEMA_DIR)
        self.assertIn("run_dir_unexpected_files", self.codes(r, WARN))


class SeedGroupTests(Base):
    def reports(self, n, vary_decision=False, vary_manifest=None):
        out = []
        for s in range(n):
            m = dict(MANIFEST, run_id=f"r{s}", seed=s)
            if vary_manifest and s == n - 1:
                m.update(vary_manifest)
            recs = [rec(self.g[0], run_id=f"r{s}", decision="merge" if vary_decision and s else "no_merge"),
                    rec(self.g[1], run_id=f"r{s}")]
            out.append(self.validate(recs, manifest=m, name=f"run{s}"))
        return out

    def test_three_identical_seeds(self):
        issues = check_seed_group(self.reports(3))
        self.assertEqual({i.code for i in issues}, {"seedgroup_identical_outputs"})
        self.assertTrue(all(i.severity != ERROR for i in issues))

    def test_few_seeds_and_differences_are_warnings_only(self):
        issues = check_seed_group(self.reports(2, vary_decision=True, vary_manifest={"backbone": "q"}))
        codes = {i.code for i in issues}
        self.assertTrue({"seedgroup_fewer_than_min_seeds", "seedgroup_manifest_value_differs",
                         "seedgroup_outputs_differ"} <= codes)
        self.assertNotIn(ERROR, {i.severity for i in issues})

    def test_run_specific_fields_may_differ_silently(self):
        codes = {i.code for i in check_seed_group(self.reports(3))}
        self.assertNotIn("seedgroup_manifest_value_differs", codes)

    def test_duplicate_seed_warns(self):
        reps = self.reports(3)
        reps[2].manifest["seed"] = 0
        self.assertIn("seedgroup_duplicate_seed", {i.code for i in check_seed_group(reps)})


class LoaderAndSchemaTests(unittest.TestCase):
    def test_load_jsonl_collects_malformed_and_keeps_unicode(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.jsonl"
            p.write_text('{"a": "मौसी"}\n\nnot json\n[1]\n', encoding="utf-8")
            res = load_jsonl(p)
        self.assertEqual(res.records, [{"a": "मौसी"}])
        self.assertEqual([n for n, _ in res.malformed], [3, 4])

    def test_schema_lite_basics(self):
        s = {"type": "object", "required": ["x"], "properties": {"x": {"type": ["string", "null"], "enum": ["a", None]}}}
        self.assertEqual(jsonschema_lite.validate({"x": None}, s), [])
        self.assertEqual(len(jsonschema_lite.validate({"x": "b"}, s)), 1)
        self.assertEqual(len(jsonschema_lite.validate({}, s)), 1)
        self.assertEqual(len(jsonschema_lite.validate({"x": 1}, s)), 1)


if __name__ == "__main__":
    unittest.main()
