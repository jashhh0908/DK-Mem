"""Unit tests of the handoff acceptance checker. Fixtures are SYNTHETIC placeholders (nonsense Latin tokens), not benchmark items and not
candidate test examples; they only exercise the checker's logic."""
import copy
import unittest

from tier2 import handoff_check_v1 as H

HDR = {"schema_version": "1", "revision": 1, "author_id": "x", "language": "hi", "native_language_region": "-", "authored_utc": "-",
       "tools_used": [], "nonce": "a" * 64, "conflict_report": "none flagged; checked all pairs", "attests_no_benchmark_exposure": True, "attests_no_llm_drafting": True}


def _case():
    items = [{"item_id": "ind_aaaaaaaaaaaa", "language": "hi", "text_a": "zork alpha one", "text_b": "blip beta two"}]
    rec = {"item_id": "ind_aaaaaaaaaaaa", "relation": "different", "distinction_class": "kinship", "marker_a": "zork", "marker_b": "blip",
           "fact_a": {"attribute": "t", "value": "alpha"}, "fact_b": {"attribute": "t", "value": "beta"},
           "qa": [{"role": "a", "question": "what for zork", "slots": [["alpha"]], "forbidden": [["beta"]]},
                  {"role": "b", "question": "what for blip", "slots": [["beta"]], "forbidden": [["alpha"]]}],
           "author_confidence": "certain", "source_kind": "original", "conflict": {"flagged": False}}
    return items, {"header": dict(HDR), "records": [rec]}


def codes(rep):
    return {c for _, c in rep["defects"]}


class HandoffCheck(unittest.TestCase):
    def run_check(self, items, key, refs=(), target=1):
        return H.check(items, key, references=list(refs), target=target)

    def test_clean_case_has_no_defects(self):
        items, key = _case()
        rep = self.run_check(items, key)
        self.assertEqual(rep["defects"], [])
        self.assertEqual(rep["shortfalls"].get("hi|different"), None)
        self.assertIn("mr|same", rep["shortfalls"])                      # absent cells are reported as shortfalls, never hidden

    def test_label_bearing_item_field_is_rejected(self):
        items, key = _case()
        items[0]["relation"] = "same"
        self.assertIn("ITEM_KEYS_NOT_LABEL_FREE", codes(self.run_check(items, key)))

    def test_answer_in_other_text_and_question_leak_are_caught(self):
        items, key = _case()
        items[0]["text_b"] = "blip alpha beta"
        self.assertIn("QA_ANSWER_IN_OTHER_TEXT", codes(self.run_check(items, key)))
        items, key = _case()
        key["records"][0]["qa"][0]["question"] = "alpha for zork"
        self.assertIn("QA_QUESTION_CONTAINS_ANSWER", codes(self.run_check(items, key)))

    def test_class_relation_and_ambiguity_rules(self):
        items, key = _case()
        key["records"][0]["distinction_class"] = "name_variant"
        self.assertIn("CLASS_RELATION_NOT_ALLOWED", codes(self.run_check(items, key)))
        items, key = _case()
        key["records"][0]["author_confidence"] = "probable"
        self.assertIn("AMBIGUITY_NOT_EXPLAINED", codes(self.run_check(items, key)))

    def test_conflict_deliverable_is_required_and_consistent(self):
        items, key = _case()
        del key["records"][0]["conflict"]
        self.assertIn("RECORD_KEYS", codes(self.run_check(items, key)))
        items, key = _case()
        key["header"]["conflict_report"] = ""
        self.assertIn("CONFLICT_REPORT_EMPTY", codes(self.run_check(items, key)))
        items, key = _case()
        key["records"][0]["author_confidence"] = "ambiguous"
        key["records"][0]["ambiguity_note"] = "rival reading exists"
        self.assertIn("AMBIGUOUS_NOT_FLAGGED", codes(self.run_check(items, key)))
        items, key = _case()
        key["records"][0]["conflict"] = {"flagged": True}
        self.assertTrue({"CONFLICT_KINDS", "CONFLICT_DESCRIPTION_EMPTY"} <= codes(self.run_check(items, key)))
        key["records"][0]["conflict"] = {"flagged": True, "kinds": ["kinship_term_unclear"], "description": "term maps to two relatives by region"}
        rep = self.run_check(items, key)
        self.assertEqual(rep["defects"], [])
        self.assertIn("FLAGGED_CONFLICT_NEEDS_REVIEWER_RESOLUTION", {c for _, c in rep["warnings"]})

    def test_reviews_must_resolve_every_flagged_conflict(self):
        items, key = _case()
        key["records"][0]["conflict"] = {"flagged": True, "kinds": ["relation_label_unclear"], "description": "two readings defensible here"}
        iid = "ind_aaaaaaaaaaaa"
        ok = {"item_id": iid, "relation_correct": "yes", "conflict_resolution": "label_stands", "verdict": "include", "reason": "reading A is standard"}
        self.assertEqual(H.check_reviews(key, [ok]), [])
        self.assertIn((iid, "FLAGGED_CONFLICT_UNRESOLVED"), H.check_reviews(key, [{**ok, "conflict_resolution": "not_flagged"}]))
        self.assertIn((iid, "REVIEW_REASON_MISSING"), H.check_reviews(key, [{**ok, "reason": ""}]))
        self.assertIn((iid, "REVIEW_EXCLUDED_BUT_INCLUDED"), H.check_reviews(key, [{**ok, "conflict_resolution": "excluded"}]))
        self.assertIn((iid, "REVIEW_MISSING_OR_DUPLICATE"), H.check_reviews(key, []))

    def test_attestation_and_nonce(self):
        items, key = _case()
        key["header"]["attests_no_llm_drafting"] = False
        self.assertIn("ATTESTATION_NOT_TRUE", codes(self.run_check(items, key)))
        items, key = _case()
        key["header"]["nonce"] = "zz"
        self.assertIn("NONCE_FORMAT", codes(self.run_check(items, key)))

    def test_reference_duplicate_is_a_defect_and_similar_frame_a_warning(self):
        items, key = _case()
        ref = {"text_a": "zork alpha one", "text_b": "other words entirely"}
        self.assertIn("TEXT_DUPLICATES_REFERENCE", codes(self.run_check(items, key, [ref])))
        items2 = copy.deepcopy(items)
        items2[0]["text_a"], items2[0]["text_b"] = "zork gamma one", "blip delta two"
        key2 = copy.deepcopy(key)
        key2["records"][0]["qa"][0]["slots"] = [["gamma"]]
        key2["records"][0]["qa"][1]["slots"] = [["delta"]]
        key2["records"][0]["qa"][0]["forbidden"] = [["delta"]]
        key2["records"][0]["qa"][1]["forbidden"] = [["gamma"]]
        rep = self.run_check(items2, key2, [{"text_a": "zork alpha one", "text_b": "blip beta two"}])
        self.assertEqual(rep["defects"], [])
        self.assertIn("FRAME_SIMILAR_TO_REFERENCE", {c for _, c in rep["warnings"]})


if __name__ == "__main__":
    unittest.main()
