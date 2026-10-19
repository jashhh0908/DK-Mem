"""The DRAFT lexicon v2 (dkmem/memory/distinction_features_v2.json, built by tools/build_lexicon_v2.py).

v1 stays untouched and its Hindi entries are carried over unchanged; the Marathi/Gujarati entries are the proposal's, with the (model, not
native) review recorded on each; on the v2.1 dev specs the gate's features are what the pair markers say, except the two documented gaps.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from dkmem.memory.lexicon import load_lexicon  # noqa: E402
from dkmem.memory.matcher import match_distinctions  # noqa: E402

V1 = REPO / "dkmem" / "memory" / "distinction_features.json"
V2 = REPO / "dkmem" / "memory" / "distinction_features_v2.json"
PROPOSAL = REPO / "data" / "team_b_delivery" / "member3_lexicon_proposal" / "proposed_distinction_features.json"
SPECS = REPO / "data" / "tier2_v2_1" / "dev" / "pair_specs_v2_1.json"
KNOWN_UNTAGGED = {"t2s_v2_0119", "t2s_v2_0142"}       # Marathi periphrasis; Gujarati naani held (homograph of 'small/young')


def load_builder():
    spec = importlib.util.spec_from_file_location("_build_lexicon_v2", REPO / "tools" / "build_lexicon_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLexiconV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = json.loads(V1.read_text(encoding="utf-8"))
        cls.v2 = json.loads(V2.read_text(encoding="utf-8"))

    def test_the_file_is_exactly_what_the_builder_derives_from_its_sources(self):
        self.assertEqual(V2.read_bytes(), load_builder().build().encode("utf-8"))
        self.assertNotIn(b"\r", V2.read_bytes())

    def test_it_loads_with_the_production_loader(self):
        lex = load_lexicon(V2)
        self.assertEqual(sorted(lex.langs), ["gu", "hi", "mr"])
        self.assertEqual(len(lex), 33)

    def test_v1_hindi_entries_are_carried_over_unchanged_and_nothing_else_hindi_is_added(self):
        hi = [e for e in self.v2["entries"] if e["lang"] == "hi"]
        self.assertEqual(hi, self.v1["entries"])

    def test_mr_gu_entries_are_the_proposal_with_only_the_review_appended_to_the_notes(self):
        prop = {e["id"]: e for e in json.loads(PROPOSAL.read_text(encoding="utf-8"))["entries"] if e["lang"] in ("mr", "gu")}
        new = {e["id"]: e for e in self.v2["entries"] if e["lang"] in ("mr", "gu")}
        self.assertEqual(set(new), set(prop))
        for i, e in new.items():
            self.assertEqual({k: v for k, v in e.items() if k != "notes"}, {k: v for k, v in prop[i].items() if k != "notes"}, i)
            self.assertTrue(e["notes"].startswith(prop[i]["notes"] + " || MODEL REVIEW (Claude, not native-verified): "), i)

    def test_the_review_status_is_stated_honestly(self):
        d = self.v2["description"]
        self.assertIn("DRAFT", d)
        self.assertIn("NOT by native speakers", d)
        self.assertNotIn("native-reviewed", d.lower().replace("not by native", ""))

    @unittest.skipUnless(SPECS.is_file(), "the v2.1 dev specs are not present (git-ignored, local only)")
    def test_on_the_dev_specs_every_mr_gu_marker_is_tagged_except_the_documented_gaps(self):
        lex = load_lexicon(V2)
        specs = json.loads(SPECS.read_text(encoding="utf-8"))
        specs = specs["specs"] if isinstance(specs, dict) else specs
        seen = set()
        for s in specs:
            if s["language"] not in ("mr", "gu") or s["distinction_class"] not in ("kinship", "honorific_register"):
                continue
            cls = "register" if s["distinction_class"] == "honorific_register" else "kinship"
            fa = match_distinctions(lex, s["text_a"], s["language"]).get(cls)
            fb = match_distinctions(lex, s["text_b"], s["language"]).get(cls)
            if s["spec_id"] in KNOWN_UNTAGGED:
                self.assertTrue(fa is None or fb is None, s["spec_id"])
                seen.add(s["spec_id"])
                continue
            self.assertIsNotNone(fa, s["spec_id"])
            self.assertIsNotNone(fb, s["spec_id"])
            if cls == "kinship":
                self.assertEqual(fa == fb, s["relation"] == "same", s["spec_id"])
            else:
                self.assertNotEqual(fa, fb, s["spec_id"])        # a register pair always carries two different registers
        self.assertEqual(seen, KNOWN_UNTAGGED)

    def test_v1_is_still_hindi_only(self):
        self.assertEqual({e["lang"] for e in self.v1["entries"]}, {"hi"})


if __name__ == "__main__":
    unittest.main()
