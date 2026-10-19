"""Code-mixed lexicon resolution, Unicode composition and loader conflict checks.

Run: python -m unittest tests.test_matcher_multilang  (or pytest)
"""

import json
import tempfile
import unittest
from pathlib import Path

from dkmem.memory.lexicon import Lexicon, LexiconEntry, LexiconError, load_lexicon
from dkmem.memory.dkmem_extract import resolve_distinction
from dkmem.memory.matcher import match_distinctions, match_distinctions_multi
from dkmem.pipeline.trace import lexicon_distinction
from dkmem.tier1.extraction import extract_lexicon_only

PRODUCTION_LEXICON = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


def entry(**kw):
    d = dict(id="e", lang="hi", distinction_class="kinship", match_type="exact_word", surface_forms=("x",), value="v", priority=0, notes="")
    d.update(kw)
    d["surface_forms"] = tuple(d["surface_forms"])
    return LexiconEntry(**d)


def conflicting_lexicon(hi_priority=0, mr_priority=0):
    """The same written form means two different relatives in hi and mr (the kind of clash Member 3 documented)."""
    return Lexicon((entry(id="hi_a", lang="hi", surface_forms=("zzz",), value="dada", priority=hi_priority),
                    entry(id="mr_a", lang="mr", surface_forms=("zzz",), value="bhai", priority=mr_priority)))


class TestMultiLanguageResolution(unittest.TestCase):
    def test_result_does_not_depend_on_component_order(self):
        lx = conflicting_lexicon()
        self.assertEqual(match_distinctions_multi(lx, "zzz aaya", ["hi", "mr"]), match_distinctions_multi(lx, "zzz aaya", ["mr", "hi"]))

    def test_the_old_per_component_update_was_order_dependent(self):
        lx = conflicting_lexicon()
        a, b = {}, {}
        for c in ("hi", "mr"):
            a.update(match_distinctions(lx, "zzz", c))
        for c in ("mr", "hi"):
            b.update(match_distinctions(lx, "zzz", c))
        self.assertNotEqual(a, b)                    # documents the defect the pooled resolution removes

    def test_priority_decides_across_languages(self):
        self.assertEqual(match_distinctions_multi(conflicting_lexicon(mr_priority=5), "zzz", ["hi", "mr"]), {"kinship": "bhai"})
        self.assertEqual(match_distinctions_multi(conflicting_lexicon(hi_priority=5), "zzz", ["mr", "hi"]), {"kinship": "dada"})

    def test_single_language_equals_match_distinctions(self):
        lx = load_lexicon(PRODUCTION_LEXICON)
        for text in ("Meri chachi aa rahi hai", "Maa, tum ghar aao", "kuch nahi"):
            self.assertEqual(match_distinctions_multi(lx, text, ["hi"]), match_distinctions(lx, text, "hi"))

    def test_call_sites_agree_for_both_tag_orders(self):
        # the extractor only supports 'L' or 'L-en' tags (derive_lang_profile), so the pipeline-reachable clash is hi vs en
        lx = Lexicon((entry(id="hi_a", lang="hi", surface_forms=("zzz",), value="dada"),
                      entry(id="en_a", lang="en", surface_forms=("zzz",), value="bhai")))
        for fn in (lambda tag: lexicon_distinction(lx, "zzz", tag),
                   lambda tag: extract_lexicon_only("p", "a", "zzz", tag, lx, seed=0).distinction):
            self.assertEqual(fn("hi-en"), fn("en-hi"))
        self.assertEqual(lexicon_distinction(lx, "zzz", "hi-en"), lexicon_distinction(lx, "zzz", "en-hi"))
        # resolve_distinction treats the clash as ambiguous (to be settled by the model) in either order
        self.assertIsNone(resolve_distinction(lx, "zzz", "hi-en", None))
        self.assertIsNone(resolve_distinction(lx, "zzz", "en-hi", None))


class TestUnicodeComposition(unittest.TestCase):
    NUKTA_COMPOSED = "ज़"                      # ज़ as one code point
    NUKTA_DECOMPOSED = "ज़"              # ज + nukta

    def test_decomposed_text_matches_composed_form_and_vice_versa(self):
        lx = Lexicon((entry(surface_forms=(f"{self.NUKTA_COMPOSED}ा",), value="dada"),))
        for text in (f"{self.NUKTA_COMPOSED}ा aaya", f"{self.NUKTA_DECOMPOSED}ा aaya"):
            self.assertEqual(match_distinctions(lx, text, "hi"), {"kinship": "dada"}, repr(text))
        lx2 = Lexicon((entry(surface_forms=(f"{self.NUKTA_DECOMPOSED}ा",), value="dada"),))
        self.assertEqual(match_distinctions(lx2, f"{self.NUKTA_COMPOSED}ा", "hi"), {"kinship": "dada"})

    def test_shipped_lexicon_behaviour_is_unchanged_for_nfc_text(self):
        lx = load_lexicon(PRODUCTION_LEXICON)
        self.assertEqual(match_distinctions(lx, "मेरी चाची दिल्ली आ रही है", "hi"), {"kinship": "chachi"})
        self.assertEqual(match_distinctions(lx, "Meri mausi Pune mein rehti hai", "hi"), {"kinship": "mausi"})


class TestLoaderConflicts(unittest.TestCase):
    def _load(self, entries):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lx.json"
            p.write_text(json.dumps({"schema_version": "1.0", "entries": entries}), encoding="utf-8")
            return load_lexicon(p)

    def _e(self, id, form, value, match_type="exact_word"):
        return {"id": id, "lang": "hi", "distinction_class": "kinship", "match_type": match_type, "surface_forms": [form], "value": value}

    def test_case_variant_with_different_value_is_rejected(self):
        with self.assertRaises(LexiconError):
            self._load([self._e("a", "Aaji", "dadi"), self._e("b", "aaji", "nani")])

    def test_composition_variant_with_different_value_is_rejected(self):
        with self.assertRaises(LexiconError):
            self._load([self._e("a", "ज़ा", "dada"), self._e("b", "ज़ा", "nana")])

    def test_case_variant_with_same_value_is_accepted(self):
        self.assertEqual(len(self._load([self._e("a", "Aaji", "dadi"), self._e("b", "aaji", "dadi")])), 2)

    def test_regex_patterns_are_not_casefolded_for_the_conflict_check(self):
        self.assertEqual(len(self._load([self._e("a", "(?i)^A$", "dadi", "regex"), self._e("b", "^a$", "nani", "regex")])), 2)

    def test_production_lexicon_still_loads(self):
        self.assertEqual(len(load_lexicon(PRODUCTION_LEXICON)), 15)


if __name__ == "__main__":
    unittest.main()
