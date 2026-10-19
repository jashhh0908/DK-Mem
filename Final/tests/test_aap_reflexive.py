"""Hindi ``aap`` and the reflexive "apne aap" ("by itself"): what the matcher does today, and that no benchmark result is affected.

Expected behaviour per the existing protocol: ``dkmem/memory/distinction_features.md`` states that the matcher is "deliberately
context-free" and that anything needing sentence-level context to disambiguate is out of scope. The v1 lexicon entry ``hi-register-aap``
is documented as the second-person pronoun; it matches the token ``aap`` wherever it occurs, including inside the reflexive
(``apne aap``, where ``aap`` means "self" and is not the polite pronoun; recorded by Member 3 as a false match, probe ``fp04``).

These tests pin that behaviour and measure its reach. They do NOT assert it is right: they exist so that any change (a negative-context
rule, a new lexicon version) is deliberate, needs native-speaker review and a new lexicon hash, and cannot silently alter frozen results.
The v1 lexicon file is frozen (its sha256 is recorded by every run), so it is not edited here.
"""

import glob
import json
import re
import unittest
from pathlib import Path

from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.matcher import find_matches, match_distinctions

REPO = Path(__file__).parent.parent
LEXICON = REPO / "dkmem" / "memory" / "distinction_features.json"


class TestAapMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lx = load_lexicon(LEXICON)

    def tokens(self, text):
        return [m.token for m in find_matches(self.lx, text, "hi") if m.entry.id == "hi-register-aap"]

    def test_the_polite_pronoun_is_tagged_in_both_scripts_and_cases(self):
        for text in ("aap kaise hain", "Aap kaise hain", "आप कैसे हैं", "Maa, aap somvar ko aayengi"):
            self.assertEqual(match_distinctions(self.lx, text, "hi"), {"register": "aap"}, text)

    def test_the_entry_is_a_context_free_exact_word_match(self):
        entry = next(e for e in self.lx if e.id == "hi-register-aap")
        self.assertEqual((entry.match_type, entry.value, entry.distinction_class), ("exact_word", "aap", "register"))
        self.assertEqual(sorted(entry.surface_forms), ["aap", "आप"])

    def test_the_reflexive_is_currently_tagged_as_the_polite_pronoun_a_documented_known_limitation(self):
        # KNOWN FALSE MATCH (context-free by design). Changing this needs a lexicon/matcher decision + native review; see module docstring.
        for text in ("woh apne aap ghar chala gaya", "वह अपने आप आ गया", "Apni baat apne aap kehni chahiye"):
            self.assertEqual(match_distinctions(self.lx, text, "hi"), {"register": "aap"}, text)
            self.assertTrue(self.tokens(text), text)

    def test_each_occurrence_is_independent_there_is_no_phrase_level_state(self):
        self.assertEqual(len(self.tokens("aap aap kehte rahe")), 2)

    def test_unrelated_words_containing_aap_are_not_matched(self):
        for text in ("aapka ghar", "aapne dekha", "paap kiya", "baap ka naam"):      # exact_word: whole token only (inflections are NOT in v1)
            self.assertEqual(self.tokens(text), [], text)


def _strings():
    def walk(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, dict):
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    files = (glob.glob(str(REPO / "dkmem/data/**/*.jsonl"), recursive=True) + glob.glob(str(REPO / "eval/tier2/data/*.jsonl"))
             + glob.glob(str(REPO / "eval/tier2/mirror_v1_1/data/*.jsonl")))
    out = []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.extend(walk(json.loads(line)))
    return out


STRINGS = _strings()
REFLEXIVE = re.compile(r"(?<!\w)(apne|apni|apna|अपने|अपनी|अपना)\s+(aap|आप)(?!\w)", re.I)


@unittest.skipUnless(len(STRINGS) > 10000, "frozen benchmark data (Tier-1 input, Tier-2 v1.0 and mirror) not present")
class TestNoStoredBenchmarkTextHitsTheReflexive(unittest.TestCase):
    def test_no_frozen_utterance_contains_the_reflexive(self):
        hits = [s for s in STRINGS if REFLEXIVE.search(s)]
        self.assertEqual(hits, [], "a frozen benchmark text contains the reflexive: lexicon results on it would carry the false match")
