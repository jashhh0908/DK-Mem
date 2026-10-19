"""QA alias normalisation v2 (Indic-safe). The frozen v1 normaliser (generate_tier2.normalize_text) keeps word characters plus the
Devanagari block; Gujarati vowel signs are combining marks, not word characters, so v1 deletes them and distinct Gujarati words collide
(મામા and મામી both become "મ મ"). v2 also keeps the Gujarati block and is otherwise identical: for text without Gujarati characters it
returns exactly the v1 string (tests check this on every v1 QA reference and answer).

A QA reference names its normaliser in ``scoring``: SCORING_V1 (generate_tier2.SCORING) is always scored with v1, SCORING_V2 with v2,
so frozen v1 references keep their frozen semantics.
"""
from __future__ import annotations

import re
import unicodedata

from . import generate_tier2 as G

SCORING_V1 = G.SCORING
SCORING_V2 = "all_slots_contained_and_no_forbidden_after_normalisation_v2_indic"

# Devanagari (hi, mr) and Gujarati (gu): letters, vowel signs, virama, nukta and digits of these blocks survive normalisation.
_DROP = re.compile(r"[^\wऀ-ॿ઀-૿]+")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).casefold()
    out: list[str] = []
    for ch in s:
        if unicodedata.combining(ch) and out and out[-1].isascii() and out[-1].isalpha():
            continue                                  # strip Latin diacritics only (as v1)
        out.append(ch)
    s = unicodedata.normalize("NFC", "".join(out))
    s = _DROP.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_alias(answer: str, alias: str) -> bool:
    a, b = normalize_text(answer), normalize_text(alias)
    return bool(b) and f" {b} " in f" {a} "


def score_answer_v2(answer: str, reference: dict) -> dict:
    slots_hit = [any(contains_alias(answer, al) for al in slot) for slot in reference["slots"]]
    forbidden_hit = any(contains_alias(answer, al) for group in reference.get("forbidden", []) for al in group)
    return {"correct": all(slots_hit) and not forbidden_hit, "slots_hit": slots_hit, "forbidden_hit": forbidden_hit}


def score_answer(answer: str, reference: dict) -> dict:
    """Score with the normaliser the reference names; an unknown scoring rule is an error, never a silent default."""
    rule = reference.get("scoring", SCORING_V1)
    if rule == SCORING_V1:
        return G.score_answer(answer, reference)
    if rule == SCORING_V2:
        return score_answer_v2(answer, reference)
    raise ValueError(f"unknown QA scoring rule {rule!r}")
