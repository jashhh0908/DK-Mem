"""Kinship naming diagnostic for the frozen Config C v2 judge: build cases_v1.jsonl and review_sheet.csv (EVALUATOR side).

    python -m tier2.kinship_probe.make_cases [--check]        # --check: verify the files equal what this script builds

Question tested (diagnostic, not a judge evaluation): does merge_judge_entity_t2_v2 on Qwen2.5-7B merge two DIFFERENT Hindi
kinship terms when its reason names both relatives with the same English word, and does sentence structure change that?

Groups (72 cases; every case is judged in both orders):
  H-diff        different kinship terms, no names, Roman Hinglish (12 term pairs x 3 structure cells)      label keep_both   36
  H-same-named  the same named relative (same kinship term AND same first name) in both sentences       label merge       18
  E-explicit    English with the relation spelled out ("my father's mother" vs "my mother's mother")   label keep_both    6
  E-collapsed   English with the distinction removed ("my grandmother" twice)                          no label (ref.)    6
  possessive    as H-diff P-same but without "Meri"/"Mere" ("Dadi ji ..." / "Nani ji ...")             label keep_both    6
Structure cells: P-same (identical frame, same attribute, different value), N-same (different frames, same attribute),
N-diff (different frames, different attributes).

Labels are by construction from the kinship definitions below; no model labels anything. All sentences, names and places are new:
disjoint() checks them against both Tier-2 benchmark arms (all user turns) and the three judge-validation sets.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER2 = HERE.parent
CASES = HERE / "cases_v1.jsonl"
REVIEW_SHEET = HERE / "review_sheet.csv"

# Standard meanings (Hindi kinship terms, from the speaker's point of view). Reviewed before freezing (review_log.jsonl).
DEFINITIONS = {
    "dadi": "father's mother (paternal grandmother)",
    "nani": "mother's mother (maternal grandmother)",
    "dadaji": "father's father (paternal grandfather), honorific -ji",
    "nanaji": "mother's father (maternal grandfather), honorific -ji",
    "chachi": "wife of the father's younger brother",
    "chacha": "father's younger brother",
    "tauji": "father's elder brother, honorific -ji",
    "bua": "father's sister",
    "fufaji": "husband of the father's sister (bua), honorific -ji",
    "mausi": "mother's sister",
    "mausaji": "husband of the mother's sister (mausi), honorific -ji",
    "mama": "mother's brother",
    "mami": "wife of the mother's brother (mama)",
    "bhabhi": "elder brother's wife",
    "saali": "wife's sister (speaker is a married man)",
    "jija": "sister's husband",
    "devar": "husband's younger brother (speaker is a married woman)",
}
FEMININE = {"dadi", "nani", "chachi", "bua", "mausi", "mami", "bhabhi", "saali"}

# 12 term pairs: the 5 pairs that occur in the Tier-2 benchmark + 7 that do not (generality).
BENCHMARK_PAIRS = [("dadi", "nani"), ("chachi", "mausi"), ("chachi", "bua"), ("mami", "mausi"), ("bua", "mami")]
NEW_PAIRS = [("chachi", "mami"), ("nanaji", "dadaji"), ("mama", "chacha"), ("mausaji", "fufaji"), ("tauji", "chacha"),
             ("bhabhi", "saali"), ("jija", "devar")]

# New places (none occurs in the benchmark or any validation set; checked by disjoint()).
H_PLACES = ["Tinsukia", "Nagaon", "Dimapur", "Lunglei", "Balasore", "Berhampur", "Bhadrak", "Angul", "Keonjhar", "Baripada",
            "Koraput", "Jagdalpur", "Raigarh", "Ambikapur", "Dhamtari", "Mehsana", "Palanpur", "Amreli", "Gondal", "Veraval",
            "Botad", "Dahod", "Jamshedpur", "Chaibasa", "Lohardaga", "Gumla", "Simdega"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def poss(term: str) -> str:
    return "Meri" if term in FEMININE else "Mere"


def verbs(term: str) -> dict:
    f = term in FEMININE
    return {"go": "ja rahi hain" if f else "ja rahe hain", "live": "rehti hain" if f else "rehte hain"}


def h_diff() -> list[dict]:
    out, k = [], 0
    for i, (x, y) in enumerate(BENCHMARK_PAIRS + NEW_PAIRS):
        origin = "benchmark_pair" if (x, y) in BENCHMARK_PAIRS else "new_pair"
        p = lambda j: H_PLACES[(k + j) % len(H_PLACES)]
        cells = {
            "P-same": (f"{poss(x)} {x} {p(0)} {verbs(x)['go']}", f"{poss(y)} {y} {p(1)} {verbs(y)['go']}", "travel_destination|travel_destination"),
            "N-same": (f"{poss(x)} {x} {p(2)} mein {verbs(x)['live']}", f"{poss(y)} {y} ka ghar {p(3)} mein hai", "residence|residence"),
            "N-diff": (f"{poss(x)} {x} {p(4)} mein {verbs(x)['live']}", f"{poss(y)} {y} ka janamdin {MONTHS[i % 12]} mein hai", "residence|birthday_month"),
        }
        k += 5
        for cell, (a, b, attr) in cells.items():
            out.append({"group": "H-diff", "cell": cell, "terms": [x, y], "term_origin": origin, "attributes": attr, "text_a": a, "text_b": b,
                        "expected": "keep_both", "intended_identity": f"two different people: {x} = {DEFINITIONS[x]}; {y} = {DEFINITIONS[y]}",
                        "rationale": "different kinship terms name different relatives; nothing in either sentence links them"})
    return out


# Same-person controls: the same kinship term AND the same first name in both sentences (user-approved design, residence
# changes instead of trips for P-same).
CONTROLS = [
    ("mausi", "Kamla", ("Meri mausi Kamla Shillong mein rehti hain", "Meri mausi Kamla ab Imphal mein rehti hain"),
     ("Meri mausi Kamla Agartala mein rehti hain", "Kamla mausi ka ghar ab Aizawl mein hai"),
     ("Meri mausi Kamla Kohima mein padhati hain", "Kamla mausi ka janamdin November mein hai")),
    ("bua", "Sudha", ("Meri bua Sudha Gangtok mein rehti hain", "Meri bua Sudha ab Itanagar mein rehti hain"),
     ("Meri bua Sudha Dibrugarh mein rehti hain", "Sudha bua ka ghar ab Tezpur mein hai"),
     ("Meri bua Sudha Silchar mein kaam karti hain", "Sudha bua ki shaadi ki saalgirah March mein hai")),
    ("chachi", "Usha", ("Meri chachi Usha Jorhat mein rehti hain", "Meri chachi Usha ab Puri mein rehti hain"),
     ("Meri chachi Usha Cuttack mein rehti hain", "Usha chachi ka ghar ab Rourkela mein hai"),
     ("Meri chachi Usha Sambalpur mein dukaan chalati hain", "Usha chachi ka janamdin August mein hai")),
    ("mami", "Leela", ("Meri mami Leela Korba mein rehti hain", "Meri mami Leela ab Bastar mein rehti hain"),
     ("Meri mami Leela Porbandar mein rehti hain", "Leela mami ka ghar ab Junagadh mein hai"),
     ("Meri mami Leela Navsari mein school chalati hain", "Leela mami ka janamdin February mein hai")),
    ("mama", "Vinod", ("Mere mama Vinod Valsad mein rehte hain", "Mere mama Vinod ab Bharuch mein rehte hain"),
     ("Mere mama Vinod Morbi mein rehte hain", "Vinod mama ka ghar ab Nadiad mein hai"),
     ("Mere mama Vinod Godhra mein naukri karte hain", "Vinod mama ka janamdin December mein hai")),
    ("chacha", "Dinesh", ("Mere chacha Dinesh Hazaribagh mein rehte hain", "Mere chacha Dinesh ab Dhanbad mein rehte hain"),
     ("Mere chacha Dinesh Giridih mein rehte hain", "Dinesh chacha ka ghar ab Deoghar mein hai"),
     ("Mere chacha Dinesh Dumka mein dawakhana chalate hain", "Dinesh chacha ka janamdin June mein hai")),
]
CONTROL_RATIONALE = {
    "P-same": "same kinship term and same first name in both sentences; 'ab' marks a changed residence (changed/conflicting fact -> merge under the frozen policy)",
    "N-same": "same term and name; the reversed order '<name> <term>' is the usual way to refer to the same relative; 'ab' marks a changed residence",
    "N-diff": "same term and name; two compatible facts (occupation and a date) about one person",
}


def h_same_named() -> list[dict]:
    out = []
    for term, name, *cells in CONTROLS:
        for cell, (a, b) in zip(("P-same", "N-same", "N-diff"), cells):
            out.append({"group": "H-same-named", "cell": cell, "terms": [term, term], "term_origin": "control", "name": name,
                        "attributes": {"P-same": "residence|residence", "N-same": "residence|residence", "N-diff": "occupation|date"}[cell],
                        "text_a": a, "text_b": b, "expected": "merge",
                        "intended_identity": f"one person: {name}, the speaker's {term} ({DEFINITIONS[term]})", "rationale": CONTROL_RATIONALE[cell]})
    return out


ENGLISH = {
    ("dadi", "nani"): ("my father's mother", "my mother's mother", "my grandmother"),
    ("chachi", "mausi"): ("my father's younger brother's wife", "my mother's sister", "my aunt"),
    ("chachi", "bua"): ("my father's younger brother's wife", "my father's sister", "my aunt"),
    ("mami", "mausi"): ("my mother's brother's wife", "my mother's sister", "my aunt"),
    ("bua", "mami"): ("my father's sister", "my mother's brother's wife", "my aunt"),
}
E_PLACES = [("Tinsukia", "Nagaon"), ("Dimapur", "Lunglei"), ("Balasore", "Berhampur"), ("Bhadrak", "Angul"), ("Keonjhar", "Baripada"),
            ("Koraput", "Jagdalpur")]


def english() -> list[dict]:
    out = []
    pairs = list(ENGLISH.items()) + [(("chachi", "mami"), ("my father's younger brother's wife", "my mother's brother's wife", "my aunt"))]
    for (pair, (ex_a, ex_b, collapsed)), (c1, c2) in zip(pairs, E_PLACES):
        cap = lambda s: s[0].upper() + s[1:]
        out.append({"group": "E-explicit", "cell": "P-same", "terms": list(pair), "term_origin": "english", "attributes": "travel_destination|travel_destination",
                    "text_a": f"{cap(ex_a)} is travelling to {c1}", "text_b": f"{cap(ex_b)} is travelling to {c2}", "expected": "keep_both",
                    "intended_identity": f"two different people: {ex_a}; {ex_b}", "rationale": "the English spells out two different relations"})
        out.append({"group": "E-collapsed", "cell": "P-same", "terms": list(pair), "term_origin": "english", "attributes": "travel_destination|travel_destination",
                    "text_a": f"{cap(collapsed)} is travelling to {c1}", "text_b": f"{cap(collapsed)} is travelling to {c2}", "expected": None,
                    "intended_identity": "underdetermined (reference condition: the distinction is absent in English)",
                    "rationale": "what the judge does when both relatives are named with the same English word; not labelled, not gated"})
    return out


def possessive() -> list[dict]:
    out = []
    pairs = BENCHMARK_PAIRS + [("chachi", "mami")]
    places = [("Raigarh", "Ambikapur"), ("Dhamtari", "Mehsana"), ("Palanpur", "Amreli"), ("Gondal", "Veraval"), ("Botad", "Dahod"),
              ("Jamshedpur", "Chaibasa")]
    for (x, y), (c1, c2) in zip(pairs, places):
        out.append({"group": "possessive", "cell": "P-same", "terms": [x, y], "term_origin": "benchmark_pair" if (x, y) in BENCHMARK_PAIRS else "new_pair",
                    "attributes": "travel_destination|travel_destination",
                    "text_a": f"{x.capitalize()} ji {c1} {verbs(x)['go']}", "text_b": f"{y.capitalize()} ji {c2} {verbs(y)['go']}", "expected": "keep_both",
                    "intended_identity": f"two different people: {x} = {DEFINITIONS[x]}; {y} = {DEFINITIONS[y]}",
                    "rationale": "as H-diff P-same without the possessive Meri/Mere (tests whether 'Meri' is read as a name)"})
    return out


def build() -> list[dict]:
    rows = h_diff() + h_same_named() + english() + possessive()
    for i, r in enumerate(rows, 1):
        r["case_id"] = f"kp_{i:03d}"
    keys = ["case_id", "group", "cell", "terms", "term_origin", "name", "attributes", "text_a", "text_b", "expected", "intended_identity", "rationale"]
    out = [{k: r.get(k) for k in keys} for r in rows]
    assert len({(r["text_a"], r["text_b"]) for r in out}) == len(out), "duplicate case"
    return out


def render_cases(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def render_review_sheet(rows: list[dict]) -> str:
    """Hindi/Hinglish cases only (the English ones need no Hindi review), with the definitions, and empty reviewer columns."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["case_id", "group", "cell", "text_a", "text_b", "terms", "definitions", "intended_label", "intended_identity",
                "reviewer_correct(Y/N)", "reviewer_natural(Y/N)", "reviewer_ambiguous(Y/N)", "reviewer_suggested_fix", "reviewer_comment"])
    for r in rows:
        if r["group"].startswith("E-"):
            continue
        w.writerow([r["case_id"], r["group"], r["cell"], r["text_a"], r["text_b"], "/".join(r["terms"]),
                    " | ".join(f"{t}: {DEFINITIONS[t]}" for t in dict.fromkeys(r["terms"])), r["expected"], r["intended_identity"], "", "", "", "", ""])
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ disjointness
CLOSED = set(MONTHS) | {"Meri", "Mere", "My", "January"}


def reference_texts() -> dict[str, list[str]]:
    rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    ref = {}
    for name, p in (("benchmark v1.0", TIER2 / "data" / "tier2_conversations.jsonl"), ("mirror v1.1m", TIER2 / "mirror_v1_1" / "data" / "tier2_conversations.jsonl")):
        ref[name] = [t["text"] for c in rd(p) for s in c["sessions"] for t in s["turns"] if t["speaker"] == "user"]
    for name, p in (("judge validation v1", TIER2 / "judge_validation" / "cases_v1.jsonl"), ("judge validation v2", TIER2 / "judge_validation_v2" / "cases_v2.jsonl"),
                    ("judge validation 7b", TIER2 / "judge_validation_7b" / "cases_7b.jsonl")):
        ref[name] = [x for c in rd(p) for x in (c["text_a"], c["text_b"])]
    return ref


def disjoint(rows: list[dict]) -> dict:
    """Shared sentences and shared capitalised words (names, places) with every reference set. Kinship words, months and
    possessives are closed classes and are allowed to recur."""
    ours = [x for r in rows for x in (r["text_a"], r["text_b"])]
    kin = {k.capitalize() for k in DEFINITIONS} | {"Ji"}
    caps = lambda ts: {w for t in ts for w in re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]+\b", t)} - CLOSED - kin
    ours_caps = caps(ours)
    rep = {}
    for name, ts in reference_texts().items():
        rep[name] = {"identical_sentences": sorted(set(ours) & set(ts)), "shared_capitalised_words": sorted(ours_caps & caps(ts))}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    rows = build()
    texts = {CASES: render_cases(rows), REVIEW_SHEET: render_review_sheet(rows)}
    if a.check:
        bad = [p.name for p, t in texts.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        print("files equal the build:", not bad, bad)
        return 1 if bad else 0
    for p, t in texts.items():
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.write_text(t, encoding="utf-8", newline="\n")
    print("wrote", CASES.name, len(rows), "cases and", REVIEW_SHEET.name)
    print(json.dumps(disjoint(rows), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
