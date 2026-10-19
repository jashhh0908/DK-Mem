"""Build the DRAFT distinction lexicon v2 from its three sources (deterministic; no model call, nothing hand-typed into the output).

    python tools/build_lexicon_v2.py            # writes dkmem/memory/distinction_features_v2.json (LF)
    python tools/build_lexicon_v2.py --check    # exit 1 if the written file differs from what the sources give

Sources:
* dkmem/memory/distinction_features.json (v1): the 15 pilot-reviewed Hindi entries, copied UNCHANGED (v1 itself is never modified). The
  proposal's extra Hindi variants are not taken: they were never reviewed, and the Hindi 'aap' rule stays as it is until a native review.
* data/team_b_delivery/member3_lexicon_proposal/proposed_distinction_features.json: the 18 Marathi/Gujarati entries, surface forms UNCHANGED.
* paper/review/{mr,gu}_lexicon_review.csv: the review verdict and comment of every one of those entries, appended to its notes. That review was
  made by the model (Claude), NOT by a native speaker; the lexicon says so in its description and in every mr/gu entry.

Every mr/gu entry must have a verdict. A 'wrong' entry is left out (none at present). 'unsure' entries are kept: their concerns (fictive kinship,
collision with English 'mama', tu/tame marking number as well as register) are the same risks v1 already accepts for Hindi chacha/mama, and
dropping tumhi/tame would leave the register class unable to fire for Marathi and Gujarati at all.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "dkmem" / "memory" / "distinction_features.json"
PROPOSAL = REPO / "data" / "team_b_delivery" / "member3_lexicon_proposal" / "proposed_distinction_features.json"
REVIEWS = {"mr": REPO / "paper" / "review" / "mr_lexicon_review.csv", "gu": REPO / "paper" / "review" / "gu_lexicon_review.csv"}
OUT = REPO / "dkmem" / "memory" / "distinction_features_v2.json"
NEW_LANGS = ("mr", "gu")
DESCRIPTION = ("DK-Mem distinction lexicon v2 (DRAFT, not frozen): the 15 pilot-reviewed Hindi entries of v1, unchanged, plus 18 Marathi (mr) and "
               "Gujarati (gu) kinship/register entries from the Member-3 proposal with their surface forms unchanged. The mr/gu entries were "
               "reviewed by the model (Claude), NOT by native speakers: no human verification was available. Each mr/gu entry's notes carry its "
               "review verdict; 'unsure' entries are kept and their risks are stated. Still held, as in the proposal: Gujarati naani (homograph of the "
               "adjective 'small/young') and Marathi naani, so a daadi/naani pair is tagged on one side only and the gate cannot veto it; Marathi "
               "periphrases (babanchi aai / aaichi aai) are not tagged. Built by tools/build_lexicon_v2.py; v1 is unchanged.")


def build() -> str:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    prop = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    entries = [dict(e) for e in v1["entries"]]
    assert all(e["lang"] == "hi" for e in entries), "v1 is expected to hold Hindi entries only"
    for lang in NEW_LANGS:
        with open(REVIEWS[lang], encoding="utf-8-sig", newline="") as fh:
            review = {r["id"]: r for r in csv.DictReader(fh)}
        proposed = [e for e in prop["entries"] if e["lang"] == lang]
        if set(review) != {e["id"] for e in proposed}:
            raise SystemExit(f"{lang}: review sheet ids differ from the proposal: {sorted(set(review) ^ {e['id'] for e in proposed})}")
        for e in proposed:
            r = review[e["id"]]
            if (r["value"], r["class"], r["match_type"]) != (e["value"], e["distinction_class"], e["match_type"]) \
                    or [s.strip() for s in r["surface_forms"].split(" | ")] != e["surface_forms"]:
                raise SystemExit(f"{e['id']}: the review sheet row does not describe the proposed entry")
            verdict = r["verdict(ok/wrong/unsure)"].strip().lower()
            if verdict not in ("ok", "unsure", "wrong"):
                raise SystemExit(f"{e['id']}: no review verdict ({verdict!r}); every mr/gu entry must be reviewed")
            if verdict == "wrong":
                continue
            out = dict(e)
            out["notes"] = f"{e['notes']} || MODEL REVIEW (Claude, not native-verified): {verdict}. {r['comment'].strip()}"
            entries.append(out)
    doc = {"schema_version": v1["schema_version"], "description": DESCRIPTION, "entries": entries}
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = build()
    sys.path.insert(0, str(REPO))
    from dkmem.memory.lexicon import load_lexicon        # the production loader validates the result (schema, regexes, conflicts)
    if a.check:
        ok = OUT.exists() and OUT.read_bytes() == text.encode("utf-8")
        print("up to date" if ok else f"STALE: {OUT.relative_to(REPO)} differs from its sources")
        if ok:
            load_lexicon(OUT)
        return 0 if ok else 1
    OUT.write_bytes(text.encode("utf-8"))
    lex = load_lexicon(OUT)
    print(f"wrote {OUT.relative_to(REPO)}: {len(lex)} entries, languages {sorted(lex.langs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
