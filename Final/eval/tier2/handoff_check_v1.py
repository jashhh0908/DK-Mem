"""Custodian-side acceptance checks for an independently authored Tier-2 test set (items.jsonl + key.json). Evaluator side only: it reads
the key, so no tuner may run it. It prints item ids and defect codes, never labels, QA values or texts.

    python -m tier2.handoff_check_v1 --items items.jsonl --key key.json [--expect-items-sha256 H] [--expect-key-sha256 H]
        [--reviews reviews.jsonl] [--reference-specs specs.json ...] [--target 20] [--require-target]

Checks mirror ``generate_tier2_v2.validate_parts`` at spec level (QA contract) and the plan's per-relation target. Exit 0 only if there
are no defects (and, with --require-target, no shortfall). Near-duplicate frames against reference specs are WARNINGS for human decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import qa_norm as Q
from .heldout_integrity_v2_1 import THRESHOLD, _pair_sim

LANGS = ("hi", "mr", "gu")
CLASSES = ("kinship", "honorific_register", "name_variant", "contextual_qualifier")
ITEM_KEYS = {"item_id", "language", "text_a", "text_b"}
REC_REQUIRED = {"item_id", "relation", "distinction_class", "marker_a", "marker_b", "fact_a", "fact_b", "qa", "author_confidence", "source_kind",
                "conflict"}
REC_ALLOWED = REC_REQUIRED | {"ambiguity_note", "plausible_alternative_label"}
HEADER_KEYS = {"schema_version", "revision", "author_id", "language", "native_language_region", "authored_utc", "tools_used", "nonce",
               "conflict_report", "attests_no_benchmark_exposure", "attests_no_llm_drafting"}
CONFLICT_KINDS = {"relation_label_unclear", "kinship_term_unclear", "pronoun_or_register_unclear", "wording_or_script_unclear",
                  "does_not_map_to_class_definition", "convention_1_wrong", "convention_2_wrong", "convention_3_wrong", "convention_4_wrong"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(items: list[dict], key: dict, *, references: list[dict], target: int) -> dict:
    defects: list[tuple[str, str]] = []      # (item id or "-", code)
    warnings: list[tuple[str, str]] = []
    add = lambda iid, code: defects.append((iid, code))
    hdr = key.get("header", {}) if isinstance(key, dict) else {}
    if set(hdr) != HEADER_KEYS:
        add("-", "KEY_HEADER_KEYS")
    elif not (isinstance(hdr["nonce"], str) and len(hdr["nonce"]) == 64 and all(c in "0123456789abcdef" for c in hdr["nonce"])):
        add("-", "NONCE_FORMAT")
    elif not (hdr["attests_no_benchmark_exposure"] is True and hdr["attests_no_llm_drafting"] is True):
        add("-", "ATTESTATION_NOT_TRUE")
    if "conflict_report" in hdr and len(str(hdr["conflict_report"]).strip()) < 10:
        add("-", "CONFLICT_REPORT_EMPTY")
    ids = [i.get("item_id") for i in items]
    if len(set(ids)) != len(ids):
        add("-", "DUPLICATE_ITEM_ID")
    recs = {r.get("item_id"): r for r in key.get("records", [])} if isinstance(key, dict) else {}
    if set(recs) != set(ids):
        add("-", "ITEM_KEY_ID_MISMATCH")
    texts = Counter()
    ref_texts = {Q.normalize_text(t) for s in references for t in (s["text_a"], s["text_b"])}
    for it in items:
        iid = it.get("item_id", "?")
        if set(it) != ITEM_KEYS:
            add(iid, "ITEM_KEYS_NOT_LABEL_FREE")
            continue
        if not str(iid).startswith("ind_"):
            add(iid, "ITEM_ID_NOT_OPAQUE")
        if it["language"] not in LANGS:
            add(iid, "LANGUAGE")
        if not it["text_a"].strip() or not it["text_b"].strip() or it["text_a"] == it["text_b"]:
            add(iid, "TEXT_EMPTY_OR_IDENTICAL")
        for t in (it["text_a"], it["text_b"]):
            texts[Q.normalize_text(t)] += 1
            if Q.normalize_text(t) in ref_texts:
                add(iid, "TEXT_DUPLICATES_REFERENCE")
        for s in references:
            if _pair_sim(it, s) >= THRESHOLD:
                warnings.append((iid, "FRAME_SIMILAR_TO_REFERENCE"))
                break
        r = recs.get(iid)
        if r is None:
            continue
        miss, extra = REC_REQUIRED - set(r), set(r) - REC_ALLOWED
        if miss or extra:
            add(iid, "RECORD_KEYS")
            continue
        if r["relation"] not in ("same", "different") or r["distinction_class"] not in CLASSES:
            add(iid, "RELATION_OR_CLASS")
            continue
        if (r["distinction_class"] == "name_variant" and r["relation"] != "same") or (r["distinction_class"] == "contextual_qualifier" and r["relation"] != "different"):
            add(iid, "CLASS_RELATION_NOT_ALLOWED")
        if r["author_confidence"] not in ("certain", "probable", "ambiguous"):
            add(iid, "CONFIDENCE")
        elif r["author_confidence"] != "certain" and not str(r.get("ambiguity_note", "")).strip():
            add(iid, "AMBIGUITY_NOT_EXPLAINED")
        c = r["conflict"]
        if not isinstance(c, dict) or not isinstance(c.get("flagged"), bool) or set(c) - {"flagged", "kinds", "description"}:
            add(iid, "CONFLICT_FIELD")
        elif c["flagged"]:
            if not c.get("kinds") or not set(c["kinds"]) <= CONFLICT_KINDS:
                add(iid, "CONFLICT_KINDS")
            if len(str(c.get("description", "")).strip()) < 10:
                add(iid, "CONFLICT_DESCRIPTION_EMPTY")
            warnings.append((iid, "FLAGGED_CONFLICT_NEEDS_REVIEWER_RESOLUTION"))
        elif r["author_confidence"] == "ambiguous":
            add(iid, "AMBIGUOUS_NOT_FLAGGED")
        if r["author_confidence"] == "ambiguous":
            warnings.append((iid, "AUTHOR_MARKED_AMBIGUOUS_NEEDS_REVIEW"))
        for m, t in (("marker_a", it["text_a"]), ("marker_b", it["text_b"])):
            if not Q.contains_alias(t, r[m]):
                add(iid, "MARKER_NOT_IN_TEXT")
        roles = sorted(q.get("role") for q in r["qa"])
        if roles != (["a", "b"] if r["relation"] == "different" else ["ab"]):
            add(iid, "QA_ROLES")
            continue
        for q in r["qa"]:
            role = q["role"]
            own = {"a": [it["text_a"]], "b": [it["text_b"]], "ab": [it["text_a"], it["text_b"]]}[role]
            other = {"a": it["text_b"], "b": it["text_a"]}.get(role)
            slots = [a for g in q["slots"] for a in g]
            forb = [a for g in q["forbidden"] for a in g]
            if not q["slots"] or not all(q["slots"]):
                add(iid, "QA_EMPTY_SLOT")
            if (role == "ab") != (not q["forbidden"]):
                add(iid, "QA_FORBIDDEN_PRESENCE")
            if {Q.normalize_text(a) for a in slots} & {Q.normalize_text(a) for a in forb}:
                add(iid, "QA_SLOT_FORBIDDEN_OVERLAP")
            if not all(any(Q.contains_alias(t, a) for t in own for a in g) for g in q["slots"]):
                add(iid, "QA_ANSWER_NOT_IN_SUPPORTING_TEXT")
            if other and any(Q.contains_alias(other, a) for a in slots):
                add(iid, "QA_ANSWER_IN_OTHER_TEXT")
            if other and not all(any(Q.contains_alias(other, a) for a in g) for g in q["forbidden"]):
                add(iid, "QA_FORBIDDEN_NOT_IN_OTHER_TEXT")
            if any(Q.contains_alias(q["question"], a) for a in slots):
                add(iid, "QA_QUESTION_CONTAINS_ANSWER")
    if any(v > 1 for v in texts.values()):
        add("-", "DUPLICATE_TEXT_ACROSS_ITEMS")
    lang_of = {i.get("item_id"): i.get("language") for i in items}
    counts = Counter((lang_of[i], recs[i]["relation"]) for i in recs if i in lang_of and recs[i].get("relation") in ("same", "different"))
    short = {f"{l}|{r}": {"have": counts.get((l, r), 0), "need": target} for l in LANGS for r in ("different", "same") if counts.get((l, r), 0) < target}
    return {"n_items": len(items), "defects": sorted(set(defects)), "warnings": sorted(set(warnings)),
            "counts_language_relation": {f"{l}|{r}": n for (l, r), n in sorted(counts.items())}, "target_per_language_relation": target,
            "shortfalls": short}


def check_reviews(key: dict, reviews: list[dict]) -> list[tuple[str, str]]:
    """Reviewer completeness: every item reviewed once; every flagged item resolved (label_stands/excluded) with a reason; consistency."""
    out: list[tuple[str, str]] = []
    recs = {r["item_id"]: r for r in key.get("records", [])}
    seen = Counter(r.get("item_id") for r in reviews)
    for iid in recs:
        if seen.get(iid, 0) != 1:
            out.append((iid, "REVIEW_MISSING_OR_DUPLICATE"))
    for rv in reviews:
        iid = rv.get("item_id")
        if iid not in recs:
            out.append((str(iid), "REVIEW_UNKNOWN_ITEM"))
            continue
        flagged = bool(recs[iid].get("conflict", {}).get("flagged"))
        res = rv.get("conflict_resolution")
        if res not in ("not_flagged", "label_stands", "excluded"):
            out.append((iid, "REVIEW_CONFLICT_RESOLUTION_MISSING"))
        elif flagged and res == "not_flagged":
            out.append((iid, "FLAGGED_CONFLICT_UNRESOLVED"))
        elif not flagged and res == "label_stands":
            out.append((iid, "REVIEW_RESOLVES_UNFLAGGED"))
        if (flagged or rv.get("verdict") == "exclude" or res == "excluded") and not str(rv.get("reason", "")).strip():
            out.append((iid, "REVIEW_REASON_MISSING"))
        if res == "excluded" and rv.get("verdict") != "exclude":
            out.append((iid, "REVIEW_EXCLUDED_BUT_INCLUDED"))
        if flagged and rv.get("verdict") == "include" and rv.get("relation_correct") != "yes":
            out.append((iid, "REVIEW_INCLUDES_UNCONFIRMED_LABEL"))
    return sorted(set(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--expect-items-sha256")
    ap.add_argument("--expect-key-sha256")
    ap.add_argument("--reference-specs", type=Path, nargs="*", default=[])
    ap.add_argument("--reviews", type=Path, default=None, help="reviewer records (jsonl): every flagged conflict must have a resolution")
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--require-target", action="store_true")
    a = ap.parse_args(argv)
    items = [json.loads(l) for l in a.items.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = json.loads(a.key.read_text(encoding="utf-8"))
    refs = [s for p in a.reference_specs for s in json.loads(p.read_text(encoding="utf-8"))["specs"]]
    rep = check(items, key, references=refs, target=a.target)
    if a.reviews:
        rev = [json.loads(l) for l in a.reviews.read_text(encoding="utf-8").splitlines() if l.strip()]
        rep["review_defects"] = check_reviews(key, rev)
        rep["n_excluded_by_review"] = sum(r.get("verdict") == "exclude" for r in rev)
    rep["items_sha256"], rep["key_sha256"] = _sha(a.items), _sha(a.key)
    rep["hash_ok"] = {"items": a.expect_items_sha256 in (None, rep["items_sha256"]), "key": a.expect_key_sha256 in (None, rep["key_sha256"])}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    bad = bool(rep["defects"]) or bool(rep.get("review_defects")) or not all(rep["hash_ok"].values()) or (a.require_target and bool(rep["shortfalls"]))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
