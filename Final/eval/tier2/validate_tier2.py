"""Validate the Tier-2 dataset package (read-only). Exit code 0 only if every check passes.

Checks: manifest hashes and counts; deterministic regeneration (the files equal what generate_tier2.py produces); all 69 in-scope Tier-1 pairs exactly once
with the Tier-1 relation / class / language / opaque entity ids; gold in the Tier-1 gold format (and accepted by the existing Tier-1 gold loader); no label
leakage into tier2_conversations.jsonl; conversation structure (6 sessions x 4 turns, alternating speakers, unique ids); injected sentences = the pair specs
with their markers; the 1/3/5 session-distance allocation and its balance; QA coverage (2 questions per gold-different pair, 1 combined per gold-same pair),
alias consistency and that answers are carried by the right injected turn; distractor / assistant turns free of injected names, lexicon terms, register
pronouns, family terms and QA values; frozen Tier-1 artifacts unchanged. Lexicon coverage of the injected turns is reported as information only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tier2 import generate_tier2 as G

TIER1_GOLD_MD5 = "7c6945e8a2869ab52e84ebcb09a2931c"        # the frozen Tier-1 gold file
EXTRA_BANNED = {"maa", "papa", "bhai", "behen", "chefin", "tu", "tum", "aap", "tumne", "aapne", "du", "dein", "deine", "deiner", "deinen", "dir", "dich",
                "sie", "ihr", "ihre", "ihren", "ihnen", "kannst", "chachi", "mausi", "chacha", "mausa", "mama", "mami", "nana", "nani", "dada", "dadi", "bua", "tai", "tau"}
LEAK_TOKENS = ("ent_", "injected", "distractor", "relation", "kinship", "honorific", "name_variant", "t2p_", "t1_", "ep_", "gold", "pair_id")
CONV_KEYS = {"conversation_id", "n_sessions", "sessions"}
SESSION_KEYS = {"session_id", "session_index", "date_label", "turns"}
TURN_KEYS = {"turn_id", "speaker", "language", "text"}
QUESTION_KEYS = {"qa_id", "conversation_id", "question", "language", "ask_after_session_index"}
GOLD_KEYS = {"eval_pair_id", "pair_id", "relation", "distinction_class", "language", "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b"}


class Report:
    def __init__(self):
        self.failures: list[str] = []
        self.info: dict = {}

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.failures.append(msg)
        return bool(ok)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate(data_dir: Path = G.DATA_DIR) -> Report:
    R = Report()
    paths = {n: data_dir / n for n in (*G.FILES, "tier2_manifest.json")}
    if not R.check(all(p.exists() for p in paths.values()), f"missing files in {data_dir}"):
        return R
    conv, meta, gold, qs, refs = (G.read_jsonl(paths[n]) for n in G.FILES)
    man = json.loads(paths["tier2_manifest.json"].read_text(encoding="utf-8"))

    # ---- manifest and determinism
    for n in G.FILES:
        R.check(man["files"][n]["sha256"] == G.sha256_file(paths[n]), f"manifest sha256 mismatch: {n}")
    R.check(man["seed"] == G.SEED and man["version"] == G.BENCHMARK_VERSION, "manifest seed/version mismatch")
    R.check(man["sources"]["pair_specs_sha256"] == G.sha256_file(G.PAIR_SPECS) and man["sources"]["distractors_sha256"] == G.sha256_file(G.DISTRACTORS),
            "banks changed since the data were generated")
    regen = G.render(G.build_all())
    for n in (*G.FILES, "tier2_manifest.json"):
        R.check(regen[n] == paths[n].read_text(encoding="utf-8"), f"{n} differs from a fresh deterministic regeneration")

    # ---- frozen Tier-1 artifacts untouched
    R.check(hashlib.md5(G.TIER1_GOLD.read_bytes()).hexdigest() == TIER1_GOLD_MD5, "frozen Tier-1 gold changed")
    try:
        from harness import _dkmem  # noqa: F401  (puts ipdV2 on sys.path; read-only)
        from dkmem.tier1.io import TIER1_INPUT_SHA256
        R.check(man["sources"]["tier1_input_sha256_lf"] == TIER1_INPUT_SHA256, "Tier-1 input hash differs from the documented one")
    except ImportError:
        R.info["tier1_input_hash_check"] = "skipped (dkmem not importable)"

    # ---- counts and Tier-1 provenance
    if not R.check((len(conv), len(meta), len(gold)) == (69, 69, 69), f"expected 69 conversations/metadata/gold, got {(len(conv), len(meta), len(gold))}"):
        return R                                   # the remaining checks join the files on ids and need all 69 rows
    t1 = {g["eval_pair_id"]: g for g in G.read_jsonl(G.TIER1_GOLD)}
    t1_in = {k: g for k, g in t1.items() if g["distinction_class"] in G.IN_SCOPE}
    R.check(len(t1_in) == 69, "Tier-1 in-scope pair count is not 69")
    src = [m["source_tier1_eval_pair_id"] for m in meta]
    R.check(sorted(src) == sorted(t1_in), "the 69 in-scope Tier-1 pairs are not each represented exactly once")
    R.check(Counter(g["relation"] for g in gold) == {"different": 42, "same": 27}, "relation counts are not 42 different / 27 same")
    ids = [g["pair_id"] for g in gold] + [g["eval_pair_id"] for g in gold]
    R.check(len(set(ids)) == len(ids), "duplicate gold ids")
    mby = {m["injected_pair_id"]: m for m in meta}
    for g in gold:
        R.check(set(g) == GOLD_KEYS, f"gold keys differ from the Tier-1 gold format: {g['eval_pair_id']}")
        m = mby[g["eval_pair_id"]]
        s1 = t1_in[m["source_tier1_eval_pair_id"]]
        R.check((g["relation"], g["distinction_class"], g["language"]) == (s1["relation"], s1["distinction_class"], s1["language"]), f"{g['eval_pair_id']}: relation/class/language differ from Tier-1")
        R.check((g["opaque_entity_id_a"], g["opaque_entity_id_b"]) == (s1["opaque_entity_id_a"], s1["opaque_entity_id_b"]), f"{g['eval_pair_id']}: opaque entity ids not reused exactly")
        R.check(g["relation"] in ("same", "different"), f"{g['eval_pair_id']}: invalid relation")
        R.check((g["utterance_a_id"], g["utterance_b_id"]) == (m["turn_id_a"], m["turn_id_b"]), f"{g['eval_pair_id']}: gold utterance ids differ from the injection metadata")
    ents = [g[k] for g in gold for k in ("opaque_entity_id_a", "opaque_entity_id_b")]
    R.check(len(set(ents)) == len(ents), "opaque entity ids are not unique across Tier-2")

    # ---- no label leakage into the system-visible conversation file
    raw_conv = paths["tier2_conversations.jsonl"].read_text(encoding="utf-8").lower()
    for tok in LEAK_TOKENS:
        R.check(tok not in raw_conv, f"conversations file contains the label-like token {tok!r}")
    R.check(all(ent.lower() not in raw_conv for ent in ents), "an opaque entity id appears in the conversations file")
    classes_seq = [mby[c["conversation_id"].replace("t2c_", "t2p_")]["distinction_class"] for c in conv]
    R.check(sum(a != b for a, b in zip(classes_seq, classes_seq[1:])) >= 20, "conversation order is grouped by class (ids leak the class)")
    R.check([c["conversation_id"] for c in conv] == sorted(c["conversation_id"] for c in conv), "conversations are not in id order")

    # ---- conversation structure and injected content
    specs = {s["spec_id"]: s for s in json.loads(G.PAIR_SPECS.read_text(encoding="utf-8"))["specs"]}
    all_turn_ids: set[str] = set()
    for c in conv:
        R.check(set(c) == CONV_KEYS and c["n_sessions"] == G.N_SESSIONS and len(c["sessions"]) == G.N_SESSIONS, f"{c['conversation_id']}: bad conversation structure")
        text_by_id = {}
        for si, sess in enumerate(c["sessions"], start=1):
            R.check(set(sess) == SESSION_KEYS and sess["session_index"] == si and sess["session_id"] == f"{c['conversation_id']}_s{si}", f"{c['conversation_id']}: bad session {si}")
            R.check(len(sess["turns"]) == G.TURNS_PER_SESSION, f"{sess['session_id']}: not {G.TURNS_PER_SESSION} turns")
            for ti, t in enumerate(sess["turns"], start=1):
                R.check(set(t) == TURN_KEYS and t["turn_id"] == f"{sess['session_id']}_t{ti}", f"{sess['session_id']}: bad turn {ti}")
                R.check(t["speaker"] == ("user" if ti % 2 else "assistant"), f"{t['turn_id']}: speaker does not alternate")
                R.check(t["turn_id"] not in all_turn_ids, f"duplicate turn id {t['turn_id']}")
                all_turn_ids.add(t["turn_id"])
                text_by_id[t["turn_id"]] = t
        m = mby[c["conversation_id"].replace("t2c_", "t2p_")]
        sp = specs[m["spec_id"]]
        for mem in ("a", "b"):
            t = text_by_id[m[f"turn_id_{mem}"]]
            R.check(t["text"] == sp[f"text_{mem}"] and t["speaker"] == "user", f"{m['injected_pair_id']}: injected turn {mem} differs from its pair spec")
            R.check(sp[f"marker_{mem}"] in t["text"], f"{m['injected_pair_id']}: marker {mem} missing from its injected turn")
            R.check(m[f"turn_id_{mem}"].startswith(f"{m[f'session_id_{mem}']}_t"), f"{m['injected_pair_id']}: turn/session mismatch for {mem}")
        R.check(set(m["injected_turn_ids"]) == {m["turn_id_a"], m["turn_id_b"]}, f"{m['injected_pair_id']}: injected_turn_ids wrong")
        user_turns = [t for t in text_by_id.values() if t["speaker"] == "user"]
        R.check(len(user_turns) == 2 + G.DISTRACTOR_USER_TURNS and len(m["distractor_user_turn_ids"]) == G.DISTRACTOR_USER_TURNS, f"{c['conversation_id']}: wrong user-turn counts")
        R.check(m["session_index_b" if m["first_member"] == "a" else "session_index_a"] - m["session_index_a" if m["first_member"] == "a" else "session_index_b"] == m["session_distance"],
                f"{m['injected_pair_id']}: session distance inconsistent with the first member")

    # ---- session-distance allocation
    dist = Counter(m["session_distance"] for m in meta)
    R.check(set(dist) == set(G.DISTANCES) and max(dist.values()) - min(dist.values()) <= 1, f"session distances not balanced: {dict(dist)}")
    for cls in G.IN_SCOPE:
        d = Counter(m["session_distance"] for m in meta if m["distinction_class"] == cls)
        R.check(max(d.get(x, 0) for x in G.DISTANCES) - min(d.get(x, 0) for x in G.DISTANCES) <= 1, f"session distances not balanced within class {cls}: {dict(d)}")
    for rel in ("same", "different"):
        d = Counter(mby[g["eval_pair_id"]]["session_distance"] for g in gold if g["relation"] == rel)
        R.check(max(d.get(x, 0) for x in G.DISTANCES) - min(d.get(x, 0) for x in G.DISTANCES) <= 1, f"session distances not balanced within relation {rel}: {dict(d)}")

    # ---- QA coverage and consistency
    rel_of = {g["eval_pair_id"]: g["relation"] for g in gold}
    R.check(len(qs) == len(refs) == 111, f"expected 111 QA questions and references, got {(len(qs), len(refs))}")
    R.check([q["qa_id"] for q in qs] == [r["qa_id"] for r in refs], "questions and references are not aligned")
    per_pair = {}
    for r in refs:
        per_pair.setdefault(r["injected_pair_id"], []).append(r)
    for pid, rel in rel_of.items():
        roles = sorted(r["role"] for r in per_pair.get(pid, []))
        R.check(roles == (["a", "b"] if rel == "different" else ["ab"]), f"{pid}: QA roles {roles} wrong for a gold-{rel} pair")
    for q in qs:
        R.check(set(q) == QUESTION_KEYS, f"{q['qa_id']}: question keys wrong (question file must not carry labels)")
    qtext = {q["qa_id"]: q["question"] for q in qs}
    for r in refs:
        m = mby[r["injected_pair_id"]]
        sp = specs[m["spec_id"]]
        R.check(r["scoring"] == G.SCORING and r["slots"] and all(r["slots"]) and r["supporting_turn_ids"], f"{r['qa_id']}: incomplete reference")
        R.check((r["qa_type"] == "disambiguation") == (rel_of[r["injected_pair_id"]] == "different"), f"{r['qa_id']}: qa_type inconsistent with the relation")
        slot_norm = {G.normalize_text(a) for slot in r["slots"] for a in slot}
        forb_norm = {G.normalize_text(a) for grp in r["forbidden"] for a in grp}
        R.check(not (slot_norm & forb_norm), f"{r['qa_id']}: slot and forbidden aliases overlap")
        R.check((r["role"] == "ab") == (not r["forbidden"]), f"{r['qa_id']}: forbidden aliases must exist exactly for the per-member questions")
        text_a, text_b = sp["text_a"], sp["text_b"]
        own = {"a": [text_a], "b": [text_b], "ab": [text_a, text_b]}[r["role"]]
        other = {"a": text_b, "b": text_a}.get(r["role"])
        for slot in r["slots"]:                       # every required value is carried by (one of) the supporting injected turns
            R.check(any(G.contains_alias(t, al) for t in own for al in slot), f"{r['qa_id']}: no accepted alias appears in its supporting injected turn(s)")
        if other:                                      # a per-member question must not be answerable from the other member's turn, and forbids its value
            R.check(not any(G.contains_alias(text_a if r["role"] == "b" else text_b, al) for slot in r["slots"] for al in slot), f"{r['qa_id']}: its answer also appears in the other member's turn")
            for grp in r["forbidden"]:
                R.check(any(G.contains_alias(other, al) for al in grp), f"{r['qa_id']}: forbidden aliases do not occur in the other member's turn")
        R.check(not any(G.contains_alias(qtext[r["qa_id"]], al) for slot in r["slots"] for al in slot), f"{r['qa_id']}: the question text already contains the answer")

    # ---- distractor / assistant turns are clean
    bank = json.loads(G.DISTRACTORS.read_text(encoding="utf-8"))
    banned_tokens = set(EXTRA_BANNED)
    banned_phrases = set()
    for sp in specs.values():
        for key in ("marker_a", "marker_b"):
            banned_phrases.add(G.normalize_text(sp[key]))
            banned_tokens.update(G.normalize_text(sp[key]).split())
        for f in (sp["fact_a"]["value"], sp["fact_b"]["value"]):
            banned_phrases.add(G.normalize_text(f))
        for q in sp["qa"]:
            for grp in (*q["slots"], *q["forbidden"]):
                banned_phrases.update(G.normalize_text(a) for a in grp)
    lex = json.loads((G.EVAL.parent / "ipdV2" / "dkmem" / "memory" / "distinction_features.json").read_text(encoding="utf-8"))["entries"]
    for e in lex:
        banned_phrases.update(G.normalize_text(f) for f in e["surface_forms"])
    texts = [("user", lang, t) for lang, ts in bank["user_turns"].items() for t in ts] + [("assistant", lang, e["text"]) for lang, es in bank["assistant_turns"].items() for e in es]
    for kind, lang, t in texts:
        nt = G.normalize_text(t)
        toks = set(nt.split())
        R.check(not (toks & banned_tokens), f"{kind}/{lang} bank text contains a banned token {sorted(toks & banned_tokens)}: {t!r}")
        R.check(not any(f" {p} " in f" {nt} " for p in banned_phrases if p), f"{kind}/{lang} bank text contains an injected name, lexicon term or QA value: {t!r}")
    # the conversations really use only bank text (or the specs) for non-injected turns
    bank_users = {t for ts in bank["user_turns"].values() for t in ts}
    bank_assist = {e["text"] for es in bank["assistant_turns"].values() for e in es}
    for c in conv:
        m = mby[c["conversation_id"].replace("t2c_", "t2p_")]
        inj = set(m["injected_turn_ids"])
        for sess in c["sessions"]:
            for t in sess["turns"]:
                if t["turn_id"] in inj:
                    continue
                R.check(t["text"] in (bank_users if t["speaker"] == "user" else bank_assist), f"{t['turn_id']}: non-injected turn text is not from the fixed bank")

    # ---- information only: does the existing DK-Mem lexicon see the injected features?
    try:
        from dkmem.memory.lexicon import load_lexicon
        from dkmem.memory.matcher import match_distinctions
        lx = load_lexicon(G.EVAL.parent / "ipdV2" / "dkmem" / "memory" / "distinction_features.json")
        hit = Counter()
        for g in gold:
            if g["distinction_class"] in ("kinship", "honorific_register") and g["language"] == "hi":
                m = mby[g["eval_pair_id"]]
                sp = specs[m["spec_id"]]
                for mem in ("a", "b"):
                    hit[(g["distinction_class"], bool(match_distinctions(lx, sp[f"text_{mem}"], "hi")))] += 1
        R.info["lexicon_detects_injected_features(class, detected)->turns"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(hit.items())}
    except Exception as e:      # information only
        R.info["lexicon_check"] = f"skipped ({type(e).__name__})"
    R.info["counts"] = man["counts"]
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=G.DATA_DIR)
    a = ap.parse_args(argv)
    rep = validate(a.data_dir)
    print(json.dumps(rep.info, ensure_ascii=False, indent=1))
    if rep.failures:
        print(f"\nVALIDATION FAILED: {len(rep.failures)} failure(s)")
        for f in rep.failures[:60]:
            print("  -", f)
        return 1
    print("\nVALIDATION OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
