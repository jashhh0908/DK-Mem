"""Generate a Tier-2 v2 benchmark DATASET PACKAGE from Team B pair specs (data only; no inference). The frozen v1 generator
(generate_tier2.py) is unchanged: it needs a Tier-1 link per spec, reads Team B's Tier-1 key and has a Hindi/German bank only.

Differences from v1 (everything else is the v1 construction: 6 sessions x 4 turns, 2 injected + 10 distractor user turns, session
distance 1/3/5, first member and user slot drawn per conversation, QA asked after the last session):
  * specs need no Tier-1 link; the Tier-1 key is never read. Opaque entity ids are minted per (version, spec, side): unique, carrying no
    relation (as v1, where all 138 ids were distinct);
  * any language with a distractor bank (several bank files may be given; a language may come from one file only);
  * session distances cycle over the specs sorted by (language, class, relation, spec id), so every stratum and the whole set are
    balanced to within one;
  * the evaluator-side metadata carries ``split`` (dev/test, from Team B's split files) and ``spec_source``;
  * QA references use the Indic-safe normaliser (``qa_norm.SCORING_V2``).

    python -m tier2.generate_tier2_v2 build --specs <pair_specs_v2.json> --splits <dev gold> <test gold> --distractors <bank> [<bank> ...]
        [--lexicon <file> ...] [--version 2.0] [--seed 20261015] --out <new empty dir>
    python -m tier2.generate_tier2_v2 validate --data-dir <dir> --specs ... --splits ... --distractors ... [--lexicon ...]

``build`` validates before it writes and writes nothing on a failure; the package is a draft until Team B's review and the FROZEN file.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from . import generate_tier2 as G
from . import qa_norm as Q
from .validate_tier2 import CONV_KEYS, EXTRA_BANNED, GOLD_KEYS, LEAK_TOKENS, QUESTION_KEYS, SESSION_KEYS, TURN_KEYS

DEFAULT_SEED = 20261015
N_SESSIONS, TURNS_PER_SESSION, DISTANCES, N_DISTRACTORS = G.N_SESSIONS, G.TURNS_PER_SESSION, G.DISTANCES, G.DISTRACTOR_USER_TURNS
FILES = G.FILES
STATEMENT = ("Tier 2 v2 is a controlled injection benchmark built from constructed sentences; it makes no claim about the natural frequency "
             "of these distinctions in conversation. Labels are Team B's v2 labels (see the review record pinned with the FROZEN file).")
META_EXTRA = {"split", "spec_source"}


class GenError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ inputs
def load_specs(path: Path) -> list[dict]:
    specs = json.loads(Path(path).read_text(encoding="utf-8"))["specs"]
    ids = [s["spec_id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise GenError("duplicate spec ids")
    need = {"spec_id", "relation", "distinction_class", "language", "text_a", "text_b", "marker_a", "marker_b", "fact_a", "fact_b", "qa"}
    for s in specs:
        if need - set(s):
            raise GenError(f"{s.get('spec_id')}: spec misses {sorted(need - set(s))}")
        if s["relation"] not in ("same", "different"):
            raise GenError(f"{s['spec_id']}: relation {s['relation']!r}")
    return specs


def load_splits(paths: list[Path], specs: list[dict]) -> dict[str, str]:
    """{spec id: split}; every spec exactly once, and the split files agree with the specs on relation, class and language."""
    by = {s["spec_id"]: s for s in specs}
    split = {}
    for p in paths:
        for row in G.read_jsonl(Path(p)):
            sid = row["eval_pair_id"]
            if sid in split:
                raise GenError(f"{sid} appears in more than one split row")
            if sid not in by:
                raise GenError(f"split file {Path(p).name} names {sid}, which has no spec")
            if (row["relation"], row["distinction_class"], row["language"]) != (by[sid]["relation"], by[sid]["distinction_class"], by[sid]["language"]):
                raise GenError(f"{sid}: split file and spec disagree on relation/class/language")
            if row.get("split") not in ("dev", "test"):
                raise GenError(f"{sid}: split {row.get('split')!r}")
            split[sid] = row["split"]
    missing = set(by) - set(split)
    if missing:
        raise GenError(f"{len(missing)} specs have no split: {sorted(missing)[:5]}")
    return split


def load_banks(paths: list[Path]) -> dict:
    bank = {"user_turns": {}, "assistant_turns": {}, "sources": {}}
    for p in paths:
        b = json.loads(Path(p).read_text(encoding="utf-8"))
        for kind in ("user_turns", "assistant_turns"):
            for lang, rows in b[kind].items():
                if lang in bank[kind]:
                    raise GenError(f"language {lang} has {kind} in more than one bank file")
                bank[kind][lang] = rows
        bank["sources"][str(p)] = G.sha256_file(Path(p))
    return bank


def check_bank_covers(bank: dict, languages: set[str]) -> None:
    for lang in sorted(languages):
        u, a = len(bank["user_turns"].get(lang, [])), len(bank["assistant_turns"].get(lang, []))
        if u < N_DISTRACTORS or a < 2 * N_SESSIONS:
            raise GenError(f"no usable distractor bank for language {lang!r}: {u} user turns (need >= {N_DISTRACTORS}), "
                           f"{a} assistant turns (need >= {2 * N_SESSIONS})")


def opaque_id(version: str, spec_id: str, side: str) -> str:
    return "ent_" + hashlib.sha256(f"tier2-v{version}|{spec_id}|{side}".encode("utf-8")).hexdigest()[:16]


def allocate_distances(specs: list[dict]) -> dict[str, int]:
    order = sorted(specs, key=lambda s: (s["language"], s["distinction_class"], s["relation"], s["spec_id"]))
    return {s["spec_id"]: DISTANCES[k % len(DISTANCES)] for k, s in enumerate(order)}


# ------------------------------------------------------------------------------------------------ build
def build(specs: list[dict], split: dict[str, str], bank: dict, *, version: str, seed: int) -> dict:
    check_bank_covers(bank, {s["language"] for s in specs})
    rng_for = lambda label: random.Random(int.from_bytes(hashlib.sha256(f"{seed}|{label}".encode("utf-8")).digest()[:8], "big"))
    order = sorted(specs, key=lambda s: s["spec_id"])
    rng_for("conversation-order").shuffle(order)            # ids reveal neither class, language nor relation
    alloc = allocate_distances(specs)
    counters = Counter()
    conversations, metadata, gold, questions, references = [], [], [], [], []
    base_date = datetime.date(2026, 3, 2)
    for n, s in enumerate(order, start=1):
        conv_id, pair_id = f"t2c_{n:04d}", f"t2p_{n:04d}"
        rng = rng_for(conv_id)
        d = alloc[s["spec_id"]]
        first_idx = 1 + counters[d] % (N_SESSIONS - d)
        counters[d] += 1
        first_member = rng.choice(["a", "b"])
        idx = {first_member: first_idx, ("b" if first_member == "a" else "a"): first_idx + d}
        lang = s["language"]
        users = rng.sample(bank["user_turns"][lang], N_DISTRACTORS)
        assist = list(bank["assistant_turns"][lang])
        rng.shuffle(assist)
        slot_of = {m: rng.choice([0, 1]) for m in ("a", "b")}
        sessions, turn_of, distractor_ids, injected_ids = [], {}, [], []
        u_i = a_i = 0
        for si in range(1, N_SESSIONS + 1):
            turns = []
            for ui in (0, 1):
                t_user = 2 * ui + 1
                member = next((m for m in ("a", "b") if idx[m] == si and slot_of[m] == ui), None)
                tid = f"{conv_id}_s{si}_t{t_user}"
                if member:
                    text = s["text_" + member]
                    turn_of[member] = tid
                    injected_ids.append(tid)
                else:
                    text = users[u_i]
                    u_i += 1
                    distractor_ids.append(tid)
                turns.append({"turn_id": tid, "speaker": "user", "language": lang, "text": text})
                a = assist[a_i]
                a_i += 1
                turns.append({"turn_id": f"{conv_id}_s{si}_t{t_user + 1}", "speaker": "assistant", "language": a["language"], "text": a["text"]})
            sessions.append({"session_id": f"{conv_id}_s{si}", "session_index": si, "date_label": (base_date + datetime.timedelta(days=7 * (si - 1))).isoformat(),
                             "turns": turns})
        conversations.append({"conversation_id": conv_id, "n_sessions": N_SESSIONS, "sessions": sessions})
        metadata.append({"conversation_id": conv_id, "injected_pair_id": pair_id, "spec_id": s["spec_id"],
                         "source_tier1_pair_id": s.get("source_tier1_pair_id"), "source_tier1_eval_pair_id": s.get("source_tier1_eval_pair_id"),
                         "turn_id_a": turn_of["a"], "turn_id_b": turn_of["b"], "session_id_a": f"{conv_id}_s{idx['a']}", "session_id_b": f"{conv_id}_s{idx['b']}",
                         "session_index_a": idx["a"], "session_index_b": idx["b"], "session_distance": d, "first_member": first_member, "language": lang,
                         "distinction_class": s["distinction_class"], "fact_a": s["fact_a"], "fact_b": s["fact_b"], "marker_a": s["marker_a"], "marker_b": s["marker_b"],
                         "n_turns": N_SESSIONS * TURNS_PER_SESSION, "injected_turn_ids": injected_ids, "distractor_user_turn_ids": distractor_ids,
                         "split": split[s["spec_id"]], "spec_source": s.get("source", "v1.0-retained" if s.get("source_tier1_pair_id") else None)})
        gold.append({"eval_pair_id": pair_id, "pair_id": f"t2_{n:04d}", "relation": s["relation"], "distinction_class": s["distinction_class"], "language": lang,
                     "utterance_a_id": turn_of["a"], "utterance_b_id": turn_of["b"],
                     "opaque_entity_id_a": opaque_id(version, s["spec_id"], "a"), "opaque_entity_id_b": opaque_id(version, s["spec_id"], "b")})
        for q in s["qa"]:
            qid = f"t2q_{n:04d}_{q['role']}"
            questions.append({"qa_id": qid, "conversation_id": conv_id, "question": q["question"], "language": lang, "ask_after_session_index": N_SESSIONS})
            support = [turn_of["a"]] if q["role"] == "a" else [turn_of["b"]] if q["role"] == "b" else [turn_of["a"], turn_of["b"]]
            references.append({"qa_id": qid, "conversation_id": conv_id, "injected_pair_id": pair_id, "role": q["role"],
                               "qa_type": "disambiguation" if s["relation"] == "different" else "link",
                               "depends_on": "preserving_distinction" if s["relation"] == "different" else "linking_mentions",
                               "slots": q["slots"], "forbidden": q["forbidden"], "scoring": Q.SCORING_V2, "supporting_turn_ids": support})
    return {"tier2_conversations.jsonl": conversations, "tier2_injection_metadata.jsonl": metadata, "tier2_gold.jsonl": gold,
            "tier2_qa_questions.jsonl": questions, "tier2_qa_reference.jsonl": references}


def render(parts: dict, *, version: str, seed: int, sources: dict) -> dict:
    texts = {n: G.dumps_jsonl(rows) for n, rows in parts.items()}
    gold, meta = parts["tier2_gold.jsonl"], parts["tier2_injection_metadata.jsonl"]
    by = lambda key: dict(sorted(Counter(key(g, m) for g, m in zip(gold, meta)).items()))
    counts = {"conversations": len(parts["tier2_conversations.jsonl"]), "injected_pairs": len(gold), "qa_questions": len(parts["tier2_qa_questions.jsonl"]),
              "by_relation": by(lambda g, m: g["relation"]), "by_language": by(lambda g, m: g["language"]), "by_class": by(lambda g, m: g["distinction_class"]),
              "by_split": by(lambda g, m: m["split"]), "by_session_distance": by(lambda g, m: str(m["session_distance"])),
              "by_language_class_relation": by(lambda g, m: f"{g['language']}|{g['distinction_class']}|{g['relation']}")}
    man = {"benchmark": "DK-Mem Tier-2 controlled multi-session injection benchmark", "version": version, "status": "DRAFT (not frozen)",
           "statement": STATEMENT, "seed": seed, "generator": "eval/tier2/generate_tier2_v2.py",
           "parameters": {"n_sessions": N_SESSIONS, "turns_per_session": TURNS_PER_SESSION, "distances": list(DISTANCES),
                          "distractor_user_turns_per_conversation": N_DISTRACTORS, "qa_scoring": Q.SCORING_V2,
                          "shared_candidate_retrieval": {"embedder": "BAAI/bge-m3", "top_k": G.RETRIEVAL_K}, "date_labels": "weekly from 2026-03-02"},
           "sources": {**sources, "generator_sha256": G.sha256_file(Path(__file__)), "qa_norm_sha256": G.sha256_file(Path(Q.__file__))},
           "counts": counts,
           "files": {n: {"sha256": G.sha256_bytes(t.encode("utf-8")), "bytes": len(t.encode("utf-8")), "lines": t.count("\n")} for n, t in texts.items()}}
    texts["tier2_manifest.json"] = json.dumps(man, ensure_ascii=False, indent=1) + "\n"
    return texts


# ------------------------------------------------------------------------------------------------ validation
def validate_parts(parts: dict, specs: list[dict], split: dict[str, str], bank: dict, lexicons: list) -> list[str]:
    """Structural, leakage, QA-consistency and distractor-cleanliness checks (the v1 validator's checks, parameterised)."""
    fail = []
    chk = lambda ok, msg: ok or fail.append(msg)
    conv, meta, gold, qs, refs = (parts[n] for n in FILES)
    spec = {s["spec_id"]: s for s in specs}
    chk(len(conv) == len(meta) == len(gold) == len(specs), f"row counts {len(conv)}/{len(meta)}/{len(gold)} vs {len(specs)} specs")
    chk(sorted(m["spec_id"] for m in meta) == sorted(spec), "every spec must be injected exactly once")
    mby = {m["injected_pair_id"]: m for m in meta}
    for g in gold:
        m = mby[g["eval_pair_id"]]
        s = spec[m["spec_id"]]
        chk(set(g) == GOLD_KEYS, f"{g['eval_pair_id']}: gold keys differ from the Tier-1 gold format")
        chk((g["relation"], g["distinction_class"], g["language"]) == (s["relation"], s["distinction_class"], s["language"]), f"{g['eval_pair_id']}: gold differs from spec")
        chk((g["utterance_a_id"], g["utterance_b_id"]) == (m["turn_id_a"], m["turn_id_b"]), f"{g['eval_pair_id']}: gold utterance ids differ from metadata")
        chk(m["split"] == split[m["spec_id"]], f"{g['eval_pair_id']}: split differs from Team B's split files")
    ents = [g[k] for g in gold for k in ("opaque_entity_id_a", "opaque_entity_id_b")]
    chk(len(set(ents)) == len(ents), "opaque entity ids are not unique")
    raw_conv = G.dumps_jsonl(conv).lower()
    for tok in LEAK_TOKENS:
        chk(tok not in raw_conv, f"conversations contain the label-like token {tok!r}")
    chk(all(e.lower() not in raw_conv for e in ents), "an opaque entity id appears in the conversations")
    for key in ("distinction_class", "language", "relation"):
        seq = [(mby[c["conversation_id"].replace("t2c_", "t2p_")] | {"relation": spec[mby[c["conversation_id"].replace("t2c_", "t2p_")]["spec_id"]]["relation"]})[key]
               for c in conv]
        if len(set(seq)) > 1:
            chk(sum(a != b for a, b in zip(seq, seq[1:])) >= len(seq) // 4, f"conversation order is grouped by {key}")
    seen = set()
    for c in conv:
        chk(set(c) == CONV_KEYS and c["n_sessions"] == N_SESSIONS and len(c["sessions"]) == N_SESSIONS, f"{c['conversation_id']}: bad structure")
        by_id = {}
        for si, sess in enumerate(c["sessions"], start=1):
            chk(set(sess) == SESSION_KEYS and sess["session_index"] == si and sess["session_id"] == f"{c['conversation_id']}_s{si}", f"{c['conversation_id']}: bad session {si}")
            chk(len(sess["turns"]) == TURNS_PER_SESSION, f"{sess['session_id']}: not {TURNS_PER_SESSION} turns")
            for ti, t in enumerate(sess["turns"], start=1):
                chk(set(t) == TURN_KEYS and t["turn_id"] == f"{sess['session_id']}_t{ti}", f"{sess['session_id']}: bad turn {ti}")
                chk(t["speaker"] == ("user" if ti % 2 else "assistant"), f"{t['turn_id']}: speakers do not alternate")
                chk(t["turn_id"] not in seen, f"duplicate turn id {t['turn_id']}")
                seen.add(t["turn_id"])
                by_id[t["turn_id"]] = t
        m = mby[c["conversation_id"].replace("t2c_", "t2p_")]
        s = spec[m["spec_id"]]
        for mem in ("a", "b"):
            t = by_id[m[f"turn_id_{mem}"]]
            chk(t["text"] == s[f"text_{mem}"] and t["speaker"] == "user", f"{m['injected_pair_id']}: injected turn {mem} differs from its spec")
        chk(set(m["injected_turn_ids"]) == {m["turn_id_a"], m["turn_id_b"]}, f"{m['injected_pair_id']}: injected_turn_ids wrong")
        chk(sum(t["speaker"] == "user" for t in by_id.values()) == 2 + N_DISTRACTORS and len(m["distractor_user_turn_ids"]) == N_DISTRACTORS,
            f"{c['conversation_id']}: wrong user-turn counts")
        lo, hi = sorted((m["session_index_a"], m["session_index_b"]))
        chk(hi - lo == m["session_distance"], f"{m['injected_pair_id']}: session distance inconsistent")
        users = set(bank["user_turns"][m["language"]])
        assist = {e["text"] for e in bank["assistant_turns"][m["language"]]}
        for t in by_id.values():
            if t["turn_id"] not in m["injected_turn_ids"]:
                chk(t["text"] in (users if t["speaker"] == "user" else assist), f"{t['turn_id']}: non-injected text is not from the bank")
    dist = Counter(m["session_distance"] for m in meta)
    chk(max(dist.values()) - min(dist.values()) <= 1, f"session distances not balanced: {dict(dist)}")
    for key in {(m["language"], m["distinction_class"], spec[m["spec_id"]]["relation"]) for m in meta}:
        d = Counter(m["session_distance"] for m in meta if (m["language"], m["distinction_class"], spec[m["spec_id"]]["relation"]) == key)
        chk(max(d.get(x, 0) for x in DISTANCES) - min(d.get(x, 0) for x in DISTANCES) <= 1 or sum(d.values()) < len(DISTANCES),
            f"session distances not balanced within {key}: {dict(d)}")
    # QA
    chk([q["qa_id"] for q in qs] == [r["qa_id"] for r in refs], "questions and references are not aligned")
    qtext = {q["qa_id"]: q["question"] for q in qs}
    for q in qs:
        chk(set(q) == QUESTION_KEYS, f"{q['qa_id']}: question keys wrong")
    roles = Counter()
    for r in refs:
        m = mby[r["injected_pair_id"]]
        s = spec[m["spec_id"]]
        roles[(r["injected_pair_id"], r["role"])] += 1
        tag = f"{r['qa_id']} ({m['spec_id']})"
        chk(r["scoring"] == Q.SCORING_V2 and r["slots"] and all(r["slots"]) and r["supporting_turn_ids"], f"{tag}: incomplete reference")
        chk(not ({Q.normalize_text(a) for sl in r["slots"] for a in sl} & {Q.normalize_text(a) for gr in r["forbidden"] for a in gr}), f"{tag}: slot and forbidden aliases overlap")
        chk((r["role"] == "ab") == (not r["forbidden"]), f"{tag}: forbidden aliases must exist exactly for per-member questions")
        own = {"a": [s["text_a"]], "b": [s["text_b"]], "ab": [s["text_a"], s["text_b"]]}[r["role"]]
        other = {"a": s["text_b"], "b": s["text_a"]}.get(r["role"])
        for slot in r["slots"]:
            chk(any(Q.contains_alias(t, al) for t in own for al in slot), f"{tag}: no accepted alias appears in its supporting turn(s)")
        if other:
            chk(not any(Q.contains_alias(other, al) for sl in r["slots"] for al in sl), f"{tag}: its answer also appears in the other member's turn")
            for grp in r["forbidden"]:
                chk(any(Q.contains_alias(other, al) for al in grp), f"{tag}: forbidden aliases do not occur in the other member's turn")
        chk(not any(Q.contains_alias(qtext[r["qa_id"]], al) for sl in r["slots"] for al in sl), f"{tag}: the question already contains the answer")
    for g in gold:
        got = sorted(role for (pid, role) in roles if pid == g["eval_pair_id"])
        chk(got == (["a", "b"] if g["relation"] == "different" else ["ab"]), f"{g['eval_pair_id']} ({mby[g['eval_pair_id']]['spec_id']}): QA roles {got} wrong for a gold-{g['relation']} pair")
    # distractor cleanliness: no marker, fact value, QA alias or lexicon term of any spec (lexicon terms via the production matcher)
    from harness import _dkmem  # noqa: F401
    from dkmem.memory.matcher import match_distinctions
    banned_tokens, banned_phrases = set(EXTRA_BANNED), set()
    for s in specs:
        for key in ("marker_a", "marker_b"):
            nm = Q.normalize_text(s[key])
            banned_phrases.add(nm)
            if len(nm.split()) == 1:                    # a multi-word marker ("Papa ki behen") is banned as a phrase, not per token
                banned_tokens.add(nm)
        banned_phrases.update(Q.normalize_text(str(f["value"])) for f in (s["fact_a"], s["fact_b"]))
        for q in s["qa"]:
            for grp in (*q["slots"], *q["forbidden"]):
                banned_phrases.update(Q.normalize_text(a) for a in grp)
    used = {m["language"] for m in meta}
    for kind in ("user_turns", "assistant_turns"):
        for lang in sorted(used):
            for row in bank[kind][lang]:
                t = row if kind == "user_turns" else row["text"]
                nt = Q.normalize_text(t)
                chk(not (set(nt.split()) & banned_tokens), f"{kind}/{lang} bank text has a banned token {sorted(set(nt.split()) & banned_tokens)}: {t!r}")
                chk(not any(p and f" {p} " in f" {nt} " for p in banned_phrases), f"{kind}/{lang} bank text has an injected name, marker or QA value: {t!r}")
                for lx in lexicons:
                    hits = match_distinctions(lx, t, lang)
                    chk(not hits, f"{kind}/{lang} bank text matches the lexicon {hits}: {t!r}")
    return fail


def _load_inputs(a) -> tuple:
    from harness import _dkmem  # noqa: F401
    from dkmem.memory.lexicon import load_lexicon
    specs = load_specs(a.specs)
    split = load_splits(a.splits, specs)
    if a.languages:                                     # a staged build (e.g. Hindi while the mr/gu banks are pending)
        specs = [s for s in specs if s["language"] in set(a.languages.split(","))]
        split = {k: v for k, v in split.items() if k in {s["spec_id"] for s in specs}}
    bank = load_banks(a.distractors)
    lexicons = [load_lexicon(p) for p in (a.lexicon or [_dkmem.LEXICON_PATH])]
    sources = {"pair_specs_sha256": G.sha256_file(a.specs), "split_files_sha256": {Path(p).name: G.sha256_file(p) for p in a.splits},
               "distractor_banks_sha256": bank["sources"], "lexicons_checked_sha256": [G.sha256_file(Path(p)) for p in (a.lexicon or [_dkmem.LEXICON_PATH])]}
    return specs, split, bank, lexicons, sources


def _utf8_stdout() -> None:
    """Reports contain Devanagari/Gujarati; a Windows console encoding (cp1252) must not crash the CLI."""
    import sys
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--specs", type=Path, required=True)
        p.add_argument("--splits", type=Path, nargs="+", required=True)
        p.add_argument("--distractors", type=Path, nargs="+", required=True)
        p.add_argument("--lexicon", type=Path, nargs="*", default=None, help="lexicon(s) whose terms distractors must not contain")
        p.add_argument("--version", default="2.0")
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--languages", default=None, help="comma list: build only these languages' specs (a staged, separate package)")
        if name == "build":
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--data-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        specs, split, bank, lexicons, sources = _load_inputs(a)
        parts = build(specs, split, bank, version=a.version, seed=a.seed)
    except GenError as e:
        raise SystemExit(f"refused: {e}")
    texts = render(parts, version=a.version, seed=a.seed, sources=sources)
    if a.cmd == "validate":
        fails = validate_parts({n: G.read_jsonl(a.data_dir / n) for n in FILES}, specs, split, bank, lexicons)
        fails += [f"{n} differs from a fresh deterministic regeneration" for n in texts if (a.data_dir / n).read_text(encoding="utf-8") != texts[n]]
        print(json.dumps({"ok": not fails, "failures": fails}, ensure_ascii=False, indent=1))
        return 0 if not fails else 1
    fails = validate_parts(parts, specs, split, bank, lexicons)
    if fails:
        print(json.dumps({"ok": False, "written": False, "failures": fails}, ensure_ascii=False, indent=1))
        return 1
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"refusing to write into the non-empty directory {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    for n, t in texts.items():
        with open(a.out / n, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
    print(f"wrote {len(texts)} files to {a.out}: {json.loads(texts['tier2_manifest.json'])['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
