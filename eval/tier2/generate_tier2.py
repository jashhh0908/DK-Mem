"""Generate the Tier-2 controlled injection benchmark DATASET PACKAGE (data only; no DK-Mem runner).

Tier 2 is a CONTROLLED INJECTION benchmark: each of the 69 in-scope Tier-1 pairs (kinship, honorific_register, name_variant; 42 gold-different + 27
gold-same) is injected, as two templated fact sentences, into one synthetic 6-session conversation with fixed hand-authored distractor turns. It says
nothing about how often these distinctions occur in natural conversation.

Inputs (read-only): the frozen Tier-1 input and gold (for the provenance, class, language, relation and opaque entity ids), banks/pair_specs.json
(the 69 explicit hand-reviewable pair specs) and banks/distractors.json (the fixed hand-authored distractor bank). Everything else is deterministic
from SEED. Outputs are written to an empty directory and never overwrite anything.

Separation of concerns (one file each):
  tier2_conversations.jsonl        - the ONLY file the system under test receives (no labels, no injected/distractor marks)
  tier2_injection_metadata.jsonl   - where each pair was injected (evaluator side)
  tier2_gold.jsonl                 - gold relation / class / language / opaque entity ids, in the exact Tier-1 gold format
  tier2_qa_questions.jsonl         - QA questions (what the answerer sees)
  tier2_qa_reference.jsonl         - accepted aliases, forbidden aliases, supporting turns (evaluator side)
  tier2_manifest.json              - seed, parameters, counts and SHA-256 of every input and output
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
TIER1_INPUT = EVAL.parent / "ipdV2" / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"
TIER1_GOLD = EVAL / "teamB_answer_key.jsonl"
PAIR_SPECS = HERE / "banks" / "pair_specs.json"
DISTRACTORS = HERE / "banks" / "distractors.json"
DATA_DIR = HERE / "data"

BENCHMARK_VERSION = "1.0"
SEED = 20261009
N_SESSIONS = 6
TURNS_PER_SESSION = 4          # user, assistant, user, assistant
DISTANCES = (1, 3, 5)          # session distance between the two injected members
DISTRACTOR_USER_TURNS = 10     # 12 user turns per conversation: 2 injected + 10 distractors
RETRIEVAL_K = 5                # benchmark parameter for the shared bge-m3 top-k candidate retrieval (used by the runner, not by this package)
IN_SCOPE = ("kinship", "honorific_register", "name_variant")
STRATA = (("kinship", "different"), ("name_variant", "same"), ("name_variant", "different"), ("honorific_register", "same"), ("honorific_register", "different"))
FILES = ("tier2_conversations.jsonl", "tier2_injection_metadata.jsonl", "tier2_gold.jsonl", "tier2_qa_questions.jsonl", "tier2_qa_reference.jsonl")
STATEMENT = ("Tier 2 is a controlled injection benchmark built from constructed sentences; it makes no claim about the natural frequency of these "
             "distinctions in conversation. Labels are by construction from the provisional Tier-1 gold.")
SCORING = "all_slots_contained_and_no_forbidden_after_normalisation"


# ------------------------------------------------------------------------------------------------ deterministic scoring helpers
def normalize_text(s: str) -> str:
    """casefold, strip Latin diacritics (not Devanagari marks), drop punctuation, collapse spaces. Used for alias matching."""
    s = unicodedata.normalize("NFKD", s).casefold()
    out: list[str] = []
    for ch in s:
        if unicodedata.combining(ch) and out and out[-1].isascii() and out[-1].isalpha():
            continue
        out.append(ch)
    s = unicodedata.normalize("NFC", "".join(out))
    s = re.sub(r"[^\wऀ-ॿ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_alias(answer: str, alias: str) -> bool:
    a, b = normalize_text(answer), normalize_text(alias)
    return bool(b) and f" {b} " in f" {a} "


def score_answer(answer: str, reference: dict) -> dict:
    """Deterministic QA scoring: every slot needs one accepted alias in the answer, and no forbidden alias may appear (a conflated answer fails)."""
    slots_hit = [any(contains_alias(answer, al) for al in slot) for slot in reference["slots"]]
    forbidden_hit = any(contains_alias(answer, al) for group in reference.get("forbidden", []) for al in group)
    return {"correct": all(slots_hit) and not forbidden_hit, "slots_hit": slots_hit, "forbidden_hit": forbidden_hit}


# ------------------------------------------------------------------------------------------------ helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def rng_for(label: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()[:8], "big"))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dumps_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def allocate_distances(items: list[dict]) -> dict[str, int]:
    """Balanced session-distance allocation: each (class, relation) stratum cycles through 1/3/5; the per-stratum offsets are chosen to minimise the
    global spread, then the per-class spreads (deterministic tie-break on the offset tuple)."""
    by_stratum = {s: sorted([i for i in items if (i["distinction_class"], i["relation"]) == s], key=lambda i: i["source_tier1_pair_id"]) for s in STRATA}
    best, best_key = None, None
    for offs in itertools.product(range(3), repeat=len(STRATA)):
        alloc, per_class, glob = {}, {c: Counter() for c in IN_SCOPE}, Counter()
        for s, off in zip(STRATA, offs):
            for k, it in enumerate(by_stratum[s]):
                d = DISTANCES[(k + off) % 3]
                alloc[it["source_tier1_pair_id"]] = d
                per_class[s[0]][d] += 1
                glob[d] += 1
        spread = lambda c: max(c.get(d, 0) for d in DISTANCES) - min(c.get(d, 0) for d in DISTANCES)
        key = (spread(glob), sum(spread(c) for c in per_class.values()), offs)
        if best_key is None or key < best_key:
            best, best_key = alloc, key
    return best


# ------------------------------------------------------------------------------------------------ build
def build_all() -> dict:
    specs = json.loads(PAIR_SPECS.read_text(encoding="utf-8"))["specs"]
    bank = json.loads(DISTRACTORS.read_text(encoding="utf-8"))
    gold1 = {g["eval_pair_id"]: g for g in read_jsonl(TIER1_GOLD)}
    inp1 = {r["eval_pair_id"]: r for r in read_jsonl(TIER1_INPUT)}
    for s in specs:
        g = gold1[s["source_tier1_eval_pair_id"]]
        assert (g["pair_id"], g["relation"], g["distinction_class"], g["language"]) == (s["source_tier1_pair_id"], s["relation"], s["distinction_class"], s["language"]), s["spec_id"]
        assert s["distinction_class"] in IN_SCOPE and s["source_tier1_eval_pair_id"] in inp1
    order = sorted(specs, key=lambda s: s["source_tier1_pair_id"])
    rng_for("conversation-order").shuffle(order)               # conversation ids do not reveal class or relation
    alloc = allocate_distances(order)
    counters = Counter()
    conversations, metadata, gold, questions, references = [], [], [], [], []
    base_date = datetime.date(2026, 3, 2)
    for n, s in enumerate(order, start=1):
        conv_id, pair_id = f"t2c_{n:04d}", f"t2p_{n:04d}"
        rng = rng_for(conv_id)
        d = alloc[s["source_tier1_pair_id"]]
        options = N_SESSIONS - d                                # possible first-member sessions: 1..options
        first_idx = 1 + counters[d] % options
        counters[d] += 1
        first_member = rng.choice(["a", "b"])
        idx = {first_member: first_idx, ("b" if first_member == "a" else "a"): first_idx + d}
        lang = s["language"]
        users = rng.sample(bank["user_turns"][lang], DISTRACTOR_USER_TURNS)
        assist = list(bank["assistant_turns"][lang])
        rng.shuffle(assist)
        slot_of = {m: rng.choice([0, 1]) for m in ("a", "b")}   # which of the two user slots of its session carries the injected sentence
        sessions, turn_of, distractor_ids, injected_ids = [], {}, [], []
        u_i = a_i = 0
        for si in range(1, N_SESSIONS + 1):
            turns = []
            for ui in (0, 1):
                t_user = 2 * ui + 1
                member = next((m for m in ("a", "b") if idx[m] == si and slot_of[m] == ui), None)
                tid = f"{conv_id}_s{si}_t{t_user}"
                if member:
                    text, tlang = s["text_" + member], lang
                    turn_of[member] = tid
                    injected_ids.append(tid)
                else:
                    text, tlang = users[u_i], lang
                    u_i += 1
                    distractor_ids.append(tid)
                turns.append({"turn_id": tid, "speaker": "user", "language": tlang, "text": text})
                a = assist[a_i]
                a_i += 1
                turns.append({"turn_id": f"{conv_id}_s{si}_t{t_user + 1}", "speaker": "assistant", "language": a["language"], "text": a["text"]})
            sessions.append({"session_id": f"{conv_id}_s{si}", "session_index": si, "date_label": (base_date + datetime.timedelta(days=7 * (si - 1))).isoformat(), "turns": turns})
        conversations.append({"conversation_id": conv_id, "n_sessions": N_SESSIONS, "sessions": sessions})
        g1 = gold1[s["source_tier1_eval_pair_id"]]
        metadata.append({"conversation_id": conv_id, "injected_pair_id": pair_id, "spec_id": s["spec_id"], "source_tier1_pair_id": s["source_tier1_pair_id"],
                         "source_tier1_eval_pair_id": s["source_tier1_eval_pair_id"], "turn_id_a": turn_of["a"], "turn_id_b": turn_of["b"],
                         "session_id_a": f"{conv_id}_s{idx['a']}", "session_id_b": f"{conv_id}_s{idx['b']}", "session_index_a": idx["a"], "session_index_b": idx["b"],
                         "session_distance": d, "first_member": first_member, "language": lang, "distinction_class": s["distinction_class"],
                         "fact_a": s["fact_a"], "fact_b": s["fact_b"], "marker_a": s["marker_a"], "marker_b": s["marker_b"],
                         "n_turns": N_SESSIONS * TURNS_PER_SESSION, "injected_turn_ids": injected_ids, "distractor_user_turn_ids": distractor_ids})
        gold.append({"eval_pair_id": pair_id, "pair_id": f"t2_{n:04d}", "relation": s["relation"], "distinction_class": s["distinction_class"], "language": lang,
                     "utterance_a_id": turn_of["a"], "utterance_b_id": turn_of["b"],
                     "opaque_entity_id_a": g1["opaque_entity_id_a"], "opaque_entity_id_b": g1["opaque_entity_id_b"]})
        for q in s["qa"]:
            qid = f"t2q_{n:04d}_{q['role']}"
            questions.append({"qa_id": qid, "conversation_id": conv_id, "question": q["question"], "language": lang, "ask_after_session_index": N_SESSIONS})
            support = [turn_of["a"]] if q["role"] == "a" else [turn_of["b"]] if q["role"] == "b" else [turn_of["a"], turn_of["b"]]
            references.append({"qa_id": qid, "conversation_id": conv_id, "injected_pair_id": pair_id, "role": q["role"],
                               "qa_type": "disambiguation" if s["relation"] == "different" else "link",
                               "depends_on": "preserving_distinction" if s["relation"] == "different" else "linking_mentions",
                               "slots": q["slots"], "forbidden": q["forbidden"], "scoring": SCORING, "supporting_turn_ids": support})
    return {"tier2_conversations.jsonl": conversations, "tier2_injection_metadata.jsonl": metadata, "tier2_gold.jsonl": gold,
            "tier2_qa_questions.jsonl": questions, "tier2_qa_reference.jsonl": references}


def counts(parts: dict) -> dict:
    gold, meta = parts["tier2_gold.jsonl"], parts["tier2_injection_metadata.jsonl"]
    by = lambda key: dict(sorted(Counter(key(g, m) for g, m in zip(gold, meta)).items()))
    return {"conversations": len(parts["tier2_conversations.jsonl"]), "injected_pairs": len(gold),
            "qa_questions": len(parts["tier2_qa_questions.jsonl"]), "qa_references": len(parts["tier2_qa_reference.jsonl"]),
            "by_relation": by(lambda g, m: g["relation"]), "by_class": by(lambda g, m: g["distinction_class"]), "by_language": by(lambda g, m: g["language"]),
            "by_session_distance": by(lambda g, m: str(m["session_distance"])),
            "by_class_relation": by(lambda g, m: f"{g['distinction_class']}|{g['relation']}"),
            "by_class_session_distance": by(lambda g, m: f"{g['distinction_class']}|{m['session_distance']}"),
            "by_relation_session_distance": by(lambda g, m: f"{g['relation']}|{m['session_distance']}"),
            "by_language_class": by(lambda g, m: f"{g['language']}|{g['distinction_class']}"),
            "qa_by_type": dict(sorted(Counter(r["qa_type"] for r in parts["tier2_qa_reference.jsonl"]).items()))}


def manifest_for(parts: dict, texts: dict) -> dict:
    raw_in = TIER1_INPUT.read_bytes().replace(b"\r\n", b"\n")
    return {"benchmark": "DK-Mem Tier-2 controlled multi-session injection benchmark", "version": BENCHMARK_VERSION, "statement": STATEMENT,
            "seed": SEED,
            "parameters": {"n_sessions": N_SESSIONS, "turns_per_session": TURNS_PER_SESSION, "distances": list(DISTANCES), "user_turns_per_conversation": 2 + DISTRACTOR_USER_TURNS,
                           "distractor_user_turns_per_conversation": DISTRACTOR_USER_TURNS, "shared_candidate_retrieval": {"embedder": "BAAI/bge-m3", "top_k": RETRIEVAL_K},
                           "qa_scoring": SCORING, "turn_id_format": "t2c_NNNN_sS_tT", "date_labels": "weekly from 2026-03-02"},
            "sources": {"tier1_input_sha256_lf": sha256_bytes(raw_in), "tier1_gold_sha256": sha256_file(TIER1_GOLD), "pair_specs_sha256": sha256_file(PAIR_SPECS),
                        "distractors_sha256": sha256_file(DISTRACTORS), "generator_sha256": sha256_file(Path(__file__))},
            "counts": counts(parts),
            "files": {n: {"sha256": sha256_bytes(t.encode("utf-8")), "bytes": len(t.encode("utf-8")), "lines": t.count("\n")} for n, t in texts.items()}}


def render(parts: dict) -> dict:
    texts = {n: dumps_jsonl(rows) for n, rows in parts.items()}
    texts["tier2_manifest.json"] = json.dumps(manifest_for(parts, texts), ensure_ascii=False, indent=1) + "\n"
    return texts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    existing = [n for n in (*FILES, "tier2_manifest.json") if (a.out / n).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing Tier-2 files in {a.out}: {existing}")
    texts = render(build_all())
    for n, t in texts.items():
        with open(a.out / n, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
    c = json.loads(texts["tier2_manifest.json"])["counts"]
    print(f"wrote {len(texts)} files to {a.out}: {c['conversations']} conversations, {c['qa_questions']} QA questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
