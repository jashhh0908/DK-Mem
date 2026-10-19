"""Generate the Tier-2 v1.1m MIRROR arm (counter-confound) from the frozen v1.0 benchmark and banks/pair_specs_mirror.json.

EVALUATOR side: reads the frozen v1.0 conversations, injection metadata and gold (never the Tier-1 answer key). Deterministic; writes
to an empty directory and never overwrites.

Each mirror conversation ``t2m_NNNN`` is the v1.0 conversation ``t2c_NNNN`` with
  * every id renamed (``t2c_`` -> ``t2m_``; turn, session and conversation ids keep their structure), and
  * exactly ONE user turn changed: member b's injected sentence, replaced by the mirror bank's sentence for that pair.
Sessions, dates, distractors, assistant turns, member a's sentence, the session distance and which member comes first are unchanged,
so v1.0 and the mirror arm form a paired design (same conversation, flipped attribute pattern).

Labels are by construction and equal v1.0's for the same pair (the markers are unchanged): ``relation``, ``distinction_class``,
``language`` and the opaque entity ids are copied from the v1.0 gold. The mirror arm has NO QA questions: it exists to break the
attribute-pattern shortcut in merge decisions (FCR/MCR); v1.0 keeps its QA. Output files have the v1.0 names and formats.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_DATA = HERE.parent / "data"
BANK = HERE / "banks" / "pair_specs_mirror.json"
DATA_DIR = HERE / "data"
VERSION = "1.1m"
FILES = ("tier2_conversations.jsonl", "tier2_injection_metadata.jsonl", "tier2_gold.jsonl", "tier2_qa_questions.jsonl", "tier2_qa_reference.jsonl")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def dumps_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def rename(s: str) -> str:
    if not s.startswith("t2c_"):
        raise ValueError(f"unexpected v1.0 id {s!r}")
    return "t2m_" + s[4:]


def build(v1_dir: Path = V1_DATA, bank_path: Path = BANK) -> dict:
    convs = read_jsonl(v1_dir / "tier2_conversations.jsonl")
    meta = {m["conversation_id"]: m for m in read_jsonl(v1_dir / "tier2_injection_metadata.jsonl")}
    gold = {g["eval_pair_id"]: g for g in read_jsonl(v1_dir / "tier2_gold.jsonl")}
    bank = {s["spec_id"]: s for s in json.loads(bank_path.read_text(encoding="utf-8"))["specs"]}
    if len(bank) != len(convs) or len(meta) != len(convs):
        raise ValueError("the mirror bank, the metadata and the conversations must cover the same pairs")
    out_c, out_m, out_g = [], [], []
    for c in convs:
        m = meta[c["conversation_id"]]
        g = gold[m["injected_pair_id"]]
        spec = bank[m["spec_id"]]
        if (spec["relation"], spec["distinction_class"], spec["language"]) != (g["relation"], g["distinction_class"], g["language"]):
            raise ValueError(f"{m['spec_id']}: bank labels differ from the v1.0 gold")
        if spec["marker_b"] != m["marker_b"]:
            raise ValueError(f"{m['spec_id']}: member b's marker changed")
        changed = 0
        sessions = []
        for s in c["sessions"]:
            turns = []
            for t in s["turns"]:
                t2 = {**t, "turn_id": rename(t["turn_id"])}
                if t["turn_id"] == m["turn_id_b"]:
                    t2["text"] = spec["text_b"]
                    changed += 1
                turns.append(t2)
            sessions.append({**s, "session_id": rename(s["session_id"]), "turns": turns})
        if changed != 1:
            raise ValueError(f"{c['conversation_id']}: expected exactly one replaced turn, got {changed}")
        cid = rename(c["conversation_id"])
        n = cid[4:]
        pid = f"t2mp_{n}"
        out_c.append({**c, "conversation_id": cid, "sessions": sessions})
        out_m.append({**m, "conversation_id": cid, "injected_pair_id": pid, "turn_id_a": rename(m["turn_id_a"]), "turn_id_b": rename(m["turn_id_b"]),
                      "session_id_a": rename(m["session_id_a"]), "session_id_b": rename(m["session_id_b"]), "fact_b": spec["fact_b"],
                      "injected_turn_ids": [rename(x) for x in m["injected_turn_ids"]],
                      "distractor_user_turn_ids": [rename(x) for x in m["distractor_user_turn_ids"]],
                      "arm": "mirror_v1.1m", "mirror_of_conversation_id": c["conversation_id"], "mirror_of_injected_pair_id": m["injected_pair_id"],
                      "attribute_pattern": spec["attribute_pattern"], "v1_0_attribute_pattern": spec["v1_0_attribute_pattern"]})
        out_g.append({**g, "eval_pair_id": pid, "pair_id": f"t2m_{n}", "utterance_a_id": rename(g["utterance_a_id"]),
                      "utterance_b_id": rename(g["utterance_b_id"])})
    return {"tier2_conversations.jsonl": out_c, "tier2_injection_metadata.jsonl": out_m, "tier2_gold.jsonl": out_g,
            "tier2_qa_questions.jsonl": [], "tier2_qa_reference.jsonl": []}


def counts(parts: dict) -> dict:
    meta = parts["tier2_injection_metadata.jsonl"]
    by = lambda f: dict(sorted(Counter(f(m) for m in meta).items()))
    gold = {g["eval_pair_id"]: g for g in parts["tier2_gold.jsonl"]}
    return {"conversations": len(parts["tier2_conversations.jsonl"]), "injected_pairs": len(meta), "qa_questions": 0,
            "by_relation_attribute_pattern": by(lambda m: f"{gold[m['injected_pair_id']]['relation']}|{m['attribute_pattern']}"),
            "by_class_relation": by(lambda m: f"{m['distinction_class']}|{gold[m['injected_pair_id']]['relation']}"),
            "by_session_distance": by(lambda m: str(m["session_distance"]))}


def render(parts: dict) -> dict[str, str]:
    return {n: dumps_jsonl(rows) for n, rows in parts.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    existing = [n for n in FILES if (a.out / n).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing mirror files in {a.out}: {existing}")
    parts = build()
    for n, t in render(parts).items():
        with open(a.out / n, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
    print("wrote", a.out, json.dumps(counts(parts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
