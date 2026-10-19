"""Validate the Tier-2 v1.1m mirror arm (EVALUATOR side; reads v1.0 and mirror data, never the Tier-1 answer key).

    python -m tier2.mirror_v1_1.validate_mirror        # exit 0 only if every check passes

Checks: deterministic regeneration; the frozen pins; each mirror conversation equals its v1.0 conversation except for the renamed ids
and exactly member b's sentence; labels equal v1.0's; member b keeps its marker and states the new fact; the attribute pattern is
flipped as designed; across v1.0 + mirror the relation is balanced against the attribute pattern within every class x relation
stratum; the system-side files carry no label fields; the replacement sentences add no new cue (no distractor text, no Devanagari
outside the name-variant members that already carried it in v1.0).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from . import generate_mirror as M

HERE = M.HERE
FROZEN = HERE / "FROZEN_v1.1m.json"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class Report:
    def __init__(self):
        self.failures, self.n = [], 0

    def check(self, ok: bool, msg: str) -> bool:
        self.n += 1
        if not ok:
            self.failures.append(msg)
        return bool(ok)


def _strip_ids(c: dict) -> dict:
    return {**c, "conversation_id": None, "sessions": [{**s, "session_id": None, "turns": [{**t, "turn_id": t["turn_id"][4:]} for t in s["turns"]]}
                                                      for s in c["sessions"]]}


def validate(data_dir: Path = M.DATA_DIR) -> Report:
    R = Report()
    parts = M.build()
    texts = M.render(parts)
    for n in M.FILES:
        R.check((data_dir / n).read_text(encoding="utf-8") == texts[n], f"{n} is not what generate_mirror.py produces")
    if FROZEN.exists():
        fz = json.loads(FROZEN.read_text(encoding="utf-8"))
        for rel, want in fz["sha256"].items():
            p = (data_dir.parent / rel)
            R.check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == want, f"frozen pin differs: {rel}")
        v1 = json.loads((M.HERE.parent / "FROZEN_v1.0.json").read_text(encoding="utf-8"))["sha256"]
        for rel in ("data/tier2_conversations.jsonl", "data/tier2_injection_metadata.jsonl", "data/tier2_gold.jsonl", "banks/pair_specs.json"):
            R.check(fz["derived_from_v1_0"][rel] == v1[rel], f"derived_from_v1_0 pin differs from FROZEN_v1.0.json: {rel}")
    v1c = {c["conversation_id"]: c for c in M.read_jsonl(M.V1_DATA / "tier2_conversations.jsonl")}
    v1m = {m["conversation_id"]: m for m in M.read_jsonl(M.V1_DATA / "tier2_injection_metadata.jsonl")}
    v1g = {g["eval_pair_id"]: g for g in M.read_jsonl(M.V1_DATA / "tier2_gold.jsonl")}
    conv = M.read_jsonl(data_dir / "tier2_conversations.jsonl")
    meta = {m["conversation_id"]: m for m in M.read_jsonl(data_dir / "tier2_injection_metadata.jsonl")}
    gold = {g["eval_pair_id"]: g for g in M.read_jsonl(data_dir / "tier2_gold.jsonl")}
    bank = {s["spec_id"]: s for s in json.loads(M.BANK.read_text(encoding="utf-8"))["specs"]}
    distractors = json.loads((M.HERE.parent / "banks" / "distractors.json").read_text(encoding="utf-8"))["user_turns"]
    distractor_texts = {t for v in distractors.values() for t in v}
    R.check(len(conv) == len(meta) == len(gold) == 69, "expected 69 conversations, metadata rows and gold rows")
    R.check((data_dir / "tier2_qa_questions.jsonl").read_text(encoding="utf-8") == "" and (data_dir / "tier2_qa_reference.jsonl").read_text(encoding="utf-8") == "",
            "the mirror arm has no QA")
    cells = Counter()
    for c in conv:
        cid = c["conversation_id"]
        R.check(re.fullmatch(r"t2m_\d{4}", cid) is not None, f"{cid}: id format")
        R.check(set(c) == {"conversation_id", "n_sessions", "sessions"} and all(set(t) == {"turn_id", "speaker", "language", "text"}
                for s in c["sessions"] for t in s["turns"]), f"{cid}: label-free layout")
        m = meta[cid]
        src = v1c[m["mirror_of_conversation_id"]]
        src_m = v1m[src["conversation_id"]]
        a, b = _strip_ids(src), _strip_ids(c)
        diff = [(ta["turn_id"], ta["text"], tb["text"]) for sa, sb in zip(a["sessions"], b["sessions"]) for ta, tb in zip(sa["turns"], sb["turns"]) if ta != tb]
        R.check(len(a["sessions"]) == len(b["sessions"]) and all(len(x["turns"]) == len(y["turns"]) for x, y in zip(a["sessions"], b["sessions"])),
                f"{cid}: structure differs from v1.0")
        R.check(len(diff) == 1 and "t2c_" + diff[0][0] == src_m["turn_id_b"], f"{cid}: exactly member b's turn may differ, got {len(diff)}")
        g, g1 = gold[m["injected_pair_id"]], v1g[src_m["injected_pair_id"]]
        R.check({k: g[k] for k in ("relation", "distinction_class", "language", "opaque_entity_id_a", "opaque_entity_id_b")} ==
                {k: g1[k] for k in ("relation", "distinction_class", "language", "opaque_entity_id_a", "opaque_entity_id_b")}, f"{cid}: labels differ from v1.0")
        R.check((g["utterance_a_id"], g["utterance_b_id"]) == (m["turn_id_a"], m["turn_id_b"]), f"{cid}: gold utterance ids")
        spec = bank[m["spec_id"]]
        new = next(t["text"] for s in c["sessions"] for t in s["turns"] if t["turn_id"] == m["turn_id_b"])
        R.check(new == spec["text_b"], f"{cid}: member b is not the bank sentence")
        R.check(m["marker_b"].split()[-1] in new and str(spec["fact_b"]["value"]) in new, f"{cid}: member b lost its marker or value")
        R.check(new not in distractor_texts, f"{cid}: member b equals a distractor")
        R.check(bool(DEVANAGARI.search(new)) == bool(DEVANAGARI.search(src_m["marker_b"])), f"{cid}: Devanagari only where v1.0's member b carried it")
        fa, fb = m["fact_a"], m["fact_b"]
        if g["relation"] == "different":
            R.check(fa["attribute"] != fb["attribute"] and m["attribute_pattern"] == "different_attribute", f"{cid}: gold-different must change the attribute")
        else:
            R.check(fa["attribute"] == fb["attribute"] and str(fa["value"]).casefold() != str(fb["value"]).casefold()
                    and m["attribute_pattern"] == "same_attribute_conflicting_value", f"{cid}: gold-same must conflict on the same attribute")
        R.check(src_m["fact_a"] == fa and src_m["session_distance"] == m["session_distance"] and src_m["first_member"] == m["first_member"],
                f"{cid}: member a, distance or order changed")
        same_attr = fa["attribute"] == fb["attribute"]
        v1_same_attr = src_m["fact_a"]["attribute"] == src_m["fact_b"]["attribute"]
        cells[(g["distinction_class"], g["relation"], "same_attribute" if same_attr else "different_attribute")] += 1
        cells[(g["distinction_class"], g["relation"], "same_attribute" if v1_same_attr else "different_attribute")] += 1
    for (cls, rel) in {(k[0], k[1]) for k in cells}:
        R.check(cells[(cls, rel, "same_attribute")] == cells[(cls, rel, "different_attribute")],
                f"v1.0 + mirror: {cls}|{rel} is not balanced over the attribute pattern")
    return R


def main(argv=None) -> int:
    R = validate()
    print(f"{R.n - len(R.failures)}/{R.n} checks pass")
    for f in R.failures[:30]:
        print("FAIL", f)
    return 0 if not R.failures else 1


if __name__ == "__main__":
    sys.exit(main())
