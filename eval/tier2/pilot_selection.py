"""Deterministic selection of the 6 frozen Tier-2 conversations used for the system pilot (evaluator side: reads gold + injection metadata).

Rule: for each (class, relation, session distance) cell below take the lowest conversation id of the frozen benchmark.
  3 gold-different: kinship d=1, name_variant d=3, honorific_register (German du/Sie) d=5
  3 gold-same:      name_variant d=5, honorific_register (Hindi) d=1, name_variant d=3
This covers kinship, name_variant and honorific_register and all three session distances in each relation group.
The system under test is only ever given the conversation ids, never this file's labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import generate_tier2 as G

CELLS = (("kinship", "different", 1, None), ("name_variant", "different", 3, None), ("honorific_register", "different", 5, "de"),
         ("name_variant", "same", 5, None), ("honorific_register", "same", 1, "hi"), ("name_variant", "same", 3, None))


def select(data_dir: Path = G.DATA_DIR) -> list[dict]:
    gold = {g["eval_pair_id"]: g for g in G.read_jsonl(data_dir / "tier2_gold.jsonl")}
    meta = sorted(G.read_jsonl(data_dir / "tier2_injection_metadata.jsonl"), key=lambda m: m["conversation_id"])
    out, used = [], set()
    for cls, rel, dist, lang in CELLS:
        m = next(m for m in meta if m["conversation_id"] not in used and gold[m["injected_pair_id"]]["distinction_class"] == cls
                 and gold[m["injected_pair_id"]]["relation"] == rel and m["session_distance"] == dist and (lang is None or gold[m["injected_pair_id"]]["language"] == lang))
        used.add(m["conversation_id"])
        out.append({"conversation_id": m["conversation_id"], "injected_pair_id": m["injected_pair_id"], "distinction_class": cls, "relation": rel, "session_distance": dist,
                    "language": gold[m["injected_pair_id"]]["language"], "source_tier1_pair_id": m["source_tier1_pair_id"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=G.DATA_DIR)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    sel = select(a.data_dir)
    doc = {"purpose": "Tier-2 system pilot: 6 frozen conversations (3 gold-different + 3 gold-same). Evaluator-side file; the runner gets only the ids.",
           "benchmark_version": G.BENCHMARK_VERSION, "rule": __doc__.strip().splitlines()[2:6], "conversation_ids": [s["conversation_id"] for s in sel], "selected": sel}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", a.out, doc["conversation_ids"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
