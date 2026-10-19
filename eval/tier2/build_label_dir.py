"""Build a v2 LABEL directory over an existing, frozen Tier-2 arm (EVALUATOR side; no inference).

The v1.0 runs saw tier2_conversations.jsonl of the v1.0 arm. A pair whose two injected sentences, markers and facts are unchanged in the
v2 specs can be rescored from those runs with its v2 label (relation, class, split); everything else is EXCLUDED and listed with the reason
(not in v2, or its text/markers/facts changed: those need the v2 package and new inference). The conversations file is copied byte for
byte, so tier2.replay / tier2.score_v2 accept the directory for the arm's existing runs.

    python -m tier2.build_label_dir --arm-data-dir tier2/data --arm-frozen tier2/FROZEN_v1.0.json --specs <pair_specs_v2.json>
        --splits <dev gold> <test gold> --out <new empty dir>

Writes: tier2_conversations.jsonl (copy), tier2_gold.jsonl, tier2_injection_metadata.jsonl (+ split, spec_source), tier2_qa_questions.jsonl
(the questions the runs were asked, unchanged), tier2_qa_reference.jsonl (v2 aliases; a question whose text changed in v2 is dropped),
excluded_pairs.txt, label_dir_manifest.json. Only retained pairs appear in gold/metadata/references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from . import generate_tier2 as G
from .generate_tier2_v2 import GenError, load_specs, load_splits

SAME_FIELDS = ("text_a", "text_b", "marker_a", "marker_b", "fact_a", "fact_b")


def verify_arm(data_dir: Path, frozen: Path) -> dict:
    fz = json.loads(Path(frozen).read_text(encoding="utf-8"))
    root = Path(frozen).parent
    bad = [rel for rel, want in fz["sha256"].items() if rel.startswith("data/") and hashlib.sha256((root / rel).read_bytes()).hexdigest() != want]
    if bad:
        raise GenError(f"arm files differ from {Path(frozen).name}: {bad}")
    if (root / "data").resolve() != Path(data_dir).resolve():
        raise GenError(f"{frozen} pins {root / 'data'}, not {data_dir}")
    return fz


def build(data_dir: Path, specs: list[dict], split: dict[str, str]) -> dict:
    spec = {s["spec_id"]: s for s in specs}
    meta = G.read_jsonl(Path(data_dir) / "tier2_injection_metadata.jsonl")
    gold = {g["eval_pair_id"]: g for g in G.read_jsonl(Path(data_dir) / "tier2_gold.jsonl")}
    qs = G.read_jsonl(Path(data_dir) / "tier2_qa_questions.jsonl")
    refs = G.read_jsonl(Path(data_dir) / "tier2_qa_reference.jsonl")
    v1_bank = {s["spec_id"]: s for s in json.loads(G.PAIR_SPECS.read_text(encoding="utf-8"))["specs"]}
    out_meta, out_gold, excluded, kept = [], [], {}, {}
    for m in meta:
        pid, s = m["injected_pair_id"], spec.get(m["spec_id"])
        v1 = v1_bank.get(m["spec_id"])
        if s is None:
            excluded[pid] = f"spec {m['spec_id']} ({m['language']}) is not in the v2 specs"
            continue
        if v1 is None or any(s[f] != v1[f] for f in SAME_FIELDS):
            excluded[pid] = f"spec {m['spec_id']}: injected text, markers or facts changed in v2 (new inference)"
            continue
        kept[pid] = s
        g = dict(gold[pid])
        g.update(relation=s["relation"], distinction_class=s["distinction_class"], language=s["language"])
        out_gold.append(g)
        out_meta.append({**m, "distinction_class": s["distinction_class"], "language": s["language"], "split": split[s["spec_id"]],
                         "spec_source": s.get("source", "v1.0-retained")})
    out_refs, dropped_qa = [], []
    qtext = {q["qa_id"]: q["question"] for q in qs}
    for r in refs:
        s = kept.get(r["injected_pair_id"])
        if s is None:
            continue
        new = next((q for q in s["qa"] if q["role"] == r["role"]), None)
        if new is None or new["question"] != qtext[r["qa_id"]]:
            dropped_qa.append(r["qa_id"])                 # the runs answered a different question
            continue
        rel = s["relation"]
        out_refs.append({**r, "slots": new["slots"], "forbidden": new["forbidden"],
                         "qa_type": "disambiguation" if rel == "different" else "link",
                         "depends_on": "preserving_distinction" if rel == "different" else "linking_mentions"})
    return {"gold": out_gold, "meta": out_meta, "refs": out_refs, "questions": qs, "excluded": excluded, "dropped_qa": dropped_qa}


def write(res: dict, data_dir: Path, out: Path, sources: dict) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise GenError(f"refusing to write into the non-empty directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(data_dir) / "tier2_conversations.jsonl", out / "tier2_conversations.jsonl")
    texts = {"tier2_gold.jsonl": G.dumps_jsonl(res["gold"]), "tier2_injection_metadata.jsonl": G.dumps_jsonl(res["meta"]),
             "tier2_qa_questions.jsonl": G.dumps_jsonl(res["questions"]), "tier2_qa_reference.jsonl": G.dumps_jsonl(res["refs"]),
             "excluded_pairs.txt": "".join(f"{p}\n" for p in sorted(res["excluded"]))}
    for n, t in texts.items():
        with open(out / n, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
    man = {"kind": "Tier-2 v2 label directory over a frozen arm (no new conversations, no inference)", "status": "DRAFT (labels not final)",
           "builder": "eval/tier2/build_label_dir.py", "builder_sha256": G.sha256_file(Path(__file__)), "sources": sources,
           "counts": {"retained_pairs": len(res["gold"]), "excluded_pairs": len(res["excluded"]), "qa_references": len(res["refs"]),
                      "qa_dropped_changed_question": len(res["dropped_qa"]),
                      "by_language_class_relation_split": dict(sorted(Counter(f"{m['language']}|{m['distinction_class']}|{g['relation']}|{m['split']}"
                                                                                for g, m in zip(res["gold"], res["meta"])).items()))},
           "excluded": dict(sorted(res["excluded"].items())), "dropped_qa": res["dropped_qa"],
           "files_sha256": {n: G.sha256_file(out / n) for n in ("tier2_conversations.jsonl", *texts)}}
    (out / "label_dir_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return man


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
    ap.add_argument("--arm-data-dir", type=Path, required=True)
    ap.add_argument("--arm-frozen", type=Path, required=True)
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--splits", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        fz = verify_arm(a.arm_data_dir, a.arm_frozen)
        specs = load_specs(a.specs)
        split = load_splits(a.splits, specs)
        res = build(a.arm_data_dir, specs, split)
        man = write(res, a.arm_data_dir, a.out, {"arm_frozen": str(a.arm_frozen), "arm_version": fz.get("version"),
                                                  "pair_specs_sha256": G.sha256_file(a.specs),
                                                  "split_files_sha256": {Path(p).name: G.sha256_file(p) for p in a.splits}})
    except GenError as e:
        raise SystemExit(f"refused: {e}")
    print(json.dumps({"out": str(a.out), **man["counts"], "excluded": man["excluded"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
