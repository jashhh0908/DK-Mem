"""Score the Qwen2.5-7B validation run of merge_judge_entity_t2_v2 (cases_7b.jsonl) against criteria_7b.json (EVALUATOR side).

    python -m tier2.judge_validation_7b.score_7b --outputs <judge_outputs.jsonl> [--run-info <run_info.json>] [--out <report.json>]

Gates: the frozen v1 scorer (``tier2.judge_validation.score.score``), unchanged. Added here, pre-registered as REPORTED (not gated):
reason-decision contradictions, and (with ``--run-info``) a check that the run used the pinned 7B revision, sharded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..judge_validation.score import _jl, score

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_7b.jsonl"
CRITERIA = HERE / "criteria_7b.json"
FREEZE = HERE / "FREEZE.json"


def verify_freeze() -> dict:
    fz = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = [rel for rel, want in fz["files_sha256"].items() if hashlib.sha256((HERE.parent / rel).read_bytes()).hexdigest() != want]
    if bad:
        raise RuntimeError(f"frozen 7B judge-validation files changed: {bad}")
    return fz


def reason_verdict(raw) -> str:
    try:
        r = json.loads(raw).get("reason", "").lower()
    except (ValueError, AttributeError, TypeError):
        return "unparsed"
    if "same entity" in r:
        return "same"
    if "different entit" in r:
        return "different"
    if "not identifiable" in r:
        return "not_identifiable"
    return "other"


def contradictions(outputs: list[dict]) -> dict:
    cells = Counter()
    for o in outputs:
        try:
            d = json.loads(o["raw_output"]).get("decision")
        except (ValueError, AttributeError, TypeError):
            d = None
        cells[(reason_verdict(o["raw_output"]), str(d))] += 1
    bad = cells[("same", "keep_both")] + cells[("different", "merge")] + cells[("not_identifiable", "merge")]
    return {"reason_verdict_x_decision": {f"{a}|{b}": n for (a, b), n in sorted(cells.items())}, "contradictions": bad, "calls": len(outputs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--run-info", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    fz = verify_freeze()
    criteria = json.loads(CRITERIA.read_text(encoding="utf-8"))
    criteria["prompt_sha256"] = fz["prompt_sha256"]
    outs = _jl(a.outputs)
    rep = score(outs, _jl(CASES), criteria)
    rep["reported_diagnostics"] = {"reason_decision": contradictions(outs)}
    if a.run_info:
        info = json.loads(a.run_info.read_text(encoding="utf-8"))
        rep["reported_diagnostics"]["backbone"] = {
            "resolved_model_revision": info.get("resolved_model_revision"), "sharded": info.get("sharded"), "device_name": info.get("device_name"),
            "is_pinned_7b": info.get("resolved_model_revision") == criteria["decoding"]["model_revision"],
            "generation_params": (info.get("generation_run_info") or {}).get("generation_params"),
            "chat_template_sha256": (info.get("tokenizer") or {}).get("chat_template_sha256")}
    rep["freeze"] = {"frozen_at_utc": fz["frozen_at_utc"], "files_sha256": fz["files_sha256"]}
    if a.out:
        if a.out.exists():
            raise SystemExit(f"refusing to overwrite {a.out}")
        a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({"accepted": rep["accepted"], "gates": {k: v["pass"] for k, v in rep["gates"].items()}, "decisions": rep["decisions"],
                      "contradictions": rep["reported_diagnostics"]["reason_decision"]["contradictions"]}))
    return 0 if rep["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
