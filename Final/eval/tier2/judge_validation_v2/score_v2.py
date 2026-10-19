"""Score a v2 judge-validation run (merge_judge_entity_t2_v2 on cases_v2.jsonl) against criteria_v2.json (EVALUATOR side).

    python -m tier2.judge_validation_v2.score_v2 --outputs <judge_outputs.jsonl> [--out <report.json>]

The scoring rules are the frozen v1 scorer's (``tier2.judge_validation.score.score``, imported unchanged so that both validations
are scored identically); this module only points it at the v2 files and checks the v2 freeze first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..judge_validation.score import _jl, score

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_v2.jsonl"
CRITERIA = HERE / "criteria_v2.json"
FREEZE = HERE / "FREEZE.json"


def verify_freeze() -> dict:
    fz = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = [rel for rel, want in fz["files_sha256"].items() if hashlib.sha256((HERE.parent / rel).read_bytes()).hexdigest() != want]
    if bad:
        raise RuntimeError(f"frozen v2 judge-validation files changed: {bad}")
    return fz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    fz = verify_freeze()
    criteria = json.loads(CRITERIA.read_text(encoding="utf-8"))
    criteria["prompt_sha256"] = fz["prompt_sha256"]
    rep = score(_jl(a.outputs), _jl(CASES), criteria)
    rep["freeze"] = {"frozen_at_utc": fz["frozen_at_utc"], "files_sha256": fz["files_sha256"]}
    if a.out:
        if a.out.exists():
            raise SystemExit(f"refusing to overwrite {a.out}")
        a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({"accepted": rep["accepted"], "gates": {k: v["pass"] for k, v in rep["gates"].items()}, "decisions": rep["decisions"]}))
    return 0 if rep["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
