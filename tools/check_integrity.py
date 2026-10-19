"""Merge gate: run before every pull request and after every merge. Exit 0 only if nothing frozen changed and the paper's numbers are current.

    python tools/check_integrity.py

1. The 46 frozen / delivery files (tools/frozen_baseline_sha256.txt) are byte-identical. Files you do not have locally (gitignored data held by
   the lead) are reported as "absent", never as passed.
2. The four ANALYSIS_PLAN_FREEZE_v1.json pins (entity_report.py, contract.py, ...) match.
3. The vendored harness/evaluator files equal eval/harness/VENDORED.json.
4. paper/generated/* is up to date with the verified report (only checked where the report is present).
5. No generated paper file was hand-edited (each carries its GENERATED header).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    problems, notes = [], []
    ok = absent = 0
    for line in (REPO / "tools" / "frozen_baseline_sha256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, rel = line.split(maxsplit=1)
        p = REPO / rel.lstrip("*")
        if not p.exists():
            absent += 1
        elif sha(p) != want:
            problems.append(f"FROZEN FILE CHANGED: {rel.lstrip('*')}")
        else:
            ok += 1
    notes.append(f"frozen baseline: {ok} identical, {absent} absent locally (held by the lead), {len([p for p in problems if 'FROZEN' in p])} changed")

    fz = json.loads((REPO / "eval" / "tier2" / "ANALYSIS_PLAN_FREEZE_v1.json").read_text(encoding="utf-8"))
    for rel, want in fz["files_sha256"].items():
        if sha(REPO / "eval" / "tier2" / rel) != want:
            problems.append(f"PINNED ANALYSIS FILE CHANGED: eval/tier2/{rel}")
    notes.append(f"analysis-plan pins: {len(fz['files_sha256'])} checked")

    v = json.loads((REPO / "eval" / "harness" / "VENDORED.json").read_text(encoding="utf-8"))
    for rel, row in v["files"].items():
        if sha(REPO / "eval" / rel) != row["local_sha256"]:
            problems.append(f"VENDORED FILE CHANGED: eval/{rel}")
    notes.append(f"vendored files: {len(v['files'])} checked")

    lx = subprocess.run([sys.executable, str(REPO / "tools" / "build_lexicon_v2.py"), "--check"], capture_output=True, text=True)
    if lx.returncode != 0:
        problems.append("dkmem/memory/distinction_features_v2.json is STALE or invalid: run python tools/build_lexicon_v2.py ("
                        + lx.stdout.strip() + lx.stderr.strip()[-200:] + ")")
    else:
        notes.append("draft lexicon v2: matches its sources")

    report = REPO / "eval" / "tier2_entity_runs" / "TIER2_FULL_entity_report_v1.json"
    if report.exists():
        r = subprocess.run([sys.executable, str(REPO / "paper" / "tools" / "make_numbers.py"), "--check"], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("paper/generated is STALE: run python paper/tools/make_numbers.py (" + r.stdout.strip() + r.stderr.strip()[-200:] + ")")
        f = subprocess.run([sys.executable, str(REPO / "paper" / "tools" / "make_figures.py"), "--check"], capture_output=True, text=True)
        if f.returncode != 0:
            problems.append("paper/generated/figure_numbers.tex is STALE: run python paper/tools/make_figures.py (" + f.stdout.strip() + f.stderr.strip()[-200:] + ")")
        notes.append("paper numbers: checked against the verified report")
    else:
        notes.append("paper numbers: report absent locally, not checked (the lead checks this)")
    if (REPO / "results" / "full_ad_3b_s0_20261010_scored" / "score.json").exists():
        r = subprocess.run([sys.executable, str(REPO / "paper" / "tools" / "make_tier1_numbers.py"), "--check"], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("paper/generated tier1 files are STALE: run python paper/tools/make_tier1_numbers.py (" + r.stdout.strip() + r.stderr.strip()[-200:] + ")")
        notes.append("Tier-1 paper numbers: checked against the pinned key-holder scoring")
    else:
        notes.append("Tier-1 paper numbers: scored files absent locally, not checked (the key holder checks this)")
    if (REPO / "eval" / "tier2_entity_runs" / "v2_full_20261017" / "score_v2_provisional_dev.json").exists():
        r = subprocess.run([sys.executable, str(REPO / "paper" / "tools" / "make_v2_numbers.py"), "--check"], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("paper/generated v2 files are STALE: run python paper/tools/make_v2_numbers.py (" + r.stdout.strip() + r.stderr.strip()[-200:] + ")")
        notes.append("Tier-2 v2 paper numbers: checked against the pinned score_v2 reports")
    else:
        notes.append("Tier-2 v2 paper numbers: score reports absent locally, not checked (the lead checks this)")
    for p in sorted((REPO / "paper" / "generated").glob("*.tex")):
        if not p.read_text(encoding="utf-8").startswith("% GENERATED"):
            problems.append(f"HAND-EDITED GENERATED FILE: {p.relative_to(REPO)}")

    print("\n".join(notes))
    if problems:
        print("\nFAIL:", *problems, sep="\n  ")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
