"""Independent verification of the runs behind an entity report (evaluator side; no model, no gold, writes only ``--out``).

``entity_report.py`` is hash-pinned by ``ANALYSIS_PLAN_FREEZE_v1.json`` and validates each run with the pinned ``contract.validate_run``,
which trusts a judge log's own status/proposal and accepts any embedder revision. It cannot be edited without breaking the freeze, so this
separate step re-validates the SAME runs with ``judge_provenance.validate_run_verified`` (per-row judge identity, strict re-parse of every
raw output, embedder revision against the registered pin) and, given the report, cross-checks that the report analysed exactly those runs
and recorded the same contract outcome.

What this does and does not protect (verified, not assumed): it detects, after the fact and before a report is trusted, a run whose log or
manifest does not match its declared judge or embedder. It does NOT stop ``entity_report.py`` from computing numbers from such a run; the
protection exists only if this verifier is run and its exit status honoured.

    python -m tier2.verify_entity_inputs --runs <dir>/runs --arm v1.0 --evaluator <kernel>.evaluator \\
        [--runs ... --arm mirror --evaluator ...] [--report TIER2_FULL_entity_report_v1.json] --out verification.json
Exit status 0 only if every run validates, no judge log or embedder check failed, and (with --report) the report's runs and contract
outcomes match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import embedder_pin, judge_provenance
from .entity_report import ARMS


def verify_arm(runs_dir: Path, arm: str, evaluator: Path) -> dict:
    data, frozen = ARMS[arm]
    qmap = evaluator / "qa_id_map.json"
    out = {}
    for rd in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        rep = judge_provenance.validate_run_verified(
            rd, data, passthrough_file=evaluator / "t2_passthrough_used.jsonl", qa_id_map=qmap if qmap.exists() else None, frozen_file=frozen)
        jv = rep["judge_log_verification"]
        out[rd.name] = {"ok": rep["ok"], "pipeline_config": rep["pipeline_config"], "errors": [e["code"] for e in rep["errors"]],
                        "warnings": [w["code"] for w in rep["warnings"]],
                        "judge_log": None if jv is None else {k: jv[k] for k in ("status", "family", "n_rows", "n_identity_verified", "n_reparsed_ok",
                                                                                 "n_reparsed_rejected", "problems")},
                        "embedder": rep["embedder_verification"]["status"]}
    return out


def cross_check(report: dict, verified: dict[str, dict]) -> list[str]:
    """Problems between a stored entity report and this verification (empty = consistent)."""
    problems = []
    for arm, ver in verified.items():
        rep_runs = ((report.get("arms") or {}).get(arm) or {}).get("runs")
        if rep_runs is None:
            problems.append(f"arm {arm}: not in the report")
            continue
        if set(rep_runs) != set(ver):
            problems.append(f"arm {arm}: report analysed {sorted(set(rep_runs) ^ set(ver))} differently from the runs verified here")
        for rid in set(rep_runs) & set(ver):
            if bool(rep_runs[rid].get("contract_ok")) != ver[rid]["ok"]:
                problems.append(f"{arm}/{rid}: report contract_ok={rep_runs[rid].get('contract_ok')!r}, verified here ok={ver[rid]['ok']}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, action="append", required=True)
    ap.add_argument("--arm", action="append", required=True, choices=tuple(ARMS))
    ap.add_argument("--evaluator", type=Path, action="append", required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not (len(a.runs) == len(a.arm) == len(a.evaluator)):
        ap.error("--runs, --arm and --evaluator must be given once per arm")
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    verified = {arm: verify_arm(r, arm, e) for r, arm, e in zip(a.runs, a.arm, a.evaluator)}
    failed = [f"{arm}/{rid}: {v['errors']}" for arm, runs in verified.items() for rid, v in runs.items() if not v["ok"]]
    cross = cross_check(json.loads(a.report.read_text(encoding="utf-8")), verified) if a.report else []
    result = {"embedder_registered_revision": embedder_pin.BGE_M3_REVISION, "arms": verified, "failed_runs": failed, "report_cross_check": cross,
              "report_checked": str(a.report) if a.report else None, "ok": not failed and not cross}
    a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("wrote", a.out, "| ok:", result["ok"], "| runs:", {k: len(v) for k, v in verified.items()}, "| failed:", len(failed), "| report mismatches:", len(cross))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
