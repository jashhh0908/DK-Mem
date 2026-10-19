"""Verified entry point for the entity report: ``entity_report.py`` itself is hash-pinned (ANALYSIS_PLAN_FREEZE_v1.json) and stays
byte-identical; this module runs it only after, and with, the independent run verification the pinned code does not make.

    python -m tier2.entity_report_verified --runs <dir>/runs --arm v1.0 --evaluator <kernel>.evaluator [--runs ... --arm mirror ...] \\
        --out report.json [--allow-excluded]

1. PREFLIGHT (before any report exists): every run of every arm is re-validated with ``contract.validate_run`` + ``judge_provenance.augment_report``
   (per-row judge identity, strict re-parse of every raw output, embedder revision against ``embedder_pin``). If any run fails, NO report is
   written and the exit status is 1 -- unless ``--allow-excluded``.
2. GENERATION: the unchanged ``entity_report.main`` runs with ``contract.validate_run`` replaced, for the duration of the call only, by the
   verified validator. The pinned report logic then applies its own rule to the extra errors: a run that fails is excluded and listed under
   ``failures`` exactly like any other contract failure (plan section 7). For runs that pass, the report is byte-identical to what the
   pinned code alone produces (tested).
3. SIDECAR ``<out>.verification.json``: the report's sha256, the per-run verification, the report/verification cross-check and ``ok``.

What this protects, precisely: reports produced through THIS entry point cannot silently include a run whose judge log or embedder
revision contradicts its manifest. It cannot stop someone from running ``python -m tier2.entity_report`` directly: making the pinned
entry point itself refuse would require editing a pinned file, i.e. re-freezing the analysis plan (see the repository report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import contract, entity_report, judge_provenance, verify_entity_inputs


def _sidecar(out: Path) -> Path:
    return out.with_name(out.name + ".verification.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, action="append", required=True)
    ap.add_argument("--arm", action="append", required=True, choices=tuple(entity_report.ARMS))
    ap.add_argument("--evaluator", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-excluded", action="store_true",
                    help="write the report even if some runs fail verification (they are excluded and listed as contract failures)")
    a = ap.parse_args(argv)
    if not (len(a.runs) == len(a.arm) == len(a.evaluator)):
        ap.error("--runs, --arm and --evaluator must be given once per arm")
    if a.out.exists() or _sidecar(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out} or {_sidecar(a.out)}")

    verified = {arm: verify_entity_inputs.verify_arm(r, arm, e) for r, arm, e in zip(a.runs, a.arm, a.evaluator)}
    failed = [f"{arm}/{rid}: {v['errors']}" for arm, runs in verified.items() for rid, v in runs.items() if not v["ok"]]
    if failed and not a.allow_excluded:
        print("verification FAILED; no report written:", *failed, sep="\n  ")
        return 1

    original = contract.validate_run

    def verified_validate(run_dir, *args, **kwargs):
        return judge_provenance.augment_report(original(run_dir, *args, **kwargs), run_dir)

    inner = []
    for r, arm, e in zip(a.runs, a.arm, a.evaluator):
        inner += ["--runs", str(r), "--arm", arm, "--evaluator", str(e)]
    contract.validate_run = verified_validate          # visible to entity_report (it calls contract.validate_run); restored below
    try:
        rc = entity_report.main([*inner, "--out", str(a.out)])
    finally:
        contract.validate_run = original
    if rc != 0:
        return rc

    report = json.loads(a.out.read_text(encoding="utf-8"))
    cross = verify_entity_inputs.cross_check(report, verified)
    side = {"report": a.out.name, "report_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
            "generated_via": "tier2.entity_report_verified", "allow_excluded": a.allow_excluded,
            "entity_report_py_sha256": hashlib.sha256(Path(entity_report.__file__).read_bytes()).hexdigest(),
            "embedder_registered_revision": judge_provenance.embedder_pin.BGE_M3_REVISION,
            "verification": verified, "failed_runs": failed, "report_cross_check": cross, "ok": not failed and not cross}
    _sidecar(a.out).write_text(json.dumps(side, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("wrote", _sidecar(a.out), "| ok:", side["ok"], "| failed runs:", len(failed), "| cross-check problems:", len(cross))
    return 0 if side["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
