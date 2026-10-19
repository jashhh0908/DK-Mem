"""Run-directory writer: exactly `run_manifest.json` + `pairwise_eval.jsonl`, schema-checked on write.

ipdV2's own writers only accept the two DK-Mem strategies, so this serialises the same schemas for any
strategy. Records are validated with the evaluator's schema checker against ipdV2's schema files.
Refuses to write into an existing run directory (never overwrites a run).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluator import jsonschema_lite
from evaluator.loaders import load_schema, relax_pairwise_schema

from ._dkmem import SCHEMA_DIR


class RunWriteError(ValueError):
    pass


def build_manifest(*, run_id: str, strategy: str, seed: int, pipeline_config: str, dkmem_mode: str,
                   backbone: str | None = None, extra: dict | None = None) -> dict:
    m = {"run_id": run_id, "strategy": strategy, "dataset_tier": "tier1_minimal_pairs", "seed": seed,
         "backbone": backbone, "created_at": datetime.now(timezone.utc).isoformat(),
         "pipeline_config": pipeline_config, "dkmem_mode": dkmem_mode}
    m.update(extra or {})
    return m


def write_run(run_dir: Path, manifest: dict, records: list[dict], schema_dir: Path = SCHEMA_DIR) -> None:
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise RunWriteError(f"{run_dir} already exists; runs are never overwritten")
    errs = [f"manifest {p}: {m}" for p, m in
            jsonschema_lite.validate(manifest, load_schema(schema_dir / "run_manifest.schema.json"))]
    pw = relax_pairwise_schema(load_schema(schema_dir / "pairwise_eval.schema.json"), manifest)   # raw-cosine runs only
    for r in records:
        errs += [f"{r.get('record_id')} {p}: {m}" for p, m in jsonschema_lite.validate(r, pw)]
        if r.get("run_id") != manifest["run_id"] or r.get("strategy") != manifest["strategy"]:
            errs.append(f"{r.get('record_id')}: run_id/strategy differ from manifest")
    if errs:
        raise RunWriteError("; ".join(errs[:5]))
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "pairwise_eval.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
