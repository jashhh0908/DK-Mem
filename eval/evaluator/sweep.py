"""Offline tau sweep for the FCR-MCR frontier (experiment_output_spec.md, tier1_eval_contract.md section 6).

Decisions are re-derived from the raw, pre-gating `similarity_score` and `compatibility`:

    compatibility == "incompatible"     -> never merge, at any tau
    compatibility == "underdetermined"  -> never merge (not merged, per contract section 4)
    compatibility == "compatible"       -> merge iff similarity_score >  tau     (contract/spec: strict '>')
    compatibility is None (no gating)   -> merge iff similarity_score >  tau

NOTE: ipdV2's gate() merges at `sim >= tau`; the contract says `>`. The headline sweep follows the
contract. `check_recorded_decisions` compares the sweep rule with each record's own recorded decision
at its own threshold and reports every disagreement (ties at sim == tau are separated out).

FCR/MCR are NOT redefined here: each tau re-classifies the pairs through `metrics.pair_outcome`
(with the swept merge rule) and the numbers come from `metrics.fcr` / `metrics.mcr`. Pairs that were not
compared or had no entry have no records, hence no similarity: they are unmerged at every tau (a constant
MCR term) and can never be false merges.

Everything produced here from the legacy runs is LEGACY / PROVISIONAL / MACHINERY VALIDATION ONLY: the
legacy similarity is the placeholder difflib ratio.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .metrics import fcr, mcr, pair_details, pair_outcome, recorded_merge
from .report import find_runs
from .validate import DEFAULT_SCHEMA_DIR, GoldIndex, RunReport, validate_gold_file, validate_run

BANNER = "LEGACY / PROVISIONAL / MACHINERY VALIDATION ONLY"
BANNER_DETAIL = ("Legacy Tier 1 runs scored with the placeholder difflib similarity. "
                 "Tier 1 v2, LLM-annotated, provisional. Not a research result.")
NEVER_MERGE = {"incompatible", "underdetermined"}


def default_taus() -> list[float]:
    return [round(i / 100, 2) for i in range(101)]


def derived_merge(record: dict, tau: float) -> bool:
    """Would this record be a merge at `tau` under the contract's sweep rule?"""
    if record.get("compatibility") in NEVER_MERGE:
        return False
    return record["similarity_score"] > tau


def merge_rule(tau: float) -> Callable[[dict], bool]:
    return lambda record: derived_merge(record, tau)


def rows_at_tau(base_rows: list[dict], report: RunReport, tau: float) -> list[dict]:
    """Per-gold-pair rows re-classified at `tau` (structural status and gold fields unchanged)."""
    rule = merge_rule(tau)
    return [{**row, **pair_outcome(row["structural_status"], report.pair_records.get(row["eval_pair_id"], []), rule)}
            for row in base_rows]


def sweep_run(report: RunReport, gold: GoldIndex, taus: Iterable[float]) -> dict:
    base = pair_details(report, gold)
    classes = sorted({r["distinction_class"] for r in base})
    manifest = report.manifest or {}
    points = []
    for tau in taus:
        rows = rows_at_tau(base, report, tau)
        f, m = fcr(rows), mcr(rows)
        by_cls = defaultdict(list)
        for r in rows:
            by_cls[r["distinction_class"]].append(r)
        points.append({
            "tau": tau, "fcr": f, "mcr": m,
            "n_pairs_compared": sum(r["structural_status"] == "compared" for r in rows),
            "n_pairs_merged": sum(r["merged"] for r in rows),
            # tau-independent MCR term: gold-same pairs that were never compared / had no entry
            "constant_mcr_misses": sum(r["relation"] == "same" and r["structural_status"] != "compared" for r in rows),
            "per_class": {c: {"fcr": _nd(fcr(by_cls[c])), "mcr": _nd(mcr(by_cls[c]))} for c in classes},
        })
    return {"run_id": report.run_id, "strategy": report.strategy, "seed": manifest.get("seed"),
            "backbone": manifest.get("backbone"), "banner": BANNER, "points": points,
            "recorded_decision_check": check_recorded_decisions(report, gold)}


def _nd(cell: dict) -> dict:
    return {"n": cell["numerator"], "d": cell["denominator"], "rate": cell["rate"]}


def check_recorded_decisions(report: RunReport, gold: GoldIndex | None = None) -> dict:
    """Does the sweep rule reproduce each scored record's recorded decision at the record's own tau?

    Disagreements are reported, never fixed. `at_tie` marks sim == tau, where the contract's '>' and the
    gate's '>=' differ by construction.
    """
    total = agree = 0
    disagreements = []
    for pid, recs in report.pair_records.items():
        for r in recs:
            total += 1
            derived, recorded = derived_merge(r, r["threshold"]), recorded_merge(r)
            if derived == recorded:
                agree += 1
                continue
            disagreements.append({
                "eval_pair_id": pid, "record_id": r["record_id"], "tau": r["threshold"],
                "similarity_score": r["similarity_score"], "compatibility": r.get("compatibility"),
                "recorded_decision": r["decision"], "sweep_merge": derived, "at_tie": r["similarity_score"] == r["threshold"]})
    non_tie = [d for d in disagreements if not d["at_tie"]]
    return {"records_checked": total, "agree": agree, "disagree": len(disagreements),
            "disagree_at_tie_sim_equals_tau": len(disagreements) - len(non_tie),
            "disagree_not_at_tie": len(non_tie),
            "severity": "WARN" if disagreements else "OK", "disagreements": disagreements}


def build(gold_path: Path, runs_root: Path, taus: list[float] | None = None, schema_dir: Path = DEFAULT_SCHEMA_DIR) -> dict:
    taus = taus or default_taus()
    gold = validate_gold_file(gold_path)
    runs, skipped = [], []
    for d in find_runs(runs_root):
        rep = validate_run(d, gold, schema_dir)
        if (rep.manifest or {}).get("merge_mechanism") == "llm_judge":
            # a judge has no tau: its decisions are not a function of similarity_score, so a sweep is undefined
            skipped.append({"run_id": rep.run_id, "reason": "merge_mechanism is llm_judge: tau sweep undefined"})
            continue
        rec_taus = sorted({r["threshold"] for recs in rep.pair_records.values() for r in recs})
        grid = sorted(set(taus) | set(rec_taus))
        run = sweep_run(rep, gold, grid)
        run["validation_ok"] = rep.ok
        run["recorded_taus"] = rec_taus
        runs.append(run)
    return {"banner": BANNER, "detail": BANNER_DETAIL,
            "rule": "merge iff similarity_score > tau AND compatibility not in {incompatible, underdetermined}",
            "gold": str(gold_path), "tau_grid": taus, "runs": runs, "skipped_runs": skipped}


def _fmt(cell: dict) -> str:
    return f"{cell['display']} ({cell['numerator']}/{cell['denominator']})" + ("*" if cell["indicative"] else "")


def render_markdown(out: dict, show_taus=(0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)) -> str:
    L = [f"# {out['banner']}", "", out["detail"], "",
         f"Rule: {out['rule']}. Not-compared and no-entry pairs are unmerged at every tau (constant MCR term). "
         "`*` = n<20. No plots; no research claim.", ""]
    L += ["## Sweep rule vs recorded decisions at each record's own tau (validation)", "",
          "| run | records | agree | disagree | at tie (sim == tau) | not at tie | status |", "|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        c = r["recorded_decision_check"]
        L.append(f"| {r['run_id']} | {c['records_checked']} | {c['agree']} | {c['disagree']} | "
                 f"{c['disagree_at_tie_sim_equals_tau']} | {c['disagree_not_at_tie']} | {c['severity']} |")
    seen = set()
    for r in out["runs"]:
        if r["strategy"] in seen:
            continue
        seen.add(r["strategy"])
        same = [x for x in out["runs"] if x["strategy"] == r["strategy"]]
        ident = all([p["fcr"] for p in x["points"]] == [p["fcr"] for p in r["points"]] and
                    [p["mcr"] for p in x["points"]] == [p["mcr"] for p in r["points"]] for x in same)
        L += ["", f"## {r['strategy']} (shown for {r['run_id']}; "
                  f"{'all ' + str(len(same)) + ' seeds give identical sweeps' if ident else 'SEEDS DIFFER - see JSON'})", "",
              "| tau | FCR | MCR | pairs compared | pairs merged | constant MCR misses |", "|---|---|---|---|---|---|"]
        pts = {p["tau"]: p for p in r["points"]}
        for t in show_taus:
            p = pts.get(t)
            if p:
                L.append(f"| {t:.2f} | {_fmt(p['fcr'])} | {_fmt(p['mcr'])} | {p['n_pairs_compared']} | "
                         f"{p['n_pairs_merged']} | {p['constant_mcr_misses']} |")
        p0 = pts[min(pts)]
        L.append(f"\nFloor note: MCR cannot go below the constant term ({p0['constant_mcr_misses']} of "
                 f"{p0['mcr']['denominator']} gold-same pairs) at any tau.")
    return "\n".join(L) + "\n"


def write_outputs(out: dict, outdir: Path, prefix: str = "legacy_tier1_sweep") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{prefix}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(outdir / f"{prefix}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["banner", "run_id", "strategy", "seed", "tau", "fcr", "fcr_num", "fcr_den", "mcr", "mcr_num", "mcr_den",
                    "n_pairs_compared", "n_pairs_merged", "constant_mcr_misses"])
        for r in out["runs"]:
            for p in r["points"]:
                w.writerow([BANNER, r["run_id"], r["strategy"], r["seed"], p["tau"], p["fcr"]["rate"], p["fcr"]["numerator"],
                            p["fcr"]["denominator"], p["mcr"]["rate"], p["mcr"]["numerator"], p["mcr"]["denominator"],
                            p["n_pairs_compared"], p["n_pairs_merged"], p["constant_mcr_misses"]])
    (outdir / f"{prefix}_summary.md").write_text(render_markdown(out), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline tau sweep over all run directories (" + BANNER + ").")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    a = ap.parse_args(argv)
    out = build(a.gold, a.runs, schema_dir=a.schema_dir)
    write_outputs(out, a.out)
    for r in out["runs"]:
        c = r["recorded_decision_check"]
        print(f"{r['run_id']}: agree {c['agree']}/{c['records_checked']}, disagree {c['disagree']} "
              f"(tie {c['disagree_at_tie_sim_equals_tau']}, non-tie {c['disagree_not_at_tie']}) [{c['severity']}]")
    print(f"wrote legacy_tier1_sweep.json/.csv/_summary.md to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
