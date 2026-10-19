"""Contract section 8 reporting + statistics for the legacy Tier 1 runs.

Per run: overall / per-class / per-language / class x language FCR and MCR (rates, counts,
n/a, n<20 flag) with paired-bootstrap 95% CIs, plus Cohen's kappa against the gold labels.
Per strategy: mean +/- SD across seeds, with an explicit seed-identity check.

    python -m evaluator.contract_report --gold teamB_answer_key.jsonl --runs tier1_results --out results

Every output is LEGACY / PROVISIONAL / MACHINERY VALIDATION ONLY (placeholder difflib similarity).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import cell, evaluate_run
from .report import CONTRACT_LABEL, find_runs
from .stats import BOOTSTRAP_SEED, METHOD, N_BOOT, bootstrap_ci, cohens_kappa, mean_sd
from .validate import DEFAULT_SCHEMA_DIR, validate_gold_file, validate_run

BANNER = "LEGACY / PROVISIONAL / MACHINERY VALIDATION ONLY"
BANNER_DETAIL = (f"{CONTRACT_LABEL}. Legacy Tier 1 runs scored with the placeholder difflib similarity; "
                 "not a research result.")
SEED_NOTE = ("Legacy seeds are decision-identical, so SD = 0 here is NOT empirical seed variance; "
             "it only reflects that the seed has no effect on these runs.")


def groupings(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    g: dict[str, dict[str, list[dict]]] = {"overall": {"all": rows}, "per_class": defaultdict(list),
                                           "per_language": defaultdict(list), "class_x_language": defaultdict(list)}
    for r in rows:
        g["per_class"][r["distinction_class"]].append(r)
        g["per_language"][r["language"]].append(r)
        g["class_x_language"][f"{r['distinction_class']}|{r['language']}"].append(r)
    return {k: dict(sorted(v.items())) for k, v in g.items()}


def cell_with_ci(rows: list[dict], label: str, n_boot: int, cache: dict) -> dict:
    out = cell(rows)
    out["n_pairs"] = len(rows)
    for m in ("fcr", "mcr"):
        key = (label, m, n_boot, tuple(sorted((r["relation"], r["merged"]) for r in rows)))
        if key not in cache:                      # seed depends only on label+metric, so identical data -> identical CI
            cache[key] = bootstrap_ci(rows, m, label=label, n_boot=n_boot)
        out[m] = {**out[m], "ci": {k: cache[key][k] for k in ("lo", "hi", "n_boot", "seed")}}
    return out


def run_tables(result: dict, n_boot: int, cache: dict) -> dict:
    rows = result["_rows"]
    tables = {name: {k: cell_with_ci(v, f"{name}:{k}", n_boot, cache) for k, v in groups.items()}
              for name, groups in groupings(rows).items()}
    return {"run_id": result["run_id"], "strategy": result["strategy"], "seed": result["seed"],
            "backbone": result["backbone"], "validation_ok": result["validation_ok"],
            "tables": tables, "kappa": {"overall": cohens_kappa(rows),
                                         "per_class": {k: cohens_kappa(v) for k, v in groupings(rows)["per_class"].items()}}}


def signature(result: dict) -> list:
    return sorted((r["eval_pair_id"], r["structural_status"], r["merged"]) for r in result["_rows"])


def strategy_seed_stats(runs: list[dict], identical: bool) -> dict:
    """Mean/SD across seeds for every cell's FCR and MCR rate."""
    out = {}
    for table in runs[0]["tables"]:
        out[table] = {}
        for key in runs[0]["tables"][table]:
            out[table][key] = {m: {**mean_sd([r["tables"][table][key][m]["rate"] for r in runs]),
                                   "denominator": runs[0]["tables"][table][key][m]["denominator"]}
                               for m in ("fcr", "mcr")}
    return {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs], "seeds_decision_identical": identical,
            "note": SEED_NOTE if identical else "Seeds differ; SD is empirical.",
            "min_seeds_met": len(runs) >= 3, "cells": out}


def build(gold_path: Path, runs_root: Path, n_boot: int = N_BOOT, schema_dir: Path = DEFAULT_SCHEMA_DIR) -> dict:
    gold = validate_gold_file(gold_path)
    results = [evaluate_run(validate_run(d, gold, schema_dir), gold) for d in find_runs(runs_root)]
    cache: dict = {}
    runs = [run_tables(r, n_boot, cache) for r in results]
    by_strategy = defaultdict(list)
    for res, run in zip(results, runs):
        by_strategy[res["strategy"]].append((res, run))
    strategies = {}
    for s, pairs in by_strategy.items():
        identical = len({json.dumps(signature(res)) for res, _ in pairs}) == 1
        strategies[s] = strategy_seed_stats([run for _, run in pairs], identical)
    return {"banner": BANNER, "detail": BANNER_DETAIL,
            "statistics": {"bootstrap": METHOD, "n_boot": n_boot, "base_seed": BOOTSTRAP_SEED, "ci": "95% percentile",
                           "kappa": "gold same/different vs predicted merged/not merged; no-entry and not-compared = not merged",
                           "sd": "sample SD (n-1) across seeds"},
            "gold": str(gold_path), "runs": runs, "strategies": strategies}


# ---------------------------------------------------------------- markdown

def _r(c: dict, m: str) -> str:
    x = c[m]
    if x["denominator"] == 0:
        return "n/a (n=0)"
    ci = x["ci"]
    s = f"{x['rate']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] ({x['numerator']}/{x['denominator']})"
    return s + "*" if x["indicative"] else s


def _table(cells: dict, first: str) -> list[str]:
    L = [f"| {first} | FCR [95% CI] | MCR [95% CI] |", "|---|---|---|"]
    for k, c in cells.items():
        L.append(f"| {k} | {_r(c, 'fcr')} | {_r(c, 'mcr')} |")
    return L


def _sd(x: dict) -> str:
    if x["mean"] is None:
        return "n/a"
    return f"{x['mean']:.3f} ± {x['sd']:.3f}" if x["sd"] is not None else f"{x['mean']:.3f} (1 seed)"


def render_markdown(out: dict) -> str:
    st = out["statistics"]
    L = [f"# {out['banner']}", "", out["detail"], "",
         f"- Bootstrap: {st['bootstrap']}; {st['n_boot']} replicates, base seed {st['base_seed']}; 95% percentile CI. "
         "Uncertainty over evaluation pairs only, not over seeds.",
         f"- Kappa: {st['kappa']}. This is system-vs-gold agreement, not inter-annotator kappa.",
         "- `*` = n<20 (indicative). `n/a (n=0)` = no gold pairs of that relation. A point estimate of 0 or 1 gives a degenerate CI.",
         "- Do not headline overall FCR/MCR alone: class and relation are largely confounded (see contract section 8).", "",
         "## Runs", "", "| run | strategy | seed | backbone | valid |", "|---|---|---|---|---|"]
    for r in out["runs"]:
        L.append(f"| {r['run_id']} | {r['strategy']} | {r['seed']} | {r['backbone']} | {'ok' if r['validation_ok'] else 'ERRORS'} |")
    L += ["", "## Mean ± SD across seeds (overall)", "", "| strategy | seeds | FCR | MCR | decision-identical |", "|---|---|---|---|---|"]
    for s, v in out["strategies"].items():
        o = v["cells"]["overall"]["all"]
        L.append(f"| {s} | {v['n_seeds']} {v['seeds']} | {_sd(o['fcr'])} | {_sd(o['mcr'])} | {v['seeds_decision_identical']} |")
    L += ["", f"> {SEED_NOTE}", ""]
    shown, seen = [], set()
    for r in out["runs"]:
        if r["strategy"] not in seen:
            seen.add(r["strategy"])
            shown.append(r)
    for r in shown:
        t = r["tables"]
        L += [f"## {r['strategy']} (run {r['run_id']}; other seeds identical when the table above says so)", "",
              "### Overall", ""] + _table(t["overall"], "scope")
        L += ["", "### Per distinction class", ""] + _table(t["per_class"], "class")
        L += ["", "### Per language", ""] + _table(t["per_language"], "language")
        L += ["", "### Class × language", ""] + _table(t["class_x_language"], "class|language")
        k = r["kappa"]["overall"]
        kv = "undefined (" + k["undefined_reason"] + ")" if k["kappa"] is None else f"{k['kappa']:.3f}"
        L += ["", f"### Cohen's kappa vs gold (n={k['n']})", "",
              f"kappa = {kv}; observed agreement {k['observed_agreement']:.3f}, expected {k['expected_agreement']:.3f}; "
              f"confusion {k['confusion']}", "",
              "Per class: " + "; ".join(
                  f"{c}: n={v['n']}, " + (f"kappa {v['kappa']:.3f}" if v["kappa"] is not None else "kappa undefined")
                  for c, v in r["kappa"]["per_class"].items()), ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Contract reporting + bootstrap CIs + kappa (" + BANNER + ").")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    a = ap.parse_args(argv)
    out = build(a.gold, a.runs, a.n_boot, a.schema_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "legacy_tier1_contract_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (a.out / "legacy_tier1_contract_report.md").write_text(render_markdown(out), encoding="utf-8")
    print("wrote legacy_tier1_contract_report.json/.md to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
