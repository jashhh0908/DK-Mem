"""Evaluate every run directory under a results root and write machine- and human-readable output.

    python -m evaluator.report --gold teamB_answer_key.jsonl --runs tier1_results --out results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import evaluate_run, seed_summary
from .validate import DEFAULT_SCHEMA_DIR, check_seed_group, validate_gold_file, validate_run

DEFAULT_LABEL = "LEGACY / INITIAL Tier 1 results (not final research results)"
CONTRACT_LABEL = "Tier 1 v2, LLM-annotated, provisional"


def find_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/run_manifest.json") if (p.parent / "pairwise_eval.jsonl").exists())


def _c(cell: dict) -> str:
    s = f"{cell['display']} ({cell['numerator']}/{cell['denominator']})"
    return s + "*" if cell["indicative"] else s


def build(gold_path: Path, runs_root: Path, schema_dir: Path = DEFAULT_SCHEMA_DIR, label: str = DEFAULT_LABEL) -> dict:
    gold = validate_gold_file(gold_path)
    reports = [validate_run(d, gold, schema_dir) for d in find_runs(runs_root)]
    results = [evaluate_run(r, gold) for r in reports]
    groups = defaultdict(list)
    for rep, res in zip(reports, results):
        groups[rep.strategy].append((rep, res))
    return {
        "label": label, "contract_label": CONTRACT_LABEL,
        "definitions": {"FCR": "merged gold-different pairs / all gold-different pairs",
                        "MCR": "unmerged gold-same pairs / all gold-same pairs",
                        "note": "denominators include not-compared and no-entry pairs; * = n<20 (indicative); n/a = n=0"},
        "gold": {"path": str(gold_path), "pairs": len(gold.pairs), "distributions": gold.distributions,
                 "validation_errors": [i.to_dict() for i in gold.errors]},
        "runs": [{k: v for k, v in r.items() if k != "_rows"} for r in results],
        "strategies": {s: {"seed_summary": seed_summary([res for _, res in v]),
                           "seed_group_issues": [i.to_dict() for i in check_seed_group([rep for rep, _ in v])]}
                       for s, v in groups.items()},
        "_rows": [row for r in results for row in r["_rows"]],
    }


def render_markdown(out: dict) -> str:
    L = [f"# DK-Mem Tier 1 FCR / MCR - {out['label']}", "",
         f"**{out['contract_label']}.** Gold: `{out['gold']['path']}` ({out['gold']['pairs']} pairs).", "",
         "- FCR = merged gold-different pairs / all gold-different pairs (lower is better)",
         "- MCR = unmerged gold-same pairs / all gold-same pairs (lower is better)",
         "- Denominators include not-compared and no-entry pairs. `*` = n<20 (indicative). `n/a` = n=0.",
         "- Legacy runs: placeholder difflib similarity (tau 0.85); seeds within a strategy are identical, so there is no seed variance.",
         "- Do not headline overall FCR/MCR alone: class and relation are largely confounded.", "",
         "## Overall, by strategy and seed", "",
         "| run | strategy | seed | FCR | MCR | valid | no-entry pairs |", "|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        o = r["overall"]
        L.append(f"| {r['run_id']} | {r['strategy']} | {r['seed']} | {_c(o['fcr'])} | {_c(o['mcr'])} | "
                 f"{'ok' if r['validation_ok'] else 'ERRORS'} | {r['no_entry_diagnostics']['count']} |")
    for title, key in (("In-scope classes only (kinship, honorific_register, name_variant)", "in_scope"),
                       ("Legacy classes only (evidentiality, classifier_measure, spatial_temporal_deixis, politeness_relationship)", "legacy")):
        L += ["", f"## {title}", "", "| run | FCR | MCR |", "|---|---|---|"]
        for r in out["runs"]:
            s = r["scope_subsets"][key]
            L.append(f"| {r['run_id']} | {_c(s['fcr'])} | {_c(s['mcr'])} |")
    L += ["", "## Per distinction class (gold class)", ""]
    for r in out["runs"]:
        L += [f"### {r['run_id']}", "", "| class | scope | FCR | MCR | merged / cmp-not-merged / not-cmp / no-entry (same) | (different) |",
              "|---|---|---|---|---|---|"]
        for c, v in r["per_class"].items():
            o = v["outcomes"]
            f = lambda d: " / ".join(str(d[k]) for k in ("merged", "compared_not_merged", "not_compared", "no_entry"))
            L.append(f"| {c} | {v['scope']} | {_c(v['fcr'])} | {_c(v['mcr'])} | {f(o['same'])} | {f(o['different'])} |")
        L.append("")
    L += ["## No-entry diagnostics (kept separate from model decisions)", "",
          "An utterance with zero entries; the output does not record why (e.g. extraction failure). These pairs are unmerged under the contract.", ""]
    for r in out["runs"]:
        d = r["no_entry_diagnostics"]
        if d["count"]:
            L.append(f"- **{r['run_id']}**: {d['count']} pairs {d['by_structural_status']}; by relation {d['by_relation']}; "
                     f"by class {d['by_class']}; by language {d['by_language']}")
        else:
            L.append(f"- {r['run_id']}: 0")
    L += ["", "## Validation", ""]
    for r in out["runs"]:
        L.append(f"- {r['run_id']}: errors={r['validation_errors'] or 'none'}; warnings={r['validation_warnings'] or 'none'}")
    for s, v in out["strategies"].items():
        L.append(f"- seed group {s}: " + "; ".join(i["code"] for i in v["seed_group_issues"]))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FCR/MCR over all run directories under --runs.")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--prefix", default="legacy_tier1")
    a = ap.parse_args(argv)
    out = build(a.gold, a.runs, a.schema_dir, a.label)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = out.pop("_rows")
    (a.out / f"{a.prefix}_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(a.out / f"{a.prefix}_pair_details.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (a.out / f"{a.prefix}_summary.md").write_text(render_markdown(out), encoding="utf-8")
    print(f"wrote {a.prefix}_metrics.json, _pair_details.jsonl, _summary.md to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
