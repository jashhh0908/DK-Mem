"""CLI: python -m evaluator --gold teamB_answer_key.jsonl --run DIR [--run DIR ...] [--json OUT]"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .validate import DEFAULT_SCHEMA_DIR, check_seed_group, validate_gold_file, validate_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate Tier 1 run(s) against the Team B answer key (no metrics).")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, type=Path, help="run directory (repeatable)")
    ap.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    ap.add_argument("--json", type=Path, help="write the full machine-readable report here")
    args = ap.parse_args(argv)

    gold = validate_gold_file(args.gold)
    print(f"GOLD {args.gold}: {len(gold.pairs)} pairs, {len(gold.errors)} error type(s)")
    for i in gold.issues:
        print(f"  [{i.severity}] {i.code} x{i.count}: {i.message} {i.examples}")
    reports = [validate_run(r, gold, args.schema_dir) for r in args.run]
    for rep in reports:
        c = rep.counts
        print(f"\nRUN {rep.run_id} ({rep.strategy}) ok={rep.ok}")
        print(f"  records={c['pairwise_records']} pair_status={c['pair_status']} ignored={c['ignored_records']}")
        for i in rep.issues:
            ex = f" e.g. {i.examples[:2]}" if i.examples else ""
            print(f"  [{i.severity}] {i.code} x{i.count}: {i.message}{ex}")
    groups = defaultdict(list)
    for rep in reports:
        groups[rep.strategy].append(rep)
    group_issues = {s: check_seed_group(rs) for s, rs in groups.items()}
    for s, issues in group_issues.items():
        print(f"\nSEED GROUP {s} ({len(groups[s])} run(s))")
        for i in issues:
            print(f"  [{i.severity}] {i.code} x{i.count}: {i.message} {i.examples}")
    if args.json:
        args.json.write_text(json.dumps({
            "gold": {"path": str(args.gold), "pairs": len(gold.pairs),
                     "issues": [i.to_dict() for i in gold.issues], "distributions": gold.distributions},
            "runs": [r.to_dict() for r in reports],
            "seed_groups": {str(s): [i.to_dict() for i in v] for s, v in group_issues.items()},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if gold.errors or any(not r.ok for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
