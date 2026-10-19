"""FCR and MCR for Tier 1, exactly as defined in tier1_eval_contract.md sections 4-6.

    FCR = #{gold 'different' pairs that are merged}   / #{gold 'different' pairs}
    MCR = #{gold 'same' pairs that are NOT merged}    / #{gold 'same' pairs}

Denominators are ALL gold pairs of that relation, including pairs that were not compared
or had no entry (those are never merged, so they are MCR misses and never FCR hits).

A pair is merged iff some entry of utt_a and some entry of utt_b end up in one cluster
under union-find over merge/supersede records (section 4). Only cross-utterance records
(already isolated by validate_run) can connect the two utterances, and any cross record
with decision merge/supersede directly unions an utt_a entry with an utt_b entry, so
"merged" == "at least one cross-utterance merge/supersede record".
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Any

from .validate import GoldIndex, IN_SCOPE_CLASSES, LEGACY_CLASSES, RunReport

MERGE_DECISIONS = {"merge", "supersede"}
INDICATIVE_N = 20          # contract section 8: flag any cell with n < 20
# outcome labels (contract section 4); no_entry is kept distinct from not_compared
OUTCOMES = ("merged", "compared_not_merged", "not_compared", "no_entry")


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    """One metric cell. n == 0 -> rate None ('n/a'), never 0. n < 20 -> indicative."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
        "display": "n/a" if denominator == 0 else f"{numerator / denominator:.3f}",
        "indicative": 0 < denominator < INDICATIVE_N,
    }


def recorded_merge(record: dict) -> bool:
    """The system's own recorded decision: merge or supersede."""
    return record["decision"] in MERGE_DECISIONS


def pair_outcome(status: str, cross_records: list[dict], is_merge=recorded_merge) -> dict[str, Any]:
    """Classify one gold pair from its Phase 2 structural status + cross-utterance records.

    `is_merge(record)` decides whether one record counts as a merge; the default is the
    recorded decision. The tau sweep passes its re-derived rule here, so the metric
    definitions are never duplicated.
    """
    merged = any(is_merge(r) for r in cross_records)
    if merged:
        outcome = "merged"
    elif status == "compared":
        outcome = "compared_not_merged"
    elif status == "not_compared":
        outcome = "not_compared"
    elif status.startswith("no_entry"):
        outcome = "no_entry"
    else:
        raise ValueError(f"unknown pair status {status!r}")
    return {
        "outcome": outcome,
        "merged": merged,
        "structural_status": status,
        "decisions": sorted(r["decision"] for r in cross_records),
        "compatibility": sorted({str(r.get("compatibility")) for r in cross_records}),
        "max_similarity": max((r["similarity_score"] for r in cross_records), default=None),
    }


def pair_details(report: RunReport, gold: GoldIndex) -> list[dict]:
    """One row per gold pair (all of them, nothing filtered) with its outcome."""
    rows = []
    for pid, g in gold.pairs.items():
        row = {
            "run_id": report.run_id, "strategy": report.strategy,
            "seed": (report.manifest or {}).get("seed"),
            "eval_pair_id": pid, "pair_id": g["pair_id"], "relation": g["relation"],
            "distinction_class": g["distinction_class"], "language": g["language"],
            "scope": "in_scope" if g["distinction_class"] in IN_SCOPE_CLASSES else "legacy",
        }
        row.update(pair_outcome(report.pair_status[pid], report.pair_records.get(pid, [])))
        rows.append(row)
    return rows


def fcr(rows: list[dict]) -> dict[str, Any]:
    """False consolidation rate over gold-'different' rows."""
    diff = [r for r in rows if r["relation"] == "different"]
    return rate(sum(r["merged"] for r in diff), len(diff))


def mcr(rows: list[dict]) -> dict[str, Any]:
    """Missed consolidation rate over gold-'same' rows."""
    same = [r for r in rows if r["relation"] == "same"]
    return rate(sum(not r["merged"] for r in same), len(same))


def outcome_breakdown(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Counts of the four contract outcomes split by gold relation; no_entry stays separate."""
    out = {rel: Counter() for rel in ("same", "different")}
    for r in rows:
        out[r["relation"]][r["outcome"]] += 1
    return {rel: {o: c[o] for o in OUTCOMES} for rel, c in out.items()}


def cell(rows: list[dict]) -> dict[str, Any]:
    return {"fcr": fcr(rows), "mcr": mcr(rows), "outcomes": outcome_breakdown(rows)}


def no_entry_diagnostics(rows: list[dict]) -> dict[str, Any]:
    """Pairs with an utterance that produced zero entries, kept apart from model decisions.

    The output files do not say WHY an utterance has no entry (e.g. extraction failure),
    so these are reported as 'no_entry', not as extraction failures.
    """
    ne = [r for r in rows if r["outcome"] == "no_entry"]
    return {
        "count": len(ne),
        "by_structural_status": dict(Counter(r["structural_status"] for r in ne)),
        "by_relation": dict(Counter(r["relation"] for r in ne)),
        "by_class": dict(Counter(r["distinction_class"] for r in ne)),
        "by_language": dict(Counter(r["language"] for r in ne)),
        "eval_pair_ids": sorted(r["eval_pair_id"] for r in ne),
    }


def evaluate_run(report: RunReport, gold: GoldIndex) -> dict[str, Any]:
    """FCR/MCR for one validated run. Returns (summary dict, pair rows) under keys."""
    rows = pair_details(report, gold)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["distinction_class"]].append(r)
    classes = sorted(set(IN_SCOPE_CLASSES) | set(LEGACY_CLASSES) | set(by_class))
    return {
        "run_id": report.run_id, "strategy": report.strategy,
        "seed": (report.manifest or {}).get("seed"),
        "backbone": (report.manifest or {}).get("backbone"),
        "validation_ok": report.ok,
        "validation_errors": [i.code for i in report.errors],
        "validation_warnings": [i.code for i in report.warnings],
        "overall": cell(rows),
        "scope_subsets": {
            "in_scope": cell([r for r in rows if r["scope"] == "in_scope"]),
            "legacy": cell([r for r in rows if r["scope"] == "legacy"]),
        },
        "per_class": {c: {"scope": "in_scope" if c in IN_SCOPE_CLASSES else "legacy", **cell(by_class.get(c, []))}
                      for c in classes},
        "no_entry_diagnostics": no_entry_diagnostics(rows),
        "_rows": rows,
    }


def seed_summary(results: list[dict]) -> dict[str, Any]:
    """Mean/range of overall FCR/MCR over a strategy's seeds (descriptive, no CIs)."""
    out = {"n_seeds": len(results)}
    for m in ("fcr", "mcr"):
        vals = [r["overall"][m]["rate"] for r in results if r["overall"][m]["rate"] is not None]
        out[m] = {"mean": fmean(vals) if vals else None, "min": min(vals, default=None), "max": max(vals, default=None)}
    return out
