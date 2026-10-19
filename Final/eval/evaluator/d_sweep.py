"""Offline tau sweep and FCR-MCR frontier for REAL bge-m3 config-D runs (raw cosine similarity).

Built entirely from the stored `pairwise_eval.jsonl` (`similarity_score` = the raw, pre-gating cosine, and
`compatibility`); no model is loaded or needed. Per tau the host decision is recomputed with the existing strict
sweep rule (evaluator.sweep.derived_merge):

    compatibility in {incompatible, underdetermined} -> never merge
    otherwise                                        -> merge iff similarity_score > tau      (strict)

so a gate-OFF run (compatibility null) is the embedding threshold alone, and a gate-ON run (D/lexicon) is the
embedding threshold plus the DK-Mem lexicon gate. The two conditions are always kept separate. FCR/MCR come from the
existing `metrics.pair_outcome` / `fcr` / `mcr`; CIs from the existing relation-stratified bootstrap
(`contract_report.cell_with_ci`). Nothing here redefines a metric.

Only runs whose manifest identifies them as real bge-m3 runs are swept. Placeholder difflib D runs stay valid
but are listed as excluded (a difflib score is not a cosine and is never mixed in). The run-level tau stored in a
manifest (0.5 for the first artifact) is provenance only: it is not part of the grid and no result at it is reported,
except an internal consistency check that the sweep rule reproduces the recorded decisions.

Grid (BGE_M3_GRID, a documented choice, NOT a selected threshold): the first real D/off run scored 0.714-0.998, so the
generic 0.00-1.00 grid is replaced, for bge-m3 D only, by [0.700, 1.000]:
    0.700-0.800 step 0.010   (11 points, sparse tail where only same-entity pairs live)
    0.805-0.990 step 0.005   (38 points, the dense transition region)
    0.991-1.000 step 0.001   (10 points, resolves the top end)
In addition the EXACT frontier is computed at every distinct observed score (the step function is constant between
scores), so the grid cannot hide a vertex of the frontier.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .contract_report import N_BOOT, cell_with_ci
from .metrics import fcr, mcr, pair_details
from .report import find_runs
from .sweep import check_recorded_decisions, rows_at_tau
from .validate import DEFAULT_SCHEMA_DIR, GoldIndex, RunReport, validate_gold_file, validate_run

PREFIX = "D_BGE_M3_PROVISIONAL_sweep"
BANNER = ("D_BGE_M3 / PROVISIONAL: offline tau sweep from stored raw cosine scores (real BAAI/bge-m3), seed 0. "
          "Tier 1 v2, LLM-annotated, provisional gold. The embedding-threshold baseline alone is not evidence for or "
          "against DK-Mem. Not a final research result.")
LEGACY_NOTE = ("The 173-pair gold set includes four legacy classes (evidentiality, classifier_measure, "
               "spatial_temporal_deixis, politeness_relationship) that lie outside the pairwise schema enum; they are "
               "reported separately and are not final research results.")
GRID_VERSION = "bge_m3_d_grid_v1"
COSINE_RANGE = [-1.0, 1.0]
FCR_TARGETS = (0.20, 0.15, 0.10, 0.05, 0.02, 0.0)
MCR_TARGETS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def bge_m3_grid() -> list[float]:
    g = [round(0.70 + 0.01 * i, 3) for i in range(11)]
    g += [round(0.805 + 0.005 * i, 3) for i in range(38)]
    g += [round(0.991 + 0.001 * i, 3) for i in range(10)]
    return g


BGE_M3_GRID = bge_m3_grid()


# ------------------------------------------------------------------ which runs are swept

def classify_manifest(m: dict | None) -> str:
    """'bge-m3' for a real bge-m3 D run, 'difflib_placeholder' for a placeholder D run, else 'other'."""
    m = m or {}
    if m.get("pipeline_config") != "D":
        return "other"
    if (m.get("similarity_backend") == "bge-m3" and m.get("similarity_model_id") == "BAAI/bge-m3"
            and m.get("similarity_metric") == "bge-m3_dense_cosine_v1" and m.get("similarity_is_placeholder") is False
            and m.get("similarity_range") == COSINE_RANGE and m.get("similarity_model_revision")):
        return "bge-m3"
    return "difflib_placeholder" if m.get("similarity_metric") == "difflib_ratio_v1" else "other"


# ------------------------------------------------------------------ sweep

def _pt(cell: dict) -> dict:
    return {"n": cell["numerator"], "d": cell["denominator"], "rate": cell["rate"],
            "lo": cell.get("ci", {}).get("lo"), "hi": cell.get("ci", {}).get("hi")}


def exact_taus(report: RunReport) -> list[float]:
    """tau values that enumerate every distinct frontier state: below the minimum (everything above tau) and each
    observed score (merge iff score > tau, so a pair AT tau is not merged)."""
    scores = sorted({r["similarity_score"] for recs in report.pair_records.values() for r in recs})
    return [scores[0] - 1e-9] + scores if scores else []


def sweep_one(report: RunReport, gold: GoldIndex, grid: list[float], n_boot: int, cache: dict) -> dict:
    base = pair_details(report, gold)
    classes = sorted({r["distinction_class"] for r in base})
    points = []
    for tau in grid:
        rows = rows_at_tau(base, report, tau)
        c = cell_with_ci(rows, f"d_sweep|tau={tau:.4f}", n_boot, cache)
        by_cls = defaultdict(list)
        for r in rows:
            by_cls[r["distinction_class"]].append(r)
        points.append({"tau": tau, "fcr": _pt(c["fcr"]), "mcr": _pt(c["mcr"]), "n_pairs_merged": sum(r["merged"] for r in rows),
                       "per_class": {k: {"fcr": _pt_nd(fcr(by_cls[k])), "mcr": _pt_nd(mcr(by_cls[k]))} for k in classes}})
    exact = []
    for tau in exact_taus(report):
        rows = rows_at_tau(base, report, tau)
        f, m = fcr(rows), mcr(rows)
        exact.append({"tau": tau, "fcr": f["rate"], "mcr": m["rate"], "fcr_n": f["numerator"], "mcr_n": m["numerator"],
                      "n_pairs_merged": sum(r["merged"] for r in rows)})
    manifest = report.manifest or {}
    return {"run_id": report.run_id, "strategy": report.strategy, "dkmem_mode": manifest.get("dkmem_mode"),
            "gate": "off (embedding threshold only)" if manifest.get("dkmem_mode") == "off"
                    else f"on ({manifest.get('dkmem_mode')}: embedding threshold + DK-Mem gate)",
            "seed": manifest.get("seed"), "model_revision": manifest.get("similarity_model_revision"), "validation_ok": report.ok,
            "run_level_tau_provenance_only": manifest.get("tau"),
            "rule_consistency_at_recorded_tau": _consistency(check_recorded_decisions(report, gold)),
            "n_scored_pairs": len(report.pair_records), "points": points, "exact": exact,
            "pareto_exact": pareto(exact), "operating_points": operating_points(exact)}


def _pt_nd(cell: dict) -> dict:
    return {"n": cell["numerator"], "d": cell["denominator"], "rate": cell["rate"]}


def _consistency(c: dict) -> dict:
    return {"records_checked": c["records_checked"], "agree": c["agree"], "disagree": c["disagree"],
            "disagree_not_at_tie": c["disagree_not_at_tie"], "severity": c["severity"],
            "note": "internal check only: does the strict sweep rule reproduce the decisions recorded at the run-level tau?"}


# ------------------------------------------------------------------ frontier

def pareto(exact: list[dict]) -> list[dict]:
    """Non-dominated (FCR, MCR) states (both lower is better) with the tau interval that realises each.

    A state holds for tau in [tau_from, tau_to): tau_from is the observed score (or just below the minimum) that
    produces it and tau_to is the next observed score (None for the last state)."""
    states, taus = {}, [p["tau"] for p in exact]
    for i, p in enumerate(exact):
        key = (p["fcr"], p["mcr"])
        lo = p["tau"]
        hi = taus[i + 1] if i + 1 < len(taus) else None
        s = states.setdefault(key, {"fcr": p["fcr"], "mcr": p["mcr"], "fcr_n": p["fcr_n"], "mcr_n": p["mcr_n"], "intervals": []})
        s["intervals"].append([lo, hi])
    pts = list(states.values())
    front = [p for p in pts if not any(q["fcr"] <= p["fcr"] and q["mcr"] <= p["mcr"] and (q["fcr"] < p["fcr"] or q["mcr"] < p["mcr"]) for q in pts)]
    return sorted(front, key=lambda p: (p["fcr"], p["mcr"]))


def operating_points(exact: list[dict]) -> dict:
    """Best achievable trade-offs read off the exact frontier: lowest MCR at each FCR ceiling and lowest FCR at each
    MCR ceiling (the 'lower FCR at matched MCR' view). Descriptive: no threshold is selected."""
    front = pareto(exact)

    def pick(points, key):
        best = min(points, key=key) if points else None
        return None if best is None else {"fcr": best["fcr"], "mcr": best["mcr"], "fcr_n": best["fcr_n"], "mcr_n": best["mcr_n"],
                                         "tau_intervals": best["intervals"]}
    return {"min_mcr_given_fcr_at_most": {str(t): pick([p for p in front if p["fcr"] <= t + 1e-12], lambda p: (p["mcr"], p["fcr"])) for t in FCR_TARGETS},
            "min_fcr_given_mcr_at_most": {str(t): pick([p for p in front if p["mcr"] <= t + 1e-12], lambda p: (p["fcr"], p["mcr"])) for t in MCR_TARGETS}}


def compare_frontiers(off: dict, gated: dict) -> dict:
    """Matched-MCR comparison of the exact frontiers of a gate-OFF and a gate-ON run (same embedding, same pairs)."""
    def min_fcr(run, ceiling):
        v = [p["fcr"] for p in run["pareto_exact"] if p["mcr"] <= ceiling + 1e-12]
        return min(v) if v else None
    levels = sorted({p["mcr"] for r in (off, gated) for p in r["pareto_exact"]})
    rows = [{"mcr_ceiling": m, "fcr_off": min_fcr(off, m), "fcr_gated": min_fcr(gated, m)} for m in levels]
    both = [r for r in rows if r["fcr_off"] is not None and r["fcr_gated"] is not None]
    # per-tau containment: gating can only remove merges
    po, pg = {p["tau"]: p for p in off["points"]}, {p["tau"]: p for p in gated["points"]}
    common = sorted(set(po) & set(pg))
    return {"off_run": off["run_id"], "gated_run": gated["run_id"], "matched_mcr": rows,
            "levels_where_gated_fcr_lower": sum(r["fcr_gated"] < r["fcr_off"] for r in both),
            "levels_where_gated_fcr_equal": sum(r["fcr_gated"] == r["fcr_off"] for r in both),
            "levels_where_gated_fcr_higher": sum(r["fcr_gated"] > r["fcr_off"] for r in both),
            "grid_taus_where_gated_fcr_not_above_off": sum(pg[t]["fcr"]["rate"] <= po[t]["fcr"]["rate"] + 1e-12 for t in common),
            "grid_taus_where_gated_mcr_not_below_off": sum(pg[t]["mcr"]["rate"] >= po[t]["mcr"]["rate"] - 1e-12 for t in common),
            "grid_taus_compared": len(common)}


def score_identity(off_rep: RunReport, gated_rep: RunReport) -> dict:
    """Are the stored similarity scores of a gate-off and a gate-on run bit-identical, record for record?"""
    def scores(rep):
        return {r["record_id"]: r["similarity_score"] for recs in rep.pair_records.values() for r in recs}
    a, b = scores(off_rep), scores(gated_rep)
    same_ids = set(a) == set(b)
    diffs = [abs(a[k] - b[k]) for k in set(a) & set(b)]
    return {"off_run": off_rep.run_id, "gated_run": gated_rep.run_id, "n_off": len(a), "n_gated": len(b),
            "same_record_ids": same_ids, "n_compared": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
            "identical": bool(same_ids and diffs and all(d == 0.0 for d in diffs))}


def matched_tau(off: dict, gated: dict) -> dict:
    """Per grid tau: gate-off vs gate-on FCR/MCR (counts are exact integers; differences are gated minus off).
    Both runs share the embedding, so at one tau the gate can only remove merges (FCR can only fall, MCR only rise)."""
    po, pg = {p["tau"]: p for p in off["points"]}, {p["tau"]: p for p in gated["points"]}
    rows = []
    for t in sorted(set(po) & set(pg)):
        o, g = po[t], pg[t]
        dfn, dmn = g["fcr"]["n"] - o["fcr"]["n"], g["mcr"]["n"] - o["mcr"]["n"]
        kind = ("no_change" if dfn == 0 and dmn == 0 else
                "strict_improvement (FCR lower, MCR not higher)" if dfn < 0 and dmn <= 0 else
                "strict_improvement (MCR lower, FCR not higher)" if dmn < 0 and dfn <= 0 else
                "trade_off (FCR lower, MCR higher)" if dfn < 0 and dmn > 0 else
                "worse (FCR not lower, MCR higher)" if dmn > 0 else "other")
        rows.append({"tau": t, "fcr_off": o["fcr"]["rate"], "fcr_gated": g["fcr"]["rate"], "d_fcr": g["fcr"]["rate"] - o["fcr"]["rate"],
                     "fcr_off_n": o["fcr"]["n"], "fcr_gated_n": g["fcr"]["n"],
                     "mcr_off": o["mcr"]["rate"], "mcr_gated": g["mcr"]["rate"], "d_mcr": g["mcr"]["rate"] - o["mcr"]["rate"],
                     "mcr_off_n": o["mcr"]["n"], "mcr_gated_n": g["mcr"]["n"], "outcome": kind})
    counts: dict = defaultdict(int)
    for r in rows:
        counts[r["outcome"]] += 1
    return {"off_run": off["run_id"], "gated_run": gated["run_id"], "rows": rows, "outcome_counts": dict(counts),
            "strict_improvement_taus": [r["tau"] for r in rows if r["outcome"].startswith("strict_improvement")]}


def _dominates(q: dict, p: dict) -> bool:
    """q Pareto-dominates p: no worse on both rates, strictly better on at least one (counts are exact integers)."""
    return q["fcr_n"] <= p["fcr_n"] and q["mcr_n"] <= p["mcr_n"] and (q["fcr_n"] < p["fcr_n"] or q["mcr_n"] < p["mcr_n"])


def _weakly(q: dict, p: dict) -> bool:
    return q["fcr_n"] <= p["fcr_n"] and q["mcr_n"] <= p["mcr_n"]


def dominance(off: dict, gated: dict) -> dict:
    """Empirical dominance checks between a gate-off and a gate-on run; neither direction is assumed."""
    def gp(run):     # grid points as count states
        return [{"tau": p["tau"], "fcr_n": p["fcr"]["n"], "mcr_n": p["mcr"]["n"]} for p in run["points"]]
    og, gg = gp(off), gp(gated)
    g_over_o = [(q["tau"], p["tau"]) for q in gg for p in og if _dominates(q, p)]
    o_over_g = [(q["tau"], p["tau"]) for q in og for p in gg if _dominates(q, p)]
    of, gf = off["pareto_exact"], gated["pareto_exact"]
    return {
        "grid": {"gated_points_dominating_an_off_point": sorted({a for a, _ in g_over_o}),
                 "off_points_dominated_by_a_gated_point": sorted({b for _, b in g_over_o}),
                 "n_dominating_pairs_gated_over_off": len(g_over_o),
                 "off_points_dominating_a_gated_point": sorted({a for a, _ in o_over_g}),
                 "gated_points_dominated_by_an_off_point": sorted({b for _, b in o_over_g}),
                 "n_dominating_pairs_off_over_gated": len(o_over_g),
                 "n_off_points": len(og), "n_gated_points": len(gg),
                 "within_run_dominating_pairs": {"off": sum(_dominates(q, p) for q in og for p in og),
                                                 "gated": sum(_dominates(q, p) for q in gg for p in gg)},
                 "within_run_note": "a run's own grid points also dominate each other (a higher tau can keep MCR equal and lower FCR); "
                                    "these counts are the baseline for reading the cross-run pair counts, and the exact-frontier "
                                    "check below is the clean test"},
        "exact_frontier": {
            "gated_vertices_dominating_an_off_vertex": [{"gated": (q["fcr_n"], q["mcr_n"]), "off": (p["fcr_n"], p["mcr_n"])}
                                                        for q in gf for p in of if _dominates(q, p)],
            "off_vertices_dominated_by_a_gated_vertex": [(p["fcr_n"], p["mcr_n"]) for p in of if any(_dominates(q, p) for q in gf)],
            "gated_vertices_not_matched_or_beaten_by_off": [{"fcr": p["fcr"], "mcr": p["mcr"], "fcr_n": p["fcr_n"], "mcr_n": p["mcr_n"],
                                                             "tau_intervals": p["intervals"]}
                                                            for p in gf if not any(_weakly(q, p) for q in of)],
            "off_vertices_not_matched_or_beaten_by_gated": [{"fcr": p["fcr"], "mcr": p["mcr"], "fcr_n": p["fcr_n"], "mcr_n": p["mcr_n"],
                                                             "tau_intervals": p["intervals"]}
                                                            for p in of if not any(_weakly(q, p) for q in gf)],
            "gated_vertices_identical_to_an_off_vertex": sum(any((q["fcr_n"], q["mcr_n"]) == (p["fcr_n"], p["mcr_n"]) for q in of) for p in gf),
            "n_off_vertices": len(of), "n_gated_vertices": len(gf)},
        "note": "empirical, count-based (FCR numerator /98, MCR numerator /75); 'dominates' = no worse on both and strictly better on one"}


# ------------------------------------------------------------------ build / outputs

def build(gold_path: Path, runs_root: Path, n_boot: int = N_BOOT, grid: list[float] | None = None,
          schema_dir: Path = DEFAULT_SCHEMA_DIR) -> dict:
    grid = grid or BGE_M3_GRID
    gold = validate_gold_file(gold_path)
    cache: dict = {}
    runs, excluded, ignored, reports = [], [], 0, {}
    for d in find_runs(Path(runs_root)):
        try:
            m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        kind = classify_manifest(m)
        if kind == "other":
            ignored += 1
            continue
        if kind == "difflib_placeholder":
            excluded.append({"run_id": m.get("run_id"), "reason": "placeholder difflib similarity (valid, kept as is); "
                             "not a cosine, never mixed into the bge-m3 sweep"})
            continue
        rep = validate_run(d, gold, schema_dir)
        runs.append(sweep_one(rep, gold, grid, n_boot, cache))
        reports[rep.run_id] = rep
    runs.sort(key=lambda r: (r["dkmem_mode"] != "off", r["seed"] if r["seed"] is not None else -1, r["run_id"]))
    off = [r for r in runs if r["dkmem_mode"] == "off"]
    gated = [r for r in runs if r["dkmem_mode"] != "off"]
    pairs = [(o, g) for o in off for g in gated if o["model_revision"] == g["model_revision"] and o["seed"] == g["seed"]]      # a gate-off run is only paired with the gate-on run of the SAME seed
    identity = [score_identity(reports[o["run_id"]], reports[g["run_id"]]) for o, g in pairs]
    ok = [(o, g) for (o, g), i in zip(pairs, identity) if i["identical"]]      # compare only after the scores are verified identical
    out = {"banner": BANNER, "legacy_note": LEGACY_NOTE, "gold": str(gold_path),
           "rule": "merge iff similarity_score > tau (strict) AND compatibility not in {incompatible, underdetermined}",
           "similarity": "raw cosine of L2-normalised BAAI/bge-m3 dense embeddings, range [-1, 1], not mapped",
           "grid": {"version": GRID_VERSION, "taus": grid, "n": len(grid), "min": min(grid), "max": max(grid),
                    "segments": [{"from": 0.70, "to": 0.80, "step": 0.01}, {"from": 0.805, "to": 0.99, "step": 0.005},
                                 {"from": 0.991, "to": 1.0, "step": 0.001}],
                    "rationale": "covers the observed 0.714-0.998 range; finest steps in the dense region; no tau is selected"},
           "bootstrap": {"n_boot": n_boot, "method": "existing relation-stratified percentile bootstrap, 95% CI, "
                         "label-seeded per tau; evaluation-pair uncertainty only (one seed)"},
           "runs": runs, "excluded_runs": excluded, "ignored_other_config_runs": ignored,
           "frontier_comparison": [compare_frontiers(o, g) for o, g in ok],
           "score_identity": identity,
           "matched_tau": [matched_tau(o, g) for o, g in ok],
           "dominance": [dict(off_run=o["run_id"], gated_run=g["run_id"], **dominance(o, g)) for o, g in ok],
           "frontier_comparison_status": ("computed" if ok else "BLOCKED: similarity scores not identical" if pairs else
                                          "pending: needs both a gate-off and a gate-on (D/lexicon) bge-m3 run")}
    return out


def _c(p: dict) -> str:
    if p["d"] == 0:
        return "n/a"
    ci = f" [{p['lo']:.3f}, {p['hi']:.3f}]" if p["lo"] is not None else ""
    return f"{p['rate']:.3f}{ci} ({p['n']}/{p['d']})"


def render_markdown(out: dict) -> str:
    g = out["grid"]
    L = [f"> **{out['banner']}**", "", f"> {out['legacy_note']}", "", "# Config D (real bge-m3) tau sweep and FCR-MCR frontier", "",
         f"- Rule: {out['rule']}.", f"- Similarity: {out['similarity']}.",
         f"- FCR = merged gold-different / 98; MCR = unmerged gold-same / 75; no-entry pairs stay in the denominators (D has none).",
         f"- Grid `{g['version']}`: {g['n']} taus in [{g['min']}, {g['max']}]: 0.700-0.800 step 0.010; 0.805-0.990 step 0.005; "
         f"0.991-1.000 step 0.001. {g['rationale']}.",
         f"- CIs: {out['bootstrap']['method']} ({out['bootstrap']['n_boot']} replicates).",
         "- The run-level tau stored in a manifest is provenance only and is not analysed.", ""]
    for r in out["runs"]:
        L += [f"## {r['run_id']}: {r['gate']}", "", f"Model revision `{r['model_revision']}`; validation ok = {r['validation_ok']}; "
              f"rule-consistency check at the recorded tau: {r['rule_consistency_at_recorded_tau']['agree']}/"
              f"{r['rule_consistency_at_recorded_tau']['records_checked']} records agree.", "",
              "| tau | FCR [95% CI] | MCR [95% CI] | pairs merged |", "|---|---|---|---|"]
        for p in r["points"]:
            L.append(f"| {p['tau']:.3f} | {_c(p['fcr'])} | {_c(p['mcr'])} | {p['n_pairs_merged']} |")
        L += ["", f"### Exact Pareto frontier ({len(r['pareto_exact'])} vertices; tau intervals are [from, to), merge iff score > tau)", "",
              "| FCR | MCR | tau interval(s) |", "|---|---|---|"]
        for p in r["pareto_exact"]:
            iv = "; ".join(f"[{a:.4f}, {('inf' if b is None else format(b, '.4f'))})" for a, b in p["intervals"])
            L.append(f"| {p['fcr']:.3f} ({p['fcr_n']}/98) | {p['mcr']:.3f} ({p['mcr_n']}/75) | {iv} |")
        L += ["", "### Operating points on the exact frontier (descriptive, no tau selected)", "",
              "| constraint | FCR | MCR | tau interval |", "|---|---|---|---|"]
        for kind, label in (("min_mcr_given_fcr_at_most", "FCR <="), ("min_fcr_given_mcr_at_most", "MCR <=")):
            for t, v in r["operating_points"][kind].items():
                if v is None:
                    L.append(f"| {label} {t} | not achievable | | |")
                else:
                    iv = "; ".join(f"[{a:.4f}, {('inf' if b is None else format(b, '.4f'))})" for a, b in v["tau_intervals"])
                    L.append(f"| {label} {t} | {v['fcr']:.3f} | {v['mcr']:.3f} | {iv} |")
        L.append("")
    L += ["## Gate-off vs gate-on frontier", "", f"Status: {out['frontier_comparison_status']}.", ""]
    for c in out["frontier_comparison"]:
        L.append(f"- {c['off_run']} vs {c['gated_run']}: gated FCR lower at {c['levels_where_gated_fcr_lower']} / equal at "
                 f"{c['levels_where_gated_fcr_equal']} / higher at {c['levels_where_gated_fcr_higher']} matched-MCR levels.")
    for i in out.get("score_identity", []):
        L.append(f"- Score identity check ({i['off_run']} vs {i['gated_run']}): same record ids = {i['same_record_ids']}, "
                 f"{i['n_compared']} records compared, max |diff| = {i['max_abs_diff']}, identical = {i['identical']}.")
    for c in out["frontier_comparison"]:
        L += ["", "### Matched-MCR comparison (min FCR achievable at MCR <= level, exact frontiers)", "",
              "| MCR ceiling | FCR gate-off | FCR gate-on |", "|---|---|---|"]
        for r in c["matched_mcr"]:
            f = lambda v: "n/a" if v is None else f"{v:.3f}"
            L.append(f"| {r['mcr_ceiling']:.3f} | {f(r['fcr_off'])} | {f(r['fcr_gated'])} |")
    for m in out.get("matched_tau", []):
        L += ["", f"### Matched-tau comparison ({m['off_run']} vs {m['gated_run']}; delta = gate-on minus gate-off)", "",
              f"Outcome counts over the grid: {m['outcome_counts']}.", "",
              "| tau | FCR off | FCR on | dFCR | MCR off | MCR on | dMCR | outcome |", "|---|---|---|---|---|---|---|---|"]
        for r in m["rows"]:
            L.append(f"| {r['tau']:.3f} | {r['fcr_off']:.3f} ({r['fcr_off_n']}) | {r['fcr_gated']:.3f} ({r['fcr_gated_n']}) | {r['d_fcr']:+.3f} | "
                     f"{r['mcr_off']:.3f} ({r['mcr_off_n']}) | {r['mcr_gated']:.3f} ({r['mcr_gated_n']}) | {r['d_mcr']:+.3f} | {r['outcome']} |")
    for d in out.get("dominance", []):
        g, e = d["grid"], d["exact_frontier"]
        L += ["", f"### Dominance ({d['off_run']} vs {d['gated_run']}; {d['note']})", "",
              f"- Grid: gate-on points dominating a gate-off point: {len(g['gated_points_dominating_an_off_point'])} of {g['n_gated_points']} "
              f"({g['n_dominating_pairs_gated_over_off']} dominating pairs; within-run baseline: off {g['within_run_dominating_pairs']['off']}, gate-on {g['within_run_dominating_pairs']['gated']}); gate-off points dominating a gate-on point: "
              f"{len(g['off_points_dominating_a_gated_point'])} ({g['n_dominating_pairs_off_over_gated']} pairs).",
              f"- Exact frontier: gate-on vertices dominating a gate-off vertex: {len(e['gated_vertices_dominating_an_off_vertex'])}; "
              f"gate-off vertices dominated by a gate-on vertex: {len(e['off_vertices_dominated_by_a_gated_vertex'])} of {e['n_off_vertices']}; "
              f"gate-on vertices identical to a gate-off vertex: {e['gated_vertices_identical_to_an_off_vertex']} of {e['n_gated_vertices']}.",
              f"- Gate-on vertices NOT matched or beaten by any gate-off state: {len(e['gated_vertices_not_matched_or_beaten_by_off'])}; "
              f"gate-off vertices NOT matched or beaten by any gate-on state: {len(e['off_vertices_not_matched_or_beaten_by_gated'])}."]
        for k, lab in (("gated_vertices_not_matched_or_beaten_by_off", "gate-on only"), ("off_vertices_not_matched_or_beaten_by_gated", "gate-off only")):
            for v in e[k]:
                L.append(f"  - {lab}: FCR {v['fcr']:.3f} ({v['fcr_n']}/98), MCR {v['mcr']:.3f} ({v['mcr_n']}/75)")
    if out["excluded_runs"]:
        L += ["", "## Excluded D runs (valid, but not real bge-m3)", ""] + [f"- {e['run_id']}: {e['reason']}" for e in out["excluded_runs"]]
    return "\n".join(L) + "\n"


def write_outputs(out: dict, outdir: Path, prefix: str = PREFIX) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)

    def text(name, s):
        with open(outdir / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
    text(f"{prefix}.json", json.dumps(out, indent=1, ensure_ascii=False))
    text(f"{prefix}_summary.md", render_markdown(out))
    with open(outdir / f"{prefix}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["run_id", "dkmem_mode", "tau", "fcr_n", "fcr_d", "fcr", "fcr_lo", "fcr_hi", "mcr_n", "mcr_d", "mcr", "mcr_lo", "mcr_hi", "pairs_merged"])
        for r in out["runs"]:
            for p in r["points"]:
                w.writerow([r["run_id"], r["dkmem_mode"], p["tau"], p["fcr"]["n"], p["fcr"]["d"], p["fcr"]["rate"], p["fcr"]["lo"], p["fcr"]["hi"],
                            p["mcr"]["n"], p["mcr"]["d"], p["mcr"]["rate"], p["mcr"]["lo"], p["mcr"]["hi"], p["n_pairs_merged"]])
    with open(outdir / f"{prefix}_frontier_exact.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["run_id", "dkmem_mode", "fcr_n", "fcr", "mcr_n", "mcr", "tau_from", "tau_to"])
        for r in out["runs"]:
            for p in r["pareto_exact"]:
                for lo, hi in p["intervals"]:
                    w.writerow([r["run_id"], r["dkmem_mode"], p["fcr_n"], p["fcr"], p["mcr_n"], p["mcr"], lo, "" if hi is None else hi])
    names = [f"{prefix}.json", f"{prefix}.csv", f"{prefix}_frontier_exact.csv", f"{prefix}_summary.md"]
    if out.get("matched_tau"):
        with open(outdir / f"{prefix}_matched_tau.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["off_run", "gated_run", "tau", "fcr_off_n", "fcr_off", "fcr_gated_n", "fcr_gated", "d_fcr",
                        "mcr_off_n", "mcr_off", "mcr_gated_n", "mcr_gated", "d_mcr", "outcome"])
            for m in out["matched_tau"]:
                for r in m["rows"]:
                    w.writerow([m["off_run"], m["gated_run"], r["tau"], r["fcr_off_n"], r["fcr_off"], r["fcr_gated_n"], r["fcr_gated"], r["d_fcr"],
                                r["mcr_off_n"], r["mcr_off"], r["mcr_gated_n"], r["mcr_gated"], r["d_mcr"], r["outcome"]])
        names.append(f"{prefix}_matched_tau.csv")
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline tau sweep + FCR-MCR frontier for real bge-m3 D runs (no model is loaded).")
    ap.add_argument("--gold", type=Path, default=Path(__file__).resolve().parents[1] / "teamB_answer_key.jsonl")
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--prefix", default=PREFIX, help="output file prefix (use a new one so earlier sweep outputs are not overwritten)")
    a = ap.parse_args(argv)
    for n in (f"{a.prefix}.json", f"{a.prefix}_summary.md"):
        if (a.out / n).exists():
            raise SystemExit(f"refusing to overwrite existing {a.out / n}; choose a different --prefix")
    out = build(a.gold, a.runs, a.n_boot)
    print("wrote", *write_outputs(out, a.out, a.prefix), sep="\n  ")
    print(f"runs swept: {[r['run_id'] for r in out['runs']]}; excluded: {[e['run_id'] for e in out['excluded_runs']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
