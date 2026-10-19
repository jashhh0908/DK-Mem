"""Preliminary Tier-2 full-run report (EVALUATOR side). Reads the four t2full runs + their contract reports and the frozen benchmark; changes nothing.

TIER-2 FULL RUN - PROVISIONAL, NOT YET AUDITED. FCR-T2 / MCR-T2 use the Tier-1 definitions on the injected pairs:
  FCR-T2 = injected gold-different pairs whose two turns were directly merged (record decision merge|supersede) / all gold-different injected pairs
  MCR-T2 = injected gold-same pairs NOT directly merged / all gold-same injected pairs
Pairs that were never compared (retrieval miss) or whose judge answer was unparseable stay in the denominators as unmerged and are ALSO reported separately
(the retrieval-miss rate is never folded into a merge decision). CIs, where shown, reuse the existing relation-stratified percentile bootstrap.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.contract_report import N_BOOT, cell_with_ci

from . import contract
from . import generate_tier2 as G
from . import judge_provenance, run_tier2

BANNER = "TIER-2 FULL RUN — PROVISIONAL, NOT YET AUDITED"
CONFIGS = (("C/off", "t2full-store-surface-only-C_seed0"), ("C/lexicon", "t2full-dk-mem-lexicon-C_seed0"),
           ("D/off", "t2full-embedding-threshold-D_seed0"), ("D/lexicon", "t2full-dk-mem-lexicon-D_seed0"))


def use_run_prefix(prefix: str) -> None:
    """Report on another run set (e.g. a rerun with the frozen judge); the default stays the historical t2full set."""
    global CONFIGS
    CONFIGS = (("C/off", f"{prefix}-store-surface-only-C_seed0"), ("C/lexicon", f"{prefix}-dk-mem-lexicon-C_seed0"),
               ("D/off", f"{prefix}-embedding-threshold-D_seed0"), ("D/lexicon", f"{prefix}-dk-mem-lexicon-D_seed0"))


TIER1 = {"C/off": ("C", "store-surface-only-C_seed0"), "C/lexicon": ("C", "dk-mem-lexicon-C_seed0"),
         "D/off": ("D", "embedding-threshold-D_bge-m3_tau0.5_seed0"), "D/lexicon": ("D", "dk-mem-lexicon-D_bge-m3_tau0.5_seed0")}
TAU = 0.90


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def quant(xs: list[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    s = sorted(xs)
    return {f"p{int(q * 100):02d}": s[min(len(s) - 1, int(round(q * (len(s) - 1))))] for q in qs}


def rate(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.3f})" if d else "n/a"


def tier1_outcomes(eval_dir: Path, cfg: str) -> dict[str, bool]:
    """Tier-1 merged? per Tier-1 eval_pair_id for the same configuration (3B C runs as stored; frozen D scores re-thresholded at the SAME tau as Tier 2)."""
    kind, rid = TIER1[cfg]
    if kind == "C":
        rows = jl(eval_dir / "results" / "C_REAL_PROVISIONAL_tier1_pair_details.jsonl")
        return {r["eval_pair_id"]: bool(r["merged"]) for r in rows if r["run_id"] == rid}
    utt = {}
    for g in jl(eval_dir / "teamB_answer_key.jsonl"):
        utt[g["utterance_a_id"]] = utt[g["utterance_b_id"]] = g["eval_pair_id"]
    out = {}
    for r in jl(eval_dir / "harness_runs" / rid / "pairwise_eval.jsonl"):
        pid = utt[r["entry_a"]["source_utterance_id"]]
        host = r["similarity_score"] > TAU
        out[pid] = out.get(pid, False) or (host and (r.get("compatibility") in (None, "compatible")))
    return out


def build(runs_dir: Path, data_dir: Path = G.DATA_DIR, eval_dir: Path = G.EVAL, n_boot: int = N_BOOT) -> dict:
    gold = {g["eval_pair_id"]: g for g in G.read_jsonl(data_dir / "tier2_gold.jsonl")}
    meta = {m["injected_pair_id"]: m for m in G.read_jsonl(data_dir / "tier2_injection_metadata.jsonl")}
    n_conv = len(G.read_jsonl(data_dir / "tier2_conversations.jsonl"))
    run_tier2.verify_frozen(data_dir)                                   # raises if any frozen Tier-2 v1.0 file changed
    rep = {"banner": BANNER, "configs": {}, "frozen_hashes_verified": True, "n_conversations_frozen": n_conv, "tau_note": "D tau=0.90 is a pilot/execution threshold, NOT a final operating point; raw cosine kept for every compared pair"}
    reports, recs, rlogs, cache = {}, {}, {}, {}
    for name, rid in CONFIGS:
        d = runs_dir / rid
        reports[name] = json.loads((runs_dir / f"{rid}.tier2_contract_report.json").read_text(encoding="utf-8"))
        recs[name] = jl(d / "pairwise_eval.jsonl")
        rlogs[name] = jl(runs_dir / f"{rid}.tier2_retrieval_log.jsonl")
    # ---- per configuration
    for name, rid in CONFIGS:
        R, rows, rlog = reports[name], recs[name], rlogs[name]
        man = json.loads((runs_dir / rid / "run_manifest.json").read_text(encoding="utf-8"))
        info = json.loads((runs_dir / f"{rid}.tier2_run_info.json").read_text(encoding="utf-8"))
        ps = R["injected_pairs"]
        diff, same = [p for p in ps if p["relation"] == "different"], [p for p in ps if p["relation"] == "same"]
        dec = Counter(r["decision"] for r in rows)
        merges = dec.get("merge", 0) + dec.get("supersede", 0)
        sims = [r["similarity_score"] for r in rows]
        sess_ok = all({x["session_index"] for x in rlog if x["conversation_id"] == c} == set(range(1, 7)) for c in {x["conversation_id"] for x in rlog})
        checks = {"69/69 conversations completed": R["counts"]["conversations"] == n_conv == len({x["conversation_id"] for x in rlog}),
                  "all 6 sessions per conversation": sess_ok, "user turns 828": R["counts"]["user_turns"] == 69 * 12,
                  "pairwise records = logged comparisons with a record": len(rows) == sum(1 for x in rlog for c in x["comparisons"] if c["status"] == "record"),
                  "all 69 injected pairs mapped": len(ps) == 69 and len({p["injected_pair_id"] for p in ps}) == 69,
                  "contract 0 errors": R["ok"] and not R["errors"], "no NaN/Inf similarity": all(s == s and abs(s) != float("inf") for s in sims),
                  "frozen Tier-2 hashes match v1.0": True}
        rows_b = [{"relation": p["relation"], "merged": p["status"] == "merged", "outcome": p["status"]} for p in ps]
        cw = cell_with_ci(rows_b, f"tier2_full|{name}", n_boot, cache)
        qa_by = defaultdict(lambda: [0, 0])
        for p in ps:
            for q in p["qa"]:
                k = f"{p['status']}|{p['relation']}"
                qa_by[k][1] += 1
                qa_by[k][0] += bool(q["correct"])
        rep["configs"][name] = {
            "run_id": rid, "conversations_completed": R["counts"]["conversations"], "completion_checks": checks, "all_checks_pass": all(checks.values()),
            "pairwise_records": len(rows), "merge": merges, "non_merge": len(rows) - merges, "decisions": dict(dec), "compatibility": R["counts"]["compatibility"],
            "injected_pair_status": R["counts"]["injected_pair_status"], "retrieval_misses": R["counts"]["injected_pair_status"].get("retrieval_miss", 0),
            "retrieval_miss_rate": rate(R["counts"]["injected_pair_status"].get("retrieval_miss", 0), len(ps)),
            "judge_parse_errors": info["judge_parse_errors"],
            "gold_different": dict(Counter(p["status"] for p in diff)), "gold_same": dict(Counter(p["status"] for p in same)),
            "FCR_T2": {"n": cw["fcr"]["numerator"], "d": cw["fcr"]["denominator"], "ci95": [cw["fcr"]["ci"]["lo"], cw["fcr"]["ci"]["hi"]]},
            "MCR_T2": {"n": cw["mcr"]["numerator"], "d": cw["mcr"]["denominator"], "ci95": [cw["mcr"]["ci"]["lo"], cw["mcr"]["ci"]["hi"]]},
            "QA": {"correct": R["counts"]["qa_correct"], "total": R["counts"]["qa_answers"], "accuracy": R["counts"]["qa_correct"] / R["counts"]["qa_answers"],
                   "by_pair_status_and_relation": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(qa_by.items())}},
            "contract_ok": R["ok"], "contract_errors": R["errors"][:5], "memory_records": R["counts"]["memory_records"],
            "judge": judge_provenance.classify(man) if name.startswith("C") else None,
            "by_session_distance": {}, "by_class": {}}
        for key, getter in (("by_session_distance", lambda p: str(p["session_distance"])), ("by_class", lambda p: p["distinction_class"])):
            groups = defaultdict(list)
            for p in ps:
                groups[getter(p)].append(p)
            for g_, L in sorted(groups.items()):
                gd, gs = [p for p in L if p["relation"] == "different"], [p for p in L if p["relation"] == "same"]
                rep["configs"][name][key][g_] = {"FCR": rate(sum(p["status"] == "merged" for p in gd), len(gd)), "MCR": rate(sum(p["status"] != "merged" for p in gs), len(gs)),
                                                 "retrieval_misses": sum(p["status"] == "retrieval_miss" for p in L), "n_different": len(gd), "n_same": len(gs)}
    # ---- retrieval is identical across configurations?
    base = {r["record_id"]: r["similarity_score"] for r in recs["C/off"]}
    rep["raw_scores_identical_across_configs"] = all({r["record_id"]: r["similarity_score"] for r in recs[n]} == base for n, _ in CONFIGS)
    # ---- Tier-1 vs Tier-2 per-pair outcome changes (same configuration)
    t1v2 = {}
    for name, _ in CONFIGS:
        t1 = tier1_outcomes(eval_dir, name)
        trans = Counter()
        changed = []
        for p in reports[name]["injected_pairs"]:
            src = meta[p["injected_pair_id"]]["source_tier1_eval_pair_id"]
            a, b = t1[src], p["status"] == "merged"
            trans[(p["relation"], "T1 merged" if a else "T1 not merged", "T2 merged" if b else "T2 not merged")] += 1
            if a != b:
                changed.append({"injected_pair_id": p["injected_pair_id"], "source_tier1_pair_id": meta[p["injected_pair_id"]]["source_tier1_pair_id"], "relation": p["relation"],
                                "class": p["distinction_class"], "tier1_merged": a, "tier2_status": p["status"]})
        t1v2[name] = {"transitions": {" | ".join(k): v for k, v in sorted(trans.items())}, "n_changed": len(changed), "changed": changed}
    rep["tier1_vs_tier2"] = t1v2
    rep["tier1_comparison_note"] = "same configuration; C: stored Tier-1 3B runs; D: frozen Tier-1 bge-m3 scores re-thresholded at the same tau=0.90 (+ lexicon compatibility for D/lexicon)"
    # ---- D raw scores
    d_off = recs["D/off"]
    sims = [r["similarity_score"] for r in d_off]
    inj = {}
    for p in reports["D/off"]["injected_pairs"]:
        if p["similarity"] is not None:
            inj.setdefault(p["relation"], []).append(p["similarity"])
    edges = [-1, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0000001]
    hist = {f"[{edges[i]:.2f},{edges[i + 1]:.2f})": sum(edges[i] <= s < edges[i + 1] for s in sims) for i in range(len(edges) - 1)}
    rep["d_raw_scores"] = {"n_compared_pairs": len(sims), "min": min(sims), "max": max(sims), "mean": statistics.mean(sims), "quantiles": quant(sims),
                           "above_tau_strict": sum(s > TAU for s in sims), "histogram": hist,
                           "injected_pairs_by_relation": {k: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v), "above_tau": sum(x > TAU for x in v)} for k, v in inj.items()},
                           "stored_for_every_compared_pair": all("similarity_score" in r for r in d_off)}
    # ---- gate effects
    gate = {}
    for off, on in (("D/off", "D/lexicon"), ("C/off", "C/lexicon")):
        a = {r["record_id"]: r for r in recs[off]}
        b = {r["record_id"]: r for r in recs[on]}
        assert set(a) == set(b)
        non_compat = {k for k, r in b.items() if r["compatibility"] in ("incompatible", "underdetermined")}
        eff = {k for k in non_compat if a[k]["decision"] in ("merge", "supersede")}
        created = [k for k in b if b[k]["decision"] in ("merge", "supersede") and a[k]["decision"] not in ("merge", "supersede")]
        injected_ids = {}
        for p in reports[on]["injected_pairs"]:
            injected_ids[f"{p['earlier_turn_id']}_e0::{p['later_turn_id']}_e0"] = p
        vp = [(k, p) for k, p in injected_ids.items() if k in non_compat]
        ep = [(k, p) for k, p in vp if k in eff]
        gate[f"{off} -> {on}"] = {"compared_pairs": len(b), "gate_verdicts": dict(Counter(r["compatibility"] for r in b.values())), "pairs_vetoed_by_gate (non-compatible verdict)": len(non_compat),
                                  "effective_vetoes (host would have merged)": len(eff), "merges_created_by_gate": len(created),
                                  "injected_pairs_vetoed": [{"injected_pair_id": p["injected_pair_id"], "relation": p["relation"], "class": p["distinction_class"], "verdict": b[k]["compatibility"], "effective": k in eff} for k, p in vp],
                                  "injected_pairs_vetoed_count": len(vp), "injected_effective_vetoes_count": len(ep),
                                  "decisions_differ_only_by_gate": all((a[k]["decision"] != b[k]["decision"]) <= (k in non_compat) for k in a)}
    rep["gate_effects"] = gate
    return rep


def render(rep: dict) -> str:
    L = [f"# {BANNER}", "", f"> {rep['tau_note']}.", "", "Descriptive preliminary numbers for the frozen Tier-2 v1.0 benchmark (69 conversations; 42 gold-different + 27 gold-same injected pairs). Single seed. "
         "FCR-T2/MCR-T2 follow the Tier-1 definitions on the injected pairs; never-compared (retrieval-miss) and judge-failure pairs stay in the denominators as unmerged and are also reported separately. "
         "CIs: existing relation-stratified percentile bootstrap (pair = conversation). No audit has been done.", ""]
    names = list(rep["configs"])
    for n in names:                     # state which judge each Config C run actually used, before any number
        j = rep["configs"][n].get("judge")
        if j is None:
            continue
        kind = ("the FROZEN Tier-1 merge_judge_v1" if j["frozen_tier1"] else
                "the HISTORICAL harness judge (NOT the frozen Tier-1 prompt; results not comparable with Tier 1)" if j["historical_harness_judge"] else
                "an UNRECOGNISED judge prompt (do not report)")
        L += [f"> **{n}** judge: {kind}; prompt id `{j['prompt_id']}`, sha256 `{str(j['prompt_sha256'])[:16]}...`, max_new_tokens {j['max_new_tokens']}.", ""]
    row = lambda label, f: [label] + [f(rep["configs"][n]) for n in names]
    tbl = lambda rows: ["| " + " | ".join(r) + " |" for r in rows]
    head = ["", *names]
    L += ["## Per configuration", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    pct = lambda x: f"{x['n']}/{x['d']} ({x['n'] / x['d']:.3f}) [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    L += tbl([row("run id", lambda c: c["run_id"]), row("conversations completed", lambda c: f"{c['conversations_completed']}/69"), row("all completion checks", lambda c: "PASS" if c["all_checks_pass"] else "FAIL"),
              row("pairwise records", lambda c: str(c["pairwise_records"])), row("merge / non-merge", lambda c: f"{c['merge']} / {c['non_merge']}"), row("gate verdicts", lambda c: json.dumps(c["compatibility"])),
              row("retrieval misses (injected pairs)", lambda c: f"{c['retrieval_misses']}/69"), row("judge parse errors", lambda c: str(c["judge_parse_errors"])),
              row("injected gold-different (n=42)", lambda c: json.dumps(c["gold_different"])), row("injected gold-same (n=27)", lambda c: json.dumps(c["gold_same"])),
              row("FCR-T2 [95% CI]", lambda c: pct(c["FCR_T2"])), row("MCR-T2 [95% CI]", lambda c: pct(c["MCR_T2"])),
              row("QA correct", lambda c: f"{c['QA']['correct']}/{c['QA']['total']} ({c['QA']['accuracy']:.3f})"), row("contract", lambda c: "OK, 0 errors" if c["contract_ok"] else f"FAILED {len(c['contract_errors'])}+")])
    for key, title in (("by_session_distance", "FCR-T2 / MCR-T2 by session distance"), ("by_class", "FCR-T2 / MCR-T2 by distinction class")):
        L += ["", f"## {title}", "", "| group | " + " | ".join(f"{n} FCR | {n} MCR | {n} misses" for n in names) + " |", "|---|" + "---|---|---|" * len(names)]
        for g in rep["configs"][names[0]][key]:
            L.append(f"| {g} (diff {rep['configs'][names[0]][key][g]['n_different']}, same {rep['configs'][names[0]][key][g]['n_same']}) | " +
                     " | ".join(f"{rep['configs'][n][key][g]['FCR']} | {rep['configs'][n][key][g]['MCR']} | {rep['configs'][n][key][g]['retrieval_misses']}" for n in names) + " |")
    L += ["", "## Tier-1 vs Tier-2 per-pair outcome changes (same configuration)", "", rep["tier1_comparison_note"], ""]
    for n in names:
        t = rep["tier1_vs_tier2"][n]
        L.append(f"* **{n}**: {t['n_changed']} of 69 pairs changed merge outcome. Transitions: {t['transitions']}")
    d = rep["d_raw_scores"]
    L += ["", "## D raw bge-m3 cosine scores (all compared pairs; identical across the four runs: " + str(rep["raw_scores_identical_across_configs"]) + ")", "",
          f"* {d['n_compared_pairs']} compared pairs; min {d['min']:.4f}, max {d['max']:.4f}, mean {d['mean']:.4f}; quantiles {{" + ", ".join(f"{k}: {v:.3f}" for k, v in d['quantiles'].items()) + f"}}; pairs above tau (strict >) {d['above_tau_strict']}.",
          f"* histogram: {d['histogram']}", f"* injected pairs by relation: {d['injected_pairs_by_relation']}", f"* raw score stored for every compared pair: {d['stored_for_every_compared_pair']}", "", "## Gate effects", ""]
    for k, g in rep["gate_effects"].items():
        L += [f"### {k}", "", f"* compared pairs {g['compared_pairs']}; gate verdicts {g['gate_verdicts']}", f"* pairs vetoed (non-compatible verdict): {g['pairs_vetoed_by_gate (non-compatible verdict)']}; effective vetoes (host would have merged): {g['effective_vetoes (host would have merged)']}; merges created by the gate: {g['merges_created_by_gate']}",
              f"* injected pairs vetoed: {g['injected_pairs_vetoed_count']}; effective: {g['injected_effective_vetoes_count']}; detail: {g['injected_pairs_vetoed']}", ""]
    L += ["## Retrieval misses", "", "Reported separately from FCR/MCR: " + "; ".join(f"{n} {rep['configs'][n]['retrieval_miss_rate']}" for n in names) + ".", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix, e.g. results/TIER2_FULL_RUN_PROVISIONAL_report")
    ap.add_argument("--run-prefix", default="t2full", help="run-id prefix of the four runs to report (default: the historical t2full set)")
    a = ap.parse_args(argv)
    use_run_prefix(a.run_prefix)
    js, md = a.out.with_suffix(".json"), a.out.with_suffix(".md")
    if js.exists() or md.exists():
        raise SystemExit("refusing to overwrite an existing report")
    rep = build(a.runs_dir)
    js.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    md.write_text(render(rep), encoding="utf-8", newline="\n")
    print("wrote", js, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
