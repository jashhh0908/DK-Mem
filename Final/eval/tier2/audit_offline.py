"""Tier-2 full-run AUDIT + OFFLINE D threshold sweep (PROVISIONAL, read-only).

Reads only: the frozen Tier-2 v1.0 data, the completed `t2full-*` run artifacts, the Tier-1 gold key and the Tier-1 D run
artifacts. Loads no model and recomputes no embedding. Writes only new report files (refuses to overwrite).

Sweep semantics = the Tier-1 D semantics (evaluator.sweep.derived_merge, strict):
    D/off     : merge iff similarity_score > tau
    D/lexicon : merge iff similarity_score > tau AND compatibility == compatible   (apply_gate blocks BOTH `incompatible`
                (keep_both) and `underdetermined` (link_unresolved); the two are reported separately here)
A pair never compared (retrieval miss) would stay unmerged and in the denominators; there are 0 here.
FCR-T2 = merged gold-different injected / 42, MCR-T2 = unmerged gold-same injected / 27 (both lower is better).
The compared pairs, rankings and scores do not depend on tau (the store ranks all earlier RAW entries), so re-thresholding the
stored scores is exact for the injected pairs; a tau sweep does NOT change which pairs were compared.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluator import d_sweep
from evaluator.contract_report import N_BOOT, cell_with_ci
from evaluator.sweep import derived_merge

from . import generate_tier2 as G
from . import judge_provenance, run_tier2

BANNER = "TIER-2 AUDIT + OFFLINE D ANALYSIS — PROVISIONAL, NOT A FINAL RESULT"
RUNS = {"C/off": "t2full-store-surface-only-C_seed0", "C/lexicon": "t2full-dk-mem-lexicon-C_seed0",
        "D/off": "t2full-embedding-threshold-D_seed0", "D/lexicon": "t2full-dk-mem-lexicon-D_seed0"}
EXEC_TAU = 0.90                      # pilot/execution threshold only


PASSTHROUGH_FILE: Path | None = None     # the exact passthrough a rerun used (default: the historical gold-linked ids)


def use_run_prefix(prefix: str) -> None:
    """Point the audit at another run set (e.g. a rerun with the frozen judge) without touching the historical default."""
    RUNS.clear()
    RUNS.update({"C/off": f"{prefix}-store-surface-only-C_seed0", "C/lexicon": f"{prefix}-dk-mem-lexicon-C_seed0",
                 "D/off": f"{prefix}-embedding-threshold-D_seed0", "D/lexicon": f"{prefix}-dk-mem-lexicon-D_seed0"})
FCR_TARGETS = (0.20, 0.15, 0.10, 0.05, 0.02, 0.0)
MCR_TARGETS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.0)
BLOCKING = ("incompatible", "underdetermined")


def _dkmem_dir(eval_dir: Path) -> Path:
    """dkmem next to eval/ in this repository; the original sibling layout (<root>/ipdV2/dkmem) is still honoured."""
    return (eval_dir.parent / "ipdV2" / "dkmem") if (eval_dir.parent / "ipdV2" / "dkmem").is_dir() else (eval_dir.parent / "dkmem")


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def grid() -> list[float]:
    """Documented choice, NOT a selected threshold. Observed Tier-2 raw cosine range is 0.26-0.96."""
    g = [round(0.30 + 0.05 * i, 3) for i in range(7)]            # 0.30 .. 0.60
    g += [round(0.61 + 0.01 * i, 3) for i in range(39)]          # 0.61 .. 0.99
    g += [0.995, 1.0]
    return g


# --------------------------------------------------------------------------------------------- injected pairs
def load_pairs(data_dir: Path, runs_dir: Path) -> dict:
    """For every gold injected pair and run: the ONE cross-turn pairwise record (found independently of the contract report)."""
    gold = G.read_jsonl(data_dir / "tier2_gold.jsonl")
    meta = {m["injected_pair_id"]: m for m in G.read_jsonl(data_dir / "tier2_injection_metadata.jsonl")}
    runs = {n: {r["record_id"]: r for r in jl(runs_dir / rid / "pairwise_eval.jsonl")} for n, rid in RUNS.items()}
    pairs = []
    for g in gold:
        m = meta[g["eval_pair_id"]]
        a, b = g["utterance_a_id"], g["utterance_b_id"]
        found = {}
        for n, recs in runs.items():
            hits = [recs[k] for k in (f"{a}_e0::{b}_e0", f"{b}_e0::{a}_e0") if k in recs]
            found[n] = hits[0] if len(hits) == 1 else None
        pairs.append({"id": g["eval_pair_id"], "gold": g, "meta": m, "records": found})
    return {"pairs": pairs, "runs": runs}


# --------------------------------------------------------------------------------------------- audit
def audit(runs_dir: Path, data_dir: Path) -> dict:
    out = {"raw_artifacts_byte_identical_to_download_manifest": None, "frozen_hashes": None, "configs": {}}
    man = json.loads((runs_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [f["file"] for f in man["files"] if hashlib.sha256((runs_dir / f["file"]).read_bytes()).hexdigest() != f["sha256"]]
    out["raw_artifacts_byte_identical_to_download_manifest"] = {"files_checked": len(man["files"]), "mismatches": bad}
    try:
        run_tier2.verify_frozen(data_dir)
        out["frozen_hashes"] = "all Tier-2 v1.0 frozen files match FROZEN_v1.0.json"
    except Exception as e:                                          # noqa: BLE001 - reported, not hidden
        out["frozen_hashes"] = f"MISMATCH: {e}"
    convs = G.read_jsonl(data_dir / "tier2_conversations.jsonl")
    conv_ids = [c["conversation_id"] for c in convs]
    n_turn_users = {c["conversation_id"]: sum(1 for s in c["sessions"] for t in s["turns"] if t["speaker"] == "user") for c in convs}
    qa = G.read_jsonl(data_dir / "tier2_qa_questions.jsonl")
    P = load_pairs(data_dir, runs_dir)
    for name, rid in RUNS.items():
        recs = list(P["runs"][name].values())
        rlog = jl(runs_dir / f"{rid}.tier2_retrieval_log.jsonl")
        rep = json.loads((runs_dir / f"{rid}.tier2_contract_report.json").read_text(encoding="utf-8"))
        info = json.loads((runs_dir / f"{rid}.tier2_run_info.json").read_text(encoding="utf-8"))
        qa_rows = jl(runs_dir / f"{rid}.tier2_qa_answers.jsonl")
        by_conv = Counter(r["record_id"].split("_s")[0] for r in recs)
        sessions = defaultdict(set)
        turns = Counter()
        for x in rlog:
            sessions[x["conversation_id"]].add(x["session_index"])
            turns[x["conversation_id"]] += 1
        linked = {c["record_id"] for x in rlog for c in x["comparisons"] if c["status"] == "record"}
        sims = [r["similarity_score"] for r in recs]
        pr = P["pairs"]
        mapping_ok = all(p["records"][name] is not None for p in pr)
        if PASSTHROUGH_FILE is None:        # historical runs: injected turns carry the Tier-1 opaque ids of the gold pair
            want_ids = lambda p: {p["gold"]["utterance_a_id"]: p["gold"]["opaque_entity_id_a"], p["gold"]["utterance_b_id"]: p["gold"]["opaque_entity_id_b"]}
        else:                               # reruns: whatever the run's own passthrough file says for the two turns
            pt_map = {r["turn_id"]: r["gold_entity_id"] for r in jl(PASSTHROUGH_FILE)}
            want_ids = lambda p: {p["gold"]["utterance_a_id"]: pt_map[p["gold"]["utterance_a_id"]], p["gold"]["utterance_b_id"]: pt_map[p["gold"]["utterance_b_id"]]}
        ent_ok = all(
            p["records"][name] is not None and
            {p["records"][name][k]["source_utterance_id"]: p["records"][name][k]["gold_entity_id"] for k in ("entry_a", "entry_b")} == want_ids(p)
            for p in pr)
        meta_ok = all(p["meta"]["language"] == p["gold"]["language"] and p["meta"]["distinction_class"] == p["gold"]["distinction_class"] and
                      {p["meta"]["turn_id_a"], p["meta"]["turn_id_b"]} == {p["gold"]["utterance_a_id"], p["gold"]["utterance_b_id"]} for p in pr)
        crel = {q["injected_pair_id"]: q["relation"] for q in rep["injected_pairs"]}
        contract_rel_ok = all(crel[p["id"]] == p["gold"]["relation"] for p in pr)
        misses = [p["id"] for p in pr if p["records"][name] is None or p["records"][name]["record_id"] not in linked]
        c = {
            "conversations_in_retrieval_log": len(sessions), "conversations_all_69": sorted(sessions) == sorted(conv_ids),
            "all_6_sessions_each": all(s == set(range(1, 7)) for s in sessions.values()),
            "user_turns_each_12_and_logged": all(turns[k] == n_turn_users[k] == 12 for k in conv_ids),
            "pairwise_records": len(recs), "pairwise_records_expected": 69 * 45,
            "records_per_conversation_all_45": all(by_conv[k] == 45 for k in conv_ids),
            "record_ids_unique_and_all_in_retrieval_log": len(recs) == len(P["runs"][name]) == len(linked) and set(P["runs"][name]) == linked,
            "judge_parse_errors": info["judge_parse_errors"], "comparison_status_counts": dict(Counter(c["status"] for x in rlog for c in x["comparisons"])),
            "injected_pairs_mapped_to_one_record_each": mapping_ok, "injected_entity_ids_match_gold_relation": ent_ok,
            "injected_metadata_matches_gold": meta_ok, "contract_relation_matches_gold": contract_rel_ok,
            "retrieval_misses": len(misses), "retrieval_miss_ids": misses,
            "contract_ok": rep["ok"], "contract_errors": len(rep["errors"]),
            "similarity_present_and_finite_all_records": all("similarity_score" in r and finite(r["similarity_score"]) for r in recs),
            "similarity_min": min(sims), "similarity_max": max(sims),
            "qa_answers": len(qa_rows), "qa_questions_expected": len(qa), "qa_correct": rep["counts"]["qa_correct"],
        }
        if name.startswith("C"):
            c["c_recorded_threshold_values"] = sorted({r["threshold"] for r in recs}, key=repr)
        if name.startswith("D"):
            c["d_rule_reproduced_at_tau_0.90"] = all(
                (r["decision"] in ("merge", "supersede")) == (r["similarity_score"] > EXEC_TAU and (name == "D/off" or r["compatibility"] == "compatible")) for r in recs)
            c["d_recorded_threshold_values"] = sorted({r["threshold"] for r in recs})
        if name.endswith("lexicon"):
            c["compat_counts"] = dict(Counter(r["compatibility"] for r in recs))
        else:
            c["compat_all_null"] = all(r["compatibility"] is None for r in recs)
        c["all_audit_checks_pass"] = all([c["conversations_all_69"], c["all_6_sessions_each"], c["user_turns_each_12_and_logged"], c["pairwise_records"] == 3105,
                                         c["records_per_conversation_all_45"], c["record_ids_unique_and_all_in_retrieval_log"], mapping_ok, ent_ok, meta_ok,
                                         contract_rel_ok, not misses, rep["ok"], not rep["errors"], c["similarity_present_and_finite_all_records"],
                                         c["qa_answers"] == len(qa)] + ([c["d_rule_reproduced_at_tau_0.90"]] if name.startswith("D") else []))
        out["configs"][name] = c
    base = {k: r["similarity_score"] for k, r in P["runs"]["D/off"].items()}
    out["raw_similarity_identical_across_all_four_runs"] = all({k: r["similarity_score"] for k, r in P["runs"][n].items()} == base for n in RUNS)
    out["_pairs"] = P
    # ---- C judge: identity and behaviour from the manifest and the judge log (never a hard-coded sentence)
    deg, prov = {}, {}
    for name in ("C/off", "C/lexicon"):
        jlog = jl(runs_dir / f"{RUNS[name]}.judge_log.jsonl")
        man_c = json.loads((runs_dir / RUNS[name] / "run_manifest.json").read_text(encoding="utf-8"))
        prov[name] = judge_provenance.describe(man_c, jlog)
        recs = P["runs"][name]
        same_ids = {p["records"][name]["record_id"] for p in P["pairs"] if p["gold"]["relation"] == "same"}
        jbyrec = {r["record_id"]: r for r in jlog}
        b = prov[name]["behaviour"]
        deg[name] = {"judge_calls": b["judge_calls"], "status_ok": b["status_ok"], "parse_errors": b["parse_errors"], "proposal_counts": b["proposal_counts"],
                     "raw_output_distinct": b["raw_output_distinct"], "pairs_judged_same": sum(r["proposal"] == "merge" for r in jlog),
                     "gold_same_injected_judged": len(same_ids), "gold_same_injected_judged_same": sum(jbyrec[i]["proposal"] == "merge" for i in same_ids),
                     "merge_decisions_in_pairwise_records": sum(r["decision"] in ("merge", "supersede") for r in recs.values()),
                     "judge": {k: prov[name][k] for k in ("prompt_id", "prompt_sha256", "frozen_tier1", "historical_harness_judge", "max_new_tokens",
                                                          "comparable_with_tier1_config_c")}}
    out["c_judge_behaviour"] = deg
    out["c_judge_provenance"] = prov
    off, lex = prov["C/off"]["classification"], prov["C/lexicon"]["classification"]
    out["c_judge_classification"] = off if off == lex else f"C/off: {off} C/lexicon: {lex}"
    return out


# --------------------------------------------------------------------------------------------- five false merges
def false_merges(P: dict) -> list[dict]:
    lex = P["runs"]["D/lexicon"]
    out = []
    de_entries = Counter()
    for r in lex.values():
        for e in (r["entry_a"], r["entry_b"]):
            if e["language"] == "de":
                de_entries[e["entry_id"]] += 1
    for p in P["pairs"]:
        off, on = p["records"]["D/off"], p["records"]["D/lexicon"]
        if p["gold"]["relation"] == "different" and off["decision"] in ("merge", "supersede"):
            out.append({
                "conversation_id": p["meta"]["conversation_id"], "injected_pair_id": p["id"], "distinction_class": p["gold"]["distinction_class"], "language": p["gold"]["language"],
                "session_distance": p["meta"]["session_distance"], "raw_cosine": off["similarity_score"],
                "D_off_decision": off["decision"], "D_lexicon_decision": on["decision"], "gate_verdict": on["compatibility"],
                "texts": [on["entry_a"]["surface"], on["entry_b"]["surface"]], "markers": [p["meta"]["marker_a"], p["meta"]["marker_b"]],
                "extracted_distinction_a": on["entry_a"]["distinction"], "extracted_distinction_b": on["entry_b"]["distinction"],
                "scores_identical_off_vs_lexicon": off["similarity_score"] == on["similarity_score"],
                "margin_over_tau_0.90": off["similarity_score"] - EXEC_TAU})
    return out


def lexicon_coverage(lex_path: Path) -> dict:
    d = json.loads(lex_path.read_text(encoding="utf-8"))
    return {"n_entries": len(d["entries"]), "by_lang_and_class": {f"{k[0]}|{k[1]}": v for k, v in Counter((e["lang"], e["distinction_class"]) for e in d["entries"]).items()},
            "languages": sorted({e["lang"] for e in d["entries"]}), "register_surface_forms": sorted(f for e in d["entries"] if e["distinction_class"] == "register" for f in e["surface_forms"])}


# --------------------------------------------------------------------------------------------- gate verdicts
def gate_analysis(P: dict, taus: list[float]) -> dict:
    lex, off = P["runs"]["D/lexicon"], P["runs"]["D/off"]
    inj_ids = {p["records"]["D/lexicon"]["record_id"]: p for p in P["pairs"]}
    cats = {}
    for v in ("compatible", "underdetermined", "incompatible"):
        rs = [r for r in lex.values() if r["compatibility"] == v]
        inj = [inj_ids[r["record_id"]] for r in rs if r["record_id"] in inj_ids]
        cats[v] = {"all_compared_pairs": len(rs), "injected_pairs": len(inj),
                   "host_would_merge_at_tau_0.90 (all pairs)": sum(r["similarity_score"] > EXEC_TAU for r in rs),
                   "host_would_merge_at_tau_0.90 (injected)": sum(inj_ids[r["record_id"]]["records"]["D/off"]["similarity_score"] > EXEC_TAU for r in rs if r["record_id"] in inj_ids),
                   "actual_D_lexicon_merges": sum(r["decision"] in ("merge", "supersede") for r in rs),
                   "injected_by_relation": dict(Counter(p["gold"]["relation"] for p in inj)),
                   "injected_by_class": dict(Counter(p["gold"]["distinction_class"] for p in inj)),
                   "injected_by_language": dict(Counter(p["gold"]["language"] for p in inj)),
                   "injected_by_distance": dict(Counter(str(p["meta"]["session_distance"]) for p in inj))}
    incompat = []
    for p in P["pairs"]:
        r = p["records"]["D/lexicon"]
        if r["compatibility"] == "incompatible":
            s = r["similarity_score"]
            incompat.append({"injected_pair_id": p["id"], "conversation_id": p["meta"]["conversation_id"], "relation": p["gold"]["relation"], "distinction_class": p["gold"]["distinction_class"],
                             "language": p["gold"]["language"], "session_distance": p["meta"]["session_distance"], "raw_cosine": s,
                             "host_would_merge_at_tau_0.90": s > EXEC_TAU, "effective_veto_at_tau_0.90": s > EXEC_TAU and r["decision"] not in ("merge", "supersede"),
                             "distinction_a": r["entry_a"]["distinction"], "distinction_b": r["entry_b"]["distinction"], "D_lexicon_decision": r["decision"]})
    # effective vetoes as a function of tau (host would merge, gate blocks), by verdict, injected and all pairs; both are offline re-thresholdings
    by_tau = []
    for t in taus:
        row = {"tau": t}
        for v in BLOCKING:
            allp = [r for r in lex.values() if r["compatibility"] == v and r["similarity_score"] > t]
            inj = [p for p in P["pairs"] if p["records"]["D/lexicon"]["compatibility"] == v and p["records"]["D/lexicon"]["similarity_score"] > t]
            row[f"{v}_effective_all"] = len(allp)
            row[f"{v}_effective_injected"] = len(inj)
            row[f"{v}_effective_injected_gold_different"] = sum(p["gold"]["relation"] == "different" for p in inj)
            row[f"{v}_effective_injected_gold_same"] = sum(p["gold"]["relation"] == "same" for p in inj)
        by_tau.append(row)
    summ = {"incompatible_total": len(incompat), "incompatible_by_relation": dict(Counter(x["relation"] for x in incompat)),
            "incompatible_by_class": dict(Counter(x["distinction_class"] for x in incompat)), "incompatible_by_language": dict(Counter(x["language"] for x in incompat)),
            "incompatible_by_distance": dict(Counter(str(x["session_distance"]) for x in incompat)),
            "incompatible_by_class_and_relation": dict(Counter(f"{x['distinction_class']}|{x['relation']}" for x in incompat)),
            "incompatible_host_would_merge_at_0.90": sum(x["host_would_merge_at_tau_0.90"] for x in incompat),
            "effective_vetoes_at_0.90": sum(x["effective_veto_at_tau_0.90"] for x in incompat),
            "incompatible_cosine_range": [min(x["raw_cosine"] for x in incompat), max(x["raw_cosine"] for x in incompat)] if incompat else None,
            "incompatible_gold_different_cosine_median": statistics.median([x["raw_cosine"] for x in incompat if x["relation"] == "different"]) if incompat else None}
    # which injected pairs are non-vetoed but D-merged / what the gate would need to catch
    return {"verdict_categories": cats, "incompatible_injected_pairs": incompat, "incompatible_summary": summ, "effective_vetoes_by_tau": by_tau,
            "note": "apply_gate blocks both incompatible (keep_both) and underdetermined (link_unresolved). `veto` below means the explicit cannot-link category `incompatible`; "
                    "`underdetermined` is reported separately and is not called a veto. 'effective' = the host (cosine > tau) would have merged the pair."}


# --------------------------------------------------------------------------------------------- sweep
def pair_states(P: dict, name: str) -> list[dict]:
    return [{"id": p["id"], "relation": p["gold"]["relation"], "class": p["gold"]["distinction_class"], "language": p["gold"]["language"],
             "distance": p["meta"]["session_distance"], "record": {"similarity_score": p["records"][name]["similarity_score"], "compatibility": p["records"][name]["compatibility"]}}
            for p in P["pairs"]]


def at_tau(states: list[dict], tau: float) -> list[dict]:
    return [{"relation": s["relation"], "merged": derived_merge(s["record"], tau), "outcome": "merged" if derived_merge(s["record"], tau) else "compared_not_merged",
             "distinction_class": s["class"], "language": s["language"], "distance": s["distance"]} for s in states]


def counts(rows: list[dict]) -> dict:
    d = [r for r in rows if r["relation"] == "different"]
    s = [r for r in rows if r["relation"] == "same"]
    fn, mn = sum(r["merged"] for r in d), sum(not r["merged"] for r in s)
    return {"fcr_n": fn, "fcr_d": len(d), "mcr_n": mn, "mcr_d": len(s), "fcr": fn / len(d) if d else None, "mcr": mn / len(s) if s else None}


def sweep_run(states: list[dict], run_id: str, gate: str, taus: list[float], n_boot: int, cache: dict) -> dict:
    points, classes = [], sorted({s["class"] for s in states})
    for t in taus:
        rows = at_tau(states, t)
        c = cell_with_ci(rows, f"t2sweep|{gate}|{t}", n_boot, cache)
        per = {}
        for k in classes:
            cc = counts([r for r in rows if r["distinction_class"] == k])
            per[k] = {"fcr_n": cc["fcr_n"], "fcr_d": cc["fcr_d"], "mcr_n": cc["mcr_n"], "mcr_d": cc["mcr_d"]}
        points.append({"tau": t, "fcr": d_sweep._pt(c["fcr"]), "mcr": d_sweep._pt(c["mcr"]), "n_pairs_merged": sum(r["merged"] for r in rows), "per_class": per})
    scores = sorted({s["record"]["similarity_score"] for s in states})
    exact = []
    for t in [scores[0] - 1e-9] + scores:
        cc = counts(at_tau(states, t))
        exact.append({"tau": t, "fcr": cc["fcr"], "mcr": cc["mcr"], "fcr_n": cc["fcr_n"], "mcr_n": cc["mcr_n"], "n_pairs_merged": cc["fcr_n"] + cc["mcr_d"] - cc["mcr_n"]})
    front = d_sweep.pareto(exact)
    return {"run_id": run_id, "gate": gate, "dkmem_mode": "off" if gate == "off" else gate, "points": points, "exact": exact, "pareto_exact": front}


def op_table(front: list[dict]) -> dict:
    def pick(pts, key):
        b = min(pts, key=key) if pts else None
        return None if b is None else {"fcr": b["fcr"], "mcr": b["mcr"], "fcr_n": b["fcr_n"], "mcr_n": b["mcr_n"], "tau_intervals": b["intervals"]}
    return {"min_fcr_given_mcr_at_most": {str(t): pick([p for p in front if p["mcr"] <= t + 1e-12], lambda p: (p["fcr"], p["mcr"])) for t in MCR_TARGETS},
            "min_mcr_given_fcr_at_most": {str(t): pick([p for p in front if p["fcr"] <= t + 1e-12], lambda p: (p["mcr"], p["fcr"])) for t in FCR_TARGETS}}


def sweep_all(P: dict, n_boot: int) -> dict:
    taus = grid()
    cache: dict = {}
    off = sweep_run(pair_states(P, "D/off"), RUNS["D/off"], "off", taus, n_boot, cache)
    lex = sweep_run(pair_states(P, "D/lexicon"), RUNS["D/lexicon"], "lexicon", taus, n_boot, cache)
    cmp_ = d_sweep.compare_frontiers(off, lex)
    dom = d_sweep.dominance(off, lex)
    mt = d_sweep.matched_tau(off, lex)
    ef = dom["exact_frontier"]
    return {"grid": {"taus": taus, "n": len(taus), "note": "documented choice, not a selected threshold"}, "off": off, "lexicon": lex,
            "operating_points": {"D/off": op_table(off["pareto_exact"]), "D/lexicon": op_table(lex["pareto_exact"])},
            "matched_mcr": cmp_, "matched_tau": mt, "dominance": dom,
            "frontier_summary": {"off_vertices": len(off["pareto_exact"]), "lexicon_vertices": len(lex["pareto_exact"]),
                                 "off_vertices_dominated_by_lexicon": len(ef["off_vertices_dominated_by_a_gated_vertex"]),
                                 "lexicon_vertices_not_matched_or_beaten_by_off": len(ef["gated_vertices_not_matched_or_beaten_by_off"]),
                                 "off_vertices_not_matched_or_beaten_by_lexicon": len(ef["off_vertices_not_matched_or_beaten_by_gated"]),
                                 "identical_vertices": ef["gated_vertices_identical_to_an_off_vertex"],
                                 "frontiers_identical": [(p["fcr_n"], p["mcr_n"]) for p in off["pareto_exact"]] == [(p["fcr_n"], p["mcr_n"]) for p in lex["pareto_exact"]]}}


# --------------------------------------------------------------------------------------------- Tier 1 vs Tier 2
def tier1_states(eval_dir: Path, P: dict) -> dict:
    """Tier-1 D states for the SAME 69 source pairs: per pair, all cross-utterance records (Tier-1 isolated pair context)."""
    key = {g["eval_pair_id"]: g for g in jl(eval_dir / "teamB_answer_key.jsonl")}
    utt = {}
    for g in key.values():
        utt[g["utterance_a_id"]] = utt[g["utterance_b_id"]] = g["eval_pair_id"]
    out = {}
    for name, rid in (("D/off", "embedding-threshold-D_bge-m3_tau0.5_seed0"), ("D/lexicon", "dk-mem-lexicon-D_bge-m3_tau0.5_seed0")):
        per = defaultdict(list)
        for r in jl(eval_dir / "harness_runs" / rid / "pairwise_eval.jsonl"):
            ea, eb = utt.get(r["entry_a"]["source_utterance_id"]), utt.get(r["entry_b"]["source_utterance_id"])
            if ea is not None and ea == eb and r["entry_a"]["source_utterance_id"] != r["entry_b"]["source_utterance_id"]:
                per[ea].append({"similarity_score": r["similarity_score"], "compatibility": r["compatibility"]})
        out[name] = per
    states = {n: [] for n in out}
    for p in P["pairs"]:
        src = p["meta"]["source_tier1_eval_pair_id"]
        for n in out:
            recs = out[n].get(src, [])
            states[n].append({"id": p["id"], "relation": p["gold"]["relation"], "class": p["gold"]["distinction_class"], "recs": recs})
    return states


def tier1_vs_tier2(eval_dir: Path, P: dict) -> dict:
    t1 = tier1_states(eval_dir, P)
    res = {"note": "Same 69 source pairs. Tier 1 = isolated pair (frozen Tier-1 D runs, bge-m3); Tier 2 = the pair embedded in a 6-session conversation with its text re-rendered by the Tier-2 generator. Descriptive; "
                   "no operating point is selected and tau=0.90 is only the Tier-2 execution threshold."}
    for name in ("D/off", "D/lexicon"):
        pairs1 = t1[name]
        score1 = [max((x["similarity_score"] for x in s["recs"]), default=None) for s in pairs1]
        score2 = [p["records"][name]["similarity_score"] for p in P["pairs"]]
        rel = [p["gold"]["relation"] for p in P["pairs"]]
        cl = [p["gold"]["distinction_class"] for p in P["pairs"]]
        stat = {}
        for key, sel in (("gold_different", lambda i: rel[i] == "different"), ("gold_same", lambda i: rel[i] == "same"),
                         *[(f"class={c}", (lambda c: lambda i: cl[i] == c)(c)) for c in sorted(set(cl))]):
            idx = [i for i in range(69) if sel(i) and score1[i] is not None]
            if idx:
                d = [score2[i] - score1[i] for i in idx]
                stat[key] = {"n": len(idx), "tier1_median": statistics.median(score1[i] for i in idx), "tier2_median": statistics.median(score2[i] for i in idx),
                             "median_diff_t2_minus_t1": statistics.median(d), "pairs_t2_lower": sum(x < 0 for x in d)}
        sw = {}
        for tier, states in (("tier1", None), ("tier2", None)):
            pass
        def fc(tier, tau):
            m = n = a = b = 0
            for i, p in enumerate(P["pairs"]):
                if tier == "tier1":
                    recs = pairs1[i]["recs"]
                    merged = any(derived_merge(r, tau) for r in recs)
                else:
                    merged = derived_merge({"similarity_score": p["records"][name]["similarity_score"], "compatibility": p["records"][name]["compatibility"]}, tau)
                if p["gold"]["relation"] == "different":
                    m += merged
                else:
                    n += not merged
            return {"fcr": f"{m}/42", "mcr": f"{n}/27"}
        for tau in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95):
            sw[str(tau)] = {"tier1_context": fc("tier1", tau), "tier2_context": fc("tier2", tau)}
        # Tier-1 context exact frontier on these same 69 pairs
        ex = []
        scs = sorted({x["similarity_score"] for s in pairs1 for x in s["recs"]})
        for t in [scs[0] - 1e-9] + scs:
            f_ = sum(any(derived_merge(r, t) for r in s["recs"]) for s in pairs1 if s["relation"] == "different")
            m_ = sum(not any(derived_merge(r, t) for r in s["recs"]) for s in pairs1 if s["relation"] == "same")
            ex.append({"tau": t, "fcr": f_ / 42, "mcr": m_ / 27, "fcr_n": f_, "mcr_n": m_})
        res[name] = {"score_shift": stat, "fcr_mcr_at_selected_taus": sw, "tier1_context_frontier": [(p["fcr_n"], p["mcr_n"]) for p in d_sweep.pareto(ex)],
                     "tier2_context_frontier": [(p["fcr_n"], p["mcr_n"]) for p in d_sweep.pareto(sweep_run(pair_states(P, name), "x", "off" if name == "D/off" else "lexicon", [0.5], 1, {})["exact"])]}
    return res


# --------------------------------------------------------------------------------------------- build / render
def build(runs_dir: Path, data_dir: Path = G.DATA_DIR, eval_dir: Path = G.EVAL, n_boot: int = N_BOOT) -> dict:
    a = audit(runs_dir, data_dir)
    P = a.pop("_pairs")
    rep = {"banner": BANNER, "tau_note": "tau=0.90 is a pilot/execution threshold, NOT an operating point; no threshold is selected here",
           "audit": a, "false_merges_tau_0.90": false_merges(P), "lexicon_coverage": lexicon_coverage(_dkmem_dir(eval_dir) / "memory" / "distinction_features.json")}
    ents = {}
    for r in P["runs"]["D/lexicon"].values():
        for e in (r["entry_a"], r["entry_b"]):
            ents[e["entry_id"]] = e
    rep["language_entry_counts"] = dict(Counter(e["language"] for e in ents.values()))
    rep["entries_with_extracted_distinction_by_language"] = dict(Counter(e["language"] for e in ents.values() if e["distinction"]))
    rep["german_honorific_gold_different"] = [{"injected_pair_id": p["id"], "cosine": p["records"]["D/off"]["similarity_score"], "merged_at_0.90": p["records"]["D/off"]["decision"] in ("merge", "supersede")}
                                              for p in P["pairs"] if p["gold"]["language"] == "de" and p["gold"]["relation"] == "different"]
    rep["gate"] = gate_analysis(P, grid())
    rep["sweep"] = sweep_all(P, n_boot)
    rep["tier1_vs_tier2"] = tier1_vs_tier2(eval_dir, P)
    # raw score summary, injected pairs by relation/class (D)
    ps = pair_states(P, "D/off")
    rep["injected_scores"] = {k: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)} for k, v in
                              {f"{rel}|{c}": [s["record"]["similarity_score"] for s in ps if s["relation"] == rel and s["class"] == c] for rel, c in sorted({(s["relation"], s["class"]) for s in ps})}.items()}
    return rep


def fr(c: dict, n: str, d: str) -> str:
    return f"{c[n]}/{c[d]} ({c[n] / c[d]:.3f})"


def render(rep: dict) -> str:
    A, S, Gt = rep["audit"], rep["sweep"], rep["gate"]
    L = [f"# {BANNER}", "", f"> {rep['tau_note']}. Read-only analysis of the completed `t2full-*` artifacts and frozen Tier-2 v1.0; no model or embedding was run. Single seed, no superiority claims.", "", "## 1. Audit", ""]
    L += [f"* raw run artifacts vs the download manifest: {A['raw_artifacts_byte_identical_to_download_manifest']['files_checked']} files checked, mismatches: {A['raw_artifacts_byte_identical_to_download_manifest']['mismatches']}", f"* frozen Tier-2 v1.0: {A['frozen_hashes']}",
          f"* raw cosine identical across all four runs: {A['raw_similarity_identical_across_all_four_runs']}", ""]
    names = list(A["configs"])
    L += ["| check | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    keys = [("conversations_all_69", "69 conversations (set equals frozen ids)"), ("all_6_sessions_each", "6 sessions per conversation"), ("user_turns_each_12_and_logged", "12 user turns logged per conversation"), ("pairwise_records", "pairwise records (expected 3105)"),
            ("records_per_conversation_all_45", "45 records per conversation"), ("record_ids_unique_and_all_in_retrieval_log", "record ids unique & all in retrieval log"), ("injected_pairs_mapped_to_one_record_each", "69 injected pairs -> exactly one record"),
            ("injected_entity_ids_match_gold_relation", "passthrough ids on the pair's entries == gold opaque ids"), ("injected_metadata_matches_gold", "metadata == gold (class/language/turns)"), ("contract_relation_matches_gold", "contract-report relation == gold"),
            ("retrieval_misses", "retrieval misses"), ("judge_parse_errors", "judge parse errors"), ("contract_ok", "contract ok"), ("contract_errors", "contract errors"),
            ("similarity_present_and_finite_all_records", "similarity present & finite"), ("qa_answers", "QA answers (111)"), ("all_audit_checks_pass", "ALL AUDIT CHECKS")]
    for k, label in keys:
        L.append(f"| {label} | " + " | ".join(str(A['configs'][n].get(k)) for n in names) + " |")
    L += ["", "D-rule reproduction at tau=0.90 (D/off: sim > tau; D/lexicon: sim > tau and compatible): " + ", ".join(f"{n} {A['configs'][n]['d_rule_reproduced_at_tau_0.90']}" for n in names if n.startswith('D')), "",
          "### C judge behaviour", "", f"**{A['c_judge_classification']}**", ""]
    for n, d in A["c_judge_behaviour"].items():
        L.append(f"* {n}: {d['judge_calls']} judge calls, {d['status_ok']} parsed ok; proposals {d['proposal_counts']}; raw outputs {d['raw_output_distinct']}; judged-same: {d['pairs_judged_same']}/{d['judge_calls']}; "
                 f"gold-same injected judged same: {d['gold_same_injected_judged_same']}/{d['gold_same_injected_judged']}; merge decisions: {d['merge_decisions_in_pairwise_records']}")
    L += ["", "## 2. D's five gold-different false merges at tau=0.90", "", "| conv | pair | class | lang | dist | raw cosine | D/off | D/lexicon | gate | distinction A / B |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in rep["false_merges_tau_0.90"]:
        L.append(f"| {f['conversation_id']} | {f['injected_pair_id']} | {f['distinction_class']} | {f['language']} | {f['session_distance']} | {f['raw_cosine']:.4f} | {f['D_off_decision']} | {f['D_lexicon_decision']} | {f['gate_verdict']} | {f['extracted_distinction_a']} / {f['extracted_distinction_b']} |")
    for f in rep["false_merges_tau_0.90"]:
        L.append(f"* {f['injected_pair_id']}: “{f['texts'][0]}” vs “{f['texts'][1]}” (markers {f['markers']})")
    lc = rep["lexicon_coverage"]
    L += ["", f"**Why the gate did not veto them.** The lexicon has {lc['n_entries']} entries, languages {lc['languages']} only ({lc['by_lang_and_class']}); the register surface forms are {lc['register_surface_forms']}. "
          f"All five pairs are German du/Sie. Entries per language: {rep['language_entry_counts']}; entries with an extracted distinction: {rep['entries_with_extracted_distinction_by_language']} (none German). "
          "Both sides therefore carry no distinction → verdict `compatible` → the host decision passes through (cosine > 0.90 → merge). This is a lexicon-coverage gap (no German entries), not a gate-logic error. "
          f"German honorific gold-different pairs: {rep['german_honorific_gold_different']}.", ""]
    L += ["## 3. Gate verdicts (D/lexicon; C/lexicon verdicts are identical)", "", "| verdict | all compared | injected | host would merge @0.90 (all) | host would merge @0.90 (injected) | actual D/lexicon merges |", "|---|---|---|---|---|---|"]
    for v, c in Gt["verdict_categories"].items():
        L.append(f"| {v} | {c['all_compared_pairs']} | {c['injected_pairs']} | {c['host_would_merge_at_tau_0.90 (all pairs)']} | {c['host_would_merge_at_tau_0.90 (injected)']} | {c['actual_D_lexicon_merges']} |")
    s = Gt["incompatible_summary"]
    L += ["", f"Injected pairs with an `incompatible` verdict: **{s['incompatible_total']}**; by relation {s['incompatible_by_relation']}; by class {s['incompatible_by_class']}; by language {s['incompatible_by_language']}; by session distance {s['incompatible_by_distance']}; "
          f"class×relation {s['incompatible_by_class_and_relation']}. Host would otherwise merge at tau=0.90: {s['incompatible_host_would_merge_at_0.90']}; **effective vetoes at tau=0.90: {s['effective_vetoes_at_0.90']}**. "
          f"Cosine range of the incompatible injected pairs {s['incompatible_cosine_range']}; gold-different median {s['incompatible_gold_different_cosine_median']}.", "",
          "Underdetermined verdicts are reported separately and are not counted as vetoes: " + str({v: Gt['verdict_categories']['underdetermined'][v] for v in ('all_compared_pairs', 'injected_pairs')}) + ".", "",
          "Effective blocking by tau (host would merge, gate blocks) — selected taus:", "", "| tau | incompat. effective (all) | incompat. injected diff | incompat. injected same | underdet. effective (all) | underdet. injected |", "|---|---|---|---|---|---|"]
    for r in Gt["effective_vetoes_by_tau"]:
        if r["tau"] in (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            L.append(f"| {r['tau']} | {r['incompatible_effective_all']} | {r['incompatible_effective_injected_gold_different']} | {r['incompatible_effective_injected_gold_same']} | {r['underdetermined_effective_all']} | {r['underdetermined_effective_injected']} |")
    L += ["", Gt["note"], "", "## 4. Offline D threshold sweep (stored raw cosine; strict `>`; injected pairs only: 42 gold-different, 27 gold-same)", "",
          f"Grid ({S['grid']['n']} taus, documented choice, not a selected threshold): 0.30–0.60 step 0.05, 0.61–0.99 step 0.01, 0.995, 1.0; exact frontiers use every distinct observed injected-pair score. CIs: existing relation-stratified bootstrap.", "",
          "### Selected grid points (D/off | D/lexicon)", "", "| tau | D/off FCR | D/off MCR | D/lexicon FCR | D/lexicon MCR |", "|---|---|---|---|---|"]
    po, pl = {p["tau"]: p for p in S["off"]["points"]}, {p["tau"]: p for p in S["lexicon"]["points"]}
    f_ = lambda q: f"{q['n']}/{q['d']} ({q['rate']:.3f}) [{q['lo']:.2f},{q['hi']:.2f}]"
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.96):
        if t in po:
            L.append(f"| {t} | {f_(po[t]['fcr'])} | {f_(po[t]['mcr'])} | {f_(pl[t]['fcr'])} | {f_(pl[t]['mcr'])} |")
    for nm, key in (("D/off", "off"), ("D/lexicon", "lexicon")):
        L += ["", f"### Exact Pareto frontier — {nm} ({len(S[key]['pareto_exact'])} vertices)", "", "| FCR | MCR | tau interval (state holds for tau in [from, to)) |", "|---|---|---|"]
        for p in S[key]["pareto_exact"]:
            L.append(f"| {p['fcr_n']}/42 ({p['fcr']:.3f}) | {p['mcr_n']}/27 ({p['mcr']:.3f}) | " + "; ".join(f"[{a:.6f}, {('%.6f' % b) if b is not None else '∞'})" for a, b in p["intervals"][:3]) + " |")
    L += ["", "### Operating-point table (descriptive; nothing selected)", "", "Lowest FCR at an MCR ceiling:", "", "| MCR ≤ | D/off FCR (MCR) | D/lexicon FCR (MCR) |", "|---|---|---|"]
    for t in MCR_TARGETS:
        a = S["operating_points"]["D/off"]["min_fcr_given_mcr_at_most"][str(t)]
        b = S["operating_points"]["D/lexicon"]["min_fcr_given_mcr_at_most"][str(t)]
        g_ = lambda x: "n/a" if x is None else f"{x['fcr_n']}/42 = {x['fcr']:.3f} (MCR {x['mcr_n']}/27)"
        L.append(f"| {t} | {g_(a)} | {g_(b)} |")
    L += ["", "Lowest MCR at an FCR ceiling:", "", "| FCR ≤ | D/off MCR (FCR) | D/lexicon MCR (FCR) |", "|---|---|---|"]
    for t in FCR_TARGETS:
        a = S["operating_points"]["D/off"]["min_mcr_given_fcr_at_most"][str(t)]
        b = S["operating_points"]["D/lexicon"]["min_mcr_given_fcr_at_most"][str(t)]
        g_ = lambda x: "n/a" if x is None else f"{x['mcr_n']}/27 = {x['mcr']:.3f} (FCR {x['fcr_n']}/42)"
        L.append(f"| {t} | {g_(a)} | {g_(b)} |")
    fs, mm, mt = S["frontier_summary"], S["matched_mcr"], S["matched_tau"]
    L += ["", "### Does the gate shift the Tier-2 frontier?", "",
          f"* D/off frontier vertices: {fs['off_vertices']}; D/lexicon: {fs['lexicon_vertices']}; identical (FCR,MCR) vertices: {fs['identical_vertices']}; frontiers identical as sets of (FCR,MCR) counts: **{fs['frontiers_identical']}**.",
          f"* D/off vertices dominated by a D/lexicon vertex: **{fs['off_vertices_dominated_by_lexicon']}**; D/lexicon vertices not matched or beaten by D/off: **{fs['lexicon_vertices_not_matched_or_beaten_by_off']}**; D/off vertices not matched or beaten by D/lexicon: {fs['off_vertices_not_matched_or_beaten_by_lexicon']}.",
          f"* matched-MCR levels where D/lexicon FCR is lower / equal / higher: {mm['levels_where_gated_fcr_lower']} / {mm['levels_where_gated_fcr_equal']} / {mm['levels_where_gated_fcr_higher']}.",
          f"* grid taus by matched-tau outcome: {mt['outcome_counts']}; strict-improvement taus: {mt['strict_improvement_taus']}.",
          f"* dominance (grid points): D/lexicon points dominating a D/off point: {len(S['dominance']['grid']['gated_points_dominating_an_off_point'])}; D/off points dominated by a D/lexicon point: {len(S['dominance']['grid']['off_points_dominated_by_a_gated_point'])}; "
          f"D/off points dominating a D/lexicon point: {len(S['dominance']['grid']['off_points_dominating_a_gated_point'])}.", "",
          "### Injected-pair raw cosine by relation × class (D)", ""] + [f"* {k}: n={v['n']}, min {v['min']:.3f}, median {v['median']:.3f}, max {v['max']:.3f}" for k, v in rep["injected_scores"].items()]
    T = rep["tier1_vs_tier2"]
    L += ["", "## 5. Tier 1 vs Tier 2 (descriptive; same 69 source pairs)", "", T["note"], ""]
    for nm in ("D/off", "D/lexicon"):
        L += [f"### {nm}", "", "| tau | Tier-1 context FCR / MCR | Tier-2 context FCR / MCR |", "|---|---|---|"]
        for t, v in T[nm]["fcr_mcr_at_selected_taus"].items():
            L.append(f"| {t} | {v['tier1_context']['fcr']} / {v['tier1_context']['mcr']} | {v['tier2_context']['fcr']} / {v['tier2_context']['mcr']} |")
        L += ["", f"* Tier-1-context frontier vertices (FCR n, MCR n): {T[nm]['tier1_context_frontier']}", f"* Tier-2-context frontier vertices: {T[nm]['tier2_context_frontier']}"]
        for k, v in T[nm]["score_shift"].items():
            L.append(f"* score shift {k}: n={v['n']}, Tier-1 median {v['tier1_median']:.3f}, Tier-2 median {v['tier2_median']:.3f}, median diff {v['median_diff_t2_minus_t1']:+.3f}, {v['pairs_t2_lower']} pairs lower in Tier 2")
        L.append("")
    return "\n".join(L) + "\n"


def write_csvs(rep: dict, prefix: Path) -> list[Path]:
    S = rep["sweep"]
    paths = []
    p = prefix.with_name(prefix.name + "_sweep.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau", "off_fcr_n", "off_fcr_d", "off_mcr_n", "off_mcr_d", "lex_fcr_n", "lex_fcr_d", "lex_mcr_n", "lex_mcr_d", "off_fcr_lo", "off_fcr_hi", "lex_fcr_lo", "lex_fcr_hi"])
        for a, b in zip(S["off"]["points"], S["lexicon"]["points"]):
            w.writerow([a["tau"], a["fcr"]["n"], a["fcr"]["d"], a["mcr"]["n"], a["mcr"]["d"], b["fcr"]["n"], b["fcr"]["d"], b["mcr"]["n"], b["mcr"]["d"], a["fcr"]["lo"], a["fcr"]["hi"], b["fcr"]["lo"], b["fcr"]["hi"]])
    paths.append(p)
    p = prefix.with_name(prefix.name + "_frontier_exact.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "fcr_n", "fcr_d", "mcr_n", "mcr_d", "tau_from", "tau_to"])
        for cond, key in (("D/off", "off"), ("D/lexicon", "lexicon")):
            for v in S[key]["pareto_exact"]:
                for lo, hi in v["intervals"]:
                    w.writerow([cond, v["fcr_n"], 42, v["mcr_n"], 27, lo, "" if hi is None else hi])
    paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix, e.g. results/TIER2_AUDIT_OFFLINE_D_PROVISIONAL")
    ap.add_argument("--run-prefix", default="t2full", help="run-id prefix of the four runs to audit (default: the historical t2full set)")
    ap.add_argument("--passthrough", type=Path, default=None, help="the exact passthrough file a rerun used (default: the historical gold-linked ids)")
    a = ap.parse_args(argv)
    use_run_prefix(a.run_prefix)
    global PASSTHROUGH_FILE
    PASSTHROUGH_FILE = a.passthrough
    outs = [a.out.with_suffix(".json"), a.out.with_suffix(".md"), a.out.with_name(a.out.name + "_sweep.csv"), a.out.with_name(a.out.name + "_frontier_exact.csv")]
    if any(p.exists() for p in outs):
        raise SystemExit("refusing to overwrite an existing report")
    rep = build(a.runs_dir)
    outs[0].write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=list), encoding="utf-8", newline="\n")
    outs[1].write_text(render(rep), encoding="utf-8", newline="\n")
    write_csvs(rep, a.out)
    print("wrote", *outs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
