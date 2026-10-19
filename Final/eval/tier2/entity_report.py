"""Tier-2 full-run analysis, exactly as pre-registered in analysis_plan_full_v1.md (EVALUATOR side: reads gold and injection metadata).

    python -m tier2.entity_report --runs <output dir>/runs --arm v1.0|mirror --evaluator <kernel dir>.evaluator --out <report.json>

Every run is first validated with ``contract.validate_run`` (exact passthrough, QA-id map, the arm's FROZEN file). A run that fails the
contract is reported as a failure and contributes no metric. Nothing here tunes, filters or re-scores a decision: outcomes are read from
the pairwise records, the judge log and the final store as written by the run. The plan's sha256 is pinned in ANALYSIS_PLAN_FREEZE_v1.json;
this module and that plan must not change after full-run results exist.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

from . import contract

HERE = Path(__file__).resolve().parent
ARMS = {"v1.0": (HERE / "data", HERE / "FROZEN_v1.0.json"), "mirror": (HERE / "mirror_v1_1" / "data", HERE / "mirror_v1_1" / "FROZEN_v1.1m.json")}
MERGING = ("merge", "supersede")
B_BOOT = 10_000
SEED = 20261014
Z = 1.959963984540054


# ------------------------------------------------------------------------------------------------ interval / test helpers (plan §5)
def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": p, "ci95": [max(0.0, c - h), min(1.0, c + h)], "method": "wilson"}


def cluster_ratio(clusters: list[tuple[int, int]], seed: int = SEED) -> dict:
    """Ratio of sums (k/n) over clusters, 95% percentile bootstrap resampling clusters (conversations) with replacement."""
    k, n = sum(c[0] for c in clusters), sum(c[1] for c in clusters)
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    rng = random.Random(seed)
    m = len(clusters)
    stats = []
    for _ in range(B_BOOT):
        s = [clusters[rng.randrange(m)] for _ in range(m)]
        sn = sum(c[1] for c in s)
        if sn:
            stats.append(sum(c[0] for c in s) / sn)
    stats.sort()
    return {"k": k, "n": n, "clusters": m, "rate": k / n, "ci95": [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]],
            "method": f"cluster percentile bootstrap over conversations, B={B_BOOT}, seed {SEED}"}


def paired(x: list[int], y: list[int], seed: int = SEED) -> dict:
    """Paired difference mean(y) - mean(x) of 0/1 outcomes on the same units: percentile bootstrap over units + exact McNemar."""
    assert len(x) == len(y)
    n = len(x)
    if n == 0:
        return {"n": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(B_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(y[i] - x[i] for i in idx) / n)
    diffs.sort()
    b = sum(1 for a, c in zip(x, y) if a == 1 and c == 0)      # x only
    c_ = sum(1 for a, c in zip(x, y) if a == 0 and c == 1)     # y only
    m = b + c_
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(b, c_) + 1)) / 2 ** m)
    return {"n": n, "x_rate": sum(x) / n, "y_rate": sum(y) / n, "diff_y_minus_x": (sum(y) - sum(x)) / n,
            "ci95": [diffs[int(0.025 * B_BOOT)], diffs[int(0.975 * B_BOOT) - 1]], "discordant_x_only": b, "discordant_y_only": c_,
            "mcnemar_exact_p": p, "method": f"paired percentile bootstrap over units, B={B_BOOT}, seed {SEED}; exact two-sided McNemar"}


# ------------------------------------------------------------------------------------------------ loading
def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def attribute_pattern(m: dict) -> str:
    if "attribute_pattern" in m:
        return "same_attribute" if m["attribute_pattern"].startswith("same_attribute") else "different_attribute"
    return "same_attribute" if m["fact_a"]["attribute"] == m["fact_b"]["attribute"] else "different_attribute"


def judge_contradiction(raw: str) -> bool:
    try:
        d = json.loads(raw)
        r = d.get("reason", "").lower()
    except (ValueError, AttributeError, TypeError):
        return False
    return ("same entity" in r and d.get("decision") == "keep_both") or (("different entit" in r or "not identifiable" in r) and d.get("decision") == "merge")


def user_only_reason(raw: str) -> bool:
    try:
        return "about the user; b is about the user" in json.loads(raw).get("reason", "").lower()
    except (ValueError, AttributeError, TypeError):
        return False


# ------------------------------------------------------------------------------------------------ per run
def analyse_run(run_dir: Path, arm: str, evaluator: Path) -> dict:
    data, frozen = ARMS[arm]
    rep = contract.validate_run(run_dir, data, passthrough_file=evaluator / "t2_passthrough_used.jsonl",
                                qa_id_map=(evaluator / "qa_id_map.json") if (evaluator / "qa_id_map.json").exists() else None, frozen_file=frozen)
    out = {"run_id": rep["run_id"], "pipeline_config": rep["pipeline_config"], "dkmem_mode": rep["dkmem_mode"], "contract_ok": rep["ok"],
           "contract_errors": rep["errors"][:20], "contract_warnings": rep["warnings"], "counts": rep["counts"], "judge": rep["judge"]}
    if not rep["ok"]:
        out["excluded"] = "contract failed: no metric is computed for this run (plan §7)"
        return out
    man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    meta = {m["injected_pair_id"]: m for m in _jl(data / "tier2_injection_metadata.jsonl")}
    injected = {t for m in meta.values() for t in (m["turn_id_a"], m["turn_id_b"])}
    side = lambda s: run_dir.parent / f"{run_dir.name}.{s}"
    recs = _jl(run_dir / "pairwise_eval.jsonl")
    store = _jl(side("tier2_store_final.jsonl"))
    record_of = {e: m["memory_id"] for m in store for e in m["member_entry_ids"]}
    # merge graph (plan §3: connectivity = the final union-find record; paths through it)
    graph = defaultdict(set)
    for r in recs:
        if r["decision"] in MERGING:
            a, b = r["entry_a"]["entry_id"], r["entry_b"]["entry_id"]
            graph[a].add(b)
            graph[b].add(a)

    def path(a: str, b: str):
        prev, q = {a: None}, deque([a])
        while q:
            x = q.popleft()
            if x == b:
                p = [x]
                while prev[p[-1]] is not None:
                    p.append(prev[p[-1]])
                return p[::-1]
            for y in sorted(graph[x]):
                if y not in prev:
                    prev[y] = x
                    q.append(y)
        return None

    # ---- injected pairs: direct and record-level outcomes (plan §2, §3)
    pairs = []
    for p in rep["injected_pairs"]:
        m = meta[p["injected_pair_id"]]
        ea, eb = p["earlier_turn_id"] + "_e0", p["later_turn_id"] + "_e0"
        direct = p["status"] == "merged"
        connected = bool(p["same_final_memory_record"])
        via = None
        if connected and not direct:
            pth = path(ea, eb)
            inter = [e[:-3] for e in (pth or [])[1:-1]]
            via = {"path_length_edges": (len(pth) - 1) if pth else None, "intermediate_turns": inter,
                   "intermediates_all_distractors": all(t not in injected for t in inter)}
        pairs.append({"injected_pair_id": p["injected_pair_id"], "conversation_id": p["conversation_id"], "relation": p["relation"],
                      "distinction_class": p["distinction_class"], "language": m["language"], "session_distance": p["session_distance"],
                      "earlier_turn_id": p["earlier_turn_id"], "later_turn_id": p["later_turn_id"],
                      "attribute_pattern": attribute_pattern(m), "mirror_of": m.get("mirror_of_injected_pair_id", p["injected_pair_id"]),
                      "direct_status": p["status"], "direct_merged": direct, "connected_final_record": connected,
                      "connection": "direct" if direct else ("indirect_only" if connected else "none"), "indirect": via,
                      "compatibility": p["compatibility"], "similarity": p["similarity"], "rank": p["rank_of_earlier_among_candidates"],
                      "qa": p["qa"]})
    out["pairs"] = pairs
    # ---- comparisons by endpoint role (plan §4)
    kind = lambda r: {0: "distractor_distractor", 1: "injected_distractor", 2: "injected_injected"}[
        (r["entry_a"]["source_utterance_id"] in injected) + (r["entry_b"]["source_utterance_id"] in injected)]
    by_conv = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in recs:
        c = r["entry_a"]["source_utterance_id"].rsplit("_s", 1)[0]
        cell = by_conv[kind(r)][c]
        cell[1] += 1
        cell[0] += r["decision"] in MERGING
    conv_ids = sorted({p["conversation_id"] for p in pairs})
    out["merge_rates"] = {k: cluster_ratio([tuple(by_conv[k].get(c, [0, 0])) for c in conv_ids])
                          for k in ("distractor_distractor", "injected_distractor", "injected_injected")}
    # ---- gate vetoes and bypasses (plan §4)
    if out["dkmem_mode"] != "off":
        if out["pipeline_config"] == "C":
            prop = {j["record_id"]: j["proposal"] for j in _jl(side("judge_log.jsonl"))}
            host_merge = lambda r: prop.get(r["record_id"]) == "merge"
        else:
            tau = man["tau"]
            host_merge = lambda r: r["similarity_score"] > tau
        vetoed = [r for r in recs if host_merge(r) and r["decision"] not in MERGING]
        bypassed = [r for r in vetoed if record_of.get(r["entry_a"]["entry_id"]) == record_of.get(r["entry_b"]["entry_id"])]
        out["gate"] = {"vetoes": len(vetoed), "vetoes_by_compatibility": dict(Counter(r["compatibility"] for r in vetoed)),
                       "vetoes_bypassed_via_intermediate_merges": len(bypassed),
                       "bypassed_by_kind": dict(Counter(kind(r) for r in bypassed)),
                       "injected_pair_vetoes": sum(kind(r) == "injected_injected" for r in vetoed),
                       "injected_pair_vetoes_bypassed": sum(kind(r) == "injected_injected" for r in bypassed)}
    # ---- judge diagnostics (plan §6)
    if out["pipeline_config"] == "C":
        jl = _jl(side("judge_log.jsonl"))
        inj_rec = {f"{p['earlier_turn_id']}_e0::{p['later_turn_id']}_e0" for p in pairs}
        dd = [j for j in jl if j["turn_id_a"] not in injected and j["turn_id_b"] not in injected]
        out["judge_diagnostics"] = {
            "calls": len(jl), "parse_failures": sum(j["status"] != "ok" for j in jl), "possibly_truncated": sum(bool(j.get("possibly_truncated")) for j in jl),
            "at_budget": sum(bool(j.get("at_budget")) for j in jl), "proposals": dict(Counter(str(j["proposal"]) for j in jl)),
            "reason_decision_contradictions": sum(judge_contradiction(j["raw_output"]) for j in jl),
            "distractor_distractor_calls": len(dd), "distractor_distractor_user_only_reasons": sum(user_only_reason(j["raw_output"]) for j in dd),
            "distractor_distractor_user_only_reasons_merged": sum(user_only_reason(j["raw_output"]) and j["proposal"] == "merge" for j in dd),
            "injected_pair_reasons": {j["record_id"]: j["reason"] for j in jl if j["record_id"] in inj_rec}}
    # ---- QA (plan §5; v1.0 arm only)
    qa = rep["qa_scored"]
    if qa:
        by_pair = {p["injected_pair_id"]: p for p in pairs}
        clus = lambda rows: cluster_ratio([(sum(r["correct"] for r in rows if r["conv"] == c), sum(1 for r in rows if r["conv"] == c)) for c in conv_ids])
        rows = [{**q, "conv": by_pair[q["injected_pair_id"]]["conversation_id"], "relation": by_pair[q["injected_pair_id"]]["relation"],
                 "direct": by_pair[q["injected_pair_id"]]["direct_merged"], "connected": by_pair[q["injected_pair_id"]]["connected_final_record"]} for q in qa]
        out["qa"] = {"all": clus(rows), "forbidden_hits": sum(q["forbidden_hit"] for q in qa)}
        for name, f in (("supporting_retrieved", lambda r: r["supporting_memory_retrieved"]), ("supporting_not_retrieved", lambda r: not r["supporting_memory_retrieved"])):
            out["qa"][name] = clus([r for r in rows if f(r)])
        for rel in ("different", "same"):
            for conn in ("connected", "not_connected"):
                sel = [r for r in rows if r["relation"] == rel and (r["connected"] if conn == "connected" else not r["connected"])]
                out["qa"][f"{rel}|{conn}"] = clus(sel)
    return out


# ------------------------------------------------------------------------------------------------ per arm
def summarise(pairs: list[dict]) -> dict:
    diff = [p for p in pairs if p["relation"] == "different"]
    same = [p for p in pairs if p["relation"] == "same"]
    s = {"FCR_direct": wilson(sum(p["direct_merged"] for p in diff), len(diff)),
         "MCR_direct": wilson(sum(not p["direct_merged"] for p in same), len(same)),
         "FCR_record": wilson(sum(p["connected_final_record"] for p in diff), len(diff)),
         "MCR_record": wilson(sum(not p["connected_final_record"] for p in same), len(same)),
         "indirect_only_connections": {"different": sum(p["connection"] == "indirect_only" for p in diff),
                                        "same": sum(p["connection"] == "indirect_only" for p in same)},
         "direct_status": {"different": dict(Counter(p["direct_status"] for p in diff)), "same": dict(Counter(p["direct_status"] for p in same))},
         "retrieval_misses": [p["injected_pair_id"] for p in pairs if p["direct_status"] == "retrieval_miss"],
         "judge_parse_error_pairs": [p["injected_pair_id"] for p in pairs if p["direct_status"] == "judge_parse_error"]}
    strata = {}
    for key in ("distinction_class", "language", "session_distance", "attribute_pattern"):
        st = defaultdict(lambda: {"n": 0, "direct_merged": 0, "connected": 0})
        for p in pairs:
            cell = st[f"{p['relation']}|{key}={p[key]}"]
            cell["n"] += 1
            cell["direct_merged"] += p["direct_merged"]
            cell["connected"] += p["connected_final_record"]
        strata[key] = dict(sorted(st.items()))
    s["strata_counts"] = strata
    return s


def analyse_arm(runs_dir: Path, arm: str, evaluator: Path) -> dict:
    runs = {}
    for rd in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        runs[rd.name] = analyse_run(rd, arm, evaluator)
    out = {"arm": arm, "runs": {}, "failures": [r for r, v in runs.items() if not v["contract_ok"]]}
    for rid, v in runs.items():
        out["runs"][rid] = {**{k: v[k] for k in v if k != "pairs"}, **({"summary": summarise(v["pairs"]), "pairs": v["pairs"]} if v["contract_ok"] else {})}
    # paired contrasts within the arm (plan §5)
    ok = {rid: v for rid, v in runs.items() if v["contract_ok"]}
    key = lambda v: (v["pipeline_config"], v["dkmem_mode"])
    by = {key(v): v for v in ok.values()}
    contrasts = {}
    for cfg in ("C", "D"):
        off, on = by.get((cfg, "off")), by.get((cfg, "lexicon"))
        if off and on:
            for rel in ("different", "same"):
                x = [int(p["direct_merged"]) for p in off["pairs"] if p["relation"] == rel]
                y = [int(p["direct_merged"]) for p in on["pairs"] if p["relation"] == rel]
                contrasts[f"{cfg}: gate lexicon - off, direct merge rate, {rel}"] = paired(x, y)
    for mode in ("off", "lexicon"):
        c, d = by.get(("C", mode)), by.get(("D", mode))
        if c and d:
            for rel in ("different", "same"):
                x = [int(p["direct_merged"]) for p in d["pairs"] if p["relation"] == rel]
                y = [int(p["direct_merged"]) for p in c["pairs"] if p["relation"] == rel]
                contrasts[f"gate {mode}: C - D, direct merge rate, {rel}"] = paired(x, y)
    out["paired_contrasts"] = contrasts
    return out


def across_arms(v10: dict, mirror: dict) -> dict:
    """Mirror - v1.0 on the same source pairs (paired by mirror_of), per config/mode and relation; merge decisions only, never QA."""
    out = {}
    key = lambda r: (r["pipeline_config"], r["dkmem_mode"])
    a = {key(r): r for r in v10["runs"].values() if r["contract_ok"]}
    b = {key(r): r for r in mirror["runs"].values() if r["contract_ok"]}
    for k in sorted(set(a) & set(b)):
        pa = {p["injected_pair_id"]: p for p in a[k]["pairs"]}
        for rel in ("different", "same"):
            mp = [p for p in b[k]["pairs"] if p["relation"] == rel]
            x = [int(pa[p["mirror_of"]]["direct_merged"]) for p in mp]
            y = [int(p["direct_merged"]) for p in mp]
            out[f"{k[0]}/{k[1]}: mirror - v1.0, direct merge rate, {rel}"] = paired(x, y)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, action="append", required=True, help="<output dir>/runs (repeat with --arm for each arm)")
    ap.add_argument("--arm", action="append", required=True, choices=tuple(ARMS))
    ap.add_argument("--evaluator", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not (len(a.runs) == len(a.arm) == len(a.evaluator)):
        ap.error("--runs, --arm and --evaluator must be given once per arm")
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    fz = json.loads((HERE / "ANALYSIS_PLAN_FREEZE_v1.json").read_text(encoding="utf-8"))
    import hashlib
    for rel, want in fz["files_sha256"].items():
        if hashlib.sha256((HERE / rel).read_bytes()).hexdigest() != want:
            raise SystemExit(f"the frozen analysis plan changed: {rel}")
    arms = {arm: analyse_arm(r, arm, e) for r, arm, e in zip(a.runs, a.arm, a.evaluator)}
    rep = {"analysis_plan": fz, "arms": arms}
    if "v1.0" in arms and "mirror" in arms:
        rep["mirror_vs_v1_0_paired"] = across_arms(arms["v1.0"], arms["mirror"])
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("wrote", a.out, "| failures:", {k: v["failures"] for k, v in arms.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
