"""Tier-2 scorer for Evaluation Plan v2 (EVALUATOR side). Parameterised: labels, lexicon, languages and classes are inputs.

    python -m tier2.score_v2 --runs <out>/runs --run-data-dir <arm data dir the runs used> --arm-frozen <that arm's FROZEN json>
        --evaluator <kernel>.evaluator --data-dir <label data dir> [--lexicon <file>] [--languages hi] [--classes kinship,...]
        [--exclude-pairs ids.txt] [--boot 10000] --label PROVISIONAL|FINAL --out <new file>

Per run: (1) provenance: tier2.contract_v2.validate_run (all four configs) + judge-log verification against the ORIGINAL arm (exact
passthrough, QA-id map, frozen file, and the arm's pinned lexicon when it has one); a failing run
is excluded and reported; (2) outcomes: tier2.replay with the given lexicon (C: the logged judge proposals; D: the run's tau), so a new
lexicon needs no inference; (3) metrics with the labels of --data-dir, whose conversations file must equal the runs' (else: rerun).

Plan v2 definitions: PRIMARY = final-memory-record level (same union-find record), SECONDARY = direct pair record. FCR = gold-different
merged / gold-different; MCR = gold-same not merged / gold-same; retrieval misses and judge parse errors stay in the denominators as not
merged. Wilson 95% for rates; paired bootstrap + exact McNemar for contrasts; conversation-cluster bootstrap for comparison-level rates
and QA. Config D: full frontier over every distinct cosine, gate off and lexicon, NO tau selected; matched-MCR on the fixed grid
MCR_GRID with a relation-stratified pair bootstrap; C's points against D's frontiers at C's MCR.

--label FINAL is refused unless eval/EVALUATION_PLAN_FREEZE_v2.json exists and every file it pins still matches.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from harness import _dkmem  # noqa: F401
from dkmem.memory.gate import apply_gate
from dkmem.eval.sensitivity import FEATURE_SETS
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.trace import sha256_file_lf

from . import contract, judge_provenance

from . import qa_norm as Q
from .entity_report import cluster_ratio, paired, wilson
from .replay import JUDGE_HOSTS, Distinctions, _jl, check_same_conversations, compared_pairs, load_turns, replay_run

EVAL = Path(__file__).resolve().parents[1]
FREEZE_V2 = EVAL / "EVALUATION_PLAN_FREEZE_v2.json"
MERGING = ("merge", "supersede")
MCR_GRID = [round(i * 0.05, 2) for i in range(21)]
SEED = 20261014


def check_freeze_v2() -> dict:
    if not FREEZE_V2.exists():
        raise SystemExit("label FINAL refused: eval/EVALUATION_PLAN_FREEZE_v2.json does not exist (plan v2 not frozen)")
    fz = json.loads(FREEZE_V2.read_text(encoding="utf-8"))
    # LF-normalised hashes, so the check is independent of the checkout's line endings (dkmem.pipeline.trace.sha256_file_lf)
    bad = [rel for rel, want in fz["files_sha256"].items() if sha256_file_lf(EVAL.parent / rel) != want]
    if bad:
        raise SystemExit(f"label FINAL refused: frozen v2 files changed: {bad}")
    return fz


# ------------------------------------------------------------------------------------------------ pairs and outcomes
def select_pairs(data_dir: Path, order: dict, languages, classes, exclude, splits=("dev",)) -> tuple[list[dict], set[str], dict]:
    """Pairs of the label directory after the filters. A metadata row without ``split`` (v1 data) counts as dev: those pairs were seen."""
    meta = _jl(Path(data_dir) / "tier2_injection_metadata.jsonl")
    gold = {g["eval_pair_id"]: g for g in _jl(Path(data_dir) / "tier2_gold.jsonl")}
    injected = {t for m in meta for t in (m["turn_id_a"], m["turn_id_b"])}
    pairs, dropped = [], Counter()
    for m in meta:
        g = gold[m["injected_pair_id"]]
        why = ("excluded_id" if m["injected_pair_id"] in exclude else "split" if m.get("split", "dev") not in splits
               else "language" if languages and m["language"] not in languages
               else "class" if classes and m["distinction_class"] not in classes else None)
        if why:
            dropped[why] += 1
            continue
        a, b = sorted((m["turn_id_a"], m["turn_id_b"]), key=order.get)
        pairs.append({"pair": m["injected_pair_id"], "conversation_id": m["conversation_id"], "earlier": a, "later": b,
                      "relation": g["relation"], "class": m["distinction_class"], "language": m["language"], "split": m.get("split", "dev")})
    return pairs, injected, dict(dropped)


def outcomes(pairs: list[dict], rlog: list[dict], replayed: dict) -> list[dict]:
    rec = {r["record_id"]: r for r in replayed["records"]}
    member = {e: m["memory_id"] for m in replayed["store"] for e in m["member_entry_ids"]}
    by_turn = {x["turn_id"]: x for x in rlog}
    no_entry = {x["turn_id"] for x in rlog if x.get("status") == "extraction_failed"}      # configs A and B: the extraction failed, nothing was stored
    out = []
    for p in pairs:
        ea, eb = p["earlier"] + "_e0", p["later"] + "_e0"
        if p["earlier"] in no_entry or p["later"] in no_entry:
            # no entry on a side: the pair cannot be merged. It stays in the denominators as NOT merged (as retrieval misses and judge parse errors do)
            out.append({**p, "direct_status": "no_entry", "direct_merged": False, "record_merged": False})
            continue
        r = rec.get(f"{ea}::{eb}")
        if r is not None:
            status = "merged" if r["decision"] in MERGING else "not_merged"
        elif ea not in by_turn[p["later"]]["candidates"]:
            status = "retrieval_miss"
        else:
            status = "judge_parse_error"
        out.append({**p, "direct_status": status, "direct_merged": status == "merged", "record_merged": member[ea] == member[eb]})
    return out


def rates(outs: list[dict]) -> dict:
    diff = [o for o in outs if o["relation"] == "different"]
    same = [o for o in outs if o["relation"] == "same"]
    w = lambda k, n: {**wilson(k, n), "indicative": n < 20}
    return {"FCR_record": w(sum(o["record_merged"] for o in diff), len(diff)), "MCR_record": w(sum(not o["record_merged"] for o in same), len(same)),
            "FCR_direct": w(sum(o["direct_merged"] for o in diff), len(diff)), "MCR_direct": w(sum(not o["direct_merged"] for o in same), len(same)),
            "outcome_breakdown": {rel: dict(Counter(("merged_record" if o["record_merged"] else o["direct_status"]) for o in outs if o["relation"] == rel))
                                  for rel in ("different", "same")},
            "joined_only_through_intermediates": sum(o["record_merged"] and not o["direct_merged"] for o in outs)}


def by_stratum(outs: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for o in outs:
        groups[o[key]].append(o)
    return {k: rates(v) for k, v in sorted(groups.items())}


# ------------------------------------------------------------------------------------------------ one run
def score_run(rd: Path, *, run_data_dir: Path, arm_frozen: Path, evaluator: Path, turns: dict, pairs: list[dict], injected: set[str],
              lexicon, data_dir: Path, conv_ids: list[str], features=None) -> dict:
    man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
    host, mode = man["pipeline_config"], man["dkmem_mode"]
    qmap = evaluator / "qa_id_map.json"
    # the contract plus the independent judge-log verification (per-row prompt identity, strict re-parse of every raw output): the
    # replay below reads the log's ``proposal`` field, so a log that does not match its manifest must exclude the run
    # an arm that pins the gate's lexicon (v2: <arm root>/lexicon/distinction_features.json): a gated run must have recorded exactly that file
    arm_lexicon = Path(arm_frozen).parent / "lexicon" / "distinction_features.json"
    prov = judge_provenance.validate_run_verified(rd, run_data_dir, passthrough_file=evaluator / "t2_passthrough_used.jsonl",
                                                  qa_id_map=qmap if qmap.exists() else None, frozen_file=arm_frozen,
                                                  lexicon_file=arm_lexicon if arm_lexicon.is_file() else None)
    res = {"run_id": rd.name, "config": host, "mode": mode, "tau": man.get("tau"), "provenance_contract_ok": prov["ok"],
           "provenance_errors": prov["errors"][:10],
           "judge_log_verification": (None if prov["judge_log_verification"] is None else
                                      {k: prov["judge_log_verification"][k] for k in ("status", "family", "parser", "n_rows", "n_identity_verified",
                                                                                      "n_reparsed_ok", "n_reparsed_rejected", "problems")}),
           "embedder_verification": prov["embedder_verification"],
           "provenance_warnings": [w["code"] for w in prov["warnings"]]}
    if not prov["ok"]:
        res["excluded"] = "provenance contract failed: no metric (plan v2 section 2)"
        return res
    side = lambda s: rd.parent / f"{rd.name}.{s}"
    rlog = _jl(side("tier2_retrieval_log.jsonl"))
    props = {j["record_id"]: j["proposal"] for j in _jl(side("judge_log.jsonl"))} if host in JUDGE_HOSTS else None
    rep = replay_run(rlog, turns, host=host, mode=mode, distinctions=Distinctions(lexicon, turns) if mode == "lexicon" else None,
                     tau=man.get("tau"), judge_proposals=props, features=features)
    written_store = _jl(side("tier2_store_final.jsonl"))
    res["final_records_changed_by_lexicon"] = rep["store"] != written_store
    outs = outcomes(pairs, rlog, rep)
    res["overall"] = rates(outs)
    res["by_class"] = by_stratum(outs, "class")
    res["by_language"] = by_stratum(outs, "language")
    res["pairs"] = outs
    # comparison-level merge rates (conversation-cluster bootstrap)
    kind = lambda a, b: "dd" if (a not in injected and b not in injected) else ("ii" if (a in injected and b in injected) else "id")
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rep["records"]:
        a, b = r["record_id"].split("::")
        a, b = a[:-3], b[:-3]
        c = turns[b]["conversation_id"]
        cell = cells[kind(a, b)][c]
        cell[1] += 1
        cell[0] += r["decision"] in MERGING
    res["merge_rates"] = {k: cluster_ratio([tuple(cells[k].get(c, [0, 0])) for c in conv_ids]) for k in ("dd", "id", "ii")}
    # gate vetoes and bypasses
    if mode == "lexicon":
        prop = (lambda r: props.get(r["record_id"]) == "merge") if host in JUDGE_HOSTS else (lambda r: r["similarity_score"] > man["tau"])
        member = {e: m["memory_id"] for m in rep["store"] for e in m["member_entry_ids"]}
        vetoed = [r for r in rep["records"] if prop(r) and r["decision"] not in MERGING]
        byp = [r for r in vetoed if member[r["record_id"].split("::")[0]] == member[r["record_id"].split("::")[1]]]
        res["gate"] = {"vetoes": len(vetoed), "by_compatibility": dict(Counter(r["compatibility"] for r in vetoed)), "bypassed": len(byp)}
    # QA: valid only if the final records are unchanged; rescored with the label directory's references
    refs_p = Path(data_dir) / "tier2_qa_reference.jsonl"
    answers = _jl(side("tier2_qa_answers.jsonl"))
    if not answers or not refs_p.exists() or not refs_p.read_text(encoding="utf-8").strip():
        res["qa"] = {"status": "no QA in this arm"}
    elif res["final_records_changed_by_lexicon"]:
        res["qa"] = {"status": "STALE: the replayed final records differ from the run's; rerun the QA step for this run"}
    else:
        idmap = json.loads(qmap.read_text(encoding="utf-8")) if qmap.exists() else {}
        refs = {r["qa_id"]: r for r in _jl(refs_p)}
        conv_of = {o["pair"]: o for o in outs}
        rows = []
        for a in answers:
            ref = refs.get(idmap.get(a["qa_id"], a["qa_id"]))
            if ref is None or ref["injected_pair_id"] not in conv_of:
                continue
            sc = Q.score_answer(a["raw_answer"] or "", ref)          # the normaliser the reference names (v1 frozen / v2 Indic-safe)
            o = conv_of[ref["injected_pair_id"]]
            sup = all(any(t in x["member_turn_ids"] for x in a["retrieved"]) for t in ref["supporting_turn_ids"])
            rows.append({"conv": o["conversation_id"], "correct": sc["correct"], "forbidden": sc["forbidden_hit"], "supporting": sup,
                         "relation": o["relation"], "connected": o["record_merged"]})
        clus = lambda rs: cluster_ratio([(sum(r["correct"] for r in rs if r["conv"] == c), sum(1 for r in rs if r["conv"] == c)) for c in conv_ids])
        res["qa"] = {"status": "rescored", "all": clus(rows), "forbidden_hits": sum(r["forbidden"] for r in rows),
                     "supporting_retrieved": clus([r for r in rows if r["supporting"]]), "supporting_not_retrieved": clus([r for r in rows if not r["supporting"]]),
                     **{f"{rel}|{'connected' if c else 'not_connected'}": clus([r for r in rows if r["relation"] == rel and r["connected"] == c])
                        for rel in ("different", "same") for c in (True, False)}}
    return res


# ------------------------------------------------------------------------------------------------ D frontier with uncertainty
def d_frontier(rlog: list[dict], turns: dict, pairs: list[dict], lexicon, features=None) -> dict:
    """Per pair, the first index of the descending tau grid at which it is merged (direct) / connected (record), per gate mode.
    Lowering tau only adds merges, so every curve is monotone and a pair's state is fixed by that index."""
    dist = Distinctions(lexicon, turns)
    edges = []
    for x, comps in compared_pairs(rlog):
        for cand, sim, _s, _r in comps:
            a, b = cand[:-3], x["turn_id"]
            g = apply_gate("merge", dist(a), dist(b)) if features is None else apply_gate("merge", dist(a), dist(b), features)
            edges.append((sim, a, b, g.decision == "merge"))
    edges.sort(key=lambda e: -e[0])
    taus = [None] + sorted({e[0] for e in edges}, reverse=True)
    out = {"taus": taus}
    pair_key = {(p["earlier"], p["later"]): i for i, p in enumerate(pairs)}
    for mode in ("off", "lexicon"):
        parent = {t: t for t in turns}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        inf = len(taus)
        direct, record = [inf] * len(pairs), [inf] * len(pairs)
        i = 0
        for k, tau in enumerate(taus):
            changed = False
            while tau is not None and i < len(edges) and edges[i][0] > tau:
                sim, a, b, allowed = edges[i]
                i += 1
                if mode == "lexicon" and not allowed:
                    continue
                j = pair_key.get((a, b))
                if j is not None and direct[j] == inf:
                    direct[j] = k
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                    changed = True
            if changed or k == 0:
                for j, p in enumerate(pairs):
                    if record[j] == inf and find(p["earlier"]) == find(p["later"]):
                        record[j] = k
        out[mode] = {"direct": direct, "record": record}
    return out


def curve(idx: list[int], rel: list[str], sample: list[int], n_grid: int) -> list[tuple[int, float, float]]:
    """(grid index, FCR, MCR) at every index where the sampled pairs' state changes, plus index 0."""
    nd = sum(rel[j] == "different" for j in sample)
    ns = len(sample) - nd
    ev = sorted((idx[j], rel[j]) for j in sample if idx[j] < n_grid)
    pts, fd, ms, e = [(0, 0.0, 1.0 if ns else 0.0)], 0, 0, 0
    for k in sorted({i for i, _ in ev}):
        while e < len(ev) and ev[e][0] <= k:
            if ev[e][1] == "different":
                fd += 1
            else:
                ms += 1
            e += 1
        pts.append((k, fd / nd if nd else 0.0, (ns - ms) / ns if ns else 0.0))
    # include the zero-change state at index 0 properly: pairs merged already at index 0 are counted by the loop
    return pts


def min_fcr_at(pts, m: float) -> float | None:
    """Lowest FCR on the curve with MCR <= m (curves are monotone: FCR rises and MCR falls along the grid)."""
    for _k, f, mm in pts:
        if mm <= m + 1e-12:
            return f
    return None


def frontier_report(fr: dict, pairs: list[dict], c_points: dict, level: str, boot: int) -> dict:
    rel = [p["relation"] for p in pairs]
    n = len(fr["taus"])
    full = list(range(len(pairs)))
    curves = {m: curve(fr[m][level], rel, full, n) for m in ("off", "lexicon")}
    point = lambda k: fr["taus"][k]
    res = {"frontier": {m: [{"tau": point(k), "FCR": f, "MCR": mm} for k, f, mm in curves[m]] for m in curves},
           "matched_mcr": [{"MCR_at_most": m, "FCR_gate_off": min_fcr_at(curves["off"], m), "FCR_gate_lexicon": min_fcr_at(curves["lexicon"], m)}
                           for m in MCR_GRID]}
    # relation-stratified pair bootstrap of the matched-MCR differences and of C versus D at C's MCR
    diff_ix = [j for j in full if rel[j] == "different"]
    same_ix = [j for j in full if rel[j] == "same"]
    rng = random.Random(SEED)
    samples = defaultdict(list)
    c_out = {name: [int(o) for o in outs] for name, outs in c_points.items()}       # per pair 1 = merged at this level
    for _ in range(boot):
        s = [rng.choice(diff_ix) for _ in diff_ix] + [rng.choice(same_ix) for _ in same_ix]
        cv = {m: curve(fr[m][level], rel, s, n) for m in ("off", "lexicon")}
        for m in MCR_GRID:
            a, b = min_fcr_at(cv["off"], m), min_fcr_at(cv["lexicon"], m)
            if a is not None and b is not None:
                samples[f"lexicon_minus_off@{m}"].append(b - a)
        for name, merged in c_out.items():
            nd = len(diff_ix)
            fcr_c = sum(merged[j] for j in s[:nd]) / nd
            mcr_c = sum(1 - merged[j] for j in s[nd:]) / len(same_ix)
            for mode in ("off", "lexicon"):
                d = min_fcr_at(cv[mode], mcr_c)
                if d is not None:
                    samples[f"{name}_minus_D_{mode}@C_MCR"].append(fcr_c - d)
    ci = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[max(0, int(0.975 * len(v)) - 1)]] if v else None
    res["bootstrap"] = {"B": boot, "seed": SEED, "method": "relation-stratified pair bootstrap; frontiers recomputed per resample",
                        "intervals": {k: {"n": len(v), "ci95": ci(v)} for k, v in sorted(samples.items())}}
    for name, merged in c_out.items():
        nd = len(diff_ix)
        fcr_c = sum(merged[j] for j in diff_ix) / nd
        mcr_c = sum(1 - merged[j] for j in same_ix) / len(same_ix)
        res.setdefault("C_points", {})[name] = {"FCR": fcr_c, "MCR": mcr_c,
                                               "D_off_min_FCR_at_C_MCR": min_fcr_at(curves["off"], mcr_c),
                                               "D_lexicon_min_FCR_at_C_MCR": min_fcr_at(curves["lexicon"], mcr_c)}
    return res


# ------------------------------------------------------------------------------------------------ arm
def replay_lexicon(arm_frozen, override=None) -> tuple[Path, bool]:
    """The lexicon the replay applies, and whether it is the arm's pin. Default: the lexicon the arm pins (the one its gated runs used,
    v2: <arm root>/lexicon/distinction_features.json), else the v1 default; any other --lexicon is a counterfactual replay."""
    arm_lexicon = Path(arm_frozen).parent / "lexicon" / "distinction_features.json"
    path = Path(override or (arm_lexicon if arm_lexicon.is_file() else _dkmem.LEXICON_PATH))
    return path, arm_lexicon.is_file() and sha256_file_lf(path) == sha256_file_lf(arm_lexicon)


def score_arm(a) -> dict:
    check_same_conversations(a.runs, a.data_dir)
    turns = load_turns(a.data_dir)
    lexicon_path, lexicon_is_arm_pin = replay_lexicon(a.arm_frozen, a.lexicon)
    lexicon = load_lexicon(lexicon_path)
    runs = sorted(p for p in Path(a.runs).iterdir() if p.is_dir())
    rlog0 = _jl(runs[0].parent / f"{runs[0].name}.tier2_retrieval_log.jsonl")
    order = {x["turn_id"]: i for i, x in enumerate(rlog0)}
    exclude = set(Path(a.exclude_pairs).read_text(encoding="utf-8").split()) if a.exclude_pairs else set()
    pairs, injected, dropped = select_pairs(a.data_dir, order, set(a.languages.split(",")) if a.languages else None,
                                            set(a.classes.split(",")) if a.classes else None, exclude,
                                            ("dev", "test") if a.split == "all" else (a.split,))
    features = None if a.feature_set == "frozen" else FEATURE_SETS[a.feature_set]
    conv_ids = sorted({p["conversation_id"] for p in pairs})
    res = {"inputs": {"runs": str(a.runs), "data_dir": str(a.data_dir), "lexicon": str(lexicon_path), "lexicon_sha256_lf": sha256_file_lf(lexicon_path),
                      "lexicon_is_arm_pin": lexicon_is_arm_pin, "languages": a.languages or "all", "classes": a.classes or "all", "excluded_pairs": sorted(exclude), "dropped": dropped,
                      "split": a.split, "feature_set": {"name": a.feature_set, "classes": sorted(FEATURE_SETS[a.feature_set])},
                      "by_split": dict(Counter(p["split"] for p in pairs)),
                      "pairs": len(pairs), "by_relation": dict(Counter(p["relation"] for p in pairs))},
           "runs": {}}
    for rd in runs:
        res["runs"][rd.name] = score_run(rd, run_data_dir=a.run_data_dir, arm_frozen=a.arm_frozen, evaluator=a.evaluator, turns=turns,
                                         pairs=pairs, injected=injected, lexicon=lexicon, data_dir=a.data_dir, conv_ids=conv_ids, features=features)
    ok = {k: v for k, v in res["runs"].items() if v["provenance_contract_ok"]}
    by = {(v["config"], v["mode"]): v for v in ok.values()}
    contrasts = {}
    for cfg in ("A", "B", "C", "D"):
        off, on = by.get((cfg, "off")), by.get((cfg, "lexicon"))
        if off and on:
            for lvl in ("record", "direct"):
                for rel in ("different", "same"):
                    x = [int(o[f"{lvl}_merged"]) for o in off["pairs"] if o["relation"] == rel]
                    y = [int(o[f"{lvl}_merged"]) for o in on["pairs"] if o["relation"] == rel]
                    contrasts[f"{cfg} gate lexicon - off | {lvl} merge rate | {rel}"] = paired(x, y)
    res["gate_contrasts"] = contrasts
    d_run = next((v for v in ok.values() if v["config"] == "D"), None)
    if d_run:
        rlog = _jl(Path(a.runs) / f"{d_run['run_id']}.tier2_retrieval_log.jsonl")
        fr = d_frontier(rlog, turns, pairs, lexicon, features)
        res["d_frontier"] = {}
        for lvl in ("record", "direct"):
            c_points = {f"{v['config']}_{v['mode']}": [o[f"{lvl}_merged"] for o in v["pairs"]] for v in ok.values() if v["config"] in JUDGE_HOSTS}
            res["d_frontier"][lvl] = frontier_report(fr, pairs, c_points, lvl, a.boot)
        res["d_frontier"]["n_taus"] = len(fr["taus"])
        res["d_frontier"]["note"] = "no tau is selected; tau 0.90 was the executed value only"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--run-data-dir", type=Path, required=True, help="the arm data dir the runs were made with (provenance)")
    ap.add_argument("--arm-frozen", type=Path, required=True, help="that arm's FROZEN json (provenance)")
    ap.add_argument("--evaluator", type=Path, required=True, help="<kernel>.evaluator with t2_passthrough_used.jsonl and qa_id_map.json")
    ap.add_argument("--data-dir", type=Path, required=True, help="label directory (same conversations; final gold, metadata, QA references)")
    ap.add_argument("--lexicon", type=Path, default=None,
                    help="default: the arm's pinned lexicon (v2), else the v1 file; anything else is a counterfactual replay")
    ap.add_argument("--languages", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                    help="held-out test pairs are scored only with --label FINAL (plan v2 section 1); rows without a split count as dev")
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="frozen",
                    help="the gate's cannot-link classes in the replay (frozen = kinship + register; kinship_only = pre-registered sensitivity)")
    ap.add_argument("--exclude-pairs", type=Path, default=None, help="file of injected_pair_ids excluded by the label review")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--label", choices=("PROVISIONAL", "FINAL"), default="PROVISIONAL")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.split != "dev" and a.label != "FINAL":
        raise SystemExit("refused: held-out test pairs are scored only with --label FINAL after the plan v2 freeze")
    freeze = check_freeze_v2() if a.label == "FINAL" else None
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    rep = score_arm(a)
    rep["label"] = a.label
    rep["plan"] = "eval/evaluation_plan_v2.md" + (f" (frozen {freeze['frozen_at_utc']})" if freeze else " (NOT frozen: provisional output)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("wrote", a.out, "| label", a.label, "| pairs", rep["inputs"]["pairs"], "| runs", {k: v["provenance_contract_ok"] for k, v in rep["runs"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
