"""Tier-2 run contract validation for ALL four configurations (EVALUATOR side).

``contract.py`` is hash-pinned by ANALYSIS_PLAN_FREEZE_v1.json and validates verbatim runs (configs C and D) only. This module is its
generalisation, derived from it by asserted edits and tested to return the SAME report as the pinned validator on every stored C/D run
(tests/test_contract_v2.py). What it adds:

* Configs A (English gloss) and B (native-language fact): the stored text is the extraction, not the surface. Every turn's extraction is
  logged (``<run>.extraction_log.jsonl``) and independently re-parsed with the strict parser of its prompt (extraction_verify); a turn whose
  extraction failed has NO entry (never a decision), appears in the retrieval log as ``extraction_failed`` and, for an injected pair, makes
  its status ``no_entry``. The record similarity of A/B is a cosine of the two stored texts, logged as ``record_similarity``.
* ``lexicon_file``: a run that used the gate must have recorded the sha256 of the lexicon it is validated against (v2 final lexicon).
* Final memory records are checked to list exactly the stored entries, with texts joined in write order.

Injected-pair status: merged / not_merged / retrieval_miss / judge_parse_error / no_entry (A, B: a side's extraction failed) /
missing_record (a contract violation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluator import jsonschema_lite
from evaluator.loaders import load_schema, relax_pairwise_schema

from . import generate_tier2 as G
from .contract import FROZEN_V1_0, MERGING, SCHEMA_DIR, _jl, _Rep, build_passthrough_ids, serialize_passthrough
from .frozen_judge import (
    IDENTITY_JUDGES, FROZEN_JUDGE_PROMPT_ID, FROZEN_JUDGE_PROMPT_SHA256, JUDGE_ROW_THRESHOLD,
    LEGACY_HARNESS_JUDGE_PROMPT_SHA256, LEGACY_JUDGE_ROW_THRESHOLD,
)

VERBATIM_CONFIGS = ("C", "D")
JUDGE_CONFIGS = ("A", "B", "C")
EXPECTED_STRATEGY_V2 = {("A", "off"): "mem0", ("A", "lexicon"): "dk-mem-lexicon", ("B", "off"): "mem0", ("B", "lexicon"): "dk-mem-lexicon",
                        ("C", "off"): "store-surface-only", ("C", "lexicon"): "dk-mem-lexicon",
                        ("D", "off"): "embedding-threshold", ("D", "lexicon"): "dk-mem-lexicon"}


def validate_run(run_dir: Path, data_dir: Path = G.DATA_DIR, schema_dir: Path = SCHEMA_DIR, *, k: int = 5,
                 passthrough_file: Path | None = None, qa_id_map: Path | dict | None = None, frozen_file: Path = FROZEN_V1_0,
                 lexicon_file: Path | None = None) -> dict:
    """``passthrough_file``: the exact turn-id -> opaque-id file the run was given. Default (None): the historical gold-linked ids
    (``build_passthrough_ids``); the run's recorded ``passthrough_sha256`` must then equal their serialization, so validating a
    run made from another passthrough without naming it is an error, not a silent pass.

    ``qa_id_map``: ``{opaque qa id: frozen qa id}`` for a run whose questions were blinded (the Kaggle smoke payload). Required
    when the manifest records a blinded questions view.

    ``frozen_file``: the FROZEN json of the benchmark arm ``data_dir`` belongs to (v1.0 by default; the mirror arm's is
    mirror_v1_1/FROZEN_v1.1m.json). The run must record that file's version and status.

    ``lexicon_file``: when given, a run that used the lexicon gate must have recorded this file's LF-normalised sha256 (``lexicon_sha256``).

    Differences from ``contract.validate_run`` (pinned, unchanged): configs A and B are accepted (stored text = English gloss / native fact, a
    turn whose extraction failed has NO entry and is reported ``no_entry``, an extraction log is verified); for configs C and D the checks and
    the report are identical (tested on every stored run)."""
    run_dir = Path(run_dir)
    R = _Rep()
    if isinstance(qa_id_map, (str, Path)):
        qa_id_map = json.loads(Path(qa_id_map).read_text(encoding="utf-8"))
    run_id = run_dir.name
    side = lambda suffix: run_dir.parent / f"{run_id}.{suffix}"
    R.check(sorted(p.name for p in run_dir.iterdir()) == ["pairwise_eval.jsonl", "run_manifest.json"], "run_dir_files", "run dir must hold exactly run_manifest.json + pairwise_eval.jsonl")
    man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    recs = _jl(run_dir / "pairwise_eval.jsonl")
    rlog = _jl(side("tier2_retrieval_log.jsonl"))
    qa_rows = _jl(side("tier2_qa_answers.jsonl"))
    if qa_id_map is not None:
        unknown = [a["qa_id"] for a in qa_rows if a["qa_id"] not in qa_id_map]
        R.check(not unknown, "qa_id_not_in_map", f"{len(unknown)} answer rows carry an id the map does not know")
        qa_rows = [{**a, "qa_id": qa_id_map.get(a["qa_id"], a["qa_id"])} for a in qa_rows]
    store = _jl(side("tier2_store_final.jsonl"))

    # ---- frozen benchmark inputs (evaluator side)
    conv_all = {c["conversation_id"]: c for c in G.read_jsonl(data_dir / "tier2_conversations.jsonl")}
    gold = {g["eval_pair_id"]: g for g in G.read_jsonl(data_dir / "tier2_gold.jsonl")}
    meta = {m["injected_pair_id"]: m for m in G.read_jsonl(data_dir / "tier2_injection_metadata.jsonl")}
    questions = {q["qa_id"]: q for q in G.read_jsonl(data_dir / "tier2_qa_questions.jsonl")}
    refs = {r["qa_id"]: r for r in G.read_jsonl(data_dir / "tier2_qa_reference.jsonl")}
    pt_rows = _jl(passthrough_file) if passthrough_file else build_passthrough_ids(data_dir)
    passthrough = {r["turn_id"]: r["gold_entity_id"] for r in pt_rows}

    # ---- manifest
    errs = jsonschema_lite.validate(man, load_schema(schema_dir / "run_manifest.schema.json"))
    R.check(not errs, "manifest_schema", f"manifest fails schema: {errs[:3]}")
    R.check(man.get("dataset_tier") == "tier2_injected_sessions", "manifest_tier", f"dataset_tier {man.get('dataset_tier')!r}")
    cfg, mode = man.get("pipeline_config"), man.get("dkmem_mode")
    verbatim = cfg in VERBATIM_CONFIGS
    R.check(EXPECTED_STRATEGY_V2.get((cfg, mode)) == man.get("strategy"), "manifest_strategy", f"strategy {man.get('strategy')!r} for {cfg}/{mode}")
    t2 = man.get("tier2", {})
    conv_ids = t2.get("conversations", [])
    R.check(bool(conv_ids) and all(c in conv_all for c in conv_ids), "manifest_conversations", "manifest conversations missing or unknown")
    R.check(t2.get("conversations_sha256") == G.sha256_file(data_dir / "tier2_conversations.jsonl"), "manifest_conversations_hash", "conversations hash differs from the frozen file")
    fz = json.loads(Path(frozen_file).read_text(encoding="utf-8"))
    R.check(t2.get("benchmark_version") == fz["version"] and t2.get("frozen_status") == "FROZEN", "manifest_frozen",
            f"run was not made against the frozen benchmark version {fz['version']!r}")
    tau = man.get("tau")
    R.check((cfg == "D") == (tau is not None), "manifest_tau", "tau must be set exactly for config D")
    used_pt_sha = (hashlib.sha256(Path(passthrough_file).read_bytes()).hexdigest() if passthrough_file is not None
                   else hashlib.sha256(serialize_passthrough(pt_rows).encode("utf-8")).hexdigest())
    if t2.get("passthrough_sha256") is not None:
        R.check(t2["passthrough_sha256"] == used_pt_sha, "manifest_passthrough_hash",
                "the run was not made with the passthrough it is being validated against"
                + ("" if passthrough_file is not None else " (pass the run's own file with --passthrough)"))
    blinded = (t2.get("blinded_views") or {}).get("data/tier2_qa_questions.jsonl")
    R.check(not blinded or qa_id_map is not None, "qa_ids_blinded_without_map",
            "the run's questions are a blinded view: pass the evaluator-side map with --qa-id-map")
    if lexicon_file is not None and mode == "lexicon":
        from dkmem.pipeline.trace import sha256_file_lf
        R.check(man.get("lexicon_sha256") == sha256_file_lf(lexicon_file), "lexicon_sha256_mismatch",
                f"the run did not record the sha256 of the lexicon it is being validated against ({Path(lexicon_file).name}): "
                f"recorded {man.get('lexicon_sha256')!r}")
    # ---- config C judge identity and the row-threshold convention that goes with it
    judge_info = None
    if cfg in JUDGE_CONFIGS:
        jid, jsha = man.get("judge_prompt_id"), man.get("judge_prompt_sha256")
        frozen = jid == FROZEN_JUDGE_PROMPT_ID and jsha == FROZEN_JUDGE_PROMPT_SHA256
        identity = jid in IDENTITY_JUDGES and jsha == IDENTITY_JUDGES[jid]
        legacy = jid == FROZEN_JUDGE_PROMPT_ID and jsha == LEGACY_HARNESS_JUDGE_PROMPT_SHA256
        R.check(frozen or identity or legacy, "judge_prompt_not_recognised",
                f"config C judge {jid!r} sha256 {jsha!r} is none of the frozen Tier-1 merge_judge_v1 ({FROZEN_JUDGE_PROMPT_SHA256[:12]}...), "
                f"a frozen Tier-2 identity judge ({IDENTITY_JUDGES}) or the recognised historical harness judge "
                f"({LEGACY_HARNESS_JUDGE_PROMPT_SHA256[:12]}...)")
        if legacy:
            R.warnings.append({"code": "historical_harness_judge", "message": "this run used the harness judge prompt, not the frozen Tier-1 "
                               "merge_judge_v1; its Config C results are not comparable with Tier 1 and are kept as a historical record"})
        judge_info = {"prompt_id": jid, "prompt_sha256": jsha, "frozen_tier1": frozen, "tier2_identity_judge": identity,
                      "historical_harness_judge": legacy,
                      "row_threshold_expected": JUDGE_ROW_THRESHOLD if (frozen or identity) else LEGACY_JUDGE_ROW_THRESHOLD}

    # ---- stored texts and extraction failures (configs A and B; for C and D every user turn is stored verbatim and none can fail)
    failed_turns = {x["turn_id"] for x in rlog if x.get("status") == "extraction_failed"}
    stored_text = {x["turn_id"]: x.get("stored_text") for x in rlog if x.get("status") != "extraction_failed"}
    if verbatim:
        R.check(not failed_turns, "extraction_failed_in_verbatim_config", f"config {cfg} stores every turn verbatim; {len(failed_turns)} rows say extraction failed")

    # ---- turn index of the selected conversations
    turn = {}
    for cid in conv_ids:
        order = 0
        for s in conv_all[cid]["sessions"]:
            for t in s["turns"]:
                if t["speaker"] == "user":
                    turn[t["turn_id"]] = {"conv": cid, "order": order, "session": s["session_index"], "text": t["text"], "language": t["language"]}
                    order += 1
    sel_pairs = {pid: g for pid, g in gold.items() if meta[pid]["conversation_id"] in conv_ids}

    # ---- pairwise records
    pw = relax_pairwise_schema(load_schema(schema_dir / "pairwise_eval.schema.json"), man)
    seen, bad_schema = Counter(), 0
    rec_by_id = {}
    for r in recs:
        e = jsonschema_lite.validate(r, pw)
        if e:
            bad_schema += 1
            R.err("record_schema", f"{r.get('record_id')}: {e[:2]}")
        seen[r.get("record_id")] += 1
        rec_by_id[r.get("record_id")] = r
        R.check(r.get("run_id") == man.get("run_id") and r.get("strategy") == man.get("strategy"), "record_run_strategy", f"{r.get('record_id')}: run_id/strategy differ from the manifest")
        ea, eb = r["entry_a"], r["entry_b"]
        ta, tb = turn.get(ea["source_utterance_id"]), turn.get(eb["source_utterance_id"])
        if not R.check(ta is not None and tb is not None, "record_unknown_turn", f"{r['record_id']}: source_utterance_id is not a user turn of a selected conversation"):
            continue
        R.check(ta["conv"] == tb["conv"], "record_cross_conversation", f"{r['record_id']}: entries from two conversations")
        R.check(ta["order"] < tb["order"], "record_order", f"{r['record_id']}: entry_a must be the earlier turn")
        R.check(r["record_id"] == f"{ea['entry_id']}::{eb['entry_id']}", "record_id_format", r["record_id"])
        for e, t in ((ea, ta), (eb, tb)):
            R.check(e["entry_id"] == f"{e['source_utterance_id']}_e0", "entry_id_format", e["entry_id"])
            if verbatim:
                R.check(e["surface"] == t["text"] and e["gloss"] == t["text"], "entry_not_verbatim", f"{e['entry_id']}: gloss/surface differ from the turn text")
            else:
                src_turn = e["source_utterance_id"]
                R.check(e["surface"] == t["text"], "entry_surface_not_the_turn_text", f"{e['entry_id']}: surface differs from the turn text")
                R.check(isinstance(stored_text.get(src_turn), str) and e["gloss"] == stored_text.get(src_turn), "entry_gloss_not_the_stored_text",
                        f"{e['entry_id']}: gloss differs from the stored text the retrieval log recorded for its turn")
            R.check(e["language"] == t["language"], "entry_language", e["entry_id"])
            R.check(e["gold_entity_id"] == passthrough[e["source_utterance_id"]], "entry_passthrough_id", f"{e['entry_id']}: gold_entity_id is not the passthrough id")
            R.check((e.get("distinction") is None) or mode == "lexicon", "distinction_with_gate_off", e["entry_id"])
        comp, dec, sim = r.get("compatibility"), r["decision"], r["similarity_score"]
        if mode == "off":
            R.check(comp is None, "compat_not_null_gate_off", r["record_id"])
        else:
            R.check(comp in ("compatible", "incompatible", "underdetermined"), "compat_missing_gate_on", r["record_id"])
            if comp in ("incompatible", "underdetermined"):
                R.check(dec not in MERGING, "gate_veto_violated", f"{r['record_id']}: merged despite compat={comp}")
        if cfg == "D":
            R.check(r["threshold"] == tau, "threshold_value", r["record_id"])
            host_merge = sim > tau
            R.check((dec in MERGING) == (host_merge and (mode == "off" or comp == "compatible")), "d_rule_violated", f"{r['record_id']}: decision {dec} vs sim {sim} tau {tau} compat {comp}")
        else:
            # a judge has no tau: null for the frozen judge (dkmem's convention); the historical harness judge wrote 0.0
            want = judge_info["row_threshold_expected"] if judge_info else JUDGE_ROW_THRESHOLD
            R.check(r["threshold"] == want and (r["threshold"] is None) == (want is None), "threshold_value",
                    f"{r['record_id']}: threshold {r['threshold']!r}, expected {want!r}")
    R.check(all(n == 1 for n in seen.values()), "duplicate_record_id", "record_id repeated")

    # ---- retrieval log: structure and consistency with the records
    by_turn = {x["turn_id"]: x for x in rlog}
    R.check(len(by_turn) == len(rlog) == len(turn), "retrieval_log_rows", f"retrieval log has {len(rlog)} rows for {len(turn)} user turns")
    n_comp_records = 0
    for tid, info in turn.items():
        x = by_turn.get(tid)
        if x is None:
            continue
        earlier = [t for t, v in turn.items() if v["conv"] == info["conv"] and v["order"] < info["order"] and t not in failed_turns]
        if tid in failed_turns:
            R.check(x["store_size_before"] == len(earlier) and not x["ranked"] and not x["candidates"] and not x["comparisons"], "extraction_failed_row", tid)
            continue
        R.check(x["store_size_before"] == len(earlier) == len(x["ranked"]), "retrieval_store_size", tid)
        sims = [c["similarity"] for c in x["ranked"]]
        R.check(sims == sorted(sims, reverse=True), "retrieval_not_sorted", tid)
        R.check([c["rank"] for c in x["ranked"]] == list(range(1, len(sims) + 1)), "retrieval_ranks", tid)
        R.check(x["candidates"] == [c["entry_id"] for c in x["ranked"][:k]] and x["top_k"] == k, "retrieval_candidates", tid)
        R.check([c["candidate_entry_id"] for c in x["comparisons"]] == x["candidates"], "comparisons_not_candidates", tid)
        sim_of = {c["entry_id"]: c["similarity"] for c in x["ranked"]}
        for c in x["comparisons"]:
            if c["status"] == "record":
                n_comp_records += 1
                r = rec_by_id.get(c["record_id"])
                if R.check(r is not None, "comparison_without_record", c["record_id"]):
                    want_sim = sim_of[c["candidate_entry_id"]] if verbatim else c.get("record_similarity")
                    R.check(r["decision"] == c["decision"] and r["similarity_score"] == want_sim, "comparison_record_mismatch", c["record_id"])
            else:
                R.check(c["status"] == "judge_parse_error" and c["record_id"] not in rec_by_id, "comparison_status", c["record_id"])
    R.check(n_comp_records == len(recs), "record_count", f"{len(recs)} records vs {n_comp_records} logged comparisons with status 'record'")
    if cfg in JUDGE_CONFIGS:
        jl = _jl(side("judge_log.jsonl"))
        R.check(len(jl) == sum(len(x["comparisons"]) for x in rlog), "judge_log_rows", "judge log rows != comparisons")
        R.check(man.get("judge_parse_failures") == sum(1 for j in jl if j["status"] != "ok"), "judge_parse_failures", "manifest count differs from the judge log")

    ext_info = None
    if not verbatim:
        from . import extraction_verify
        xl_path = side("extraction_log.jsonl")
        xrows = _jl(xl_path) if R.check(xl_path.exists(), "extraction_log_missing", f"config {cfg} writes <run>.extraction_log.jsonl") else []
        ext_info = extraction_verify.verify_extraction_log(cfg, man, xrows, {tid: v["text"] for tid, v in turn.items()})
        for code, n_rows in sorted(ext_info["problems"].items()):
            R.err("extraction_log_" + code, f"{n_rows} extraction-log rows/fields: {code} (first at row {ext_info['first_bad_row'].get(code)}); "
                  "the log does not match the manifest's declared extraction")
        R.check({r["turn_id"] for r in xrows if r.get("status") == "failed"} == failed_turns, "extraction_log_vs_retrieval_log",
                "the turns whose extraction failed differ between the extraction log and the retrieval log")
        R.check({r["turn_id"]: r.get("stored_text") for r in xrows if r.get("status") == "ok"} == stored_text, "extraction_log_stored_text_vs_retrieval_log",
                "the stored texts differ between the extraction log and the retrieval log")

    # ---- the final memory records list exactly the stored entries; a record's text is its members' stored texts, write order, ' | ' separated
    text_of = {f"{tid}_e0": (info["text"] if verbatim else stored_text.get(tid)) for tid, info in turn.items() if tid not in failed_turns}
    R.check(sorted(e for m in store for e in m["member_entry_ids"]) == sorted(text_of), "store_members_not_the_stored_entries",
            "the final memory records do not list exactly the entries that were stored")
    for m in store:
        R.check(m["text"] == " | ".join(str(text_of.get(e)) for e in m["member_entry_ids"]), "store_record_text", m["memory_id"])

    # ---- injected pairs: map decisions back
    final_rec = {}
    for m in store:
        for e in m["member_entry_ids"]:
            final_rec[e] = m["memory_id"]
    pairs_out = []
    for pid, g in sel_pairs.items():
        m = meta[pid]
        ta_, tb_ = g["utterance_a_id"], g["utterance_b_id"]
        first, later = (ta_, tb_) if turn[ta_]["order"] < turn[tb_]["order"] else (tb_, ta_)
        ea_id, eb_id = f"{first}_e0", f"{later}_e0"
        r = rec_by_id.get(f"{ea_id}::{eb_id}")
        x = by_turn.get(later, {})
        rank = next((c["rank"] for c in x.get("ranked", []) if c["entry_id"] == ea_id), None)
        sim = next((c["similarity"] for c in x.get("ranked", []) if c["entry_id"] == ea_id), None)
        comp = next((c for c in x.get("comparisons", []) if c["candidate_entry_id"] == ea_id), None)
        if r is not None:
            status = "merged" if r["decision"] in MERGING else "not_merged"
        elif first in failed_turns or later in failed_turns:
            status = "no_entry"
        elif comp is None:
            status = "retrieval_miss"
        elif comp["status"] == "judge_parse_error":
            status = "judge_parse_error"
        else:
            status = "missing_record"
            R.err("injected_pair_missing_record", pid)
        pairs_out.append({"injected_pair_id": pid, "conversation_id": m["conversation_id"], "relation": g["relation"], "distinction_class": g["distinction_class"],
                          "session_distance": m["session_distance"], "earlier_turn_id": first, "later_turn_id": later, "status": status,
                          "decision": r["decision"] if r else None, "compatibility": r.get("compatibility") if r else None,
                          "similarity": sim, "rank_of_earlier_among_candidates": rank, "top_k": k, "n_earlier_entries": x.get("store_size_before"),
                          "same_final_memory_record": final_rec.get(ea_id) == final_rec.get(eb_id) and ea_id in final_rec})

    # ---- QA
    need = [qid for qid, q in questions.items() if q["conversation_id"] in conv_ids]
    got = Counter(a["qa_id"] for a in qa_rows)
    R.check(sorted(got) == sorted(need) and all(n == 1 for n in got.values()), "qa_coverage", f"QA answers {len(got)} vs questions {len(need)}")
    pair_qa = {}
    qa_scored = []
    for a in qa_rows:
        ref = refs.get(a["qa_id"])
        if ref is None:
            R.err("qa_unknown_id", a["qa_id"])
            continue
        R.check(len(a["retrieved"]) <= k and a["question"] == questions[a["qa_id"]]["question"], "qa_row", a["qa_id"])
        sc = G.score_answer(a["raw_answer"] or "", ref)
        qa_scored.append({"qa_id": a["qa_id"], "injected_pair_id": ref["injected_pair_id"], "role": ref["role"], "qa_type": ref["qa_type"], "correct": sc["correct"],
                          "forbidden_hit": sc["forbidden_hit"], "slots_hit": sc["slots_hit"], "supporting_memory_retrieved": all(
                              any(t in x["member_turn_ids"] for x in a["retrieved"]) for t in ref["supporting_turn_ids"]), "answer": a["raw_answer"]})
    for s in qa_scored:
        pair_qa.setdefault(s["injected_pair_id"], []).append(s)
    for p in pairs_out:
        p["qa"] = [{"qa_id": s["qa_id"], "correct": s["correct"], "forbidden_hit": s["forbidden_hit"], "supporting_memory_retrieved": s["supporting_memory_retrieved"]} for s in pair_qa.get(p["injected_pair_id"], [])]
    report = {"run_id": man.get("run_id"), "strategy": man.get("strategy"), "pipeline_config": cfg, "dkmem_mode": mode, "tau_run_level": tau,
              "benchmark_version": fz["version"],
              "judge": judge_info, "ok": not R.errors, "errors": R.errors, "warnings": R.warnings,
              "counts": {"conversations": len(conv_ids), "user_turns": len(turn), "pairwise_records": len(recs), "schema_invalid_records": bad_schema,
                         "decisions": dict(Counter(r["decision"] for r in recs)), "compatibility": {str(k_): v for k_, v in Counter(r.get("compatibility") for r in recs).items()},
                         "injected_pairs": len(pairs_out), "injected_pair_status": dict(Counter(p["status"] for p in pairs_out)),
                         "qa_answers": len(qa_rows), "qa_correct": sum(s["correct"] for s in qa_scored), "memory_records": len(store)},
              "injected_pairs": pairs_out, "qa_scored": qa_scored,
              "note": "pilot integration report: a handful of conversations; descriptive only, not an evaluation result"}
    if not verbatim:
        report["extraction"] = ext_info
        report["counts"]["extraction_failures"] = len(failed_turns)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a Tier-2 run (configs A-D) against its frozen benchmark arm.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--passthrough", type=Path, default=None, help="the exact passthrough file the run used")
    ap.add_argument("--qa-id-map", type=Path, default=None)
    ap.add_argument("--frozen", type=Path, default=FROZEN_V1_0, help="FROZEN json of the benchmark arm")
    ap.add_argument("--lexicon", type=Path, default=None, help="the lexicon a gated run must have recorded the sha256 of")
    a = ap.parse_args(argv)
    rep = validate_run(a.run_dir, a.data_dir, passthrough_file=a.passthrough, qa_id_map=a.qa_id_map, frozen_file=a.frozen, lexicon_file=a.lexicon)
    out = a.out or a.run_dir.parent / f"{a.run_dir.name}.tier2_contract_report.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(("CONTRACT OK" if rep["ok"] else f"CONTRACT FAILED ({len(rep['errors'])} errors)"), "|", rep["counts"]["injected_pair_status"], "->", out)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
