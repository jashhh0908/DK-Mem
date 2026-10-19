"""Offline lexicon replay for Tier 1 A-D run groups (no inference; nothing in the source group is modified).

A Tier 1 A-D group logs, per pair, the host's proposal (``trace.gate.off.proposed``) and the two utterances. The gate's distinctions
are a pure function of the lexicon and the utterance (mode ``lexicon``: ``dkmem.pipeline.trace.lexicon_distinction``; mode
``lexicon+llm``: ``dkmem.memory.dkmem_extract.resolve_distinction``, which needs the model's distinction only for a class the lexicon
flags as ambiguous, and the trace keeps that distinction as ``extraction.<side>.model_distinction`` whenever it was extracted). So
re-gating with a new lexicon needs no model call, EXCEPT when the new lexicon makes an utterance ambiguous and no model distinction
was logged: such a pair is reported as ``needs_inference`` and skipped in that mode exactly as the runner skips it.

    python -m tier1.lexicon_replay verify --group <group dir>                                   # replay with the group's own lexicon
    python -m tier1.lexicon_replay replay --group <group dir> --lexicon <file> --out <NEW dir>  # write a re-gated copy of the group

The written group has the source layout (run_config.json, trace.jsonl, <mode>/{run_manifest.json, pairwise_eval.jsonl}), so
``dkmem.eval.cli`` scores it unchanged; ``run_config.json`` records the replay in ``lexicon_replay`` and carries the new lexicon sha.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from harness import _dkmem  # noqa: F401  (puts dkmem on sys.path)
from dkmem.memory.dkmem_extract import resolve_distinction
from dkmem.memory.gate import apply_gate
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.trace import lexicon_distinction, sha256_file_lf
from dkmem.tier1.io import DECISION_TRANSLATION, translate_distinction

GATED_MODES = ("lexicon", "lexicon+llm")
SLUG = {"off": "off", "lexicon": "lexicon", "lexicon+llm": "lexicon-llm"}


class ReplayError(ValueError):
    pass


def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load(group: Path) -> dict:
    group = Path(group)
    cfg = json.loads((group / "run_config.json").read_text(encoding="utf-8"))
    pc = cfg.get("pipeline_config")
    if not (isinstance(pc, dict) and pc.get("config_id") in ("A", "B", "C", "D")):
        raise ReplayError(f"{group.name}: not an A-D group (the replay needs one host proposal per pair)")
    rows = {m: _jl(group / SLUG[m] / "pairwise_eval.jsonl") for m in cfg["dkmem_modes"]}
    return {"path": group, "cfg": cfg, "trace": _jl(group / "trace.jsonl"), "rows": rows}


def _distinctions(mode: str, t: dict, lexicon) -> tuple:
    inp, lang = t["input"], t["language"]
    if mode == "lexicon":
        return lexicon_distinction(lexicon, inp["utterance_a"], lang), lexicon_distinction(lexicon, inp["utterance_b"], lang)
    ext = lambda side: (SimpleNamespace(distinction=t["extraction"][side]["model_distinction"])
                        if (t.get("extraction") or {}).get(side, {}).get("model_distinction") is not None else None)
    return (resolve_distinction(lexicon, inp["utterance_a"], lang, ext("a")),
            resolve_distinction(lexicon, inp["utterance_b"], lang, ext("b")))


def _marker(distinction: dict, mode: str):
    """The pairwise-row form of a distinction, exactly as MemoryEntry.to_dict writes it."""
    m = translate_distinction(distinction, extraction_method=mode)
    return m.to_dict() if m is not None else None


def replay(g: dict, lexicon) -> dict:
    """New trace and pairwise rows for every gated mode; the off mode is copied unchanged."""
    trace, rows = copy.deepcopy(g["trace"]), {m: [] for m in g["rows"]}
    by_rid = {m: {r["record_id"]: r for r in g["rows"][m]} for m in g["rows"]}
    off_rows = by_rid.get("off", {})
    stats = {m: {"changed_decisions": 0, "needs_inference": 0, "pairs": 0} for m in GATED_MODES if m in g["rows"]}
    for t in trace:
        if t.get("status") != "ok":
            continue
        proposed = t["gate"]["off"]["proposed"]
        rid = t["record_id"]
        if "off" in rows:
            rows["off"].append(off_rows[rid])
        for mode in (m for m in GATED_MODES if m in rows):
            stats[mode]["pairs"] += 1
            da, db = _distinctions(mode, t, lexicon)
            old_final = (t.get("final") or {}).get(mode)
            if da is None or db is None:
                stats[mode]["needs_inference"] += 1
                t["gate"][mode] = {"applied": False, "skipped": "no V4 extraction for the model fallback"}
                t["final"][mode] = None
                continue
            gr = apply_gate(proposed, da, db)
            t["gate"][mode] = {"applied": True, "distinction_a": da, "distinction_b": db, "compatibility": gr.compatibility,
                               "proposed": gr.proposed, "decision": gr.decision, "vetoed": gr.vetoed, "reason": gr.reason}
            ext = DECISION_TRANSLATION[gr.decision]
            t["final"][mode] = {"decision": gr.decision, "pairwise_decision": ext, "record_id": rid}
            if old_final is None or old_final["decision"] != gr.decision:
                stats[mode]["changed_decisions"] += 1
            base = by_rid[mode].get(rid)
            if base is None:                                 # skipped before, resolvable now: build the row from the off row
                if not g["rows"][mode] and rid not in off_rows:
                    raise ReplayError(f"{rid}: no template row for mode {mode}")
                base = copy.deepcopy(off_rows[rid])
                tmpl = g["rows"][mode][0] if g["rows"][mode] else None
                base["run_id"] = tmpl["run_id"] if tmpl else base["run_id"].rsplit("-", 1)[0] + "-" + SLUG[mode]
                base["strategy"] = tmpl["strategy"] if tmpl else base["strategy"]
            r = copy.deepcopy(base)
            r["decision"], r["compatibility"] = ext, gr.compatibility
            r["predicted_entity_id"] = r["entry_a"]["entry_id"] if ext in ("merge", "supersede") else None
            r["entry_a"]["distinction"] = _marker(da, mode)
            r["entry_b"]["distinction"] = _marker(db, mode)
            rows[mode].append(r)
    return {"trace": trace, "rows": rows, "stats": stats}


def verify(group: Path, lexicon_path: Path | None = None) -> dict:
    """Replay with the lexicon the group was run with: trace gate/final blocks and every pairwise row must be identical."""
    g = load(group)
    lp = Path(lexicon_path or _dkmem.LEXICON_PATH)
    if sha256_file_lf(lp) != g["cfg"]["lexicon"]["sha256"]:
        raise ReplayError(f"{lp} is not the lexicon this group used ({g['cfg']['lexicon']['sha256'][:12]}...)")
    rep = replay(g, load_lexicon(lp))
    same_rows = {m: rep["rows"][m] == g["rows"][m] for m in g["rows"]}
    same_trace = [({k: t.get(k) for k in ("gate", "final")}) for t in rep["trace"]] == [({k: t.get(k) for k in ("gate", "final")}) for t in g["trace"]]
    return {"group": g["cfg"]["group_id"], "rows_identical": same_rows, "trace_identical": same_trace, "stats": rep["stats"]}


def write(group: Path, lexicon_path: Path, out: Path) -> dict:
    """Write a re-gated copy of ``group`` to the NEW directory ``out`` (verifies the source replay first)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ReplayError(f"{out} exists and is not empty")
    v = verify(group)
    if not (v["trace_identical"] and all(v["rows_identical"].values())):
        raise ReplayError(f"{group}: replay with the original lexicon does not reproduce the group: {v}")
    g = load(group)
    rep = replay(g, load_lexicon(lexicon_path))
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(g["cfg"])
    cfg["lexicon_replay"] = {"source_group": str(Path(group)), "source_lexicon_sha256": g["cfg"]["lexicon"]["sha256"],
                             "replay_lexicon_sha256": sha256_file_lf(lexicon_path), "replay_lexicon_path": str(lexicon_path),
                             "stats": rep["stats"], "note": "host proposals reused from the source trace; only the gate was re-applied"}
    cfg["lexicon"] = {"sha256": sha256_file_lf(lexicon_path)}
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    with open(out / "trace.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for t in rep["trace"]:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    for mode, rows in rep["rows"].items():
        d = out / SLUG[mode]
        d.mkdir()
        shutil.copyfile(Path(group) / SLUG[mode] / "run_manifest.json", d / "run_manifest.json")
        with open(d / "pairwise_eval.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for extra in ("summary.json", "invocation.json"):
        if (Path(group) / extra).exists():
            shutil.copyfile(Path(group) / extra, out / f"source_{extra}")
    return {"out": str(out), "stats": rep["stats"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--group", type=Path, nargs="+", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--group", type=Path, required=True)
    r.add_argument("--lexicon", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        ok = True
        for grp in a.group:
            try:
                res = verify(grp)
            except ReplayError as e:
                print(json.dumps({"group": str(grp), "skipped": str(e)}))
                continue
            ok &= res["trace_identical"] and all(res["rows_identical"].values())
            print(json.dumps(res))
        return 0 if ok else 1
    print(json.dumps(write(a.group, a.lexicon, a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
