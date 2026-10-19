"""Score a kinship-probe run of the frozen v2 judge (EVALUATOR side). Implements criteria_v1.json exactly.

    python -m tier2.kinship_probe.score --outputs <judge_outputs.jsonl> --out <report.json>

Refuses to score unless FREEZE.json exists and every pinned file still matches it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES, CRITERIA, FREEZE = HERE / "cases_v1.jsonl", HERE / "criteria_v1.json", HERE / "FREEZE.json"
REASON = re.compile(r"^\s*A is about (.*?);\s*B is about (.*?);\s*(.*)$", re.S)


def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def verify_freeze() -> dict:
    if not FREEZE.exists():
        raise SystemExit("kinship probe is not frozen yet (FREEZE.json missing): scoring refused")
    fz = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = [rel for rel, want in fz["files_sha256"].items() if hashlib.sha256((HERE / rel).read_bytes()).hexdigest() != want]
    if bad:
        raise SystemExit(f"frozen kinship-probe files changed: {bad}")
    return fz


def _roots(term: str) -> list[str]:
    t = term.lower()
    return [t, t[:-2]] if t.endswith("ji") and len(t) > 4 else [t]


def entity_kind(entity: str, term: str, english_words: list[str]) -> tuple[str, str | None]:
    """('source', term) | ('english', word) | ('other', None) for one named entity."""
    e = (entity or "").lower()
    if any(re.search(rf"\b{re.escape(r)}", e) for r in _roots(term)):
        return "source", term
    for w in sorted(english_words, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", e):
            return "english", w
    return "other", None


def parse_decision(raw: str):
    from harness import _dkmem  # noqa: F401
    from dkmem.memory.judge import parse_judge_output
    r = parse_judge_output(raw)
    return (r.decision, r.reason) if r.ok else (None, None)


def naming_mode(reason: str | None, case: dict, order: str, english_words: list[str]) -> tuple[str, dict]:
    if reason is None:
        return "unparsed", {}
    m = REASON.match(reason)
    if not m:
        return "unparsed", {}
    ta, tb = case["terms"] if order == "ab" else case["terms"][::-1]
    ka, wa = entity_kind(m.group(1), ta, english_words)
    kb, wb = entity_kind(m.group(2), tb, english_words)
    info = {"entity_a": m.group(1), "entity_b": m.group(2), "kind_a": ka, "kind_b": kb, "word_a": wa, "word_b": wb,
            "meri_as_name": any(re.search(r"\b(meri|mere)('s|\s+(" + "|".join(map(re.escape, english_words)) + r"))\b", x.lower())
                                for x in (m.group(1), m.group(2)))}
    if ka == kb == "source" and ta != tb:
        return "source_distinct", info
    if ka == kb == "english":
        return ("english_same" if wa == wb else "english_different"), info
    return "mixed", info


def score(outputs: list[dict], cases: list[dict], criteria: dict) -> dict:
    from tier2.entity_report import paired
    words = criteria["naming_classifier"]["english_kin_words"]
    by = {c["case_id"]: c for c in cases}
    calls = []
    for o in outputs:
        c = by[o["case_id"]]
        dec, reason = parse_decision(o["raw_output"])
        mode, info = naming_mode(reason, c, o["order"], words)
        calls.append({"case_id": c["case_id"], "order": o["order"], "group": c["group"], "cell": c["cell"], "term_origin": c["term_origin"],
                      "terms": c["terms"], "expected": c["expected"], "decision": dec, "mode": mode, **info,
                      "contradiction": bool(reason) and (("same entity" in reason.lower() and dec == "keep_both") or
                                                         (("different entit" in reason.lower() or "not identifiable" in reason.lower()) and dec == "merge"))})
    missing = {(c, o) for c in by for o in ("ab", "ba")} - {(x["case_id"], x["order"]) for x in calls}
    if missing:
        raise ValueError(f"outputs do not cover every case in both orders: {sorted(missing)[:5]}")
    # raw counts: mode x group x cell x order -> merges / calls
    table = defaultdict(lambda: [0, 0])
    for x in calls:
        cell = table[f"{x['group']}|{x['cell']}|{x['order']}|{x['mode']}"]
        cell[0] += x["decision"] == "merge"
        cell[1] += 1
    counts = {k: {"merges": v[0], "calls": v[1]} for k, v in sorted(table.items())}
    hd = [x for x in calls if x["group"] == "H-diff"]
    es = [x for x in hd if x["mode"] == "english_same"]
    ot = [x for x in hd if x["mode"] != "english_same"]
    rate = lambda xs: (sum(x["decision"] == "merge" for x in xs), len(xs))
    m_es, m_ot = rate(es), rate(ot)
    mech = criteria["preregistered_diagnostics"]["mechanism"]
    if m_es[1] >= 8 and m_es[0] / m_es[1] >= 0.80 and (m_ot[1] == 0 or m_ot[0] / m_ot[1] <= 0.10):
        verdict = "consistent_with_hypothesis"
    elif m_es[1] >= 8 and m_es[0] / m_es[1] <= 0.30:
        verdict = "inconsistent_with_hypothesis"
    else:
        verdict = "inconclusive"
    mechanism = {"english_same": {"merges": m_es[0], "calls": m_es[1]}, "other_modes": {"merges": m_ot[0], "calls": m_ot[1]},
                 "by_mode": {m: dict(zip(("merges", "calls"), rate([x for x in hd if x["mode"] == m]))) for m in sorted({x["mode"] for x in hd})},
                 "verdict": verdict, "wording": mech["wording"]}
    # structure: P-same vs N-diff paired by term pair and order
    key = lambda x: (tuple(x["terms"]), x["order"])
    ps = {key(x): x for x in hd if x["cell"] == "P-same"}
    nd = {key(x): x for x in hd if x["cell"] == "N-diff"}
    units = sorted(set(ps) & set(nd))
    structure = {}
    for outcome, f in (("merge", lambda x: int(x["decision"] == "merge")), ("english_same", lambda x: int(x["mode"] == "english_same"))):
        r = paired([f(nd[u]) for u in units], [f(ps[u]) for u in units])
        r["structure_matters"] = r["diff_y_minus_x"] >= 0.25 and r["ci95"][0] > 0
        r["note"] = "y = P-same, x = N-diff"
        structure[outcome] = r
    structure["N-same"] = {m: dict(zip(("merges", "calls"), rate([x for x in hd if x["cell"] == "N-same" and x["mode"] == m])))
                           for m in sorted({x["mode"] for x in hd if x["cell"] == "N-same"})}
    grp = lambda g: [x for x in calls if x["group"] == g]
    report = {
        "criteria_version": criteria["version"], "n_calls": len(calls),
        "parse_failures": sum(x["decision"] is None for x in calls), "reason_decision_contradictions": sum(x["contradiction"] for x in calls),
        "meri_as_name": sum(bool(x.get("meri_as_name")) for x in calls),
        "order_consistency": {"same_decision_cases": sum(1 for c in by if len({x["decision"] for x in calls if x["case_id"] == c}) == 1), "cases": len(by)},
        "counts": counts, "mechanism": mechanism, "structure": structure,
        "controls": {"H-same-named": {cell: dict(zip(("merges", "calls"), rate([x for x in grp("H-same-named") if x["cell"] == cell]))) for cell in ("P-same", "N-same", "N-diff")},
                     "E-explicit": dict(zip(("merges", "calls"), rate(grp("E-explicit")))),
                     "E-collapsed": dict(zip(("merges", "calls"), rate(grp("E-collapsed")))),
                     "possessive": {"without_possessive": dict(zip(("merges", "calls"), rate(grp("possessive")))),
                                    "with_possessive_same_pairs": dict(zip(("merges", "calls"), rate(
                                        [x for x in hd if x["cell"] == "P-same" and tuple(x["terms"]) in {tuple(y["terms"]) for y in grp("possessive")}])))},
                     "term_origin": {o: dict(zip(("merges", "calls"), rate([x for x in hd if x["term_origin"] == o]))) for o in ("benchmark_pair", "new_pair")}},
        "limitations": criteria["limitations_reported_with_every_result"],
        "calls": calls}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    fz = verify_freeze()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    outs = _jl(a.outputs)
    ids = {(o.get("prompt_id"), o.get("prompt_sha256")) for o in outs}
    if ids != {(fz["prompt_id"], fz["prompt_sha256"])}:
        raise SystemExit(f"outputs were not made with the frozen v2 prompt: {ids}")
    rep = score(outs, _jl(CASES), json.loads(CRITERIA.read_text(encoding="utf-8")))
    rep["freeze"] = fz
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({"mechanism": rep["mechanism"]["verdict"], "structure_merge": rep["structure"]["merge"]["structure_matters"],
                      "structure_english_same": rep["structure"]["english_same"]["structure_matters"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
