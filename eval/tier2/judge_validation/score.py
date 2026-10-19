"""Score a judge-validation run against the frozen cases and the pre-registered criteria (EVALUATOR side: reads the labels).

    python -m tier2.judge_validation.score --outputs <judge_outputs.jsonl> [--out <report.json>]

``judge_outputs.jsonl`` is what the validation kernel writes: one row per call with ``case_id``, ``order`` (ab|ba), ``raw_output``,
``output_tokens``, ``max_new_tokens`` and the prompt id/sha256 it used. The kernel never receives the labels. Every frozen file is
checked against FREEZE.json before anything is scored; a parse failure is a wrong answer, never repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_v1.jsonl"
CRITERIA = HERE / "criteria_v1.json"
FREEZE = HERE / "FREEZE.json"
ORDERS = ("ab", "ba")


def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_freeze() -> dict:
    """Every file FREEZE.json pins must still hash to its pin (raises otherwise)."""
    fz = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = [rel for rel, want in fz["files_sha256"].items() if _sha(HERE.parent / rel) != want]
    if bad:
        raise RuntimeError(f"frozen judge-validation files changed: {bad}")
    return fz


def parse(raw):
    from harness import _dkmem  # noqa: F401  (puts dkmem on sys.path)
    from dkmem.memory.judge import parse_judge_output
    r = parse_judge_output(raw)
    return r.decision if r.ok else None


def _select(cases: list[dict], sub: dict) -> list[dict]:
    out = []
    for c in cases:
        if "role" in sub and c["role"] != sub["role"]:
            continue
        if "expected" in sub and c["expected"] != sub["expected"]:
            continue
        if "category" in sub and c["category"] not in sub["category"]:
            continue
        if c["category"] in sub.get("exclude_category", ()):
            continue
        if "attribute_pattern" in sub and c["attribute_pattern"] not in sub["attribute_pattern"]:
            continue
        out.append(c)
    return out


def score(outputs: list[dict], cases: list[dict], criteria: dict) -> dict:
    by_case = {c["case_id"]: c for c in cases}
    calls = {}
    for o in outputs:
        key = (o["case_id"], o["order"])
        if key in calls:
            raise ValueError(f"duplicate call {key}")
        calls[key] = {**o, "decision": parse(o["raw_output"])}
    missing = [(c, o) for c in by_case for o in ORDERS if (c, o) not in calls]
    unknown = [k for k in calls if k[0] not in by_case or k[1] not in ORDERS]
    if missing or unknown:
        raise ValueError(f"outputs do not cover the frozen cases exactly: missing {missing[:5]}, unknown {unknown[:5]}")
    ids = {(o.get("prompt_id"), o.get("prompt_sha256")) for o in outputs}
    if ids != {(criteria["prompt_id"], criteria["prompt_sha256"])}:
        raise ValueError(f"outputs were not made with the frozen prompt: {ids}")

    def acc(sel: list[dict]) -> dict:
        n = ok = 0
        for c in sel:
            for o in ORDERS:
                n += 1
                ok += calls[(c["case_id"], o)]["decision"] == c["expected"]
        return {"n_calls": n, "correct": ok, "rate": (ok / n) if n else None}

    all_calls = list(calls.values())
    decisions = Counter(str(c["decision"]) for c in all_calls)
    gates = {}
    for g in criteria["gates"]:
        gid = g["id"]
        if gid == "C1_parse":
            pf = sum(c["decision"] is None for c in all_calls)
            ab = sum(1 for c in all_calls if c.get("output_tokens") is not None and c["output_tokens"] >= c["max_new_tokens"])
            gates[gid] = {"parse_failures": pf, "at_budget": ab, "pass": pf <= g["max_parse_failures"] and ab <= g["max_at_budget"]}
        elif gid == "C6_order_consistency":
            sel = _select(cases, g["subset"])
            same = sum(calls[(c["case_id"], "ab")]["decision"] == calls[(c["case_id"], "ba")]["decision"] is not None for c in sel)
            rate = same / len(sel)
            gates[gid] = {"n_cases": len(sel), "consistent": same, "rate": rate, "pass": rate >= g["min"]}
        elif gid == "C7_not_degenerate":
            shares = {d: decisions.get(d, 0) / len(all_calls) for d in ("merge", "keep_both")}
            gates[gid] = {"shares": shares, "pass": all(s >= g["min_share_each_decision"] for s in shares.values())}
        elif gid == "C8_attribute_shortcut":
            sel = _select(cases, g["subset"])
            cells = defaultdict(list)
            for c in sel:
                cells[f"{c['expected']}|{c['attribute_pattern']}"].append(c)
            res = {k: acc(v) for k, v in sorted(cells.items())}
            gates[gid] = {"cells": res, "pass": len(res) == 4 and all(r["rate"] >= g["min_each_cell"] for r in res.values())}
        else:
            r = acc(_select(cases, g["subset"]))
            gates[gid] = {**r, "pass": r["rate"] is not None and r["rate"] >= g["min"]}
    by_cat = {cat: acc([c for c in cases if c["category"] == cat]) for cat in sorted({c["category"] for c in cases})}
    by_lang = {f"{role}|{lang}": acc([c for c in cases if c["language"] == lang and c["role"] == role])
               for role in ("control", "probe") for lang in sorted({c["language"] for c in cases if c["role"] == role})}
    per_case = [{"case_id": c["case_id"], "category": c["category"], "role": c["role"], "expected": c["expected"],
                 "contested": c["contested"], "ab": calls[(c["case_id"], "ab")]["decision"], "ba": calls[(c["case_id"], "ba")]["decision"],
                 "reason_ab": _reason(calls[(c["case_id"], "ab")]["raw_output"])} for c in cases]
    tok = sorted(c["output_tokens"] for c in all_calls if c.get("output_tokens") is not None)
    return {"criteria_version": criteria["version"], "prompt_id": criteria["prompt_id"], "prompt_sha256": criteria["prompt_sha256"],
            "n_cases": len(cases), "n_calls": len(all_calls), "decisions": dict(decisions),
            "output_tokens": {"n": len(tok), "min": tok[0], "max": tok[-1], "median": tok[len(tok) // 2]} if tok else {"n": 0},
            "gates": gates, "accepted": all(g["pass"] for g in gates.values()),
            "reported_not_gated": {"by_category": by_cat, "by_role_language": by_lang,
                                   "probes": acc([c for c in cases if c["role"] == "probe"]),
                                   "probes_uncontested": acc([c for c in cases if c["role"] == "probe" and not c["contested"]])},
            "per_case": per_case}


def _reason(raw):
    try:
        return json.loads(raw).get("reason")
    except (ValueError, AttributeError, TypeError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    fz = verify_freeze()
    criteria = json.loads(CRITERIA.read_text(encoding="utf-8"))
    criteria["prompt_sha256"] = fz["prompt_sha256"]
    rep = score(_jl(a.outputs), _jl(CASES), criteria)
    rep["freeze"] = {"frozen_at_utc": fz["frozen_at_utc"], "files_sha256": fz["files_sha256"]}
    if a.out:
        if a.out.exists():
            raise SystemExit(f"refusing to overwrite {a.out}")
        a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({"accepted": rep["accepted"], "gates": {k: v["pass"] for k, v in rep["gates"].items()}, "decisions": rep["decisions"]}))
    return 0 if rep["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
