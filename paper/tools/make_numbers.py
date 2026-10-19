"""Every number in the paper, generated from the verified stored entity report (no inference, no hand-typed numbers).

    python paper/tools/make_numbers.py                  # writes paper/generated/{numbers.tex, results_table.tex, numbers.json}
    python paper/tools/make_numbers.py --check          # exit 1 if the committed generated files are stale

Input: eval/tier2_entity_runs/TIER2_FULL_entity_report_v1.json (sha256 pinned below; produced/verified by tier2.entity_report_verified).
That file holds per-pair gold labels and is gitignored, so only the lead (who holds the artifacts) runs this; the OUTPUTS are aggregates
with no per-pair labels and are committed, so everyone can build the PDF.

Exclusions: paper/review/exclusions.txt (injected pair ids flagged wrong/ambiguous by the review, written by summarize_reviews.py) is read
BY DEFAULT when it exists, for writing and for --check alike, so the committed outputs always match the committed exclusion list. It adds a
sensitivity table computed without those pairs; the primary numbers never change because of it. --exclude FILE uses another list,
--no-exclude ignores it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "eval" / "tier2_entity_runs" / "TIER2_FULL_entity_report_v1.json"
REPORT_SHA256 = "18b5d9df89087f44bf74fc36db022e79db88b6b897595018be32e2dd77e7f096"
OUT = REPO / "paper" / "generated"

ARM_TAG = {"v1.0": "Main", "mirror": "Mirror"}
RUN_TAG = {("C", "off"): "Coff", ("C", "lexicon"): "Clex", ("D", "off"): "Doff", ("D", "lexicon"): "Dlex"}
RUN_LABEL = {("C", "off"): "C, gate off", ("C", "lexicon"): "C, gate on", ("D", "off"): "D, gate off", ("D", "lexicon"): "D, gate on"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval (the method the stored report uses)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rates(pairs: list[dict], *, merged_key: str = "direct_merged") -> dict:
    diff = [p for p in pairs if p["relation"] == "different"]
    same = [p for p in pairs if p["relation"] == "same"]
    fk, mk = sum(bool(p[merged_key]) for p in diff), sum(not p[merged_key] for p in same)
    return {"FCR": {"k": fk, "n": len(diff), "ci95": wilson(fk, len(diff))}, "MCR": {"k": mk, "n": len(same), "ci95": wilson(mk, len(same))}}


def build(report: dict, exclude: set[str]) -> dict:
    out = {"source": {"report": str(REPORT.relative_to(REPO)).replace("\\", "/"), "sha256": REPORT_SHA256}, "arms": {}}
    for arm, a in report["arms"].items():
        arm_out = {"runs": {}, "paired_contrasts": a.get("paired_contrasts", {})}
        for rid, run in a["runs"].items():
            key = (run["pipeline_config"], run["dkmem_mode"])
            pairs = run["pairs"]
            entry = {"run_id": rid, "config": key[0], "mode": key[1],
                     "all": rates(pairs), "hindi": rates([p for p in pairs if p["language"] == "hi"]),
                     "by_language_n": {lang: sum(p["language"] == lang for p in pairs) for lang in sorted({p["language"] for p in pairs})},
                     "gate": run.get("gate"),
                     "qa": {k: run["qa"]["all"][k] for k in ("k", "n", "rate", "ci95")} if "qa" in run else None,   # mirror arm: no QA
                     "retrieval_misses": run["summary"].get("retrieval_misses"), "judge_parse_error_pairs": run["summary"].get("judge_parse_error_pairs")}
            s = run["summary"]
            # the generator must agree with the stored report exactly (lossless: nothing recomputed differently)
            assert (entry["all"]["FCR"]["k"], entry["all"]["FCR"]["n"]) == (s["FCR_direct"]["k"], s["FCR_direct"]["n"]), rid
            assert (entry["all"]["MCR"]["k"], entry["all"]["MCR"]["n"]) == (s["MCR_direct"]["k"], s["MCR_direct"]["n"]), rid
            for m in ("FCR", "MCR"):
                assert all(abs(x - y) < 1e-9 for x, y in zip(entry["all"][m]["ci95"], s[f"{m}_direct"]["ci95"])), (rid, m)
            if exclude:
                kept = [p for p in pairs if p["injected_pair_id"] not in exclude]
                entry["sensitivity_excluding_review_flags"] = {"excluded": sorted({p["injected_pair_id"] for p in pairs} & exclude),
                                                               "all": rates(kept), "hindi": rates([p for p in kept if p["language"] == "hi"])}
            arm_out["runs"][RUN_TAG[key]] = entry
        out["arms"][ARM_TAG[arm]] = arm_out
    return out


def frac(c: dict) -> str:
    return f"{c['k']}/{c['n']}"


def pct(c: dict) -> str:
    return f"{100 * c['k'] / c['n']:.1f}" if c["n"] else "n/a"


def ci(c: dict) -> str:
    lo, hi = c["ci95"]
    return f"[{lo:.2f}, {hi:.2f}]"


def macros(nums: dict) -> str:
    L = ["% GENERATED by paper/tools/make_numbers.py from " + nums["source"]["report"] + " (sha256 " + nums["source"]["sha256"][:12] + "...). Do not edit.",
         "% Usage: \\MainCoffFCR -> 15/42, \\MainCoffFCRpct -> 35.7, \\MainCoffFCRci -> [0.23, 0.51]; prefix Hi for the Hindi-only subset."]
    for arm, a in nums["arms"].items():
        for tag, e in a["runs"].items():
            for sub, pre in (("all", ""), ("hindi", "Hi")):
                for m in ("FCR", "MCR"):
                    c = e[sub][m]
                    base = f"{arm}{tag}{pre}{m}"
                    L += [f"\\newcommand{{\\{base}}}{{{frac(c)}}}", f"\\newcommand{{\\{base}pct}}{{{pct(c)}}}", f"\\newcommand{{\\{base}ci}}{{{ci(c)}}}"]
            g = e.get("gate") or {}
            L.append(f"\\newcommand{{\\{arm}{tag}Vetoes}}{{{g.get('vetoes', 0)}}}")
            if e["qa"]:
                L.append(f"\\newcommand{{\\{arm}{tag}QA}}{{{e['qa']['k']}/{e['qa']['n']}}}")
        pc = a["paired_contrasts"].get("C: gate lexicon - off, direct merge rate, different")
        if pc:
            L += [f"\\newcommand{{\\{arm}CgateDiff}}{{{pc['diff_y_minus_x']:.2f}}}", f"\\newcommand{{\\{arm}CgateDiffci}}{{[{pc['ci95'][0]:.2f}, {pc['ci95'][1]:.2f}]}}",
                  f"\\newcommand{{\\{arm}CgateMcNemar}}{{{pc['mcnemar_exact_p']:.4f}}}",
                  f"\\newcommand{{\\{arm}CgateDiscordant}}{{{pc['discordant_x_only']} vs {pc['discordant_y_only']}}}"]
    return "\n".join(L) + "\n"


def table(nums: dict) -> str:
    L = ["% GENERATED by paper/tools/make_numbers.py. Do not edit.",
         "\\begin{tabular}{llcccc}", "\\toprule",
         "Arm & Run & FCR (all) & MCR (all) & FCR (Hindi) & MCR (Hindi) \\\\", "\\midrule"]
    for arm, a in nums["arms"].items():
        for key, tag in RUN_TAG.items():
            e = a["runs"].get(tag)
            if not e:
                continue
            cells = [f"{frac(e[s][m])} {ci(e[s][m])}" for s in ("all", "hindi") for m in ("FCR", "MCR")]
            L.append(f"{arm} & {RUN_LABEL[key]} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


def render(nums: dict) -> dict[str, str]:
    return {"numbers.tex": macros(nums), "results_table.tex": table(nums),
            "numbers.json": json.dumps(nums, indent=1, ensure_ascii=False, sort_keys=True) + "\n"}


EXCLUSIONS = Path(__file__).resolve().parents[2] / "paper" / "review" / "exclusions.txt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="file of injected pair ids (one per line, # comments); default paper/review/exclusions.txt when it exists")
    ap.add_argument("--no-exclude", action="store_true", help="ignore the exclusion list (no sensitivity table)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if paper/generated differs")
    a = ap.parse_args(argv)
    raw = a.report.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise SystemExit(f"{a.report} is not the verified report (sha256 mismatch): regenerate it with tier2.entity_report_verified")
    if a.no_exclude and a.exclude:
        ap.error("--exclude and --no-exclude are exclusive")
    excl_file = None if a.no_exclude else (a.exclude or (EXCLUSIONS if EXCLUSIONS.exists() else None))
    exclude = set()
    if excl_file:
        exclude = {l.split("#")[0].strip() for l in excl_file.read_text(encoding="utf-8").splitlines()} - {""}
    files = render(build(json.loads(raw), exclude))
    if a.check:
        stale = [n for n, t in files.items() if not (OUT / n).exists() or (OUT / n).read_text(encoding="utf-8") != t]
        print("stale:" if stale else "up to date", *stale)
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (OUT / n).write_text(t, encoding="utf-8", newline="\n")
    print("wrote", *[f"paper/generated/{n}" for n in files])
    return 0


if __name__ == "__main__":
    sys.exit(main())
