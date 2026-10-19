"""Tier-2 v2 numbers for the paper (Hindi, Marathi, Gujarati; configs A-D; 7B judge; seed 0), from the stored score_v2 reports. No inference.

    python paper/tools/make_v2_numbers.py           # writes paper/generated/{v2_numbers.tex, v2_table.tex, v2_numbers.json}
    python paper/tools/make_v2_numbers.py --check   # exit 1 if those files are stale

Inputs (gitignored, per-pair labels; held by the lead), pinned by sha256 below, written by tier2.score_v2 from the v2 full run
(eval/tier2_entity_runs/v2_full_20261017/README.md has the commands):
* score_v2_provisional_dev.json: the frozen feature set (kinship + register), the PRIMARY numbers;
* score_v2_provisional_dev_kinship_only.json: the pre-registered kinship-only sensitivity (gate rates only; its QA is stale by design).
Both are PROVISIONAL (plan v2 not frozen) development results. Primary level = final memory record (plan v2). The outputs are aggregates
with no per-pair labels and are committed, so everyone can build the PDF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "eval" / "tier2_entity_runs" / "v2_full_20261017"
PINS = {"score_v2_provisional_dev.json": "ed2fc561a156a7abaf05660abe8cfb1f31587d9fdebefda8db782a6d235f5b56",
        "score_v2_provisional_dev_kinship_only.json": "02c00b59aaa41013f0a099b24e0134fc6e98d06e278600553c6a2795aee4350a"}
OUT = REPO / "paper" / "generated"
MODES = {"off": "Off", "lexicon": "Lex"}
LANGS = {"hi": "Hi", "mr": "Mr", "gu": "Gu"}
CLASSES = {"kinship": "Kin", "honorific_register": "Reg", "name_variant": "Name", "contextual_qualifier": "Qual"}
CONFIG_LABEL = {"A": "A (English gloss)", "B": "B (native-language fact)", "C": "C (verbatim)", "D": "D (cosine, $\\tau$=0.90)"}   # as Table tab:hosts


def rate(m: dict) -> dict:
    return {"k": m["k"], "n": m["n"], "ci95": None if m["ci95"] is None else [m["ci95"][0], m["ci95"][1]]}     # None: empty cell (n = 0)


def check_inputs(primary: dict, kin: dict) -> None:
    for name, r, fs in (("primary", primary, "frozen"), ("kinship_only", kin, "kinship_only")):
        i = r["inputs"]
        assert r["label"] == "PROVISIONAL" and i["split"] == "dev", name
        assert i["feature_set"]["name"] == fs, name
        assert i["lexicon_is_arm_pin"] is True, f"{name}: the replay did not use the arm's pinned lexicon"
        assert len(r["runs"]) == 8 and all(v["provenance_contract_ok"] for v in r["runs"].values()), f"{name}: a run failed provenance"
        assert sorted((v["config"], v["mode"]) for v in r["runs"].values()) == sorted((c, m) for c in "ABCD" for m in MODES), name
    # the primary replay reproduces every run's written final store, so its QA is the run's own
    assert not any(v["final_records_changed_by_lexicon"] for v in primary["runs"].values()), "primary replay differs from a written store"
    assert primary["inputs"]["by_relation"] == kin["inputs"]["by_relation"]


def build(primary: dict, kin: dict) -> dict:
    check_inputs(primary, kin)
    i = primary["inputs"]
    out = {"source": dict(PINS), "label": "PROVISIONAL, development, single seed",
           "pairs": {"total": i["pairs"], "different": i["by_relation"]["different"], "same": i["by_relation"]["same"]}, "runs": {}}
    by = {(v["config"], v["mode"]): v for v in primary["runs"].values()}
    kby = {(v["config"], v["mode"]): v for v in kin["runs"].values()}
    lang_n = {}
    for (cfg, mode), v in sorted(by.items()):
        o = v["overall"]
        e ={"run_id": v["run_id"], "FCR": rate(o["FCR_record"]), "MCR": rate(o["MCR_record"]),
             "by_language": {l: {"FCR": rate(s["FCR_record"]), "MCR": rate(s["MCR_record"])} for l, s in v["by_language"].items()},
             "by_class": {c: {m: rate(s[f"{m}_record"]) for m in ("FCR", "MCR")} for c, s in v["by_class"].items()},
             "qa": {k: v["qa"]["all"][k] for k in ("k", "n", "rate", "ci95")},
             "gate": v.get("gate")}
        if mode == "lexicon":
            ko = kby[(cfg, mode)]["overall"]
            e["kinship_only"] = {"FCR": rate(ko["FCR_record"]), "MCR": rate(ko["MCR_record"])}
        for l, s in e["by_language"].items():
            lang_n.setdefault(l, (s["FCR"]["n"], s["MCR"]["n"]))
            assert lang_n[l] == (s["FCR"]["n"], s["MCR"]["n"]), (l, cfg, mode)
        out["runs"][cfg + MODES[mode]] = e
    out["pairs"]["by_language"] = {l: {"different": d, "same": s} for l, (d, s) in sorted(lang_n.items())}
    out["contrasts"] = {}
    for cfg in "ABCD":
        for rel in ("different", "same"):
            c = primary["gate_contrasts"][f"{cfg} gate lexicon - off | record merge rate | {rel}"]
            out["contrasts"][f"{cfg}|{rel}"] = {k: c[k] for k in ("n", "x_rate", "y_rate", "diff_y_minus_x", "ci95", "discordant_x_only",
                                                                   "discordant_y_only", "mcnemar_exact_p")}
    cp = primary["d_frontier"]["record"]["C_points"]
    out["d_frontier"] = {"n_taus": primary["d_frontier"]["n_taus"],
                         "at_judge_points": {k: {"FCR": p["FCR"], "MCR": p["MCR"], "D_off_min_FCR": p["D_off_min_FCR_at_C_MCR"],
                                                 "D_lexicon_min_FCR": p["D_lexicon_min_FCR_at_C_MCR"]} for k, p in sorted(cp.items())}}
    return out


def frac(c: dict) -> str:
    return f"{c['k']}/{c['n']}"


def pct(c: dict) -> str:
    return f"{100 * c['k'] / c['n']:.1f}" if c["n"] else "n/a"


def ci(c: dict) -> str:
    return "n/a" if c["ci95"] is None else f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"


def p100(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def macros(n: dict) -> str:
    L = ["% GENERATED by paper/tools/make_v2_numbers.py (Tier 2 v2: hi/mr/gu, PROVISIONAL, development, seed 0). Do not edit.",
         "% e.g. \\VTwoCOffFCR -> 18/68 (record level), \\VTwoCOffFCRpct, \\VTwoCOffFCRci; \\VTwoCLexHiFCR per language; \\VTwoCLexKinFCR per class;",
         "% \\VTwoCLexKOFCR = kinship-only sensitivity; \\VTwoCgateDiffDifferent etc. = paired contrasts (record merge rate, gate on - off)."]
    p = n["pairs"]
    L += [f"\\newcommand{{\\VTwoNPairs}}{{{p['total']}}}", f"\\newcommand{{\\VTwoNDiff}}{{{p['different']}}}",
          f"\\newcommand{{\\VTwoNSame}}{{{p['same']}}}"]
    for l, t in LANGS.items():
        d = p["by_language"][l]
        L += [f"\\newcommand{{\\VTwoN{t}Diff}}{{{d['different']}}}", f"\\newcommand{{\\VTwoN{t}Same}}{{{d['same']}}}"]
    for tag, e in n["runs"].items():
        for m in ("FCR", "MCR"):
            c = e[m]
            L += [f"\\newcommand{{\\VTwo{tag}{m}}}{{{frac(c)}}}", f"\\newcommand{{\\VTwo{tag}{m}pct}}{{{pct(c)}}}",
                  f"\\newcommand{{\\VTwo{tag}{m}ci}}{{{ci(c)}}}"]
            for l, t in LANGS.items():
                L.append(f"\\newcommand{{\\VTwo{tag}{t}{m}}}{{{frac(e['by_language'][l][m])}}}")
            for cl, t in CLASSES.items():
                L.append(f"\\newcommand{{\\VTwo{tag}{t}{m}}}{{{frac(e['by_class'][cl][m])}}}")
            if "kinship_only" in e:
                c = e["kinship_only"][m]
                L += [f"\\newcommand{{\\VTwo{tag}KO{m}}}{{{frac(c)}}}", f"\\newcommand{{\\VTwo{tag}KO{m}ci}}{{{ci(c)}}}"]
        L.append(f"\\newcommand{{\\VTwo{tag}QA}}{{{e['qa']['k']}/{e['qa']['n']}}}")
        L.append(f"\\newcommand{{\\VTwo{tag}QApct}}{{{100 * e['qa']['rate']:.1f}}}")
        if e["gate"]:
            L.append(f"\\newcommand{{\\VTwo{tag}Vetoes}}{{{e['gate']['vetoes']}}}")
    for key, c in n["contrasts"].items():
        cfg, rel = key.split("|")
        b = f"VTwo{cfg}gate{rel.capitalize()}"
        L += [f"\\newcommand{{\\{b}Diff}}{{{c['diff_y_minus_x']:.2f}}}", f"\\newcommand{{\\{b}Diffci}}{{[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]}}",
              f"\\newcommand{{\\{b}McNemar}}{{{c['mcnemar_exact_p']:.4f}}}",
              f"\\newcommand{{\\{b}Discordant}}{{{c['discordant_x_only']} vs {c['discordant_y_only']}}}"]
    for k, pt in n["d_frontier"]["at_judge_points"].items():
        cfg, mode = k.split("_")
        b = f"VTwoDFront{cfg}{MODES[mode]}"
        L += [f"\\newcommand{{\\{b}Off}}{{{p100(pt['D_off_min_FCR'])}}}", f"\\newcommand{{\\{b}Lex}}{{{p100(pt['D_lexicon_min_FCR'])}}}"]
    L.append(f"\\newcommand{{\\VTwoDFrontNTaus}}{{{n['d_frontier']['n_taus']}}}")
    return "\n".join(L) + "\n"


def table(n: dict) -> str:
    L = ["% GENERATED by paper/tools/make_v2_numbers.py. Do not edit.", "\\begin{tabular}{llccc}", "\\toprule",
         "Config & DK-Mem & FCR & MCR & QA \\\\", "\\midrule"]
    for cfg in "ABCD":
        for mode, mt in MODES.items():
            e = n["runs"][cfg + mt]
            L.append(f"{CONFIG_LABEL[cfg]} & {'gate on' if mode == 'lexicon' else 'gate off'} & {frac(e['FCR'])} {ci(e['FCR'])} & "
                     f"{frac(e['MCR'])} {ci(e['MCR'])} & {e['qa']['k']}/{e['qa']['n']} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


def render(n: dict) -> dict[str, str]:
    return {"v2_numbers.tex": macros(n), "v2_table.tex": table(n), "v2_numbers.json": json.dumps(n, indent=1, sort_keys=True) + "\n"}


def load_pinned(run_dir: Path = RUN_DIR) -> tuple[dict, dict]:
    raw = {}
    for name, want in PINS.items():
        b = (run_dir / name).read_bytes()
        if hashlib.sha256(b).hexdigest() != want:
            raise SystemExit(f"{run_dir / name} does not match its pinned sha256: rescore into a new file and re-pin")
        raw[name] = json.loads(b)
    return raw["score_v2_provisional_dev.json"], raw["score_v2_provisional_dev_kinship_only.json"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files = render(build(*load_pinned()))
    if a.check:
        stale = [f for f, t in files.items() if not (OUT / f).exists() or (OUT / f).read_text(encoding="utf-8") != t]
        print("stale:" if stale else "up to date", *stale)
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for f, t in files.items():
        (OUT / f).write_text(t, encoding="utf-8", newline="\n")
    print("wrote", *[f"paper/generated/{f}" for f in files])
    return 0


if __name__ == "__main__":
    sys.exit(main())
