"""Offline readiness of the frozen Tier-2 identity judge (merge_judge_entity_t2_v2 on Qwen2.5-7B) for new languages (no inference).

    python -m tier2.judge_readiness check --specs <pair_specs_v2.json> --splits <dev gold> <test gold> [--lexicon <file>] [--languages mr,gu]
    python -m tier2.judge_readiness check-cases --cases <labelled cases.jsonl> --specs <pair_specs_v2.json> [--distractors <bank> ...]

``check`` reads DEV specs only (held-out test sentences are never read here) and reports: the frozen prompt is intact and carries no
language-specific text; prompt rendering keeps the new scripts byte for byte; the frozen validation sets' language coverage (the judge
counts as VALIDATED for a language only if a frozen validation set has items in it and passed); the verbatim extractor and the lexicon on
the new languages; whether the frozen v1 QA normaliser would conflate a slot with a forbidden alias (it does for Gujarati: use qa_norm v2).

``check-cases`` validates a native-authored validation set for a new language before it is frozen: the 7B case schema, per language
the same (category, role, expected) composition as the Hindi part of cases_7b.jsonl (so criteria_7b.json applies unchanged, per
language), unique ids, and disjointness from every v2 benchmark sentence, the distractor banks and the prompt's examples.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from harness import _dkmem  # noqa: F401
from dkmem.memory.judge import render_judge_prompt
from dkmem.memory.lexicon import load_lexicon

from harness.extractors import extract_verbatim
from . import generate_tier2 as G
from . import qa_norm as Q
from .generate_tier2_v2 import load_banks, load_specs, load_splits
from .judge_entity import MERGE_JUDGE_ENTITY_T2_V2, _EXAMPLES_V2
from .judge_host import check_frozen_prompt

HERE = Path(__file__).resolve().parent
FROZEN_SETS = {"v2": HERE / "judge_validation_v2" / "cases_v2.jsonl", "7b": HERE / "judge_validation_7b" / "cases_7b.jsonl"}
TEMPLATE_SET, TEMPLATE_LANG = FROZEN_SETS["7b"], "hi"
LANG_WORDS = ("hindi", "marathi", "gujarati", "german", "english", "language", "script", "devanagari", "kinship", "register")


def _probe(cat: str) -> str:
    return "probe_register" if cat.startswith("probe_register") else cat


def template() -> Counter:
    rows = G.read_jsonl(TEMPLATE_SET)
    return Counter((_probe(r["category"]), r["role"], r["expected"]) for r in rows if r["language"] == TEMPLATE_LANG)


def check(specs: list[dict], split: dict[str, str], lexicon, languages: list[str]) -> dict:
    out: dict = {"languages": languages}
    check_frozen_prompt(MERGE_JUDGE_ENTITY_T2_V2)
    text = (MERGE_JUDGE_ENTITY_T2_V2.system + " " + " ".join(q + " " + a for q, a in MERGE_JUDGE_ENTITY_T2_V2.examples)).lower()
    out["prompt"] = {"id": MERGE_JUDGE_ENTITY_T2_V2.prompt_id, "sha256": MERGE_JUDGE_ENTITY_T2_V2.sha256, "frozen_check": "ok",
                     "language_specific_words": [w for w in LANG_WORDS if w in text]}
    dev = [s for s in specs if split[s["spec_id"]] == "dev"]
    per = {}
    for lang in languages:
        rows = [s for s in dev if s["language"] == lang]
        r = {"dev_specs": len(rows)}
        bad_render, ext_err, detect = 0, 0, Counter()
        for s in rows:
            msgs = render_judge_prompt(s["text_a"], s["text_b"], MERGE_JUDGE_ENTITY_T2_V2)
            user = msgs[-1]["content"]
            bad_render += not (s["text_a"] in user and s["text_b"] in user and unicodedata.is_normalized("NFC", user) == unicodedata.is_normalized("NFC", s["text_a"] + s["text_b"]))
            for mem in ("a", "b"):
                try:
                    e = extract_verbatim(f"{s['spec_id']}_{mem}", mem, s[f"text_{mem}"], lang, seed=0, lexicon=lexicon)
                    detect[(s["distinction_class"], "marked" if e.distinction else "unmarked")] += 1
                except Exception:                      # noqa: BLE001  (reported, not raised)
                    ext_err += 1
        conflated_v1 = conflated_v2 = altered = n_alias = 0
        for s in rows:
            for q in s["qa"]:
                for al in (a for grp in (*q["slots"], *q["forbidden"]) for a in grp):
                    n_alias += 1
                    altered += G.normalize_text(al) != Q.normalize_text(al)
                slots = {G.normalize_text(a) for sl in q["slots"] for a in sl}
                forb = {G.normalize_text(a) for gr in q["forbidden"] for a in gr}
                conflated_v1 += bool(slots & forb) or any(not x for x in slots)
                slots2 = {Q.normalize_text(a) for sl in q["slots"] for a in sl}
                forb2 = {Q.normalize_text(a) for gr in q["forbidden"] for a in gr}
                conflated_v2 += bool(slots2 & forb2) or any(not x for x in slots2)
        r.update(render_mismatches=bad_render, extractor_errors=ext_err, lexicon_detection={f"{k[0]}|{k[1]}": v for k, v in sorted(detect.items())},
                 qa_questions_with_slot_forbidden_collision={"v1_normaliser": conflated_v1, "v2_normaliser": conflated_v2},
                 qa_aliases_mangled_by_v1_normaliser=f"{altered}/{n_alias}")
        per[lang] = r
    out["per_language"] = per
    cover = {}
    for name, p in FROZEN_SETS.items():
        cover[name] = dict(Counter(r["language"] for r in G.read_jsonl(p)))
    out["frozen_validation_coverage"] = cover
    out["tokenizer_length_check"] = "not run (no local Qwen tokenizer); the Kaggle validation kernel logs prompt_tokens per call"
    out["verdict"] = {lang: ("VALIDATED" if any(lang in c for c in cover.values()) else
                             "NOT VALIDATED: no frozen validation items in this language; a native-authored set (check-cases), its freeze and one "
                             "7B validation kernel run (new inference) are required before Config C results in this language are reported")
                      for lang in languages}
    return out


def check_cases(cases_path: Path, specs: list[dict], banks: list[Path]) -> dict:
    rows = G.read_jsonl(cases_path)
    keys = set(G.read_jsonl(TEMPLATE_SET)[0])
    fails = []
    ids = [r.get("case_id") for r in rows]
    if len(set(ids)) != len(ids):
        fails.append("duplicate case ids")
    for r in rows:
        if set(r) != keys:
            fails.append(f"{r.get('case_id')}: keys {sorted(set(r) ^ keys)} differ from the 7B case schema")
    langs = sorted({r.get("language") for r in rows})
    want = template()
    comp = {}
    for lang in langs:
        got = Counter((_probe(r["category"]), r["role"], r["expected"]) for r in rows if r.get("language") == lang)
        comp[lang] = {"matches_template": got == want, "missing": {"|".join(k): v for k, v in (want - got).items()},
                      "extra": {"|".join(k): v for k, v in (got - want).items()}}
        if got != want:
            fails.append(f"{lang}: composition differs from the Hindi part of cases_7b")
    taken = {Q.normalize_text(s[f"text_{m}"]) for s in specs for m in ("a", "b")}
    if banks:
        b = load_banks(banks)
        taken |= {Q.normalize_text(t) for ts in b["user_turns"].values() for t in ts}
        taken |= {Q.normalize_text(e["text"]) for es in b["assistant_turns"].values() for e in es}
    for q, _a in _EXAMPLES_V2:
        taken |= {Q.normalize_text(x.split(": ", 1)[1]) for x in q.split("\n")}
    for r in rows:
        for m in ("text_a", "text_b"):
            if Q.normalize_text(r.get(m, "")) in taken:
                fails.append(f"{r.get('case_id')}: {m} repeats a benchmark, distractor or prompt-example sentence")
    return {"cases": len(rows), "languages": langs, "composition": comp, "ok": not fails, "failures": fails}


def _utf8_stdout() -> None:
    """Reports contain Devanagari/Gujarati; a Windows console encoding (cp1252) must not crash the CLI."""
    import sys
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--specs", type=Path, required=True)
    c.add_argument("--splits", type=Path, nargs="+", required=True)
    c.add_argument("--lexicon", type=Path, default=None)
    c.add_argument("--languages", default="mr,gu")
    k = sub.add_parser("check-cases")
    k.add_argument("--cases", type=Path, required=True)
    k.add_argument("--specs", type=Path, required=True)
    k.add_argument("--distractors", type=Path, nargs="*", default=[])
    a = ap.parse_args(argv)
    specs = load_specs(a.specs)
    if a.cmd == "check":
        res = check(specs, load_splits(a.splits, specs), load_lexicon(a.lexicon or _dkmem.LEXICON_PATH), a.languages.split(","))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    res = check_cases(a.cases, specs, a.distractors)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
