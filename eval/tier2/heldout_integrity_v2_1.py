"""Read-only integrity audit of the v2.1 Hindi held-out specs against the Team B items whose labels were exposed in VERIFICATION_PROMPT.md.
Reports, per held-out spec, its nearest structural neighbour among the v2.1 dev specs (frame similarity after dropping city/weekday/month
values) and whether that neighbour is an exposed test item. A high-similarity exposed neighbour is a signal only (function words such as 'mein rehti' inflate it); the status
of the set rests on provenance (the author saw the exposed labels), which no similarity score can undo.
    python -m tier2.heldout_integrity_v2_1 --pkg data/tier2_v2_1 [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import qa_norm as Q

VALUES = set("pune delhi patna agra kanpur ranchi nagpur jaipur surat mumbai nashik lucknow chennai indore kolkata bhopal ahmedabad rajkot somvar mangalvar "
             "budhvar guruvar shukravar shanivar ravivar january may august march november june july october december april september".split())
THRESHOLD = 0.40       # frame similarity at or above which an exposed neighbour counts as a template source


def _skel(t: str) -> set:
    return {w for w in Q.normalize_text(t).split() if w not in VALUES}


def _sim(a: str, b: str) -> float:
    A, B = _skel(a), _skel(b)
    return len(A & B) / max(1, len(A | B))


def _pair_sim(h: dict, s: dict) -> float:
    return max(_sim(h["text_a"], s["text_a"]) + _sim(h["text_b"], s["text_b"]), _sim(h["text_a"], s["text_b"]) + _sim(h["text_b"], s["text_a"])) / 2


def audit(pkg: Path) -> dict:
    dev = json.loads((pkg / "dev/pair_specs_v2_1.json").read_text(encoding="utf-8"))["specs"]
    ho = json.loads((pkg / "heldout_sealed/pair_specs_heldout_v1.json").read_text(encoding="utf-8"))["specs"]
    rows = {r["eval_pair_id"]: r for r in map(json.loads, (pkg / "dev/splits_v2_1.jsonl").read_text(encoding="utf-8").splitlines())}
    exposed = {k for k, v in rows.items() if v["origin_split"] == "team_b_test_EXPOSED"}
    items = []
    for h in ho:
        best = max(dev, key=lambda s: _pair_sim(h, s))
        best_ex = max((s for s in dev if s["spec_id"] in exposed), key=lambda s: _pair_sim(h, s))
        items.append({"spec_id": h["spec_id"], "class": h["distinction_class"], "relation": h["relation"],
                      "nearest": best["spec_id"], "nearest_sim": round(_pair_sim(h, best), 2), "nearest_is_exposed": best["spec_id"] in exposed,
                      "nearest_exposed": best_ex["spec_id"], "nearest_exposed_sim": round(_pair_sim(h, best_ex), 2),
                      "template_derived_from_exposed": _pair_sim(h, best_ex) >= THRESHOLD})
    n = sum(i["template_derived_from_exposed"] for i in items)
    return {"threshold": THRESHOLD, "n_heldout": len(items), "n_template_derived_from_exposed": n,
            "n_frames_identical_to_dev_items": sum(i["nearest_sim"] >= 0.99 for i in items),
            "verdict": "NOT_INDEPENDENT" if n else "no exposed template overlap detected (still not blind: author saw the exposed labels)", "items": items}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkg", type=Path, default=Path("data/tier2_v2_1"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = audit(a.pkg)
    txt = json.dumps(rep, ensure_ascii=False, indent=1) + "\n"
    if a.out:
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; refusing to overwrite")
        a.out.write_text(txt, encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "items"}, ensure_ascii=False, indent=1))
    for i in rep["items"]:
        if i["template_derived_from_exposed"]:
            print(i["spec_id"], "<-", i["nearest_exposed"], i["nearest_exposed_sim"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
