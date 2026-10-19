"""Exact offline replay of a Tier-2 run's consolidation, and the Config D tau x gate sweep built on it (EVALUATOR side for the sweep).

Why a replay is exact. run_tier2 ranks candidates over RAW earlier entries only (memory_store.PersistentStore.rank), so the compared
pairs, their order and their bge-m3 cosines (tier2_retrieval_log.jsonl) do not depend on any merge decision. The host proposal is
  A, B, C: the judge's proposal for that record (judge_log.jsonl; a parse error made no record). A and B store the extraction (gloss / fact), so a
     turn whose extraction failed has NO entry (its retrieval-log row is ``extraction_failed``) and each entry carries its logged stored text and
     each comparison its logged ``record_similarity``
  D: "merge" iff cosine > tau (harness.hosts.EmbeddingThresholdHost.propose)
and with the gate on, dkmem.memory.gate.apply_gate on the lexicon distinctions of the two verbatim entries
(harness.extractors.extract_verbatim with the lexicon), exactly as run_tier2.run_conversation does. Merges join records with the
same PersistentStore. Nothing is re-inferred: a new tau or a new lexicon is replayed from logs alone.

    python -m tier2.replay verify --runs <out dir>/runs --data-dir <arm data dir>          # replay every run and compare byte-for-byte
    python -m tier2.replay sweep  --runs <out dir>/runs --data-dir <arm data dir> --out <new file.json> [--lexicon <file>]

The sweep reports FCR/MCR frontiers for every distinct tau; it never selects a tau.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from harness import _dkmem  # noqa: F401
from dkmem.memory.gate import apply_gate
from dkmem.memory.lexicon import load_lexicon
from dkmem.tier1.io import DECISION_TRANSLATION

from harness.extractors import extract_verbatim
from .memory_store import PersistentStore, StoredEntry

MERGING = ("merge", "supersede")
JUDGE_HOSTS = ("A", "B", "C")        # configs whose host is an LLM judge


def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def check_same_conversations(runs_dir: Path, data_dir: Path) -> None:
    """A replay is valid only on the exact conversation texts the run saw: refuse if the data directory's conversations file differs
    from the one recorded in every run manifest (changed or new texts need an inference rerun, not a replay)."""
    import hashlib
    have = hashlib.sha256((Path(data_dir) / "tier2_conversations.jsonl").read_bytes()).hexdigest()
    for rd in (p for p in Path(runs_dir).iterdir() if p.is_dir()):
        want = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))["tier2"]["conversations_sha256"]
        if want != have:
            raise SystemExit(f"{rd.name} was run on conversations {want[:12]}..., the data directory has {have[:12]}...: rerun, do not replay")


def load_turns(data_dir: Path) -> dict[str, dict]:
    out = {}
    for c in _jl(Path(data_dir) / "tier2_conversations.jsonl"):
        for s in c["sessions"]:
            for t in s["turns"]:
                if t["speaker"] == "user":
                    out[t["turn_id"]] = {"text": t["text"], "language": t["language"], "session_index": s["session_index"], "conversation_id": c["conversation_id"]}
    return out


class Distinctions:
    """Lexicon distinction of each verbatim entry, computed exactly as run_tier2 does (cached)."""

    def __init__(self, lexicon, turns: dict, seed: int = 0):
        self.lexicon, self.turns, self.seed, self._c = lexicon, turns, seed, {}

    def __call__(self, turn_id: str) -> dict:
        if turn_id not in self._c:
            t = self.turns[turn_id]
            self._c[turn_id] = extract_verbatim(turn_id, "a", t["text"], t["language"], seed=self.seed, lexicon=self.lexicon).distinction
        return self._c[turn_id]


def compared_pairs(rlog: list[dict]):
    """(conversation_id, later turn row, [(candidate entry id, similarity, logged status, record id)]) in processing order.

    ``similarity`` is the ranking cosine of the two surfaces, except for configs A and B whose comparisons log ``record_similarity``, the cosine
    of the two STORED texts (what the pairwise record carries). Rows of turns whose extraction failed have no comparisons."""
    for x in rlog:
        sim = {c["entry_id"]: c["similarity"] for c in x["ranked"]}
        comps = [(c["candidate_entry_id"], c.get("record_similarity", sim[c["candidate_entry_id"]]), c["status"], c["record_id"]) for c in x["comparisons"]]
        yield x, comps


def replay_run(rlog: list[dict], turns: dict, *, host: str, mode: str, distinctions: Distinctions | None, tau: float | None = None,
               judge_proposals: dict | None = None, features=None) -> dict:
    """Records and final store of one run, recomputed from its retrieval log (and judge proposals for C). ``features``: the gate's
    cannot-link classes (None = the frozen default, kinship + register; dkmem.eval.sensitivity.FEATURE_SETS names the variants)."""
    if host in JUDGE_HOSTS and judge_proposals is None:
        raise ValueError(f"a {host} replay needs the judge proposals")
    if host == "D" and tau is None:
        raise ValueError("a D replay needs tau")
    if (mode == "lexicon") != (distinctions is not None):
        raise ValueError("mode lexicon needs distinctions; mode off must not have them")
    stores, records = {}, []
    for x, comps in compared_pairs(rlog):
        cid = turns[x["turn_id"]]["conversation_id"]
        store = stores.setdefault(cid, PersistentStore(cid))
        if x.get("status") == "extraction_failed":      # A/B: no entry was written for this turn
            continue
        t = turns[x["turn_id"]]
        store.add(StoredEntry(entry_id=x["entry_id"], turn_id=x["turn_id"], session_index=x["session_index"], surface=t["text"],
                              language=t["language"], extraction=None, entry_dict={}, text=x.get("stored_text")))
        for cand, sim, status, rid in comps:
            if host in JUDGE_HOSTS:
                proposal = judge_proposals.get(rid)
                if proposal is None:                     # judge parse error: no record, no merge (as the runner)
                    continue
            else:
                proposal = "merge" if sim > tau else "keep_both"
            if mode == "lexicon":
                g = (apply_gate(proposal, distinctions(cand[:-3]), distinctions(x["turn_id"])) if features is None
                     else apply_gate(proposal, distinctions(cand[:-3]), distinctions(x["turn_id"]), features))
                decision, compat = g.decision, g.compatibility
            else:
                decision, compat = proposal, None
            ext = DECISION_TRANSLATION[decision]
            records.append({"record_id": rid, "decision": ext, "compatibility": compat, "similarity_score": sim,
                            "threshold": tau if host == "D" else None})
            if ext in MERGING:
                store.merge(cand, x["entry_id"])
    final = [{"conversation_id": cid, **r} for cid, s in stores.items() for r in s.records()]
    return {"records": records, "store": final}


# ------------------------------------------------------------------------------------------------ verification
def run_kind(manifest: dict) -> tuple[str, str]:
    return manifest["pipeline_config"], manifest["dkmem_mode"]


def verify(runs_dir: Path, data_dir: Path, lexicon_path: Path | None = None) -> dict:
    """Replay every run of an output directory and compare records and final store with what the run wrote."""
    check_same_conversations(runs_dir, data_dir)
    turns = load_turns(data_dir)
    lex = load_lexicon(lexicon_path or _dkmem.LEXICON_PATH)
    out = {}
    for rd in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
        host, mode = run_kind(man)
        side = lambda s: rd.parent / f"{rd.name}.{s}"
        rlog = _jl(side("tier2_retrieval_log.jsonl"))
        props = {j["record_id"]: j["proposal"] for j in _jl(side("judge_log.jsonl"))} if host in JUDGE_HOSTS else None
        rep = replay_run(rlog, turns, host=host, mode=mode, distinctions=Distinctions(lex, turns) if mode == "lexicon" else None,
                         tau=man.get("tau"), judge_proposals=props)
        written = _jl(rd / "pairwise_eval.jsonl")
        key = lambda r: (r["record_id"], r["decision"], r["compatibility"], r["similarity_score"], r["threshold"])
        rec_equal = [key(r) for r in rep["records"]] == [key(r) for r in written]
        store_equal = rep["store"] == _jl(side("tier2_store_final.jsonl"))
        out[rd.name] = {"config": host, "mode": mode, "tau": man.get("tau"), "records": len(written),
                        "records_identical": rec_equal, "store_identical": store_equal}
    return out


# ------------------------------------------------------------------------------------------------ Config D sweep
def _wilson(k: int, n: int) -> list | None:
    if n == 0:
        return None
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def sweep_d(rlog: list[dict], turns: dict, pairs: list[dict], distinctions: Distinctions, injected: set[str]) -> dict:
    """Every distinct tau (descending), gate off and lexicon. Incremental union-find: lowering tau only adds merge edges.

    pairs: injected pairs [{pair id, earlier, later, relation}]. FCR = gold-different merged / #different; MCR = gold-same not merged /
    #same; 'direct' = the pair's own record, 'record' = same final memory record."""
    edges = []
    for x, comps in compared_pairs(rlog):
        for cand, sim, _status, _rid in comps:
            a, b = cand[:-3], x["turn_id"]
            allowed = apply_gate("merge", distinctions(a), distinctions(b)).decision == "merge"
            kind = "dd" if (a not in injected and b not in injected) else ("ii" if (a in injected and b in injected) else "id")
            edges.append((sim, a, b, allowed, kind))
    edges.sort(key=lambda e: -e[0])
    sims = sorted({e[0] for e in edges}, reverse=True)
    taus = [None] + sims          # None: tau above every score (no merge); then each distinct score (merge iff sim > tau)
    n_entries = len(turns)
    by_pair = {(p["earlier"], p["later"]): p for p in pairs}
    out = {}
    for mode in ("off", "lexicon"):
        parent = {t: t for t in turns}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        comps, direct, kinds, i, points = n_entries, set(), {"dd": 0, "id": 0, "ii": 0}, 0, []
        for tau in taus:
            # edges with sim > tau are merges at this tau
            while tau is not None and i < len(edges) and edges[i][0] > tau:
                sim, a, b, allowed, kind = edges[i]
                i += 1
                if mode == "lexicon" and not allowed:
                    continue
                kinds[kind] += 1
                if (a, b) in by_pair:
                    direct.add((a, b))
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                    comps -= 1
            diff = [p for p in pairs if p["relation"] == "different"]
            same = [p for p in pairs if p["relation"] == "same"]
            fd = sum((p["earlier"], p["later"]) in direct for p in diff)
            md = sum((p["earlier"], p["later"]) not in direct for p in same)
            fr = sum(find(p["earlier"]) == find(p["later"]) for p in diff)
            mr = sum(find(p["earlier"]) != find(p["later"]) for p in same)
            points.append({"tau": tau, "merge_edges": dict(kinds), "memory_records": comps,
                           "FCR_direct": [fd, len(diff)], "MCR_direct": [md, len(same)], "FCR_record": [fr, len(diff)], "MCR_record": [mr, len(same)]})
        out[mode] = points
    return out


def frontier(points: list[dict], level: str = "direct") -> list[dict]:
    """Pareto points (no other point has FCR <= and MCR <= with one strictly lower)."""
    f = lambda p: (p[f"FCR_{level}"][0] / p[f"FCR_{level}"][1], p[f"MCR_{level}"][0] / p[f"MCR_{level}"][1])
    pts = sorted({f(p): p for p in points}.items())
    out, best_mcr = [], float("inf")
    for (fcr, mcr), p in pts:
        if mcr < best_mcr:
            out.append({"tau": p["tau"], "FCR": fcr, "MCR": mcr})
            best_mcr = mcr
    return out


def matched_mcr(off: list[dict], lex: list[dict], level: str = "direct") -> list[dict]:
    """For every MCR value reached by gate off, the lowest FCR each mode reaches at MCR <= that value (a frontier comparison;
    no tau is chosen)."""
    rate = lambda p, k: p[f"{k}_{level}"][0] / p[f"{k}_{level}"][1]
    rows = []
    for m in sorted({rate(p, "MCR") for p in off}):
        best = lambda ps: min((rate(p, "FCR") for p in ps if rate(p, "MCR") <= m + 1e-12), default=None)
        rows.append({"MCR_at_most": m, "FCR_gate_off": best(off), "FCR_gate_lexicon": best(lex)})
    return rows


def sweep_arm(runs_dir: Path, data_dir: Path, lexicon_path: Path | None, languages: set[str] | None) -> dict:
    check_same_conversations(runs_dir, data_dir)
    turns = load_turns(data_dir)
    lex = load_lexicon(lexicon_path or _dkmem.LEXICON_PATH)
    d_runs = [rd for rd in sorted(Path(runs_dir).iterdir()) if rd.is_dir()
              and json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))["pipeline_config"] == "D"]
    rlog = _jl(d_runs[0].parent / f"{d_runs[0].name}.tier2_retrieval_log.jsonl")      # identical across D runs (verified by verify)
    meta = _jl(Path(data_dir) / "tier2_injection_metadata.jsonl")
    gold = {g["eval_pair_id"]: g for g in _jl(Path(data_dir) / "tier2_gold.jsonl")}
    order = {x["turn_id"]: i for i, x in enumerate(rlog)}
    injected = {t for m in meta for t in (m["turn_id_a"], m["turn_id_b"])}
    pairs = []
    for m in meta:
        if languages and m["language"] not in languages:
            continue
        a, b = sorted((m["turn_id_a"], m["turn_id_b"]), key=order.get)
        pairs.append({"pair": m["injected_pair_id"], "earlier": a, "later": b, "relation": gold[m["injected_pair_id"]]["relation"],
                      "class": m["distinction_class"], "language": m["language"]})
    sw = sweep_d(rlog, turns, pairs, Distinctions(lex, turns), injected)
    for mode in sw:
        for p in sw[mode]:
            for k in ("FCR_direct", "MCR_direct", "FCR_record", "MCR_record"):
                p[k] = {"k": p[k][0], "n": p[k][1], "rate": p[k][0] / p[k][1] if p[k][1] else None, "wilson95": _wilson(*p[k])}
    flat = lambda pts: [{**p, **{k: [p[k]["k"], p[k]["n"]] for k in ("FCR_direct", "MCR_direct", "FCR_record", "MCR_record")}} for p in pts]
    return {"pairs": len(pairs), "languages": sorted(languages) if languages else "all", "taus": len(sw["off"]),
            "points": sw, "frontier_direct": {m: frontier(flat(sw[m])) for m in sw},
            "matched_mcr_direct": matched_mcr(flat(sw["off"]), flat(sw["lexicon"]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--runs", type=Path, required=True)
    v.add_argument("--data-dir", type=Path, required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--runs", type=Path, required=True)
    s.add_argument("--data-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--lexicon", type=Path, default=None, help="lexicon file (default: the repository's dkmem lexicon)")
    s.add_argument("--languages", default=None, help="comma-separated injected-pair languages to score (default: all)")
    s.add_argument("--label", default="PROVISIONAL", help="written into the output; use FINAL only with frozen labels and lexicon")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        rep = verify(a.runs, a.data_dir)
        print(json.dumps(rep, indent=1))
        return 0 if all(r["records_identical"] and r["store_identical"] for r in rep.values()) else 1
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    rep = sweep_arm(a.runs, a.data_dir, a.lexicon, set(a.languages.split(",")) if a.languages else None)
    rep["label"] = a.label
    rep["lexicon"] = str(a.lexicon or _dkmem.LEXICON_PATH)
    rep["note"] = "frontier over every distinct tau; no tau is selected. Rates are descriptive; labels/lexicon status as in 'label'."
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print("wrote", a.out, "| taus", rep["taus"], "| pairs", rep["pairs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
