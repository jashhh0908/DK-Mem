"""Input validation and alignment for Tier 1 runs (Team B contract).

Scope: load, validate, align. No metrics. Findings are split by severity:
  ERROR - the input violates the contract/schema (run is not trustworthy as-is)
  WARN  - suspicious or ignored-by-contract input, or a soft consistency rule
  INFO  - normal model outcomes / descriptive facts (e.g. a pair not compared)
Alignment rule (tier1_eval_contract.md sections 2-4): entry.source_utterance_id ->
key.utterance_a_id / utterance_b_id -> eval_pair_id. gold_entity_id is only a
passthrough check.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import jsonschema_lite
from .loaders import JsonlResult, load_gold, load_manifest, load_pairwise, load_schema, relax_pairwise_schema

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"
MAX_EXAMPLES = 5
RELATIONS = {"same", "different"}
GOLD_REQUIRED = [
    "eval_pair_id", "pair_id", "relation", "distinction_class", "language",
    "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b",
]
EXPECTED_DKMEM_MODE = {"dk-mem-lexicon": "lexicon", "dk-mem-lexicon-llm": "lexicon+llm"}
SEED_GROUP_IGNORED_KEYS = {"run_id", "seed", "created_at"}
# Gold/contract define seven classes; the schema enum allows only the first three.
IN_SCOPE_CLASSES = ("kinship", "honorific_register", "name_variant")
LEGACY_CLASSES = ("evidentiality", "classifier_measure", "spatial_temporal_deixis", "politeness_relationship")
DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "ipdV2" / "dkmem"


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    count: int = 0
    examples: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message,
                "count": self.count, "examples": self.examples}


class _Issues:
    """Collects issues, merging repeats of the same code into one with a count."""

    def __init__(self) -> None:
        self._by_code: dict[str, Issue] = {}

    def add(self, severity: str, code: str, message: str, example: Any = None, n: int = 1) -> None:
        it = self._by_code.get(code)
        if it is None:
            it = self._by_code[code] = Issue(severity, code, message)
        it.count += n
        if example is not None and len(it.examples) < MAX_EXAMPLES:
            it.examples.append(example)

    def list(self) -> list[Issue]:
        return list(self._by_code.values())


def _count(it) -> dict[str, int]:
    return dict(sorted(Counter(it).items()))


# --------------------------------------------------------------------------- gold

@dataclass
class GoldIndex:
    pairs: dict[str, dict]                       # eval_pair_id -> gold record
    utt: dict[str, tuple[str, str, str, str]]    # utterance_id -> (eval_pair_id, side, entity_id, language)
    issues: list[Issue]
    distributions: dict[str, dict[str, int]]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]


def build_gold_index(gold: JsonlResult) -> GoldIndex:
    iss = _Issues()
    for n, err in gold.malformed:
        iss.add(ERROR, "gold_malformed_line", "gold line is not a valid JSON object", f"line {n}: {err}")
    pairs: dict[str, dict] = {}
    utt: dict[str, tuple[str, str, str, str]] = {}
    dup_keys = ("pair_id", "utterance_a_id", "utterance_b_id", "opaque_entity_id_a", "opaque_entity_id_b")
    seen: dict[str, Counter] = {k: Counter() for k in dup_keys}
    for rec in gold.records:
        missing = [f for f in GOLD_REQUIRED if not isinstance(rec.get(f), str) or not rec.get(f)]
        if missing:
            iss.add(ERROR, "gold_missing_field", "gold record missing/non-string required field(s)",
                    {"eval_pair_id": rec.get("eval_pair_id"), "fields": missing})
            continue
        if rec["relation"] not in RELATIONS:
            iss.add(ERROR, "gold_bad_relation", f"relation not in {sorted(RELATIONS)}",
                    {"eval_pair_id": rec["eval_pair_id"], "relation": rec["relation"]})
        if rec["eval_pair_id"] in pairs:
            iss.add(ERROR, "gold_duplicate_eval_pair_id", "duplicate eval_pair_id in gold", rec["eval_pair_id"])
            continue
        pairs[rec["eval_pair_id"]] = rec
        for k in dup_keys:
            seen[k][rec[k]] += 1
    for k, c in seen.items():
        for v, n in c.items():
            if n > 1:
                iss.add(ERROR, f"gold_duplicate_{k}", f"duplicate {k} in gold", v)
    for rec in pairs.values():
        for side in ("a", "b"):
            uid = rec[f"utterance_{side}_id"]
            utt.setdefault(uid, (rec["eval_pair_id"], side, rec[f"opaque_entity_id_{side}"], rec["language"]))
        if rec["utterance_a_id"] == rec["utterance_b_id"]:
            iss.add(ERROR, "gold_same_utterance_ids", "utterance_a_id == utterance_b_id", rec["eval_pair_id"])
        if rec["opaque_entity_id_a"] == rec["opaque_entity_id_b"]:
            iss.add(ERROR, "gold_same_entity_ids", "opaque entity ids must differ", rec["eval_pair_id"])
    dist = {
        "class_x_relation": _count(f"{r['distinction_class']}|{r['relation']}" for r in pairs.values()),
        "language_x_relation": _count(f"{r['language']}|{r['relation']}" for r in pairs.values()),
        "relation": _count(r["relation"] for r in pairs.values()),
    }
    return GoldIndex(pairs, utt, iss.list(), dist)


def validate_gold_file(path: str | Path) -> GoldIndex:
    return build_gold_index(load_gold(path))


# --------------------------------------------------------------------------- run

@dataclass
class RunReport:
    run_dir: str
    run_id: str | None
    strategy: str | None
    manifest: dict | None
    issues: list[Issue]
    # eval_pair_id -> records linking a utt_a entry with a utt_b entry (either orientation)
    pair_records: dict[str, list[dict]]
    # eval_pair_id -> structural status: compared | not_compared | no_entry_a | no_entry_b | no_entry_both
    pair_status: dict[str, str]
    counts: dict[str, Any]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"run_dir": self.run_dir, "run_id": self.run_id, "strategy": self.strategy,
                "ok": self.ok, "manifest": self.manifest, "counts": self.counts,
                "issues": [i.to_dict() for i in self.issues]}


def _value_at(rec: dict, path: str) -> Any:
    node: Any = rec
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def validate_manifest(manifest: dict, schema: dict, iss: _Issues) -> None:
    for path, msg in jsonschema_lite.validate(manifest, schema):
        iss.add(ERROR, "manifest_schema_violation", "run_manifest fails schema", f"{path}: {msg}")
    if manifest.get("dataset_tier") != "tier1_minimal_pairs":
        iss.add(ERROR, "manifest_wrong_dataset_tier", "dataset_tier must be tier1_minimal_pairs",
                manifest.get("dataset_tier"))
    strat, mode = manifest.get("strategy"), manifest.get("dkmem_mode")
    if mode is not None and strat in EXPECTED_DKMEM_MODE and mode != EXPECTED_DKMEM_MODE[strat]:
        iss.add(ERROR, "manifest_dkmem_mode_mismatch", "dkmem_mode disagrees with strategy", f"{strat} vs {mode}")
    if mode is not None and strat not in EXPECTED_DKMEM_MODE and mode != "off":
        iss.add(ERROR, "manifest_dkmem_mode_mismatch", "non-DK-Mem strategy must have dkmem_mode 'off'", mode)
    if strat == "dk-mem-lexicon-llm" and manifest.get("backbone") is None:
        iss.add(WARN, "manifest_llm_strategy_without_backbone", "LLM strategy has backbone null")
    # a lexicon-only run has no model; but with an LLM host (pipeline config A-C) the backbone is the host's
    if strat == "dk-mem-lexicon" and manifest.get("backbone") is not None and manifest.get("pipeline_config") not in ("A", "B", "C"):
        iss.add(WARN, "manifest_lexicon_strategy_with_backbone", "lexicon-only strategy has a backbone",
                manifest.get("backbone"))


def validate_run(
    run_dir: str | Path,
    gold: GoldIndex,
    schema_dir: str | Path = DEFAULT_SCHEMA_DIR,
) -> RunReport:
    """Validate and align one run directory against the gold index. No metrics."""
    run_dir = Path(run_dir)
    iss = _Issues()
    pw_schema = load_schema(Path(schema_dir) / "pairwise_eval.schema.json")
    mf_schema = load_schema(Path(schema_dir) / "run_manifest.schema.json")

    files = sorted(p.name for p in run_dir.iterdir()) if run_dir.is_dir() else []
    if files != ["pairwise_eval.jsonl", "run_manifest.json"]:
        iss.add(WARN, "run_dir_unexpected_files",
                "run directory should contain exactly pairwise_eval.jsonl and run_manifest.json", files)

    manifest: dict | None = None
    try:
        manifest = load_manifest(run_dir / "run_manifest.json")
        validate_manifest(manifest, mf_schema, iss)
    except (OSError, ValueError) as e:
        iss.add(ERROR, "manifest_unreadable", "cannot load run_manifest.json", str(e))
    try:
        data = load_pairwise(run_dir / "pairwise_eval.jsonl")
    except OSError as e:
        iss.add(ERROR, "pairwise_unreadable", "cannot load pairwise_eval.jsonl", str(e))
        data = JsonlResult()
    pw_schema = relax_pairwise_schema(pw_schema, manifest)      # raw-cosine runs only (manifest-declared); else unchanged
    for n, err in data.malformed:
        iss.add(ERROR, "pairwise_malformed_line", "pairwise line is not a valid JSON object", f"line {n}: {err}")

    m_run, m_strat = (manifest or {}).get("run_id"), (manifest or {}).get("strategy")
    pair_records: dict[str, list[dict]] = defaultdict(list)
    seen_record_ids: Counter = Counter()
    entry_src: dict[str, Any] = {}
    entries_by_utt: dict[str, set] = defaultdict(set)
    seen_links: Counter = Counter()
    n_schema_bad = 0
    decisions: Counter = Counter()
    compat: Counter = Counter()
    entry_classes: Counter = Counter()
    scored_compat: list[Any] = []
    scored_has_key: list[bool] = []
    ignored: Counter = Counter()
    run_ids: Counter = Counter()

    for idx, rec in enumerate(data.records):
        rid = rec.get("record_id", f"<record #{idx}>")
        errs = jsonschema_lite.validate(rec, pw_schema)
        if errs:
            n_schema_bad += 1
            for path, msg in errs:
                if path.endswith("distinction.class") and _value_at(rec, path) in LEGACY_CLASSES:
                    iss.add(WARN, "legacy_distinction_class_outside_schema_enum",
                            "delivered legacy Tier 1 data contains distinction classes outside the current "
                            "pairwise_eval.schema.json enum (3 classes: kinship, honorific_register, name_variant), "
                            "while the gold/contract define 7 classes; the 4 legacy classes are kept, not filtered",
                            f"{rid} {path}: {_value_at(rec, path)!r}")
                else:
                    iss.add(ERROR, "schema_violation", "pairwise record fails schema", f"{rid} {path}: {msg}")
        if isinstance(rec.get("record_id"), str):
            seen_record_ids[rec["record_id"]] += 1
        run_ids[rec.get("run_id")] += 1
        if m_run is not None and rec.get("run_id") != m_run:
            iss.add(ERROR, "record_run_id_mismatch", "record run_id differs from manifest", rid)
        if m_strat is not None and rec.get("strategy") != m_strat:
            iss.add(ERROR, "record_strategy_mismatch", "record strategy differs from manifest", rid)
        ea, eb = rec.get("entry_a"), rec.get("entry_b")
        if not (isinstance(ea, dict) and isinstance(eb, dict)):
            continue  # already reported as a schema violation
        decisions[rec.get("decision")] += 1
        compat[rec.get("compatibility")] += 1
        if rec.get("decision") == "supersede" and rec.get("superseded_entry_id") not in (ea.get("entry_id"), eb.get("entry_id")):
            iss.add(WARN, "supersede_id_not_an_entry", "superseded_entry_id is neither entry_a nor entry_b", rid)

        sources = []
        for e in (ea, eb):
            eid, su = e.get("entry_id"), e.get("source_utterance_id")
            if isinstance(eid, str):
                if eid in entry_src and entry_src[eid] != su:
                    iss.add(ERROR, "entry_id_reused_across_utterances",
                            "one entry_id used for different source_utterance_ids", eid)
                entry_src.setdefault(eid, su)
            if su in gold.utt:
                entries_by_utt[su].add(eid)
                _, _, gold_ent, gold_lang = gold.utt[su]
                if e.get("gold_entity_id") != gold_ent:
                    iss.add(ERROR, "gold_entity_id_mismatch",
                            "entry gold_entity_id differs from the key for its source_utterance_id",
                            {"record": rid, "entry_id": eid})
                if e.get("language") != gold_lang:
                    iss.add(WARN, "language_mismatch", "entry language differs from gold language",
                            {"record": rid, "entry_id": eid, "got": e.get("language"), "want": gold_lang})
            sources.append(su)
            d = e.get("distinction")
            entry_classes[d.get("class") if isinstance(d, dict) else None] += 1

        sa, sb = sources
        if sa not in gold.utt or sb not in gold.utt:
            ignored["unknown_utterance"] += 1
            iss.add(WARN, "unknown_utterance_id", "record references a source_utterance_id not in the key (ignored)",
                    {"record": rid, "source_utterance_ids": [sa, sb]})
            continue
        if sa == sb:
            ignored["same_utterance"] += 1
            iss.add(WARN, "same_utterance_record", "record compares entries of one utterance (ignored)", rid)
            continue
        pa, pb = gold.utt[sa][0], gold.utt[sb][0]
        if pa != pb:
            ignored["cross_episode"] += 1
            iss.add(WARN, "cross_episode_record", "record links utterances of different eval_pair_ids (ignored)",
                    {"record": rid, "pairs": [pa, pb]})
            continue
        pair_records[pa].append(rec)
        link = (pa, frozenset((ea.get("entry_id"), eb.get("entry_id"))))
        seen_links[link] += 1
        if seen_links[link] == 2:
            iss.add(WARN, "duplicate_comparison", "same entry pair compared in more than one record", rid)
        scored_has_key.append("compatibility" in rec)
        scored_compat.append(rec.get("compatibility"))

    for r, n in seen_record_ids.items():
        if n > 1:
            iss.add(ERROR, "duplicate_record_id", "record_id repeated", r)
    if len([r for r in run_ids if r is not None]) > 1:
        iss.add(ERROR, "multiple_run_ids", "records carry more than one run_id", dict(run_ids))

    # compatibility consistency on scored (cross-utterance) records
    if scored_has_key and not all(scored_has_key):
        iss.add(WARN, "compatibility_field_missing", "some scored records lack the compatibility field",
                f"{scored_has_key.count(False)} records")
    nulls = sum(c is None for c in scored_compat)
    if scored_compat and 0 < nulls < len(scored_compat):
        iss.add(WARN, "compatibility_mixed_null", "run mixes null and non-null compatibility",
                f"{nulls} null of {len(scored_compat)}")
    if m_strat in EXPECTED_DKMEM_MODE and scored_compat and nulls == len(scored_compat):
        iss.add(WARN, "compatibility_all_null_for_dkmem", "DK-Mem run has compatibility null on every scored record")

    # coverage vs gold: structural outcome only (normal model outcomes -> INFO)
    pair_status: dict[str, str] = {}
    for pid, g in gold.pairs.items():
        has_a = bool(entries_by_utt.get(g["utterance_a_id"]))
        has_b = bool(entries_by_utt.get(g["utterance_b_id"]))
        if pair_records.get(pid):
            st = "compared"
        elif not has_a and not has_b:
            st = "no_entry_both"
        elif not has_a:
            st = "no_entry_a"
        elif not has_b:
            st = "no_entry_b"
        else:
            st = "not_compared"
        pair_status[pid] = st
        if st.startswith("no_entry"):
            iss.add(INFO, "pair_no_entry",
                    "pair has an utterance with zero entries (model outcome; the cause, e.g. extraction failure, "
                    "is not recorded in the output)", pid)
        elif st == "not_compared":
            iss.add(INFO, "pair_not_compared",
                    "both utterances have entries but no cross-utterance record exists (model outcome)", pid)
    for pid, recs in pair_records.items():
        if len(recs) > 1:
            iss.add(INFO, "pair_multiple_cross_records",
                    "pair has more than one cross-utterance record (multi-entry utterances)", pid)

    classes = {pid: g["distinction_class"] for pid, g in gold.pairs.items()}
    counts = {
        "pairwise_records": len(data.records),
        "gold_pairs": len(gold.pairs),
        "pairs_with_cross_record": len(pair_records),
        "pair_status": _count(pair_status.values()),
        "pair_status_by_class_relation": _count(
            f"{pair_status[p]}|{classes[p]}|{gold.pairs[p]['relation']}" for p in gold.pairs),
        "ignored_records": dict(ignored),
        "schema_invalid_records": n_schema_bad,
        "decisions": dict(decisions),
        "compatibility": {str(k): v for k, v in compat.items()},
        "entry_distinction_classes": {str(k): v for k, v in entry_classes.items()},
        "gold_class_x_relation": gold.distributions["class_x_relation"],
        "gold_language_x_relation": gold.distributions["language_x_relation"],
    }
    return RunReport(str(run_dir), m_run, m_strat, manifest, iss.list(), dict(pair_records), pair_status, counts)


# --------------------------------------------------------------------------- multi-seed (warn-only)

def check_seed_group(reports: list[RunReport], min_seeds: int = 3) -> list[Issue]:
    """Warn-only consistency checks across the runs of ONE strategy. Never ERROR."""
    iss = _Issues()
    mans = [r.manifest for r in reports if r.manifest]
    strategies = {m.get("strategy") for m in mans}
    if len(strategies) > 1:
        iss.add(WARN, "seedgroup_mixed_strategies", "runs in the group have different strategies",
                sorted(map(str, strategies)))
    if len({m.get("seed") for m in mans}) < min_seeds:
        iss.add(WARN, "seedgroup_fewer_than_min_seeds", f"fewer than {min_seeds} distinct seeds",
                f"{len({m.get('seed') for m in mans})} distinct")
    for k, code in (("seed", "seedgroup_duplicate_seed"), ("run_id", "seedgroup_duplicate_run_id")):
        for v, n in Counter(m.get(k) for m in mans).items():
            if n > 1:
                iss.add(WARN, code, f"duplicate {k} within group", v)
    keys = set().union(*(m.keys() for m in mans)) if mans else set()
    for k in sorted(keys - SEED_GROUP_IGNORED_KEYS):
        if not all(k in m for m in mans):
            iss.add(WARN, "seedgroup_manifest_key_in_some_runs", "manifest key present in only some runs", k)
        elif len({json.dumps(m[k], sort_keys=True) for m in mans}) > 1:
            iss.add(WARN, "seedgroup_manifest_value_differs", "manifest value differs between runs", k)

    def signature(r: RunReport) -> tuple:
        return tuple(sorted(
            (pid, tuple(sorted((x["entry_a"]["entry_id"], x["entry_b"]["entry_id"], x["decision"],
                                x["similarity_score"]) for x in recs)))
            for pid, recs in r.pair_records.items()))

    if len(reports) > 1:
        if len({signature(r) for r in reports}) == 1:
            iss.add(INFO, "seedgroup_identical_outputs",
                    "all runs have identical pair decisions and scores (seed has no effect; sd across seeds will be 0)")
        else:
            iss.add(INFO, "seedgroup_outputs_differ", "pair decisions/scores differ between seeds")
    return iss.list()
