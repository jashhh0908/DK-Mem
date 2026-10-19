"""Tier-2 multi-session SYSTEM runner (configs A, B, C and D, DK-Mem gate off or lexicon).

The system under test receives ONLY tier2_conversations.jsonl (turns, no labels) and, after the last session, tier2_qa_questions.jsonl, plus an opaque
turn-id -> entity-id passthrough file (ids only; injected and distractor turns have indistinguishable ids). It never reads the gold, injection metadata or
QA references (a unit test checks that this module names none of them).

Configs A and B store an LLM extraction of each turn instead of its surface (A: English gloss, B: native-language fact; see
entry_extraction.py). A turn whose strict extraction fails has no entry; every turn is logged in <run_id>.extraction_log.jsonl. The judge of
A/B/C sees the STORED texts (gloss for A and B, surface for C); candidate ranking is by bge-m3 cosine of the SURFACES for every config.

Pipeline per conversation (stores are never shared between conversations):
  for each session in order, for each USER turn in order:
    1. extraction: verbatim storage of the surface text (configs C, D); with the gate on, the lexicon distinction (no LLM)
    2. candidate retrieval: rank every earlier raw entry of this conversation by bge-m3 cosine, keep the top-k (k=5)
    3. consolidation: for each (candidate, new entry) pair the host decides (C: a frozen merge judge chosen with --judge-prompt: the Tier-1
       merge_judge_v1 from dkmem.memory.judge, or the Tier-2 identity judge merge_judge_entity_t2_v1; D: cosine > tau, strict), then, if the gate is on,
       `apply_gate` can only veto; every compared pair is written as ONE pairwise record; a merge joins the entries into one memory record
    4. the new entry is added to the persistent store
  after the last session: every QA question retrieves the top-k MEMORY RECORDS by bge-m3 cosine and a fixed answerer prompt (same model for every config)
  answers from them.
Outputs per run: <run_id>/{run_manifest.json,pairwise_eval.jsonl} (Tier-1 schema, dataset_tier tier2_injected_sessions) and next to it
<run_id>.tier2_retrieval_log.jsonl, .tier2_store_final.jsonl, .tier2_qa_answers.jsonl, (C) .judge_log.jsonl, .tier2_run_info.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from harness import _dkmem  # noqa: F401  (puts ipdV2 on sys.path; read-only)
from dkmem.backends.llm import GenerationParams
from dkmem.memory.gate import apply_gate
from dkmem.memory.judge import MERGE_JUDGE_V1
from dkmem.memory.lexicon import load_lexicon
from dkmem.pipeline.trace import sha256_file_lf
from dkmem.tier1.io import DECISION_TRANSLATION, MemoryEntry, translate_distinction

from harness.extractors import extract_verbatim
from harness.hosts import EmbeddingThresholdHost
from harness.writer import build_manifest, write_run

from .entry_extraction import EXTRACTION_CONFIGS, TurnExtractor
from .judge_host import FROZEN_JUDGE_MAX_NEW_TOKENS, JUDGES_BY_ID, FrozenJudgeHost, TokenMeter, check_frozen_prompt, summarize_counts
from .memory_store import PersistentStore, StoredEntry

HERE = Path(__file__).resolve().parent

# Strategy names (same values as the Tier-1 harness' episode.py, which is not vendored: it imports the superseded harness
# judge). Gate on: the DK-Mem mode names the strategy; gate off: the config does.
GATE_ON_STRATEGY = {"lexicon": "dk-mem-lexicon", "lexicon+llm": "dk-mem-lexicon-llm"}
GATE_OFF_STRATEGY = {"A": "mem0", "B": "mem0", "C": "store-surface-only", "D": "embedding-threshold"}
DATA_DIR = HERE / "data"
FROZEN_FILE = HERE / "FROZEN_v1.0.json"     # the original benchmark; the mirror arm passes --frozen mirror_v1_1/FROZEN_v1.1m.json
TOP_K = 5
JUDGE_MAX_NEW_TOKENS = FROZEN_JUDGE_MAX_NEW_TOKENS   # Tier-1's judge budget; the frozen prompt answers with a reason sentence
QA_MAX_NEW_TOKENS = 48
EXTRACTION_MAX_NEW_TOKENS = 256                      # as the Tier-1 A-D runs (generation_params.max_new_tokens)
JUDGE_CONFIGS = ("A", "B", "C")                     # configs whose host is an LLM judge
QA_PROMPT_ID = "tier2_qa_v1"
QA_SYSTEM = ("You answer questions about the user using only the stored memories listed below. Reply with a short answer only (a few words), in the "
             "language of the question. If the memories do not contain the answer, reply \"unknown\".")
QA_USER_TEMPLATE = "Memories:\n{memories}\n\nQuestion: {question}"
STORE_SEMANTICS = ("non-destructive: every raw entry is kept; a merge decision joins entries into one memory record that lists all member surface texts "
                   "(write order, ' | ' separated) and all member source turn ids; candidate ranking uses raw entries only, so compared pairs do not depend on "
                   "earlier merge decisions")


_STAGE = {"name": "startup", "t0": time.time()}


def stage(name: str) -> None:
    """Flushed progress marker; remembers the current stage so a failure can name it."""
    _STAGE["name"] = name
    print(f"[tier2 +{time.time() - _STAGE['t0']:7.1f}s] STAGE: {name}", flush=True)


# The files the system under test genuinely reads. Every other file the frozen manifest pins (gold, injection metadata, QA
# references, the pair specs, the distractor bank, the generator and validator) is evaluator-side: it is verified when it is
# present (a local checkout) and may be absent from a Kaggle payload, which must never carry it.
SYSTEM_FROZEN_FILES = ("data/tier2_conversations.jsonl", "data/tier2_qa_questions.jsonl")


def verify_frozen(data_dir: Path, frozen_file: Path = FROZEN_FILE) -> dict:
    """Every frozen Tier-2 file that is present must still match the sha256 in ``frozen_file`` (raises otherwise). ``data/...`` pins
    resolve against the data directory's parent, every other pin against the frozen file's own directory.

    A file of ``SYSTEM_FROZEN_FILES`` must be present. An evaluator-side file may be absent (the smoke payload excludes
    them); the result lists which were verified and which were not in the payload, so the run manifest can say so.
    """
    frozen_file = Path(frozen_file)
    fz = json.loads(frozen_file.read_text(encoding="utf-8"))
    bad, verified, absent = [], [], []
    for rel, want in fz["sha256"].items():
        p = (data_dir.parent / rel) if rel.startswith("data/") else (frozen_file.parent / rel)
        if not p.is_file():
            if rel in SYSTEM_FROZEN_FILES:
                bad.append(f"{rel} (missing)")
            else:
                absent.append(rel)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            bad.append(rel)
        else:
            verified.append(rel)
    if bad:
        raise RuntimeError(f"frozen Tier-2 files changed: {bad}")
    if not all(r in verified for r in SYSTEM_FROZEN_FILES):
        raise RuntimeError("the system-side Tier-2 inputs were not verified")
    return {**fz, "verified_files": sorted(verified), "not_in_payload_evaluator_only": sorted(absent)}


def qa_prompt_sha256() -> str:
    return hashlib.sha256("\x00".join([QA_PROMPT_ID, QA_SYSTEM, QA_USER_TEMPLATE]).encode("utf-8")).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class Setup:
    config: str                    # "C" | "D"
    mode: str                      # "off" | "lexicon"
    host: object                   # FrozenJudgeHost (C: a frozen merge judge) | EmbeddingThresholdHost (D)
    metric: object                 # bge-m3 CosineMetric (or a test metric with the same interface)
    generator: object              # TextGenerator used by the C judge and by the QA answerer
    lexicon: object | None
    passthrough: dict[str, str]
    seed: int = 0
    k: int = TOP_K
    tau: float | None = None
    qa_params: GenerationParams | None = None
    progress: object | None = None          # optional callable(str) for stage markers
    meter: object | None = None             # TokenMeter over the generator's own tokenizer (None for test doubles)
    extractor: object | None = None         # entry_extraction.TurnExtractor (configs A and B only)
    lexicon_sha256: str | None = None       # LF-normalised sha256 of the lexicon file the gate used (recorded in the manifest when given)

    @property
    def strategy(self) -> str:
        return GATE_ON_STRATEGY.get(self.mode) or GATE_OFF_STRATEGY[self.config]

    def __post_init__(self) -> None:
        if self.config not in ("A", "B", "C", "D") or self.mode not in ("off", "lexicon"):
            raise NotImplementedError(f"Tier 2 supports configs A-D with mode off/lexicon, got {self.config}/{self.mode}")
        if (self.mode == "lexicon") != (self.lexicon is not None):
            raise ValueError("mode 'lexicon' needs a lexicon; mode 'off' must not have one")
        if self.config in JUDGE_CONFIGS:
            if not isinstance(self.host, FrozenJudgeHost):
                raise TypeError(f"config {self.config} needs a FrozenJudgeHost with a frozen merge judge")
            check_frozen_prompt(self.host.prompt)
            want_field = "surface" if self.config == "C" else "gloss"
            if self.host.text_field != want_field:
                raise ValueError(f"config {self.config}'s judge must see the stored text ({want_field!r}), not {self.host.text_field!r}")
        if self.config in EXTRACTION_CONFIGS:
            if not isinstance(self.extractor, TurnExtractor) or self.extractor.config != self.config:
                raise TypeError(f"config {self.config} needs a TurnExtractor for config {self.config}")
            if (self.extractor.lexicon is not None) != (self.mode == "lexicon"):
                raise ValueError("the extractor must carry the lexicon exactly when the gate is on")
        elif self.extractor is not None:
            raise ValueError(f"config {self.config} stores the surface text and takes no extractor")
        if self.config == "D" and self.tau is None:
            raise ValueError("config D needs an explicit tau (a recorded run-level value, not an operating point)")


def _entry_dict(turn: dict, ext, setup: Setup) -> dict:
    method = setup.mode if setup.mode != "off" else None
    uid = turn["turn_id"]
    return MemoryEntry(entry_id=f"{uid}_e0", source_utterance_id=uid, gold_entity_id=setup.passthrough[uid], language=turn["language"],
                       gloss=ext.gloss, surface=ext.surface, distinction=translate_distinction(ext.distinction, extraction_method=method)).to_dict()


def run_conversation(conv: dict, questions: list[dict], setup: Setup, run_id: str) -> dict:
    cid = conv["conversation_id"]
    store = PersistentStore(cid)
    rows, rlog, jrows = [], [], []
    lex = setup.lexicon
    P = setup.progress or (lambda *_: None)
    n_merges = 0
    verbatim = setup.config in ("C", "D")
    extractions = {}
    if not verbatim:       # one batched, strict extraction of the conversation's user turns (independent of the store)
        extractions = setup.extractor.extract_turns([{"turn_id": t["turn_id"], "text": t["text"], "language": t["language"], "session_index": s["session_index"]}
                                                      for s in conv["sessions"] for t in s["turns"] if t["speaker"] == "user"], cid)
        P(f"{cid}: extraction complete ({sum(not r.ok for r in extractions.values())} of {len(extractions)} turns failed)")
    for sess in conv["sessions"]:
        P(f"{cid}: session {sess['session_index']} begins")
        for turn in sess["turns"]:
            if turn["speaker"] != "user":
                continue                                           # assistant turns are not memory sources
            if verbatim:
                ext, stored = extract_verbatim(turn["turn_id"], "a", turn["text"], turn["language"], seed=setup.seed, lexicon=lex), None
            else:
                te = extractions[turn["turn_id"]]
                if not te.ok:                                          # no entry: nothing to rank, compare or store (never a decision)
                    rlog.append({"conversation_id": cid, "turn_id": turn["turn_id"], "session_index": sess["session_index"], "entry_id": None,
                                 "status": "extraction_failed", "store_size_before": len(store.entries), "top_k": setup.k, "ranked": [],
                                 "candidates": [], "comparisons": []})
                    P(f"{cid}: {turn['turn_id']} extraction failed: no entry")
                    continue
                ext, stored = te.extraction, te.stored_text
            new = StoredEntry(entry_id=f"{turn['turn_id']}_e0", turn_id=turn["turn_id"], session_index=sess["session_index"], surface=ext.surface,
                              language=turn["language"], extraction=ext, entry_dict=_entry_dict(turn, ext, setup), text=stored)
            ranked = store.rank(turn["text"], setup.metric)          # ranks EARLIER entries only; the new entry is added right after
            store.add(new)
            top = ranked[:setup.k]
            log = {"conversation_id": cid, "turn_id": turn["turn_id"], "session_index": sess["session_index"], "entry_id": new.entry_id,
                   "store_size_before": len(ranked), "top_k": setup.k,
                   "ranked": [{"rank": i + 1, "entry_id": e.entry_id, "turn_id": e.turn_id, "similarity": s} for i, (e, s) in enumerate(ranked)],
                   "candidates": [e.entry_id for e, _ in top], "comparisons": []}
            if not verbatim:
                log["stored_text"] = new.text
            if top:
                outcomes = setup.host.decide_many([(e.extraction, ext) for e, _ in top])
                if setup.config in JUDGE_CONFIGS:                   # tag the judge-log rows of this call with their pair
                    for (e, _), jr in zip(top, setup.host.log[-len(top):]):
                        jr.update(conversation_id=cid, record_id=f"{e.entry_id}::{new.entry_id}", turn_id_a=e.turn_id, turn_id_b=new.turn_id)
                        jrows.append(jr)
                for (cand, _), out in zip(top, outcomes):
                    rid = f"{cand.entry_id}::{new.entry_id}"
                    if out.proposal is None:
                        log["comparisons"].append({"record_id": rid, "candidate_entry_id": cand.entry_id, "status": "judge_parse_error", "decision": None})
                        continue
                    if setup.mode == "lexicon":
                        g = apply_gate(out.proposal, cand.extraction.distinction, ext.distinction)
                        decision, compat = g.decision, g.compatibility
                    else:
                        decision, compat = out.proposal, None
                    external = DECISION_TRANSLATION[decision]
                    rows.append({"record_id": rid, "run_id": run_id, "strategy": setup.strategy, "entry_a": cand.entry_dict, "entry_b": new.entry_dict,
                                 "decision": external, "similarity_score": out.similarity, "threshold": out.threshold, "compatibility": compat,
                                 "predicted_entity_id": cand.entry_id if external in ("merge", "supersede") else None, "superseded_entry_id": None})
                    comp = {"record_id": rid, "candidate_entry_id": cand.entry_id, "status": "record", "decision": external}
                    if not verbatim:
                        comp["record_similarity"] = out.similarity      # cosine of the two STORED texts (the ranking score is of the surfaces)
                    log["comparisons"].append(comp)
                    if external in ("merge", "supersede"):
                        store.merge(cand.entry_id, new.entry_id)
                        n_merges += 1
            rlog.append(log)
            P(f"{cid}: {turn['turn_id']} stored (ranked {len(ranked)} earlier entries, compared {len(top)})")
        P(f"{cid}: session {sess['session_index']} extraction complete")
        P(f"{cid}: session {sess['session_index']} retrieval/consolidation complete (entries {len(store.entries)}, merges so far {n_merges}, pairwise records so far {len(rows)})")
    records = store.records()
    P(f"{cid}: final memory state created ({len(records)} memory records from {len(store.entries)} entries)")
    P(f"{cid}: QA begins")
    qa_rows = answer_questions(cid, questions, records, setup)
    P(f"{cid}: QA completes ({len(qa_rows)} answers)")
    return {"rows": rows, "retrieval_log": rlog, "judge_rows": jrows, "store": [{"conversation_id": cid, **r} for r in records], "qa": qa_rows}


class LazyQuestions:
    """The QA questions of a conversation, read from disk on first use.

    The question file's ids and per-conversation counts structurally separate same pairs (one combined question) from
    different pairs (two); ``run_conversation`` calls this only after the conversation's final memory state exists, so no
    consolidation decision can depend on it. ``loaded`` stays False until the first call.
    """

    def __init__(self, path: Path) -> None:
        self.path, self.loaded, self._rows = Path(path), False, None

    def __call__(self, conversation_id: str) -> list[dict]:
        if self._rows is None:
            self._rows = read_jsonl(self.path)
            self.loaded = True
        return [q for q in self._rows if q["conversation_id"] == conversation_id]


def answer_questions(cid: str, questions, records: list[dict], setup: Setup) -> list[dict]:
    """``questions``: a list of question rows, or a callable ``conversation_id -> rows`` (``LazyQuestions``)."""
    qs = questions(cid) if callable(questions) else [q for q in questions if q["conversation_id"] == cid]
    if not qs:
        return []
    setup.metric.prepare([q["question"] for q in qs] + [r["text"] for r in records])
    prompts, retrieved = [], []
    for q in qs:
        ranked = sorted(((r, setup.metric(r["text"], q["question"])) for r in records), key=lambda p: (-p[1], p[0]["member_entry_ids"][0]))[:setup.k]
        retrieved.append([{"rank": i + 1, "memory_id": r["memory_id"], "member_turn_ids": r["member_turn_ids"], "similarity": s, "text": r["text"]}
                          for i, (r, s) in enumerate(ranked)])
        mem = "\n".join(f"- {x['text']}" for x in retrieved[-1])
        prompts.append([{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": QA_USER_TEMPLATE.format(memories=mem, question=q["question"])}])
    answers = setup.generator.generate(prompts, setup.qa_params)
    if len(answers) != len(qs):
        raise RuntimeError("answerer returned the wrong number of outputs")
    meter = setup.meter
    return [{"qa_id": q["qa_id"], "conversation_id": cid, "question": q["question"], "language": q.get("language"), "prompt_id": QA_PROMPT_ID,
             "retrieved": ret, "raw_answer": a, "status": "ok" if isinstance(a, str) and a.strip() else "empty_answer",
             "output_tokens": meter.text_tokens(a) if (meter and isinstance(a, str)) else None}
            for q, ret, a in zip(qs, retrieved, answers)]


def run_setup(setup: Setup, convs: list[dict], questions, *, run_id: str, out: Path, meta: dict) -> dict:
    results = [run_conversation(c, questions, setup, run_id) for c in convs]
    rows = [r for res in results for r in res["rows"]]
    rlog = [r for res in results for r in res["retrieval_log"]]
    qa = [r for res in results for r in res["qa"]]
    store = [r for res in results for r in res["store"]]
    jrows = [r for res in results for r in res["judge_rows"]]
    extra = {"merge_mechanism": "llm_judge" if setup.config == "C" else "embedding_threshold", "tau": setup.tau, "tau_note": ("pilot/execution threshold, NOT an operating point and not claimed optimal; the raw bge-m3 cosine of every compared pair is stored" if setup.tau is not None else None), "similarity_note": "similarity_score is the raw bge-m3 cosine of the two surface texts",
             "tier2": {**meta, "pilot_scope": "configs C and D, gate off/lexicon", "shared_candidate_retrieval": {"embedder": "BAAI/bge-m3", "top_k": setup.k, "ranked_over": "all earlier raw entries of the conversation"},
                       "store_semantics": STORE_SEMANTICS, "qa": {"prompt_id": QA_PROMPT_ID, "prompt_sha256": qa_prompt_sha256(), "top_k_memory_records": setup.k, "max_new_tokens": QA_MAX_NEW_TOKENS,
                                                                  "answerer": getattr(setup.generator, "backbone", None)},
                       "conversations": [c["conversation_id"] for c in convs]},
             **setup.metric.describe()}
    sims = sorted(r["similarity_score"] for r in rows)
    pct = lambda f: sims[min(len(sims) - 1, int(round(f * (len(sims) - 1))))] if sims else None
    extra["tier2"]["decision_distribution"] = {
        "pairwise_decisions": dict(Counter(r["decision"] for r in rows)),
        "compatibility": {str(k): v for k, v in Counter(r["compatibility"] for r in rows).items()},
        "similarity_score": {"n": len(sims), "min": sims[0] if sims else None, "median": pct(0.5), "p95": pct(0.95), "max": sims[-1] if sims else None},
        "pairs_above_tau": sum(r["similarity_score"] > setup.tau for r in rows) if setup.tau is not None else None}
    extra["tier2"]["qa"].update(
        questions_read="after each conversation's final memory state (LazyQuestions)" if callable(questions) else "before the run (list)",
        output_tokens=summarize_counts([a["output_tokens"] for a in qa if a.get("output_tokens") is not None]))
    if setup.meter is not None:
        extra["tier2"]["tokenizer"] = setup.meter.describe()
    if setup.config in JUDGE_CONFIGS:
        # the decoding settings the judge host ACTUALLY used (what the manifest and the budget fields must describe), not a constant
        used_params = setup.host.params or GenerationParams(max_new_tokens=JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=setup.seed, batch_size=8)
        proposals = Counter(str(j["proposal"]) for j in jrows)
        extra.update(judge_decision_distribution={
            "proposals": dict(proposals), "raw_decisions": dict(Counter(_raw_decision(j["raw_output"]) for j in jrows)),
            "n_calls": len(jrows), "n_parsed": sum(1 for j in jrows if j["status"] == "ok"),
            "constant": len([k for k in proposals if k != "None"]) == 1 and "None" not in proposals},
            judge_tokenization={
                "prompt_tokens": summarize_counts([j["prompt_tokens"] for j in jrows if j.get("prompt_tokens") is not None]),
                "output_tokens": summarize_counts([j["output_tokens"] for j in jrows if j.get("output_tokens") is not None]),
                "n_at_budget": sum(1 for j in jrows if j.get("at_budget")), "budget": used_params.max_new_tokens,
                "measured": all(j.get("output_tokens") is not None for j in jrows) and bool(jrows)})
        judge_gen = setup.host.generator
        extra.update(judge_prompt_id=setup.host.prompt.prompt_id, judge_prompt_sha256=setup.host.prompt.sha256, judge_text_field="surface",
                     judge_calls=len(jrows), judge_parse_failures=sum(1 for j in jrows if j["status"] != "ok"),
                     judge_possibly_truncated=sum(1 for j in jrows if j.get("possibly_truncated")),
                     # the JUDGE model's provenance (it may differ from the QA answerer's: --judge-model-id)
                     generation_run_info=judge_gen.run_info(used_params),
                     judge_backbone=getattr(judge_gen, "backbone", None), judge_model_separate_from_qa=judge_gen is not setup.generator,
                     judge_device_map=_device_map(judge_gen),
                     judge_tokenizer=setup.host.meter.describe() if setup.host.meter is not None else None)
    if setup.config in EXTRACTION_CONFIGS:
        xlog = setup.extractor.log
        extra.update(extraction_prompt_id=setup.extractor.prompt.prompt_id, extraction_prompt_sha256=setup.extractor.prompt.sha256,
                     extraction_calls=len(xlog), extraction_failures=sum(1 for r in xlog if r["status"] != "ok"),
                     extraction_backbone=getattr(setup.generator, "backbone", None), extraction_max_new_tokens=setup.extractor.params.max_new_tokens)
    if setup.lexicon_sha256 is not None:
        extra["lexicon_sha256"] = setup.lexicon_sha256
    extra["tier2"]["qa"]["generation_run_info"] = setup.generator.run_info(setup.qa_params)
    # backbone: the model that wrote the stored texts and decided merges. A/B: the extractor (the judge's own backbone is judge_backbone);
    # C: the judge; D: none
    backbone = (getattr(setup.generator, "backbone", None) if setup.config in EXTRACTION_CONFIGS
                else getattr(setup.host.generator, "backbone", None) if setup.config == "C" else None)
    manifest = build_manifest(run_id=run_id, strategy=setup.strategy, seed=setup.seed, pipeline_config=setup.config, dkmem_mode=setup.mode,
                              backbone=backbone, extra=extra)
    manifest["dataset_tier"] = "tier2_injected_sessions"
    write_run(out / run_id, manifest, rows)
    (setup.progress or (lambda *_: None))(f"{run_id}: pairwise output written ({len(rows)} records)")
    write_jsonl(out / f"{run_id}.tier2_retrieval_log.jsonl", rlog)
    write_jsonl(out / f"{run_id}.tier2_store_final.jsonl", store)
    write_jsonl(out / f"{run_id}.tier2_qa_answers.jsonl", qa)
    if setup.config in JUDGE_CONFIGS:
        write_jsonl(out / f"{run_id}.judge_log.jsonl", jrows)
    if setup.config in EXTRACTION_CONFIGS:
        write_jsonl(out / f"{run_id}.extraction_log.jsonl", setup.extractor.log)
    info = {"run_id": run_id, "conversations": len(convs), "pairwise_records": len(rows), "decisions": dict(Counter(r["decision"] for r in rows)),
            "judge_parse_errors": sum(1 for r in rlog for c in r["comparisons"] if c["status"] == "judge_parse_error"), "memory_records": len(store), "qa_answers": len(qa),
            "questions_loaded_lazily": bool(getattr(questions, "loaded", False))}
    if setup.config in EXTRACTION_CONFIGS:
        info["extraction_failures"] = sum(1 for r in setup.extractor.log if r["status"] != "ok")
    (out / f"{run_id}.tier2_run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def _device_map(generator):
    """Layer -> device placement of a sharded model (None for a single-device model or a test double)."""
    m = getattr(getattr(generator, "model", None), "hf_device_map", None)
    return {str(k): str(v) for k, v in m.items()} if isinstance(m, dict) else None


def _raw_decision(raw) -> str:
    """The ``decision`` value of a raw judge answer, or ``unparsed`` (for the recorded distribution only; never a decision)."""
    try:
        d = json.loads(raw).get("decision")
        return d if isinstance(d, str) else "unparsed"
    except (ValueError, AttributeError, TypeError):
        return "unparsed"


def load_conversations(data_dir: Path, conversation_ids: list[str] | None) -> list[dict]:
    convs = read_jsonl(data_dir / "tier2_conversations.jsonl")
    if conversation_ids:
        by = {c["conversation_id"]: c for c in convs}
        missing = [c for c in conversation_ids if c not in by]
        if missing:
            raise SystemExit(f"unknown conversation ids: {missing}")
        convs = [by[c] for c in conversation_ids]
    return convs


def load_inputs(data_dir: Path, conversation_ids: list[str] | None) -> tuple[list[dict], list[dict]]:
    convs = read_jsonl(data_dir / "tier2_conversations.jsonl")
    if conversation_ids:
        by = {c["conversation_id"]: c for c in convs}
        missing = [c for c in conversation_ids if c not in by]
        if missing:
            raise SystemExit(f"unknown conversation ids: {missing}")
        convs = [by[c] for c in conversation_ids]
    questions = read_jsonl(data_dir / "tier2_qa_questions.jsonl")
    return convs, questions


def build_models(args):
    """Real backends (lazy imports; patched in tests): bge-m3 metric and the fixed QA answerer LLM (which is also the config C judge
    unless --judge-model-id names a separate judge model, see ``build_judge_model``)."""
    from dkmem.backends.llm import HFGenerator, ModelConfig
    from harness.similarity import build_metric
    stage("model/tokenizer loading begins (bge-m3 metric is lazy; Qwen loads now)")
    metric = build_metric(revision=args.similarity_revision, dtype="float16", device="cuda:0", batch_size=32, max_length=512)
    gen = HFGenerator.load(ModelConfig(model_id=args.model_id, revision=args.model_revision, device=getattr(args, "qa_device", "cuda:0")))
    stage(f"model/tokenizer loading complete ({gen.backbone}, revision {getattr(getattr(gen.model, 'config', None), '_commit_hash', None)})")
    return metric, gen


def parse_max_memory(spec):
    """``"0=5GiB,1=11GiB"`` -> ``{0: "5GiB", 1: "11GiB"}``; None or empty -> None (the loader's own default)."""
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        dev, _, cap = part.partition("=")
        if not dev.strip().isdigit() or not cap.strip():
            raise ValueError(f"--judge-max-memory must look like 0=5GiB,1=11GiB, got {spec!r}")
        out[int(dev)] = cap.strip()
    return out


def build_judge_model(args, qa_generator):
    """The config C judge model: the QA answerer itself (no --judge-model-id: the historical behaviour) or a separately loaded, pinned
    model (lazy imports; patched in tests). With --judge-shard its weights are split over the visible GPUs with the given caps."""
    if not args.judge_model_id:
        return qa_generator
    from dkmem.backends.llm import HFGenerator, ModelConfig
    stage(f"judge model loading begins ({args.judge_model_id}@{args.judge_model_revision}, shard={args.judge_shard})")
    cfg = ModelConfig(model_id=args.judge_model_id, revision=args.judge_model_revision, device=args.judge_device)
    judge = HFGenerator.load_sharded(cfg, max_memory=parse_max_memory(args.judge_max_memory)) if args.judge_shard else HFGenerator.load(cfg)
    stage(f"judge model loading complete ({judge.backbone}, revision {getattr(getattr(judge.model, 'config', None), '_commit_hash', None)}, "
          f"device {judge.device_name})")
    return judge


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--progress", action="store_true", help="print flushed stage markers")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--frozen", type=Path, default=FROZEN_FILE, help="the FROZEN json of the benchmark arm the data directory belongs to")
    ap.add_argument("--judge-prompt", choices=sorted(JUDGES_BY_ID), default=MERGE_JUDGE_V1.prompt_id,
                    help="judge of configs A, B and C (frozen ids only); the Tier-2 identity judge is merge_judge_entity_t2_v1")
    ap.add_argument("--lexicon", type=Path, default=None, help="the gate's lexicon file (default: the production lexicon); its sha256 is recorded")
    ap.add_argument("--passthrough", type=Path, required=True, help="JSONL of {turn_id, gold_entity_id} (opaque ids only)")
    ap.add_argument("--conversations", help="comma-separated conversation ids (default: all)")
    ap.add_argument("--configs", default="C:off,C:lexicon,D:off,D:lexicon", help="comma list of CONFIG:MODE, config A-D, mode off|lexicon")
    ap.add_argument("--tau", type=float, default=0.90, help="config D run-level threshold (pilot value; NOT an operating point; scores are stored raw)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-prefix", default="t2pilot")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--model-revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--similarity-revision", default="5617a9f61b028005a4858fdac845db406aefb181")
    ap.add_argument("--qa-device", default="cuda:0", help="device of the QA answerer (single-device load)")
    ap.add_argument("--judge-model-id", default=None, help="config C judge model; default: the QA answerer (--model-id) itself")
    ap.add_argument("--judge-model-revision", default=None, help="pinned revision of --judge-model-id (required with it)")
    ap.add_argument("--judge-shard", action="store_true", help="split the judge model over the visible GPUs (HFGenerator.load_sharded)")
    ap.add_argument("--judge-max-memory", default=None, help="per-GPU weight caps for --judge-shard, e.g. 0=5GiB,1=11GiB")
    ap.add_argument("--judge-device", default="cuda:0", help="device of an unsharded judge model")
    a = ap.parse_args(argv)
    try:
        return _main(a)
    except BaseException as e:                     # name the failing stage, keep a failure artifact, then fail loudly
        tb = traceback.format_exc()
        print(f"[tier2] FAILURE in stage {_STAGE['name']!r}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        print(f"[tier2] FAILURE in stage {_STAGE['name']!r}: {type(e).__name__}: {e}", flush=True)
        try:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "TIER2_FAILURE.json").write_text(json.dumps({"stage": _STAGE['name'], "exception_type": type(e).__name__, "message": str(e), "traceback": tb}, indent=1), encoding="utf-8")
        except OSError:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return 1


def _main(a) -> int:
    stage("process startup")
    has_c = any(spec.split(":")[0] in JUDGE_CONFIGS for spec in a.configs.split(","))       # a run with an LLM judge (A, B or C)
    if a.judge_model_id and not has_c:
        raise SystemExit("--judge-model-id is only meaningful with a config A, B or C run")
    if a.judge_model_id and not a.judge_model_revision:
        raise SystemExit("--judge-model-id needs a pinned --judge-model-revision")
    if (a.judge_shard or a.judge_max_memory) and not a.judge_model_id:
        raise SystemExit("--judge-shard/--judge-max-memory need --judge-model-id")
    parse_max_memory(a.judge_max_memory)               # fail before any model is loaded
    verified = verify_frozen(a.data_dir, a.frozen)
    stage("frozen Tier-2 manifest/hash verification passed")
    convs = load_conversations(a.data_dir, a.conversations.split(",") if a.conversations else None)
    questions = LazyQuestions(a.data_dir / "tier2_qa_questions.jsonl")   # read only after a conversation's final memory state
    stage(f"frozen Tier-2 data loaded ({len(convs)} conversations; questions are read after consolidation)")
    passthrough = {r["turn_id"]: r["gold_entity_id"] for r in read_jsonl(a.passthrough)}
    metric, gen = build_models(a)
    meter = TokenMeter.from_generator(gen)                                   # the QA answerer's tokenizer
    judge_gen = build_judge_model(a, gen) if has_c else gen
    judge_meter = meter if judge_gen is gen else TokenMeter.from_generator(judge_gen)   # the judge's own tokenizer
    lexicon_path = a.lexicon or _dkmem.LEXICON_PATH
    lexicon = load_lexicon(lexicon_path)
    lexicon_sha = sha256_file_lf(lexicon_path)
    qa_params = GenerationParams(max_new_tokens=QA_MAX_NEW_TOKENS, do_sample=False, seed=a.seed, batch_size=8)
    judge_params = GenerationParams(max_new_tokens=JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=a.seed, batch_size=8)
    extraction_params = GenerationParams(max_new_tokens=EXTRACTION_MAX_NEW_TOKENS, do_sample=False, seed=a.seed, batch_size=8)
    frozen = json.loads(a.frozen.read_text(encoding="utf-8"))
    meta = {"benchmark_version": frozen.get("version"), "frozen_status": frozen.get("status"),
            "conversations_sha256": hashlib.sha256((a.data_dir / "tier2_conversations.jsonl").read_bytes()).hexdigest(),
            "qa_questions_sha256": hashlib.sha256((a.data_dir / "tier2_qa_questions.jsonl").read_bytes()).hexdigest(),
            "passthrough_sha256": hashlib.sha256(a.passthrough.read_bytes()).hexdigest(),
            "frozen_files_verified": verified["verified_files"], "frozen_files_not_in_payload": verified["not_in_payload_evaluator_only"],
            "blinded_views": frozen.get("blinded_views")}
    a.out.mkdir(parents=True, exist_ok=True)
    infos = []
    for spec in a.configs.split(","):
        cfg, mode = spec.split(":")
        lex = lexicon if mode == "lexicon" else None
        host = (FrozenJudgeHost(judge_gen, judge_params, JUDGES_BY_ID[a.judge_prompt], metric, "surface" if cfg == "C" else "gloss", meter=judge_meter)
                if cfg in JUDGE_CONFIGS else EmbeddingThresholdHost(a.tau, metric, "surface"))
        extractor = TurnExtractor(cfg, gen, extraction_params, lex, a.seed) if cfg in EXTRACTION_CONFIGS else None
        setup = Setup(cfg, mode, host, metric, gen, lex, passthrough, seed=a.seed, tau=a.tau if cfg == "D" else None, qa_params=qa_params,
                      meter=meter, extractor=extractor, lexicon_sha256=lexicon_sha if mode == "lexicon" else None, progress=(lambda m: print(f"[tier2 +{time.time() - _STAGE['t0']:7.1f}s]   {m}", flush=True)) if a.progress else None)
        run_id = f"{a.run_prefix}-{setup.strategy}-{cfg}_seed{a.seed}"
        stage(f"{run_id}: {len(convs)} conversation(s) begin")
        infos.append(run_setup(setup, convs, questions, run_id=run_id, out=a.out, meta=meta))
        print("[tier2]  ", json.dumps(infos[-1]), flush=True)
    stage("run finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
