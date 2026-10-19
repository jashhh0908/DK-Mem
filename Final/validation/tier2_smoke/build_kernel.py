"""Build the Kaggle notebook for the Tier-2 Config C smoke test (frozen judge, real Qwen2.5-3B + bge-m3, 3 conversations).

The payload is the SYSTEM side of Tier 2 only: the runner and its real dependencies, plus the two label-free inputs and a
uniform opaque passthrough. It never carries the evaluator-only files (relation labels, QA answer aliases, injection positions,
pair specs, distractor bank, the generator/validator that read Team B's key) or the answer key. This builder refuses to write
a notebook if any of that would be embedded (`check_payload`), and the notebook repeats the integrity check on Kaggle before it
writes a file. Nothing is pushed by this script; `kaggle kernels push` is a separate, explicit step.

    python build_kernel.py --user <kaggle-user> --slug dk-mem-tier2-smoke-frozen-judge --out <new dir>

Payload (exact, 70 files; 69 when the 2026-10-13 frozen-judge smoke kernel was built, before judge_host imported judge_entity):
  dkmem/**          .py / .json / .txt, minus dkmem/data/ (Tier-1 input, not needed) and dkmem/eval/ (reads the key)
  eval/tier2/       __init__, run_tier2, memory_store, judge_host, frozen_judge, judge_entity, FROZEN_v1.0.json (a PROJECTION, see below)
  eval/harness/     __init__, _dkmem, extractors, hosts, similarity, writer   (vendored, see eval/harness/VENDORED.json)
  eval/evaluator/   __init__, jsonschema_lite, loaders                        (all that harness.writer imports)
  eval/diagnostics/no_entry/classify.py                                       (pure; imported by harness.extractors)
  eval/tier2/data/  tier2_conversations.jsonl, tier2_qa_questions.jsonl       (frozen, hash-pinned)
  inputs/           t2_passthrough.jsonl                                      (turn id -> opaque id, one rule for every turn)

Two things differ from a plain copy of the repository files, both for blinding:

* ``inputs/t2_passthrough.jsonl`` is generated here from the conversations file. The historical passthrough
  (eval/tier2_full_runs/t2_passthrough.jsonl) gives distractor turns ``sha256("tier2-distractor|" + turn_id)`` ids and injected
  turns their Tier-1 opaque ids, so anyone who knows that rule can tell injected turns from distractors (690 of its 828 ids are
  recomputable, the other 138 are exactly the injected turns). Here every user turn gets the SAME rule, so the file carries no
  injected/distractor signal. The ids are only a passthrough: relation comes from the gold, never from entity ids.
* ``eval/tier2/FROZEN_v1.0.json`` is projected to what the runner and the hash check read (benchmark, version, status, frozen
  timestamp, sha256 map). The original also lists audit edits, limitations and per-stratum counts and names a conversation.
* ``eval/tier2/data/tier2_qa_questions.jsonl`` is a BLINDED VIEW of the frozen file: every ``qa_id`` is replaced by an opaque
  ``qa_<12 hex>``; text, language, conversation and row order are unchanged. The frozen ids end in ``_a``/``_b`` (a different pair,
  two questions) or ``_ab`` (a same pair, one combined question), which would hand the relation to the system. The view's pin
  and its derivation are recorded in the projected FROZEN file (``blinded_views``); the map back to the frozen ids is written to
  the output directory only (``qa_id_map.json``, evaluator side). The runner additionally reads the questions only after a
  conversation's final memory state exists, so the per-conversation question count cannot reach a consolidation decision.

Two directories are written: ``<out>`` holds only what ``kaggle kernels push -p <out>`` should see (the notebook and
``kernel-metadata.json``); ``<out>.evaluator`` holds the evaluator-side artifacts (payload manifest, the exact passthrough file the
run is given, the qa id map) and is never pushed.

Freshness: ``--check <kernel dir>`` re-derives the payload from the working tree and fails if the notebook is stale or the pushed
directory holds anything but the notebook and its metadata. Generated kernel directories (validation/tier2_smoke/kernel_*/,
including ``.evaluator``) are git-ignored; the builder and this check are the source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOTEBOOK_NAME = "dkmem_tier2_smoke.ipynb"
MAX_NOTEBOOK_BYTES = 1_000_000

# reuse the Tier-1 builder's generic pieces (zip payload, round-trip check, unpack cell) instead of copying them
_spec = importlib.util.spec_from_file_location("_abcd_build_kernel", os.path.join(REPO, "validation", "abcd_first_run", "build_kernel.py"))
_abcd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_abcd)
BuildError, make_payload, verify_payload = _abcd.BuildError, _abcd.make_payload, _abcd.verify_payload
manifest_of, tree_sha256 = _abcd.manifest_of, _abcd.tree_sha256
_code, _md, _UNPACK_CELL = _abcd._code, _abcd._md, _abcd._UNPACK_CELL

DKMEM_SUFFIXES = (".py", ".json", ".txt")
# the draft lexicon v2 is not shipped: a v2 run gets the gate's lexicon from its frozen arm root (lexicon/, pinned), never from dkmem/
DKMEM_EXCLUDED_PREFIXES = ("dkmem/data/", "dkmem/eval/", "dkmem/memory/distinction_features_v2.json")
SYSTEM_FILES = (
    "eval/tier2/__init__.py", "eval/tier2/run_tier2.py", "eval/tier2/memory_store.py", "eval/tier2/judge_host.py", "eval/tier2/entry_extraction.py",
    "eval/tier2/frozen_judge.py", "eval/tier2/judge_entity.py", "eval/tier2/FROZEN_v1.0.json",
    "eval/harness/__init__.py", "eval/harness/_dkmem.py", "eval/harness/extractors.py", "eval/harness/hosts.py",
    "eval/harness/similarity.py", "eval/harness/writer.py",
    "eval/evaluator/__init__.py", "eval/evaluator/jsonschema_lite.py", "eval/evaluator/loaders.py",
    "eval/diagnostics/no_entry/classify.py",
)
FROZEN_FILE = "eval/tier2/FROZEN_v1.0.json"
FROZEN_PROJECTION_KEYS = ("benchmark", "version", "status", "frozen_at_utc", "sha256", "blinded_views")
QUESTIONS_PIN_KEY = "data/tier2_qa_questions.jsonl"
QA_RULE = "tier2-smoke-qa-v1|"
_OPAQUE_QA_ID = re.compile(r"qa_[0-9a-f]{12}")
QA_ID_MAP_FILE = "qa_id_map.json"
PUSHED_FILES = (NOTEBOOK_NAME, "kernel-metadata.json")      # the only files the pushed directory may hold
CONVERSATIONS = "eval/tier2/data/tier2_conversations.jsonl"
QUESTIONS = "eval/tier2/data/tier2_qa_questions.jsonl"
DATA_FILES = (CONVERSATIONS, QUESTIONS)
PASSTHROUGH = "inputs/t2_passthrough.jsonl"
ALLOWED_NON_DKMEM = set(SYSTEM_FILES) | set(DATA_FILES) | {PASSTHROUGH}
REQUIRED_FILES = (
    "dkmem/memory/judge.py", "dkmem/memory/distinction_features.json", "dkmem/pairwise_eval.schema.json",
    "dkmem/run_manifest.schema.json", "dkmem/tier1/io.py", "dkmem/backends/llm.py", *ALLOWED_NON_DKMEM,
)

# Passthrough rule: ONE derivation for every user turn (builder-side only; it is not in the payload).
PASSTHROUGH_RULE = "tier2-smoke-passthrough-v1|"
LEGACY_DISTRACTOR_RULE = "tier2-distractor|"          # the historical file's distractor rule (contract.build_passthrough_ids)

# Names that identify evaluator-only material: refused anywhere in a payload path.
FORBIDDEN_NAME_PARTS = ("answer_key", "teamb", "tier2_gold", "tier2_qa_reference", "tier2_injection_metadata", "pair_specs",
                        "pilot_selection", "distractors", "generate_tier2", "validate_tier2", "contract_report", "audit_offline",
                        "full_report")
# Field names that only evaluator-side data carries; refused in the eval/ and inputs/ part of a payload.
FORBIDDEN_FIELDS = ("source_tier1_pair_id", "source_tier1_eval_pair_id", "injected_turn_ids", "distractor_user_turn_ids",
                    "opaque_entity_id_a", "opaque_entity_id_b", "supporting_turn_ids", "forbidden_aliases")
_REAL_EP_ID = re.compile(r"ep_[0-9a-f]{12}")
_RELATION = re.compile(r"\"(?:gold_)?relation\"\s*:\s*\"(?:same|different)\"")
_ORIGINAL_PAIR_ID = re.compile(r"\bt1_\d{4}\b")
_ENT_ID = re.compile(r"ent_[0-9a-f]{16}")

CONVERSATION_KEYS = {"conversation_id", "n_sessions", "sessions"}
SESSION_KEYS = {"date_label", "session_id", "session_index", "turns"}
TURN_KEYS = {"language", "speaker", "text", "turn_id"}
QUESTION_KEYS = {"ask_after_session_index", "conversation_id", "language", "qa_id", "question"}
PASSTHROUGH_KEYS = {"turn_id", "gold_entity_id"}


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_lf(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n").decode("utf-8")


def uniform_id(turn_id: str) -> str:
    return "ent_" + hashlib.sha256((PASSTHROUGH_RULE + turn_id).encode("utf-8")).hexdigest()[:16]


def legacy_distractor_id(turn_id: str) -> str:
    return "ent_" + hashlib.sha256((LEGACY_DISTRACTOR_RULE + turn_id).encode("utf-8")).hexdigest()[:16]


def user_turn_ids(conversations_text: str) -> list[str]:
    return [t["turn_id"] for c in _jsonl(conversations_text) for s in c["sessions"] for t in s["turns"] if t["speaker"] == "user"]


def make_passthrough(conversations_text: str) -> str:
    """Opaque ids for every user turn by one rule: nothing separates an injected turn from a distractor."""
    return "".join(json.dumps({"turn_id": t, "gold_entity_id": uniform_id(t)}) + "\n" for t in user_turn_ids(conversations_text))


def opaque_qa_id(frozen_id: str) -> str:
    return "qa_" + hashlib.sha256((QA_RULE + frozen_id).encode("utf-8")).hexdigest()[:12]


def blind_questions(text: str) -> tuple[str, dict[str, str]]:
    """``(view text, {opaque id: frozen id})``: only ``qa_id`` changes; fields, values and row order are untouched."""
    out, mapping = [], {}
    for row in _jsonl(text):
        new = opaque_qa_id(row["qa_id"])
        if new in mapping:
            raise BuildError("opaque qa id collision")
        mapping[new] = row["qa_id"]
        out.append(json.dumps({**row, "qa_id": new}, ensure_ascii=False))
    return "".join(line + "\n" for line in out), mapping


def project_frozen(text: str, *, questions_view_sha256: str | None = None) -> str:
    """The part of FROZEN_v1.0.json the runner reads: no audit edits, limitations, counts or conversation ids.

    With ``questions_view_sha256`` the questions pin becomes the blinded view's hash and ``blinded_views`` records the frozen
    hash it derives from (a verifier can re-derive the view from the frozen file and compare)."""
    full = json.loads(text)
    pins = dict(full["sha256"])
    views = {}
    if questions_view_sha256 is not None:
        views[QUESTIONS_PIN_KEY] = {"derived_from_frozen_sha256": pins[QUESTIONS_PIN_KEY], "view_sha256": questions_view_sha256,
                                    "transform": "qa_id -> qa_ + sha256(QA_RULE + qa_id)[:12]; every other field and the row order unchanged"}
        pins[QUESTIONS_PIN_KEY] = questions_view_sha256
    proj = {"benchmark": full["benchmark"], "version": full["version"], "status": full["status"],
            "frozen_at_utc": full["frozen_at_utc"], "sha256": pins, "blinded_views": views}
    return json.dumps(proj, indent=1) + "\n"


# --- collecting and checking -------------------------------------------------------------------------------------


def collect_files(repo: str = REPO) -> dict[str, str]:
    """``{relative posix path: LF text}`` of exactly the declared payload (nothing is picked up by wildcard outside dkmem/)."""
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(os.path.join(repo, "dkmem")):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, fn), repo).replace(os.sep, "/")
            if fn.endswith(DKMEM_SUFFIXES) and not rel.startswith(DKMEM_EXCLUDED_PREFIXES):
                files[rel] = _read_lf(os.path.join(repo, rel))
    for rel in (*SYSTEM_FILES, *DATA_FILES):
        files[rel] = _read_lf(os.path.join(repo, rel))
    pins = json.loads(files[FROZEN_FILE])["sha256"]
    if hashlib.sha256(files[QUESTIONS].encode("utf-8")).hexdigest() != pins[QUESTIONS_PIN_KEY]:
        raise BuildError("the repository's tier2_qa_questions.jsonl no longer matches its frozen pin")
    files[QUESTIONS], _ = blind_questions(files[QUESTIONS])
    files[FROZEN_FILE] = project_frozen(files[FROZEN_FILE], questions_view_sha256=hashlib.sha256(files[QUESTIONS].encode("utf-8")).hexdigest())
    files[PASSTHROUGH] = make_passthrough(files[CONVERSATIONS])
    return files


def qa_id_map(repo: str = REPO) -> dict[str, str]:
    """``{opaque id: frozen id}`` (evaluator side: written next to the notebook, never into the payload)."""
    return blind_questions(_read_lf(os.path.join(repo, QUESTIONS)))[1]


SYSTEM_PIN_PREFIXES = ("lexicon/",)      # a v2 freeze also pins the gate's lexicon: the runner reads it, so it is NOT evaluator-only


def protected_hashes(files: dict[str, str]) -> set[str]:
    """sha256 of every evaluator-only frozen file, read from the payload's own FROZEN json (no protected file is opened)."""
    fz = json.loads(files[FROZEN_FILE])["sha256"]
    return {h for rel, h in fz.items()
            if rel not in ("data/tier2_conversations.jsonl", "data/tier2_qa_questions.jsonl") and not rel.startswith(SYSTEM_PIN_PREFIXES)}


def check_payload(files: dict[str, str]) -> None:
    """Refuse a payload that carries evaluator-only material, gold-like data, or cannot run the smoke test."""
    if not files:
        raise BuildError("nothing to embed")
    for n in files:
        if n.startswith("/") or ".." in n.split("/") or n.split("/", 1)[0] not in ("dkmem", "eval", "inputs"):
            raise BuildError(f"path outside dkmem/, eval/ and inputs/: {n}")
        low = n.lower()
        hit = [p for p in FORBIDDEN_NAME_PARTS if p in low]
        if hit:
            raise BuildError(f"evaluator-only or key material would be embedded ({hit}): {n}")
        if n.startswith(DKMEM_EXCLUDED_PREFIXES):
            raise BuildError(f"excluded dkmem tree would be embedded: {n}")
        if not n.startswith("dkmem/") and n not in ALLOWED_NON_DKMEM:
            raise BuildError(f"file is not part of the declared Tier-2 smoke payload: {n}")
    protected = protected_hashes(files)
    for n, t in files.items():
        if hashlib.sha256(t.encode("utf-8")).hexdigest() in protected:
            raise BuildError(f"{n} is byte-identical to a frozen evaluator-only file (renamed copy?)")
        if "\r" in t:
            raise BuildError(f"carriage returns would change the file's hash on Kaggle: {n}")
        for lineno, line in enumerate(t.splitlines(), 1):
            if (_REAL_EP_ID.search(line) and _RELATION.search(line)) or _ORIGINAL_PAIR_ID.search(line):
                raise BuildError(f"gold label or original Tier-1 pair id would be embedded: {n}:{lineno}")
            if _RELATION.search(line):
                raise BuildError(f"a relation label would be embedded: {n}:{lineno}")
        if not n.startswith("dkmem/"):
            leaked = [f for f in FORBIDDEN_FIELDS if f in t]
            if leaked:
                raise BuildError(f"evaluator-side field name(s) {leaked} in {n}")
    missing = [n for n in REQUIRED_FILES if n not in files]
    if missing:
        raise BuildError(f"files the smoke test needs are missing: {missing}")
    _check_inputs(files)


def _check_inputs(files: dict[str, str]) -> None:
    """Structure of the data files (field NAMES only: nothing here depends on a label) and their pinned hashes."""
    frozen = json.loads(files[FROZEN_FILE])
    if sorted(frozen) != sorted(FROZEN_PROJECTION_KEYS):
        raise BuildError(f"the payload's FROZEN_v1.0.json must be the projection {FROZEN_PROJECTION_KEYS}, has {sorted(frozen)}")
    fz = frozen["sha256"]
    for rel in DATA_FILES:
        got = hashlib.sha256(files[rel].encode("utf-8")).hexdigest()
        want = fz[rel.removeprefix("eval/tier2/")]
        if got != want:
            raise BuildError(f"{rel} hashes to {got[:16]}..., not the frozen {want[:16]}...")
    convs = _jsonl(files[CONVERSATIONS])
    user_turns = set()
    for c in convs:
        if set(c) != CONVERSATION_KEYS:
            raise BuildError(f"conversation fields {sorted(c)} differ from the label-free layout")
        for s in c["sessions"]:
            if set(s) != SESSION_KEYS:
                raise BuildError(f"session fields {sorted(s)} differ from the label-free layout")
            for t in s["turns"]:
                if set(t) != TURN_KEYS:
                    raise BuildError(f"turn fields {sorted(t)} differ from the label-free layout")
                if t["speaker"] == "user":
                    user_turns.add(t["turn_id"])
    ids = {c["conversation_id"] for c in convs}
    view = frozen["blinded_views"].get(QUESTIONS_PIN_KEY)
    if not view or view["view_sha256"] != fz[QUESTIONS_PIN_KEY] or not re.fullmatch(r"[0-9a-f]{64}", view.get("derived_from_frozen_sha256", "")):
        raise BuildError("the questions file is not a recorded blinded view of the frozen file")
    qa_ids = []
    for q in _jsonl(files[QUESTIONS]):
        if set(q) != QUESTION_KEYS:
            raise BuildError(f"question fields {sorted(q)} differ from the label-free layout")
        if q["conversation_id"] not in ids:
            raise BuildError("a question refers to an unknown conversation")
        if not _OPAQUE_QA_ID.fullmatch(q["qa_id"]):
            raise BuildError("question ids must be opaque qa_<12 hex>: the frozen ids separate same pairs from different pairs")
        qa_ids.append(q["qa_id"])
    if len(set(qa_ids)) != len(qa_ids):
        raise BuildError("opaque question ids must be unique")
    rows = _jsonl(files[PASSTHROUGH])
    if any(set(r) != PASSTHROUGH_KEYS or not _ENT_ID.fullmatch(r["gold_entity_id"]) for r in rows):
        raise BuildError("passthrough rows must hold exactly turn_id and an opaque ent_<16 hex> id")
    if len({r["gold_entity_id"] for r in rows}) != len(rows) or {r["turn_id"] for r in rows} != user_turns:
        raise BuildError("passthrough ids must be unique and cover exactly the user turns")
    if any(r["gold_entity_id"] != uniform_id(r["turn_id"]) for r in rows):
        raise BuildError("passthrough ids must all follow the one uniform rule (a mixed rule separates injected from distractor turns)")
    if any(r["gold_entity_id"] == legacy_distractor_id(r["turn_id"]) for r in rows):
        raise BuildError("passthrough ids follow the historical distractor rule: injected turns would be identifiable")


SMOKE_PLANS = {"v1": (("hi", 2), ("de", 1)),                  # the original benchmark: two Hindi and one German conversation
               "v2": (("hi", 2), ("mr", 1), ("gu", 1))}       # the v2 benchmark: every language is exercised (German is out of v2)


def select_conversations(files: dict[str, str], plan=SMOKE_PLANS["v1"]) -> list[str]:
    """Label-free, deterministic: per ``plan`` the first ``n`` conversations of each language, in file order (default: two Hindi + one German)."""
    by_lang: dict[str, list[str]] = {}
    for c in _jsonl(files[CONVERSATIONS]):
        langs = {t["language"] for s in c["sessions"] for t in s["turns"] if t["speaker"] == "user"}
        if len(langs) == 1:
            by_lang.setdefault(next(iter(langs)), []).append(c["conversation_id"])
    if any(len(by_lang.get(lang, [])) < n for lang, n in plan):
        raise BuildError("the conversations file lacks the smoke selection " + ", ".join(f"{n} {lang}" for lang, n in plan))
    return [cid for lang, n in plan for cid in by_lang[lang][:n]]


# --- the notebook ------------------------------------------------------------------------------------------------

_FORBIDDEN_SCAN_CELL = '''
FORBIDDEN = @@FORBIDDEN@@
found = []
for dp, dn, fn in os.walk(SRC):
    for name in fn + dn:
        low = os.path.join(dp, name).lower()
        found += [p for p in FORBIDDEN if p in low]
assert not found, "evaluator-only or key material is present in the payload tree: " + str(sorted(set(found)))
sys.path[:0] = [os.path.join(SRC, "eval"), SRC]
import importlib
importlib.import_module("tier2.run_tier2")   # fail before any model is loaded
print("payload tree clean; runner imports")
'''

# ``acceptance`` are pipeline-correctness checks and fail the kernel; ``observations`` are findings about the judge's behaviour
# on three conversations (a conservative judge is a result, not a defect) and are reported without failing. The summary is
# written before the assertion, so a failed acceptance still leaves its evidence in the kernel output.
_SUMMARY_CELL = '''
import glob, hashlib, re, shutil
OUTDIR = "@@OUTDIR@@"
PT_PATH = "@@PTPATH@@"
pt_bytes = open(PT_PATH, "rb").read()
PT_SHA = hashlib.sha256(pt_bytes).hexdigest()
shutil.copyfile(PT_PATH, OUTDIR + "/t2_passthrough_used.jsonl")   # the exact file the runs were given: validate with it


def stats(v):
    v = sorted(v)
    if not v:
        return {"n": 0}
    q = lambda f: v[min(len(v) - 1, int(round(f * (len(v) - 1))))]
    return {"n": len(v), "min": v[0], "median": q(0.5), "p95": q(0.95), "max": v[-1]}


summary = {"passthrough_sha256": PT_SHA, "payload_tree_sha256": TREE_SHA256, "runs": {}}
for info_path in sorted(glob.glob(OUTDIR + "/runs/*.tier2_run_info.json")):
    rid = os.path.basename(info_path).removesuffix(".tier2_run_info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    man = json.load(open(OUTDIR + "/runs/" + rid + "/run_manifest.json", encoding="utf-8"))
    rows = [json.loads(l) for l in open(OUTDIR + "/runs/" + rid + "/pairwise_eval.jsonl", encoding="utf-8") if l.strip()]
    qa = [json.loads(l) for l in open(OUTDIR + "/runs/" + rid + ".tier2_qa_answers.jsonl", encoding="utf-8") if l.strip()]
    t2 = man["tier2"]
    entry = {"info": info, "strategy": man["strategy"], "pipeline_config": man["pipeline_config"], "dkmem_mode": man["dkmem_mode"],
             "dataset_tier": man["dataset_tier"], "rows": len(rows), "thresholds": sorted({repr(r["threshold"]) for r in rows}),
             "frozen_files_verified": t2["frozen_files_verified"], "frozen_files_not_in_payload": t2["frozen_files_not_in_payload"],
             "passthrough_sha256": t2["passthrough_sha256"], "tokenizer": t2.get("tokenizer"),
             "decision_distribution": t2.get("decision_distribution"),
             "qa": {"answers": len(qa), "ids_opaque": all(re.fullmatch(r"qa_[0-9a-f]{12}", a["qa_id"]) for a in qa),
                    "output_tokens": stats([a["output_tokens"] for a in qa if a.get("output_tokens") is not None]),
                    "questions_read": t2["qa"].get("questions_read")}}
    jl_path = OUTDIR + "/runs/" + rid + ".judge_log.jsonl"
    if os.path.exists(jl_path):
        jl = [json.loads(l) for l in open(jl_path, encoding="utf-8") if l.strip()]
        props = {}
        for r in jl:
            props[str(r["proposal"])] = props.get(str(r["proposal"]), 0) + 1
        out_tok = [r["output_tokens"] for r in jl if r.get("output_tokens") is not None]
        in_tok = [r["prompt_tokens"] for r in jl if r.get("prompt_tokens") is not None]
        sizes = [len(r["raw_output"].encode("utf-8")) for r in jl]
        entry["judge"] = {"prompt_id": man["judge_prompt_id"], "prompt_sha256": man["judge_prompt_sha256"], "calls": len(jl),
                          "parse_failures": sum(r["status"] != "ok" for r in jl), "possibly_truncated": sum(bool(r.get("possibly_truncated")) for r in jl),
                          "proposals": props, "decision_distribution": man.get("judge_decision_distribution"),
                          "raw_output_bytes": stats(sizes),
                          "tokens": {"measured": bool(jl) and len(out_tok) == len(jl) and len(in_tok) == len(jl), "prompt": stats(in_tok),
                                     "output": stats(out_tok), "at_budget": sum(bool(r.get("at_budget")) for r in jl)},
                          "max_new_tokens": man["generation_run_info"]["generation_params"]["max_new_tokens"],
                          "judge_parse_failures_manifest": man["judge_parse_failures"]}
    summary["runs"][rid] = entry
FROZEN = "3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff"
runs = list(summary["runs"].values())
c_runs = [e for e in runs if e["pipeline_config"] == "C"]
accept = {
    "four_runs": len(runs) == 4,
    "frozen_prompt_sha": bool(c_runs) and all(e["judge"]["prompt_sha256"] == FROZEN and e["judge"]["prompt_id"] == "merge_judge_v1" for e in c_runs),
    "budget_128": bool(c_runs) and all(e["judge"]["max_new_tokens"] == 128 for e in c_runs),
    "no_parse_failures": bool(c_runs) and all(e["judge"]["parse_failures"] == 0 and e["judge"]["possibly_truncated"] == 0 for e in c_runs),
    "tokenization_recorded": bool(c_runs) and all(e["judge"]["tokens"]["measured"] and (e["tokenizer"] or {}).get("name_or_path") for e in c_runs),
    "budget_not_binding": bool(c_runs) and all(e["judge"]["tokens"]["at_budget"] == 0 and e["judge"]["tokens"]["output"].get("max", 10**9) < 128 for e in c_runs),
    "c_thresholds_null": bool(c_runs) and all(e["thresholds"] == ["None"] for e in c_runs),
    "tier2_manifests": bool(runs) and all(e["dataset_tier"] == "tier2_injected_sessions" for e in runs),
    "evaluator_files_absent": bool(runs) and all(len(e["frozen_files_not_in_payload"]) > 0 for e in runs),
    "passthrough_is_the_payload_file": bool(runs) and all(e["passthrough_sha256"] == PT_SHA for e in runs),
    "qa_ids_opaque": bool(runs) and all(e["qa"]["ids_opaque"] and e["qa"]["answers"] > 0 for e in runs),
    "questions_read_after_consolidation": bool(runs) and all(e["info"].get("questions_loaded_lazily") for e in runs),
}
summary["acceptance"] = accept
summary["observations"] = {
    "judge_not_constant": {rid: len([k for k in e["judge"]["proposals"] if k != "None"]) > 1 for rid, e in summary["runs"].items() if "judge" in e},
    "judge_proposals": {rid: e["judge"]["proposals"] for rid, e in summary["runs"].items() if "judge" in e},
    "judge_output_tokens": {rid: e["judge"]["tokens"]["output"] for rid, e in summary["runs"].items() if "judge" in e},
    "pairwise_decisions": {rid: (e["decision_distribution"] or {}).get("pairwise_decisions") for rid, e in summary["runs"].items()},
}
json.dump(summary, open(OUTDIR + "/smoke_summary.json", "w", encoding="utf-8"), indent=1)
json.dump({"nvidia_smi_L": GPU_LIST, "cli_args": ARGS, "conversations": CONVERSATIONS, "payload_tree_sha256": TREE_SHA256},
          open(OUTDIR + "/run_environment.json", "w", encoding="utf-8"), indent=2)
print(json.dumps({"acceptance": accept, "observations": summary["observations"]}, indent=1))
assert all(accept.values()), "smoke acceptance failed: " + str([k for k, v in accept.items() if not v])
'''


def build_run_args(conversations: list[str], seed: int, tau: float, model_id: str, model_revision: str, similarity_revision: str) -> list[str]:
    return ["--progress", "--passthrough", "/kaggle/working/src/" + PASSTHROUGH, "--data-dir", "/kaggle/working/src/eval/tier2/data",
            "--conversations", ",".join(conversations), "--configs", "C:off,C:lexicon,D:off,D:lexicon", "--tau", str(tau),
            "--seed", str(seed), "--run-prefix", "t2frozen-smoke", "--out", "/kaggle/working/runs",
            "--model-id", model_id, "--model-revision", model_revision, "--similarity-revision", similarity_revision]


def build_notebook(files: dict[str, str], run_args: list[str], conversations: list[str]) -> tuple[dict, str]:
    payload = make_payload(files)
    manifest = manifest_of(files)
    tree = tree_sha256(manifest)
    unpack = (_UNPACK_CELL.replace("@@PAYLOAD@@", payload)
              .replace("@@PAYLOAD_SHA256@@", hashlib.sha256(payload.encode("ascii")).hexdigest())
              .replace("@@MANIFEST@@", json.dumps(json.dumps(manifest, sort_keys=True))))
    cells = [
        _md("# DK-Mem Tier 2 smoke: Config C with the frozen Tier-1 merge judge (3 conversations, Qwen2.5-3B, seed 0)"),
        _code(unpack),
        _code(_FORBIDDEN_SCAN_CELL.replace("@@FORBIDDEN@@", json.dumps(list(FORBIDDEN_NAME_PARTS)))),
        _code(f"""
ARGS = {json.dumps(run_args)}
CONVERSATIONS = {json.dumps(conversations)}
TREE_SHA256 = {json.dumps(tree)}
print("python -m tier2.run_tier2", " ".join(ARGS), flush=True)
t0 = time.time()
proc = subprocess.Popen([sys.executable, "-m", "tier2.run_tier2", *ARGS], cwd=os.path.join(SRC, "eval"), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, env={{**os.environ, "PYTHONPATH": os.path.join(SRC, "eval") + os.pathsep + SRC}})
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print("exit code", rc, "| seconds", round(time.time() - t0, 1))
assert rc == 0
"""),
        _code(_SUMMARY_CELL.replace("@@OUTDIR@@", "/kaggle/working").replace("@@PTPATH@@", "/kaggle/working/src/" + PASSTHROUGH)),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return nb, payload


def payload_listing(files: dict[str, str]) -> dict:
    return {"tree_sha256": tree_sha256(manifest_of(files)), "n_files": len(files),
            "files": [{"path": n, "bytes": len(t.encode("utf-8")), "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest()} for n, t in sorted(files.items())]}


def evaluator_dir(out: str) -> str:
    return os.fspath(out).rstrip("/" + os.sep) + ".evaluator"


def check_kernel_dir(path: str) -> int:
    """Freshness: does the built notebook still equal what the working tree would build? (0 = fresh)"""
    try:
        ev = evaluator_dir(path)
        nb = json.load(open(os.path.join(path, NOTEBOOK_NAME), encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        manifest = json.load(open(os.path.join(ev, "payload_manifest.json"), encoding="utf-8"))
        files = collect_files()
        check_payload(files)
        problems = []
        extra = sorted(set(os.listdir(path)) - set(PUSHED_FILES))
        if extra or set(PUSHED_FILES) - set(os.listdir(path)):
            problems.append(f"the pushed directory must hold exactly {list(PUSHED_FILES)}; unexpected {extra}")
        payload_sha = hashlib.sha256(make_payload(files).encode("ascii")).hexdigest()
        tree = tree_sha256(manifest_of(files))
        m = re.search(r'PAYLOAD_SHA256 = "([0-9a-f]{64})"', src)
        t = re.search(r'TREE_SHA256 = "([0-9a-f]{64})"', src)
        if not m or m.group(1) != payload_sha:
            problems.append("the notebook's payload is not what the working tree builds")
        if not t or t.group(1) != tree or manifest.get("tree_sha256") != tree:
            problems.append("the payload tree hash differs from the working tree")
        used = open(os.path.join(ev, "t2_passthrough_used.jsonl"), "rb").read()
        if used != files[PASSTHROUGH].encode("utf-8") or hashlib.sha256(used).hexdigest() != manifest.get("passthrough_sha256"):
            problems.append("t2_passthrough_used.jsonl is not the payload's passthrough")
        qmap = json.load(open(os.path.join(ev, QA_ID_MAP_FILE), encoding="utf-8"))
        if qmap != qa_id_map():
            problems.append("qa_id_map.json does not match the frozen questions")
    except (OSError, ValueError, KeyError, BuildError) as e:
        print(f"build_kernel --check: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if problems:
        print("build_kernel --check: STALE: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"build_kernel --check: fresh | {len(files)} files | tree sha256 {tree[:16]} | payload sha256 {payload_sha[:16]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None, metavar="KERNEL_DIR", help="verify a built kernel directory against the working tree and exit")
    ap.add_argument("--user")
    ap.add_argument("--slug")
    ap.add_argument("--out", help="a NEW directory (and a new <out>.evaluator); nothing existing is overwritten")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.90, help="run-level threshold of Config D (an execution value, not an operating point)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--model-revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--similarity-revision", default="5617a9f61b028005a4858fdac845db406aefb181")
    ap.add_argument("--conversations", default=None, help="comma-separated ids (default: first two Hindi + first German, label-free)")
    ap.add_argument("--max-notebook-bytes", type=int, default=MAX_NOTEBOOK_BYTES)
    args = ap.parse_args(argv)
    if args.check:
        return check_kernel_dir(args.check)
    if not (args.user and args.slug and args.out):
        ap.error("--user, --slug and --out are required (or use --check KERNEL_DIR)")
    ev = evaluator_dir(args.out)
    try:
        for d in (args.out, ev):
            if os.path.exists(d) and os.listdir(d):
                raise BuildError(f"{d} exists and is not empty; use a new directory")
        files = collect_files()
        check_payload(files)
        convs = args.conversations.split(",") if args.conversations else select_conversations(files)
        known = {c["conversation_id"] for c in _jsonl(files[CONVERSATIONS])}
        if not set(convs) <= known:
            raise BuildError(f"unknown conversation ids: {sorted(set(convs) - known)}")
        run_args = build_run_args(convs, args.seed, args.tau, args.model_id, args.model_revision, args.similarity_revision)
        nb, payload = build_notebook(files, run_args, convs)
        verify_payload(payload, files)
        text = json.dumps(nb, indent=1, ensure_ascii=False)
        size = len(text.encode("utf-8"))
        if size > args.max_notebook_bytes:
            raise BuildError(f"the notebook is {size:,} bytes, over the {args.max_notebook_bytes:,}-byte limit")
    except BuildError as e:
        print(f"build_kernel: refusing to write the notebook: {e}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(args.out, NOTEBOOK_NAME), "w", encoding="utf-8") as f:
        f.write(text)
    meta = {"id": f"{args.user}/{args.slug}", "title": args.slug.replace("-", " ").title()[:60], "code_file": NOTEBOOK_NAME,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    with open(os.path.join(args.out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # ---- evaluator side: never pushed
    qmap_text = json.dumps(qa_id_map(), indent=1, sort_keys=True)
    with open(os.path.join(ev, QA_ID_MAP_FILE), "w", encoding="utf-8", newline="\n") as f:
        f.write(qmap_text)    # opaque id -> frozen id
    with open(os.path.join(ev, "payload_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({**payload_listing(files), "conversations": convs, "run_args": run_args,
                   "payload_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
                   "passthrough_rule": "uniform: ent_ + sha256(PASSTHROUGH_RULE + turn_id)[:16] for every user turn",
                   "passthrough_sha256": hashlib.sha256(files[PASSTHROUGH].encode("utf-8")).hexdigest(),
                   "qa_questions_view_sha256": hashlib.sha256(files[QUESTIONS].encode("utf-8")).hexdigest(),
                   "qa_id_map_file": QA_ID_MAP_FILE, "qa_id_map_sha256": hashlib.sha256(qmap_text.encode("utf-8")).hexdigest()}, f, indent=1)
    # the exact file the run is given (contract validate --passthrough): same bytes as inputs/ in the payload
    with open(os.path.join(ev, "t2_passthrough_used.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.write(files[PASSTHROUGH])
    print("wrote", args.out, "(+", ev, ") |", len(files), "files |", f"payload {len(payload):,} B, notebook {size:,} B | conversations {convs} | "
          f"tree sha256 {tree_sha256(manifest_of(files))[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
