"""What a Tier-2 run's Config C judge actually was, read from its manifest and judge log (evaluator side; no model, no gold).

Used by the audit and the report so that a run is never described as "the frozen Tier-1 judge" unless its manifest says so,
and so that a degenerate (constant) judge is reported from the data, not from a hard-coded sentence.

``verify_log`` does not trust the log's own ``status`` / ``proposal`` / counts. For a run whose manifest names a current judge it
(1) checks that every row carries the manifest's prompt id and sha256, and (2) re-parses every ``raw_output`` with the strict
parser of that judge family and compares the result with the logged status and proposal.

Parser per family (the only parsers used; none is guessed):

* ``tier1_merge_judge_v1`` (``merge_judge_v1``, frozen sha) and ``tier2_identity_judge`` (``merge_judge_entity_t2_v1`` / ``_v2``):
  ``dkmem.memory.judge.parse_judge_output`` -- the parser ``judge_pairs`` / ``FrozenJudgeHost`` applied when the log was written. It
  accepts exactly ``{"decision": "merge" | "keep_both", "reason": str}`` in either key order. Logged ``status`` is ``ok`` iff the parse
  succeeded (else ``parse_error``); logged ``proposal`` is the parsed decision, or ``None`` for a failed parse.
* ``historical_harness_judge`` (id ``merge_judge_v1`` with the harness sha): its parser is not in this repository, so its rows cannot be
  re-parsed. They are reported ``legacy_unverifiable`` and never counted as verified.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import embedder_pin
from .frozen_judge import (
    ENTITY_JUDGE_MAX_NEW_TOKENS, IDENTITY_JUDGES, FROZEN_JUDGE_MAX_NEW_TOKENS, FROZEN_JUDGE_PROMPT_ID, FROZEN_JUDGE_PROMPT_SHA256, JUDGE_ROW_THRESHOLD,
    LEGACY_HARNESS_JUDGE_PROMPT_SHA256, LEGACY_JUDGE_MAX_NEW_TOKENS, LEGACY_JUDGE_ROW_THRESHOLD,
)


FAMILY_TIER1 = "tier1_merge_judge_v1"
FAMILY_IDENTITY = "tier2_identity_judge"
FAMILY_LEGACY = "historical_harness_judge"
STRICT_PARSER_NAME = "dkmem.memory.judge.parse_judge_output"
CURRENT_FAMILIES = (FAMILY_TIER1, FAMILY_IDENTITY)


def family(manifest: dict):
    """The judge family a manifest declares, by prompt id AND sha256 (``None`` if it is none of the known judges)."""
    jid, sha = manifest.get("judge_prompt_id"), manifest.get("judge_prompt_sha256")
    if jid == FROZEN_JUDGE_PROMPT_ID and sha == FROZEN_JUDGE_PROMPT_SHA256:
        return FAMILY_TIER1
    if jid in IDENTITY_JUDGES and sha == IDENTITY_JUDGES[jid]:
        return FAMILY_IDENTITY
    if jid == FROZEN_JUDGE_PROMPT_ID and sha == LEGACY_HARNESS_JUDGE_PROMPT_SHA256:
        return FAMILY_LEGACY
    return None


def strict_parser(fam):
    """The strict parser of a judge family, or ``None`` when the family has none in this repository (legacy, unknown)."""
    if fam in CURRENT_FAMILIES:
        from dkmem.memory.judge import parse_judge_output      # lazy: evaluator-side callers need dkmem on the path only here
        return parse_judge_output
    return None


def verify_log(manifest: dict, judge_log: list[dict]) -> dict:
    """Independent verification of a judge log against its manifest (see the module docstring).

    ``status``: ``verified`` (every row identity-checked and re-parsed, nothing disagrees), ``failed`` (a current-judge run with a
    missing identity, a mismatch or a parse disagreement), ``legacy_unverifiable`` (declared historical run whose rows cannot be
    re-parsed and carry no per-row prompt hash; nothing contradicts it, nothing is claimed verified), ``unrecognised``, or ``empty``.
    ``problems`` maps a code to the number of rows with that problem.
    """
    fam = family(manifest)
    n = len(judge_log)
    parser = strict_parser(fam)
    mid, msha = manifest.get("judge_prompt_id"), manifest.get("judge_prompt_sha256")
    c = {"identity_ok": 0, "identity_missing": 0, "reparse_ok": 0, "reparse_rejected": 0, "agree": 0}
    problems: dict[str, int] = {}
    proposals: Counter = Counter()
    first_bad: dict[str, int] = {}

    def bad(code, i):
        problems[code] = problems.get(code, 0) + 1
        first_bad.setdefault(code, i)

    for i, r in enumerate(judge_log):
        if not isinstance(r, dict):
            bad("row_not_object", i)
            continue
        rid, rsha = r.get("prompt_id"), r.get("prompt_sha256")
        if rid is None or rsha is None:
            c["identity_missing"] += 1
        else:
            if rid != mid:
                bad("row_prompt_id_mismatch", i)
            elif rsha != msha:
                bad("row_prompt_sha_mismatch", i)
            else:
                c["identity_ok"] += 1
        if parser is None:
            continue
        parsed = parser(r.get("raw_output"))
        c["reparse_ok" if parsed.ok else "reparse_rejected"] += 1
        proposals[parsed.decision if parsed.ok else None] += 1
        logged_ok = r.get("status") == "ok"
        if parsed.ok != logged_ok:
            bad("row_status_mismatch", i)
        elif r.get("proposal") != (parsed.decision if parsed.ok else None):
            bad("row_proposal_mismatch", i)
        else:
            c["agree"] += 1
    if fam in CURRENT_FAMILIES and c["identity_missing"]:
        problems["row_identity_missing"] = c["identity_missing"]       # a current judge's rows must say which prompt produced them
        first_bad.setdefault("row_identity_missing", next(i for i, r in enumerate(judge_log) if isinstance(r, dict) and (r.get("prompt_id") is None or r.get("prompt_sha256") is None)))
    if fam is None:
        status = "unrecognised"
    elif n == 0:
        status = "empty"
    elif problems:
        status = "failed"
    elif fam == FAMILY_LEGACY:
        status = "legacy_unverifiable"
    else:
        status = "verified"
    return {"status": status, "family": fam, "parser": STRICT_PARSER_NAME if parser else None, "n_rows": n,
            "n_identity_verified": c["identity_ok"], "n_identity_missing": c["identity_missing"],
            "n_reparsed_ok": c["reparse_ok"] if parser else None, "n_reparsed_rejected": c["reparse_rejected"] if parser else None,
            "n_logged_agree_with_reparse": c["agree"] if parser else None,
            "reparsed_proposal_counts": {str(k): v for k, v in proposals.items()} if parser else None,
            "problems": problems, "first_bad_row": first_bad}


def judge_budget(manifest: dict):
    """The judge's ``max_new_tokens`` as the manifest recorded it (None if absent)."""
    return ((manifest.get("generation_run_info") or {}).get("generation_params") or {}).get("max_new_tokens")


def classify(manifest: dict) -> dict:
    jid, sha = manifest.get("judge_prompt_id"), manifest.get("judge_prompt_sha256")
    frozen = jid == FROZEN_JUDGE_PROMPT_ID and sha == FROZEN_JUDGE_PROMPT_SHA256
    legacy = jid == FROZEN_JUDGE_PROMPT_ID and sha == LEGACY_HARNESS_JUDGE_PROMPT_SHA256
    identity = jid in IDENTITY_JUDGES and sha == IDENTITY_JUDGES[jid]
    return {"prompt_id": jid, "prompt_sha256": sha, "frozen_tier1": frozen, "tier2_identity_judge": identity, "historical_harness_judge": legacy,
            "recognised": frozen or identity or legacy, "max_new_tokens": judge_budget(manifest),
            "expected_max_new_tokens": FROZEN_JUDGE_MAX_NEW_TOKENS if frozen else (ENTITY_JUDGE_MAX_NEW_TOKENS if identity else (LEGACY_JUDGE_MAX_NEW_TOKENS if legacy else None)),
            "row_threshold_convention": JUDGE_ROW_THRESHOLD if (frozen or identity) else (LEGACY_JUDGE_ROW_THRESHOLD if legacy else "unknown"),
            "comparable_with_tier1_config_c": frozen}


def behaviour(judge_log: list[dict]) -> dict:
    """Counts only. ``constant`` is True when every parsed call got the same proposal (then FCR/MCR are trivial)."""
    n = len(judge_log)
    proposals = Counter(r.get("proposal") for r in judge_log)
    parsed = {k: v for k, v in proposals.items() if k is not None}
    return {"judge_calls": n, "status_ok": sum(r.get("status") == "ok" for r in judge_log),
            "parse_errors": sum(r.get("status") != "ok" for r in judge_log),
            "proposal_counts": {str(k): v for k, v in proposals.items()},
            "raw_output_distinct": dict(Counter((r.get("raw_output") or "").strip() for r in judge_log)),
            "source": "self-reported log fields (status/proposal); see verify_log for the independent re-parse",
            "constant": n > 0 and len(parsed) == 1 and sum(parsed.values()) == n}


def describe(manifest: dict, judge_log: list[dict]) -> dict:
    """``classification`` is the sentence the audit and the report print for this run's judge."""
    c, b, v = classify(manifest), behaviour(judge_log), verify_log(manifest, judge_log)
    n = b["judge_calls"]
    if c["frozen_tier1"] or c["tier2_identity_judge"]:
        # counts come from the INDEPENDENT re-parse of every raw_output, never from the log's own status field
        evidence = (f"{v['n_reparsed_ok']}/{n} raw outputs accepted by the frozen strict parser {v['parser']} on independent re-parse, "
                    f"{v['n_reparsed_rejected']} parse failures (never repaired); logged status/proposal agree with the re-parse on "
                    f"{v['n_logged_agree_with_reparse']}/{n} rows; per-row prompt identity verified on {v['n_identity_verified']}/{n}; "
                    f"re-parsed proposals {v['reparsed_proposal_counts']}.")
        if v["status"] == "verified":
            tail = " VERIFIED: every row matches the manifest's prompt and the strict re-parse."
        else:
            tail = (f" NOT VERIFIED ({v['status']}): problems {v['problems']}; the log cannot be taken as evidence of this judge's "
                    f"answers until it is regenerated or explained.")
        if c["frozen_tier1"]:
            text = (f"FROZEN TIER-1 JUDGE: merge_judge_v1 sha256 {c['prompt_sha256'][:12]}..., decoding budget {c['max_new_tokens']} new tokens "
                    f"(Tier-1 value {FROZEN_JUDGE_MAX_NEW_TOKENS}); {evidence}{tail}")
        else:
            text = (f"TIER-2 IDENTITY JUDGE: {c['prompt_id']} sha256 {c['prompt_sha256'][:12]}... (merge = same specific person, thing or "
                    f"event, whatever the facts; NOT the Tier-1 judge, so its Config C numbers are not comparable with Tier-1 Config C), decoding budget "
                    f"{c['max_new_tokens']} new tokens; {evidence}{tail}")
    elif c["historical_harness_judge"]:
        text = (f"NOT THE FROZEN TIER-1 JUDGE (historical harness run): the prompt id is merge_judge_v1 but the text is the harness "
                f"neutral same/different prompt, sha256 {c['prompt_sha256'][:12]}... instead of {FROZEN_JUDGE_PROMPT_SHA256[:12]}..., "
                f"decoded with {c['max_new_tokens']} new tokens. Its Config C results are not comparable with Tier 1 and are kept only as a record. "
                f"LEGACY_UNVERIFIABLE: its parser is not in this repository and {v['n_identity_missing']}/{n} rows carry no prompt hash, so the "
                f"logged status/proposal of these rows are self-reported and NOT independently verified (verification status: {v['status']}).")
    else:
        text = (f"UNRECOGNISED JUDGE PROMPT {c['prompt_id']!r} sha256 {c['prompt_sha256']!r}: neither the frozen Tier-1 merge_judge_v1 "
                f"nor the Tier-2 identity judge nor the recognised historical harness judge; the run's Config C numbers must not be reported.")
    if b["constant"]:
        only = next(iter(k for k in b["proposal_counts"] if k != "None"))
        text += (f" DEGENERATE: the judge returned '{only}' on all {n}/{n} compared pairs, so FCR/MCR are the trivial constant outcome "
                 f"and carry no information about false versus missed consolidation, and the gate has nothing to act on.")
    return {**c, "behaviour": b, "verification": v, "classification": text}


def validate_run_verified(run_dir, *args, **kwargs) -> dict:
    """``contract_v2.validate_run`` (all four configs) followed by ``augment_report`` (see there)."""
    from . import contract_v2    # lazy; contract_v2 equals the pinned contract for configs C and D (tested) and also accepts A and B
    return augment_report(contract_v2.validate_run(run_dir, *args, **kwargs), run_dir)


def augment_report(report: dict, run_dir) -> dict:
    """Add to a ``contract.validate_run`` report two independent verifications that the pinned ``contract.py`` does not make:
    the embedder revision (every config; ``embedder_pin.verify_manifest_embedder``) and the judge log (Config C; ``verify_log``).

    ``contract.py`` is hash-pinned by ``ANALYSIS_PLAN_FREEZE_v1.json`` and is therefore left byte-identical; this wrapper adds to its
    report, never edits it: ``errors`` gain a ``judge_log_<problem>`` entry per problem and, for a current judge, a
    ``judge_log_parse_failure_count`` error when the manifest's ``judge_parse_failures`` differs from the independent re-parse; a
    declared historical run whose rows cannot be verified gets a ``legacy_unverifiable`` warning (still loadable, never "verified").
    A manifest whose recorded bge-m3 revision differs from ``embedder_pin.BGE_M3_REVISION`` (or records none) adds an
    ``embedder_revision_wrong_revision`` / ``_missing`` / ``_wrong_model`` error. The result carries ``judge_log_verification`` (``None``
    for Config D), ``embedder_verification`` and a recomputed ``ok``.
    """
    report["judge_log_verification"] = None
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    emb = embedder_pin.verify_manifest_embedder(manifest)
    report["embedder_verification"] = emb
    if emb["status"] not in (embedder_pin.OK, embedder_pin.NOT_APPLICABLE):
        report["errors"].append({"code": "embedder_revision_" + emb["status"],
                                 "message": f"manifest records embedder revision {emb['recorded']!r} (requested {emb['requested']!r}, model "
                                            f"{emb['model_id']!r}); the registered bge-m3 revision is {emb['expected']!r}"})
        report["ok"] = not report["errors"]
    if report.get("pipeline_config") not in ("A", "B", "C"):       # the configs whose host is an LLM judge
        return report
    log_path = run_dir.parent / f"{run_dir.name}.judge_log.jsonl"
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()] if log_path.exists() else []
    ver = verify_log(manifest, rows)
    report["judge_log_verification"] = ver
    for code, n_rows in sorted(ver["problems"].items()):
        report["errors"].append({"code": "judge_log_" + code,
                                 "message": f"{n_rows} judge-log rows: {code} (first at row {ver['first_bad_row'].get(code)}); "
                                            f"the log does not match the manifest's declared judge"})
    if ver["family"] in CURRENT_FAMILIES and manifest.get("judge_parse_failures") != ver["n_reparsed_rejected"]:
        report["errors"].append({"code": "judge_log_parse_failure_count",
                                 "message": f"manifest judge_parse_failures={manifest.get('judge_parse_failures')!r}, independent re-parse of "
                                            f"the raw outputs rejects {ver['n_reparsed_rejected']}"})
    if ver["status"] == "legacy_unverifiable":
        report["warnings"].append({"code": "legacy_unverifiable",
                                   "message": f"{ver['n_identity_missing']}/{ver['n_rows']} judge-log rows of this declared historical run carry no "
                                              f"per-row prompt hash and cannot be re-parsed by the frozen parser (its parser is not in this "
                                              f"repository): their status and proposals are self-reported and NOT independently verified"})
    report["ok"] = not report["errors"]
    return report
