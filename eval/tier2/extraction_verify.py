"""Independent verification of a Tier-2 extraction log (configs A and B; evaluator side, no model, no gold).

A and B store an LLM extraction of each user turn instead of its surface text, so (as for the judge log) the log's own ``status`` and
``stored_text`` are not trusted: every ``raw_output`` is re-parsed with the strict parser of the declared prompt and the result is
compared with what the log says was stored.

Parser per config (the only parsers used):
* A -- ``dkmem.memory.extract.parse_model_output`` with ``MEM0_EXTRACTION_V4`` (exactly the JSON object with gloss / distinction / surface;
  ``surface`` must equal the turn text); the stored text is the ``gloss``.
* B -- ``dkmem.memory.native_extract.parse_fact_output`` (exactly ``{"fact": <non-empty string>}``); the stored text is the fact.

Log row (one per user turn, processing order): ``conversation_id, turn_id, session_index, status (ok | failed), prompt_id, prompt_sha256,
raw_output, error, stored_text`` (``stored_text`` is null for a failed turn: no entry is written).
"""
from __future__ import annotations

PARSER_NAMES = {"A": "dkmem.memory.extract.parse_model_output", "B": "dkmem.memory.native_extract.parse_fact_output"}


def expected_prompt(cfg: str):
    if cfg == "A":
        from dkmem.memory.prompts import MEM0_EXTRACTION_V4
        return MEM0_EXTRACTION_V4
    if cfg == "B":
        from dkmem.memory.native_extract import NATIVE_B_V1
        return NATIVE_B_V1
    raise ValueError(f"config {cfg!r} has no extraction step")


def reparse(cfg: str, raw_output, utterance: str, prompt) -> tuple[bool, str | None]:
    """``(parsed ok, stored text)`` of one raw extraction under the strict parser of ``cfg``."""
    if cfg == "A":
        from dkmem.memory.extract import ExtractionError, parse_model_output
        try:
            gloss, _distinction, _surface = parse_model_output(raw_output, utterance=utterance, prompt=prompt)
        except ExtractionError:
            return False, None
        return True, gloss
    from dkmem.memory.native_extract import NativeExtractionError, parse_fact_output
    try:
        return True, parse_fact_output(raw_output)
    except NativeExtractionError:
        return False, None


def verify_extraction_log(cfg: str, manifest: dict, rows: list[dict], user_texts: dict[str, str]) -> dict:
    """Problems of an extraction log against its manifest and the turns it must cover (``problems``: code -> number of rows/fields)."""
    prompt = expected_prompt(cfg)
    problems: dict[str, int] = {}
    first: dict[str, int] = {}

    def bad(code, i=-1, n=1):
        problems[code] = problems.get(code, 0) + n
        first.setdefault(code, i)

    if manifest.get("extraction_prompt_id") != prompt.prompt_id or manifest.get("extraction_prompt_sha256") != prompt.sha256:
        bad("manifest_prompt")
    seen: dict[str, int] = {}
    n_ok = n_failed = 0
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or "turn_id" not in r:
            bad("row_not_a_turn_row", i)
            continue
        tid = r["turn_id"]
        if tid not in user_texts:
            bad("unknown_turn", i)
            continue
        if tid in seen:
            bad("duplicate_turn", i)
        seen[tid] = i
        if r.get("prompt_id") != prompt.prompt_id or r.get("prompt_sha256") != prompt.sha256:
            bad("row_prompt", i)
        if r.get("status") not in ("ok", "failed"):
            bad("row_status_value", i)
            continue
        ok, text = reparse(cfg, r.get("raw_output"), user_texts[tid], prompt)
        n_ok, n_failed = n_ok + ok, n_failed + (not ok)
        if ok != (r["status"] == "ok"):
            bad("row_status_mismatch", i)
        elif ok and r.get("stored_text") != text:
            bad("row_stored_text_mismatch", i)
        elif not ok and r.get("stored_text") is not None:
            bad("row_failed_has_stored_text", i)
    missing = set(user_texts) - set(seen)
    if missing:
        bad("turn_without_row", -1, len(missing))
    if manifest.get("extraction_calls") != len(rows):
        bad("manifest_call_count")
    if manifest.get("extraction_failures") != n_failed:
        bad("manifest_failure_count")
    return {"status": "failed" if problems else "verified", "parser": PARSER_NAMES[cfg], "n_rows": len(rows), "n_reparsed_ok": n_ok,
            "n_reparsed_failed": n_failed, "problems": problems, "first_bad_row": first}
