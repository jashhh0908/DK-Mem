"""Evidence-based classification of one utterance's extraction outcome (pure functions, no model).

Inputs are what the legacy path really produced: the raw model output string and, if the
legacy parser rejected it, the exception type and message. The messages matched here are
the literal strings raised by dkmem.memory.extract.parse_model_output in the legacy snapshot.
Anything not recognised is reported as `extraction_exception` with the message kept, never
guessed into a nicer category.
"""
from __future__ import annotations

import re

OK = "ok"
MODEL_CALL_FAILURE = "model_call_failure"
MODEL_EMPTY_OUTPUT = "model_empty_output"
INVALID_JSON = "invalid_json"
SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"
EXTRACTION_EXCEPTION = "extraction_exception"
UNKNOWN = "unknown"

_SCHEMA_PATTERNS = [
    (r"^output must be a JSON object", "not_a_json_object"),
    (r"^output keys must be", "wrong_keys"),
    (r"^'gloss' must be", "gloss_invalid"),
    (r"^'distinction' must be an object", "distinction_not_object"),
    (r"^'distinction' has unknown keys", "distinction_unknown_key"),
    (r"^'distinction\.[^']*' must be a non-empty string", "distinction_value_not_nonempty_string"),
    (r"^'distinction\.[^']*' must be one of", "distinction_closed_value_violation"),
    (r"^'surface' must be a string", "surface_not_string"),
    (r"^'surface' must equal the input utterance", "surface_mismatch"),
    (r"^duplicate JSON keys", "duplicate_json_keys"),
]


def classify_json_error(raw: str, message: str) -> str:
    """Sub-category for output the JSON parser rejected."""
    if raw.lstrip().startswith("```"):
        return "code_fenced"
    if "Extra data" in message:
        return "extra_text_after_json"
    if not raw.rstrip().endswith("}"):
        return "incomplete_or_truncated"
    return "other_json_syntax"


def classify_output(raw_output: object, error_type: str | None, error_message: str | None) -> tuple[str, str | None]:
    """Return (category, subcategory) for one utterance.

    `error_type`/`error_message` are None when the legacy parser accepted the output.
    """
    if error_type is None:
        return OK, None
    if isinstance(raw_output, str) and not raw_output.strip():
        return MODEL_EMPTY_OUTPUT, None
    msg = error_message or ""
    if msg.startswith("output is not a single JSON object"):
        return INVALID_JSON, classify_json_error(raw_output if isinstance(raw_output, str) else "", msg)
    for pattern, sub in _SCHEMA_PATTERNS:
        if re.match(pattern, msg):
            return SCHEMA_VALIDATION_FAILURE, sub
    return EXTRACTION_EXCEPTION, f"{error_type}: {msg[:200]}"


def classify_episode(rows_produced: int, model_error: str | None, side_categories: dict[str, str]) -> dict:
    """Final reason an episode produced zero entries, from the per-side evidence."""
    if rows_produced > 0:
        return {"final": "entries_produced", "failing_sides": [], "dropped_valid_side": False}
    if model_error is not None:
        return {"final": MODEL_CALL_FAILURE, "failing_sides": [], "dropped_valid_side": False}
    failing = [s for s, c in sorted(side_categories.items()) if c != OK]
    if not failing:
        # both sides parsed fine yet no records: nothing in the evidence explains it
        return {"final": UNKNOWN, "failing_sides": [], "dropped_valid_side": False}
    return {
        "final": side_categories[failing[0]],
        "failing_sides": failing,
        # the legacy runner drops BOTH sides when either fails, so a valid side is lost with its partner
        "dropped_valid_side": len(failing) < len(side_categories),
    }
