"""Minimal draft-07 validator for the subset used by Team B's schemas.

Supports: type (incl. unions), required, properties, enum, minimum/maximum,
$ref to #/definitions, if/then/else, format date-time. Exists because the
`jsonschema` package is not installed in this workspace; it reads the real
schema files, so the schemas stay the single source of truth.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
}


def validate(instance: Any, schema: dict, root: dict | None = None, path: str = "") -> list[tuple[str, str]]:
    """Return a list of (path, message) violations; empty means valid."""
    root = root if root is not None else schema
    errs: list[tuple[str, str]] = []
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return validate(instance, node, root, path)
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](instance) for x in types):
            return [(path, f"expected type {types}, got {type(instance).__name__}")]
    if "enum" in schema and instance not in schema["enum"]:
        errs.append((path, f"{instance!r} not in enum {schema['enum']}"))
    if _TYPES["number"](instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append((path, f"{instance} < minimum {schema['minimum']}"))
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append((path, f"{instance} > maximum {schema['maximum']}"))
    if schema.get("format") == "date-time" and isinstance(instance, str):
        try:
            datetime.fromisoformat(instance.replace("Z", "+00:00"))
        except ValueError:
            errs.append((path, f"{instance!r} is not an RFC 3339 date-time"))
    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errs.append((f"{path}.{name}".lstrip("."), "required field missing"))
        for name, sub in schema.get("properties", {}).items():
            if name in instance:
                errs += validate(instance[name], sub, root, f"{path}.{name}".lstrip("."))
    if "if" in schema:
        branch = "then" if not validate(instance, schema["if"], root, path) else "else"
        if branch in schema:
            errs += validate(instance, schema[branch], root, path)
    return errs
