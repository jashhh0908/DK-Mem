"""Read-only loaders for the gold answer key, pairwise_eval.jsonl and run_manifest.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JsonlResult:
    records: list[dict] = field(default_factory=list)
    # (1-based line number, error text) for lines that are not a JSON object
    malformed: list[tuple[int, str]] = field(default_factory=list)


def load_jsonl(path: str | Path) -> JsonlResult:
    """Load a UTF-8 JSONL file. Bad lines are collected, never raised."""
    out = JsonlResult()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                out.malformed.append((n, f"invalid JSON: {e.msg}"))
                continue
            if not isinstance(obj, dict):
                out.malformed.append((n, f"expected JSON object, got {type(obj).__name__}"))
                continue
            out.records.append(obj)
    return out


def load_gold(path: str | Path) -> JsonlResult:
    return load_jsonl(path)


def load_pairwise(path: str | Path) -> JsonlResult:
    return load_jsonl(path)


def load_manifest(path: str | Path) -> dict:
    """Load run_manifest.json; raises on unreadable/invalid JSON (caller reports it)."""
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError("run_manifest.json must contain a JSON object")
    return obj


RAW_COSINE_RANGE = [-1.0, 1.0]


def relax_pairwise_schema(schema: dict, manifest: dict | None) -> dict:
    """ipdV2's pairwise schema bounds similarity_score and threshold to [0, 1]. A run whose manifest declares
    `similarity_range: [-1.0, 1.0]` (raw cosine, e.g. config D with bge-m3) gets a copy with ONLY those two lower
    bounds set to -1. Every other schema rule, and every run that does not declare the range, is unchanged."""
    if not manifest or manifest.get("similarity_range") != RAW_COSINE_RANGE:
        return schema
    import copy
    relaxed = copy.deepcopy(schema)
    for key in ("similarity_score", "threshold"):
        relaxed["properties"][key]["minimum"] = -1
    return relaxed


def load_schema(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
