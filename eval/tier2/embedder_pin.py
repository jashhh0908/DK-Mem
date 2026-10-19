"""The registered bge-m3 revision and the check that a run manifest recorded exactly that revision (evaluator side; no model, no gold).

Why a separate module. ``harness.similarity.is_real_bge_m3_run`` (vendored byte-for-byte, see ``eval/harness/VENDORED.json``) accepts any
manifest whose ``similarity_model_revision`` is merely non-empty, so a run made with other weights would count as "the real bge-m3". It
cannot be edited without breaking the vendoring pin, and the revision default is repeated as a literal in several launchers. This module is
the one registered value that evaluator-side checks compare against; ``tests/test_embedder_pin.py`` fails if a launcher's default drifts.

A Tier-2 run manifest records, for the embedder that ranked candidates (and, in Config D, decided merges):
``similarity_model_revision`` (the resolved commit hash) and ``similarity_requested_revision`` (what the launcher asked for).
"""
from __future__ import annotations

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"     # DKMEM_FIRST_RUN.md; every stored run manifest records it

OK = "ok"
NOT_APPLICABLE = "not_applicable"
MISSING = "missing"
WRONG_REVISION = "wrong_revision"
WRONG_MODEL = "wrong_model"


def verify_manifest_embedder(manifest: dict, *, expected: str = BGE_M3_REVISION) -> dict:
    """Compare the embedder a manifest recorded with the registered one.

    ``status``:
    * ``ok``             -- model id is bge-m3 and the resolved revision equals ``expected`` (a recorded request, if any, equals it too);
    * ``not_applicable`` -- the run declared a placeholder similarity (``similarity_is_placeholder`` true, e.g. difflib): no embedder;
    * ``missing``        -- the manifest claims bge-m3 (or declares no placeholder) but records no resolved revision;
    * ``wrong_revision`` -- a different resolved revision, or a request for a different revision;
    * ``wrong_model``    -- a model id other than bge-m3.
    The recorded values are returned so a failure can be reported, never just compared silently.
    """
    recorded = manifest.get("similarity_model_revision")
    requested = manifest.get("similarity_requested_revision")
    model = manifest.get("similarity_model_id")
    out = {"expected": expected, "recorded": recorded, "requested": requested, "model_id": model}
    if manifest.get("similarity_is_placeholder") is True:
        return {**out, "status": NOT_APPLICABLE}
    if model is not None and model != BGE_M3_MODEL_ID:
        return {**out, "status": WRONG_MODEL}
    if not recorded:
        return {**out, "status": MISSING}
    if recorded != expected or (requested is not None and requested != expected):
        return {**out, "status": WRONG_REVISION}
    return {**out, "status": OK}
