"""Real embedding similarity for config D: raw cosine between L2-normalised bge-m3 dense embeddings.

* The score is the RAW cosine in [-1, 1]; it is never mapped to [0, 1] and never clipped (see `_check`). tau is
  therefore a raw cosine threshold, and the host merges iff similarity > tau (strict).
* ipdV2's `SimilarityMetric` only accepts [0, 1], so this metric is a separate duck-typed class with a declared
  `value_range`; ipdV2 is not modified. Runs using it declare `similarity_range: [-1.0, 1.0]` in the manifest, which
  makes the evaluator/writer relax only the two schema bounds that are [0, 1] in ipdV2's schema (see
  `evaluator.loaders.relax_pairwise_schema`).
* Recipe: the project has no documented bge-m3 recipe, so the model's own dense-embedding recipe is used and reported:
  last-layer CLS token, no extra projection, then L2 normalisation (applied here, in float64), cosine = dot product.
* Deterministic batching: the distinct texts are sorted and embedded in fixed-size batches in that order, so the same
  text set always forms the same batches. Embeddings are cached by the exact input text; each distinct text is embedded
  once per run.
"""
from __future__ import annotations

import math
from typing import Callable, Protocol, Sequence

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_METRIC_NAME = "bge-m3_dense_cosine_v1"
POOLING = "cls (last_hidden_state[:, 0]), no projection"
NORMALIZATION = "l2 (float64, applied by the harness before the dot product)"
COSINE_RANGE = (-1.0, 1.0)
SNAP_TOL = 1e-6        # float noise only: a cosine within 1e-6 beyond +-1 is snapped to +-1; anything else raises


class SimilarityRangeError(ValueError):
    pass


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...      # one vector per text, same order
    def describe(self) -> dict: ...


def _unit(vec: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if not math.isfinite(norm) or norm == 0.0:
        raise SimilarityRangeError("embedding has zero or non-finite norm")
    return tuple(x / norm for x in vec)


class CosineMetric:
    """name + (text_a, text_b) -> raw cosine in [-1, 1], with an exact-text embedding cache."""
    value_range = COSINE_RANGE

    def __init__(self, name: str, embedder: Embedder):
        self.name, self.embedder = name, embedder
        self._cache: dict[str, tuple[float, ...]] = {}
        self.hits = 0
        self.embedded = 0          # distinct texts sent to the embedder
        self.embed_calls = 0       # embedder.embed() invocations
        self.pairs_scored = 0

    # -- embedding ---------------------------------------------------------------------------------------------
    def prepare(self, texts: Sequence[str]) -> None:
        """Embed every not-yet-cached distinct text, sorted, in one deterministic pass."""
        todo = sorted({t for t in texts if t not in self._cache})
        if not todo:
            return
        vecs = self.embedder.embed(todo)
        if len(vecs) != len(todo):
            raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(todo)} texts")
        self.embed_calls += 1
        for t, v in zip(todo, vecs):
            self._cache[t] = _unit(v)
        self.embedded += len(todo)

    def _vec(self, text: str) -> tuple[float, ...]:
        if text in self._cache:
            self.hits += 1
        else:
            self.prepare([text])          # on-demand fallback (a pass of one); cached afterwards
        return self._cache[text]

    # -- scoring -----------------------------------------------------------------------------------------------
    def __call__(self, text_a: str, text_b: str) -> float:
        a, b = self._vec(text_a), self._vec(text_b)
        score = math.fsum(x * y for x, y in zip(a, b))      # commutative, so sim(a, b) == sim(b, a) exactly
        self.pairs_scored += 1
        return self._check(score)

    def _check(self, score: float) -> float:
        lo, hi = self.value_range
        if not math.isfinite(score) or score < lo - SNAP_TOL or score > hi + SNAP_TOL:
            raise SimilarityRangeError(f"metric {self.name!r} produced {score!r}, outside [{lo}, {hi}]")
        return min(hi, max(lo, score))                      # only float noise (<= SNAP_TOL) can be affected here

    # -- provenance --------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        return {"similarity_metric": self.name, **self.embedder.describe(),
                "similarity_range": list(self.value_range), "similarity_pooling": POOLING,
                "similarity_normalization": NORMALIZATION,
                "similarity_cache": "exact-text cache; each distinct text is embedded once per run",
                "similarity_batching": "distinct texts sorted; fixed-size batches in that order"}

    def cache_stats(self) -> dict:
        return {"distinct_texts_embedded": self.embedded, "cache_hits": self.hits, "embed_calls": self.embed_calls,
                "pairs_scored": self.pairs_scored}


class BgeM3Embedder:
    """BAAI/bge-m3 dense embeddings via plain transformers (CLS pooling). Heavy imports are lazy: nothing is
    downloaded or loaded until the first `embed` call."""

    def __init__(self, model_id: str = BGE_M3_MODEL_ID, revision: str | None = None, dtype: str = "float16",
                 device: str = "cuda:0", batch_size: int = 32, max_length: int = 512):
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be >= 1")
        self.model_id, self.revision, self.dtype, self.device = model_id, revision, dtype, device
        self.batch_size, self.max_length = batch_size, max_length
        self._tok = self._model = None
        self._resolved_revision = None

    def _load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        new = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)    # same switch as ipdV2's HFGenerator
        kw = {"dtype" if new else "torch_dtype": getattr(torch, self.dtype)}
        self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModel.from_pretrained(self.model_id, revision=self.revision, **kw).to(self.device).eval()
        self._resolved_revision = getattr(getattr(self._model, "config", None), "_commit_hash", None)
        if not (self._resolved_revision or self.revision):
            raise RuntimeError("cannot determine the exact bge-m3 revision; refusing to run without recording it")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            self._load()
        import torch
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            out.extend(hidden[:, 0].float().cpu().tolist())
        return out

    def describe(self) -> dict:
        import torch
        import transformers
        return {"similarity_backend": "bge-m3", "similarity_model_id": self.model_id,
                "similarity_model_revision": self._resolved_revision or self.revision,
                "similarity_requested_revision": self.revision, "similarity_dtype": self.dtype,
                "similarity_device": self.device, "similarity_max_seq_length": self.max_length,
                "similarity_embed_batch_size": self.batch_size, "similarity_is_placeholder": False,
                "similarity_torch_version": torch.__version__, "similarity_transformers_version": transformers.__version__}


def is_real_bge_m3_run(manifest: dict) -> bool:
    """True only for a run whose manifest says it used the real bge-m3 backend. A difflib (placeholder) D run, or a
    manifest that merely mentions bge-m3, is not one."""
    return (manifest.get("pipeline_config") == "D"
            and manifest.get("similarity_backend") == "bge-m3"
            and manifest.get("similarity_model_id") == BGE_M3_MODEL_ID
            and manifest.get("similarity_metric") == BGE_M3_METRIC_NAME
            and manifest.get("similarity_is_placeholder") is False
            and manifest.get("similarity_range") == list(COSINE_RANGE)
            and bool(manifest.get("similarity_model_revision")))


def build_metric(embedder_factory: Callable[[], Embedder] | None = None, **embedder_kw) -> CosineMetric:
    """The bge-m3 metric (embedder_factory is for tests, which must not load a model)."""
    emb = embedder_factory() if embedder_factory else BgeM3Embedder(**embedder_kw)
    return CosineMetric(BGE_M3_METRIC_NAME, emb)
