"""Host merge mechanisms (the thing the DK-Mem gate sits in front of).

A host looks at the two stored entries of an episode and returns a `HostOutcome`:
the raw similarity (always reported), a proposal in dkmem's internal vocabulary
("merge" / "keep_both"), and the threshold to record. `proposal=None` means the host produced no
usable decision (only the LLM judge can do that); such an episode gets NO pairwise record (the
contract's "not compared") and the reason is kept in `failure`, never silently converted into a decision.

Embedding-threshold host (config D): propose a merge iff similarity_score > tau (strict, per the
contract/spec; ipdV2's standalone gate() uses >=, which only matters at sim == tau).
The LLM judge host (config C) lives in `harness.judge`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import _dkmem  # noqa: F401
from dkmem.memory.schema import Extraction
from dkmem.memory.similarity import DEFAULT_SIMILARITY, SimilarityMetric, entry_similarity


@dataclass(frozen=True)
class HostOutcome:
    similarity: float                 # raw, pre-gating score for pairwise_eval.similarity_score
    proposal: str | None              # "merge" | "keep_both" | None (no usable decision)
    threshold: float                  # value for pairwise_eval.threshold
    failure: dict | None = field(default=None)


@dataclass(frozen=True)
class EmbeddingThresholdHost:
    tau: float
    metric: SimilarityMetric = DEFAULT_SIMILARITY    # difflib placeholder by default; harness.similarity.CosineMetric = real bge-m3
    text_field: str = "surface"                      # config D stores the surface text
    mechanism = "embedding_threshold"

    def __post_init__(self) -> None:
        lo, hi = getattr(self.metric, "value_range", (0.0, 1.0))      # ipdV2 metrics are [0, 1]; raw cosine is [-1, 1]
        if not lo <= self.tau <= hi:
            raise ValueError(f"tau must be in [{lo:g}, {hi:g}] for metric {self.metric.name!r}, got {self.tau!r}")

    def similarity(self, a: Extraction, b: Extraction) -> float:
        """Raw, pre-gating score (reported unchanged in pairwise_eval)."""
        return entry_similarity(a, b, self.metric, self.text_field)

    def propose(self, sim: float) -> str:
        """Internal decision vocabulary of dkmem.memory.schema.DECISIONS."""
        return "merge" if sim > self.tau else "keep_both"

    def decide_many(self, pairs: Sequence[tuple[Extraction, Extraction]]) -> list[HostOutcome]:
        out = []
        for a, b in pairs:
            sim = self.similarity(a, b)
            out.append(HostOutcome(sim, self.propose(sim), self.tau))
        return out
