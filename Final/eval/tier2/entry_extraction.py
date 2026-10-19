"""Write-side extraction of user turns for Tier-2 configs A and B (SYSTEM side: reads only turn text and language).

Config A stores the English gloss of ``dkmem.memory.prompts.MEM0_EXTRACTION_V4`` (strict ``extract_many``, as in Tier 1); config B stores the
fact of ``dkmem.memory.native_extract.NATIVE_B_V1`` (strict ``native_fact_extract_many``). Configs C and D store the surface text and use
``harness.extractors.extract_verbatim`` instead.

* One batched call per conversation: extraction depends only on the turn text, never on the store, so batching changes nothing a store
  could observe (the same convention as Tier 1, which batches every utterance).
* A turn whose output violates the strict contract is a FAILED extraction: no entry is written for it, nothing is repaired or retried, and
  the raw output is kept in the extraction log. It is never turned into a decision.
* DK-Mem gate: mode ``lexicon`` takes the distinction from the lexicon on the UTTERANCE (as Tier 1 does), mode ``off`` records none (the model's
  own distinction, A only, is discarded in both modes: the gate never consults the model on Tier 2).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from harness import _dkmem  # noqa: F401  (puts dkmem on sys.path; read-only)
from dkmem.backends.llm import GenerationParams
from dkmem.memory.extract import ExtractionBatchError, TextGenerator, derive_lang_profile, extract_many
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.native_extract import NATIVE_B_V1, native_fact_extract_many
from dkmem.memory.prompts import MEM0_EXTRACTION_V4
from dkmem.memory.schema import Extraction, ProbeItem
from dkmem.tier1.extraction import extract_lexicon_only

EXTRACTION_CONFIGS = ("A", "B")
PROMPTS = {"A": MEM0_EXTRACTION_V4, "B": NATIVE_B_V1}


@dataclass(frozen=True)
class TurnExtraction:
    """The extraction of one user turn: ``extraction`` is None (and ``ok`` False) when it failed."""

    turn_id: str
    ok: bool
    stored_text: str | None
    extraction: Extraction | None
    raw_output: str | None
    error: str | None


def _probe(turn: dict) -> ProbeItem:
    # shaped only to satisfy extract_many: it reads pair_id / lang / utt_a; the other fields are inert placeholders (as dkmem.tier1.runner)
    return ProbeItem(pair_id=turn["turn_id"], utt_a=turn["text"], utt_b=turn["text"], lang=turn["language"], distinction_class="tier2_unused",
                     distinction_value_a=None, distinction_value_b=None, gold_same_entity=False, retrieval_query="")


class TurnExtractor:
    def __init__(self, config: str, generator: TextGenerator, params: GenerationParams, lexicon: Lexicon | None, seed: int = 0) -> None:
        if config not in EXTRACTION_CONFIGS:
            raise ValueError(f"only configs {EXTRACTION_CONFIGS} extract with a model, got {config!r}")
        self.config, self.generator, self.params, self.lexicon, self.seed = config, generator, params, lexicon, seed
        self.prompt = PROMPTS[config]
        self.log: list[dict] = []          # one row per extracted turn, in order (the extraction log of the run)

    def _distinction(self, turn: dict) -> dict:
        if self.lexicon is None:
            return {}
        return dict(extract_lexicon_only(turn["turn_id"], "a", turn["text"], turn["language"], self.lexicon, seed=self.seed).distinction)

    def extract_turns(self, turns: Sequence[dict], conversation_id: str | None = None) -> dict[str, TurnExtraction]:
        """``turns``: ``[{"turn_id", "text", "language", "session_index"?}]`` -> ``{turn_id: TurnExtraction}`` in the given order."""
        turns = list(turns)
        out: dict[str, TurnExtraction] = {}
        if not turns:
            return out
        if self.config == "A":
            items = [(_probe(t), "a") for t in turns]
            failures: dict[str, Exception] = {}
            try:
                exts = extract_many(items, self.generator, self.params, prompt=self.prompt)
            except ExtractionBatchError as e:              # per-item results survive; any other ExtractionError propagates
                exts = e.extractions
                failures = {f.pair_id: f for f in e.failures}
            for t, ext in zip(turns, exts):
                if ext is None:
                    f = failures[t["turn_id"]]
                    out[t["turn_id"]] = TurnExtraction(t["turn_id"], False, None, None, f.raw_output, str(f))
                else:
                    out[t["turn_id"]] = TurnExtraction(t["turn_id"], True, ext.gloss, replace(ext, distinction=self._distinction(t)), ext.raw_output, None)
        else:
            natives = native_fact_extract_many([t["text"] for t in turns], self.generator, self.params, prompt=self.prompt)
            for t, nat in zip(turns, natives):
                if not nat.ok:
                    out[t["turn_id"]] = TurnExtraction(t["turn_id"], False, None, None, nat.raw_output, nat.error)
                    continue
                fact = nat.facts[0]
                ext = Extraction(pair_id=t["turn_id"], side="a", raw_output=nat.raw_output, gloss=fact, distinction=self._distinction(t), surface=t["text"],
                                 lang_profile=derive_lang_profile(t["language"], t["text"]), backbone=getattr(self.generator, "backbone", "unknown"),
                                 prompt_id=self.prompt.prompt_id, seed=self.seed)
                out[t["turn_id"]] = TurnExtraction(t["turn_id"], True, fact, ext, nat.raw_output, None)
        for t in turns:
            r = out[t["turn_id"]]
            self.log.append({"conversation_id": conversation_id, "turn_id": t["turn_id"], "session_index": t.get("session_index"),
                             "status": "ok" if r.ok else "failed", "prompt_id": self.prompt.prompt_id, "prompt_sha256": self.prompt.sha256,
                             "raw_output": r.raw_output, "error": r.error, "stored_text": r.stored_text})
        return out
