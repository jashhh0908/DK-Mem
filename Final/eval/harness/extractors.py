"""Write-side extraction for the harness configurations.

* Verbatim (configs C, D): no LLM, storage = the surface text. Reuses ipdV2 unchanged: with the gate ON the
  distinction comes from `dkmem.tier1.extraction.extract_lexicon_only`; with the gate OFF the same verbatim
  entry is built with an empty distinction.
* English-forced (config A): ipdV2's own `extract_many` with the English-gloss prompt (`mem0_extraction_v4`),
  unchanged and strict: nothing is repaired or lenient. A failed utterance yields no extraction; the episode
  then yields no pairwise record. Every utterance (ok or failed) is logged with its raw model output.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from . import _dkmem  # noqa: F401
from dkmem.backends.llm import GenerationParams
from dkmem.memory.dkmem_extract import apply_lexicon
from dkmem.memory.extract import ExtractionBatchError, TextGenerator, derive_lang_profile, extract_many
from dkmem.memory.lexicon import Lexicon
from dkmem.memory.prompts import MEM0_EXTRACTION_V4, PromptTemplate
from dkmem.memory.schema import Extraction
from dkmem.tier1.extraction import NO_BACKBONE, extract_lexicon_only
from dkmem.tier1.io import Tier1Record
from dkmem.tier1.runner import _probe_item_adapter

VERBATIM_PROMPT_ID = "verbatim_no_llm_v1"
ENGLISH_FORCED_PROMPT = MEM0_EXTRACTION_V4     # config A: facts normalised to an English gloss


def _load_classifier():
    """Reuse the Phase 4A failure classifier (pure functions) instead of duplicating it."""
    path = Path(__file__).resolve().parents[1] / "diagnostics" / "no_entry" / "classify.py"
    spec = importlib.util.spec_from_file_location("_no_entry_classify", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_output


classify_output = _load_classifier()


def extract_verbatim(pair_id: str, side: str, utterance: str, lang: str, *, seed: int,
                     lexicon: Lexicon | None = None) -> Extraction:
    """One verbatim entry. `lexicon=None` -> no distinction (DK-Mem off)."""
    if lexicon is not None:
        return extract_lexicon_only(pair_id, side, utterance, lang, lexicon, seed=seed)
    return Extraction(pair_id=pair_id, side=side, raw_output="", gloss=utterance, distinction={},
                      surface=utterance, lang_profile=derive_lang_profile(lang, utterance),
                      backbone=NO_BACKBONE, prompt_id=VERBATIM_PROMPT_ID, seed=seed)


@dataclass
class ExtractionResult:
    pairs: list                    # per record: (ext_a, ext_b) or None if either side failed
    log: list[dict]                # one row per utterance (ok and failed), with the raw model output
    failed_utterances: int
    failed_episodes: list[dict] = field(default_factory=list)


@dataclass
class EnglishForcedExtractor:
    """Config A extraction. `dkmem_mode`: off | lexicon (lexicon distinctions only) | lexicon+llm (lexicon first,
    model second, i.e. ipdV2's dkmem_extract overlay)."""
    generator: TextGenerator
    params: GenerationParams
    prompt: PromptTemplate = ENGLISH_FORCED_PROMPT
    lexicon: Lexicon | None = None
    dkmem_mode: str = "off"
    kind = "english_forced"

    def _validation_prompt(self) -> PromptTemplate:
        """The prompt handed to ipdV2's `extract_many` (renders the messages and picks the validation rules)."""
        return self.prompt

    def _restamp(self, ext: Extraction) -> Extraction:
        """Hook for subclasses whose `_validation_prompt` carries a different id from `self.prompt` (identity here)."""
        return ext

    def _distinction(self, ext: Extraction, lang: str) -> Extraction:
        if self.dkmem_mode == "lexicon+llm":
            return apply_lexicon(ext, self.lexicon, lang)                       # unchanged ipdV2 cascade
        if self.dkmem_mode == "lexicon":                                        # model's own distinction is discarded
            return replace(ext, distinction=extract_lexicon_only(ext.pair_id, ext.side, ext.surface, lang, self.lexicon,
                                                                 seed=ext.seed).distinction)
        return ext

    def extract_all(self, records: Sequence[Tier1Record]) -> ExtractionResult:
        """One batched, strict extraction over all utterances (a, b per record, file order)."""
        probes = [_probe_item_adapter(r) for r in records]
        items = [(p, side) for p in probes for side in ("a", "b")]
        failures: dict[tuple[str, str], Exception] = {}
        try:
            exts = extract_many(items, self.generator, self.params, prompt=self._validation_prompt())
        except ExtractionBatchError as e:                 # per-item results survive; other ExtractionErrors propagate
            exts = e.extractions
            failures = {(f.pair_id, f.side): f for f in e.failures}
        exts = [None if x is None else self._restamp(x) for x in exts]
        log, pairs, failed_eps, n_failed = [], [], [], 0
        for i, rec in enumerate(records):
            sides = {}
            for j, side in enumerate(("a", "b")):
                ext = exts[2 * i + j]
                uid = getattr(rec, f"utterance_{side}_id")
                row = {"eval_pair_id": rec.eval_pair_id, "utterance_id": uid, "side": side, "language": rec.language,
                       "utterance": getattr(rec, f"utterance_{side}"), "prompt_id": self.prompt.prompt_id}
                if ext is not None:
                    row.update(status="ok", raw_output=ext.raw_output, exception_type=None, exception_message=None,
                               category="ok", subcategory=None)
                    sides[side] = self._distinction(ext, rec.language)
                else:
                    f = failures[(rec.eval_pair_id, side)]
                    msg = str(f).removeprefix(f"{rec.eval_pair_id}/{side}: ")
                    cat, sub = classify_output(f.raw_output, type(f).__name__, msg)
                    row.update(status="failed", raw_output=f.raw_output, exception_type=type(f).__name__,
                               exception_message=msg, category=cat, subcategory=sub)
                    n_failed += 1
                log.append(row)
            if len(sides) == 2:
                pairs.append((sides["a"], sides["b"]))
            else:
                pairs.append(None)
                bad = [r for r in log[-2:] if r["status"] == "failed"]
                failed_eps.append({"eval_pair_id": rec.eval_pair_id, "stage": "extraction", "reason": "extraction_failed",
                                   "failed_sides": [r["side"] for r in bad],
                                   "categories": {r["side"]: [r["category"], r["subcategory"]] for r in bad},
                                   "valid_side_dropped": len(bad) == 1})
        return ExtractionResult(pairs, log, n_failed, failed_eps)
