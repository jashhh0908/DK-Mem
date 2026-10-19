"""Config C merge-judge host for the Tier-2 runner: the frozen Tier-1 ``merge_judge_v1`` (from ``dkmem``) or the frozen Tier-2
identity judges ``merge_judge_entity_t2_v1`` / ``_v2`` (``judge_entity.py``); nothing else is accepted.

Why this module exists. The first Tier-2 full run imported its judge from a separate ``harness.judge`` module (shipped in
the Kaggle dataset, not in this repository). That module defined its own ``MERGE_JUDGE_V1`` -- same ``prompt_id``, but a
different prompt (sha256 ``f0dc9cf7...`` instead of ``3855ef65...``, and a different answer vocabulary: it answered
``{"decision": "different"}`` where the frozen prompt answers ``{"decision": "merge" | "keep_both", "reason": ...}``). The
run therefore did not test the Tier-1 judge: 3105/3105 answers were "different".

This host takes the prompt from ``dkmem.memory.judge`` (the only definition), refuses any other prompt, renders the
messages with ``dkmem.memory.judge.render_judge_prompt`` and parses with the frozen strict parser. It keeps the interface
the runner already used (``decide_many``, ``log``) and imports nothing outside ``dkmem``. A failed judgement is a
``proposal`` of ``None`` and is never turned into a decision.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dkmem.backends.llm import GenerationParams
from dkmem.memory.judge import MERGE_JUDGE_V1, judge_pairs, render_judge_prompt

from .frozen_judge import (  # noqa: F401  (re-exported: the runner and the tests read them from here)
    FROZEN_JUDGE_MAX_NEW_TOKENS, FROZEN_JUDGE_PROMPT_ID, FROZEN_JUDGE_PROMPT_SHA256, JUDGE_ROW_THRESHOLD, TIER2_JUDGES,
)
from .judge_entity import MERGE_JUDGE_ENTITY_T2_V1, MERGE_JUDGE_ENTITY_T2_V2


def check_frozen_prompt(prompt) -> None:
    """Raise unless ``prompt`` is exactly one of the two frozen Config C judges (the object itself, its id and the sha256 of its
    exact text): the Tier-1 ``merge_judge_v1`` or the Tier-2 identity judge ``merge_judge_entity_t2_v1``."""
    if not any(prompt is p for p in (MERGE_JUDGE_V1, MERGE_JUDGE_ENTITY_T2_V1, MERGE_JUDGE_ENTITY_T2_V2)):
        raise ValueError("Config C must use dkmem.memory.judge.MERGE_JUDGE_V1 or a tier2.judge_entity identity judge itself, "
                         "not another prompt object")
    want = TIER2_JUDGES.get(prompt.prompt_id)
    if want is None or prompt.sha256 != want:
        raise ValueError(f"merge judge is not a frozen Config C prompt: {prompt.prompt_id} {prompt.sha256} (expected one of {TIER2_JUDGES})")


JUDGES_BY_ID = {p.prompt_id: p for p in (MERGE_JUDGE_V1, MERGE_JUDGE_ENTITY_T2_V1, MERGE_JUDGE_ENTITY_T2_V2)}


class TokenMeter:
    """Token counts from the SAME tokenizer the generator uses (``HFGenerator.tokenizer``): nothing is estimated.

    ``prompt_tokens`` renders the chat messages exactly as generation does (chat template, generation prompt) and counts the
    result; ``text_tokens`` counts a decoded model output by re-encoding it without special tokens (a decode/encode round trip,
    so it equals the generated length except where merges differ; it is a recorded measurement, not a bound).
    """

    def __init__(self, tokenizer, *, use_chat_template: bool = True) -> None:
        self.tokenizer, self.use_chat_template = tokenizer, use_chat_template

    @classmethod
    def from_generator(cls, generator):
        tok = getattr(generator, "tokenizer", None)
        if tok is None:
            return None
        cfg = getattr(generator, "config", None)
        return cls(tok, use_chat_template=getattr(cfg, "use_chat_template", True))

    def text_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def prompt_tokens(self, messages) -> int:
        if self.use_chat_template and not isinstance(messages, str):
            rendered = self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            return self.text_tokens(rendered)
        return self.text_tokens(messages if isinstance(messages, str) else " ".join(m["content"] for m in messages))

    def describe(self) -> dict:
        tok = self.tokenizer
        template = getattr(tok, "chat_template", None)
        try:
            size = len(tok)
        except TypeError:
            size = None
        return {"tokenizer_class": type(tok).__name__, "name_or_path": getattr(tok, "name_or_path", None), "vocab_size": size,
                "eos_token": getattr(tok, "eos_token", None), "pad_token": getattr(tok, "pad_token", None),
                "padding_side": getattr(tok, "padding_side", None),
                "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None,
                "output_count_method": "re-encode of the decoded output, special tokens excluded"}


def summarize_counts(values: list[int]) -> dict:
    """n / min / median / p95 / max of a list of token counts (empty -> n 0)."""
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda f: v[min(len(v) - 1, int(round(f * (len(v) - 1))))]
    return {"n": len(v), "min": v[0], "median": q(0.5), "p95": q(0.95), "max": v[-1]}


@dataclass(frozen=True)
class JudgeOutcome:
    """What the runner reads per compared pair: ``proposal`` is ``merge``/``keep_both`` or ``None`` (judge failed)."""

    proposal: str | None
    similarity: float
    threshold: float | None   # None: a judge has no similarity cutoff
    raw_output: str
    reason: str | None
    error: str | None


class FrozenJudgeHost:
    """Config C host: the frozen ``merge_judge_v1`` on the verbatim surface texts of two entries."""

    mechanism = "llm_judge"

    def __init__(self, generator, params: GenerationParams | None, prompt=MERGE_JUDGE_V1, metric=None,
                 text_field: str = "surface", meter: TokenMeter | None = None) -> None:
        check_frozen_prompt(prompt)
        if metric is None:
            raise ValueError("a similarity metric is required: every compared pair stores its raw cosine")
        self.generator, self.params, self.prompt, self.metric, self.text_field = generator, params, prompt, metric, text_field
        self.meter = meter          # None: no tokenizer available (test doubles); token fields are then null
        self.log: list[dict] = []

    def decide_many(self, pairs) -> list[JudgeOutcome]:
        """One batched judge call for ``[(extraction_a, extraction_b), ...]``; one log row and one outcome per pair."""
        pairs = list(pairs)
        if not pairs:
            return []
        texts = [(getattr(a, self.text_field), getattr(b, self.text_field)) for a, b in pairs]
        self.metric.prepare([t for ab in texts for t in ab])
        results = judge_pairs(texts, self.generator, self.params, prompt=self.prompt)
        out = []
        budget = getattr(self.params, "max_new_tokens", None)
        for (a, _b), (ta, tb), r in zip(pairs, texts, results):
            proposal = r.decision if r.ok else None
            p_tok = self.meter.prompt_tokens(render_judge_prompt(ta, tb, self.prompt)) if self.meter else None
            o_tok = self.meter.text_tokens(r.raw_output) if (self.meter and isinstance(r.raw_output, str)) else None
            self.log.append({"eval_pair_id": getattr(a, "pair_id", None), "text_a": ta, "text_b": tb, "raw_output": r.raw_output,
                             "reason": r.reason, "prompt_id": self.prompt.prompt_id, "prompt_sha256": self.prompt.sha256,
                             "status": "ok" if r.ok else "parse_error", "proposal": proposal, "error": r.error,
                             # a recorded diagnostic, not a repair: a failed answer that does not end in "}" was probably cut off
                             # by the decoding budget (the raw output is kept as generated either way)
                             "possibly_truncated": (not r.ok) and not (r.raw_output or "").rstrip().endswith("}"),
                             # measured with the generator's own tokenizer; ``at_budget``: the output used the whole decoding budget
                             "prompt_tokens": p_tok, "output_tokens": o_tok, "max_new_tokens": budget,
                             "at_budget": (o_tok >= budget) if (o_tok is not None and budget) else None})
            out.append(JudgeOutcome(proposal, float(self.metric(ta, tb)), JUDGE_ROW_THRESHOLD, r.raw_output, r.reason, r.error))
        return out
