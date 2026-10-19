"""Tier-2 Config C merge judge: IDENTITY consolidation (``merge_judge_entity_t2_v1``). Tier 2 only; Tier 1 keeps ``merge_judge_v1``.

Why a second judge. The frozen Tier-1 ``merge_judge_v1`` merges only when "both entries state the same fact". Tier 1 items are
minimal pairs, so there "same fact" and "same entity" coincide. Every Tier-2 pair gives its two members different attribute
values, so under that definition no Tier-2 pair can ever be merged: the smoke test got ``keep_both`` on 135/135 pairs and Config C
was uninformative. The project measures false IDENTITY (FCR counts merged pairs of distinct entities), so the Tier-2 judge asks
the identity question instead:

  merge      - both entries are about the same specific person, thing or event, whether they state the same fact, different
               compatible facts, or conflicting / changed facts (the store keeps both original texts; this is identity
               consolidation, not contradiction resolution)
  keep_both  - different people, things or events, even if they look alike; or the only shared subject is the user; or the
               entries do not show that they are about the same specific subject

Like ``merge_judge_v1`` the prompt carries no language note and no hint about kinship terms, register or names (that is the
separate prompt-informed baseline), and its chat examples are plain English with no personal names. The output format and the
strict parser are the frozen ones from ``dkmem.memory.judge`` (``{"decision": "merge" | "keep_both", "reason": str}``).

FROZEN: never edit this text; write a new id. The sha256 is pinned in ``frozen_judge.ENTITY_JUDGE_PROMPT_SHA256`` and in
``judge_validation/FREEZE.json``.
"""
from __future__ import annotations

import json

from dkmem.memory.prompts import PromptTemplate

ENTITY_JUDGE_PROMPT_ID = "merge_judge_entity_t2_v1"

_SYSTEM = """\
You maintain the long-term memory of a personal assistant. Memory is organised by subject: entries about the same specific person, thing or event belong in one memory record. Merging keeps the text of both entries. Two entries were found to be similar. Decide whether they are about the same subject, so that they should be merged, or about different subjects, so that both should be kept separately.

Answer with exactly one JSON object with exactly these two keys and nothing else (no markdown, no code fences, no commentary):
{"decision": "merge" or "keep_both", "reason": string}

- "merge": both entries are about the same specific person, thing or event. They may state the same fact or different facts about it, including a fact that has changed.
- "keep_both": the entries are about different people, things or events, even if they look alike or state similar facts; or their only shared subject is the user (for example, two unrelated things the user did or plans).
- If the entries do not show that they are about the same specific subject, keep both.
- "reason": one short sentence naming the subject of each entry."""

_EXAMPLES = (
    ("Memory A: User likes strong black coffee\nMemory B: User likes strong black coffee in the morning",
     {"decision": "merge", "reason": "Both are about the user's coffee preference."}),
    ("Memory A: User's laptop is a grey ThinkPad\nMemory B: User's laptop battery needs replacing",
     {"decision": "merge", "reason": "Both are about the user's laptop."}),
    ("Memory A: User's dentist appointment is on Tuesday\nMemory B: User's dentist appointment was moved to Thursday",
     {"decision": "merge", "reason": "Both are about the same dentist appointment; its day changed."}),
    ("Memory A: User's work laptop is slow\nMemory B: User's home laptop has a cracked screen",
     {"decision": "keep_both", "reason": "One is about the work laptop, the other about the home laptop."}),
    ("Memory A: User has a meeting at 3pm on Monday\nMemory B: User is flying to Delhi on Friday",
     {"decision": "keep_both", "reason": "They are two unrelated events."}),
)

MERGE_JUDGE_ENTITY_T2_V1 = PromptTemplate(
    prompt_id=ENTITY_JUDGE_PROMPT_ID,
    system=_SYSTEM,
    user_template="{utterance}",   # the slot carries the two-entry block, see dkmem.memory.judge.render_judge_prompt
    output_keys=("decision", "reason"),
    examples=tuple((q, json.dumps(a, ensure_ascii=False)) for q, a in _EXAMPLES),
)


# ------------------------------------------------------------------------------------------------------------------ v2
# merge_judge_entity_t2_v2 (frozen 2026-10-14 with judge_validation_v2/). v1 failed its pre-registered validation
# (eval/tier2_entity_runs/judge_validation_20261014): merge recall 0.44, same entity + compatible facts 0.10. Cause, read from the
# v1 text and the model's reasons: (1) "organised by subject" / "naming the subject of each entry" was read as the TOPIC of each
# statement ("One is about Kavya's birthday, the other about her location"); (2) even where the model named one entity ("the same
# person but in different places") it answered keep_both, because v1 never says that a shared entity decides the merge whatever
# the facts; (3) the decision came before the reason; (4) the order of the entries changed conflict decisions; (5) only one of five
# examples merged different facts about one entity. v2 defines ENTITY versus FACT, asks for the entities first (reason before
# decision; the strict parser accepts either key order), states that facts never decide, that order does not matter, and
# balances the examples (three merge, three keep_both). Still no language note and no kinship, register or script hint.
ENTITY_JUDGE_V2_PROMPT_ID = "merge_judge_entity_t2_v2"

_SYSTEM_V2 = """\
You maintain the long-term memory of a personal assistant. Each memory record is about one ENTITY: one specific person, animal, thing, place or event in the user's life. The FACTS are what an entry says about its entity (where they live, a date, a job, a plan, a preference). Two memory entries were found to be similar. Decide whether they are about the same entity.

Work in two steps:
1. For each entry, name the entity it is about: who or what is being talked about, not what is said about it. In "Tom lives in Leeds" the entity is Tom, not Leeds and not Tom's home.
2. Compare only the two entities. Do not compare the facts.

- "merge": both entries are about the same entity. This holds even when they state different facts about it, or facts that contradict each other (for example two different cities or two different dates). Merging keeps the text of both entries, so nothing is lost.
- "keep_both": the entries are about different entities, even if they look alike or state similar facts; or the only entity they share is the user (two unrelated things the user did or plans); or an entry does not identify one specific entity (for example "a friend" when the user has many friends), so you cannot tell that it is the same one.
- The order of the two entries does not matter.

Answer with exactly one JSON object with exactly these two keys and nothing else (no markdown, no code fences, no commentary). Write "reason" first, then "decision":
{"reason": string, "decision": "merge" or "keep_both"}
- "reason": one short sentence: "A is about <entity>; B is about <entity>;" followed by "same entity", "different entities" or "not identifiable"."""

_EXAMPLES_V2 = (
    ("Memory A: User's neighbour Tom is a teacher\nMemory B: Tom has two cats",
     {"reason": "A is about the neighbour Tom; B is about Tom; same entity.", "decision": "merge"}),
    ("Memory A: User's dentist appointment is on Tuesday\nMemory B: User's dentist appointment is on Thursday",
     {"reason": "A is about the dentist appointment; B is about the dentist appointment; same entity (the day differs).", "decision": "merge"}),
    ("Memory A: User's work laptop is slow\nMemory B: User's home laptop has a cracked screen",
     {"reason": "A is about the work laptop; B is about the home laptop; different entities.", "decision": "keep_both"}),
    ("Memory A: User likes strong black coffee\nMemory B: User likes strong black coffee in the morning",
     {"reason": "A is about the user's coffee habit; B is about the user's coffee habit; same entity.", "decision": "merge"}),
    ("Memory A: User has a meeting at 3pm on Monday\nMemory B: User is flying to Delhi on Friday",
     {"reason": "A is about a meeting; B is about a flight; different entities.", "decision": "keep_both"}),
    ("Memory A: A friend of the user moved to Oslo\nMemory B: A friend of the user started a new job",
     {"reason": "A is about a friend; B is about a friend; not identifiable.", "decision": "keep_both"}),
)

MERGE_JUDGE_ENTITY_T2_V2 = PromptTemplate(
    prompt_id=ENTITY_JUDGE_V2_PROMPT_ID,
    system=_SYSTEM_V2,
    user_template="{utterance}",
    output_keys=("reason", "decision"),
    examples=tuple((q, json.dumps(a, ensure_ascii=False)) for q, a in _EXAMPLES_V2),
)
