# DK-Mem Tier-2 controlled multi-session injection benchmark (dataset package)

**What this is.** A controlled *injection* benchmark: each of the 69 in-scope Tier-1 pairs (kinship, honorific_register, name_variant; 42 gold-different + 27 gold-same) is injected, as two constructed fact sentences, into one synthetic 6-session conversation filled with fixed hand-authored distractor turns. It measures false/missed consolidation and fact preservation *in situ* (across sessions, with distractors). **It makes no claim about how often these distinctions occur in natural conversation**; labels are by construction from the provisional Tier-1 gold (LLM-annotated, not final).

**Scope of this package.** Data only. The DK-Mem runner (persistent store, shared bge-m3 top-5 candidate retrieval, merge logic, QA answerer) and the Tier-2 evaluation metrics are separate, later steps. Nothing here modifies Tier-1 gold, Tier-1 runs, the evaluator, `ipdV2` or the D artifacts.

## Files (`tier2/data/`) and who may read them
| File | Content | Read by |
|---|---|---|
| `tier2_conversations.jsonl` | 69 conversations x 6 sessions x 4 turns (alternating user/assistant). **No labels, no injected/distractor marks.** | **the system under test (the only file it receives)** |
| `tier2_qa_questions.jsonl` | 111 QA questions (`qa_id`, `conversation_id`, `question`, `language`, `ask_after_session_index`) asked after the last session | the QA answerer |
| `tier2_injection_metadata.jsonl` | where each pair was injected: turn/session ids, `session_distance`, `first_member`, class, language, the attribute facts, distractor turn ids | evaluator only |
| `tier2_gold.jsonl` | gold in the **exact Tier-1 gold format** (`relation`, `distinction_class`, `language`, utterance ids = injected turn ids, `opaque_entity_id_a/b` = the Tier-1 ids of the source pair, reused exactly) | evaluator only |
| `tier2_qa_reference.jsonl` | accepted aliases (`slots`), `forbidden` aliases, `qa_type`, supporting turns | evaluator only |
| `tier2_manifest.json` | seed, parameters, counts, SHA-256 of every input and output | everyone |

As in Tier-1, `relation` (not shared entity ids) carries same/different: opaque entity ids are unique per utterance.

## Design
* **Pair specs** (`banks/pair_specs.json`): one explicit entry per source pair. Member `a` and `b` each get one templated sentence carrying the same distinction marker as the Tier-1 pair (kinship term, name script variant, register pronoun, or du/Sie) and a different attribute value (city, weekday or birthday month). Gold-different pairs: two questions, one per member; the other member's value is a *forbidden* alias, so a conflated answer fails. Gold-same pairs: one combined question that needs both facts (a missed consolidation costs accuracy).
* **Distractors** (`banks/distractors.json`): a fixed bank of hand-authored neutral turns (Hinglish / German user turns, generic assistant acknowledgements). No model sampling is involved. The validator enforces that they contain no injected names, lexicon terms, register pronouns, family terms or QA values.
* **Conversation**: 6 weekly sessions; each session is user, assistant, user, assistant (24 turns); 12 user turns = 2 injected + 10 distractors. The two members are injected `session_distance` sessions apart, with distance 1, 3 or 5 allocated cyclically per (class, relation) stratum and balanced globally (23/23/23) and within classes.
* **Shared candidate retrieval** (benchmark parameter for the runner): BAAI/bge-m3, top-5, identical for every configuration so that configurations differ only in their merge logic.
* **QA scoring** (`generate_tier2.score_answer`): deterministic. Answers are normalised (casefold, strip Latin diacritics, drop punctuation); a slot is hit if any accepted alias appears as a whole token sequence; the answer is correct iff all slots are hit and no forbidden alias appears.

## Regenerate / validate
```
python -m tier2.generate_tier2 --out <empty dir>     # deterministic (seed in the manifest); refuses to overwrite
python -m tier2.validate_tier2                        # exit 0 only if every check passes
python -m unittest tests.test_tier2_data
```

## Design assumptions and limitations (stated up front)
* **name_variant gold-different is read at the person level (design assumption, not a Tier-1 fact).** The four Tier-1 pairs ("Priya office se aa rahi hai" vs "Priya college se aa rahi hai") vary only a context word. Tier 2 realises each as two different people who share a first name and are told apart by that context ("Meri office wali Priya ..." vs "Meri college wali Priya ..."), each with a different city, and the gold relation stays `different`. If the Tier-1 label meant two *events* involving one person, these four conversations (`t2c_0049`, `0022`, `0051`, `0004`) test a different thing than intended. Treat results on them as exploratory.
* **Direct address of a third party (documented, not redesigned).** The 13 honorific_register conversations (7 Hindi, 6 German) keep the Tier-1 register-bearing form, so the user addresses Maa / Papa / Nani / Markus / Lena ... directly inside a user-assistant chat. This is less natural than reported speech but keeps the register markers (tu/tum/aap, du/Sie) intact. The German pairs rest on two different addressees sharing a first name or title, told apart in the question by "den ich duze" / "den ich mit Sie anspreche".
* **Gold-same QA does not isolate consolidation.** A combined question can be answered from two unmerged entries if the answerer retrieves both, so a missed consolidation does not always cost accuracy. Analyse QA accuracy conditional on the pair outcome.
* **Residual surface cue.** The 20 name_variant gold-same injected turns that carry a Devanagari name are the only turns with Devanagari script (all distractors are Roman); this is the manipulated script variation itself and cannot be removed. Capitalisation and terminal punctuation of injected and distractor user turns are matched (audit fix H1).
* Constructed sentences; templated phrasing; one injected pair per conversation (conversation = cluster, so pair-level bootstrap applies).
* Labels are by construction from the provisional Tier-1 gold; the honorific questions are the least natural (Hindi register: "Maa, tum ... / Maa, aap ..."; German: two different addressees who share a first name, told apart by du vs Sie).
* Only the session distance is manipulated; input form (Devanagari arm, English control) and code-mixing ratio are not (a possible v1.1).
* The current DK-Mem lexicon detects all 64 kinship injected turns but only 10 of 14 Hindi register turns (it has `tu`/`tum`/`aap` but not `tumne`/`aapne`), and no name or German entries; this is reported by the validator for information.
