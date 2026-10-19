# Tier 2 full run: frozen scoring and reporting plan (v1)

Frozen **before** any full-run output exists (`ANALYSIS_PLAN_FREEZE_v1.json` pins this file, `entity_report.py`, `contract.py` and
`generate_tier2.py`'s QA scorer by sha256). The implementation is `python -m tier2.entity_report`. Neither file may change after full-run
results exist; a needed change is a new plan version, reported as such, with the v1 results kept.

## 1. Runs and units

- Two benchmark arms, run separately: **v1.0** (`eval/tier2/data`, 69 conversations, 42 gold-different + 27 gold-same injected pairs,
  111 QA questions) and **mirror v1.1m** (`eval/tier2/mirror_v1_1/data`, the same 69 source pairs with the attribute pattern flipped, no QA).
- Per arm, four runs: C/off, C/lexicon, D/off, D/lexicon. C = verbatim + frozen identity judge `merge_judge_entity_t2_v2`
  (sha256 `36146ec0…`) on Qwen2.5-7B `a09a3545…` (sharded); D = verbatim + bge-m3 `5617a9f6…` cosine `> 0.90` (run-level execution value,
  not an operating point). QA answerer for every run: Qwen2.5-3B `aa8e7253…`, prompt `tier2_qa_v1`. Seed 0, single run: every result is
  descriptive.
- Unit of analysis for pair outcomes: the injected pair; one pair per conversation, so pair = conversation.
- Labels: gold relation from the arm's `tier2_gold.jsonl` (by construction from the provisional, LLM-annotated Tier 1 gold).

## 2. Primary outcome: direct-pair FCR / MCR

- A pair's **direct status** is `contract.validate_run`'s `status`: `merged` iff the pairwise record between the earlier and the later
  injected entry has decision `merge`/`supersede`; otherwise `not_merged` (record with `no_merge`/`underdetermined_link`),
  `retrieval_miss` (the earlier entry was not among the later entry's top-5 candidates) or `judge_parse_error`.
- **FCR-direct** = #gold-different pairs with direct status `merged` / #gold-different pairs (42).
- **MCR-direct** = #gold-same pairs whose direct status is not `merged` / #gold-same pairs (27).
- Retrieval misses and judge parse errors stay in the denominators as not merged and are listed separately.

## 3. Secondary outcome: final-memory-record connectivity

- **Connected** iff the two injected entries are in the same final memory record (`same_final_memory_record`, the union-find closure of
  all merge records of the conversation).
- **FCR-record** = #gold-different pairs connected / 42; **MCR-record** = #gold-same pairs not connected / 27.
- Connection type: `direct` (direct status `merged`), `indirect_only` (connected, direct status not `merged`: joined only through
  intermediate entries), `none`. For every indirect-only pair the shortest merge path is reported (edges, intermediate turns, whether
  every intermediate is a distractor). Direct and indirect connections are always reported separately, never pooled into one number
  without the split.

## 4. Diagnostics

- **Merge rates by comparison type**, per run, over compared pairs (pairwise records; judge parse errors produce no record and are counted
  in §6): distractor-distractor, injected-distractor, injected-injected. Rate = merges / records (ratio of sums over conversations).
- **Gate vetoes bypassed through intermediate merges** (gate-on runs): a veto is a record whose host proposed a merge (C: the judge
  proposal in the judge log; D: `similarity_score > tau`) and whose final decision is not a merge. It is **bypassed** if its two entries
  nevertheless end in the same final memory record. Reported overall, by compatibility, by comparison type, and for injected pairs.
- **QA** (v1.0 arm only; the mirror arm has no QA): accuracy with the frozen deterministic scorer (`generate_tier2.score_answer`), split by
  (a) whether every supporting turn of the question was among the retrieved memory records (`supporting_memory_retrieved`) and
  (b) gold relation × final-record connectivity of the question's pair; forbidden-alias hits counted. **QA is never compared across arms.**

## 5. Aggregation and intervals

- Pair-level proportions (FCR/MCR, direct and record): **Wilson 95% score interval** (pairs independent: one per conversation).
- Comparison-level rates (merge rates in §4) and QA accuracy (≤ 2 questions per conversation): **ratio of sums with a 95% percentile
  bootstrap over conversations** (resample conversations with replacement, B = 10 000, seed 20261014).
- Paired contrasts on the same pairs: gate lexicon − off within a config (per arm), C − D within a gate mode (per arm), and mirror − v1.0
  per config/mode on the same source pair (merge decisions only): difference of direct-merge rates with a **95% paired percentile bootstrap
  over pairs** (B = 10 000, seed 20261014) and the **exact two-sided McNemar p** on the discordant pairs. No multiplicity correction; all
  contrasts are descriptive (single seed, provisional labels).
- Strata (counts of pairs, direct merges and connections): distinction class, language, session distance, attribute pattern, each split
  by relation. No interval is reported for a stratum; small strata (e.g. 4 name_variant gold-different, 6 German register) are counts only.

## 6. Judge diagnostics (C runs)

Calls, parse failures, possibly-truncated, at-budget, proposal counts, reason-decision contradictions (reason says "same entity" but
`keep_both`, or "different entities"/"not identifiable" but `merge`), distractor-distractor calls whose reason names the user on both
sides and how many of those were merged, and the judge's reason for every injected pair. Passing technical checks is not evidence that a
decision is correct.

## 7. Failures and exclusions

- A run that fails `contract.validate_run` (exact passthrough, QA-id map, the arm's FROZEN file) is reported as a failure and contributes
  no metric. No run is re-run or replaced without reporting it.
- No pair, conversation or decision is excluded, re-scored or repaired. A judge parse error is never a decision.

## 8. Limitations reported with every result

1. **User-only / routine over-merging by the Config C judge.** In the smoke runs the 7B judge merged 4 of 100 distractor-distractor
   comparisons per run ("A is about the user; B is about the user; same entity") on routine statements, against the frozen policy; the
   audit found no implementation defect. The judge validation set had no pair of two different routines (coverage gap). §4 measures it.
2. **German du/Sie false merges.** The 7B judge treats du/Sie as one person (validation probes 0/4 kept apart; smoke: merged); under the
   benchmark's contested two-addressee convention these are false merges, and the lexicon has no German entries, so the gate cannot veto them.
3. **Indirect contamination.** Merges are transitive and the gate vetoes pairs, not records; §3 and §4 measure connections through
   intermediates and bypassed vetoes.
4. Config C's judge backbone (7B) differs from Tier 1 Config C (3B): not comparable with Tier 1 C. D's τ 0.90 is an execution value.
5. Class × relation × language are confounded in both arms; the mirror arm has no QA; labels are provisional; one seed.
