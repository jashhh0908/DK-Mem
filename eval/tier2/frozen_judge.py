"""Constants of the Config C merge judge (no imports: safe for the evaluator-side modules too).

The frozen Tier-1 judge is ``dkmem.memory.judge.MERGE_JUDGE_V1``. The first Tier-2 full run used a different prompt that the
Kaggle harness also called ``merge_judge_v1``; it is kept only to recognise (and label) those historical outputs.
"""
FROZEN_JUDGE_PROMPT_ID = "merge_judge_v1"
FROZEN_JUDGE_PROMPT_SHA256 = "3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff"   # DKMEM_FIRST_RUN.md; tests/test_baselines.py
# Tier-1's judge decoding budget (dkmem.pipeline.runner._default_judge_params: min(max_new_tokens, 128)). The frozen prompt answers
# with a "reason" sentence, so a budget of 32 tokens truncates the JSON and fails the strict parser.
FROZEN_JUDGE_MAX_NEW_TOKENS = 128
# Pairwise-row threshold of a judge host: null, as dkmem.pipeline writes it for Configs A-C and as pairwise_eval.schema.json
# describes it ("null when the host's merge decision uses no similarity cutoff").
JUDGE_ROW_THRESHOLD = None

# --- the Tier-2 identity judge (eval/tier2/judge_entity.py; frozen with its validation set in judge_validation/FREEZE.json).
# Same decoding budget, same strict parser, same null row threshold as the frozen Tier-1 judge; a different question.
ENTITY_JUDGE_PROMPT_ID = "merge_judge_entity_t2_v1"
ENTITY_JUDGE_PROMPT_SHA256 = "6f85be1900a9e95586426b084e498dfec94bb46df25ad2db2eb80bd1e273c2ed"
ENTITY_JUDGE_MAX_NEW_TOKENS = 128
# v2 (judge_entity.MERGE_JUDGE_ENTITY_T2_V2; frozen with judge_validation_v2/FREEZE.json). v1 failed its validation and is kept
# only so its validation outputs stay recognisable.
ENTITY_JUDGE_V2_PROMPT_ID = "merge_judge_entity_t2_v2"
ENTITY_JUDGE_V2_PROMPT_SHA256 = "36146ec054a700796a785a5a942c0fdaaa33a1e100103c8a8410d596c146f149"
# The Tier-2 identity judges: prompt id -> sha256 (same budget, parser and null row threshold).
IDENTITY_JUDGES = {ENTITY_JUDGE_PROMPT_ID: ENTITY_JUDGE_PROMPT_SHA256, ENTITY_JUDGE_V2_PROMPT_ID: ENTITY_JUDGE_V2_PROMPT_SHA256}
# Every judge a Tier-2 Config C run may use: prompt id -> sha256 of the exact text.
TIER2_JUDGES = {FROZEN_JUDGE_PROMPT_ID: FROZEN_JUDGE_PROMPT_SHA256, **IDENTITY_JUDGES}

# --- historical (invalid-as-Tier-1-C) harness judge, recognised only to label old runs
LEGACY_HARNESS_JUDGE_PROMPT_SHA256 = "f0dc9cf75eb8891c9651ccdd8e685af047c5813a92f61c55fa6440ed6b658bea"
LEGACY_JUDGE_ROW_THRESHOLD = 0.0          # the harness judge wrote 0.0 ("the judge has no tau")
LEGACY_JUDGE_MAX_NEW_TOKENS = 32
