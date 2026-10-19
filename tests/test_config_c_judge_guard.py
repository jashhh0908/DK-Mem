"""Config C judge integrity: a look-alike judge prompt must never run under a frozen prompt's id, and the answer vocabulary of the
first (invalidated) Tier-2 full run must never be read as a decision.

Background: that run imported ``MERGE_JUDGE_V1`` from an external module with the same ``prompt_id`` but another text and the
answers ``{"decision": "different"}``. All 3105 answers were "different" and the run was recorded as the frozen judge. These tests pin
the in-repo guards that make a repeat visible. No GPU or model.

Run: python -m unittest tests.test_config_c_judge_guard  (or pytest)
"""

import dataclasses
import json
import unittest
from pathlib import Path

from dkmem.backends.llm import GenerationParams
from dkmem.memory.judge import JUDGE_PROMPTS, MERGE_JUDGE_INFORMED_V1, MERGE_JUDGE_V1, parse_judge_output
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.prompts import PromptTemplate
from dkmem.pipeline import runner as pipeline_runner
from dkmem.pipeline.runner import build_run_config, run_stage_attribution
from dkmem.store import runner as store_runner
from dkmem.tier1.io import Tier1Record

LEXICON_PATH = Path(__file__).parent.parent / "dkmem" / "memory" / "distinction_features.json"


def record():
    return Tier1Record(eval_pair_id="ep_1", language="hi", utterance_a="meri chachi Pune mein rehti hai",
                       utterance_b="meri mausi Pune mein rehti hai", utterance_a_id="ep_1_utt_a", utterance_b_id="ep_1_utt_b",
                       opaque_entity_id_a="ent_a", opaque_entity_id_b="ent_b")


class ConstantJudge:
    """Answers every judge prompt with one fixed string, like the invalidated external judge."""

    backbone = "fake/constant"

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def generate(self, prompts, params=None):
        self.calls += len(prompts)
        return [self.answer for _ in prompts]


def look_alike(prompt=MERGE_JUDGE_V1):
    """Same id, different text: what the external ``harness.judge`` defined."""
    return dataclasses.replace(prompt, system=prompt.system + "\nAnswer same or different.")


class TestJudgePromptIdentity(unittest.TestCase):
    def test_registered_prompts_are_accepted(self):
        for prompt in (MERGE_JUDGE_V1, MERGE_JUDGE_INFORMED_V1):
            cfg = pipeline_runner.get_pipeline_config("C")
            pipeline_runner._check_judge_prompt(cfg, prompt)            # does not raise

    def test_look_alike_with_a_registered_id_is_refused(self):
        cfg = pipeline_runner.get_pipeline_config("C")
        for prompt in (MERGE_JUDGE_V1, MERGE_JUDGE_INFORMED_V1):
            fake = look_alike(prompt)
            self.assertEqual(fake.prompt_id, prompt.prompt_id)
            self.assertNotEqual(fake.sha256, prompt.sha256)
            with self.assertRaisesRegex(ValueError, "registered prompt with that id"):
                pipeline_runner._check_judge_prompt(cfg, fake)

    def test_refused_by_every_entry_point(self):
        lexicon = load_lexicon(LEXICON_PATH)
        llm = ConstantJudge('{"decision": "merge", "reason": "x"}')
        params = GenerationParams(seed=0)
        with self.assertRaises(ValueError):
            build_run_config("C", ["off"], [record()], generator=llm, lexicon_path=LEXICON_PATH, params=params, judge_prompt=look_alike())
        cfg = build_run_config("C", ["off"], [record()], generator=llm, lexicon_path=LEXICON_PATH, params=params)
        with self.assertRaises(ValueError):
            run_stage_attribution([record()], "C", lexicon, cfg, generator=llm, params=params, judge_prompt=look_alike())
        self.assertEqual(llm.calls, 0)                                   # refused before any model call
        self.assertIs(store_runner._check_judge_prompt, pipeline_runner._check_judge_prompt)   # the store runner shares the guard

    def test_a_new_id_is_not_blocked(self):
        new = PromptTemplate(prompt_id="merge_judge_new_v1", system=MERGE_JUDGE_V1.system + "\nx", user_template="{utterance}",
                             output_keys=("decision", "reason"), examples=())
        self.assertNotIn(new.prompt_id, JUDGE_PROMPTS)
        pipeline_runner._check_judge_prompt(pipeline_runner.get_pipeline_config("C"), new)


class TestDirectPathsAreGuardedToo(unittest.TestCase):
    """The library entry points that bypass the runners: rendering, judge_pairs and direct JudgeHost construction."""

    def test_rendering_and_judge_pairs_refuse_a_look_alike_without_calling_the_model(self):
        from dkmem.memory.judge import judge_pairs, render_judge_prompt
        llm = ConstantJudge('{"decision": "merge", "reason": "x"}')
        for prompt in (MERGE_JUDGE_V1, MERGE_JUDGE_INFORMED_V1):
            fake = look_alike(prompt)
            with self.assertRaisesRegex(ValueError, "registered prompt with that id"):
                render_judge_prompt("a", "b", fake)
            with self.assertRaisesRegex(ValueError, "registered prompt with that id"):
                judge_pairs([("a", "b")], llm, GenerationParams(seed=0), prompt=fake)
        self.assertEqual(llm.calls, 0)

    def test_the_registered_prompts_and_new_ids_still_render(self):
        from dkmem.memory.judge import render_judge_prompt
        for prompt in (MERGE_JUDGE_V1, MERGE_JUDGE_INFORMED_V1):
            self.assertEqual(render_judge_prompt("a", "b", prompt)[-1]["content"], "Memory A: a\nMemory B: b")
        new = PromptTemplate(prompt_id="merge_judge_new_v1", system="s", user_template="{utterance}", output_keys=("decision", "reason"), examples=())
        self.assertEqual(len(render_judge_prompt("a", "b", new)), 2)

    def test_a_judge_host_cannot_be_built_with_a_look_alike(self):
        from dkmem.store.consolidate import JudgeHost
        JudgeHost(ConstantJudge("x"), None)                                    # the default frozen prompt is fine
        with self.assertRaisesRegex(ValueError, "registered prompt with that id"):
            JudgeHost(ConstantJudge("x"), None, prompt=look_alike())

    def test_the_tier2_memory_probe_checks_its_prompt_content(self):
        import sys
        eval_dir = str(Path(__file__).parent.parent / "eval")
        sys.path.insert(0, eval_dir)
        try:
            import importlib
            src = (Path(eval_dir) / "tier2" / "judge_validation" / "memory_probe.py").read_text(encoding="utf-8")
            self.assertIn("check_frozen_prompt(prompt)", src)
            self.assertLess(src.index("prompt = JUDGES_BY_ID[a.judge_prompt]"), src.index("check_frozen_prompt(prompt)"))
            self.assertLess(src.index("check_frozen_prompt(prompt)"), src.index("render_judge_prompt(x, y, prompt)"))
        finally:
            sys.path.remove(eval_dir)


class TestInvalidatedVocabularyIsNeverADecision(unittest.TestCase):
    def test_parser_rejects_the_external_judge_vocabulary(self):
        for raw in ('{"decision": "different"}', '{"decision": "same"}', '{"decision": "different", "reason": "x"}'):
            r = parse_judge_output(raw)
            self.assertFalse(r.ok, raw)
            self.assertIsNone(r.decision)

    def test_a_constant_wrong_vocabulary_yields_failures_not_keep_both(self):
        lexicon = load_lexicon(LEXICON_PATH)
        llm = ConstantJudge('{"decision": "different"}')
        params = GenerationParams(seed=0)
        cfg = build_run_config("C", ["off", "lexicon"], [record()], generator=llm, lexicon_path=LEXICON_PATH, params=params)
        res = run_stage_attribution([record()], "C", lexicon, cfg, generator=llm, params=params)
        self.assertEqual(res.summary["pairs_judge_failed"], 1)
        self.assertEqual(res.summary["pairs_ok"], 0)
        self.assertEqual({m: len(rows) for m, rows in res.rows_by_mode.items()}, {"off": 0, "lexicon": 0})   # no row, hence no decision
        self.assertEqual(res.trace[0]["host"]["proposed"], None)

    def test_the_frozen_vocabulary_is_read_correctly(self):
        llm = ConstantJudge(json.dumps({"decision": "merge", "reason": "same aunt"}))
        lexicon = load_lexicon(LEXICON_PATH)
        params = GenerationParams(seed=0)
        cfg = build_run_config("C", ["off"], [record()], generator=llm, lexicon_path=LEXICON_PATH, params=params)
        res = run_stage_attribution([record()], "C", lexicon, cfg, generator=llm, params=params)
        self.assertEqual(res.summary["pairs_ok"], 1)
        self.assertEqual(res.trace[0]["final"]["off"]["pairwise_decision"], "merge")


if __name__ == "__main__":
    unittest.main()
