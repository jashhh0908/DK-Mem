"""The exact-equality boundary ``similarity == tau``, pinned at every layer that decides a merge.

Two conventions exist in this repository, both documented and both intentional:

* STRICT ``similarity > tau``: every HOST (Config D in the pipeline and store runners, the harness ``EmbeddingThresholdHost``), the
  dkmem offline sweep and the evaluator sweep (``experiment_output_spec.md``: Team B recomputes ``merge := similarity_score > tau``).
* ``similarity >= tau``: the standalone Tier-1 policy ``dkmem.memory.gate.gate`` (DKMEM_IMPLEMENTATION.md, the ``dkmem_v1`` fixtures
  and ``tests/test_gate.py``). Only the ``dk-mem-lexicon`` / ``dk-mem-lexicon-llm`` Tier-1 runner uses it.

They differ only when a score equals tau exactly. The evaluator reports such a record as a tie-disagreement (WARN) rather than hiding it
(``eval/evaluator/sweep.py::check_recorded_decisions``). No stored pairwise record is at its threshold (0 of 46,561 at the time of
writing). These tests do NOT choose between the conventions; they make any change to either side deliberate and show the Tier-1
record-versus-evaluator difference end to end.
"""

import math
import sys
import unittest
from pathlib import Path

from dkmem.memory.gate import gate
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.similarity import SimilarityMetric
from dkmem.store.consolidate import HostQuery, ThresholdHost
from dkmem.tier1.io import Tier1Record
from dkmem.tier1.runner import run_episode_lexicon_only

REPO = Path(__file__).parent.parent
for p in (str(REPO), str(REPO / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAU = 0.85
BELOW, ABOVE = math.nextafter(TAU, 0.0), math.nextafter(TAU, 1.0)
LEXICON = REPO / "dkmem" / "memory" / "distinction_features.json"


def q(score):
    return HostQuery("c", "stored text", "new text", score)


class TestStrictHosts(unittest.TestCase):
    def test_store_threshold_host(self):
        h = ThresholdHost(TAU)
        self.assertEqual([p.decision for p in h.propose([q(BELOW), q(TAU), q(ABOVE)])], ["keep_both", "keep_both", "merge"])

    def test_harness_embedding_threshold_host(self):
        from harness.hosts import EmbeddingThresholdHost
        h = EmbeddingThresholdHost(TAU)
        self.assertEqual([h.propose(s) for s in (BELOW, TAU, ABOVE)], ["keep_both", "keep_both", "merge"])


class TestStandaloneTier1Gate(unittest.TestCase):
    def test_gate_merges_at_equality_and_keeps_both_just_below(self):
        self.assertEqual(gate({}, {}, BELOW, TAU)[1], "keep_both")
        self.assertEqual(gate({}, {}, TAU, TAU)[1], "merge")             # >= : the documented standalone convention
        self.assertEqual(gate({}, {}, ABOVE, TAU)[1], "merge")

    def test_a_veto_does_not_depend_on_the_boundary(self):
        a, b = {"kinship": "chachi"}, {"kinship": "mausi"}
        for s in (BELOW, TAU, ABOVE):
            self.assertEqual(gate(a, b, s, TAU)[1], "keep_both")


class TestTier1RecordAtATieIsVisibleToTheEvaluator(unittest.TestCase):
    """The one place the conventions meet: a Tier-1 record written at sim == tau, read back by the evaluator's sweep rule."""

    def record_at(self, score):
        r = Tier1Record(eval_pair_id="ep_1", language="hi", utterance_a="ek do teen", utterance_b="char paanch chhe",
                        utterance_a_id="ep_1_utt_a", utterance_b_id="ep_1_utt_b", opaque_entity_id_a="ent_a", opaque_entity_id_b="ent_b")
        rows = run_episode_lexicon_only(r, load_lexicon(LEXICON), run_id="r", tau=TAU, similarity=SimilarityMetric("fixed", lambda a, b: score))
        self.assertEqual(len(rows), 1)
        return rows[0].to_dict()

    def test_tie_record_is_a_merge_but_the_evaluator_rule_says_no_merge_and_flags_it(self):
        from evaluator.sweep import derived_merge, recorded_merge
        row = self.record_at(TAU)
        self.assertEqual((row["similarity_score"], row["threshold"], row["compatibility"]), (TAU, TAU, "compatible"))
        self.assertEqual(row["decision"], "merge")                        # standalone gate: >=
        self.assertTrue(recorded_merge(row))
        self.assertFalse(derived_merge(row, row["threshold"]))            # contract rule: >
        # the disagreement is exactly a tie, which check_recorded_decisions separates out as at_tie
        self.assertEqual(row["similarity_score"], row["threshold"])

    def test_off_the_tie_both_conventions_agree(self):
        from evaluator.sweep import derived_merge, recorded_merge
        for score in (BELOW, ABOVE):
            row = self.record_at(score)
            self.assertEqual(recorded_merge(row), derived_merge(row, row["threshold"]), score)


if __name__ == "__main__":
    unittest.main()
