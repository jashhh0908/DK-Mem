"""Does the DK-Mem gate actually stop wrong merges in the A-D pipeline?

Synthetic Hindi pairs are built from the *shipped lexicon* (``dkmem/memory/distinction_features.json``: 12 kinship
terms and the register pronouns tu / tum / aap, each in romanized and Devanagari form) and sorted into the three
compatibility conditions of ``dkmem.memory.gate``:

* **incompatible**    both sides mark a discriminative feature with different values  -> ``keep_both``
* **underdetermined** exactly one side marks it                                       -> ``link_unresolved``
* **compatible**      the sides agree, or neither marks it                            -> unchanged (merge)

Each case is checked in two independent ways before the pipeline is trusted to have "triggered" anything:
(1) the lexicon matcher and ``compatible()`` are run directly on the two utterances and must produce the case's intended
marks and condition; (2) in the pipeline trace the *host proposed a merge* (the judge answered merge; for Config D the
cosine exceeded tau) before the gate acted. Only then is the final decision asserted, in DK-Mem modes ``lexicon`` and
``lexicon+llm`` of Configs A, B, C and D, with mode ``off`` as the control that shows the merge really would have happened.

What the pipeline cannot do is asserted too (``TestWhatTheGateCannotDo``): a model-only tag on a term the lexicon does
not cover is never consulted, ``name_variant`` never gates, and an ambiguous utterance whose V4 call failed has no
decision in ``lexicon+llm``. No GPU or model: scripted fake judge / extractor / embedding.

Run: python -m unittest discover -s tests   (test_pipeline's FakeLLM and run() are imported by name)
"""

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from test_pipeline import ALL_MODES, LEXICON_PATH, FakeLLM, rec, run

from dkmem.backends.llm import GenerationParams
from dkmem.memory.dkmem_extract import ambiguous_distinction_classes
from dkmem.memory.gate import COMPATIBILITY, DEFAULT_DISCRIMINATIVE_FEATURES, apply_gate, compatible
from dkmem.memory.lexicon import load_lexicon
from dkmem.memory.similarity import SimilarityMetric
from dkmem.pipeline.trace import lexicon_distinction

LEX = load_lexicon(LEXICON_PATH)
TAU = 0.85
GATED = ("lexicon", "lexicon+llm")

# lexicon value -> its surface forms (romanized first, Devanagari second), straight from the shipped file
FORMS: dict[str, dict[str, list[str]]] = {"kinship": {}, "register": {}}
for _e in LEX:
    FORMS.setdefault(_e.distinction_class, {})[_e.value] = list(_e.surface_forms)
KINSHIP, REGISTER = sorted(FORMS["kinship"]), sorted(FORMS["register"])

FINAL = {"incompatible": "no_merge", "underdetermined": "underdetermined_link", "compatible": "merge"}
INTERNAL = {"incompatible": "keep_both", "underdetermined": "link_unresolved", "compatible": "merge"}

UNMARKED = ["woh Pune mein rehti hai", "woh kahan hai"]


def kin(value, city="Pune", script=0):
    form = FORMS["kinship"][value][script]
    return f"meri {form} {city} mein rehti hai" if script == 0 else f"मेरी {form} {city} में रहती है"


def reg(value, tail="kahan ho", script=0):
    form = FORMS["register"][value][script]
    return f"{form} {tail}" if script == 0 else f"{form} कहाँ हो"


@dataclass(frozen=True)
class Case:
    cid: str
    a: str
    b: str
    expected: str  # incompatible | underdetermined | compatible
    marks_a: dict
    marks_b: dict
    group: str


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(group, cid, a, b, expected, ma, mb):
        cases.append(Case(f"ep_{len(cases):03d}", a, b, expected, ma, mb, f"{group}:{cid}"))

    for i, x in enumerate(KINSHIP):
        for y in KINSHIP[i + 1:]:
            add("kinship_incompatible", f"{x}~{y}", kin(x), kin(y, "Delhi"), "incompatible", {"kinship": x}, {"kinship": y})
    for x in KINSHIP:
        add("kinship_compatible", f"{x}=={x}", kin(x), kin(x, "Delhi"), "compatible", {"kinship": x}, {"kinship": x})
        add("kinship_script_variant", f"{x} roman/deva", kin(x), kin(x, "दिल्ली", script=1), "compatible", {"kinship": x}, {"kinship": x})
        add("kinship_underdetermined", f"{x} vs unmarked", kin(x), UNMARKED[0], "underdetermined", {"kinship": x}, {})
        add("kinship_underdetermined", f"unmarked vs {x}", UNMARKED[0], kin(x), "underdetermined", {}, {"kinship": x})
    for x in ("chachi", "mausi", "bua"):  # a Devanagari form on one side must still conflict with another value
        y = "mami" if x != "mami" else "bua"
        add("kinship_incompatible_cross_script", f"{x}(deva)~{y}", kin(x, "दिल्ली", script=1), kin(y), "incompatible", {"kinship": x}, {"kinship": y})
    for i, x in enumerate(REGISTER):
        for y in REGISTER[i + 1:]:
            add("register_incompatible", f"{x}~{y}", reg(x), reg(y, "kahan gaya"), "incompatible", {"register": x}, {"register": y})
    for x in REGISTER:
        add("register_compatible", f"{x}=={x}", reg(x), reg(x, "kahan gaya"), "compatible", {"register": x}, {"register": x})
        add("register_underdetermined", f"{x} vs unmarked", reg(x), UNMARKED[1], "underdetermined", {"register": x}, {})
        add("register_underdetermined", f"unmarked vs {x}", UNMARKED[1], reg(x), "underdetermined", {}, {"register": x})
    add("neither_marked", "no marks", UNMARKED[0], "woh Delhi mein rehta hai", "compatible", {}, {})
    # two discriminative features at once
    add("combined", "kinship incompatible outranks register underdetermined", "meri chachi, tu kahan hai",
        "meri mausi Delhi mein hai", "incompatible", {"kinship": "chachi", "register": "tu"}, {"kinship": "mausi"})
    add("combined", "register incompatible with kinship agreeing", "meri chachi, tu kahan hai",
        "meri chachi Delhi mein hai, aap kahan hain", "incompatible", {"kinship": "chachi", "register": "tu"}, {"kinship": "chachi", "register": "aap"})
    add("combined", "both features agree", "meri chachi, tu kahan hai", "meri chachi Delhi mein hai, tu kahan gaya",
        "compatible", {"kinship": "chachi", "register": "tu"}, {"kinship": "chachi", "register": "tu"})
    add("combined", "kinship agrees, register one-sided", "meri chachi, tu kahan hai", "meri chachi Delhi mein hai",
        "underdetermined", {"kinship": "chachi", "register": "tu"}, {"kinship": "chachi"})
    return cases


CASES = build_cases()
RECORDS = [rec(c.cid, c.a, c.b) for c in CASES]
BY_ID = {c.cid: c for c in CASES}


def sim_metric(score):
    return SimilarityMetric("fake_embedding", lambda a, b: score)


def host_merges():
    return FakeLLM(judge=lambda a, b: ("merge", "same"))


def host_keeps_both():
    return FakeLLM(judge=lambda a, b: ("keep_both", "different"))


def run_config(config_id, llm, records=None, modes=ALL_MODES, sim=0.99):
    records = RECORDS if records is None else records
    if config_id == "D":
        return run("D", records, llm, modes=modes, similarity=sim_metric(sim), tau=TAU)
    return run(config_id, records, llm, modes=modes)


def trace_by_pair(result):
    return {t["eval_pair_id"]: t for t in result.trace}


# ---------------------------------------------------------------------------------------------------------------------


class TestCasesTriggerTheirConditionIndependentlyOfThePipeline(unittest.TestCase):
    """The lexicon matcher and ``compatible()`` alone must put every case in the condition it is meant to test."""

    def test_every_case_marks_what_it_claims_and_lands_in_its_condition(self):
        for c in CASES:
            with self.subTest(case=c.group):
                da, db = lexicon_distinction(LEX, c.a, "hi"), lexicon_distinction(LEX, c.b, "hi")
                self.assertEqual((da, db), (c.marks_a, c.marks_b), (c.a, c.b))
                self.assertEqual(compatible(da, db), c.expected)
                self.assertEqual(ambiguous_distinction_classes(LEX, c.a, "hi") | ambiguous_distinction_classes(LEX, c.b, "hi"), set())

    def test_the_cases_cover_all_three_conditions_and_every_lexicon_value(self):
        counts = {k: sum(1 for c in CASES if c.expected == k) for k in COMPATIBILITY}
        self.assertTrue(all(counts.values()), counts)
        self.assertEqual(counts["incompatible"], 66 + 3 + 3 + 2)  # kinship pairs, cross-script, register, combined
        used = {v for c in CASES for m in (c.marks_a, c.marks_b) for v in m.values()}
        self.assertEqual(used, set(KINSHIP) | set(REGISTER))
        self.assertEqual(len(KINSHIP), 12)
        self.assertEqual(REGISTER, ["aap", "tu", "tum"])
        self.assertEqual({e.distinction_class for e in LEX}, {"kinship", "register"})  # no name_variant entries

    def test_the_unmarked_filler_words_are_not_lexicon_terms(self):
        for text in UNMARKED + ["woh Delhi mein rehta hai"]:
            self.assertEqual(lexicon_distinction(LEX, text, "hi"), {}, text)


class TestGateStopsFalseMergesInEveryConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {cfg: run_config(cfg, host_merges()) for cfg in "ABCD"}

    def test_the_host_proposed_a_merge_for_every_case_before_the_gate_acted(self):
        for cfg, res in self.results.items():
            for t in res.trace:
                with self.subTest(config=cfg, pair=t["eval_pair_id"]):
                    self.assertEqual(t["status"], "ok")
                    self.assertEqual(t["host"]["proposed"], "merge")
                    if cfg == "D":
                        self.assertGreater(t["similarity"]["score"], TAU)
                        self.assertTrue(t["host"]["tau_used"])
                        self.assertIsNone(t["host"]["judge"])
                    else:
                        self.assertEqual(t["host"]["judge"]["decision"], "merge")
                        self.assertIsNone(t["host"]["tau"])

    def test_without_the_gate_every_pair_is_merged_so_the_veto_is_real(self):
        for cfg, res in self.results.items():
            for t in res.trace:
                with self.subTest(config=cfg, pair=t["eval_pair_id"]):
                    self.assertEqual(t["final"]["off"]["pairwise_decision"], "merge")
                    self.assertFalse(t["gate"]["off"]["applied"])

    def test_final_decisions_match_the_condition_in_both_gated_modes(self):
        n = 0
        for cfg, res in self.results.items():
            traces = trace_by_pair(res)
            for c in CASES:
                t = traces[c.cid]
                for mode in GATED:
                    with self.subTest(config=cfg, mode=mode, case=c.group):
                        g = t["gate"][mode]
                        self.assertTrue(g["applied"])
                        self.assertEqual((g["distinction_a"], g["distinction_b"]), (c.marks_a, c.marks_b))
                        self.assertEqual(g["proposed"], "merge")
                        self.assertEqual(g["compatibility"], c.expected)
                        self.assertEqual(g["decision"], INTERNAL[c.expected])
                        self.assertEqual(g["vetoed"], c.expected != "compatible")
                        self.assertEqual(t["final"][mode]["decision"], INTERNAL[c.expected])
                        self.assertEqual(t["final"][mode]["pairwise_decision"], FINAL[c.expected])
                        n += 1
        self.assertEqual(n, 4 * len(CASES) * 2)

    def test_output_rows_carry_the_same_decision_compatibility_and_entity_prediction(self):
        for cfg, res in self.results.items():
            for mode in GATED:
                rows = {r.record_id.split("::")[0].rsplit("_utt_a", 1)[0]: r for r in res.rows_by_mode[mode]}
                self.assertEqual(len(rows), len(CASES))
                for c in CASES:
                    with self.subTest(config=cfg, mode=mode, case=c.group):
                        r = rows[c.cid]
                        self.assertEqual((r.decision, r.compatibility), (FINAL[c.expected], c.expected))
                        merged = c.expected == "compatible"
                        self.assertEqual(r.predicted_entity_id is not None, merged)
                        self.assertIsNone(r.superseded_entry_id)
                        self.assertEqual(r.threshold, TAU if cfg == "D" else None)
            off = res.rows_by_mode["off"]
            self.assertEqual({r.decision for r in off}, {"merge"})
            self.assertEqual({r.compatibility for r in off}, {None})

    def test_summary_counts_show_the_vetoes(self):
        expected = {k: sum(1 for c in CASES if c.expected == k) for k in COMPATIBILITY}
        for cfg, res in self.results.items():
            fd = res.summary["final_decisions"]
            self.assertEqual(fd["off"], {"merge": len(CASES)})
            for mode in GATED:
                self.assertEqual(fd[mode], {FINAL[k]: v for k, v in expected.items()}, (cfg, mode))

    def test_a_unifying_decision_is_never_left_on_a_blocked_pair(self):
        for cfg, res in self.results.items():
            for t in res.trace:
                c = BY_ID[t["eval_pair_id"]]
                if c.expected == "compatible":
                    continue
                for mode in GATED:
                    with self.subTest(config=cfg, mode=mode, case=c.group):
                        self.assertNotIn(t["final"][mode]["pairwise_decision"], ("merge", "supersede"))


class TestAHostThatWouldNotMergeIsNotAltered(unittest.TestCase):
    """Decision 6: the gate is host-first. It reports compatibility but never creates a link or a merge by itself."""

    def test_keep_both_stays_keep_both_for_every_condition(self):
        for cfg in "ABC":
            res = run_config(cfg, host_keeps_both())
            for t in res.trace:
                c = BY_ID[t["eval_pair_id"]]
                with self.subTest(config=cfg, case=c.group):
                    self.assertEqual(t["host"]["proposed"], "keep_both")
                    for mode in GATED:
                        g = t["gate"][mode]
                        self.assertEqual((g["proposed"], g["decision"], g["vetoed"]), ("keep_both", "keep_both", False))
                        self.assertEqual(g["compatibility"], c.expected)  # still reported
                        self.assertEqual(t["final"][mode]["pairwise_decision"], "no_merge")  # in particular no link

    def test_config_d_below_tau_is_not_altered(self):
        res = run_config("D", host_merges(), sim=0.5)
        for t in res.trace:
            self.assertEqual(t["host"]["proposed"], "keep_both")
            self.assertLessEqual(t["similarity"]["score"], TAU)
            for mode in GATED:
                self.assertEqual(t["final"][mode]["pairwise_decision"], "no_merge")
                self.assertFalse(t["gate"][mode]["vetoed"])

    def test_a_failed_host_answer_is_not_turned_into_a_decision_by_the_gate(self):
        llm = FakeLLM(judge=lambda a, b: "maybe?")
        res = run_config("C", llm, records=RECORDS[:6])
        self.assertTrue(all(t["status"] == "judge_failed" for t in res.trace))
        for mode in ALL_MODES:
            self.assertEqual(res.rows_by_mode[mode], [])


class TestGateUnitTruthTable(unittest.TestCase):
    """Closing the loop at the function level with real lexicon values: proposal x condition -> decision."""

    def test_all_proposals_for_one_real_pair_of_each_condition(self):
        inc = ({"kinship": "chachi"}, {"kinship": "mausi"})
        und = ({"kinship": "chachi"}, {})
        com = ({"kinship": "chachi"}, {"kinship": "chachi"})
        table = {  # proposal -> (incompatible, underdetermined, compatible)
            "merge": ("keep_both", "link_unresolved", "merge"),
            "supersede": ("keep_both", "link_unresolved", "supersede"),
            "keep_both": ("keep_both", "keep_both", "keep_both"),
            "link_unresolved": ("link_unresolved", "link_unresolved", "link_unresolved"),
        }
        for proposal, expected in table.items():
            for pair, want in zip((inc, und, com), expected):
                self.assertEqual(apply_gate(proposal, *pair).decision, want, (proposal, pair))
        self.assertEqual(DEFAULT_DISCRIMINATIVE_FEATURES, frozenset({"kinship", "register"}))


class TestWhatTheGateCannotDo(unittest.TestCase):
    """Cases that look like they should trigger the gate but cannot, asserted so the limits are on record."""

    def test_a_model_tag_on_a_term_the_lexicon_does_not_cover_is_never_consulted(self):
        # "bhaiya"/"bhai" are not in the 15-entry lexicon: lexicon+llm does not call the model for them, so the
        # pair stays compatible and the host's merge goes through -- a false merge DK-Mem cannot see.
        rec_ = rec("ep_900", "mera bhaiya Pune gaya", "mera bhai Pune gaya")
        llm = FakeLLM(judge=lambda a, b: ("merge", "same"),
                      model_distinction=lambda u: {"kinship": "bhaiya"} if "bhaiya" in u else {"kinship": "bhai"})
        res = run("C", [rec_], llm)
        t = res.trace[0]
        self.assertEqual(t["host"]["proposed"], "merge")
        for mode in GATED:
            g = t["gate"][mode]
            self.assertEqual((g["distinction_a"], g["distinction_b"], g["compatibility"]), ({}, {}, "compatible"))
            self.assertEqual(t["final"][mode]["pairwise_decision"], "merge")
        self.assertEqual(llm.calls["v4"], 0, "the model must not be called for an utterance with no ambiguous class")

    def test_a_model_tag_does_not_override_or_add_to_an_unambiguous_lexicon_result(self):
        # the model insists on "mausi" everywhere; the lexicon says chachi vs unmarked -> underdetermined, not incompatible
        rec_ = rec("ep_901", "meri chachi Pune mein rehti hai", "woh Pune mein rehti hai")
        llm = FakeLLM(judge=lambda a, b: ("merge", "same"), model_distinction=lambda u: {"kinship": "mausi"})
        t = run("A", [rec_], llm).trace[0]  # Config A calls V4 for every utterance (it is the host's extraction)
        for mode in GATED:
            self.assertEqual(t["gate"][mode]["compatibility"], "underdetermined")
            self.assertEqual(t["gate"][mode]["distinction_b"], {})
            self.assertEqual(t["final"][mode]["pairwise_decision"], "underdetermined_link")

    def test_name_variant_can_never_gate(self):
        self.assertNotIn("name_variant", DEFAULT_DISCRIMINATIVE_FEATURES)
        d = apply_gate("merge", {"name_variant": "Priya-latin"}, {"name_variant": "Priya-devanagari"})
        self.assertEqual((d.compatibility, d.decision, d.vetoed), ("compatible", "merge", False))
        d = apply_gate("merge", {"name_variant": "Priya-latin"}, {})  # one-sided is not underdetermined either
        self.assertEqual((d.compatibility, d.vetoed), ("compatible", False))
        # ...and the pipeline cannot even produce one: the lexicon has no name_variant entries, so the class is never
        # ambiguous and the model is never asked for it.
        rec_ = rec("ep_902", "mera naam Priya hai", "mera naam प्रिया hai")
        llm = FakeLLM(judge=lambda a, b: ("merge", "same"),
                      model_distinction=lambda u: {"name_variant": "Priya-latin" if "Priya" in u else "Priya-devanagari"})
        t = run("C", [rec_], llm).trace[0]
        for mode in GATED:
            self.assertEqual(t["gate"][mode]["distinction_a"], {})
            self.assertEqual(t["final"][mode]["pairwise_decision"], "merge")

    def test_terms_of_other_languages_are_invisible_to_the_lexicon(self):
        for lang, a, b in (("de", "kannst du das lesen", "können Sie das lesen"), ("tr", "amcam geldi", "dayım geldi")):
            res = run("C", [rec("ep_903", a, b, lang=lang)], host_merges())
            t = res.trace[0]
            self.assertEqual(t["host"]["proposed"], "merge", lang)
            for mode in GATED:
                self.assertEqual(t["gate"][mode]["compatibility"], "compatible", lang)
                self.assertEqual(t["final"][mode]["pairwise_decision"], "merge", lang)

    def test_inflected_register_forms_are_not_matched(self):
        # exact_word matching: "tumne"/"aapne" carry the register difference but the lexicon lists only tum/aap
        a, b = "Papa, tumne khana khaya?", "Papa, aapne khana khaya?"
        self.assertEqual((lexicon_distinction(LEX, a, "hi"), lexicon_distinction(LEX, b, "hi")), ({}, {}))
        t = run("C", [rec("ep_904", a, b)], host_merges()).trace[0]
        self.assertEqual(t["final"]["lexicon"]["pairwise_decision"], "merge")


class _V4Fails(FakeLLM):
    """FakeLLM whose V4 extraction returns unparseable output for utterances containing `` aur `` (two terms)."""

    def generate(self, prompts, params=None):
        out = super().generate(prompts, params)
        for i, p in enumerate(prompts):
            system = p[0]["content"]
            is_v4 = system.startswith("You convert one user utterance") and "exactly one key and nothing else" not in system
            if is_v4 and " aur " in p[-1]["content"]:  # (Config B's native prompt opens the same way: leave it alone)
                out[i] = "this is not json"
        return out


AMBIGUOUS = "meri chachi aur mausi Pune gayi"  # two different kinship values: the lexicon is ambiguous for kinship
OTHER_MAUSI = "meri mausi Delhi gayi"
OTHER_CHACHI = "meri chachi Delhi gayi"
OTHER_UNMARKED = "woh Delhi gayi"


class TestLexiconLlmConsultsTheModelOnlyForAmbiguousUtterances(unittest.TestCase):
    def test_the_ambiguous_utterance_really_is_ambiguous_and_lexicon_mode_takes_the_first_term(self):
        self.assertEqual(ambiguous_distinction_classes(LEX, AMBIGUOUS, "hi"), {"kinship"})
        self.assertEqual(lexicon_distinction(LEX, AMBIGUOUS, "hi"), {"kinship": "chachi"})  # documented tie-break
        for other in (OTHER_MAUSI, OTHER_CHACHI, OTHER_UNMARKED):
            self.assertEqual(ambiguous_distinction_classes(LEX, other, "hi"), set())

    def run_with_model_answer(self, config_id, other, answer):
        llm = FakeLLM(judge=lambda a, b: ("merge", "same"),
                      model_distinction=lambda u: dict(answer) if "aur" in u else {})
        res = run_config(config_id, llm, records=[rec("ep_910", AMBIGUOUS, other)])
        return res.trace[0], llm

    def test_the_models_value_replaces_the_lexicons_guess_in_every_config(self):
        for cfg in "ABCD":
            with self.subTest(config=cfg):
                # model says mausi, other side mausi -> compatible: lexicon+llm lets the merge through, lexicon vetoes
                t, llm = self.run_with_model_answer(cfg, OTHER_MAUSI, {"kinship": "mausi"})
                self.assertEqual(t["host"]["proposed"], "merge")
                self.assertEqual((t["gate"]["lexicon"]["compatibility"], t["final"]["lexicon"]["pairwise_decision"]),
                                 ("incompatible", "no_merge"))
                self.assertEqual((t["gate"]["lexicon+llm"]["compatibility"], t["final"]["lexicon+llm"]["pairwise_decision"]),
                                 ("compatible", "merge"))
                self.assertEqual(t["gate"]["lexicon+llm"]["distinction_a"], {"kinship": "mausi"})
                self.assertTrue(t["extraction"]["a"]["lexicon_ambiguous"] == ["kinship"])
                # model says chachi against a mausi side -> incompatible, vetoed
                t, _ = self.run_with_model_answer(cfg, OTHER_MAUSI, {"kinship": "chachi"})
                self.assertEqual((t["gate"]["lexicon+llm"]["compatibility"], t["final"]["lexicon+llm"]["pairwise_decision"]),
                                 ("incompatible", "no_merge"))
                # model has no kinship value -> the class is absent, the other side is marked -> underdetermined, linked
                t, _ = self.run_with_model_answer(cfg, OTHER_CHACHI, {})
                self.assertEqual(t["gate"]["lexicon+llm"]["distinction_a"], {})
                self.assertEqual((t["gate"]["lexicon+llm"]["compatibility"], t["final"]["lexicon+llm"]["pairwise_decision"]),
                                 ("underdetermined", "underdetermined_link"))
                self.assertTrue(t["gate"]["lexicon+llm"]["vetoed"])

    def test_the_model_is_called_for_the_gate_only_on_ambiguous_utterances_outside_config_a(self):
        for cfg in "BC":
            _, llm = self.run_with_model_answer(cfg, OTHER_MAUSI, {"kinship": "mausi"})
            self.assertEqual(llm.calls["v4"], 1, cfg)  # one ambiguous utterance, none for the other side

    def test_an_ambiguous_utterance_whose_v4_call_failed_gets_no_decision_in_lexicon_llm_and_no_fallback(self):
        for cfg in "BCD":
            with self.subTest(config=cfg):
                llm = _V4Fails(judge=lambda a, b: ("merge", "same"))
                res = run_config(cfg, llm, records=[rec("ep_911", AMBIGUOUS, OTHER_MAUSI)])
                t = res.trace[0]
                self.assertEqual(t["status"], "ok")
                self.assertEqual(t["host"]["proposed"], "merge")
                self.assertEqual(t["final"]["off"]["pairwise_decision"], "merge")
                self.assertEqual(t["final"]["lexicon"]["pairwise_decision"], "no_merge")  # lexicon-only veto still works
                self.assertIsNone(t["final"]["lexicon+llm"])  # not "compatible by default", not lexicon-only: no decision
                self.assertIn("skipped", t["gate"]["lexicon+llm"])
                self.assertEqual(res.rows_by_mode["lexicon+llm"], [])
                self.assertEqual(len(res.rows_by_mode["lexicon"]), 1)
        res = run_config("A", _V4Fails(judge=lambda a, b: ("merge", "same")), records=[rec("ep_912", AMBIGUOUS, OTHER_MAUSI)])
        self.assertEqual(res.trace[0]["status"], "extraction_failed")  # in A the V4 call is the host's own extraction
        self.assertTrue(all(rows == [] for rows in res.rows_by_mode.values()))


class TestStoreRunnerReachesTheSameDecisions(unittest.TestCase):
    """The store-based runner shares the gate; on these pairs it must decide exactly as the A-D pipeline does."""

    def test_store_and_pipeline_agree_on_every_case_in_every_mode(self):
        from dkmem.memory.lexicon import load_lexicon as _load
        from dkmem.store.runner import build_store_run_config, run_store_attribution

        for cfg in ("C", "D"):
            llm = host_merges()
            pipe = run_config(cfg, llm)
            sim = sim_metric(0.99) if cfg == "D" else None
            params = GenerationParams(seed=0)
            scfg = build_store_run_config(cfg, ALL_MODES, RECORDS, generator=llm, lexicon_path=LEXICON_PATH, params=params,
                                          similarity=sim, tau=TAU if cfg == "D" else None)
            store = run_store_attribution(RECORDS, cfg, _load(LEXICON_PATH), scfg, generator=llm, params=params,
                                          similarity=sim, tau=TAU if cfg == "D" else None)
            for mode in ALL_MODES:
                a = {r.record_id: (r.decision, r.compatibility) for r in pipe.rows_by_mode[mode]}
                b = {r.record_id: (r.decision, r.compatibility) for r in store.rows_by_mode[mode]}
                with self.subTest(config=cfg, mode=mode):
                    self.assertEqual(a, b)
                    self.assertEqual(len(a), len(CASES))
            # a blocked pair is not merged in the store either: the entries stay separate
            blocked = next(c for c in CASES if c.expected == "incompatible")
            stores = {s.conv_id: s for s in store.stores_by_mode["lexicon"]}
            self.assertEqual(len(stores[blocked.cid].active()), 2)
            merged = next(c for c in CASES if c.expected == "compatible")
            stores_off = {s.conv_id: s for s in store.stores_by_mode["off"]}
            self.assertEqual(len(stores_off[blocked.cid].active()), 1)  # without the gate the same pair was merged
            self.assertEqual(len({s.conv_id: s for s in store.stores_by_mode["lexicon"]}[merged.cid].active()), 1)


if __name__ == "__main__":
    unittest.main()
