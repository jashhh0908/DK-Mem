"""Tier-2 identity judge (``merge_judge_entity_t2_v1``), its frozen validation set, the v1.1m mirror arm and their kernels.

Real code under test: eval/tier2 (judge_entity, judge_host, run_tier2, contract, judge_provenance, judge_validation, mirror_v1_1),
eval/harness (vendored), validation/tier2_entity/build_kernels.py. Fake: the generator and the embedder. No model is loaded and the
Tier-1 answer key is never read. Tests that need evaluator-side Tier-2 files (pair specs, mirror gold) skip when they are absent.
"""

import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).parent.parent
EVAL = REPO / "eval"
TIER2 = EVAL / "tier2"
MIRROR = TIER2 / "mirror_v1_1"
JV = TIER2 / "judge_validation"
TIER1_JUDGE_SHA256 = "3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff"
ENTITY_SHA256 = "6f85be1900a9e95586426b084e498dfec94bb46df25ad2db2eb80bd1e273c2ed"

from dkmem.backends.llm import GenerationParams  # noqa: E402
from dkmem.memory.judge import JUDGE_PROMPTS, MERGE_JUDGE_V1  # noqa: E402
from dkmem.memory.prompts import PromptTemplate  # noqa: E402

H = {}


def setUpModule():
    if not ((TIER2 / "judge_entity.py").exists() and (EVAL / "harness" / "hosts.py").exists()):
        raise unittest.SkipTest("eval/tier2 is not present in this checkout")
    sys.path.insert(0, str(EVAL))
    import harness.similarity  # noqa: F401
    import tier2.contract as contract
    import tier2.judge_entity as je
    import tier2.judge_host as jh
    import tier2.judge_provenance as prov
    import tier2.run_tier2 as run
    import tier2.judge_validation.score as score
    import tier2.judge_validation.run_cases as run_cases
    H.update(contract=contract, je=je, jh=jh, prov=prov, run=run, score=score, run_cases=run_cases, similarity=sys.modules["harness.similarity"])


def tearDownModule():
    for name in list(sys.modules):
        if name.split(".")[0] in ("harness", "evaluator", "tier2", "diagnostics"):
            sys.modules.pop(name, None)
    if str(EVAL) in sys.path:
        sys.path.remove(str(EVAL))


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


class FakeEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 16
            for i, ch in enumerate(t):
                v[i % 16] += (ord(ch) % 17) + 1
            out.append(v)
        return out

    def describe(self):
        return {"similarity_backend": "bge-m3", "similarity_model_id": "BAAI/bge-m3", "similarity_model_revision": "fakerev",
                "similarity_requested_revision": None, "similarity_dtype": "float16", "similarity_device": "cuda:0",
                "similarity_max_seq_length": 512, "similarity_embed_batch_size": 32, "similarity_is_placeholder": False}


class RecordingJudge:
    """Records the system prompt of every judge call; answers merge for every third pair (both decisions occur)."""
    backbone = "fake-backbone"

    def __init__(self):
        self.systems, self.n = [], 0

    def generate(self, prompts, params):
        out = []
        for m in prompts:
            if m[0]["content"].startswith("You answer questions"):
                out.append("unknown")
                continue
            self.systems.append(m[0]["content"])
            self.n += 1
            out.append(json.dumps({"decision": "merge" if self.n % 3 == 0 else "keep_both", "reason": "r"}))
        return out

    def run_info(self, params):
        return {"backbone": self.backbone, "generation_params": {"max_new_tokens": params.max_new_tokens}}


# ---------------------------------------------------------------------------------------------------------------- judge
class TestIdentityJudgeIsSeparatelyVersioned(unittest.TestCase):
    def test_id_and_sha_are_pinned(self):
        p = H["je"].MERGE_JUDGE_ENTITY_T2_V1
        self.assertEqual((p.prompt_id, p.sha256), ("merge_judge_entity_t2_v1", ENTITY_SHA256))
        fz = json.loads((JV / "FREEZE.json").read_text(encoding="utf-8"))
        self.assertEqual((fz["prompt_id"], fz["prompt_sha256"]), (p.prompt_id, p.sha256))

    def test_tier1_judge_is_untouched_and_does_not_know_the_tier2_judge(self):
        self.assertEqual(MERGE_JUDGE_V1.sha256, TIER1_JUDGE_SHA256)
        self.assertNotIn("merge_judge_entity_t2_v1", JUDGE_PROMPTS)
        self.assertNotIn("entity", (REPO / "dkmem" / "memory" / "judge.py").read_text(encoding="utf-8"))

    def test_same_output_format_and_parser_as_the_tier1_judge(self):
        p = H["je"].MERGE_JUDGE_ENTITY_T2_V1
        self.assertEqual((p.output_keys, p.user_template), (MERGE_JUDGE_V1.output_keys, MERGE_JUDGE_V1.user_template))
        self.assertIn('{"decision": "merge" or "keep_both", "reason": string}', p.system)

    def test_prompt_carries_no_distinction_hint(self):
        p = H["je"].MERGE_JUDGE_ENTITY_T2_V1
        text = (p.system + " " + " ".join(q + a for q, a in p.examples)).lower()
        for hint in ("hindi", "german", "language", "kinship", "register", "formal", "polite", "script", "spelling", "devar", "jeth",
                     "mausi", "chachi", "nani", "dadi", " du ", " sie ", " tum ", " aap ", "name"):
            self.assertNotIn(hint, text, hint)
        self.assertIsNone(re.search(r"[ऀ-ॿ]", text))

    def test_policy_sentences(self):
        s = H["je"].MERGE_JUDGE_ENTITY_T2_V1.system
        self.assertIn("same specific person, thing or event", s)
        self.assertIn("including a fact that has changed", s)
        self.assertIn("If the entries do not show that they are about the same specific subject, keep both.", s)
        self.assertIn("their only shared subject is the user", s)

    def test_host_accepts_exactly_the_two_frozen_judges(self):
        jh = H["jh"]
        for p in (MERGE_JUDGE_V1, H["je"].MERGE_JUDGE_ENTITY_T2_V1):
            jh.check_frozen_prompt(p)
        e = H["je"].MERGE_JUDGE_ENTITY_T2_V1
        for bad in (PromptTemplate(e.prompt_id, e.system + " ", e.user_template, e.output_keys, e.examples),
                    PromptTemplate(e.prompt_id, e.system, e.user_template, e.output_keys, e.examples)):   # equal text, not the object
            with self.assertRaises(ValueError):
                jh.check_frozen_prompt(bad)

    def test_host_passes_the_identity_prompt_to_the_model(self):
        gen = RecordingJudge()
        gen.n = 2                                         # first call answers merge
        host = H["jh"].FrozenJudgeHost(gen, GenerationParams(max_new_tokens=128, do_sample=False), H["je"].MERGE_JUDGE_ENTITY_T2_V1,
                                       H["similarity"].build_metric(lambda: FakeEmbedder()), "surface")
        a, b = mock.Mock(surface="Kavya Gwalior mein rehti hai", pair_id=None), mock.Mock(surface="Kavya ka birthday May mein hai")
        out = host.decide_many([(a, b)])
        self.assertEqual(gen.systems, [H["je"].MERGE_JUDGE_ENTITY_T2_V1.system])
        self.assertEqual((out[0].proposal, out[0].threshold), ("merge", None))
        self.assertEqual(host.log[0]["prompt_sha256"], ENTITY_SHA256)

    def test_provenance_labels_the_identity_judge(self):
        c = H["prov"].classify({"judge_prompt_id": "merge_judge_entity_t2_v1", "judge_prompt_sha256": ENTITY_SHA256,
                                "generation_run_info": {"generation_params": {"max_new_tokens": 128}}})
        self.assertTrue(c["tier2_identity_judge"] and c["recognised"])
        self.assertFalse(c["frozen_tier1"] or c["comparable_with_tier1_config_c"])
        self.assertIsNone(c["row_threshold_convention"])
        d = H["prov"].describe({"judge_prompt_id": "merge_judge_entity_t2_v1", "judge_prompt_sha256": ENTITY_SHA256}, [])
        self.assertIn("TIER-2 IDENTITY JUDGE", d["classification"])


# ---------------------------------------------------------------------------------------------------------------- validation set
class TestJudgeValidationSetIsFrozen(unittest.TestCase):
    def test_freeze_verifies(self):
        fz = H["score"].verify_freeze()
        self.assertEqual(fz["status"], "FROZEN")
        self.assertEqual(fz["n_cases"], len(jl(JV / "cases_v1.jsonl")))

    def test_case_set_covers_both_directions_and_every_required_category(self):
        cases = jl(JV / "cases_v1.jsonl")
        cats = {c["category"] for c in cases}
        for need in ("same_fact", "same_entity_compatible", "same_entity_conflict", "different_entity_similar",
                     "different_entity_different_attribute", "user_only", "ambiguous"):
            self.assertIn(need, cats)
        self.assertEqual({c["expected"] for c in cases}, {"merge", "keep_both"})
        self.assertGreaterEqual(len({c["language"] for c in cases}), 4)
        self.assertTrue(all(c["expected"] == "keep_both" for c in cases if c["category"] == "ambiguous"))
        self.assertTrue(all(c["expected"] == "merge" for c in cases if c["category"] == "same_entity_conflict"))

    def test_cases_are_disjoint_from_the_benchmark(self):
        specs_p = TIER2 / "banks" / "pair_specs.json"
        if not specs_p.exists():
            self.skipTest("evaluator-side pair specs absent")
        specs = json.loads(specs_p.read_text(encoding="utf-8"))["specs"]
        bank = json.loads((TIER2 / "banks" / "distractors.json").read_text(encoding="utf-8"))["user_turns"]
        bench = {s["text_a"] for s in specs} | {s["text_b"] for s in specs} | {t for v in bank.values() for t in v}
        mirror_p = MIRROR / "banks" / "pair_specs_mirror.json"
        if mirror_p.exists():
            bench |= {s["text_b"] for s in json.loads(mirror_p.read_text(encoding="utf-8"))["specs"]}
        cases = jl(JV / "cases_v1.jsonl")
        self.assertFalse({c[k] for c in cases for k in ("text_a", "text_b")} & bench)
        names = {w for s in specs for w in re.findall(r"\b[A-Z][a-z]+\b", s["text_a"] + " " + s["text_b"])
                 if w not in ("Meri", "Mere", "Maa", "Papa", "Nani", "Chefin")}
        case_words = {w for c in cases for w in re.findall(r"\b[A-Z][a-z]+\b", c["text_a"] + " " + c["text_b"])}
        # closed classes are shared by design (FREEZE.json "disjointness"): months, weekdays, pronouns
        closed = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
                  "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag", "Sie", "Ihr", "Ihre"}
        self.assertFalse((names - closed) & case_words, (names - closed) & case_words)

    def _outs(self, decide):
        cases = jl(JV / "cases_v1.jsonl")
        return cases, [{"case_id": c["case_id"], "order": o, "raw_output": json.dumps({"decision": decide(c, o), "reason": "r"}),
                        "output_tokens": 20, "max_new_tokens": 128, "prompt_id": "merge_judge_entity_t2_v1", "prompt_sha256": ENTITY_SHA256}
                       for c in cases for o in ("ab", "ba")]

    def _score(self, decide):
        crit = json.loads((JV / "criteria_v1.json").read_text(encoding="utf-8"))
        crit["prompt_sha256"] = ENTITY_SHA256
        cases, outs = self._outs(decide)
        return H["score"].score(outs, cases, crit)

    def test_oracle_passes_and_degenerate_or_shortcut_judges_fail(self):
        self.assertTrue(self._score(lambda c, o: c["expected"])["accepted"])
        self.assertFalse(self._score(lambda c, o: "keep_both")["accepted"])
        self.assertFalse(self._score(lambda c, o: "merge")["accepted"])
        shortcut = self._score(lambda c, o: {"same_attribute": "keep_both", "different_attribute": "merge"}.get(c["attribute_pattern"], c["expected"]))
        self.assertFalse(shortcut["gates"]["C8_attribute_shortcut"]["pass"])
        flip = self._score(lambda c, o: c["expected"] if o == "ab" else ("merge" if c["expected"] == "keep_both" else "keep_both"))
        self.assertFalse(flip["gates"]["C6_order_consistency"]["pass"])

    def test_probe_errors_are_reported_not_gated(self):
        r = self._score(lambda c, o: c["expected"] if c["role"] == "control" else ("merge" if c["expected"] == "keep_both" else "keep_both"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["reported_not_gated"]["probes"]["correct"], 0)

    def test_parse_failures_count_as_failures_and_wrong_prompt_is_refused(self):
        cases, outs = self._outs(lambda c, o: c["expected"])
        outs[0]["raw_output"] = '{"decision": "merge"'
        crit = json.loads((JV / "criteria_v1.json").read_text(encoding="utf-8"))
        crit["prompt_sha256"] = ENTITY_SHA256
        self.assertFalse(H["score"].score(outs, cases, crit)["gates"]["C1_parse"]["pass"])
        outs[0]["prompt_sha256"] = TIER1_JUDGE_SHA256
        with self.assertRaises(ValueError):
            H["score"].score(outs, cases, crit)

    def test_runner_refuses_labelled_rows_and_judges_both_orders(self):
        rc = H["run_cases"]
        self.assertEqual([c[1] for c in rc.judge_calls([{"case_id": "x", "text_a": "A", "text_b": "B"}])], ["ab", "ba"])
        with self.assertRaises(ValueError):
            rc.judge_calls([{"case_id": "x", "text_a": "A", "text_b": "B", "expected": "merge"}])


# ---------------------------------------------------------------------------------------------------------------- mirror arm
def _mirror_available():
    return (MIRROR / "data" / "tier2_gold.jsonl").exists() and (TIER2 / "data" / "tier2_gold.jsonl").exists()


@unittest.skipUnless(_mirror_available(), "evaluator-side Tier-2 files absent")
class TestMirrorArm(unittest.TestCase):
    def test_validator_passes(self):
        from tier2.mirror_v1_1 import validate_mirror as V
        R = V.validate()
        self.assertEqual(R.failures, [])
        self.assertGreater(R.n, 800)

    def test_v1_0_files_are_unchanged(self):
        fz = json.loads((TIER2 / "FROZEN_v1.0.json").read_text(encoding="utf-8"))["sha256"]
        for rel in ("data/tier2_conversations.jsonl", "data/tier2_qa_questions.jsonl", "data/tier2_gold.jsonl"):
            self.assertEqual(hashlib.sha256((TIER2 / rel).read_bytes()).hexdigest(), fz[rel], rel)

    def test_validator_catches_a_changed_distractor_and_a_changed_label(self):
        from tier2.mirror_v1_1 import validate_mirror as V
        with tempfile.TemporaryDirectory() as d:
            d = Path(d) / "data"
            d.mkdir()
            for n in V.M.FILES:
                (d / n).write_bytes((MIRROR / "data" / n).read_bytes())
            conv = jl(d / "tier2_conversations.jsonl")
            meta = {m["conversation_id"]: m for m in jl(d / "tier2_injection_metadata.jsonl")}
            c0 = conv[0]
            victim = meta[c0["conversation_id"]]["distractor_user_turn_ids"][0]
            for s in c0["sessions"]:
                for t in s["turns"]:
                    if t["turn_id"] == victim:
                        t["text"] += " zz"
            (d / "tier2_conversations.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in conv), encoding="utf-8", newline="\n")
            gold = jl(d / "tier2_gold.jsonl")
            gold[0]["relation"] = "same" if gold[0]["relation"] == "different" else "different"
            (d / "tier2_gold.jsonl").write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold), encoding="utf-8", newline="\n")
            with mock.patch.object(V, "FROZEN", Path(d) / "absent.json"):
                fails = V.validate(d).failures
        self.assertTrue(any("exactly member b" in f for f in fails), fails[:5])
        self.assertTrue(any("labels differ" in f for f in fails), fails[:5])

    def test_runner_and_contract_on_the_mirror_arm_with_the_identity_judge(self):
        run, contract = H["run"], H["contract"]
        convs = [c["conversation_id"] for c in jl(MIRROR / "data" / "tier2_conversations.jsonl")[:2]]
        gen = RecordingJudge()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pt = d / "pt.jsonl"
            pt.write_text(contract.serialize_passthrough(contract.build_passthrough_ids(MIRROR / "data")), encoding="utf-8", newline="\n")
            with mock.patch.object(run, "build_models", lambda a: (H["similarity"].build_metric(lambda: FakeEmbedder()), gen)):
                rc = run.main(["--data-dir", str(MIRROR / "data"), "--frozen", str(MIRROR / "FROZEN_v1.1m.json"), "--passthrough", str(pt),
                               "--conversations", ",".join(convs), "--judge-prompt", "merge_judge_entity_t2_v1", "--configs", "C:off,D:lexicon",
                               "--out", str(d / "runs"), "--run-prefix", "t2m-test"])
            self.assertEqual(rc, 0)
            self.assertTrue(gen.systems and set(gen.systems) == {H["je"].MERGE_JUDGE_ENTITY_T2_V1.system})
            man = json.loads((d / "runs" / "t2m-test-store-surface-only-C_seed0" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((man["judge_prompt_id"], man["judge_prompt_sha256"]), ("merge_judge_entity_t2_v1", ENTITY_SHA256))
            self.assertEqual(man["tier2"]["benchmark_version"], "1.1m")
            self.assertIn("data/tier2_gold.jsonl", man["tier2"]["frozen_files_verified"])
            for rid in ("t2m-test-store-surface-only-C_seed0", "t2m-test-dk-mem-lexicon-D_seed0"):
                rep = contract.validate_run(d / "runs" / rid, MIRROR / "data", frozen_file=MIRROR / "FROZEN_v1.1m.json")
                self.assertTrue(rep["ok"], (rid, rep["errors"][:3]))
                self.assertEqual((rep["benchmark_version"], rep["counts"]["qa_answers"], rep["counts"]["injected_pairs"]), ("1.1m", 0, 2))
            self.assertTrue(contract.validate_run(d / "runs" / "t2m-test-store-surface-only-C_seed0", MIRROR / "data",
                                                  frozen_file=MIRROR / "FROZEN_v1.1m.json")["judge"]["tier2_identity_judge"])
            # validating the mirror run against the v1.0 frozen file is an error, not a silent pass
            wrong = contract.validate_run(d / "runs" / "t2m-test-store-surface-only-C_seed0", MIRROR / "data")
            self.assertIn("manifest_frozen", {e["code"] for e in wrong["errors"]})

    def test_the_mirror_frozen_file_is_refused_for_a_changed_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "data").mkdir()
            for n in ("tier2_conversations.jsonl", "tier2_qa_questions.jsonl"):
                (d / "data" / n).write_bytes((MIRROR / "data" / n).read_bytes())
            (d / "data" / "tier2_conversations.jsonl").write_bytes((MIRROR / "data" / "tier2_conversations.jsonl").read_bytes() + b"\n")
            with self.assertRaises(RuntimeError):
                H["run"].verify_frozen(d / "data", MIRROR / "FROZEN_v1.1m.json")


# ---------------------------------------------------------------------------------------------------------------- kernels
def _builder():
    spec = importlib.util.spec_from_file_location("_t2_entity_build", REPO / "validation" / "tier2_entity" / "build_kernels.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


class TestKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = _builder()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_judge_validation_payload_has_texts_only(self):
        files, nb, payload, tree, ev, _ = self.B.build_judge_validation()
        rows = [json.loads(l) for l in files["inputs/judge_cases.jsonl"].splitlines()]
        self.assertTrue(rows and all(set(r) == {"case_id", "text_a", "text_b"} for r in rows))
        for n in files:
            self.assertFalse(any(p in n for p in ("cases_v1", "criteria", "FREEZE", "score.py", "make_cases")), n)
        self.assertNotIn("keep_both", files["inputs/judge_cases.jsonl"])
        out = Path(self.tmp.name) / "jv"
        self.assertEqual(self.B.write({"kind": "judge-validation"}, "u", "s", str(out)), 0)
        self.assertEqual(self.B.check(str(out)), 0)
        meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["is_private"], meta["enable_internet"], meta["machine_shape"], meta["dataset_sources"]), (True, True, "NvidiaTeslaT4", []))

    @unittest.skipUnless(_mirror_available(), "evaluator-side Tier-2 files absent")
    def test_tier2_payloads_for_both_arms(self):
        for arm in ("v1.0", "mirror"):
            files, nb, payload, tree, ev, qmap = self.B.build_tier2(arm, None, True, "t2entity-smoke-" + arm)
            self.assertIn("eval/tier2/judge_entity.py", files)
            self.assertIn("--judge-prompt", ev["run_args"])
            self.assertEqual(ev["run_args"][ev["run_args"].index("--judge-prompt") + 1], "merge_judge_entity_t2_v2")
            self.assertEqual(len(ev["conversations"]), 3)
            for n in files:
                low = n.lower()
                self.assertFalse(any(p in low for p in ("tier2_gold", "injection_metadata", "qa_reference", "pair_specs", "answer_key", "mirror_bank")), n)
            frozen_rel = self.B.ARMS[arm]["frozen"]
            self.assertEqual(sorted(json.loads(files[frozen_rel])), sorted(self.B.S.FROZEN_PROJECTION_KEYS))
            self.assertEqual(bool(qmap), arm == "v1.0")
            out = Path(self.tmp.name) / ("t2-" + arm)
            spec = {"kind": "tier2", "arm": arm, "run_prefix": "t2entity-smoke-" + arm, "smoke": True, "conversations_arg": None}
            self.assertEqual(self.B.write(spec, "u", "s", str(out)), 0)
            self.assertEqual(self.B.check(str(out)), 0)

    def test_a_tampered_notebook_is_stale(self):
        out = Path(self.tmp.name) / "jv2"
        self.assertEqual(self.B.write({"kind": "judge-validation"}, "u", "s", str(out)), 0)
        nb = json.loads((out / self.B.NOTEBOOK_NAME).read_text(encoding="utf-8"))
        nb["cells"][-1]["source"].append("\n# edited")
        (out / self.B.NOTEBOOK_NAME).write_text(json.dumps(nb), encoding="utf-8")
        self.assertEqual(self.B.check(str(out)), 1)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------------------------------------------- v2
ENTITY_V2_SHA256 = "36146ec054a700796a785a5a942c0fdaaa33a1e100103c8a8410d596c146f149"
JV2 = TIER2 / "judge_validation_v2"


class TestIdentityJudgeV2(unittest.TestCase):
    def test_v2_is_pinned_and_v1_is_unchanged(self):
        je = H["je"]
        self.assertEqual((je.MERGE_JUDGE_ENTITY_T2_V2.prompt_id, je.MERGE_JUDGE_ENTITY_T2_V2.sha256), ("merge_judge_entity_t2_v2", ENTITY_V2_SHA256))
        self.assertEqual(je.MERGE_JUDGE_ENTITY_T2_V1.sha256, ENTITY_SHA256)
        self.assertEqual(MERGE_JUDGE_V1.sha256, TIER1_JUDGE_SHA256)
        self.assertNotIn("merge_judge_entity_t2_v2", JUDGE_PROMPTS)
        fz = json.loads((JV2 / "FREEZE.json").read_text(encoding="utf-8"))
        self.assertEqual((fz["prompt_id"], fz["prompt_sha256"]), ("merge_judge_entity_t2_v2", ENTITY_V2_SHA256))

    def test_v2_carries_no_distinction_hint_and_defines_entity_versus_fact(self):
        p = H["je"].MERGE_JUDGE_ENTITY_T2_V2
        text = (p.system + " " + " ".join(q + a for q, a in p.examples)).lower()
        for hint in ("hindi", "german", "language", "kinship", "register", "formal", "polite", "script", "spelling", "devar", "jeth",
                     "mausi", "chachi", "nani", "dadi", " du ", " sie ", " tum ", " aap "):
            self.assertNotIn(hint, text, hint)
        for need in ("ENTITY", "FACTS", "Compare only the two entities", "contradict", "The order of the two entries does not matter",
                     "does not identify one specific entity"):
            self.assertIn(need, p.system)

    def test_v2_examples_are_balanced_and_parse_with_the_frozen_parser(self):
        from dkmem.memory.judge import parse_judge_output
        decisions = [parse_judge_output(a).decision for _, a in H["je"].MERGE_JUDGE_ENTITY_T2_V2.examples]
        self.assertEqual(sorted(decisions), ["keep_both"] * 3 + ["merge"] * 3)

    def test_host_runner_and_contract_accept_v2(self):
        jh = H["jh"]
        jh.check_frozen_prompt(H["je"].MERGE_JUDGE_ENTITY_T2_V2)
        self.assertIn("merge_judge_entity_t2_v2", jh.JUDGES_BY_ID)
        c = H["prov"].classify({"judge_prompt_id": "merge_judge_entity_t2_v2", "judge_prompt_sha256": ENTITY_V2_SHA256})
        self.assertTrue(c["tier2_identity_judge"])
        self.assertFalse(H["prov"].classify({"judge_prompt_id": "merge_judge_entity_t2_v2", "judge_prompt_sha256": ENTITY_SHA256})["recognised"])

    def test_v2_freeze_verifies_and_criteria_equal_v1(self):
        from tier2.judge_validation_v2 import score_v2
        self.assertEqual(score_v2.verify_freeze()["status"], "FROZEN")
        c1 = json.loads((JV / "criteria_v1.json").read_text(encoding="utf-8"))
        c2 = json.loads((JV2 / "criteria_v2.json").read_text(encoding="utf-8"))
        self.assertEqual((c1["gates"], c1["decoding"], c1["parser"]), (c2["gates"], c2["decoding"], c2["parser"]))

    def test_v2_cases_are_independent_of_v1_and_the_benchmark(self):
        v1 = jl(JV / "cases_v1.jsonl")
        v2 = jl(JV2 / "cases_v2.jsonl")
        self.assertEqual(len(v2), 77)
        self.assertEqual([c["category"] for c in v1], [c["category"] for c in v2])         # same structure, so the same criteria apply
        s1 = {c[k] for c in v1 for k in ("text_a", "text_b")}
        s2 = {c[k] for c in v2 for k in ("text_a", "text_b")}
        self.assertFalse(s1 & s2)
        closed = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
                  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Montag", "Dienstag", "Mittwoch", "Donnerstag",
                  "Freitag", "Samstag", "Sonntag", "Juni", "Juli", "Mai", "März", "Oktober", "Sie", "Ihr", "Ihre"}
        names = lambda texts: {w for t in texts for w in re.findall(r"\b[A-Z][a-z]+\b", t)} - closed
        common_words = {"Meri", "Mere", "Mera", "Meine", "Mein", "Unser", "My", "The", "Our", "I", "Die", "Das", "Der", "Ein", "Eine", "Ich",
                        "Geburtstag", "Kollege", "Nachbarin", "Heute", "Kal", "Aaj", "Is", "Main", "Maine", "Mujhe", "Hamara", "Hamare"}
        self.assertFalse((names(s1) & names(s2)) - common_words, (names(s1) & names(s2)) - common_words)
        specs_p = TIER2 / "banks" / "pair_specs.json"
        if specs_p.exists():
            specs = json.loads(specs_p.read_text(encoding="utf-8"))["specs"]
            bench = {s["text_a"] for s in specs} | {s["text_b"] for s in specs}
            self.assertFalse(s2 & bench)
            self.assertFalse((names(bench) & names(s2)) - common_words - {"Urlaub", "Wohnung", "Inspektion"})

    def test_v2_kernel_payload_has_texts_only_and_selects_v2(self):
        B = _builder()
        files, nb, payload, tree, ev, _ = B.build_judge_validation("v2")
        rows = [json.loads(l) for l in files["inputs/judge_cases.jsonl"].splitlines()]
        self.assertTrue(rows[0]["case_id"].startswith("jv2_") and all(set(r) == {"case_id", "text_a", "text_b"} for r in rows))
        self.assertFalse(any("judge_validation_v2" in n or "cases_v" in n for n in files))
        self.assertEqual(ev["run_args"][-2:], ["--prompt", "merge_judge_entity_t2_v2"])
        self.assertEqual(B.ENTITY_JUDGE_ID, "merge_judge_entity_t2_v2")        # Tier-2 kernels use v2


# ---------------------------------------------------------------------------------------------------------------- 7B + Config D
JV7 = TIER2 / "judge_validation_7b"


class TestSevenBValidationSet(unittest.TestCase):
    def test_freeze_criteria_and_backbone_documentation(self):
        from tier2.judge_validation_7b import score_7b
        fz = score_7b.verify_freeze()
        self.assertEqual((fz["prompt_id"], fz["prompt_sha256"]), ("merge_judge_entity_t2_v2", ENTITY_V2_SHA256))   # v2 prompt, unchanged
        c1 = json.loads((JV / "criteria_v1.json").read_text(encoding="utf-8"))
        c7 = json.loads((JV7 / "criteria_7b.json").read_text(encoding="utf-8"))
        self.assertEqual((c1["gates"], c1["parser"]), (c7["gates"], c7["parser"]))
        self.assertEqual({k: v for k, v in c1["decoding"].items() if not k.startswith("model")},
                         {k: v for k, v in c7["decoding"].items() if not k.startswith("model") and k != "placement"})
        self.assertEqual((c7["decoding"]["model_id"], c7["decoding"]["model_revision"]), ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"))
        self.assertIn("not comparable with Tier-1 Config C", c7["backbone_change"])

    def test_cases_are_independent_of_v1_v2_and_the_benchmark(self):
        sets = [jl(JV / "cases_v1.jsonl"), jl(JV2 / "cases_v2.jsonl"), jl(JV7 / "cases_7b.jsonl")]
        self.assertEqual([c["category"] for c in sets[0]], [c["category"] for c in sets[2]])
        t = [{c[k] for c in s for k in ("text_a", "text_b")} for s in sets]
        self.assertFalse(t[2] & (t[0] | t[1]))
        common = {"Meri", "Mere", "Mera", "Meine", "Mein", "Unser", "Unsere", "My", "The", "Our", "I", "Die", "Der", "Das", "Ein", "Eine",
                  "Ich", "Heute", "Kal", "Aaj", "Is", "Main", "Maine", "Mujhe", "Hamare", "Hamara", "Geburtstag", "Kollege", "Nachbarin",
                  "Didi", "Herr", "Morgen", "Morgens", "Woche", "Wochenende", "Office", "Am", "Sie", "Ihr", "Ihre", "A", "One"}
        closed = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
                  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Wednesdays", "Tuesdays", "Saturdays",
                  "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag", "Juni", "Juli", "Mai", "März", "Oktober"}
        names = lambda ts: {w for x in ts for w in re.findall(r"\b[A-ZÄÖÜ][\wäöüß]+\b", x)} - common - closed
        self.assertFalse(names(t[2]) & (names(t[0]) | names(t[1])), names(t[2]) & (names(t[0]) | names(t[1])))

    def test_contradiction_counter(self):
        from tier2.judge_validation_7b.score_7b import contradictions
        outs = [{"raw_output": json.dumps({"reason": "A is about X; B is about Y; different entities.", "decision": "merge"})},
                {"raw_output": json.dumps({"reason": "A is about X; B is about X; same entity.", "decision": "merge"})},
                {"raw_output": json.dumps({"reason": "A is about a friend; not identifiable.", "decision": "merge"})}]
        self.assertEqual(contradictions(outs)["contradictions"], 2)

    def test_kernel_runs_v2_on_the_pinned_7b_sharded_with_texts_only(self):
        B = _builder()
        files, nb, payload, tree, ev, _ = B.build_judge_validation("7b")
        a = ev["run_args"]
        self.assertEqual(a[a.index("--model-id") + 1], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(a[a.index("--model-revision") + 1], "a09a35458c702b33eeacc393d103063234e8bc28")
        self.assertEqual(a[a.index("--prompt") + 1], "merge_judge_entity_t2_v2")
        self.assertIn("--shard", a)
        rows = [json.loads(l) for l in files["inputs/judge_cases.jsonl"].splitlines()]
        self.assertTrue(rows[0]["case_id"].startswith("jv7_") and all(set(r) == {"case_id", "text_a", "text_b"} for r in rows))
        self.assertFalse(any("judge_validation_7b" in n for n in files))


@unittest.skipUnless(_mirror_available(), "evaluator-side Tier-2 files absent")
class TestConfigDOnlyKernels(unittest.TestCase):
    def test_d_only_kernels_run_no_judge_and_check_tau(self):
        B = _builder()
        for arm in ("v1.0", "mirror"):
            files, nb, payload, tree, ev, qmap = B.build_tier2(arm, None, True, "t2d-test", B.D_CONFIGS)
            a = ev["run_args"]
            self.assertEqual(a[a.index("--configs") + 1], "D:off,D:lexicon")
            self.assertNotIn("--judge-prompt", a)
            summary = "".join(nb["cells"][-1]["source"])
            for need in ('"two_d_runs_only"', '"no_judge_log"', '"d_threshold_is_tau"', '"passthrough_is_the_payload_file"'):
                self.assertIn(need, summary)
            self.assertNotIn('"frozen_prompt_sha"', summary)
            self.assertNotIn("@@", summary)
        with self.assertRaises(B.BuildError):
            B.build_tier2("v1.0", None, True, "x", "C:off")


# ---------------------------------------------------------------------------------------------------------------- separate judge model
class NamedGen:
    """Records which prompts reach it; judge prompts answer merge every other call; QA prompts answer 'unknown'."""

    def __init__(self, backbone, revision):
        self.backbone, self.revision, self.judge_calls, self.qa_calls = backbone, revision, 0, 0
        self.device_name = "fake"
        self.model = type("M", (), {"hf_device_map": {"model.layers.0": 0, "lm_head": 1}} if "7B" in backbone else {})()

    def generate(self, prompts, params):
        out = []
        for m in prompts:
            if m[0]["content"].startswith("You answer questions"):
                self.qa_calls += 1
                out.append("unknown")
            else:
                self.judge_calls += 1
                out.append(json.dumps({"reason": "r", "decision": "merge" if self.judge_calls % 2 else "keep_both"}))
        return out

    def run_info(self, params):
        return {"backbone": self.backbone, "resolved_revision": self.revision, "generation_params": {"max_new_tokens": params.max_new_tokens}}


@unittest.skipUnless(_mirror_available(), "evaluator-side Tier-2 files absent")
class TestSeparateJudgeModel(unittest.TestCase):
    QA = ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1")
    JUDGE = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")

    def _run(self, extra_args, configs="C:off,D:off"):
        run, contract = H["run"], H["contract"]
        convs = [c["conversation_id"] for c in jl(TIER2 / "data" / "tier2_conversations.jsonl")[:2]]
        qa, judge = NamedGen(*self.QA), NamedGen(*self.JUDGE)
        seen = {}

        def fake_judge_model(a, qa_gen):
            seen["args"] = (a.judge_model_id, a.judge_model_revision, a.judge_shard, a.judge_max_memory)
            return judge if a.judge_model_id else qa_gen
        d = Path(tempfile.mkdtemp())
        pt = d / "pt.jsonl"
        pt.write_text(contract.serialize_passthrough(contract.build_passthrough_ids(TIER2 / "data")), encoding="utf-8", newline="\n")
        with mock.patch.object(run, "build_models", lambda a: (H["similarity"].build_metric(lambda: FakeEmbedder()), qa)), \
                mock.patch.object(run, "build_judge_model", fake_judge_model):
            rc = run.main(["--passthrough", str(pt), "--conversations", ",".join(convs), "--judge-prompt", "merge_judge_entity_t2_v2",
                           "--configs", configs, "--out", str(d / "runs"), "--run-prefix", "sep", *extra_args])
        return rc, d, qa, judge, seen

    def test_judge_calls_go_to_the_judge_model_and_qa_calls_to_the_qa_model(self):
        rc, d, qa, judge, seen = self._run(["--judge-model-id", self.JUDGE[0], "--judge-model-revision", self.JUDGE[1], "--judge-shard",
                                            "--judge-max-memory", "0=5GiB,1=11GiB"])
        self.assertEqual(rc, 0)
        self.assertEqual(seen["args"], (self.JUDGE[0], self.JUDGE[1], True, "0=5GiB,1=11GiB"))
        self.assertGreater(judge.judge_calls, 0)
        self.assertEqual((judge.qa_calls, qa.judge_calls), (0, 0))
        self.assertGreater(qa.qa_calls, 0)
        c = json.loads((d / "runs" / "sep-store-surface-only-C_seed0" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(c["backbone"], self.JUDGE[0])
        self.assertTrue(c["judge_model_separate_from_qa"])
        self.assertEqual((c["judge_backbone"], c["generation_run_info"]["resolved_revision"]), self.JUDGE)
        self.assertEqual(c["generation_run_info"]["generation_params"]["max_new_tokens"], 128)
        self.assertEqual(c["judge_device_map"], {"model.layers.0": "0", "lm_head": "1"})
        self.assertEqual((c["tier2"]["qa"]["answerer"], c["tier2"]["qa"]["generation_run_info"]["resolved_revision"]), self.QA)
        dm = json.loads((d / "runs" / "sep-embedding-threshold-D_seed0" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(dm["backbone"])
        self.assertEqual(dm["tier2"]["qa"]["generation_run_info"]["resolved_revision"], self.QA[1])     # same QA model for C and D
        self.assertEqual(c["tier2"]["qa"]["prompt_sha256"], dm["tier2"]["qa"]["prompt_sha256"])
        rep = H["contract"].validate_run(d / "runs" / "sep-store-surface-only-C_seed0", TIER2 / "data", passthrough_file=d / "pt.jsonl")
        self.assertTrue(rep["ok"], rep["errors"][:3])
        self.assertTrue(rep["judge"]["tier2_identity_judge"])

    def test_without_a_judge_model_the_qa_model_judges_as_before(self):
        rc, d, qa, judge, _ = self._run([])
        self.assertEqual(rc, 0)
        self.assertEqual(judge.judge_calls, 0)
        self.assertGreater(qa.judge_calls, 0)
        c = json.loads((d / "runs" / "sep-store-surface-only-C_seed0" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(c["judge_model_separate_from_qa"])
        self.assertEqual(c["backbone"], self.QA[0])

    def test_inconsistent_judge_arguments_are_refused_before_loading(self):
        for args, configs in ((["--judge-model-id", self.JUDGE[0]], "C:off"),                                         # no revision
                              (["--judge-model-id", self.JUDGE[0], "--judge-model-revision", self.JUDGE[1]], "D:off"),  # no C run
                              (["--judge-shard"], "C:off"),                                                             # shard without model
                              (["--judge-model-id", self.JUDGE[0], "--judge-model-revision", self.JUDGE[1], "--judge-max-memory", "zero"], "C:off")):
            try:
                rc, d, qa, judge, seen = self._run(args, configs)
            except SystemExit:
                continue
            self.assertEqual(rc, 1, args)                 # main() turns the ValueError into a recorded failure
            self.assertEqual((qa.qa_calls, qa.judge_calls, judge.judge_calls, seen), (0, 0, 0, {}), args)   # nothing ran
            self.assertIn("judge-max-memory", json.loads((d / "runs" / "TIER2_FAILURE.json").read_text(encoding="utf-8"))["message"])

    def test_parse_max_memory_and_real_loader_selection(self):
        run = H["run"]
        self.assertEqual(run.parse_max_memory("0=5GiB,1=11GiB"), {0: "5GiB", 1: "11GiB"})
        self.assertIsNone(run.parse_max_memory(None))
        import dkmem.backends.llm as L
        calls = {}

        class FakeHF:
            device_name = "x"
            model = None
            backbone = "b"

            @classmethod
            def load_sharded(cls, cfg, max_memory=None):
                calls["sharded"] = (cfg.model_id, cfg.revision, max_memory)
                return cls()

            @classmethod
            def load(cls, cfg):
                calls["single"] = (cfg.model_id, cfg.revision, cfg.device)
                return cls()
        a = type("A", (), dict(judge_model_id=self.JUDGE[0], judge_model_revision=self.JUDGE[1], judge_shard=True,
                               judge_max_memory="0=5GiB,1=11GiB", judge_device="cuda:0"))()
        with mock.patch.object(L, "HFGenerator", FakeHF):
            run.build_judge_model(a, object())
            a.judge_shard = False
            run.build_judge_model(a, object())
        self.assertEqual(calls["sharded"], (self.JUDGE[0], self.JUDGE[1], {0: "5GiB", 1: "11GiB"}))
        self.assertEqual(calls["single"], (self.JUDGE[0], self.JUDGE[1], "cuda:0"))
        qa = object()
        a.judge_model_id = None
        self.assertIs(run.build_judge_model(a, qa), qa)


@unittest.skipUnless(_mirror_available(), "evaluator-side Tier-2 files absent")
class TestCombinedKernelsUseTheValidated7BJudge(unittest.TestCase):
    def test_c_runs_get_the_separate_pinned_7b_and_qa_stays_3b(self):
        B = _builder()
        for arm in ("v1.0", "mirror"):
            files, nb, payload, tree, ev, qmap = B.build_tier2(arm, None, True, "t2cd-test")
            a = ev["run_args"]
            self.assertEqual(a[a.index("--judge-model-id") + 1], "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(a[a.index("--judge-model-revision") + 1], "a09a35458c702b33eeacc393d103063234e8bc28")
            self.assertIn("--judge-shard", a)
            self.assertEqual(a[a.index("--judge-max-memory") + 1], "0=5GiB,1=11GiB")
            self.assertEqual(a[a.index("--judge-prompt") + 1], "merge_judge_entity_t2_v2")
            self.assertEqual((a[a.index("--model-id") + 1], a[a.index("--model-revision") + 1]),
                             ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"))
            summary = "".join(nb["cells"][-1]["source"])
            self.assertIn('"judge_is_the_pinned_7b_and_separate"', summary)
            self.assertIn('"qa_is_the_pinned_3b_in_every_run"', summary)
            self.assertNotIn("@@", summary)
            self.assertIn("judge backbone differs from Tier 1", "".join(nb["cells"][0]["source"]))

    def test_memory_probe_carries_no_benchmark_data(self):
        B = _builder()
        files, nb, payload, tree, ev, _ = B.build_memory_probe()
        self.assertFalse([n for n in files if "tier2/data" in n or "mirror_v1_1" in n or "FROZEN" in n])
        self.assertTrue(json.loads(files["inputs/judge_cases.jsonl"].splitlines()[0])["case_id"].startswith("jv7_"))
        self.assertIn("--judge-shard", ev["run_args"])


# ---------------------------------------------------------------------------------------------------------------- analysis plan
class TestFrozenAnalysis(unittest.TestCase):
    def test_interval_helpers(self):
        from tier2 import entity_report as E
        w = E.wilson(0, 42)
        self.assertEqual((w["k"], w["n"], w["rate"]), (0, 42, 0.0))
        self.assertAlmostEqual(w["ci95"][1], 0.0838, places=3)
        self.assertAlmostEqual(sum(E.wilson(21, 42)["ci95"]) / 2, 0.5, places=6)
        self.assertIsNone(E.wilson(0, 0)["rate"])
        p = E.paired([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
        self.assertEqual((p["discordant_x_only"], p["discordant_y_only"], p["mcnemar_exact_p"], p["diff_y_minus_x"]), (1, 1, 1.0, 0.0))
        p = E.paired([1] * 10, [0] * 10)
        self.assertEqual((p["discordant_x_only"], p["diff_y_minus_x"]), (10, -1.0))
        self.assertAlmostEqual(p["mcnemar_exact_p"], 2 / 1024)
        c = E.cluster_ratio([(1, 10), (0, 10), (3, 10)])
        self.assertAlmostEqual(c["rate"], 4 / 30)
        self.assertLessEqual(c["ci95"][0], c["rate"])
        self.assertGreaterEqual(c["ci95"][1], c["rate"])
        self.assertEqual(E.cluster_ratio([(1, 10)]), E.cluster_ratio([(1, 10)]))          # deterministic (fixed seed)

    def test_contradiction_and_user_only_detectors(self):
        from tier2 import entity_report as E
        mk = lambda r, d: json.dumps({"reason": r, "decision": d})
        self.assertTrue(E.judge_contradiction(mk("A is about X; B is about Y; different entities.", "merge")))
        self.assertTrue(E.judge_contradiction(mk("A is about X; B is about X; same entity.", "keep_both")))
        self.assertFalse(E.judge_contradiction(mk("A is about X; B is about X; same entity.", "merge")))
        self.assertTrue(E.user_only_reason(mk("A is about the user; B is about the user; same entity.", "merge")))

    @unittest.skipUnless((EVAL / "tier2_entity_runs" / "cd7b_smoke_v10_20261014").exists(), "smoke outputs absent")
    def test_reproduces_the_audited_smoke_numbers(self):
        from tier2 import entity_report as E
        a = E.analyse_arm(EVAL / "tier2_entity_runs" / "cd7b_smoke_v10_20261014" / "runs", "v1.0",
                          REPO / "validation" / "tier2_entity" / "kernel_cd7b_smoke_v10.evaluator")
        self.assertEqual(a["failures"], [])
        c = a["runs"]["t2cd7b-smoke-v10-store-surface-only-C_seed0"]
        self.assertEqual((c["merge_rates"]["distractor_distractor"]["k"], c["merge_rates"]["distractor_distractor"]["n"]), (4, 100))
        self.assertEqual((c["merge_rates"]["injected_distractor"]["k"], c["merge_rates"]["injected_distractor"]["n"]), (0, 32))
        self.assertEqual(c["judge_diagnostics"]["distractor_distractor_user_only_reasons_merged"], 4)
        self.assertEqual((c["summary"]["FCR_direct"]["k"], c["summary"]["MCR_direct"]["k"]), (1, 0))
        self.assertEqual((c["qa"]["all"]["k"], c["qa"]["all"]["n"]), (4, 5))
        self.assertEqual(c["summary"]["indirect_only_connections"], {"different": 0, "same": 0})

    def test_plan_freeze_pins_the_implementation(self):
        fz = json.loads((TIER2 / "ANALYSIS_PLAN_FREEZE_v1.json").read_text(encoding="utf-8"))
        for rel, want in fz["files_sha256"].items():
            self.assertEqual(hashlib.sha256((TIER2 / rel).read_bytes()).hexdigest(), want, rel)
        self.assertIn("entity_report.py", fz["files_sha256"])
        self.assertIn("analysis_plan_full_v1.md", fz["files_sha256"])


# ---------------------------------------------------------------------------------------------------------------- kinship probe
KP = TIER2 / "kinship_probe"


class TestKinshipProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tier2.kinship_probe import make_cases as M
        cls.M = M
        cls.rows = jl(KP / "cases_v1.jsonl")

    def test_files_equal_the_deterministic_build(self):
        self.assertEqual((KP / "cases_v1.jsonl").read_text(encoding="utf-8"), self.M.render_cases(self.M.build()))
        self.assertEqual((KP / "review_sheet.csv").read_text(encoding="utf-8"), self.M.render_review_sheet(self.M.build()))

    def test_design_counts_and_labels(self):
        from collections import Counter
        c = Counter((r["group"], r["cell"]) for r in self.rows)
        self.assertEqual(len(self.rows), 72)
        for cell in ("P-same", "N-same", "N-diff"):
            self.assertEqual((c[("H-diff", cell)], c[("H-same-named", cell)]), (12, 6))
        self.assertEqual((c[("E-explicit", "P-same")], c[("E-collapsed", "P-same")], c[("possessive", "P-same")]), (6, 6, 6))
        lab = {g: {r["expected"] for r in self.rows if r["group"] == g} for g in {r["group"] for r in self.rows}}
        self.assertEqual(lab, {"H-diff": {"keep_both"}, "H-same-named": {"merge"}, "E-explicit": {"keep_both"}, "E-collapsed": {None}, "possessive": {"keep_both"}})
        self.assertTrue(all(r["terms"][0] != r["terms"][1] for r in self.rows if r["group"] == "H-diff"))
        self.assertEqual(sum(r["term_origin"] == "benchmark_pair" for r in self.rows if r["group"] == "H-diff"), 15)

    def test_same_person_controls_name_one_person_unambiguously(self):
        for r in (r for r in self.rows if r["group"] == "H-same-named"):
            term, name = r["terms"][0], r["name"]
            self.assertEqual(r["terms"][0], r["terms"][1])
            for t in (r["text_a"], r["text_b"]):
                self.assertIn(term, t.split())
                self.assertIn(name, t.split())
            self.assertTrue(r["intended_identity"].startswith(f"one person: {name}"))
            self.assertTrue(r["rationale"])
            self.assertNotIn("ja rah", r["text_a"] + r["text_b"])                      # no two-trip controls (residence changes instead)

    def test_disjoint_from_benchmark_and_validation_sets(self):
        rep = self.M.disjoint(self.rows)
        self.assertEqual(set(rep), {"benchmark v1.0", "mirror v1.1m", "judge validation v1", "judge validation v2", "judge validation 7b"})
        for name, r in rep.items():
            self.assertEqual((r["identical_sentences"], r["shared_capitalised_words"]), ([], []), name)

    def test_naming_classifier(self):
        from tier2.kinship_probe import score as S
        crit = json.loads((KP / "criteria_v1.json").read_text(encoding="utf-8"))
        w = crit["naming_classifier"]["english_kin_words"]
        case = {"terms": ["dadi", "nani"]}
        mode = lambda reason, order="ab": S.naming_mode(reason, case, order, w)
        self.assertEqual(mode("A is about Meri's grandmother; B is about Meri's grandmother; same entity.")[0], "english_same")
        self.assertTrue(mode("A is about Meri's grandmother; B is about Meri's grandmother; same entity.")[1]["meri_as_name"])
        self.assertEqual(mode("A is about Meri dadi; B is about Meri nani; different entities.")[0], "source_distinct")
        self.assertEqual(mode("A is about Meri nani; B is about Meri dadi; different entities.", "ba")[0], "source_distinct")
        self.assertEqual(mode("A is about the user's aunt; B is about the user's mother; different entities.")[0], "english_different")
        self.assertEqual(mode("A is about Meri dadi; B is about the grandmother; same entity.")[0], "mixed")
        self.assertEqual(mode("something else")[0], "unparsed")
        self.assertEqual(mode(None)[0], "unparsed")
        self.assertEqual(S.entity_kind("Mere nanaji", "nanaji", w)[0], "source")
        self.assertEqual(S.entity_kind("Mere nana", "nanaji", w)[0], "source")

    def test_scorer_on_synthetic_outputs_and_refuses_unfrozen(self):
        from tier2.kinship_probe import score as S
        crit = json.loads((KP / "criteria_v1.json").read_text(encoding="utf-8"))
        outs = []
        for r in self.rows:
            for o in ("ab", "ba"):
                same = r["group"] == "H-diff" and r["cell"] == "P-same"
                reason = "A is about Meri's aunt; B is about Meri's aunt; same entity." if same else "A is about X; B is about Y; different entities."
                outs.append({"case_id": r["case_id"], "order": o, "raw_output": json.dumps({"reason": reason, "decision": "merge" if same else "keep_both"})})
        rep = S.score(outs, self.rows, crit)
        self.assertEqual(rep["mechanism"]["english_same"], {"merges": 24, "calls": 24})
        self.assertEqual(rep["mechanism"]["verdict"], "consistent_with_hypothesis")
        self.assertTrue(rep["structure"]["merge"]["structure_matters"])
        self.assertEqual(rep["parse_failures"], 0)
        if not (KP / "FREEZE.json").exists():
            with self.assertRaises(SystemExit):
                S.verify_freeze()

    def test_kernel_payload_has_texts_only(self):
        B = _builder()
        files, nb, payload, tree, ev, _ = B.build_judge_validation("kin_probe")
        rows = [json.loads(l) for l in files["inputs/judge_cases.jsonl"].splitlines()]
        self.assertEqual(len(rows), 72)
        self.assertTrue(all(set(r) == {"case_id", "text_a", "text_b"} for r in rows))
        self.assertFalse(any("kinship_probe" in n or "review" in n for n in files))
        a = ev["run_args"]
        self.assertEqual((a[a.index("--prompt") + 1], a[a.index("--model-id") + 1], a[a.index("--model-revision") + 1]),
                         ("merge_judge_entity_t2_v2", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"))
        self.assertIn("--shard", a)


# ---------------------------------------------------------------------------------------------------------------- offline replay / D sweep
FULL = {"v1.0": (EVAL / "tier2_entity_runs" / "cd7b_full_v10_20261014" / "runs", TIER2 / "data"),
        "mirror": (EVAL / "tier2_entity_runs" / "cd7b_full_mirror_20261014" / "runs", TIER2 / "mirror_v1_1" / "data")}


@unittest.skipUnless(all(r.exists() for r, _ in FULL.values()) and _mirror_available(), "full-run outputs or evaluator files absent")
class TestOfflineReplay(unittest.TestCase):
    def test_replay_reproduces_all_eight_full_runs_byte_for_byte(self):
        from tier2 import replay as R
        for arm, (runs, data) in FULL.items():
            rep = R.verify(runs, data)
            self.assertEqual(len(rep), 4, arm)
            for rid, v in rep.items():
                self.assertTrue(v["records_identical"] and v["store_identical"], (arm, rid))

    def test_sweep_at_tau_090_equals_the_frozen_report_and_is_monotone(self):
        from tier2 import replay as R
        rep = json.loads((EVAL / "tier2_entity_runs" / "TIER2_FULL_entity_report_v1.json").read_text(encoding="utf-8"))
        for arm, (runs, data) in FULL.items():
            sw = R.sweep_arm(runs, data, None, None)
            pre = "t2cd7b-full-v10" if arm == "v1.0" else "t2cd7b-full-mirror"
            for mode, rid in (("off", f"{pre}-embedding-threshold-D_seed0"), ("lexicon", f"{pre}-dk-mem-lexicon-D_seed0")):
                pts = sw["points"][mode]
                p = next(x for x in pts if x["tau"] is not None and x["tau"] <= 0.9)
                s = rep["arms"][arm]["runs"][rid]["summary"]
                self.assertEqual((p["FCR_direct"]["k"], p["MCR_direct"]["k"]), (s["FCR_direct"]["k"], s["MCR_direct"]["k"]), (arm, mode))
                recs = [x["memory_records"] for x in pts]
                self.assertEqual(recs, sorted(recs, reverse=True))           # lowering tau only merges
                self.assertEqual(pts[0]["memory_records"], 828)                 # tau above every score: no merge
            for po, pl in zip(sw["points"]["off"], sw["points"]["lexicon"]):
                self.assertGreaterEqual(pl["memory_records"], po["memory_records"])   # the gate only removes merges

    def test_lexicon_is_swapped_offline_without_inference(self):
        from tier2 import replay as R
        runs, data = FULL["v1.0"]
        turns = R.load_turns(data)
        rd = runs / "t2cd7b-full-v10-dk-mem-lexicon-D_seed0"
        rlog = R._jl(runs / f"{rd.name}.tier2_retrieval_log.jsonl")
        from dkmem.memory.lexicon import load_lexicon
        with tempfile.TemporaryDirectory() as d:
            src = json.loads(Path(H["run"]._dkmem.LEXICON_PATH).read_text(encoding="utf-8"))
            empty = Path(d) / "empty_lexicon.json"
            e0 = dict(src["entries"][0], id="never_matches", surface_forms=["zzqxv"])
            empty.write_text(json.dumps({**src, "entries": [e0]}), encoding="utf-8")         # valid lexicon that matches nothing
            on = R.replay_run(rlog, turns, host="D", mode="lexicon", distinctions=R.Distinctions(load_lexicon(empty), turns), tau=0.9)
        off = R._jl(runs / "t2cd7b-full-v10-embedding-threshold-D_seed0" / "pairwise_eval.jsonl")
        self.assertEqual([r["decision"] for r in on["records"]], [r["decision"] for r in off])     # no distinctions -> gate never vetoes

    def test_replay_refuses_changed_conversation_texts(self):
        from tier2 import replay as R
        runs, _ = FULL["v1.0"]
        with self.assertRaises(SystemExit):
            R.check_same_conversations(runs, TIER2 / "mirror_v1_1" / "data")      # different texts -> rerun, not replay

    def test_frontier_and_matched_mcr_helpers(self):
        from tier2 import replay as R
        pts = [{"tau": t, "FCR_direct": [f, 10], "MCR_direct": [m, 10]} for t, f, m in ((0.9, 0, 10), (0.8, 2, 5), (0.7, 3, 6), (0.6, 5, 0))]
        self.assertEqual([p["tau"] for p in R.frontier(pts)], [0.9, 0.8, 0.6])
        rows = R.matched_mcr(pts, [{"tau": 0.8, "FCR_direct": [1, 10], "MCR_direct": [5, 10]}])
        self.assertEqual(next(r for r in rows if r["MCR_at_most"] == 0.5), {"MCR_at_most": 0.5, "FCR_gate_off": 0.2, "FCR_gate_lexicon": 0.1})


# ---------------------------------------------------------------------------------------------------------------- plan v2 tooling
T1_GROUPS = REPO / "results" / "full_ad_3b_s0_20261010" / "runs"


@unittest.skipUnless(T1_GROUPS.exists(), "Tier 1 full-run groups absent")
class TestTier1LexiconReplay(unittest.TestCase):
    def test_replay_reproduces_every_tier1_group(self):
        from tier1 import lexicon_replay as L
        for g in sorted(T1_GROUPS.iterdir()):
            v = L.verify(g)
            self.assertTrue(v["trace_identical"], g.name)
            self.assertEqual(set(v["rows_identical"].values()), {True}, g.name)

    def test_new_lexicon_regates_without_inference_and_writes_a_scorable_group(self):
        from tier1 import lexicon_replay as L
        from dkmem.eval.groups import load_group
        src = json.loads(Path(H["run"]._dkmem.LEXICON_PATH).read_text(encoding="utf-8"))
        g = T1_GROUPS / "D-qwen-qwen2.5-3b-instruct-s0-cdd9aca0"
        with tempfile.TemporaryDirectory() as d:
            lex = Path(d) / "lex.json"
            lex.write_text(json.dumps({**src, "entries": [e for e in src["entries"] if e["value"] not in ("dadi", "nani")]}), encoding="utf-8")
            res = L.write(g, lex, Path(d) / "out")
            self.assertGreater(res["stats"]["lexicon"]["changed_decisions"], 0)        # dadi/nani pairs are no longer vetoed
            self.assertEqual(res["stats"]["lexicon"]["needs_inference"], 0)
            grp = load_group(Path(d) / "out")                                            # dkmem.eval reads it unchanged
            self.assertEqual(len(grp.rows("lexicon")), len(L.load(g)["rows"]["lexicon"]))
            self.assertEqual(json.loads((Path(d) / "out" / "run_config.json").read_text(encoding="utf-8"))["lexicon_replay"]["source_lexicon_sha256"],
                             json.loads((g / "run_config.json").read_text(encoding="utf-8"))["lexicon"]["sha256"])
            with self.assertRaises(L.ReplayError):
                L.write(g, lex, Path(d) / "out")                                         # never overwrites

    def test_verify_refuses_a_different_lexicon(self):
        from tier1 import lexicon_replay as L
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lex.json"
            src = json.loads(Path(H["run"]._dkmem.LEXICON_PATH).read_text(encoding="utf-8"))
            p.write_text(json.dumps({**src, "entries": src["entries"][:3]}), encoding="utf-8")
            with self.assertRaises(L.ReplayError):
                L.verify(T1_GROUPS / "C-qwen-qwen2.5-3b-instruct-s0-e54539f9", p)


@unittest.skipUnless(all(r.exists() for r, _ in FULL.values()) and _mirror_available(), "full-run outputs or evaluator files absent")
class TestScoreV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tier2 import score_v2 as V
        cls.V = V
        cls.args = lambda self, arm, **kw: type("A", (), {
            "runs": FULL[arm][0], "run_data_dir": FULL[arm][1],
            "arm_frozen": TIER2 / ("FROZEN_v1.0.json" if arm == "v1.0" else "mirror_v1_1/FROZEN_v1.1m.json"),
            "evaluator": REPO / "validation" / "tier2_entity" / f"kernel_cd7b_full_{'v10' if arm == 'v1.0' else 'mirror'}.evaluator",
            "data_dir": FULL[arm][1], "lexicon": None, "languages": None, "classes": None, "exclude_pairs": None, "boot": 20,
            "split": "dev", "feature_set": "frozen", **kw})()

    def test_reproduces_the_frozen_v1_report(self):
        rep = json.loads((EVAL / "tier2_entity_runs" / "TIER2_FULL_entity_report_v1.json").read_text(encoding="utf-8"))
        for arm in ("v1.0", "mirror"):
            s = self.V.score_arm(self.args(arm))
            for rid, v in s["runs"].items():
                r = rep["arms"][arm]["runs"][rid]["summary"]
                for k in ("FCR_record", "MCR_record", "FCR_direct", "MCR_direct"):
                    self.assertEqual((v["overall"][k]["k"], v["overall"][k]["n"]), (r[k]["k"], r[k]["n"]), (arm, rid, k))
                self.assertFalse(v["final_records_changed_by_lexicon"])

    def test_language_filter_and_frontier_consistency(self):
        s = self.V.score_arm(self.args("v1.0", languages="hi"))
        self.assertEqual((s["inputs"]["pairs"], s["inputs"]["by_relation"]), (63, {"different": 36, "same": 27}))
        d_off = s["runs"]["t2cd7b-full-v10-embedding-threshold-D_seed0"]["overall"]
        fr = s["d_frontier"]["record"]["frontier"]["off"]
        at = [p for p in fr if p["tau"] is None or p["tau"] > 0.9]          # states reached before tau drops to 0.90
        self.assertEqual((round(at[-1]["FCR"] * 36), round(at[-1]["MCR"] * 27)), (d_off["FCR_record"]["k"], d_off["MCR_record"]["k"]))
        fcr = [p["FCR"] for p in fr]
        self.assertEqual(fcr, sorted(fcr))                                       # monotone frontier

    def test_curve_and_matched_mcr_helpers(self):
        V = self.V
        rel = ["different", "different", "same", "same"]
        idx = [3, 10**9, 1, 2]
        pts = V.curve(idx, rel, [0, 1, 2, 3], 5)
        self.assertEqual(pts, [(0, 0.0, 1.0), (1, 0.0, 0.5), (2, 0.0, 0.0), (3, 0.5, 0.0)])
        self.assertEqual((V.min_fcr_at(pts, 0.5), V.min_fcr_at(pts, 0.0)), (0.0, 0.0))

    def test_final_label_is_refused_without_the_v2_freeze(self):
        if (EVAL / "EVALUATION_PLAN_FREEZE_v2.json").exists():
            self.skipTest("plan v2 is frozen")
        with self.assertRaises(SystemExit):
            self.V.check_freeze_v2()

    def test_changed_lexicon_marks_qa_stale_when_records_change(self):
        src = json.loads(Path(H["run"]._dkmem.LEXICON_PATH).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            lex = Path(d) / "lex.json"
            lex.write_text(json.dumps({**src, "entries": [e for e in src["entries"] if e["value"] not in ("tu", "tum", "aap")]}), encoding="utf-8")
            s = self.V.score_arm(self.args("v1.0", lexicon=lex))
        c_lex = s["runs"]["t2cd7b-full-v10-dk-mem-lexicon-C_seed0"]
        self.assertTrue(c_lex["final_records_changed_by_lexicon"])               # register no longer vetoes Hindi same pairs
        self.assertTrue(c_lex["qa"]["status"].startswith("STALE"))
        self.assertEqual(s["runs"]["t2cd7b-full-v10-store-surface-only-C_seed0"]["qa"]["status"], "rescored")   # gate off untouched


# ---------------------------------------------------------------------------------------------------------------- benchmark v2 tooling
DELIVERY = REPO / "data" / "team_b_delivery"
V2_SPECS = DELIVERY / "banks" / "pair_specs_v2.json"
V2_SPLITS = [DELIVERY / "data" / "tier2_v2_dev_gold.jsonl", DELIVERY / "data" / "tier2_v2_test_gold.jsonl"]


def _jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


class TestQaNormV2(unittest.TestCase):
    def test_gujarati_vowel_signs_survive(self):
        from tier2 import generate_tier2 as G, qa_norm as Q
        self.assertEqual(G.normalize_text("મામા"), G.normalize_text("મામી"))         # the v1 defect
        self.assertNotEqual(Q.normalize_text("મામા"), Q.normalize_text("મામી"))
        self.assertFalse(Q.contains_alias("મામી આવી", "મામા"))
        self.assertTrue(Q.contains_alias("એ અમદાવાદ માં રહે છે", "અમદાવાદ"))

    def test_identical_to_v1_on_every_v1_reference_question_and_answer(self):
        from tier2 import generate_tier2 as G, qa_norm as Q
        texts = [a for r in _jl(TIER2 / "data" / "tier2_qa_reference.jsonl") for grp in (*r["slots"], *r["forbidden"]) for a in grp]
        texts += [q["question"] for q in _jl(TIER2 / "data" / "tier2_qa_questions.jsonl")]
        for f in (EVAL / "tier2_entity_runs").glob("cd7b_full_v10_20261014/runs/*.tier2_qa_answers.jsonl"):
            texts += [a["raw_answer"] or "" for a in _jl(f)]
        self.assertGreater(len(texts), 200)
        self.assertEqual([G.normalize_text(t) for t in texts], [Q.normalize_text(t) for t in texts])

    def test_reference_names_its_normaliser(self):
        from tier2 import qa_norm as Q
        ref = {"slots": [["મામા"]], "forbidden": []}
        self.assertTrue(Q.score_answer("મામી", {**ref, "scoring": Q.SCORING_V1})["correct"])     # frozen v1 semantics kept
        self.assertFalse(Q.score_answer("મામી", {**ref, "scoring": Q.SCORING_V2})["correct"])
        with self.assertRaises(ValueError):
            Q.score_answer("x", {**ref, "scoring": "other"})


def _v2_fixture(d: Path, *, gu_bank=True):
    """Synthetic v2 inputs (no Team B data): 6 hi + 6 gu specs, split files, a gu distractor bank."""
    cities = [("Shimla", "Ooty"), ("Kochi", "Agra"), ("Rajkot", "Bhuj"), ("Vapi", "Anand"), ("Patan", "Morbi"), ("Dahod", "Valsad")]
    kinds = [("hi", "kinship", "different")] * 3 + [("hi", "name_variant", "same")] * 3 + [("gu", "kinship", "different")] * 3 + [("gu", "honorific_register", "same")] * 3
    specs = []
    for i, (lang, cls, rel) in enumerate(kinds):
        ca, cb = cities[i % 6]
        if lang == "hi" and rel == "different":
            ta, tb, ma, mb = f"Meri phuphi{i} {ca} mein hai", f"Meri kaki{i} {cb} mein hai", f"phuphi{i}", f"kaki{i}"
        elif lang == "hi":
            ta, tb, ma, mb = f"Zoya{i} {ca} mein hai", f"Zoya{i} ka kaam {cb} mein hai", f"Zoya{i}", f"Zoya{i}"
        elif rel == "different":
            ta, tb, ma, mb = f"ફોઈ{i} {ca} માં છે", f"માસી{i} {cb} માં છે", f"ફોઈ{i}", f"માસી{i}"
        else:
            ta, tb, ma, mb = f"બા{i}, તમે {ca} માં છો", f"બા{i}, તું {cb} જઈશ", "તમે", "તું"
        qa = ([{"role": "a", "question": "kahan?", "slots": [[ca.lower()]], "forbidden": [[cb.lower()]]},
               {"role": "b", "question": "kahan?", "slots": [[cb.lower()]], "forbidden": [[ca.lower()]]}] if rel == "different"
              else [{"role": "ab", "question": "kahan kahan?", "slots": [[ca.lower()], [cb.lower()]], "forbidden": []}])
        specs.append({"spec_id": f"t2s_x_{i:02d}", "relation": rel, "distinction_class": cls, "language": lang, "text_a": ta, "text_b": tb,
                      "marker_a": ma, "marker_b": mb, "fact_a": {"attribute": "city", "value": ca}, "fact_b": {"attribute": "city", "value": cb},
                      "qa": qa, "source": "v2.0-new"})
    (d / "specs.json").write_text(json.dumps({"specs": specs}, ensure_ascii=False), encoding="utf-8")
    rows = [{"eval_pair_id": s["spec_id"], "relation": s["relation"], "distinction_class": s["distinction_class"], "language": s["language"],
             "split": "test" if k % 3 == 2 else "dev"} for k, s in enumerate(specs)]
    (d / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == "dev"), encoding="utf-8")
    (d / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == "test"), encoding="utf-8")
    bank = {"user_turns": {"gu": [f"આજે કામ ઘણું હતું {k}" for k in range(12)]} if gu_bank else {},
            "assistant_turns": {"gu": [{"text": f"સમજાયું {k}", "language": "gu"} for k in range(12)]} if gu_bank else {}}
    (d / "gu_bank.json").write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    return d / "specs.json", [d / "dev.jsonl", d / "test.jsonl"], [TIER2 / "banks" / "distractors.json", d / "gu_bank.json"]


class TestGeneratorV2(unittest.TestCase):
    def _args(self, d, **kw):
        specs, splits, banks = _v2_fixture(Path(d), **kw)
        return ["--specs", str(specs), "--splits", *map(str, splits), "--distractors", *map(str, banks)]

    def test_build_validates_and_is_deterministic(self):
        from tier2 import generate_tier2_v2 as V, contract
        from tier2.replay import load_turns
        with tempfile.TemporaryDirectory() as d:
            args = self._args(d)
            self.assertEqual(V.main(["build", *args, "--out", str(Path(d) / "out")]), 0)
            self.assertEqual(V.main(["validate", *args, "--data-dir", str(Path(d) / "out")]), 0)    # includes a byte-identical regeneration
            man = json.loads((Path(d) / "out" / "tier2_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["counts"]["by_language"], {"gu": 6, "hi": 6})
            self.assertEqual(man["counts"]["by_split"], {"dev": 8, "test": 4})
            self.assertTrue(all(r["scoring"] == "all_slots_contained_and_no_forbidden_after_normalisation_v2_indic"
                                for r in _jl(Path(d) / "out" / "tier2_qa_reference.jsonl")))
            self.assertEqual(len(contract.build_passthrough_ids(Path(d) / "out")), 12 * 12)     # the runner contract accepts the package
            self.assertEqual(len(load_turns(Path(d) / "out")), 12 * 12)
            self.assertNotIn("teamB_answer_key", Path(V.__file__).read_text(encoding="utf-8"))
            with self.assertRaises(SystemExit):
                V.main(["build", *args, "--out", str(Path(d) / "out")])                       # never overwrites

    def test_refuses_a_language_without_bank(self):
        from tier2 import generate_tier2_v2 as V
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                V.main(["build", *self._args(d, gu_bank=False), "--out", str(Path(d) / "o")])
            self.assertIn("no usable distractor bank for language 'gu'", str(cm.exception))
            self.assertEqual(V.main(["build", *self._args(d, gu_bank=False), "--languages", "hi", "--out", str(Path(d) / "o")]), 0)

    def test_detects_spec_and_bank_defects(self):
        from tier2 import generate_tier2_v2 as V
        with tempfile.TemporaryDirectory() as d:
            args = self._args(d)
            spec_p = Path(args[1])
            data = json.loads(spec_p.read_text(encoding="utf-8"))
            s = data["specs"][0]                          # a gold-different pair whose second sentence carries the first one's answer
            s["text_b"] = s["text_b"].replace(s["fact_b"]["value"], s["fact_a"]["value"])
            spec_p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            bank_p = Path(args[-1])
            bank = json.loads(bank_p.read_text(encoding="utf-8"))
            bank["user_turns"]["gu"][0] = "ફોઈ6 આવી"      # a marker inside a distractor
            bank_p.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(V.main(["build", *args, "--out", str(Path(d) / "o")]), 1)
            self.assertFalse((Path(d) / "o").exists())

    def test_split_file_must_agree_with_specs(self):
        from tier2 import generate_tier2_v2 as V
        with tempfile.TemporaryDirectory() as d:
            args = self._args(d)
            dev = Path(args[3])
            rows = _jl(dev)
            rows[0]["relation"] = "same" if rows[0]["relation"] == "different" else "different"
            dev.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaises(SystemExit):
                V.main(["build", *args, "--out", str(Path(d) / "o")])


@unittest.skipUnless(V2_SPECS.exists() and all(p.exists() for p in V2_SPLITS), "Team B v2 delivery absent")
class TestLabelDirAndScoreOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tier2 import build_label_dir as B
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "lbl"
        rc = B.main(["--arm-data-dir", str(TIER2 / "data"), "--arm-frozen", str(TIER2 / "FROZEN_v1.0.json"), "--specs", str(V2_SPECS),
                     "--splits", *map(str, V2_SPLITS), "--out", str(cls.out)])
        assert rc == 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_label_dir_contents(self):
        man = json.loads((self.out / "label_dir_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((man["counts"]["retained_pairs"], man["counts"]["excluded_pairs"]), (62, 7))
        self.assertEqual((self.out / "tier2_conversations.jsonl").read_bytes(), (TIER2 / "data" / "tier2_conversations.jsonl").read_bytes())
        meta = _jl(self.out / "tier2_injection_metadata.jsonl")
        self.assertEqual({m["split"] for m in meta}, {"dev"})                         # retained v1 pairs are all dev
        self.assertEqual(sum(m["distinction_class"] == "contextual_qualifier" for m in meta), 4)
        self.assertIn("t2p_0050", man["excluded"])                                      # t2s_t1_0191: text changed in v2

    def _score(self, data_dir=None, **kw):
        from tier2 import score_v2 as V
        a = type("A", (), {"runs": FULL["v1.0"][0], "run_data_dir": TIER2 / "data", "arm_frozen": TIER2 / "FROZEN_v1.0.json",
                           "evaluator": REPO / "validation" / "tier2_entity" / "kernel_cd7b_full_v10.evaluator", "data_dir": data_dir or self.out,
                           "lexicon": None, "languages": None, "classes": None, "exclude_pairs": None, "boot": 0, "split": "dev",
                           "feature_set": "frozen", **kw})()
        return V.score_arm(a)

    @unittest.skipUnless(FULL["v1.0"][0].exists(), "v1.0 full runs absent")
    def test_relabelled_scoring_matches_v1_on_retained_pairs(self):
        v2, v1 = self._score(), self._score(data_dir=TIER2 / "data")
        self.assertEqual(v2["inputs"]["pairs"], 62)
        for rid, r2 in v2["runs"].items():
            self.assertTrue(r2["provenance_contract_ok"])
            self.assertFalse(r2["final_records_changed_by_lexicon"])
            old = {p["pair"]: p for p in v1["runs"][rid]["pairs"]}
            for p in r2["pairs"]:
                self.assertEqual((p["record_merged"], p["direct_status"], p["relation"]),
                                 (old[p["pair"]]["record_merged"], old[p["pair"]]["direct_status"], old[p["pair"]]["relation"]))

    @unittest.skipUnless(FULL["v1.0"][0].exists(), "v1.0 full runs absent")
    def test_kinship_only_feature_set(self):
        frozen, kin = self._score(), self._score(feature_set="kinship_only")
        self.assertEqual(kin["inputs"]["feature_set"]["classes"], ["kinship"])
        for rid in frozen["runs"]:
            f, k = frozen["runs"][rid], kin["runs"][rid]
            if f["mode"] != "lexicon":
                self.assertEqual(f["overall"], k["overall"])
                continue
            self.assertLessEqual(k["gate"]["vetoes"], f["gate"]["vetoes"])
            if k["final_records_changed_by_lexicon"]:
                self.assertTrue(k["qa"]["status"].startswith("STALE"))

    def test_test_split_needs_final(self):
        from tier2 import score_v2 as V
        with self.assertRaises(SystemExit) as cm:
            V.main(["--runs", str(FULL["v1.0"][0]), "--run-data-dir", str(TIER2 / "data"), "--arm-frozen", str(TIER2 / "FROZEN_v1.0.json"),
                    "--evaluator", "x", "--data-dir", str(self.out), "--split", "all", "--out", str(Path(self.tmp.name) / "r.json")])
        self.assertIn("held-out test pairs", str(cm.exception))


class TestJudgeReadiness(unittest.TestCase):
    def test_check_cases_template_and_disjointness(self):
        from tier2 import judge_readiness as J
        rows, n = [], 0
        for lang in ("mr", "gu"):
            for (cat, role, exp), k in sorted(J.template().items()):
                for _ in range(k):
                    n += 1
                    rows.append({"case_id": f"jvx_{n:03d}", "category": f"probe_register_{lang}" if cat == "probe_register" else cat,
                                 "role": role, "expected": exp, "language": lang, "attribute_pattern": "same_fact", "contested": False,
                                 "text_a": f"{lang} sentence a {n}", "text_b": f"{lang} sentence b {n}", "note": ""})
        with tempfile.TemporaryDirectory() as d:
            specs_p, _splits, _banks = _v2_fixture(Path(d))
            specs = json.loads(specs_p.read_text(encoding="utf-8"))["specs"]
            p = Path(d) / "cases.jsonl"
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            self.assertTrue(J.check_cases(p, specs, [])["ok"])
            rows[0]["text_a"] = specs[0]["text_a"]                                       # repeats a benchmark sentence
            rows[-1]["category"] = "same_fact"                                            # breaks the composition
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            res = J.check_cases(p, specs, [])
            self.assertFalse(res["ok"])
            self.assertEqual(len(res["failures"]), 2)

    def test_new_languages_are_not_validated(self):
        from tier2 import judge_readiness as J
        from tier2.generate_tier2_v2 import load_specs, load_splits
        from dkmem.memory.lexicon import load_lexicon
        with tempfile.TemporaryDirectory() as d:
            specs_p, splits, _ = _v2_fixture(Path(d))
            specs = load_specs(specs_p)
            res = J.check(specs, load_splits(splits, specs), load_lexicon(Path(H["run"]._dkmem.LEXICON_PATH)), ["gu", "hi"])
        self.assertTrue(res["verdict"]["gu"].startswith("NOT VALIDATED"))
        self.assertEqual(res["verdict"]["hi"], "VALIDATED")
        self.assertEqual(res["prompt"]["language_specific_words"], [])
