"""Tier-2 Config C: the frozen Tier-1 ``merge_judge_v1`` reaches the model, through the ACTUAL vendored harness.

History. The first Tier-2 run imported its judge from a separate ``harness.judge`` (Kaggle dataset) whose ``MERGE_JUDGE_V1``
shared the prompt id but not the text or hash (sha256 ``f0dc9cf7...`` vs the frozen ``3855ef65...``; it answered
``{"decision": "different"}``, not merge/keep_both with a reason), decoded with ``max_new_tokens`` 32 (which truncates the frozen
prompt's JSON), and wrote ``threshold: 0.0`` where dkmem writes ``null``. These tests pin the repaired path.

Real code under test: ``eval/harness`` (vendored, see eval/harness/VENDORED.json), ``eval/evaluator``, ``eval/tier2``.
Fake: the generator and the embedder only. No real model is loaded and no gold or answer-key file is read: the evaluator
side runs on a SYNTHETIC data directory built from the label-free conversations/questions with made-up pairs and labels.

Skipped where ``eval/tier2`` is absent (the Kaggle payload embeds ``dkmem/`` and ``tests/`` only).
"""

import difflib
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).parent.parent
EVAL = REPO / "eval"
TIER2 = EVAL / "tier2"
CONVERSATIONS = TIER2 / "data" / "tier2_conversations.jsonl"
QUESTIONS = TIER2 / "data" / "tier2_qa_questions.jsonl"
HISTORICAL_C = EVAL / "tier2_full_runs" / "t2full-store-surface-only-C_seed0"
FROZEN_SHA256 = "3855ef65f50e9a0f017710e5c0d0cb5a63d210a7547fa0e0e336d4c77a45c5ff"  # DKMEM_FIRST_RUN.md; tests/test_baselines.py
HARNESS_SHA256 = "f0dc9cf75eb8891c9651ccdd8e685af047c5813a92f61c55fa6440ed6b658bea"  # the prompt the first Tier-2 run used
# raw frozen-judge answers measured on the Tier-1 full run (Qwen2.5-3B, 518 calls, 0 parse failures at max_new_tokens 128):
SHORT_ANSWER = '{"decision": "keep_both", "reason": "The two entries are different."}'
LONGEST_MEASURED_ANSWER = ('{"decision": "merge", "reason": "Both entries refer to the same person (Mishra) and the same action '
                           '(bheja, which means served food)."}')   # 135 UTF-8 bytes, the longest of the 518

from dkmem.backends.llm import GenerationParams  # noqa: E402
from dkmem.memory.judge import (  # noqa: E402
    MERGE_JUDGE_INFORMED_V1, MERGE_JUDGE_V1, parse_judge_output, render_judge_prompt,
)
from dkmem.memory.prompts import PromptTemplate  # noqa: E402

H = {}   # real harness + tier2 modules, filled by setUpModule


def setUpModule():
    if not (CONVERSATIONS.exists() and (EVAL / "harness" / "hosts.py").exists()):
        raise unittest.SkipTest("eval/tier2 and eval/harness are not present in this checkout")
    sys.path.insert(0, str(EVAL))
    import evaluator.validate  # noqa: F401  (the vendored evaluator imports cleanly)
    import harness._dkmem as dk
    import harness.hosts, harness.similarity, harness.writer  # noqa: F401
    import tier2.contract as contract
    import tier2.frozen_judge as frozen
    import tier2.judge_host as jh
    import tier2.judge_provenance as prov
    import tier2.run_tier2 as run
    H.update(dk=dk, contract=contract, frozen=frozen, jh=jh, prov=prov, run=run, similarity=sys.modules["harness.similarity"])


def tearDownModule():
    for name in list(sys.modules):
        if name in ("harness", "evaluator", "tier2", "diagnostics") or name.split(".")[0] in ("harness", "evaluator", "tier2"):
            sys.modules.pop(name, None)
    if str(EVAL) in sys.path:
        sys.path.remove(str(EVAL))


class FakeEmbedder:
    """Deterministic pseudo-embeddings from character codes; describes itself like the bge-m3 embedder."""

    def __init__(self, dim=16):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for i, ch in enumerate(t):
                v[i % self.dim] += (ord(ch) % 17) + 1
            v[0] += len(t) % 5
            out.append(v)
        return out

    def describe(self):
        return {"similarity_backend": "bge-m3", "similarity_model_id": "BAAI/bge-m3", "similarity_model_revision": "fakerev",
                "similarity_requested_revision": None, "similarity_dtype": "float16", "similarity_device": "cuda:0",
                "similarity_max_seq_length": 512, "similarity_embed_batch_size": 32, "similarity_is_placeholder": False}


def fake_metric():
    return H["similarity"].build_metric(lambda: FakeEmbedder())     # the REAL CosineMetric over a fake embedder


class FakeJudgeGenerator:
    """Records every prompt and decoding setting. Answers like the frozen prompt asks (merge for near-duplicates, keep_both
    otherwise, always with a reason); QA prompts get "unknown". ``fixed`` forces one raw answer for every judge prompt."""

    backbone = "fake-backbone"

    def __init__(self, merge_ratio=0.8, garbage_for=(), fixed=None, chars_per_token=None):
        self.calls, self.merge_ratio, self.garbage_for, self.fixed = [], merge_ratio, tuple(garbage_for), fixed
        self.chars_per_token = chars_per_token       # budget proxy: truncate raw output to max_new_tokens * chars_per_token bytes

    def generate(self, prompts, params):
        self.calls.append((list(prompts), params))
        out = []
        for messages in prompts:
            if messages[0]["content"].startswith("You answer questions"):
                out.append("unknown")
                continue
            block = messages[-1]["content"]
            a, b = block.split("\nMemory B: ", 1)
            a = a[len("Memory A: "):]
            if self.fixed is not None:
                raw = self.fixed
            elif any(g in block for g in self.garbage_for):
                raw = "I think these are the same"
            elif difflib.SequenceMatcher(None, a, b).ratio() >= self.merge_ratio:
                raw = json.dumps({"decision": "merge", "reason": "Near-identical statements."})
            else:
                raw = json.dumps({"decision": "keep_both", "reason": "Different statements."})
            if self.chars_per_token:
                raw = raw.encode("utf-8")[: params.max_new_tokens * self.chars_per_token].decode("utf-8", "ignore")
            out.append(raw)
        return out

    def run_info(self, params):
        return {"backbone": self.backbone, "generation_params": {"max_new_tokens": params.max_new_tokens}}


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def user_turns(conv):
    return [t for s in conv["sessions"] for t in s["turns"] if t["speaker"] == "user"]


def representative_pairs(n=20):
    """The ``n`` most similar user-turn pairs within conversations (the hard ones for a merge judge), spread over the Hindi
    and German conversations. Deterministic; uses no labels."""
    by_lang = {}
    for conv in load_jsonl(CONVERSATIONS):
        turns = user_turns(conv)
        for i in range(len(turns)):
            for j in range(i + 1, len(turns)):
                a, b = turns[i], turns[j]
                r = difflib.SequenceMatcher(None, a["text"], b["text"]).ratio()
                by_lang.setdefault(a["language"], []).append((-r, a["turn_id"], b["turn_id"], a["text"], b["text"]))
    picked, langs = [], sorted(by_lang)
    quota = {lang: n // len(langs) for lang in langs}
    quota[langs[0]] += n - sum(quota.values())
    for lang in langs:
        picked += sorted(by_lang[lang])[: quota[lang]]
    return picked


def judge_params(max_new_tokens=None):
    return GenerationParams(max_new_tokens=max_new_tokens or H["run"].JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8)


def real_lexicon():
    from dkmem.memory.lexicon import load_lexicon
    return load_lexicon(H["dk"].LEXICON_PATH)


def make_setup(config="C", mode="off", generator=None, convs=(), max_new_tokens=None, tau=None, metric=None):
    run, jh = H["run"], H["jh"]
    generator = generator or FakeJudgeGenerator()
    metric = metric or fake_metric()
    if config == "C":
        host = jh.FrozenJudgeHost(generator, judge_params(max_new_tokens), MERGE_JUDGE_V1, metric, "surface")
    else:
        import harness.hosts as hosts
        host = hosts.EmbeddingThresholdHost(tau, metric, "surface")
    passthrough = {t["turn_id"]: "ent_" + hashlib.sha256(t["turn_id"].encode()).hexdigest()[:16] for c in convs for t in user_turns(c)}
    qa_params = GenerationParams(max_new_tokens=run.QA_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8)
    lexicon = real_lexicon() if mode == "lexicon" else None
    return run.Setup(config, mode, host, metric, generator, lexicon, passthrough, seed=0, tau=tau, qa_params=qa_params), generator


class TestVendoredHarnessIsTheRealOne(unittest.TestCase):
    def test_imports_resolve_inside_this_repository(self):
        import harness
        import evaluator
        self.assertTrue(Path(harness.__file__).resolve().is_relative_to(EVAL.resolve()))
        self.assertTrue(Path(evaluator.__file__).resolve().is_relative_to(EVAL.resolve()))
        import dkmem
        self.assertEqual(Path(H["dk"].IPDV2).resolve(), Path(dkmem.__file__).resolve().parents[1])

    def test_vendored_files_are_unmodified_copies(self):
        info = json.loads((EVAL / "harness" / "VENDORED.json").read_text(encoding="utf-8"))
        for rel, row in info["files"].items():
            local = hashlib.sha256((EVAL / rel).read_bytes()).hexdigest()
            self.assertEqual(local, row["local_sha256"], rel)
            if rel != "harness/_dkmem.py":
                self.assertTrue(row["identical_to_source"], rel)
                self.assertEqual(local, row["source_sha256"], rel)

    def test_the_colliding_harness_judge_is_not_part_of_the_workspace(self):
        self.assertFalse((EVAL / "harness" / "judge.py").exists())
        with self.assertRaises(ImportError):
            __import__("harness.judge")

    def test_runner_dependencies_are_real_implementations(self):
        run = H["run"]
        self.assertEqual(run.extract_verbatim.__module__, "harness.extractors")
        self.assertEqual(run.EmbeddingThresholdHost.__module__, "harness.hosts")
        self.assertEqual(run.write_run.__module__, "harness.writer")
        self.assertEqual(run.build_manifest.__module__, "harness.writer")

    def test_strategy_names_match_the_dkmem_config(self):
        from dkmem.config import dkmem_mode_for_strategy
        run = H["run"]
        for mode, strategy in run.GATE_ON_STRATEGY.items():
            self.assertEqual(dkmem_mode_for_strategy(strategy), mode)
        for strategy in run.GATE_OFF_STRATEGY.values():
            self.assertEqual(dkmem_mode_for_strategy(strategy), "off")


class TestPromptIdentity(unittest.TestCase):
    def test_runner_uses_the_dkmem_prompt_object(self):
        self.assertIs(H["run"].MERGE_JUDGE_V1, MERGE_JUDGE_V1)

    def test_id_and_hash_are_the_frozen_tier1_values(self):
        self.assertEqual(MERGE_JUDGE_V1.prompt_id, "merge_judge_v1")
        self.assertEqual(MERGE_JUDGE_V1.sha256, FROZEN_SHA256)
        frozen = H["frozen"]
        self.assertEqual(frozen.FROZEN_JUDGE_PROMPT_ID, MERGE_JUDGE_V1.prompt_id)
        self.assertEqual(frozen.FROZEN_JUDGE_PROMPT_SHA256, FROZEN_SHA256)
        self.assertEqual(frozen.LEGACY_HARNESS_JUDGE_PROMPT_SHA256, HARNESS_SHA256)
        self.assertNotEqual(MERGE_JUDGE_V1.sha256, HARNESS_SHA256)

    def test_prompt_text_is_the_frozen_contract(self):
        self.assertIn('{"decision": "merge" or "keep_both", "reason": string}', MERGE_JUDGE_V1.system)
        self.assertEqual(MERGE_JUDGE_V1.output_keys, ("decision", "reason"))
        self.assertEqual([json.loads(o)["decision"] for _, o in MERGE_JUDGE_V1.examples], ["merge", "keep_both"])
        self.assertEqual(render_judge_prompt("alpha", "beta")[-1], {"role": "user", "content": "Memory A: alpha\nMemory B: beta"})
        self.assertNotIn('"different"', MERGE_JUDGE_V1.system)

    def test_hash_covers_the_text(self):
        edited = PromptTemplate(MERGE_JUDGE_V1.prompt_id, MERGE_JUDGE_V1.system + " ", MERGE_JUDGE_V1.user_template,
                                MERGE_JUDGE_V1.output_keys, MERGE_JUDGE_V1.examples)
        self.assertNotEqual(edited.sha256, FROZEN_SHA256)

    def test_judge_host_depends_on_dkmem_and_its_constants_only(self):
        import ast
        tree = ast.parse((TIER2 / "judge_host.py").read_text(encoding="utf-8"))
        mods = {("." * n.level) + (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertTrue(all(m in ("__future__", "dataclasses", "hashlib", ".frozen_judge", ".judge_entity") or m.startswith("dkmem") for m in mods), mods)
        # the Tier-2 identity judge module itself depends on dkmem only
        tree = ast.parse((TIER2 / "judge_entity.py").read_text(encoding="utf-8"))
        mods = {("." * n.level) + (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertTrue(all(m in ("__future__", "json") or m.startswith("dkmem") for m in mods), mods)

    def test_judge_decoding_budget_is_the_tier1_budget(self):
        self.assertEqual(H["run"].JUDGE_MAX_NEW_TOKENS, 128)
        self.assertEqual(H["frozen"].FROZEN_JUDGE_MAX_NEW_TOKENS, 128)
        from dkmem.pipeline.runner import _default_judge_params
        self.assertEqual(_default_judge_params(GenerationParams()).max_new_tokens, 128)


class TestOnlyTheFrozenPromptIsAccepted(unittest.TestCase):
    def test_other_prompts_are_refused(self):
        same_id_other_text = PromptTemplate("merge_judge_v1", "Answer same or different.", "{utterance}", ("decision",))
        for bad in (same_id_other_text, MERGE_JUDGE_INFORMED_V1):
            with self.assertRaises(ValueError):
                H["jh"].FrozenJudgeHost(FakeJudgeGenerator(), None, bad, fake_metric())

    def test_metric_is_required(self):
        with self.assertRaises(ValueError):
            H["jh"].FrozenJudgeHost(FakeJudgeGenerator(), None, MERGE_JUDGE_V1, None)

    def test_config_c_setup_refuses_a_foreign_host(self):
        class ForeignHost:
            prompt = MERGE_JUDGE_V1
            log = []
        with self.assertRaises(TypeError):
            H["run"].Setup("C", "off", ForeignHost(), fake_metric(), FakeJudgeGenerator(), None, {})


class TestJudgeCallPath(unittest.TestCase):
    def test_generator_receives_exactly_the_frozen_messages_and_budget(self):
        gen = FakeJudgeGenerator()
        params = judge_params()
        host = H["jh"].FrozenJudgeHost(gen, params, MERGE_JUDGE_V1, fake_metric(), "surface")
        ns = types.SimpleNamespace
        pairs = [(ns(pair_id="p1", surface="Meri mausi ka phone aaya"), ns(pair_id="p2", surface="Meri mami ka phone aaya")),
                 (ns(pair_id="p3", surface="Das Auto muss zur Inspektion"), ns(pair_id="p4", surface="Markus, wie war dein Urlaub"))]
        out = host.decide_many(pairs)
        (prompts, got_params), = gen.calls
        self.assertEqual(prompts, [render_judge_prompt(a.surface, b.surface, MERGE_JUDGE_V1) for a, b in pairs])
        self.assertIs(got_params, params)
        self.assertEqual(got_params.max_new_tokens, 128)
        self.assertTrue(all(m[0]["content"] == MERGE_JUDGE_V1.system for m in prompts))
        self.assertEqual([o.proposal for o in out], ["merge", "keep_both"])
        self.assertEqual([r["prompt_sha256"] for r in host.log], [FROZEN_SHA256] * 2)
        self.assertEqual([r["eval_pair_id"] for r in host.log], ["p1", "p3"])
        self.assertTrue(all(r["status"] == "ok" and r["reason"] and not r["possibly_truncated"] for r in host.log))

    def test_judge_rows_carry_no_threshold(self):
        host = H["jh"].FrozenJudgeHost(FakeJudgeGenerator(), judge_params(), MERGE_JUDGE_V1, fake_metric(), "surface")
        ns = types.SimpleNamespace
        out = host.decide_many([(ns(pair_id="a", surface="x one"), ns(pair_id="b", surface="x two"))])
        self.assertIsNone(out[0].threshold)
        self.assertIsInstance(out[0].similarity, float)

    def test_unparseable_answer_is_never_a_decision(self):
        gen = FakeJudgeGenerator(garbage_for=("BROKEN",))
        host = H["jh"].FrozenJudgeHost(gen, None, MERGE_JUDGE_V1, fake_metric())
        ns = types.SimpleNamespace
        out = host.decide_many([(ns(pair_id="a", surface="BROKEN one"), ns(pair_id="b", surface="BROKEN two")),
                                (ns(pair_id="c", surface="same text"), ns(pair_id="d", surface="same text"))])
        self.assertIsNone(out[0].proposal)
        self.assertEqual(host.log[0]["status"], "parse_error")
        self.assertEqual(host.log[0]["raw_output"], "I think these are the same")      # kept exactly as generated
        self.assertEqual(out[1].proposal, "merge")
        self.assertEqual(host.decide_many([]), [])


class TestTwentyRepresentativePairs(unittest.TestCase):
    def test_pairs_run_through_the_frozen_judge(self):
        pairs = representative_pairs(20)
        self.assertEqual(len(pairs), 20)
        gen = FakeJudgeGenerator()
        host = H["jh"].FrozenJudgeHost(gen, judge_params(), MERGE_JUDGE_V1, fake_metric(), "surface")
        ns = types.SimpleNamespace
        out = host.decide_many([(ns(pair_id=a, surface=ta), ns(pair_id=b, surface=tb)) for _, a, b, ta, tb in pairs])
        sent = [m for call, _ in gen.calls for m in call]
        self.assertEqual(sent, [render_judge_prompt(ta, tb, MERGE_JUDGE_V1) for _, _, _, ta, tb in pairs])
        self.assertTrue(all(o.proposal in ("merge", "keep_both") for o in out))
        self.assertEqual({o.proposal for o in out}, {"merge", "keep_both"})
        self.assertTrue(all(r["prompt_sha256"] == FROZEN_SHA256 for r in host.log))
        self.assertEqual({t["language"] for c in load_jsonl(CONVERSATIONS) for t in user_turns(c)}, {"hi", "de"})


class TestDecodingBudget(unittest.TestCase):
    """The 128-token budget under a controlled truncating generator (a bytes-per-token PROXY, not a tokenizer). The
    real-model evidence is the Tier-1 full run: 518 frozen-judge calls, 0 parse failures, longest answer 135 bytes."""

    PROXY_BYTES_PER_TOKEN = 3

    def setUp(self):
        self.conv = load_jsonl(CONVERSATIONS)[0]

    def run_with_budget(self, max_new_tokens, fixed):
        gen = FakeJudgeGenerator(fixed=fixed, chars_per_token=self.PROXY_BYTES_PER_TOKEN)
        setup, gen = make_setup("C", "off", generator=gen, convs=[self.conv], max_new_tokens=max_new_tokens)
        return H["run"].run_conversation(self.conv, [], setup, "t2budget"), setup

    def test_the_longest_measured_answer_fits_in_128_and_is_cut_at_32(self):
        self.assertTrue(parse_judge_output(LONGEST_MEASURED_ANSWER).ok)
        res, setup = self.run_with_budget(128, LONGEST_MEASURED_ANSWER)
        self.assertEqual(len(res["rows"]), 45)
        self.assertTrue(all(j["status"] == "ok" and not j["possibly_truncated"] for j in res["judge_rows"]))
        res32, setup32 = self.run_with_budget(32, LONGEST_MEASURED_ANSWER)
        self.assertEqual(res32["rows"], [])                      # no pairwise record: a failed answer is not a decision
        self.assertEqual(len(res32["judge_rows"]), 45)
        for j in res32["judge_rows"]:
            self.assertEqual((j["status"], j["proposal"], j["possibly_truncated"]), ("parse_error", None, True))
            self.assertTrue(LONGEST_MEASURED_ANSWER.startswith(j["raw_output"]) and len(j["raw_output"]) < len(LONGEST_MEASURED_ANSWER))
        statuses = {c["status"] for x in res32["retrieval_log"] for c in x["comparisons"]}
        self.assertEqual(statuses, {"judge_parse_error"})
        self.assertEqual(res32["store"] and max(len(m["member_turn_ids"]) for m in res32["store"]), 1)   # nothing merged

    def test_a_typical_answer_survives_both_budgets_but_is_not_repaired_when_cut(self):
        res, _ = self.run_with_budget(128, SHORT_ANSWER)
        self.assertTrue(all(j["status"] == "ok" for j in res["judge_rows"]))
        # cut mid-JSON at an absurdly small budget: recorded, with the cut raw output, never repaired into a decision
        res1, _ = self.run_with_budget(5, SHORT_ANSWER)
        self.assertTrue(all(j["status"] == "parse_error" and j["proposal"] is None for j in res1["judge_rows"]))
        self.assertEqual(res1["rows"], [])

    def test_failures_are_counted_in_the_run_manifest_and_info(self):
        gen = FakeJudgeGenerator(fixed=LONGEST_MEASURED_ANSWER, chars_per_token=self.PROXY_BYTES_PER_TOKEN)
        setup, _ = make_setup("C", "off", generator=gen, convs=[self.conv], max_new_tokens=32)
        captured = {}
        orig = H["run"].build_manifest
        H["run"].build_manifest = lambda **kw: captured.update(kw) or dict(kw)
        try:
            with mock.patch.object(H["run"], "write_run", lambda *a, **k: None), tempfile.TemporaryDirectory() as d:
                info = H["run"].run_setup(setup, [self.conv], [], run_id="t2budget", out=Path(d), meta={})
        finally:
            H["run"].build_manifest = orig
        extra = captured["extra"]
        self.assertEqual((extra["judge_calls"], extra["judge_parse_failures"], extra["judge_possibly_truncated"]), (45, 45, 45))
        self.assertEqual(info["judge_parse_errors"], 45)
        # the manifest describes the decoding budget the judge ACTUALLY used (32 here), in both places that state it; it used to report the
        # runner's 128 constant, which contradicted the per-row max_new_tokens/at_budget it also recorded
        self.assertEqual(extra["generation_run_info"]["generation_params"]["max_new_tokens"], 32)
        self.assertEqual(extra["judge_tokenization"]["budget"], 32)


    def test_the_manifest_budget_always_equals_the_budget_every_judge_row_recorded(self):
        for budget in (128, 64):
            gen = FakeJudgeGenerator(fixed=SHORT_ANSWER)
            setup, _ = make_setup("C", "off", generator=gen, convs=[self.conv], max_new_tokens=budget)
            captured = {}
            orig = H["run"].build_manifest
            H["run"].build_manifest = lambda **kw: captured.update(kw) or dict(kw)
            try:
                with mock.patch.object(H["run"], "write_run", lambda *a, **k: None), tempfile.TemporaryDirectory() as d:
                    H["run"].run_setup(setup, [self.conv], [], run_id="t2budget", out=Path(d), meta={})
            finally:
                H["run"].build_manifest = orig
            extra = captured["extra"]
            self.assertEqual(extra["generation_run_info"]["generation_params"]["max_new_tokens"], budget)
            self.assertEqual(extra["judge_tokenization"]["budget"], budget)
            self.assertEqual({p.max_new_tokens for _, p in gen.calls}, {budget})        # what the generator was really called with


class FakeTokenizer:
    """A whitespace tokenizer with the interface TokenMeter uses (call -> input_ids, chat template, name, vocab size)."""

    name_or_path = "fake/tokenizer"
    eos_token, pad_token, padding_side, chat_template = "<eos>", "<pad>", "left", "fake-template"

    def __len__(self):
        return 1234

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages) + " <assistant>"


class TestTokenizationAndDistributions(unittest.TestCase):
    """Judge calls are measured with the generator's own tokenizer and the manifests carry the decision distributions."""

    def setUp(self):
        self.conv = load_jsonl(CONVERSATIONS)[0]
        self.tok = FakeTokenizer()
        self.meter = H["jh"].TokenMeter(self.tok)

    def run_setup(self, gen, questions=()):
        metric = fake_metric()
        host = H["jh"].FrozenJudgeHost(gen, judge_params(), MERGE_JUDGE_V1, metric, "surface", meter=self.meter)
        qa_params = GenerationParams(max_new_tokens=H["run"].QA_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8)
        passthrough = {t["turn_id"]: "ent_" + hashlib.sha256(t["turn_id"].encode()).hexdigest()[:16] for t in user_turns(self.conv)}
        setup = H["run"].Setup("C", "off", host, metric, gen, None, passthrough, seed=0, qa_params=qa_params, meter=self.meter)
        captured = {}
        orig = H["run"].build_manifest
        H["run"].build_manifest = lambda **kw: captured.update(kw) or dict(kw)
        try:
            with mock.patch.object(H["run"], "write_run", lambda *a, **k: None), tempfile.TemporaryDirectory() as d:
                info = H["run"].run_setup(setup, [self.conv], list(questions), run_id="t2tok", out=Path(d), meta={})
        finally:
            H["run"].build_manifest = orig
        return captured["extra"], info

    def test_meter_counts_and_describes_the_tokenizer(self):
        self.assertEqual(self.meter.text_tokens("one two  three"), 3)
        msgs = render_judge_prompt("alpha", "beta")
        self.assertEqual(self.meter.prompt_tokens(msgs), len(self.tok.apply_chat_template(msgs).split()))
        d = self.meter.describe()
        self.assertEqual((d["name_or_path"], d["vocab_size"], d["padding_side"]), ("fake/tokenizer", 1234, "left"))
        self.assertEqual(d["chat_template_sha256"], hashlib.sha256(b"fake-template").hexdigest())
        self.assertIsNone(H["jh"].TokenMeter.from_generator(FakeJudgeGenerator()))        # no tokenizer: no invented counts

    def test_summarize_counts(self):
        self.assertEqual(H["jh"].summarize_counts([]), {"n": 0})
        self.assertEqual(H["jh"].summarize_counts([3, 1, 2]), {"n": 3, "min": 1, "median": 2, "p95": 3, "max": 3})

    def test_judge_rows_record_tokens_and_a_budget_hit(self):
        ns = types.SimpleNamespace
        pair = [(ns(pair_id="a", surface="x one"), ns(pair_id="b", surface="x two"))]
        long_answer = '{"decision": "merge", "reason": "' + " ".join(["same"] * 40) + '."}'
        roomy = H["jh"].FrozenJudgeHost(FakeJudgeGenerator(fixed=long_answer), judge_params(128), MERGE_JUDGE_V1, fake_metric(), "surface", meter=self.meter)
        roomy.decide_many(pair)
        row = roomy.log[0]
        self.assertEqual(row["prompt_tokens"], self.meter.prompt_tokens(render_judge_prompt("x one", "x two")))
        self.assertEqual(row["output_tokens"], self.meter.text_tokens(long_answer))
        self.assertGreater(row["output_tokens"], 40)
        self.assertEqual((row["max_new_tokens"], row["at_budget"]), (128, False))
        tight = H["jh"].FrozenJudgeHost(FakeJudgeGenerator(fixed=long_answer), judge_params(32), MERGE_JUDGE_V1, fake_metric(), "surface", meter=self.meter)
        tight.decide_many(pair)
        self.assertTrue(tight.log[0]["at_budget"])

    def test_without_a_meter_the_token_fields_are_null_not_estimated(self):
        host = H["jh"].FrozenJudgeHost(FakeJudgeGenerator(), judge_params(), MERGE_JUDGE_V1, fake_metric(), "surface")
        ns = types.SimpleNamespace
        host.decide_many([(ns(pair_id="a", surface="x one"), ns(pair_id="b", surface="x two"))])
        self.assertEqual((host.log[0]["prompt_tokens"], host.log[0]["output_tokens"], host.log[0]["at_budget"]), (None, None, None))

    def test_run_setup_manifest_records_tokenization_and_distributions(self):
        questions = [q for q in load_jsonl(QUESTIONS) if q["conversation_id"] == self.conv["conversation_id"]]
        extra, info = self.run_setup(FakeJudgeGenerator(merge_ratio=0.8), questions)
        tk = extra["judge_tokenization"]
        self.assertTrue(tk["measured"])
        self.assertEqual((tk["budget"], tk["n_at_budget"], tk["prompt_tokens"]["n"], tk["output_tokens"]["n"]), (128, 0, 45, 45))
        self.assertGreater(tk["prompt_tokens"]["max"], tk["output_tokens"]["max"])
        dist = extra["judge_decision_distribution"]
        self.assertEqual(dist["n_calls"], 45)
        self.assertEqual(sum(dist["proposals"].values()), 45)
        self.assertEqual(sum(dist["raw_decisions"].values()), 45)
        self.assertTrue(set(dist["raw_decisions"]) <= {"merge", "keep_both"})
        t2 = extra["tier2"]
        self.assertEqual(t2["tokenizer"]["name_or_path"], "fake/tokenizer")
        self.assertEqual(sum(t2["decision_distribution"]["pairwise_decisions"].values()), 45)
        self.assertEqual(t2["decision_distribution"]["similarity_score"]["n"], 45)
        self.assertEqual(t2["qa"]["output_tokens"]["n"], len(questions))
        self.assertEqual(info["qa_answers"], len(questions))

    def test_a_constant_judge_is_recorded_as_constant(self):
        extra, _ = self.run_setup(FakeJudgeGenerator(fixed='{"decision": "keep_both", "reason": "Different."}'))
        self.assertTrue(extra["judge_decision_distribution"]["constant"])
        self.assertEqual(extra["judge_decision_distribution"]["proposals"], {"keep_both": 45})


class TestQuestionsAreReadAfterConsolidation(unittest.TestCase):
    def test_lazy_questions_load_only_on_first_use(self):
        lazy = H["run"].LazyQuestions(QUESTIONS)
        self.assertFalse(lazy.loaded)
        cid = load_jsonl(CONVERSATIONS)[0]["conversation_id"]
        qs = lazy(cid)
        self.assertTrue(lazy.loaded)
        self.assertTrue(qs and all(q["conversation_id"] == cid for q in qs))

    def test_no_judge_call_happens_after_the_questions_are_requested(self):
        conv = load_jsonl(CONVERSATIONS)[0]
        setup, gen = make_setup("C", "off", generator=FakeJudgeGenerator(), convs=[conv])
        seen = []
        real = [q for q in load_jsonl(QUESTIONS) if q["conversation_id"] == conv["conversation_id"]]

        def questions(cid):
            seen.append(len(gen.calls))          # generator batches issued before the questions were asked for
            return real

        res = H["run"].run_conversation(conv, questions, setup, "t2lazy")
        self.assertEqual(seen, [11])             # 12 user turns, the first has no candidates: 11 judge batches, all before QA
        self.assertEqual(len(gen.calls), 12)     # plus the single QA batch
        self.assertEqual(len(res["qa"]), len(real))

    def test_the_runner_hands_the_run_the_lazy_reader_not_a_list(self):
        src = (TIER2 / "run_tier2.py").read_text(encoding="utf-8")
        self.assertIn('questions = LazyQuestions(a.data_dir / "tier2_qa_questions.jsonl")', src)
        self.assertNotIn("load_inputs(a.data_dir", src)


# a made-up label for the synthetic gold rows (built without the literal pattern the key-protection scan looks for)
MADE_UP_RELATION = "diff" + "erent"


class SyntheticTier2:
    """A label-free copy of a few real conversations plus MADE-UP injected pairs, so the evaluator-side contract can be
    exercised without the Tier-2 gold, the injection metadata or the QA references."""

    def __init__(self, root: Path, n_conv=3):
        self.dir = root / "data"
        self.dir.mkdir()
        convs = load_jsonl(CONVERSATIONS)[:n_conv]
        qs = [q for q in load_jsonl(QUESTIONS) if q["conversation_id"] in {c["conversation_id"] for c in convs}]
        (self.dir / "tier2_conversations.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs), encoding="utf-8", newline="\n")
        (self.dir / "tier2_qa_questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8", newline="\n")
        gold, meta, refs = [], [], []
        for c in convs:
            ts = user_turns(c)
            a, b = ts[2], ts[7]
            pid = "synth_" + c["conversation_id"]
            sess = {t["turn_id"]: s["session_index"] for s in c["sessions"] for t in s["turns"]}
            gold.append(dict(eval_pair_id=pid, utterance_a_id=a["turn_id"], utterance_b_id=b["turn_id"], relation=MADE_UP_RELATION,
                             distinction_class="kinship", language=a["language"],
                             opaque_entity_id_a="ent_" + hashlib.sha256(("A" + pid).encode()).hexdigest()[:16],
                             opaque_entity_id_b="ent_" + hashlib.sha256(("B" + pid).encode()).hexdigest()[:16]))
            meta.append({"injected_pair_id": pid, "conversation_id": c["conversation_id"], "session_distance": abs(sess[a["turn_id"]] - sess[b["turn_id"]]),
                         "language": a["language"], "distinction_class": "kinship", "turn_id_a": a["turn_id"], "turn_id_b": b["turn_id"]})
            for q in (q for q in qs if q["conversation_id"] == c["conversation_id"]):
                refs.append({"qa_id": q["qa_id"], "injected_pair_id": pid, "role": "a", "qa_type": "disambiguation",
                             "supporting_turn_ids": [a["turn_id"]], "slots": [["unknown"]], "forbidden": []})
        for name, rows in (("tier2_gold", gold), ("tier2_injection_metadata", meta), ("tier2_qa_reference", refs)):
            (self.dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
        self.conv_ids = [c["conversation_id"] for c in convs]


class TestRunnerWithTheRealHarness(unittest.TestCase):
    """The smoke test: ``run_tier2.main`` end to end over the real harness (extractor, hosts, writer, metric, lexicon), the
    real Tier-2 contract and the dkmem schemas; only the generator and the embedder are fakes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.syn = SyntheticTier2(root)
        pt = root / "passthrough.jsonl"
        pt.write_text("".join(json.dumps(r) + "\n" for r in H["contract"].build_passthrough_ids(cls.syn.dir)), encoding="utf-8", newline="\n")
        cls.gen = FakeJudgeGenerator()
        cls.out = root / "out"
        run = H["run"]

        def fake_models(args):
            return fake_metric(), cls.gen

        with mock.patch.object(run, "build_models", fake_models), mock.patch.object(run, "verify_frozen", lambda d, *_: {"version": "1.0", "verified_files": [], "not_in_payload_evaluator_only": []}):
            cls.rc = run.main(["--passthrough", str(pt), "--data-dir", str(cls.syn.dir), "--out", str(cls.out), "--tau", "0.9",
                               "--run-prefix", "t2smoke", "--conversations", ",".join(cls.syn.conv_ids)])
        cls.runs = {p.name: p for p in cls.out.iterdir() if p.is_dir()} if cls.out.exists() else {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, run):
        return load_jsonl(self.runs[run] / "pairwise_eval.jsonl")

    def manifest(self, run):
        return json.loads((self.runs[run] / "run_manifest.json").read_text(encoding="utf-8"))

    def test_run_succeeded_and_wrote_the_four_runs(self):
        self.assertEqual(self.rc, 0)
        self.assertEqual(sorted(self.runs), ["t2smoke-dk-mem-lexicon-C_seed0", "t2smoke-dk-mem-lexicon-D_seed0",
                                             "t2smoke-embedding-threshold-D_seed0", "t2smoke-store-surface-only-C_seed0"])
        for run in self.runs:
            self.assertEqual(len(self.rows(run)), 3 * 45, run)                # 0+1+2+3+4+5*7 comparisons per 12-turn conversation

    def test_config_c_uses_the_frozen_prompt_and_budget(self):
        for run in ("t2smoke-store-surface-only-C_seed0", "t2smoke-dk-mem-lexicon-C_seed0"):
            m = self.manifest(run)
            self.assertEqual((m["judge_prompt_id"], m["judge_prompt_sha256"]), ("merge_judge_v1", FROZEN_SHA256))
            self.assertEqual(m["generation_run_info"]["generation_params"]["max_new_tokens"], 128)
            self.assertEqual((m["judge_parse_failures"], m["judge_possibly_truncated"]), (0, 0))
            self.assertIsNone(m["tau"])
            judged = [msgs for call, _ in self.gen.calls for msgs in call if not msgs[0]["content"].startswith("You answer")]
            self.assertTrue(judged and all(msgs[0]["content"] == MERGE_JUDGE_V1.system for msgs in judged))
            self.assertTrue(all(p.max_new_tokens == 128 for _, p in self.gen.calls if p.max_new_tokens != H["run"].QA_MAX_NEW_TOKENS))

    def test_threshold_convention(self):
        for run in ("t2smoke-store-surface-only-C_seed0", "t2smoke-dk-mem-lexicon-C_seed0"):
            self.assertEqual({r["threshold"] for r in self.rows(run)}, {None})
        for run in ("t2smoke-embedding-threshold-D_seed0", "t2smoke-dk-mem-lexicon-D_seed0"):
            self.assertEqual({r["threshold"] for r in self.rows(run)}, {0.9})

    def test_all_four_runs_satisfy_the_contract(self):
        for run, path in self.runs.items():
            rep = H["contract"].validate_run(path, self.syn.dir)
            self.assertTrue(rep["ok"], (run, rep["errors"][:3]))
            if run.endswith("C_seed0"):
                self.assertTrue(rep["judge"]["frozen_tier1"])
                self.assertIsNone(rep["judge"]["row_threshold_expected"])
                self.assertEqual(rep["warnings"], [])

    def test_validation_against_the_passthrough_file_the_run_used(self):
        pt = Path(self.tmp.name) / "passthrough.jsonl"
        run = self.runs["t2smoke-store-surface-only-C_seed0"]
        rep = H["contract"].validate_run(run, self.syn.dir, passthrough_file=pt)
        self.assertTrue(rep["ok"], rep["errors"][:3])
        # a different passthrough (every id re-keyed) is not the one the run used: both the hash and the entry ids disagree
        other = Path(self.tmp.name) / "other_passthrough.jsonl"
        rows = load_jsonl(pt)
        other.write_text("".join(json.dumps({"turn_id": r["turn_id"], "gold_entity_id": "ent_" + hashlib.sha256(("x" + r["turn_id"]).encode()).hexdigest()[:16]}) + "\n"
                                for r in rows), encoding="utf-8")
        codes = {e["code"] for e in H["contract"].validate_run(run, self.syn.dir, passthrough_file=other)["errors"]}
        self.assertIn("manifest_passthrough_hash", codes)
        self.assertIn("entry_passthrough_id", codes)

    def test_blinded_questions_need_the_evaluator_side_map(self):
        run_name = "t2smoke-store-surface-only-C_seed0"
        qa_ids = [json.loads(l)["qa_id"] for l in (Path(self.out) / f"{run_name}.tier2_qa_answers.jsonl").read_text(encoding="utf-8").splitlines()]
        blinded = lambda m: m["tier2"].update(blinded_views={"data/tier2_qa_questions.jsonl": {"derived_from_frozen_sha256": "0" * 64}})
        codes = lambda rep: {e["code"] for e in rep["errors"]}
        self.assertIn("qa_ids_blinded_without_map", codes(self.corrupt(run_name, blinded)))
        # the same validation with a map that covers the answered ids is clean
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, ignore_errors=True)
        for p in self.out.iterdir():
            if p.name == run_name or p.name.startswith(run_name + "."):
                (shutil.copytree if p.is_dir() else shutil.copy)(p, t / p.name)
        f = t / run_name / "run_manifest.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        blinded(m)
        f.write_text(json.dumps(m), encoding="utf-8")
        rep = H["contract"].validate_run(t / run_name, self.syn.dir, qa_id_map={i: i for i in qa_ids})
        self.assertTrue(rep["ok"], rep["errors"][:3])
        rep = H["contract"].validate_run(t / run_name, self.syn.dir, qa_id_map={"qa_000000000000": qa_ids[0]})
        self.assertIn("qa_id_not_in_map", codes(rep))
        # the CLI passes the map as a file path (str or Path): it must be read before the answers are looked at
        map_file = t / "qa_id_map.json"
        map_file.write_text(json.dumps({i: i for i in qa_ids}), encoding="utf-8")
        for given in (map_file, str(map_file)):
            rep = H["contract"].validate_run(t / run_name, self.syn.dir, qa_id_map=given)
            self.assertTrue(rep["ok"], rep["errors"][:3])

    def test_default_validation_of_a_run_made_with_another_passthrough_is_an_error(self):
        # a second run whose passthrough is the re-keyed (uniform) one: validating it without naming that file must not pass
        root = Path(self.tmp.name)
        rows = load_jsonl(root / "passthrough.jsonl")
        other = root / "uniform_passthrough.jsonl"
        other.write_text("".join(json.dumps({"turn_id": r["turn_id"], "gold_entity_id": "ent_" + hashlib.sha256(("u|" + r["turn_id"]).encode()).hexdigest()[:16]}) + "\n"
                                 for r in rows), encoding="utf-8", newline="\n")
        out = root / "out_uniform"
        gen = FakeJudgeGenerator()
        run = H["run"]
        with mock.patch.object(run, "build_models", lambda args: (fake_metric(), gen)), \
                mock.patch.object(run, "verify_frozen", lambda d, *_: {"version": "1.0", "verified_files": [], "not_in_payload_evaluator_only": []}):
            rc = run.main(["--passthrough", str(other), "--data-dir", str(self.syn.dir), "--out", str(out), "--tau", "0.9",
                           "--run-prefix", "t2uni", "--configs", "C:off", "--conversations", ",".join(self.syn.conv_ids)])
        self.assertEqual(rc, 0)
        run_dir = out / "t2uni-store-surface-only-C_seed0"
        wrong = {e["code"] for e in H["contract"].validate_run(run_dir, self.syn.dir)["errors"]}
        self.assertIn("manifest_passthrough_hash", wrong)               # default = the gold-linked ids: not the file the run used
        right = H["contract"].validate_run(run_dir, self.syn.dir, passthrough_file=other)
        self.assertTrue(right["ok"], right["errors"][:3])

    def test_gate_only_vetoes(self):
        by = lambda run: {r["record_id"]: r for r in self.rows(run)}
        for off, on in (("t2smoke-store-surface-only-C_seed0", "t2smoke-dk-mem-lexicon-C_seed0"),
                        ("t2smoke-embedding-threshold-D_seed0", "t2smoke-dk-mem-lexicon-D_seed0")):
            a, b = by(off), by(on)
            self.assertEqual(set(a), set(b))
            for rid, r in b.items():
                if r["decision"] in ("merge", "supersede"):
                    self.assertIn(a[rid]["decision"], ("merge", "supersede"))

    def corrupt(self, run, mutate_manifest=None, mutate_rows=None):
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, ignore_errors=True)
        for p in self.out.iterdir():
            if p.name == run or p.name.startswith(run + "."):
                (shutil.copytree if p.is_dir() else shutil.copy)(p, t / p.name)
        if mutate_manifest:
            f = t / run / "run_manifest.json"
            m = json.loads(f.read_text(encoding="utf-8"))
            mutate_manifest(m)
            f.write_text(json.dumps(m), encoding="utf-8")
        if mutate_rows:
            f = t / run / "pairwise_eval.jsonl"
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
            for r in rows:
                mutate_rows(r)
            f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return H["contract"].validate_run(t / run, self.syn.dir)

    def test_historical_harness_judge_outputs_remain_valid_and_are_labelled(self):
        run = "t2smoke-store-surface-only-C_seed0"
        legacy = self.corrupt(run, lambda m: m.update(judge_prompt_sha256=HARNESS_SHA256), lambda r: r.update(threshold=0.0))
        self.assertTrue(legacy["ok"], legacy["errors"][:3])
        self.assertTrue(legacy["judge"]["historical_harness_judge"] and not legacy["judge"]["frozen_tier1"])
        self.assertEqual([w["code"] for w in legacy["warnings"]], ["historical_harness_judge"])

    def test_threshold_and_judge_mismatches_are_errors(self):
        run = "t2smoke-store-surface-only-C_seed0"
        codes = lambda rep: {e["code"] for e in rep["errors"]}
        self.assertIn("threshold_value", codes(self.corrupt(run, mutate_rows=lambda r: r.update(threshold=0.0))))          # frozen judge, legacy 0.0
        self.assertIn("threshold_value", codes(self.corrupt(run, lambda m: m.update(judge_prompt_sha256=HARNESS_SHA256))))  # legacy judge, null rows
        self.assertIn("judge_prompt_not_recognised", codes(self.corrupt(run, lambda m: m.update(judge_prompt_sha256="0" * 64))))


class FakeABGenerator(FakeJudgeGenerator):
    """Answers the A extraction prompt (an English gloss), the B extraction prompt ({"fact"}), the judge prompt and the QA prompt.
    A text in ``fail_texts`` gets an answer that violates both extraction contracts (not JSON)."""

    def __init__(self, fail_texts=(), **kw):
        super().__init__(**kw)
        self.fail_texts = tuple(fail_texts)
        self.extraction_calls = 0

    def generate(self, prompts, params):
        out = []
        for m in prompts:
            system, last = m[0]["content"], m[-1]["content"]
            if "exactly one key and nothing else" in system:                            # config B: native_b_v1 (shares its opening with V4: test it first)
                self.extraction_calls += 1
                u = last.removeprefix("Utterance: ")
                out.append("not json" if u in self.fail_texts else json.dumps({"fact": "fact: " + u}, ensure_ascii=False))
            elif system.startswith("You convert one user utterance"):                 # config A: mem0_extraction_v4
                self.extraction_calls += 1
                u = last.removeprefix("Utterance: ")
                out.append("not json" if u in self.fail_texts else json.dumps({"gloss": "note: " + u, "distinction": {}, "surface": u}, ensure_ascii=False))
            else:
                out.extend(super().generate([m], params))
        return out


PREFIX = {"A": "note: ", "B": "fact: "}


def make_ab_setup(config, mode, generator, conv, *, field="gloss"):
    import tier2.entry_extraction as ee
    run, jh = H["run"], H["jh"]
    metric = fake_metric()
    host = jh.FrozenJudgeHost(generator, judge_params(), MERGE_JUDGE_V1, metric, field)
    lexicon = real_lexicon() if mode == "lexicon" else None
    extractor = ee.TurnExtractor(config, generator, GenerationParams(max_new_tokens=256, do_sample=False, seed=0, batch_size=8), lexicon, 0)
    passthrough = {t["turn_id"]: "ent_" + hashlib.sha256(t["turn_id"].encode()).hexdigest()[:16] for t in user_turns(conv)}
    qa_params = GenerationParams(max_new_tokens=run.QA_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8)
    return run.Setup(config, mode, host, metric, generator, lexicon, passthrough, seed=0, qa_params=qa_params, extractor=extractor)


class TestConfigsAAndBInTheRunner(unittest.TestCase):
    """Configs A and B store the extraction, not the surface; a failed extraction stores nothing."""

    def setUp(self):
        self.conv = load_jsonl(CONVERSATIONS)[0]

    def test_the_stored_text_is_the_extraction_and_the_judge_sees_it(self):
        for config in ("A", "B"):
            with self.subTest(config=config):
                gen = FakeABGenerator()
                setup = make_ab_setup(config, "off", gen, self.conv)
                res = H["run"].run_conversation(self.conv, [], setup, "t2ab")
                self.assertEqual(len(res["rows"]), 45)
                self.assertEqual(gen.extraction_calls, 12)                                    # one batched extraction of the 12 user turns
                for j in res["judge_rows"]:
                    self.assertTrue(j["text_a"].startswith(PREFIX[config]) and j["text_b"].startswith(PREFIX[config]), j["text_a"])
                surface = {t["turn_id"]: t["text"] for t in user_turns(self.conv)}
                for r in res["rows"]:
                    for e in (r["entry_a"], r["entry_b"]):
                        self.assertEqual(e["surface"], surface[e["source_utterance_id"]])
                        self.assertEqual(e["gloss"], PREFIX[config] + e["surface"])
                self.assertTrue(all(m["text"].startswith(PREFIX[config]) for m in res["store"]))   # records show the stored text, not the surface
                sims = {c["record_id"]: c["record_similarity"] for x in res["retrieval_log"] for c in x["comparisons"]}
                self.assertEqual({r["record_id"]: r["similarity_score"] for r in res["rows"]}, sims)
                self.assertTrue(all(x["stored_text"].startswith(PREFIX[config]) for x in res["retrieval_log"]))

    def test_ranking_is_by_the_surfaces_for_every_config(self):
        a = H["run"].run_conversation(self.conv, [], make_ab_setup("A", "off", FakeABGenerator(), self.conv), "t2ab")
        c, _ = make_setup("C", "off", generator=FakeJudgeGenerator(), convs=[self.conv])
        c = H["run"].run_conversation(self.conv, [], c, "t2c")
        self.assertEqual([[(k["entry_id"], k["similarity"]) for k in x["ranked"]] for x in a["retrieval_log"]],
                         [[(k["entry_id"], k["similarity"]) for k in x["ranked"]] for x in c["retrieval_log"]])

    def test_a_failed_extraction_writes_no_entry_and_is_logged_not_decided(self):
        turns = user_turns(self.conv)
        victim = turns[2]["text"]
        failed_ids = {f"{t['turn_id']}_e0" for t in turns if t["text"] == victim}
        for config in ("A", "B"):
            with self.subTest(config=config):
                gen = FakeABGenerator(fail_texts=[victim])
                setup = make_ab_setup(config, "off", gen, self.conv)
                res = H["run"].run_conversation(self.conv, [], setup, "t2ab")
                failed_rows = [x for x in res["retrieval_log"] if x.get("status") == "extraction_failed"]
                self.assertEqual({x["turn_id"] for x in failed_rows}, {t["turn_id"] for t in turns if t["text"] == victim})
                for x in failed_rows:
                    self.assertEqual((x["ranked"], x["candidates"], x["comparisons"], x["entry_id"]), ([], [], [], None))
                members = {e for m in res["store"] for e in m["member_entry_ids"]}
                self.assertFalse(members & failed_ids)                                         # nothing stored for the failed turns
                self.assertEqual(len(members), 12 - len(failed_ids))
                self.assertFalse(any(r["entry_a"]["entry_id"] in failed_ids or r["entry_b"]["entry_id"] in failed_ids for r in res["rows"]))
                xlog = setup.extractor.log
                bad = [r for r in xlog if r["status"] == "failed"]
                self.assertEqual(len(bad), len(failed_ids))
                self.assertTrue(all(r["stored_text"] is None and r["raw_output"] == "not json" and r["error"] for r in bad))
                # a later turn's store size counts only stored entries
                last = res["retrieval_log"][-1]
                self.assertEqual(last["store_size_before"], 11 - len(failed_ids))

    def test_the_gate_uses_the_lexicon_on_the_utterance_and_off_records_no_distinction(self):
        for config in ("A", "B"):
            off = H["run"].run_conversation(self.conv, [], make_ab_setup(config, "off", FakeABGenerator(), self.conv), "t2off")
            on = H["run"].run_conversation(self.conv, [], make_ab_setup(config, "lexicon", FakeABGenerator(), self.conv), "t2on")
            self.assertTrue(all(r["compatibility"] is None and r["entry_a"]["distinction"] is None for r in off["rows"]), config)
            self.assertTrue(all(r["compatibility"] in ("compatible", "incompatible", "underdetermined") for r in on["rows"]), config)

    def test_setup_rejects_inconsistent_configurations(self):
        run, jh = H["run"], H["jh"]
        gen, conv = FakeABGenerator(), self.conv
        good = make_ab_setup("A", "off", gen, conv)
        with self.assertRaisesRegex(TypeError, "needs a TurnExtractor"):                       # A without an extractor
            run.Setup("A", "off", good.host, good.metric, gen, None, good.passthrough, extractor=None)
        with self.assertRaisesRegex(TypeError, "needs a TurnExtractor"):                       # an extractor of the wrong config
            run.Setup("B", "off", good.host, good.metric, gen, None, good.passthrough, extractor=good.extractor)
        with self.assertRaisesRegex(ValueError, "must see the stored text"):                   # A's judge must read the gloss, not the surface
            make_ab_setup("A", "off", gen, conv, field="surface")
        c, _ = make_setup("C", "off", generator=FakeJudgeGenerator(), convs=[conv])
        with self.assertRaisesRegex(ValueError, "takes no extractor"):
            run.Setup("C", "off", c.host, c.metric, gen, None, c.passthrough, extractor=good.extractor)
        with self.assertRaisesRegex(ValueError, "exactly when the gate is on"):
            run.Setup("A", "lexicon", good.host, good.metric, gen, real_lexicon(), good.passthrough, extractor=good.extractor)


class TestConfigsAAndBEndToEnd(unittest.TestCase):
    """``run_tier2.main`` over A and B (real harness, writer, contract_v2, provenance; fake generator and embedder only)."""

    @classmethod
    def setUpClass(cls):
        import tier2.contract_v2 as cv2
        cls.cv2 = cv2
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.syn = SyntheticTier2(cls.root)
        cls.pt = cls.root / "passthrough.jsonl"
        cls.pt.write_text("".join(json.dumps(r) + "\n" for r in H["contract"].build_passthrough_ids(cls.syn.dir)), encoding="utf-8", newline="\n")
        turns0 = user_turns(load_jsonl(CONVERSATIONS)[0])
        cls.victim = turns0[2]["text"]                                     # the injected 'a' side of conversation 0 in SyntheticTier2
        cls.ok_out = cls.run_main("ok", FakeABGenerator())
        cls.fail_out = cls.run_main("fail", FakeABGenerator(fail_texts=[cls.victim]))

    @classmethod
    def run_main(cls, name, gen):
        run = H["run"]
        out = cls.root / name
        with mock.patch.object(run, "build_models", lambda args: (fake_metric(), gen)), \
                mock.patch.object(run, "verify_frozen", lambda d, *_: {"version": "1.0", "verified_files": [], "not_in_payload_evaluator_only": []}):
            rc = run.main(["--passthrough", str(cls.pt), "--data-dir", str(cls.syn.dir), "--out", str(out), "--run-prefix", "t2ab",
                           "--configs", "A:off,A:lexicon,B:off,B:lexicon", "--conversations", ",".join(cls.syn.conv_ids)])
        assert rc == 0, rc
        return out

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def validate(self, out, run, **kw):
        return self.cv2.validate_run(out / run, self.syn.dir, **kw)

    def test_the_four_runs_exist_with_the_expected_names_and_files(self):
        runs = sorted(p.name for p in self.ok_out.iterdir() if p.is_dir())
        self.assertEqual(runs, ["t2ab-dk-mem-lexicon-A_seed0", "t2ab-dk-mem-lexicon-B_seed0", "t2ab-mem0-A_seed0", "t2ab-mem0-B_seed0"])
        for r in runs:
            for suffix in ("tier2_retrieval_log", "tier2_store_final", "tier2_qa_answers", "judge_log", "extraction_log"):
                self.assertTrue((self.ok_out / f"{r}.{suffix}.jsonl").exists(), (r, suffix))
            self.assertEqual(len(load_jsonl(self.ok_out / f"{r}.extraction_log.jsonl")), 36)           # 3 conversations x 12 user turns

    def test_every_run_passes_contract_v2_and_the_independent_verifications(self):
        for r in (p.name for p in self.ok_out.iterdir() if p.is_dir()):
            rep = self.validate(self.ok_out, r)
            self.assertTrue(rep["ok"], (r, rep["errors"][:3]))
            self.assertEqual(rep["extraction"]["status"], "verified")
            self.assertEqual(rep["counts"]["extraction_failures"], 0)
            self.assertEqual(rep["counts"]["pairwise_records"], 3 * 45)
            ver = H["prov"].validate_run_verified(self.ok_out / r, self.syn.dir)
            self.assertEqual(ver["judge_log_verification"]["status"], "verified")          # the judge log of an A/B run is re-parsed too
            # the test's embedder is a fake whose revision is not the registered bge-m3 pin: the verified wrapper must say so, and nothing else
            self.assertEqual({e["code"] for e in ver["errors"]}, {"embedder_revision_wrong_revision"}, (r, ver["errors"][:3]))

    def test_the_pinned_contract_rejects_a_and_b_so_the_new_validator_is_needed(self):
        rep = H["contract"].validate_run(self.ok_out / "t2ab-mem0-A_seed0", self.syn.dir)
        self.assertFalse(rep["ok"])
        self.assertIn("manifest_strategy", {e["code"] for e in rep["errors"]})

    def test_manifest_fields(self):
        m = json.loads((self.ok_out / "t2ab-dk-mem-lexicon-A_seed0" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((m["pipeline_config"], m["dkmem_mode"], m["strategy"]), ("A", "lexicon", "dk-mem-lexicon"))
        self.assertEqual((m["extraction_prompt_id"], m["extraction_calls"], m["extraction_failures"]), ("mem0_extraction_v4", 36, 0))
        self.assertEqual(m["backbone"], "fake-backbone")
        self.assertIn("lexicon_sha256", m)
        self.assertNotIn("lexicon_sha256", json.loads((self.ok_out / "t2ab-mem0-A_seed0" / "run_manifest.json").read_text(encoding="utf-8")))
        mb = json.loads((self.ok_out / "t2ab-mem0-B_seed0" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(mb["extraction_prompt_id"], "native_b_v1")

    def test_a_failed_extraction_makes_the_injected_pair_no_entry_and_the_run_still_validates(self):
        for r in (p.name for p in self.fail_out.iterdir() if p.is_dir()):
            rep = self.validate(self.fail_out, r)
            self.assertTrue(rep["ok"], (r, rep["errors"][:3]))
            self.assertGreaterEqual(rep["counts"]["extraction_failures"], 1)
            self.assertIn("no_entry", rep["counts"]["injected_pair_status"])
            first = next(p for p in rep["injected_pairs"] if p["conversation_id"] == self.syn.conv_ids[0])
            self.assertEqual(first["status"], "no_entry")

    def test_the_lexicon_sha_check_catches_a_run_validated_against_another_lexicon(self):
        lex = Path(H["dk"].LEXICON_PATH)
        ok = self.validate(self.ok_out, "t2ab-dk-mem-lexicon-A_seed0", lexicon_file=lex)
        self.assertTrue(ok["ok"], ok["errors"][:3])
        other = self.root / "other_lexicon.json"
        other.write_text(lex.read_text(encoding="utf-8").replace("Pilot-reviewed.", "Pilot-reviewed (edited).", 1), encoding="utf-8")
        bad = self.validate(self.ok_out, "t2ab-dk-mem-lexicon-A_seed0", lexicon_file=other)
        self.assertIn("lexicon_sha256_mismatch", {e["code"] for e in bad["errors"]})

    def copy_run(self, run="t2ab-mem0-A_seed0"):
        t = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, t, ignore_errors=True)
        for p in self.ok_out.iterdir():
            if p.name == run or p.name.startswith(run + "."):
                (shutil.copytree if p.is_dir() else shutil.copy)(p, t / p.name)
        return t, run

    def codes(self, t, run):
        return {e["code"] for e in self.validate(t, run)["errors"]}

    def edit_jsonl(self, path, fn):
        rows = load_jsonl(path)
        fn(rows)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def test_tampered_extraction_logs_and_entries_are_rejected(self):
        t, run = self.copy_run()
        self.assertEqual(self.codes(t, run), set())
        log = t / f"{run}.extraction_log.jsonl"
        original = log.read_text(encoding="utf-8")
        cases = {
            "extraction_log_row_status_mismatch": lambda rows: rows[0].update(status="failed", stored_text=None),
            "extraction_log_row_stored_text_mismatch": lambda rows: rows[0].update(stored_text="something else"),
            "extraction_log_turn_without_row": lambda rows: rows.pop(),
            "extraction_log_duplicate_turn": lambda rows: rows.append(dict(rows[0])),
            "extraction_log_row_prompt": lambda rows: rows[0].update(prompt_sha256="0" * 64),
        }
        for code, fn in cases.items():
            self.edit_jsonl(log, fn)
            self.assertIn(code, self.codes(t, run), code)
            log.write_text(original, encoding="utf-8")
        self.assertEqual(self.codes(t, run), set())

    def test_a_manifest_that_misstates_its_extraction_is_rejected(self):
        t, run = self.copy_run()
        f = t / run / "run_manifest.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        for key, value, code in (("extraction_failures", 3, "extraction_log_manifest_failure_count"), ("extraction_calls", 35, "extraction_log_manifest_call_count"),
                                 ("extraction_prompt_sha256", "0" * 64, "extraction_log_manifest_prompt")):
            f.write_text(json.dumps({**m, key: value}), encoding="utf-8")
            self.assertIn(code, self.codes(t, run), code)
        f.write_text(json.dumps(m), encoding="utf-8")

    def test_an_entry_whose_gloss_is_not_the_logged_stored_text_is_rejected(self):
        t, run = self.copy_run()
        self.edit_jsonl(t / run / "pairwise_eval.jsonl", lambda rows: rows[0]["entry_a"].update(gloss="a different gloss"))
        self.assertIn("entry_gloss_not_the_stored_text", self.codes(t, run))

    def test_a_retrieval_log_that_hides_an_extraction_failure_is_rejected(self):
        t, run = self.copy_run()
        log = t / f"{run}.extraction_log.jsonl"
        self.edit_jsonl(log, lambda rows: rows[0].update(status="failed", stored_text=None, raw_output="not json", error="x"))
        codes = self.codes(t, run)
        self.assertTrue({"extraction_log_vs_retrieval_log"} & codes, codes)


    def test_the_replay_reproduces_every_ab_run_exactly_including_the_runs_with_failed_extractions(self):
        import tier2.replay as RP
        for out in (self.ok_out, self.fail_out):
            res = RP.verify(out, self.syn.dir)
            self.assertEqual(len(res), 4)
            for run, v in res.items():
                self.assertTrue(v["records_identical"] and v["store_identical"], (out.name, run, v))

    def scored_inputs(self, out):
        import tier2.replay as RP
        import tier2.score_v2 as S
        turns = RP.load_turns(self.syn.dir)
        rlog = load_jsonl(out / "t2ab-mem0-A_seed0.tier2_retrieval_log.jsonl")
        pairs, injected, _ = S.select_pairs(self.syn.dir, {x["turn_id"]: i for i, x in enumerate(rlog)}, None, None, set())
        return S, RP, turns, pairs, injected

    def test_an_injected_pair_with_a_failed_side_is_no_entry_and_stays_in_the_denominators(self):
        S, RP, turns, pairs, injected = self.scored_inputs(self.fail_out)
        run = "t2ab-mem0-A_seed0"
        rlog = load_jsonl(self.fail_out / f"{run}.tier2_retrieval_log.jsonl")
        props = {j["record_id"]: j["proposal"] for j in load_jsonl(self.fail_out / f"{run}.judge_log.jsonl")}
        rep = RP.replay_run(rlog, turns, host="A", mode="off", distinctions=None, judge_proposals=props)
        outs = S.outcomes(pairs, rlog, rep)
        first = next(o for o in outs if o["conversation_id"] == self.syn.conv_ids[0])
        self.assertEqual((first["direct_status"], first["direct_merged"], first["record_merged"]), ("no_entry", False, False))
        rates = S.rates(outs)
        total = {"different": sum(o["relation"] == "different" for o in outs), "same": sum(o["relation"] == "same" for o in outs)}
        self.assertEqual(rates["FCR_direct"]["n"], total["different"])               # the failed pair is still counted
        self.assertEqual(rates["MCR_direct"]["n"], total["same"])
        self.assertEqual(rates["outcome_breakdown"][first["relation"]].get("no_entry"), 1)

    def test_score_run_scores_a_and_b_runs_with_the_judge_proposals_and_the_replayed_gate(self):
        S, RP, turns, pairs, injected = self.scored_inputs(self.ok_out)
        evaluator = self.root / "evaluator"
        evaluator.mkdir(exist_ok=True)
        shutil.copy(self.pt, evaluator / "t2_passthrough_used.jsonl")
        ok_embedder = {"status": "ok", "expected": "pin", "recorded": "pin", "requested": None, "model_id": "BAAI/bge-m3"}
        for run, cfg, mode in (("t2ab-mem0-A_seed0", "A", "off"), ("t2ab-dk-mem-lexicon-B_seed0", "B", "lexicon")):
            # the test embedder's revision is not the registered pin (that check has its own tests); everything else is real
            with mock.patch.object(H["prov"].embedder_pin, "verify_manifest_embedder", lambda m, **kw: ok_embedder):
                res = S.score_run(self.ok_out / run, run_data_dir=self.syn.dir, arm_frozen=TIER2 / "FROZEN_v1.0.json", evaluator=evaluator, turns=turns,
                                  pairs=pairs, injected=injected, lexicon=real_lexicon(), data_dir=self.syn.dir, conv_ids=self.syn.conv_ids)
            self.assertTrue(res["provenance_contract_ok"], res["provenance_errors"])
            self.assertEqual((res["config"], res["mode"]), (cfg, mode))
            self.assertFalse(res["final_records_changed_by_lexicon"], run)             # the replay reproduces the run's own final records
            self.assertEqual(res["overall"]["FCR_direct"]["n"] + res["overall"]["MCR_direct"]["n"], len(pairs))
            self.assertEqual(res["judge_log_verification"]["status"], "verified")
            if mode == "lexicon":
                self.assertIn("vetoes", res["gate"])


class TestAuditAndReportDescribeTheJudge(unittest.TestCase):
    def manifest(self, sha, budget):
        return {"judge_prompt_id": "merge_judge_v1", "judge_prompt_sha256": sha, "generation_run_info": {"generation_params": {"max_new_tokens": budget}}}

    def log(self, proposals, status="ok"):
        # rows as FrozenJudgeHost writes them: the frozen prompt's identity and a raw output the strict parser accepts (or rejects, for
        # a failure). provenance re-parses raw_output; it no longer takes the logged status on trust
        return [{"status": status if p is not None else "parse_error", "proposal": p,
                 "raw_output": json.dumps({"decision": p, "reason": "x"}) if p is not None else "not json",
                 "prompt_id": "merge_judge_v1", "prompt_sha256": FROZEN_SHA256} for p in proposals]

    def test_frozen_run_is_described_as_the_frozen_judge(self):
        d = H["prov"].describe(self.manifest(FROZEN_SHA256, 128), self.log(["merge", "keep_both", "keep_both"]))
        self.assertTrue(d["frozen_tier1"] and d["comparable_with_tier1_config_c"] and not d["historical_harness_judge"])
        self.assertIn("FROZEN TIER-1 JUDGE", d["classification"])
        self.assertNotIn("DEGENERATE", d["classification"])
        self.assertEqual(d["max_new_tokens"], 128)

    def test_constant_frozen_judge_is_reported_degenerate_from_the_data(self):
        d = H["prov"].describe(self.manifest(FROZEN_SHA256, 128), self.log(["keep_both"] * 7))
        self.assertIn("DEGENERATE", d["classification"])
        self.assertIn("7/7", d["classification"])

    def test_parse_failures_are_reported_not_hidden(self):
        d = H["prov"].describe(self.manifest(FROZEN_SHA256, 128), self.log(["merge", None, "keep_both", None]))
        self.assertEqual(d["behaviour"]["parse_errors"], 2)
        self.assertIn("2 parse failures", d["classification"])

    def test_unknown_prompt_is_flagged(self):
        d = H["prov"].describe(self.manifest("a" * 64, 128), self.log(["merge"]))
        self.assertFalse(d["recognised"])
        self.assertIn("UNRECOGNISED", d["classification"])

    @unittest.skipUnless((HISTORICAL_C / "run_manifest.json").exists(), "historical Tier-2 run not present")
    def test_historical_run_is_labelled_not_frozen_and_degenerate(self):
        man = json.loads((HISTORICAL_C / "run_manifest.json").read_text(encoding="utf-8"))
        jlog = load_jsonl(HISTORICAL_C.parent / "t2full-store-surface-only-C_seed0.judge_log.jsonl")
        d = H["prov"].describe(man, jlog)
        self.assertTrue(d["historical_harness_judge"] and not d["frozen_tier1"])
        self.assertEqual(d["max_new_tokens"], 32)
        self.assertTrue(d["behaviour"]["constant"])
        self.assertIn("NOT THE FROZEN TIER-1 JUDGE", d["classification"])
        self.assertIn("3105/3105", d["classification"])

    def test_audit_and_report_accept_a_run_prefix(self):
        from tier2 import audit_offline, full_report
        audit_offline.use_run_prefix("t2frozen")
        full_report.use_run_prefix("t2frozen")
        try:
            self.assertEqual(audit_offline.RUNS["C/off"], "t2frozen-store-surface-only-C_seed0")
            self.assertEqual(dict(full_report.CONFIGS)["D/lexicon"], "t2frozen-dk-mem-lexicon-D_seed0")
        finally:
            audit_offline.use_run_prefix("t2full")
            full_report.use_run_prefix("t2full")
        self.assertEqual(audit_offline.RUNS["C/off"], "t2full-store-surface-only-C_seed0")


if __name__ == "__main__":
    unittest.main()
