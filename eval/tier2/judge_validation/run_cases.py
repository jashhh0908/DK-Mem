"""Run a frozen Tier-2 identity judge (--prompt; default v1) on the judge-validation pairs (SYSTEM side: texts only, never the labels).

    python -m tier2.judge_validation.run_cases --cases <judge_cases.jsonl> --out <new dir>

``judge_cases.jsonl`` holds ``case_id``, ``text_a``, ``text_b`` and nothing else (the builder strips every label). Every case is judged
in both orders (``ab``: Memory A = text_a; ``ba``: swapped) with the frozen prompt, the frozen strict parser's input format and the
Tier-1 decoding budget. Raw outputs are written as generated; scoring happens evaluator-side (``score.py``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness import _dkmem  # noqa: F401  (puts dkmem on sys.path)
from dkmem.backends.llm import GenerationParams
from dkmem.memory.judge import render_judge_prompt

from ..frozen_judge import IDENTITY_JUDGES
from ..judge_entity import MERGE_JUDGE_ENTITY_T2_V1
from ..judge_host import FROZEN_JUDGE_MAX_NEW_TOKENS, JUDGES_BY_ID, TokenMeter, check_frozen_prompt

CASE_KEYS = {"case_id", "text_a", "text_b"}


def judge_calls(cases: list[dict]) -> list[tuple[str, str, str, str]]:
    """``(case_id, order, memory A, memory B)`` for both orders of every case, in file order."""
    out = []
    for c in cases:
        if set(c) != CASE_KEYS:
            raise ValueError(f"judge input rows must hold exactly {sorted(CASE_KEYS)} (no labels), got {sorted(c)}")
        out += [(c["case_id"], "ab", c["text_a"], c["text_b"]), (c["case_id"], "ba", c["text_b"], c["text_a"])]
    return out


def run(cases: list[dict], generator, params: GenerationParams, meter=None, prompt=MERGE_JUDGE_ENTITY_T2_V1) -> list[dict]:
    check_frozen_prompt(prompt)
    calls = judge_calls(cases)
    messages = [render_judge_prompt(a, b, prompt) for _, _, a, b in calls]
    raw = generator.generate(messages, params)
    if not isinstance(raw, list) or len(raw) != len(calls):
        raise RuntimeError("generator returned the wrong number of outputs")
    return [{"case_id": cid, "order": order, "text_a": a, "text_b": b, "raw_output": r, "prompt_id": prompt.prompt_id,
             "prompt_sha256": prompt.sha256, "max_new_tokens": params.max_new_tokens,
             "prompt_tokens": meter.prompt_tokens(m) if meter else None,
             "output_tokens": meter.text_tokens(r) if (meter and isinstance(r, str)) else None}
            for (cid, order, a, b), m, r in zip(calls, messages, raw)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", choices=sorted(IDENTITY_JUDGES), default=MERGE_JUDGE_ENTITY_T2_V1.prompt_id)
    ap.add_argument("--shard", action="store_true", help="split the weights over the visible GPUs (HFGenerator.load_sharded; Qwen2.5-7B on 2 x T4)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--model-revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    a = ap.parse_args(argv)
    from dkmem.backends.llm import HFGenerator, ModelConfig
    cases = [json.loads(l) for l in a.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    cfg = ModelConfig(model_id=a.model_id, revision=a.model_revision)
    gen = HFGenerator.load_sharded(cfg) if a.shard else HFGenerator.load(cfg)
    params = GenerationParams(max_new_tokens=FROZEN_JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=a.seed, batch_size=8)
    meter = TokenMeter.from_generator(gen)
    t1 = time.time()
    prompt = JUDGES_BY_ID[a.prompt]
    rows = run(cases, gen, params, meter, prompt)
    t2 = time.time()
    with open(a.out / "judge_outputs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"prompt_id": prompt.prompt_id, "prompt_sha256": prompt.sha256, "n_cases": len(cases),
            "n_calls": len(rows), "load_seconds": round(t1 - t0, 1), "judge_seconds": round(t2 - t1, 1),
            "generation_run_info": gen.run_info(params), "tokenizer": meter.describe() if meter else None,
            "resolved_model_revision": getattr(getattr(gen.model, "config", None), "_commit_hash", None),
            "sharded": bool(a.shard), "device_name": getattr(gen, "device_name", None),
            "device_map": {str(k): str(v) for k, v in (getattr(gen.model, "hf_device_map", None) or {}).items()} or None}
    (a.out / "run_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "generation_run_info"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
