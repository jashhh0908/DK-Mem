"""GPU-memory probe for a Tier-2 run with SEPARATE models: the QA answerer (Qwen2.5-3B, one GPU) and the config C judge (Qwen2.5-7B,
sharded), plus the bge-m3 metric, all resident at once on 2 x T4 (SYSTEM side: no benchmark data, no labels).

    python -m tier2.judge_validation.memory_probe --cases <judge_cases.jsonl> --out <new dir> [runner model arguments]

Models are built by the runner's own ``build_models`` / ``build_judge_model`` with the runner's argument names, so the placement is
exactly what a Tier-2 run would get. Stages: load QA + metric, load judge, embed, judge the given pairs (both orders, file order, batch 8:
the same calls as the validation run, so the outputs can be compared byte for byte), answer worst-case QA prompts. Per stage it records
torch's allocated / peak-allocated / reserved bytes and the driver's free/total bytes per device.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness import _dkmem  # noqa: F401
from dkmem.backends.llm import GenerationParams
from dkmem.memory.judge import render_judge_prompt

from .. import run_tier2 as R
from ..judge_host import JUDGES_BY_ID, TokenMeter, check_frozen_prompt
from .run_cases import judge_calls

GIB = 1024 ** 3


def snapshot(torch) -> dict:
    out = {}
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        out[str(d)] = {"allocated_gib": round(torch.cuda.memory_allocated(d) / GIB, 3), "peak_allocated_gib": round(torch.cuda.max_memory_allocated(d) / GIB, 3),
                       "reserved_gib": round(torch.cuda.memory_reserved(d) / GIB, 3), "driver_free_gib": round(free / GIB, 3), "total_gib": round(total / GIB, 3)}
    return out


def worst_case_qa_prompts(n: int = 8) -> list:
    """QA prompts at the runner's worst case: top-k (5) memory records, each a merged record of several long turns."""
    turn = ("Meri office wali dost ne kaha ki woh agle mahine apne parivaar ke saath naye ghar mein shift ho rahi hai aur "
            "sab ko invite karegi")
    record = " | ".join([turn] * 6)
    mem = "\n".join(f"- {record}" for _ in range(R.TOP_K))
    return [[{"role": "system", "content": R.QA_SYSTEM}, {"role": "user", "content": R.QA_USER_TEMPLATE.format(memories=mem, question=f"Question {i}?")}]
            for i in range(n)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge-prompt", default="merge_judge_entity_t2_v2", choices=sorted(JUDGES_BY_ID))
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--model-revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--similarity-revision", default="5617a9f61b028005a4858fdac845db406aefb181")
    ap.add_argument("--qa-device", default="cuda:0")
    ap.add_argument("--judge-model-id", required=True)
    ap.add_argument("--judge-model-revision", required=True)
    ap.add_argument("--judge-shard", action="store_true")
    ap.add_argument("--judge-max-memory", default=None)
    ap.add_argument("--judge-device", default="cuda:0")
    a = ap.parse_args(argv)
    import torch
    a.out.mkdir(parents=True, exist_ok=False)
    rep = {"args": vars(a) | {"cases": str(a.cases), "out": str(a.out)}, "stages": {}}
    t0 = time.time()
    rep["stages"]["start"] = snapshot(torch)
    metric, qa = R.build_models(a)
    metric.prepare(["warm up the embedder"])
    metric("warm up", "the embedder")
    rep["stages"]["qa_model_and_metric_loaded"] = snapshot(torch)
    judge = R.build_judge_model(a, qa)
    rep["stages"]["judge_model_loaded"] = snapshot(torch)
    rep["judge"] = {"backbone": judge.backbone, "resolved_revision": getattr(getattr(judge.model, "config", None), "_commit_hash", None),
                    "device_name": judge.device_name, "device_map": R._device_map(judge), "separate_from_qa": judge is not qa}
    rep["qa"] = {"backbone": qa.backbone, "resolved_revision": getattr(getattr(qa.model, "config", None), "_commit_hash", None),
                 "device": str(qa.model.device), "device_name": qa.device_name}
    texts = [f"probe sentence number {i} about a person and a city" for i in range(64)]
    metric.prepare(texts)
    [metric(texts[i], texts[i + 1]) for i in range(0, 63)]
    rep["stages"]["embedding_batch_done"] = snapshot(torch)
    cases = [json.loads(l) for l in a.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    prompt = JUDGES_BY_ID[a.judge_prompt]
    check_frozen_prompt(prompt)
    calls = judge_calls(cases)
    params = GenerationParams(max_new_tokens=R.JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8)
    t1 = time.time()
    raw = judge.generate([render_judge_prompt(x, y, prompt) for _, _, x, y in calls], params)
    rep["judge_seconds"] = round(time.time() - t1, 1)
    rep["stages"]["judge_batches_done"] = snapshot(torch)
    meter = TokenMeter.from_generator(judge)
    with open(a.out / "judge_outputs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for (cid, order, x, y), r in zip(calls, raw):
            f.write(json.dumps({"case_id": cid, "order": order, "raw_output": r, "prompt_id": prompt.prompt_id, "prompt_sha256": prompt.sha256,
                                "max_new_tokens": params.max_new_tokens, "output_tokens": meter.text_tokens(r)}, ensure_ascii=False) + "\n")
    qa_prompts = worst_case_qa_prompts()
    qmeter = TokenMeter.from_generator(qa)
    rep["qa_prompt_tokens_worst_case"] = qmeter.prompt_tokens(qa_prompts[0])
    t2 = time.time()
    qa.generate(qa_prompts, GenerationParams(max_new_tokens=R.QA_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8))
    rep["qa_seconds"] = round(time.time() - t2, 1)
    rep["stages"]["qa_batch_done"] = snapshot(torch)
    rep["seconds_total"] = round(time.time() - t0, 1)
    (a.out / "memory_probe.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "stages"}, indent=1))
    print(json.dumps(rep["stages"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
