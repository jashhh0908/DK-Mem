"""Batch-composition diagnostic for the frozen Config C judge (SYSTEM side: pair texts only, no labels, no injected/distractor marks).

    python -m tier2.judge_validation.batch_diagnostic --calls <calls.jsonl> --out <new dir> [judge model arguments]

``calls.jsonl`` rows: ``call_id``, ``group`` (the later entry: one runner judge call per group), ``text_a``, ``text_b``, in the runner's
order. The frozen prompt is rendered exactly as the runner renders it and decoded with the frozen settings in three batchings:

  insitu  one generate() per group, as run_tier2's FrozenJudgeHost does (<= top-k prompts, batch_size 8)
  b1      every prompt alone (batch_size 1)
  b8      all prompts in order, batch_size 8

Raw outputs are written per mode; comparison happens evaluator-side. It changes no prompt, criterion or decision of any run.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import groupby
from pathlib import Path

from harness import _dkmem  # noqa: F401
from dkmem.backends.llm import GenerationParams
from dkmem.memory.judge import render_judge_prompt

from .. import run_tier2 as R
from ..judge_host import JUDGES_BY_ID, check_frozen_prompt

KEYS = {"call_id", "group", "text_a", "text_b"}


def run(calls: list[dict], judge, prompt) -> dict:
    check_frozen_prompt(prompt)
    for c in calls:
        if set(c) != KEYS:
            raise ValueError(f"call rows must hold exactly {sorted(KEYS)}")
    msgs = [render_judge_prompt(c["text_a"], c["text_b"], prompt) for c in calls]
    p8 = GenerationParams(max_new_tokens=R.JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=8)
    p1 = GenerationParams(max_new_tokens=R.JUDGE_MAX_NEW_TOKENS, do_sample=False, seed=0, batch_size=1)
    out, timing = {}, {}
    t = time.time()
    insitu = []
    for _, grp in groupby(range(len(calls)), key=lambda i: calls[i]["group"]):
        idx = list(grp)
        insitu += judge.generate([msgs[i] for i in idx], p8)
    out["insitu"], timing["insitu"] = insitu, round(time.time() - t, 1)
    t = time.time()
    out["b1"], timing["b1"] = judge.generate(msgs, p1), round(time.time() - t, 1)
    t = time.time()
    out["b8"], timing["b8"] = judge.generate(msgs, p8), round(time.time() - t, 1)
    return {"outputs": out, "seconds": timing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge-prompt", default="merge_judge_entity_t2_v2", choices=sorted(JUDGES_BY_ID))
    ap.add_argument("--judge-model-id", required=True)
    ap.add_argument("--judge-model-revision", required=True)
    ap.add_argument("--judge-shard", action="store_true")
    ap.add_argument("--judge-max-memory", default=None)
    ap.add_argument("--judge-device", default="cuda:0")
    a = ap.parse_args(argv)
    calls = [json.loads(l) for l in a.calls.read_text(encoding="utf-8").splitlines() if l.strip()]
    a.out.mkdir(parents=True, exist_ok=False)
    judge = R.build_judge_model(a, None)
    prompt = JUDGES_BY_ID[a.judge_prompt]
    res = run(calls, judge, prompt)
    with open(a.out / "batch_outputs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(calls):
            f.write(json.dumps({"call_id": c["call_id"], **{m: res["outputs"][m][i] for m in res["outputs"]}}, ensure_ascii=False) + "\n")
    info = {"prompt_id": prompt.prompt_id, "prompt_sha256": prompt.sha256, "n_calls": len(calls), "seconds": res["seconds"],
            "judge_backbone": judge.backbone, "resolved_revision": getattr(getattr(judge.model, "config", None), "_commit_hash", None),
            "device_name": judge.device_name, "device_map": R._device_map(judge)}
    (a.out / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "device_map"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
