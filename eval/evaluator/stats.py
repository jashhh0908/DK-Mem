"""Statistics on top of the Tier 1 pair rows: paired-bootstrap CIs and Cohen's kappa.

Both operate on the per-gold-pair rows from `metrics.pair_details` / `metrics.evaluate_run`
(keys used: relation, merged). Metric values always come from `metrics.fcr` / `metrics.mcr`.

Bootstrap (percentile, deterministic)
  * Unit of resampling = one gold evaluation pair (a whole row, with all its records/outcome).
    Records are never resampled individually.
  * Relation-stratified: FCR resamples the gold-'different' pairs of the cell, MCR the gold-'same'
    pairs, each with replacement and the same count, so denominators equal the point estimate's
    and the gold relation of every pair is preserved.
  * CI = 2.5th / 97.5th percentiles (linear interpolation) of the replicate metric values.
  * Determinism: replicate k of a cell uses random.Random(seed_for(cell label, metric)), where the
    seed is a sha256 of "BOOTSTRAP_SEED|label|metric". Same data + same label -> same CI.
  * An empty stratum (n = 0) gives no CI (n/a), matching the metric convention. A cell whose
    point estimate is 0 or 1 has a degenerate CI [p, p]; that is a property of the method, not
    evidence of certainty.
  * This is uncertainty over the evaluation PAIRS only. It says nothing about seed variance.

Cohen's kappa
  * Two raters over every gold pair: gold = 'same' (should merge) vs 'different'; predicted =
    merged vs not merged. Not-compared and no-entry pairs are 'not merged' FOR THIS AGREEMENT
    CALCULATION ONLY (same convention as the contract's FCR/MCR); no pair is dropped.
  * kappa = (po - pe) / (1 - pe). If pe == 1 (both raters constant and identical) kappa is
    undefined (0/0) and reported as None with a reason, never as 1 or 0.
  * Note: this is agreement between the SYSTEM and the gold labels, not the inter-annotator kappa
    the research plan wants on Team B's annotation.
"""
from __future__ import annotations

import hashlib
import random
from typing import Callable

from .metrics import fcr, mcr

BOOTSTRAP_SEED = 20261008
N_BOOT = 10_000
ALPHA = 0.05
METHOD = "percentile bootstrap over gold pairs, relation-stratified, seeded"

_METRICS: dict[str, tuple[Callable, str]] = {"fcr": (fcr, "different"), "mcr": (mcr, "same")}


def seed_for(label: str, metric: str, base_seed: int = BOOTSTRAP_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}|{metric}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(rows: list[dict], metric: str, *, label: str, n_boot: int = N_BOOT,
                 base_seed: int = BOOTSTRAP_SEED, alpha: float = ALPHA) -> dict:
    """Point estimate and percentile CI for 'fcr' or 'mcr' over `rows` (pair rows of one cell)."""
    fn, relation = _METRICS[metric]
    stratum = [r for r in rows if r["relation"] == relation]
    point = fn(rows)
    out = {"metric": metric, "n": len(stratum), "point": point["rate"], "lo": None, "hi": None,
           "n_boot": 0, "seed": None, "method": METHOD, "confidence": 1 - alpha}
    if not stratum:
        return out                      # n = 0 -> n/a, no interval
    seed = seed_for(label, metric, base_seed)
    rng = random.Random(seed)
    values = sorted(fn(rng.choices(stratum, k=len(stratum)))["rate"] for _ in range(n_boot))
    out.update(lo=quantile(values, alpha / 2), hi=quantile(values, 1 - alpha / 2), n_boot=n_boot, seed=seed)
    return out


def cohens_kappa(rows: list[dict]) -> dict:
    """Cohen's kappa between gold (same=1) and predicted (merged=1) over all `rows`."""
    tp = sum(r["relation"] == "same" and r["merged"] for r in rows)
    fn_ = sum(r["relation"] == "same" and not r["merged"] for r in rows)
    fp = sum(r["relation"] == "different" and r["merged"] for r in rows)
    tn = sum(r["relation"] == "different" and not r["merged"] for r in rows)
    n = tp + fn_ + fp + tn
    out = {"n": n, "confusion": {"gold_same_merged": tp, "gold_same_not_merged": fn_,
                                  "gold_different_merged": fp, "gold_different_not_merged": tn},
           "observed_agreement": None, "expected_agreement": None, "kappa": None, "undefined_reason": None,
           "note": "no-entry / not-compared pairs counted as not merged for this agreement calculation only"}
    if n == 0:
        out["undefined_reason"] = "no pairs"
        return out
    po = (tp + tn) / n
    gold_pos, pred_pos = (tp + fn_) / n, (tp + fp) / n
    pe = gold_pos * pred_pos + (1 - gold_pos) * (1 - pred_pos)
    out.update(observed_agreement=po, expected_agreement=pe)
    if pe == 1:
        out["undefined_reason"] = "expected agreement is 1 (both raters constant): kappa is 0/0"
    else:
        out["kappa"] = (po - pe) / (1 - pe)
    return out


def mean_sd(values: list[float | None]) -> dict:
    """Mean and sample SD (n-1) across seeds; None values (n/a cells) are excluded."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "sd": None}
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else None
    return {"n_seeds": n, "mean": mean, "sd": sd}
