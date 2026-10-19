"""Source of paper/figures/pipeline.pdf (run: python paper/figures/pipeline.py). Draws the Tier-2 write path as implemented in
eval/tier2/run_tier2.py: the host proposes first; the gate (dkmem.memory.gate.apply_gate) can only veto a proposed merge."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(-1, 66); ax.axis("off")
BLUE, GREY, GREEN, AMB = "#e3edf7", "#f0f0f0", "#e2f2e2", "#fbf0d4"

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec="#222222", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.3)

def arrow(p, q, text=None, tx=0, ty=0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#222222", shrinkA=0, shrinkB=0))
    if text:
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=6.3, style="italic")

box(33, 56, 34, 9, "New user turn\n(verbatim text = new entry)", GREY)
box(0, 40, 43, 11, "Lexicon matcher (context-free)\nexact word, NFC + casefold\n→ distinction dict of the entry", BLUE)
box(57, 40, 43, 11, "Memory store\nbge-m3 top-5 over earlier raw entries\n→ candidate pairs (new, earlier)", GREY)
box(57, 24, 43, 10, "Host proposes first\nC: LLM identity judge (strict JSON)\nD: bge-m3 cosine > τ", AMB)
box(0, 24, 43, 10, "Compatibility gate (veto only)\ncompatible / incompatible /\nunderdetermined", BLUE)
arrow((40, 56), (21, 51)); arrow((60, 56), (78, 51))
arrow((78, 40), (78, 34), "each pair", 80, 37, ha="left")
arrow((57, 29), (43, 29), "proposal", 50, 31.5)
ax.text(50, 26.5, "merge or\nkeep_both", ha="center", va="center", fontsize=6.3, style="italic")
arrow((21, 40), (21, 34), "distinction dicts of both entries", 23, 37, ha="left")
box(0, 8, 28, 9, "keep_both\n(incompatible, or\nhost kept both)", GREY)
box(32, 8, 28, 9, "link_unresolved\n(underdetermined)", AMB)
box(64, 8, 36, 9, "merge\n(host proposed, compatible)", GREEN)
arrow((10, 24), (14, 17)); arrow((24, 24), (46, 17)); arrow((36, 24), (82, 17), "compatible: host merge passes", 64, 19.6, ha="left")
box(64, -1, 36, 6.5, "Record update: union-find join,\nall entry texts kept (non-destructive)", GREEN)
arrow((82, 8), (82, 5.5))
ax.text(61, 5.0, "keep_both / link_unresolved:\nno join, entries stay separate", ha="right", va="center", fontsize=6.3, style="italic")
fig.savefig(Path(__file__).with_name("pipeline.pdf"), bbox_inches="tight", pad_inches=0.03)
