"""Read-only access to the dkmem package of THIS repository (current HEAD, NOT the legacy snapshot).

ADAPTED from the harness shipped with the Tier-2 archive (source sha256 bc0621842ac4a059...): the original resolved
``<eval parent>/ipdV2``. Inside this repository ``eval/`` sits next to ``dkmem/``, so the dkmem root is the parent of
``eval/``. Nothing else changed; every other vendored harness file is byte-identical (see VENDORED.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]            # <repo>/eval/harness/_dkmem.py -> <repo>
IPDV2 = _ROOT / "ipdV2" if (_ROOT / "ipdV2" / "dkmem").is_dir() else _ROOT
LEXICON_PATH = IPDV2 / "dkmem" / "memory" / "distinction_features.json"
SCHEMA_DIR = IPDV2 / "dkmem"
INPUT_PATH = IPDV2 / "dkmem" / "data" / "tier1" / "eval_input" / "teamA_tier1_v2.jsonl"

if str(IPDV2) not in sys.path:
    sys.path.insert(0, str(IPDV2))

import dkmem  # noqa: E402,F401  (fail early if the wrong tree was picked up)

assert Path(dkmem.__file__).resolve().is_relative_to(IPDV2.resolve()), f"unexpected dkmem at {dkmem.__file__}"
