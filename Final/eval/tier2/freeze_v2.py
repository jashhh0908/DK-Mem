"""Assemble, freeze and verify the Tier-2 v2 benchmark ARM (the directory the runner, the kernel builder and the scorer read).

    python -m tier2.freeze_v2 assemble --build-dir <generate_tier2_v2 build output> --specs <pair specs> --splits <split files...> \\
        --distractors <bank files...> --lexicon <the FINAL lexicon> --out <new arm root, e.g. eval/tier2/v2>
    python -m tier2.freeze_v2 verify   --arm-root <arm root>
    python -m tier2.freeze_v2 freeze   --arm-root <arm root> --confirm-freeze-v2            # writes <arm root>/FROZEN_v2.0.json (once)

Arm root layout (what ``run_tier2 --data-dir <root>/data --frozen <root>/FROZEN_v2.0.json`` and ``build_kernels tier2 --arm v2`` expect)::

    data/tier2_{conversations,injection_metadata,gold,qa_questions,qa_reference}.jsonl, data/tier2_manifest.json     (generate_tier2_v2 output)
    banks/pair_specs_v2.json, banks/splits.jsonl, banks/distractors/<each bank file, verbatim>                       (evaluator-side inputs)
    lexicon/distinction_features.json        the FINAL lexicon, LF line endings: the gate's lexicon, pinned
    ASSEMBLY.json                            provenance: sha256 of every source file, the generator/normaliser hashes, the validation result
    FROZEN_v2.0.json                         only after ``freeze``; pins every file above plus the generator and normaliser code

``assemble`` validates with ``generate_tier2_v2`` against the final lexicon (distractors must not contain any of its terms) and refuses a
non-empty ``--out``. Give it EXACTLY the files (same paths) the build was made with: the generator's manifest records them, and the validator
regenerates the package byte for byte. ``freeze`` refuses without ``--confirm-freeze-v2``, refuses to overwrite, and requires a passing assembly: it is the
single step that makes the benchmark immutable, so it is run once, by the lead, after the lexicon, the specs and the test set are final.
Nothing here runs a model.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_FILES = ("tier2_conversations.jsonl", "tier2_injection_metadata.jsonl", "tier2_gold.jsonl", "tier2_qa_questions.jsonl",
              "tier2_qa_reference.jsonl", "tier2_manifest.json")
BANK_FILES = ("pair_specs_v2.json", "splits.jsonl")          # plus banks/distractors/<name> for every bank file (listed in ASSEMBLY.json)
LEXICON_REL = "lexicon/distinction_features.json"
FROZEN_NAME = "FROZEN_v2.0.json"
CODE_PINS = ("../generate_tier2_v2.py", "../generate_tier2.py", "../validate_tier2.py", "../qa_norm.py")     # relative to the arm root (= the freeze file's dir)
STATEMENT = ("Frozen Tier-2 benchmark v2: a controlled injection benchmark of constructed sentences in Hindi, Marathi and Gujarati; it makes no claim "
             "about the natural frequency of these distinctions in conversation. Labels are those of the pair specs pinned here; the review status of "
             "the specs and the lexicon is recorded in the review records pinned with the evaluation-plan freeze. Any change to a pinned file creates "
             "a new version and must not overwrite these hashes.")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _lf_bytes(p: Path) -> bytes:
    return Path(p).read_bytes().replace(b"\r\n", b"\n")


class FreezeError(SystemExit):
    pass


def assemble(build_dir: Path, specs: Path, splits: list[Path], distractors: list[Path], lexicon: Path, out: Path, *, validate: bool = True) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FreezeError(f"{out} exists and is not empty; use a new directory (an arm root is never overwritten)")
    missing = [f for f in DATA_FILES if not (Path(build_dir) / f).is_file()]
    if missing:
        raise FreezeError(f"{build_dir} is not a generate_tier2_v2 build output: missing {missing}")
    if len(distractors) < 1:
        raise FreezeError("at least one distractor bank file is needed")
    result = {"ok": None, "failures": None}
    if validate:
        from . import generate_tier2_v2 as V
        argv = ["validate", "--data-dir", str(build_dir), "--specs", str(specs), "--splits", *map(str, splits), "--distractors", *map(str, distractors),
                "--lexicon", str(lexicon)]
        manifest = json.loads((Path(build_dir) / "tier2_manifest.json").read_text(encoding="utf-8"))
        argv += ["--version", str(manifest["version"]), "--seed", str(manifest["seed"])]
        rc = V.main(argv)
        result = {"ok": rc == 0, "failures": None if rc == 0 else "see the validator output above"}
        if rc != 0:
            raise FreezeError("the build does not validate against the final lexicon, specs, splits and banks: nothing was assembled")
    (out / "data").mkdir(parents=True)
    (out / "banks" / "distractors").mkdir(parents=True)
    (out / "lexicon").mkdir()
    for f in DATA_FILES:
        shutil.copyfile(Path(build_dir) / f, out / "data" / f)
    shutil.copyfile(specs, out / "banks" / "pair_specs_v2.json")
    (out / "banks" / "splits.jsonl").write_bytes(b"".join(_lf_bytes(p) for p in splits))
    names = [Path(p).name for p in distractors]
    if len(set(names)) != len(names):
        raise FreezeError(f"distractor bank file names must be distinct: {names}")
    for p in distractors:
        shutil.copyfile(p, out / "banks" / "distractors" / Path(p).name)
    (out / LEXICON_REL).write_bytes(_lf_bytes(lexicon))
    manifest = json.loads((out / "data" / "tier2_manifest.json").read_text(encoding="utf-8"))
    assembly = {"assembled_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "validation": result,
                "sources_sha256": {"pair_specs": sha(specs), "splits": [sha(p) for p in splits], "distractor_banks": [sha(p) for p in distractors],
                                   "lexicon_raw": sha(lexicon), "lexicon_lf": hashlib.sha256(_lf_bytes(lexicon)).hexdigest()},
                "bank_files": names, "generator_sha256": sha(HERE / "generate_tier2_v2.py"), "qa_norm_sha256": sha(HERE / "qa_norm.py"),
                "benchmark_manifest_counts": manifest.get("counts"), "benchmark_version": manifest.get("version"), "seed": manifest.get("seed")}
    (out / "ASSEMBLY.json").write_text(json.dumps(assembly, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return assembly


def pin_table_keys(root: Path) -> list[str]:
    """Every pinned path of an assembled arm root (relative to the root); the distractor banks come from ASSEMBLY.json."""
    asm = json.loads((Path(root) / "ASSEMBLY.json").read_text(encoding="utf-8"))
    return ([f"data/{f}" for f in DATA_FILES] + [f"banks/{f}" for f in BANK_FILES] + [f"banks/distractors/{n}" for n in asm["bank_files"]]
            + [LEXICON_REL, "ASSEMBLY.json", *CODE_PINS])


def pin_table(root: Path) -> dict[str, str]:
    root = Path(root)
    keys = pin_table_keys(root)
    out = {}
    for k in keys:
        p = root / k
        if not p.is_file():
            raise FreezeError(f"cannot pin {k}: {p} is missing")
        out[k] = sha(p)
    return out


def freeze(root: Path, *, confirm: bool, version: str = "2.0") -> Path:
    root = Path(root)
    if not confirm:
        raise FreezeError("refusing to freeze without --confirm-freeze-v2: freezing makes the benchmark immutable (run once, by the lead, after the "
                          "lexicon, the specs and the test set are final)")
    target = root / FROZEN_NAME
    if target.exists():
        raise FreezeError(f"{target} already exists; a frozen benchmark is never overwritten (create v2.1 in a new arm root)")
    asm_p = root / "ASSEMBLY.json"
    if not asm_p.is_file():
        raise FreezeError(f"{root} is not an assembled arm root (no ASSEMBLY.json)")
    asm = json.loads(asm_p.read_text(encoding="utf-8"))
    if (asm.get("validation") or {}).get("ok") is not True:
        raise FreezeError("the assembly was not validated against the final lexicon (assemble with validation); refusing to freeze")
    if asm["sources_sha256"]["lexicon_lf"] != sha(root / LEXICON_REL):
        raise FreezeError("lexicon/distinction_features.json differs from the lexicon that was validated at assembly")
    manifest = json.loads((root / "data" / "tier2_manifest.json").read_text(encoding="utf-8"))
    for f in ("tier2_conversations.jsonl", "tier2_injection_metadata.jsonl", "tier2_gold.jsonl", "tier2_qa_questions.jsonl", "tier2_qa_reference.jsonl"):
        if sha(root / "data" / f) != manifest["files"][f]["sha256"]:
            raise FreezeError(f"data/{f} does not match the generator manifest")
    pins = pin_table(root)
    for rel in pins:
        if (root / rel).read_bytes().count(b"\r"):
            raise FreezeError(f"{rel} contains carriage returns: pinned files must use LF (their hash would change on another checkout)")
    doc = {"status": "FROZEN", "benchmark": "DK-Mem Tier-2 controlled multi-session injection benchmark (v2)", "version": version,
           "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "statement": STATEMENT,
           "counts": manifest.get("counts"), "seed": manifest.get("seed"), "sha256": pins}
    target.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    problems = verify(root)
    if problems:
        target.unlink()
        raise FreezeError(f"the freeze file did not verify and was removed: {problems}")
    return target


def verify(root: Path) -> list[str]:
    """Problems of an arm root against its freeze file (empty = every pin matches)."""
    root = Path(root)
    f = root / FROZEN_NAME
    if not f.is_file():
        return [f"{f} does not exist (not frozen)"]
    fz = json.loads(f.read_text(encoding="utf-8"))
    problems = []
    if fz.get("status") != "FROZEN":
        problems.append("status is not FROZEN")
    for rel, want in fz["sha256"].items():
        p = root / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif sha(p) != want:
            problems.append(f"{rel}: changed")
    expected = set(pin_table_keys(root))
    if set(fz["sha256"]) != expected:
        problems.append(f"the freeze pins {sorted(set(fz['sha256']) ^ expected)} differently from the arm layout")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("assemble")
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--splits", type=Path, nargs="+", required=True)
    p.add_argument("--distractors", type=Path, nargs="+", required=True)
    p.add_argument("--lexicon", type=Path, required=True, help="the FINAL lexicon (the gate's lexicon for every v2 run)")
    p.add_argument("--out", type=Path, required=True)
    for name in ("verify", "freeze"):
        q = sub.add_parser(name)
        q.add_argument("--arm-root", type=Path, required=True)
        if name == "freeze":
            q.add_argument("--confirm-freeze-v2", action="store_true")
            q.add_argument("--version", default="2.0")
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        asm = assemble(a.build_dir, a.specs, a.splits, a.distractors, a.lexicon, a.out)
        print("assembled", a.out, "| validation ok:", asm["validation"]["ok"], "| lexicon (LF) sha256", asm["sources_sha256"]["lexicon_lf"][:16])
        return 0
    if a.cmd == "verify":
        problems = verify(a.arm_root)
        print("OK: every pin matches" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    path = freeze(a.arm_root, confirm=a.confirm_freeze_v2, version=a.version)
    print("froze", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
