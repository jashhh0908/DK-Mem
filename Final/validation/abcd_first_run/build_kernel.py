"""Build the Kaggle notebook for a real A-D run: the whole test suite as a preflight, then
the exact `python -m dkmem.pipeline.cli ...` command (see DKMEM_FIRST_RUN.md).

The notebook embeds the `dkmem` package (code, schemas, lexicon, the sanctioned Tier 1
input) and the tests. It does not run anything by itself here; `kaggle kernels push`
saves the version and starts it.

Example (the first real run, Qwen2.5-3B, all 173 records, seed 0):

    python build_kernel.py --user jashnikumbhe --out kernel_3b --model-id Qwen/Qwen2.5-3B-Instruct \\
        --seed 0 --tau-d 0.80 0.85 0.90 --slug dk-mem-abcd-3b-s0

Add `--shard` for Qwen2.5-7B-Instruct (weights split over both T4s) and `--limit N` for a
smoke test. All arguments after the model are forwarded to the CLI as shown in the notebook.

Baselines (DKMEM_BASELINES.md): `--baseline mem0` or `--baseline flat` runs `python -m dkmem.baselines.cli` instead
(no `--tau-d`, `--configs` or `--modes`); `--judge-prompt merge_judge_informed_v1` with `--modes off` makes the A-D
command the prompt-informed-judge baseline. `--embedding-revision` pins bge-m3 for Config D / the baselines.

Packaging. The sources travel as a base64 zip (`PAYLOAD`), not as a JSON string: Kaggle answers a plain-text notebook
of about 1.5 MB with HTTP 400 (the 1,566,345-byte plain-text build of this notebook was rejected; the same files as a
471,724-byte zip payload were accepted and ran). The builder refuses to write a notebook that

* embeds anything outside `dkmem/` and `tests/`, a file whose name contains `answer_key`, a gold-label record (an
  `ep_` pair id next to a `relation`) or an original `t1_NNNN` pair id (Team B's blinding rule);
* lacks a file the run needs, has CRLF text, or whose embedded Tier 1 input does not hash to the sha256 pinned in the
  embedded `dkmem/tier1/io.py`;
* does not decode back, byte for byte, to the files it was built from (zip CRC and content round trip);
* is larger than `--max-notebook-bytes` (default 1,000,000).

The notebook repeats the integrity check on Kaggle before it writes a single file: the payload's sha256, the zip CRCs,
the file list and every file's sha256 must match the manifest embedded next to it, otherwise the first cell fails.
The output is deterministic: the same tree and arguments give the same notebook bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOTEBOOK_NAME = "dkmem_abcd_run.ipynb"
MAX_NOTEBOOK_BYTES = 1_000_000  # below the 1,566,345 bytes Kaggle rejected, above the 854,388 it had accepted
ZIP_DATE = (2026, 10, 10, 0, 0, 0)  # fixed, so the same files give the same zip bytes
EMBED_SUFFIXES = (".py", ".json", ".jsonl", ".md", ".txt")  # .txt: the vendored, hash-pinned Mem0 sources
REQUIRED_FILES = (
    "dkmem/pipeline/cli.py", "dkmem/pipeline/runner.py", "dkmem/baselines/cli.py", "dkmem/memory/judge.py",
    "dkmem/memory/distinction_features.json", "dkmem/pairwise_eval.schema.json", "dkmem/run_manifest.schema.json",
    "dkmem/tier1/io.py", "dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl", "tests/test_pipeline.py",
)
INPUT_FILE = "dkmem/data/tier1/eval_input/teamA_tier1_v2.jsonl"

# The same patterns tests/test_key_protection.py uses: a gold record is a real eval pair id next to a relation label.
_REAL_EP_ID = re.compile(r"ep_[0-9a-f]{12}")
_RELATION = re.compile(r"\"(?:gold_)?relation\"\s*:\s*\"(?:same|different)\"")
_ORIGINAL_PAIR_ID = re.compile(r"\bt1_\d{4}\b")


class BuildError(Exception):
    """The notebook would be unsafe, incomplete, corrupt or too large; nothing is written."""


# --- collecting and checking the payload -------------------------------------------------------------------------


def collect_files(repo: str = REPO) -> dict[str, str]:
    """``{relative posix path: LF text}`` of every embeddable file of ``dkmem/`` and ``tests/``."""
    embedded: dict[str, str] = {}
    for top in ("dkmem", "tests"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(repo, top)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in sorted(filenames):
                if fn.endswith(EMBED_SUFFIXES):
                    path = os.path.join(dirpath, fn)
                    rel = os.path.relpath(path, repo).replace(os.sep, "/")
                    with open(path, "rb") as f:
                        embedded[rel] = f.read().replace(b"\r\n", b"\n").decode("utf-8")
    return embedded


def check_payload(files: dict[str, str]) -> None:
    """Refuse a payload that breaks the blinding rule or cannot run the notebook (see the module docstring)."""
    if not files:
        raise BuildError("nothing to embed")
    outside = [n for n in files if n.split("/", 1)[0] not in ("dkmem", "tests") or ".." in n.split("/") or n.startswith("/")]
    if outside:
        raise BuildError(f"files outside dkmem/ and tests/ would be embedded: {outside[:3]}")
    keyed = [n for n in files if "answer_key" in n.lower()]
    if keyed:
        raise BuildError(f"answer-key file(s) would be embedded (Team B's blinding rule): {keyed}")
    leaks = []
    for name, text in files.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            if (_REAL_EP_ID.search(line) and _RELATION.search(line)) or _ORIGINAL_PAIR_ID.search(line):
                leaks.append(f"{name}:{lineno}")
    if leaks:
        raise BuildError(f"gold labels or original pair ids would be embedded (blinding rule): {leaks[:5]}")
    crlf = [n for n, t in files.items() if "\r" in t]
    if crlf:
        raise BuildError(f"files with carriage returns would change their hashes on Kaggle: {crlf[:3]}")
    missing = [n for n in REQUIRED_FILES if n not in files]
    if missing:
        raise BuildError(f"files the run needs are missing from the payload: {missing}")
    pinned = re.search(r'TIER1_INPUT_SHA256\s*=\s*"([0-9a-f]{64})"', files["dkmem/tier1/io.py"])
    if pinned is None:
        raise BuildError("cannot find TIER1_INPUT_SHA256 in the embedded dkmem/tier1/io.py")
    actual = hashlib.sha256(files[INPUT_FILE].encode("utf-8")).hexdigest()
    if actual != pinned.group(1):
        raise BuildError(f"the embedded Tier 1 input hashes to {actual}, not the pinned {pinned.group(1)}")


def manifest_of(files: dict[str, str]) -> dict[str, str]:
    return {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in sorted(files.items())}


def tree_sha256(manifest: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def make_payload(files: dict[str, str]) -> str:
    """Base64 of a deterministic zip of ``files``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, text in sorted(files.items()):
            z.writestr(zipfile.ZipInfo(rel, ZIP_DATE), text.encode("utf-8"), zipfile.ZIP_DEFLATED)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def verify_payload(payload: str, files: dict[str, str]) -> None:
    """The payload must decode to exactly ``files`` (CRCs, names and content)."""
    try:
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload, validate=True))) as z:
            bad = z.testzip()
            if bad is not None:
                raise BuildError(f"zip member {bad!r} fails its CRC")
            names = z.namelist()
            if sorted(names) != sorted(files) or len(set(names)) != len(names):
                raise BuildError("the zip's file list differs from the files it was built from")
            for n in names:
                if z.read(n).decode("utf-8") != files[n]:
                    raise BuildError(f"zip member {n!r} differs from its source")
    except (zipfile.BadZipFile, ValueError) as e:
        raise BuildError(f"the payload does not decode: {e}") from e


# --- the notebook --------------------------------------------------------------------------------------------------

# Cell 1 runs on Kaggle. @@NAME@@ placeholders are replaced textually (the cell is full of braces).
_UNPACK_CELL = '''
import base64, hashlib, io, json, os, subprocess, sys, time, zipfile
GPU_LIST = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(GPU_LIST)
SRC = "/kaggle/working/src"
PAYLOAD = "@@PAYLOAD@@"  # base64 of a zip of dkmem/ and tests/ (Kaggle rejects a ~1.5 MB plain-text notebook)
PAYLOAD_SHA256 = "@@PAYLOAD_SHA256@@"
MANIFEST = json.loads(@@MANIFEST@@)  # relative path -> sha256 of the file's utf-8 text
assert hashlib.sha256(PAYLOAD.encode("ascii")).hexdigest() == PAYLOAD_SHA256, "payload changed in transit"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as _z:
    assert _z.testzip() is None, "a zip member fails its CRC"
    assert sorted(_z.namelist()) == sorted(MANIFEST), "the payload's files differ from the manifest"
    FILES = {n: _z.read(n).decode("utf-8") for n in _z.namelist()}
for rel, text in FILES.items():
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == MANIFEST[rel], "file differs from the manifest: " + rel
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
print(len(FILES), "files written; payload verified, tree sha256", hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest())
try:
    import jsonschema
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)
'''


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def build_cli_args(args: argparse.Namespace) -> tuple[str, list[str]]:
    """``(module, argv)`` of the command the notebook runs. Unchanged from the frozen builder."""
    if args.baseline is not None:
        module = "dkmem.baselines.cli"
        cli_args = [args.baseline, "--seed", str(args.seed), "--out", "/kaggle/working/runs"]
        if args.baseline == "mem0":
            cli_args += ["--model-id", args.model_id]
            if args.revision:
                cli_args += ["--revision", args.revision]
            if args.shard:
                cli_args.append("--shard")
    else:
        module = "dkmem.pipeline.cli"
        cli_args = ["--configs", *args.configs, "--modes", *args.modes, "--model-id", args.model_id,
                    "--seed", str(args.seed), "--out", "/kaggle/working/runs"]
        if "D" in args.configs:
            cli_args += ["--tau-d", *args.tau_d]
        if args.revision:
            cli_args += ["--revision", args.revision]
        if args.shard:
            cli_args.append("--shard")
        if args.judge_prompt:
            cli_args += ["--judge-prompt", args.judge_prompt]
    if args.limit is not None:
        cli_args += ["--limit", str(args.limit)]
    if args.embedding_revision:
        cli_args += ["--embedding-revision", args.embedding_revision]
    return module, cli_args


def build_notebook(args: argparse.Namespace, files: dict[str, str]) -> tuple[dict, str, list[str], str]:
    """``(notebook dict, cli module, cli args, payload)``."""
    module, cli_args = build_cli_args(args)
    payload = make_payload(files)
    manifest = manifest_of(files)
    unpack = (_UNPACK_CELL.replace("@@PAYLOAD@@", payload)
              .replace("@@PAYLOAD_SHA256@@", hashlib.sha256(payload.encode("ascii")).hexdigest())
              .replace("@@MANIFEST@@", json.dumps(json.dumps(manifest, sort_keys=True))))
    cells = [
        _md(f"# DK-Mem {'baseline ' + args.baseline if args.baseline else 'stage attribution, Configs ' + ' '.join(args.configs)}: "
            f"{args.model_id}, seed {args.seed}"),
        _code(unpack),
        _code("""
# preflight: the whole unit-test suite must pass on the Kaggle image before the run starts
p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=SRC,
                   capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC})
print(p.stderr[-1500:])
assert p.returncode == 0, "unit tests failed"
"""),
        _code(f"""
ARGS = {json.dumps(cli_args)}
print("python -m {module}", " ".join(ARGS), flush=True)
t0 = time.time()
proc = subprocess.Popen([sys.executable, "-m", "{module}", *ARGS], cwd=SRC, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, env={{**os.environ, "PYTHONPATH": SRC}})
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print("exit code", rc, "| seconds", round(time.time() - t0, 1))
assert rc == 0
"""),
        _code("""
import glob
for path in sorted(glob.glob("/kaggle/working/runs/*/summary.json")):
    s = json.load(open(path, encoding="utf-8"))
    print(os.path.basename(os.path.dirname(path)), json.dumps({k: s[k] for k in s if k not in ("note",)}, ensure_ascii=False)[:1200])
json.dump({"nvidia_smi_L": GPU_LIST, "cli_args": ARGS}, open("/kaggle/working/run_environment.json", "w"), indent=2)
"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return nb, module, cli_args, payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default=None, help="pin the LLM weights to this commit hash")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-d", nargs="+", default=None)
    ap.add_argument("--baseline", choices=("mem0", "flat"), default=None,
                    help="run dkmem.baselines.cli instead of the A-D pipeline")
    ap.add_argument("--judge-prompt", default=None, help="merge judge of Configs A-C (e.g. merge_judge_informed_v1)")
    ap.add_argument("--embedding-revision", default=None)
    ap.add_argument("--configs", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--modes", nargs="+", default=["off", "lexicon", "lexicon+llm"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-notebook-bytes", type=int, default=MAX_NOTEBOOK_BYTES,
                    help="refuse a notebook larger than this (Kaggle rejects about 1.5 MB of plain text)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.baseline is None and ("D" in args.configs) and not args.tau_d:
        ap.error("--tau-d is required when Config D is run")
    if args.baseline is not None and (args.tau_d or args.judge_prompt):
        ap.error("--tau-d / --judge-prompt belong to the A-D pipeline, not to --baseline")

    try:
        files = collect_files()
        check_payload(files)
        nb, module, cli_args, payload = build_notebook(args, files)
        verify_payload(payload, files)
        text = json.dumps(nb, indent=1, ensure_ascii=False)
        size = len(text.encode("utf-8"))
        if size > args.max_notebook_bytes:
            raise BuildError(f"the notebook is {size:,} bytes, over the {args.max_notebook_bytes:,}-byte limit "
                             "(Kaggle rejects about 1.5 MB with HTTP 400)")
    except BuildError as e:
        print(f"build_kernel: refusing to write the notebook: {e}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, NOTEBOOK_NAME), "w", encoding="utf-8") as f:
        f.write(text)
    meta = {"id": f"{args.user}/{args.slug}", "title": args.slug.replace("-", " ").title()[:60], "code_file": NOTEBOOK_NAME,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    with open(os.path.join(args.out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote", args.out, "|", len(files), "files |", f"payload {len(payload):,} B, notebook {size:,} B "
          f"(limit {args.max_notebook_bytes:,}) | tree sha256 {tree_sha256(manifest_of(files))[:16]} |",
          f"python -m {module} " + " ".join(cli_args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
