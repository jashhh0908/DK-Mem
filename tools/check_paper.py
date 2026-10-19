"""Is the paper ready? Exit 0 only when it is (the definition of "done" for the team).

    python tools/check_paper.py            # full check (use before submission)
    python tools/check_paper.py --draft    # report problems but exit 0 (use during the day)

Checks: no \\todo{...} left in paper/sections; no "% UNVERIFIED" reference in refs.bib; every \\cite key exists in refs.bib; every
number macro used in the sections is defined in paper/generated/*.tex; every section file is \\input by main.tex; no result typed by hand
in results.tex/abstract.tex (a literal "k/n" fraction outside a macro).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
LATEX_BUILTINS = {"section", "paragraph", "label", "ref", "cite", "citep", "citet", "emph", "textbf", "todo", "dev", "guj", "input", "caption",
                  "centering", "small", "begin", "end", "item", "url", "footnote", "noindent", "textit", "texttt", "includegraphics", "maketitle",
                  "tau", "ge", "le", "times", "approx", "mathrm", "frac", "text", "appendix", "bibliography", "bibliographystyle", "hline",
                  "midrule", "toprule", "bottomrule", "multicolumn", "textcolor", "subsection", "subsubsection", "newcommand", "IfFileExists",
                  "abstract", "title", "author", "date"}


def main(argv) -> int:
    draft = "--draft" in argv
    problems = []
    sections = {p.stem: p.read_text(encoding="utf-8") for p in sorted((PAPER / "sections").glob("*.tex"))}
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    for name in sections:
        if f"\\input{{sections/{name}}}" not in main_tex:
            problems.append(f"sections/{name}.tex is not included in main.tex")
    for name, text in sections.items():
        n = len(re.findall(r"\\todo\{", text))
        if n:
            problems.append(f"sections/{name}.tex: {n} \\todo left")
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    unverified = len(re.findall(r"%\s*UNVERIFIED", bib))
    if unverified:
        problems.append(f"refs.bib: {unverified} entries still marked UNVERIFIED")
    defined = set()
    for g in (PAPER / "generated").glob("*.tex"):
        defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", g.read_text(encoding="utf-8")))
    defined |= set(re.findall(r"\\newcommand\{?\\(\w+)\}?", main_tex))
    for name, text in sections.items():
        body = "\n".join(l.split("%")[0] for l in text.splitlines())          # ignore comments
        for c in re.findall(r"\\cite[pt]?\{([^}]*)\}", body):
            for k in c.split(","):
                if k.strip() not in keys:
                    problems.append(f"sections/{name}.tex cites unknown key {k.strip()!r}")
        for m in set(re.findall(r"\\([A-Z]\w+)", body)):
            if m not in defined and m.lower() not in LATEX_BUILTINS:
                problems.append(f"sections/{name}.tex uses undefined macro \\{m} (regenerate paper/generated or fix the name)")
        if name in ("results", "abstract"):
            for frac in re.findall(r"(?<![\\\w{])\d+/\d+(?![\w}])", body):
                problems.append(f"sections/{name}.tex: hand-typed result {frac!r}; use a generated macro")
    print("paper check:", "OK" if not problems else f"{len(problems)} problem(s)")
    for p in problems:
        print("  -", p)
    return 0 if (draft or not problems) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
