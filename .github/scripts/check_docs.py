"""Doc check for foundation-model's Markdown (standard library only).

Four checks, on the Markdown sources:

1. Every relative link in README.md and under docs/ resolves to a file or
   directory that exists. External links and in-page anchors are not checked.
2. No Markdown file carries a second front-matter block: a `---` line, then
   only `key: value` lines, then `---`, anywhere but at the very top. That
   is what a stray fragment left by a merge looks like.
3. Every test the README cites (`tests/<file>.py::<name>`, or `::<name>`
   after one) exists: each name in the chain is a function or class in that
   file. A README must never cite a test that is not there.
4. What the README counts or tabulates matches the code: "six interlocking
   capabilities" against the overview table's rows, "Four samplers" and the
   sampler table against the `*Sampler` classes in foundation_model/sampling.py,
   and the scale presets table against the classmethods in
   foundation_model/config.py (d_model, layers, heads, d_ff).

The shape follows architecture-definition-model's .github/scripts/check_docs.py.
The README-proof convention it backs was Dermot's decision of 10 October
2026: every capability row or bullet in the README names the test that proves
it, or says "no test yet" or "not yet implemented".

Run from anywhere: python .github/scripts/check_docs.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {"node_modules", "build", "dist"}
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?")
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
KEY = re.compile(r"^[A-Za-z_][\w-]*\s*:(\s|$)")
FENCE = re.compile(r"^\s*(```|~~~)")
CITE = re.compile(r"`(tests/[\w/.-]+\.py)?((?:::\w+)+)`")
NUMBERS = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
])}


def markdown_files() -> list[Path]:
    found = []
    for path in sorted(ROOT.rglob("*.md")):
        parts = path.relative_to(ROOT).parts
        if any(p.startswith(".") or p in SKIP or p.endswith(".egg-info") for p in parts[:-1]):
            continue
        found.append(path)
    return found


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def prose(lines: list[str]) -> list[tuple[int, str]]:
    """Lines outside fenced code blocks, with their line numbers."""
    kept, fenced = [], False
    for number, line in enumerate(lines, start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            kept.append((number, line))
    return kept


def front_matter(path: Path, errors: list[str]) -> None:
    lines, fenced, i = read(path), False, 0
    while i < len(lines):
        if FENCE.match(lines[i]):
            fenced = not fenced
        elif not fenced and lines[i].strip() == "---":
            j = i + 1
            while j < len(lines) and KEY.match(lines[j]):
                j += 1
            if j > i + 1 and j < len(lines) and lines[j].strip() == "---":
                if i > 0:
                    errors.append(f"{rel(path)}:{i + 1}: a front-matter block below the top")
                i = j
        i += 1


def links(path: Path, errors: list[str]) -> None:
    for number, line in prose(read(path)):
        line = INLINE_CODE.sub("", line)
        targets = [m.group(1) for m in LINK.finditer(line)]
        ref = REF.match(line)
        if ref:
            targets.append(ref.group(1))
        for target in targets:
            if SCHEME.match(target) or target.startswith("#"):
                continue
            bare = target.split("#", 1)[0].split("?", 1)[0]
            if not bare:
                continue
            base = ROOT if bare.startswith("/") else path.parent
            if not (base / bare.lstrip("/")).exists():
                errors.append(f"{rel(path)}:{number}: link {target!r} resolves to nothing")


def defined(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return {node.name for node in ast.walk(tree) if isinstance(node, kinds)}


def citations(path: Path, errors: list[str]) -> int:
    seen, current = 0, None
    for number, line in prose(read(path)):
        for match in CITE.finditer(line):
            current = match.group(1) or current
            names = match.group(2).strip(":").split("::")
            where = f"{rel(path)}:{number}: cites {match.group(0)}"
            if current is None:
                errors.append(f"{where} with no test file before it")
                continue
            target = ROOT / current
            if not target.is_file():
                errors.append(f"{where}, and {current} does not exist")
                continue
            missing = [n for n in names if n not in defined(target)]
            if missing:
                errors.append(f"{where}, and {current} defines no {', '.join(missing)}")
            seen += 1
    return seen


def number(word: str) -> int | None:
    return int(word) if word.isdigit() else NUMBERS.get(word.lower())


def table_after(lines: list[str], header: str) -> list[list[str]]:
    """The body rows of the first table whose header row starts with `header`."""
    rows: list[list[str]] = []
    for i, line in enumerate(lines):
        if line.startswith(header):
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                rows.append([cell.strip() for cell in row.strip().strip("|").split("|")])
            break
    return rows


def code(errors: list[str]) -> None:
    lines = read(ROOT / "README.md")
    text = "\n".join(lines)

    capabilities = table_after(lines, "| Capability |")
    said = re.search(r"provides (\w+) interlocking capabilities", text)
    if not said or number(said.group(1)) != len(capabilities):
        errors.append(f"README.md: the capability count does not match the {len(capabilities)} "
                      "rows of the overview table")

    sampling = ast.parse((ROOT / "foundation_model" / "sampling.py").read_text(encoding="utf-8"))
    samplers = {n.name for n in sampling.body
                if isinstance(n, ast.ClassDef) and n.name.endswith("Sampler")}
    said = re.search(r"(\w+) samplers prevent", text)
    if not said or number(said.group(1)) != len(samplers):
        errors.append(f"README.md: the sampler count does not match the {len(samplers)} "
                      "*Sampler classes in foundation_model/sampling.py")
    tabled = {row[0].strip("`") for row in table_after(lines, "| Sampler |")}
    if tabled != samplers:
        errors.append(f"README.md: the sampler table lists {sorted(tabled)}, "
                      f"and sampling.py defines {sorted(samplers)}")

    config = ast.parse((ROOT / "foundation_model" / "config.py").read_text(encoding="utf-8"))
    presets: dict[str, dict[str, int]] = {}
    for node in ast.walk(config):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Name) and d.id == "classmethod" for d in node.decorator_list
        ):
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    presets[node.name] = {
                        k.arg: k.value.value for k in call.keywords
                        if k.arg and isinstance(k.value, ast.Constant)
                        and isinstance(k.value.value, int)
                    }
    rows = table_after(lines, "| Preset |")
    if {row[0].strip("`") for row in rows} != set(presets):
        errors.append(f"README.md: the presets table does not list exactly {sorted(presets)}")
    for row in rows:
        name, want = row[0].strip("`"), presets.get(row[0].strip("`"), {})
        shown = dict(zip(("d_model", "n_layers", "n_heads", "d_ff"), row[1:5]))
        for field, value in shown.items():
            if want.get(field) is not None and str(want[field]) != value:
                errors.append(f"README.md: preset {name} shows {field} {value}, "
                              f"and config.py has {want[field]}")


def main() -> int:
    errors: list[str] = []
    files = markdown_files()
    checked = [p for p in files if rel(p) == "README.md" or rel(p).startswith("docs/")]
    for path in files:
        front_matter(path, errors)
    for path in checked:
        links(path, errors)
    cited = citations(ROOT / "README.md", errors)
    code(errors)
    for error in errors:
        print(error)
    print(f"{len(files)} Markdown files, {len(checked)} link-checked, "
          f"{cited} test citations, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
