"""Check that the shipped policy corpus matches design D§8.2 (guide §13.2: "the corpus cannot drift").

    uv run python scripts/sync_policy_corpus.py ../VoxCareAI_Shared_Folder/research/42_voxcare_p1_detailed_design.md

The design lives in another repository, so the corpus is shipped as
``src/voxcare/domain/data/policies.json`` and this script verifies it against
the design file: the nine clause ids, clause references, titles, categories,
applicability rules and corpus fields must match exactly, and every number and
content word of each clause's terms must appear in the shipped text (the JSON
spells out the design's LaTeX, e.g. ``$500,000\\text{ VND}$`` → ``500,000 VND``).
Exit status 1 on any drift.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "src" / "voxcare" / "domain" / "data" / "policies.json"
_CLAUSE = re.compile(r"^- \*\*`(POL-[A-Z]+-\d{3})` \((Clause \d\.\d) — (.+?)\):\*\*\s*$")
_LATEX = [
    (re.compile(r"\\text\{\s*([^}]*)\}"), r" \1"),
    (re.compile(r"\\ge\b"), " >= "),
    (re.compile(r"[$]"), ""),
]


def _plain(text: str) -> str:
    for pattern, replacement in _LATEX:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+|[0-9]+", text.lower()) if len(word) > 2 or word.isdigit()}


def parse_design(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("### 8.2 "))
    end = next(i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("### 8.3 "))
    clauses: dict[str, dict] = {}
    current: dict | None = None
    for line in lines[start:end]:
        match = _CLAUSE.match(line)
        if match:
            current = {"policy_id": match[1], "clause_ref": match[2], "title": match[3], "terms": []}
            clauses[match[1]] = current
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            current = None  # a section heading ends the clause
            continue
        if stripped.startswith("- **Header:**"):
            current["header"] = stripped.split("`")[1]
        elif stripped.startswith("- **Metadata:**"):
            current["metadata"] = json.loads(stripped.split("**Metadata:**", 1)[1].strip())
        elif stripped.startswith("- **Terms:**"):
            current["terms"].append(stripped.split("**Terms:**", 1)[1])
        elif current.get("terms") and stripped:
            current["terms"].append(stripped)
    for clause in clauses.values():
        clause["terms"] = _plain(" ".join(clause["terms"]))
    return clauses


def check(design_path: Path) -> list[str]:
    design = parse_design(design_path)
    shipped = {clause["policy_id"]: clause for clause in json.loads(CORPUS.read_text(encoding="utf-8"))["clauses"]}
    problems = []
    if list(design) != list(shipped):
        problems.append(f"clause ids differ: design {list(design)} vs shipped {list(shipped)}")
    for clause_id, source in design.items():
        ours = shipped.get(clause_id)
        if ours is None:
            continue
        metadata = source.get("metadata", {})
        for key in ("clause_ref", "title"):
            if source[key] != ours[key]:
                problems.append(f"{clause_id}: {key} {ours[key]!r} != design {source[key]!r}")
        if source.get("header") != ours.get("header"):
            problems.append(f"{clause_id}: header differs")
        for key in ("category", "corpus_id", "clause_version", "is_fictional", "applicability_rules"):
            if metadata.get(key) != ours.get(key):
                problems.append(f"{clause_id}: {key} {ours.get(key)!r} != design {metadata.get(key)!r}")
        missing = _words(source["terms"]) - _words(ours["content_en"]) - {"ge"}
        if missing:
            problems.append(f"{clause_id}: terms missing words {sorted(missing)}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    problems = check(Path(argv[0]))
    for problem in problems:
        print(f"DRIFT {problem}")
    print("policy corpus matches D§8.2" if not problems else f"{len(problems)} drift finding(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
