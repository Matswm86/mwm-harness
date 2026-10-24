"""Spec (plan item 23): define, design, tasks, with a human gate on every step.

``/spec <idea>`` writes three files, in order, under ``specs/<slug>/`` in the repo being worked
on: ``requirements.md`` (user stories plus numbered acceptance criteria), ``design.md`` (stack,
structure, data, and a threat model whenever the feature touches auth, secrets or a model
surface), then ``tasks.md``, where every task names the criteria it closes. Each step waits for
the person: a draft goes through the same approval card as a model ``Write``, so "no" stops the
walk there and nothing lands on disk.

Two halves here are the harness's own, and they are the reason the flow exists rather than a
planner prompt:

- the shape is checked BEFORE a file is written. Criteria must be ``AC-N:`` lines, a task that
  names no criterion is refused, and a threat model the surface demands is not optional. A
  validator error means the step is not written and names exactly what is missing.
- every criterion is later asked for its test. ``criterion_coverage()`` maps each AC id to the
  test node ids that quote it, and ``assurance.py`` turns a criterion with no test into a FAIL
  on the correct row that NAMES that criterion. Prose cannot be checked against a diff; an id a
  test quotes can be.

The documents are the storage: ``load_spec()`` re-reads them from disk, so a spec written in one
session is checkable in the next, and an edit by hand is the edit that counts.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPEC_DIR = "specs"
STEPS = ("requirements", "design", "tasks")
DOC_NAME = {
    "requirements": "requirements.md",
    "design": "design.md",
    "tasks": "tasks.md",
}
# One draft call per step. A spec is a long answer, so it gets more room than a review comment.
DRAFT_TIMEOUT = 240.0
MAX_TEST_FILES = 2000
SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
        "build",
        "dist",
    }
)

# The whole contract, in regexes. Kept deliberately narrow: one shape to ask for, one shape to
# check, and an error message that can quote it back.
CRITERION = re.compile(r"^[ \t]*-[ \t]+(AC-[0-9]+):[ \t]*(\S.*)$", re.M | re.I)
TASK_LINE = re.compile(r"^[ \t]*-[ \t]+\[([ xX])\][ \t]+(\S.*)$", re.M)
CLOSES = re.compile(r"\b(?:closes|covers)\b[ \t]*:?[ \t]*(.*)$", re.I)
AC_TOKEN = re.compile(r"\bAC-([0-9]+)\b", re.I)
DEF_TEST = re.compile(r"^[ \t]*def (test_\w+)[ \t]*\(", re.M)
HEADING = re.compile(r"^[ \t]*##[ \t]+(\S.*)$", re.M)
SLUG_SAFE = re.compile(r"[^a-z0-9]+")

# A feature that touches any of this needs a threat model in design.md. This is a look, not a
# proof: it errs toward asking, because the cost of an unasked threat model is the one item 23
# exists to avoid.
THREAT_SURFACE = re.compile(
    r"\b(auth|authentication|authori[sz]ation|login|log ?in|password|credential|token|"
    r"secret|api[ _-]?key|private key|oauth|cookie|"
    r"payment|card number|invoice|bank|"
    r"e-?mail|phone number|personal data|pii|gdpr|patient|customer record|"
    r"upload|file path from|user input|"
    r"database|sql|shell command|subprocess|exec|deseriali[sz]|"
    r"webhook|callback|admin|multi-?tenant|permission|"
    r"model call|llm|prompt|mcp|tool call|embedding|vector|retrieval)\b",
    re.I,
)


def slug_from_idea(idea: str) -> str:
    """A folder name from the idea text: lowercase, hyphenated, capped at 48 characters."""
    slug = SLUG_SAFE.sub("-", idea.strip().lower()).strip("-")
    if not slug:
        return "spec"
    if len(slug) > 48:
        slug = slug[:48].rstrip("-")
    return slug


def spec_dir(root: Path, slug: str) -> Path:
    return root / SPEC_DIR / slug


def doc_path(root: Path, slug: str, step: str) -> Path:
    return spec_dir(root, slug) / DOC_NAME[step]


def strip_fences(text: str) -> str:
    """Unwrap a document the model wrapped in ``` fences. Deterministic, applied once."""
    body = text.strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def threat_surface(text: str) -> str:
    """The auth/data/model surface a spec text touches, or an empty string.

    Used to decide whether design.md must carry a threat model. Matches words in the idea and in
    the approved requirements, so a person who writes "login" in a story cannot dodge the check.
    """
    found = sorted({m.group(1).lower() for m in THREAT_SURFACE.finditer(text)})
    return ", ".join(found)


@dataclass
class Criterion:
    id: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    """One task line. Its id is its position: the documents carry no task-number to keep in sync."""

    id: str
    text: str
    closes: list[str]
    done: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        tail = f" (closes {', '.join(self.closes)})" if self.closes else ""
        return f"{self.id}: {self.text}{tail}"


@dataclass
class Step:
    """One step of the walk: drafted, checked, then gated on approval."""

    step: str
    file: str
    status: str  # written | refused | invalid | failed | skipped | on-disk
    reason: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Spec:
    """What is on disk under specs/<slug>/."""

    slug: str
    root: str
    criteria: list[Criterion] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    requirements: str = ""
    design: str = ""

    @property
    def directory(self) -> str:
        return f"{SPEC_DIR}/{self.slug}"

    def criterion_ids(self) -> list[str]:
        return [c.id for c in self.criteria]

    def tasks_without_criteria(self) -> list[Task]:
        return [t for t in self.tasks if not t.closes]

    def uncovered_tasks(self) -> list[Task]:
        """Tasks naming a criterion id this spec does not contain (a stale or invented link)."""
        known = set(self.criterion_ids())
        return [t for t in self.tasks if any(c not in known for c in t.closes)]

    def orphan_criteria(self) -> list[str]:
        """Criteria no task closes. Advice, not a refusal: the spec may be mid-walk."""
        closed = {c for t in self.tasks for c in t.closes}
        return [c for c in self.criterion_ids() if c not in closed]

    def text(self) -> str:
        return "\n\n".join(part for part in (self.requirements, self.design) if part)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "dir": self.directory,
            "files": list(self.files),
            "criteria": [c.as_dict() for c in self.criteria],
            "tasks": [t.as_dict() for t in self.tasks],
        }


def parse_criteria(markdown: str) -> list[Criterion]:
    seen: dict[str, Criterion] = {}
    for match in CRITERION.finditer(markdown):
        cid = match.group(1).upper()
        if cid not in seen:  # a repeat is reported by validate, and the first text wins here
            seen[cid] = Criterion(id=cid, text=match.group(2).strip())
    return sorted(seen.values(), key=lambda c: int(c.id.split("-")[1]))


def parse_tasks(markdown: str) -> list[Task]:
    tasks: list[Task] = []
    for number, match in enumerate(TASK_LINE.finditer(markdown), start=1):
        done, body = match.group(1).lower() == "x", match.group(2).strip()
        closes_match = CLOSES.search(body)
        closes: list[str] = []
        if closes_match:
            for token in AC_TOKEN.findall(closes_match.group(1)):
                cid = f"AC-{token}"
                if cid not in closes:
                    closes.append(cid)
        text = body
        if closes_match:
            text = body[: closes_match.start()].strip(" ()[]")
        tasks.append(Task(id=f"T{number}", text=text, closes=closes, done=done))
    return tasks


def _missing_heading(markdown: str, pattern: str) -> str:
    if not any(re.search(pattern, h, re.I) for h in HEADING.findall(markdown)):
        return f"a level-2 section heading matching {pattern!r}"
    return ""


def validate_requirements(markdown: str) -> list[str]:
    """Errors that stop requirements.md from being written. Empty list means it may be written."""
    errors: list[str] = []
    for pattern, name in (
        (r"user stor", "## User stories"),
        (r"acceptance criteri", "## Acceptance criteria"),
    ):
        missing = _missing_heading(markdown, pattern)
        if missing:
            errors.append(f"missing {missing} (write it as {name})")
    if not parse_criteria(markdown):
        errors.append(
            "no acceptance criterion found: each one is its own line shaped exactly "
            '"- AC-1: <testable statement>"'
        )
    ids = [m.group(1).upper() for m in CRITERION.finditer(markdown)]
    repeats = sorted({i for i in ids if ids.count(i) > 1})
    if repeats:
        errors.append(f"duplicate criterion ids: {', '.join(repeats)} (ids must be unique)")
    return errors


def validate_design(markdown: str, surface: str) -> list[str]:
    errors: list[str] = []
    for pattern, name in (
        (r"stack", "## Stack"),
        (r"structure", "## Structure"),
        (r"data", "## Data"),
    ):
        missing = _missing_heading(markdown, pattern)
        if missing:
            errors.append(f"missing {missing} (write it as {name})")
    if surface:
        missing = _missing_heading(markdown, r"threat model")
        if missing:
            errors.append(
                f"missing {missing}: this feature touches {surface}, so design.md must name the "
                "assets, the ways in, and what the harness refuses"
            )
        elif not _section_body(markdown, r"threat model"):
            errors.append(
                "the ## Threat model section is empty: name the assets, the ways in, "
                "and what the harness refuses"
            )
    return errors


def validate_tasks(markdown: str, criteria: list[Criterion]) -> list[str]:
    errors: list[str] = []
    tasks = parse_tasks(markdown)
    if not tasks:
        errors.append(
            "no task found: each task is its own line shaped exactly "
            '"- [ ] <what to build> (closes AC-1, AC-3)"'
        )
    known = {c.id for c in criteria}
    for task in tasks:
        if not task.closes:
            errors.append(
                f"{task.id} names no criterion, so it cannot be checked: "
                f'"{task.text[:90]}" needs "(closes AC-N, AC-M)"'
            )
        else:
            unknown = [c for c in task.closes if c not in known]
            if unknown:
                errors.append(
                    f"{task.id} closes {', '.join(unknown)}, "
                    f"which requirements.md does not contain (it has {', '.join(sorted(known)) or 'none'})"
                )
    if not any(re.search(r"task", h, re.I) for h in HEADING.findall(markdown)):
        errors.append(
            "missing a level-2 section heading matching 'task' (write it as \"## Tasks\")"
        )
    return errors


def _section_body(markdown: str, pattern: str) -> str:
    """Text under the first heading matching ``pattern``, up to the next heading."""
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        heading = re.match(r"^[ \t]*##[ \t]+(\S.*)$", line)
        if heading and re.search(pattern, heading.group(1), re.I):
            start = index + 1
            break
    if start is None:
        return ""
    body: list[str] = []
    for line in lines[start:]:
        if line.lstrip().startswith("##"):
            break
        body.append(line)
    return "\n".join(body).strip()


def changed_criteria(old: list[Criterion], new: list[Criterion]) -> list[str]:
    """Ids whose text moved or that disappeared: the criteria whose tests are owed a rerun."""
    before = {c.id: c.text.strip().lower() for c in old}
    after = {c.id: c.text.strip().lower() for c in new}
    out: list[str] = []
    for cid, text in before.items():
        if cid not in after or after[cid] != text:
            out.append(cid)
    return out


def test_files(root: Path) -> list[Path]:
    """The repo's pytest files, with the usual heavy directories pruned."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
                found.append(Path(dirpath) / name)
                if len(found) >= MAX_TEST_FILES:
                    return sorted(found)
    return sorted(found)


def criterion_coverage(root: Path, ids: list[str]) -> dict[str, list[str]]:
    """Map each criterion id to the test node ids that quote it.

    A criterion counts as tested when a test file names its id, inside a ``def test_...`` (the
    node id) or at module level (the bare file path, which is weaker and reads as one). The id is
    the join key, so a test that is deleted or renamed away from it shows up as a gap by itself.
    """
    wanted = {i.upper() for i in ids}
    hits: dict[str, list[str]] = {i: [] for i in sorted(wanted)}
    if not wanted:
        return hits
    for path in test_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "AC-" not in text.upper():
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = path.name
        defs = [(m.start(), m.group(1)) for m in DEF_TEST.finditer(text)]
        for match in AC_TOKEN.finditer(text):
            cid = f"AC-{match.group(1)}"
            if cid not in hits:
                continue
            offset = match.start()
            owner = next((name for pos, name in reversed(defs) if pos < offset), "")
            node = f"{rel}::{owner}" if owner else rel
            if node not in hits[cid]:
                hits[cid].append(node)
    return hits


def coverage_summary(coverage: dict[str, list[str]]) -> dict[str, Any]:
    """The card's per-criterion evidence, as plain data (no Spec object crosses into assurance)."""
    uncovered = [cid for cid, nodes in coverage.items() if not nodes]
    return {
        "criteria": list(coverage),
        "coverage": coverage,
        "uncovered": uncovered,
        "tested": len(coverage) - len(uncovered),
        "total": len(coverage),
    }


SLUG_SHAPE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def is_slug(text: str) -> bool:
    """Whether a person's argument is a folder name and not a path.

    ``/spec <slug>`` reaches into the repo by name, so only the shape ``slug_from_idea`` can
    produce is accepted: no dot, no slash, no leading dash, at most 48 characters. ``..`` and
    ``../../.ssh`` fail here instead of being resolved later.
    """
    return bool(SLUG_SHAPE.match(text))


def existing_slugs(root: Path) -> list[str]:
    """The spec folders on disk."""
    base = root / SPEC_DIR
    try:
        return sorted(p.name for p in base.iterdir() if p.is_dir() and is_slug(p.name))
    except OSError:
        return []


def load_spec(root: Path, slug: str) -> Spec | None:
    """Re-read a spec from disk, taking whichever documents are there.

    The files are the storage: a criterion edited by hand in requirements.md is the criterion the
    next card checks, and a session that never ran /spec can still check a spec a previous one
    wrote. Returns None when nothing readable is on disk.
    """
    if not is_slug(slug):
        return None
    base = spec_dir(root, slug)
    if not base.is_dir():
        return None
    texts: dict[str, str] = {}
    files: list[str] = []
    for step, name in DOC_NAME.items():
        path = base / name
        if not path.is_file():
            continue
        try:
            texts[step] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append(name)
    if not files:
        return None
    return Spec(
        slug=slug,
        root=str(root),
        criteria=parse_criteria(texts.get("requirements", "")),
        tasks=parse_tasks(texts.get("tasks", "")),
        files=files,
        requirements=texts.get("requirements", ""),
        design=texts.get("design", ""),
    )


@dataclass
class SpecReport:
    """The outcome of one ``/spec`` walk, one step, or one status read."""

    slug: str
    dir: str
    status: str  # written | partial | refused | invalid | failed | loaded | skipped
    model: str = ""
    reason: str = ""
    steps: list[Step] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    coverage: dict[str, list[str]] = field(default_factory=dict)
    surface: str = ""
    orphans: list[str] = field(default_factory=list)
    refuted: list[str] = field(default_factory=list)
    stale_tasks: list[str] = field(default_factory=list)
    tasks_loaded: int = 0
    reflagged: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def uncovered(self) -> list[str]:
        return [cid for cid, nodes in self.coverage.items() if not nodes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "dir": self.dir,
            "status": self.status,
            "model": self.model,
            "reason": self.reason,
            "steps": [s.as_dict() for s in self.steps],
            "criteria": [c.as_dict() for c in self.criteria],
            "tasks": [t.as_dict() for t in self.tasks],
            "coverage": self.coverage,
            "uncovered": self.uncovered(),
            "surface": self.surface,
            "orphans": self.orphans,
            "tasks_without_criteria": self.refuted,
            "stale_tasks": self.stale_tasks,
            "tasks_loaded": self.tasks_loaded,
            "reflagged": self.reflagged,
            "seconds": round(self.seconds, 2),
        }

    def summary(self) -> str:
        """One line: what a stranger needs first, with the number in it."""
        if self.status in ("skipped", "refused", "failed", "invalid") and not self.steps:
            return f"spec {self.status}: {self.reason}"
        written = [s.step for s in self.steps if s.status == "written"]
        head = "wrote " + ", ".join(written) if written else self.status
        bits = [f"{len(self.criteria)} criteria", f"{len(self.tasks)} tasks"]
        gap = self.uncovered()
        if gap:
            bits.append(f"{len(gap)} criterion without a test ({', '.join(gap)})")
        else:
            bits.append("every criterion has a test naming it")
        return f"spec {self.slug}: {head} ({self.dir}): " + ", ".join(bits)


def system_prompt() -> str:
    return (
        "You are the spec writer inside the MWM harness. You return one markdown document and "
        "nothing else: no preamble, no closing commentary, no code fence around the whole "
        "document. The harness parses what you return and refuses to write the file if a line "
        "does not match the contract it gives you, so match it exactly. Write the smallest "
        "correct version of what was asked: never invent a requirement the idea does not state, "
        "and never write a criterion no test could prove."
    )


def draft_prompt(step: str, slug: str, idea: str, repo_note: str, prior: str) -> str:
    """The contract for one step, with the approved documents above it quoted in full."""
    if step == "requirements":
        contract = (
            "Write the requirements document for this idea.\n\n"
            f"IDEA:\n{idea}\n\n"
            f"The file lands at specs/{slug}/requirements.md and must contain exactly two "
            "level-2 sections, in this order:\n\n"
            "## User stories\n"
            "- one bullet per story, each in the form "
            '"As a <role>, I want <thing> so that <reason>."\n\n'
            "## Acceptance criteria\n"
            "- AC-1: <one testable statement, phrased so a test can prove it>\n"
            "- AC-2: <next statement>\n\n"
            "Rules for criteria: numbered AC-1, AC-2, AC-3 in order, one per line, no gaps, no "
            "repeats, between 3 and 9 of them, the line shape exactly "
            '"- AC-N: text". Each names a behavior a person or a test can observe, never an '
            "implementation detail. If the idea is unclear, write the smallest testable version "
            "of it and say nothing else about the ambiguity."
        )
    elif step == "design":
        surface = threat_surface(prior)
        required = (
            "\n\nThis feature touches "
            f"{surface}, so a fourth section is REQUIRED:\n"
            "## Threat model\n"
            "- the asset: what an attacker wants\n"
            "- the way in: the entry point that reaches it\n"
            "- the refusal: what this harness or this feature must never do\n"
            "An empty threat model section is refused."
            if surface
            else "\n\nNo auth, data or model surface was found in the requirements, so a threat "
            "model section is optional. If you add one, give it real entries."
        )
        contract = (
            "Write the design document for a spec whose requirements are quoted below.\n\n"
            "APPROVED REQUIREMENTS:\n"
            f"{prior}\n\n"
            f"The file lands at specs/{slug}/design.md and must contain these level-2 "
            "sections, in this order:\n\n"
            "## Stack\n"
            "- what is used, and what already exists in this repo that it reuses\n\n"
            "## Structure\n"
            "- the modules and functions that change, by path, one bullet each\n\n"
            "## Data\n"
            "- what is stored, where, and what crosses a boundary" + required + "\n\n"
            "Name real paths from the repo note. Do not restate the criteria."
        )
    else:
        contract = (
            "Write the task list for a spec whose requirements and design are quoted below.\n\n"
            "APPROVED SPEC:\n"
            f"{prior}\n\n"
            f"The file lands at specs/{slug}/tasks.md and must contain one level-2 section:\n\n"
            "## Tasks\n"
            "- [ ] <what to build> (closes AC-1, AC-3)\n"
            "- [ ] <next task> (closes AC-2)\n\n"
            "Rules: every task is an unchecked box line, and every task line must end with "
            '"(closes AC-N)" naming at least one criterion that the requirements above '
            "actually contain. A task that names no criterion is refused, so never write one. "
            "Order the tasks so the first one delivers something checkable. Between 3 and 12 "
            "tasks, each one small enough to finish without a second spec."
        )
    note = f"\n\nREPO NOTE:\n{repo_note}" if repo_note else ""
    return f"{contract}{note}\n\nReturn the document now."
