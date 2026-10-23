"""Who calls this: a static call map of a Python tree, built with the standard library's ``ast``.

Two jobs. The Review Desk asks for the blast radius of the functions a turn changed, so the
reader sees "called by 4: loop.Session._run_tools, ..." before deciding what matters. And
:meth:`CodeMap.text` writes a one-line-per-function map of a package, which a model with a
small context window can read instead of whole files.

Nothing here guesses. A call whose target only exists at run time (``getattr``, a dict entry,
a local variable holding a callable, a method this class does not define) is listed as
unresolved. A bare name that no import explains and no block in the file defines is left
alone: a builtin and a local variable look identical from here, and inventing an edge would
be worse than showing none. A method reached through a variable (``session.send(...)``) is
not an edge either, for the same reason: nothing in the source says which class it is.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "build",
        "dist",
    }
)
MAX_FILES = 4_000  # a tree bigger than this is not one package, and the map stops saying so
MAX_FILE_BYTES = 400_000  # a generated file this large says nothing about who calls what
MAX_TEXT_CHARS = (
    24_000  # the map exists for small windows, so it is cut rather than allowed to grow
)
CALL_TEXT = 60  # characters of a call expression kept when it names no target


@dataclass(eq=False)
class Block:
    """One function or method. A class is not a block: its methods are, named ``Class.method``."""

    file: str  # path relative to the map's root
    name: str  # qualified inside its file: "collect_changes", "Session.send"
    label: str  # qualified across the tree: "review.collect_changes"
    kind: str  # module | function | method
    line: int
    end_line: int
    calls: list[str] = field(default_factory=list)  # labels of the repo blocks this one calls
    callers: list[str] = field(default_factory=list)  # labels of the blocks that call this one
    unresolved: list[str] = field(default_factory=list)  # "getattr(spec, name):120" style

    def covers(self, line: int) -> bool:
        return self.line <= line <= self.end_line


@dataclass
class CodeMap:
    """Every block in a tree, linked. Build one with :func:`index`; it is read-only afterwards."""

    root: str
    files: dict[str, list[Block]] = field(default_factory=dict)  # relative path -> blocks in order
    modules: dict[str, str] = field(default_factory=dict)  # dotted module name -> relative path
    errors: dict[str, str] = field(default_factory=dict)  # relative path -> why it was not parsed
    notes: list[str] = field(default_factory=list)  # what was left out, and why

    def block(self, dotted: str) -> Block | None:
        """The block a dotted name points at, or None when this tree has no such block."""
        parts = dotted.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            rel = self.modules.get(".".join(parts[:cut]))
            if rel is None:
                continue
            name = ".".join(parts[cut:])
            for block in self.files.get(rel, []):
                if block.name == name:
                    return block
        return None

    def at(self, path: str, lines: Iterable[int]) -> list[Block]:
        """The functions in ``path`` whose body covers any of ``lines``."""
        wanted = set(lines)
        return [
            block
            for block in self.files.get(path, ())
            if block.kind != "module" and any(block.covers(line) for line in wanted)
        ]

    def radius(self, path: str, lines: Iterable[int]) -> tuple[list[str], list[str]]:
        """Who calls the changed functions of one file, and which of their calls stay opaque."""
        callers: list[str] = []
        unresolved: list[str] = []
        for block in self.at(path, lines):
            callers.extend(block.callers)
            unresolved.extend(block.unresolved)
        return _unique(callers), _unique(unresolved)

    def text(self, paths: Sequence[str] = (), limit: int = MAX_TEXT_CHARS) -> str:
        """One line per function: enough for a model to choose what to read, and no code."""
        chosen = {rel for rel in self.files if not paths or rel in set(paths)}
        functions = sum(
            1
            for rel, blocks in self.files.items()
            if rel in chosen
            for b in blocks
            if b.kind != "module"
        )
        rows = [
            f"# code map of {Path(self.root).absolute().name}: {functions} function(s)"
            f" in {len(chosen)} file(s)",
            "# per file, one line each: name lines ->calls <-callers ?resolved only at run time",
            "# labels read module.Class.method; a caller label names the function that makes the call",
        ]
        for note in [
            *self.notes,
            *(f"not parsed: {rel} ({why})" for rel, why in sorted(self.errors.items())),
        ]:
            rows.append(f"# {note}")
        for rel in sorted(chosen):
            blocks = [b for b in self.files[rel] if b.kind != "module"]
            if not blocks:
                continue
            rows.append(rel)
            for block in blocks:
                rows.append(
                    f"  {block.name} {block.line}-{block.end_line}"
                    f" ->{len(block.calls)} <-{len(block.callers)}"
                    + (f" ?{len(block.unresolved)}" if block.unresolved else "")
                )
        text = "\n".join(rows)
        if len(text) > limit:
            text = text[:limit].rsplit("\n", 1)[0] + f"\n# cut at {limit:,} characters"
        return text


@dataclass
class _Site:
    """One call, as the scan saw it: what to try, and where it happened."""

    owner: Block
    line: int
    name: str = ""  # a bare-name call: "helper()"
    base: str = ""  # the name an attribute call hangs off: "self", "review", "target"
    attr: str = ""  # the attribute being called: "collect_changes"
    text: str = ""  # a call that names nothing: "handlers[kind]", "super().run"


@dataclass
class _File:
    rel: str
    tree: ast.AST
    imports: dict[str, str] = field(default_factory=dict)  # local name -> dotted target
    blocks: list[Block] = field(default_factory=list)
    bound: dict[int, set[str]] = field(default_factory=dict)  # id(block) -> names bound inside it
    sites: list[_Site] = field(default_factory=list)


# --------------------------------------------------------------------- build


def python_files(root: Path, limit: int = MAX_FILES) -> tuple[list[Path], list[str]]:
    """Every ``.py`` under ``root``, sorted, minus vendored, hidden and generated folders."""
    kept = [
        path
        for path in sorted(Path(root).rglob("*.py"))
        if not any(
            part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts
        )
    ]
    notes = (
        [f"{len(kept) - limit} file(s) left out: the map stops at {limit}"]
        if len(kept) > limit
        else []
    )
    return kept[:limit], notes


def index(root: Path) -> CodeMap:
    """Parse a tree and link its calls. A file that will not parse is reported, not raised."""
    base = Path(root)
    paths, notes = python_files(base)
    errors: dict[str, str] = {}
    scanned: list[_File] = []
    for path in paths:
        rel = path.relative_to(base).as_posix()
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                notes.append(f"{rel} left out: over {MAX_FILE_BYTES:,} bytes")
                continue
            tree = ast.parse(path.read_bytes(), filename=rel)
        except (OSError, SyntaxError, ValueError) as exc:
            errors[rel] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        scanned.append(_File(rel, tree))

    rels = [f.rel for f in scanned]
    cmap = CodeMap(
        root=str(base),
        files={f.rel: f.blocks for f in scanned},
        modules=_module_index(rels),
        errors=errors,
        notes=notes,
    )
    labels = _label_modules(rels)
    for found in scanned:
        _scan(found, labels[found.rel])
    _link(cmap, scanned)
    return cmap


def text(root: Path, paths: Sequence[str] = ()) -> str:
    """The map of ``root`` as text, in one call: :func:`index` then :meth:`CodeMap.text`."""
    return index(root).text(paths)


def _module_index(rels: Sequence[str]) -> dict[str, str]:
    """Every dotted name a file in this tree can be imported by, shortest path first."""
    modules: dict[str, str] = {}
    for rel in sorted(rels):
        parts = Path(rel).with_suffix("").parts
        for cut in range(len(parts)):
            modules.setdefault(".".join(parts[cut:]), rel)
    return modules


def _label_modules(rels: Sequence[str]) -> dict[str, str]:
    """How each file is named in a label: its stem, or a longer form when the stem repeats."""
    groups: dict[str, list[str]] = {}
    for rel in rels:
        groups.setdefault(Path(rel).stem, []).append(rel)
    labels: dict[str, str] = {}
    for stem, group in groups.items():
        if len(group) == 1:
            labels[group[0]] = stem
            continue
        for rel in group:
            parts = Path(rel).with_suffix("").parts
            labels[rel] = ".".join(parts[-2:]) if len(parts) > 1 else stem
    seen: dict[str, str] = {}
    for rel, label in sorted(labels.items()):
        if label in seen:
            for both in (rel, seen[label]):
                labels[both] = Path(both).with_suffix("").as_posix().replace("/", ".")
        else:
            seen[label] = rel
    return labels


def _scan(found: _File, label: str) -> None:
    """Imports, blocks and call sites of one file."""
    for node in ast.walk(found.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.imports[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            prefix = _import_prefix(found.rel, node)
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.imports[alias.asname or alias.name] = f"{prefix}.{alias.name}"

    end = getattr(found.tree, "end_lineno", None) or 1
    module_block = Block(found.rel, "<module>", label, "module", 1, end)
    found.blocks.append(module_block)
    _visit(found, found.tree, module_block, (), label)


def _import_prefix(rel: str, node: ast.ImportFrom) -> str:
    """The dotted package a ``from ... import`` names: as written, or resolved against this file."""
    if node.level == 0:
        return node.module or ""
    parts = list(Path(rel).parent.parts)
    if node.level > 1:
        parts = parts[: max(0, len(parts) - (node.level - 1))]
    return ".".join([*parts, node.module] if node.module else parts)


def _visit(found: _File, node: ast.AST, owner: Block, stack: tuple[str, ...], label: str) -> None:
    """Record the calls that belong to ``owner``, then descend into the blocks it contains."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            _visit(found, child, owner, (*stack, child.name), label)
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = ".".join((*stack, child.name))
            block = Block(
                found.rel,
                name,
                f"{label}.{name}",
                "method" if isinstance(node, ast.ClassDef) else "function",
                child.lineno,
                child.end_lineno or child.lineno,
            )
            found.blocks.append(block)
            found.bound[id(block)] = _bound(child)
            _visit(found, child, block, (*stack, child.name), label)
            continue
        _own_sites(found, child, owner)


def _own_sites(found: _File, node: ast.AST, owner: Block) -> None:
    """Every call, and every imported name used as a value, that belongs to ``owner``.

    A name handed to ``asyncio.to_thread`` or used in an ``isinstance`` is not a call site, but
    it does mean this block reaches that one, which is what a blast radius is for.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Call):
            found.sites.append(_site(owner, child))
        elif isinstance(child, ast.Name) and child.id in found.imports:
            found.sites.append(_Site(owner, child.lineno, name=child.id))
        _own_sites(found, child, owner)


def _site(owner: Block, call: ast.Call) -> _Site:
    """What the scan can say about one call. An empty site is one worth no line anywhere."""
    func = call.func
    if isinstance(func, ast.Name):
        if func.id == "getattr":
            return _Site(owner, call.lineno, text=_call_text(call, call.lineno))
        return _Site(owner, call.lineno, name=func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return _Site(owner, call.lineno, base=func.value.id, attr=func.attr)
    if isinstance(func, (ast.Subscript, ast.Call)):
        # a dict entry, or a call that returns one: dispatch decided while the program runs
        return _Site(owner, call.lineno, text=_call_text(func, call.lineno))
    return _Site(
        owner, call.lineno
    )  # an attribute chain: nothing but a type checker could follow it


def _call_text(func: ast.expr, line: int) -> str:
    try:
        shown = ast.unparse(func)
    except (ValueError, RecursionError):  # an exotic node is still worth naming by its line
        return f"<call>:{line}"
    shown = " ".join(shown.split())
    return f"{shown[:CALL_TEXT]}:{line}"


def _bound(node: ast.AST) -> set[str]:
    """Names this block binds itself: parameters, assignments, and the defs it contains."""
    names: set[str] = set()

    def walk(child: ast.AST) -> None:
        for kid in ast.iter_child_nodes(child):
            if isinstance(kid, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(kid.name)  # the name is bound here; its body binds inside itself
                continue
            if isinstance(kid, ast.Name) and isinstance(kid.ctx, ast.Store):
                names.add(kid.id)
            elif isinstance(kid, ast.arg):
                names.add(kid.arg)
            elif isinstance(kid, ast.ExceptHandler) and kid.name:
                names.add(kid.name)
            walk(kid)

    walk(node)
    return names


# ---------------------------------------------------------------------- link


def _link(cmap: CodeMap, scanned: Sequence[_File]) -> None:
    by_rel = {found.rel: found for found in scanned}
    for (
        found
    ) in scanned:  # cleared before any edge is written: edges cross files in both directions
        for block in found.blocks:
            block.calls = []
            block.callers = []
            block.unresolved = []
    for found in scanned:
        for site in found.sites:
            target, opaque = _resolve(cmap, by_rel[site.owner.file], site)
            if opaque:
                site.owner.unresolved.append(opaque)
            if target is None or target is site.owner:
                continue  # a recursive call tells a reader nothing about blast radius
            if target.label not in site.owner.calls:
                site.owner.calls.append(target.label)
            if site.owner.label not in target.callers:
                target.callers.append(site.owner.label)
    for blocks in cmap.files.values():
        for block in blocks:
            block.calls.sort()
            block.callers.sort()
            block.unresolved = _unique(sorted(block.unresolved))


def _resolve(cmap: CodeMap, found: _File, site: _Site) -> tuple[Block | None, str]:
    """The block a call lands on, and the text to report when only the run can say."""
    if site.text:
        return None, site.text
    bound = found.bound.get(id(site.owner), set())
    if site.attr:
        if site.base in ("self", "cls"):
            owner_class = site.owner.name.split(".")[0] if site.owner.kind == "method" else ""
            target = _in_file(found, f"{owner_class}.{site.attr}") if owner_class else None
            return target, "" if target else f"{site.base}.{site.attr}:{site.line}"
        if site.base in found.imports:
            return cmap.block(f"{found.imports[site.base]}.{site.attr}"), ""
        # An attribute call on a local object (path.exists(), data.get()) is ordinary typed code,
        # not dispatch: naming it would bury the cases below, which are the ones a reader needs.
        return None, ""
    target = _in_file(found, site.name)
    if target is not None:
        return target, ""
    if site.name in found.imports:
        return cmap.block(found.imports[site.name]), ""
    if site.name in bound:
        return (
            None,
            f"{site.name}:{site.line}",
        )  # a local holding a callable: only the run knows which
    return None, ""  # a builtin, or a name this tree does not define


def _in_file(found: _File, name: str) -> Block | None:
    """A block of this file by name, nested defs included: ``inner`` finds ``outer.inner``."""
    if not name:
        return None
    for block in found.blocks:
        if block.name == name:
            return block
    for block in found.blocks:
        if block.name.rsplit(".", 1)[-1] == name:
            return block
    return None


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = dict.fromkeys(values)
    return list(seen)
