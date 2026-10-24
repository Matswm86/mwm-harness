"""Bug flags on changed files (plan item 30): three AST looks plus one trace look.

The three static looks come from Shrike's correlator heuristics, rebuilt for Python with the
standard library's ``ast``:

- ``swallowed_exception``: an ``except`` whose body is only ``pass``/``...``/``continue``, or
  one that catches ``Exception`` (or bare) without re-raising and without any logging call.
- ``silent_failure``: an ``except`` that returns an empty value (``None``, ``[]``, ``{}``,
  ``()``, ``0``, ``False``) with no log call and no raise.
- ``query_in_loop``: a database or HTTP call (``execute``, ``fetch*``, ``query``, ``get*``,
  ``post``...) written directly inside a ``for`` or ``while`` body.

These match the workspace no-silent-failures rule; they are looks, not proofs, and they are
advice: nothing here blocks a turn. ``n_plus_one`` is the dynamic half: from a run-it trace of
one test (``runit.trace_test``), the same repo function entered 3+ times with the same
SQL/URL-shaped argument is one flag.

Unlike ``checks.py`` findings these are the harness's own code, so no external tool is needed:
a file that does not parse is reported as a ``parse_error`` flag instead of passing silently.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Argument names the run-it trace records a query/URL under, and the value shapes those
# arguments take. An argument is only compared when its name suggests data access.
SQL_NAMES = re.compile(r"^(sql|query|stmt|statement|url|path|endpoint|resource)$", re.I)
SQL_VALUE = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|PRAGMA)\b", re.I)
URL_VALUE = re.compile(r"^\s*(https?|postgres(ql)?|mysql|sqlite|mongodb(\+srv)?|redis)://")

# Attribute names that read a database or an HTTP endpoint when called inside a loop.
DATA_METHODS = {
    "execute",
    "executemany",
    "executescript",
    "fetchone",
    "fetchall",
    "fetchmany",
    "query",
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    "request",
}
# ... but only on something that reads like a client, not on a plain dict or list.
CLIENT_HINT = re.compile(r"(conn|cursor|db|session|client|api|http|requests?|self)$", re.I)

# A call whose name reads like logging keeps an except honest; these words anywhere in the
# dotted callee name count (logger.warning, log.error, self._report, print to stderr...).
LOG_CALL = re.compile(
    r"\b(log|logger|logging|warn|warning|error|exception|critical|info|debug|print|report|"
    r"anomal|notice|emit|record|audit)\b",
    re.I,
)

N_PLUS_ONE_MIN = 3  # same function, same query/URL, this many times in one traced test


@dataclass
class Flag:
    file: str  # repo-relative
    line: int
    kind: str  # swallowed_exception | silent_failure | query_in_loop | n_plus_one | parse_error
    message: str

    def label(self) -> str:
        return f"{self.file}:{self.line} {self.kind}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlagReport:
    flags: list[Flag] = field(default_factory=list)
    files: int = 0  # Python files looked at
    unparsed: int = 0  # files that did not parse (each is also a flag)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for flag in self.flags:
            out[flag.kind] = out.get(flag.kind, 0) + 1
        return dict(sorted(out.items()))

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(flag) for flag in self.flags]

    def line(self) -> str:
        if not self.flags:
            return f"bug flags: none in {self.files} changed file(s)"
        counts = ", ".join(f"{kind}:{n}" for kind, n in self.counts().items())
        return f"bug flags: {len(self.flags)} in {self.files} file(s) ({counts})"


def _is_empty_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None or node.value is False
    if isinstance(node, (ast.List, ast.Dict, ast.Tuple)):
        return not node.elts
    return False


def _raises_or_logs(handler: ast.ExceptHandler) -> bool:
    """True when the handler re-raises or calls something that reads like logging."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Attribute):
                name = callee.attr
                owner = callee.value
            elif isinstance(callee, ast.Name):
                name, owner = callee.id, callee
            else:
                continue
            owner_text = ast.dump(owner) if isinstance(owner, ast.expr) else ""
            words = f"{name} {ast.unparse(owner) if isinstance(owner, ast.expr) else ''}"
            if LOG_CALL.search(words.replace("_", " ")) or LOG_CALL.search(owner_text):
                return True
    return False


def _catches_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True  # bare except
    node = handler.type
    names = [node] if isinstance(node, ast.Tuple) else [node]
    for item in names:
        if isinstance(item, ast.Name) and item.id in {"Exception", "BaseException"}:
            return True
        if isinstance(item, ast.Attribute) and item.attr in {"Exception", "BaseException"}:
            return True
    return False


def _loops_contain(tree: ast.Module) -> set[int]:
    """Line numbers of every statement written directly inside a for/while body or an if/else
    branch of one (a query two indents deeper in the loop is still a query in the loop)."""
    inside: set[int] = set()

    def mark(body: list[ast.stmt]) -> None:
        for statement in body:
            for node in ast.walk(statement):
                if hasattr(node, "lineno"):
                    inside.add(node.lineno)

    class Walk(ast.NodeVisitor):
        def visit_For(self, node: ast.For) -> None:  # noqa: N802
            mark(node.body)
            mark(node.orelse)
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:  # noqa: N802
            mark(node.body)
            mark(node.orelse)
            self.generic_visit(node)

    Walk().visit(tree)
    return inside


def flags_for_source(source: str, path: str) -> list[Flag]:
    """All flags for one Python source string; a syntax error is itself one flag."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line = exc.lineno or 1
        return [Flag(path, line, "parse_error", f"does not parse: {exc.msg}")]
    out: list[Flag] = []
    loop_lines = _loops_contain(tree)

    for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
        body_kinds = {type(statement) for statement in handler.body}
        only_stubs = body_kinds <= {ast.Pass, ast.Continue, ast.Expr} and all(
            isinstance(statement, (ast.Pass, ast.Continue))
            or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and statement.value.value is Ellipsis
            )
            for statement in handler.body
        )
        logged = _raises_or_logs(handler)
        if only_stubs:
            out.append(
                Flag(
                    path,
                    handler.lineno,
                    "swallowed_exception",
                    "except body is only pass/.../continue: the error disappears",
                )
            )
        elif _catches_broad(handler) and not logged:
            caught = ast.unparse(handler.type) if handler.type else "bare except"
            out.append(
                Flag(
                    path,
                    handler.lineno,
                    "swallowed_exception",
                    f"catches {caught} with no raise and no log call",
                )
            )
        for statement in handler.body:
            if (
                not logged
                and isinstance(statement, ast.Return)
                and _is_empty_value(statement.value)
            ):
                out.append(
                    Flag(
                        path,
                        statement.lineno,
                        "silent_failure",
                        "except returns an empty value with no log: callers cannot tell "
                        "failure from nothing",
                    )
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.lineno not in loop_lines:
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in DATA_METHODS:
            continue
        owner = ast.unparse(node.func.value)
        tail = re.split(r"[.\[]", owner)[-1]
        if CLIENT_HINT.search(tail.lower()) or owner.lower() in DATA_METHODS:
            out.append(
                Flag(
                    path,
                    node.lineno,
                    "query_in_loop",
                    f"{owner}.{node.func.attr}() inside a loop: one round trip per item",
                )
            )
    return out


def flag_files(root: Path, files: list[Path]) -> FlagReport:
    """Look every given Python file for flags; unreadable files are skipped, unparsed flagged."""
    report = FlagReport()
    for path in files:
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            rel = path.name
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        report.files += 1
        found = flags_for_source(source, rel)
        if any(flag.kind == "parse_error" for flag in found):
            report.unparsed += 1
        report.flags.extend(found)
    report.flags.sort(key=lambda f: (f.file, f.line))
    return report


def n_plus_one(result: dict[str, Any], min_repeats: int = N_PLUS_ONE_MIN) -> list[Flag]:
    """The dynamic half: from one traced test's calls, a repo function entered 3+ times with
    the same SQL/URL-shaped argument. ``result`` is ``runit.as_dict(trace_test(...))``."""
    if result.get("status") != "ran":
        return []
    seen: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in result.get("calls", []):
        args = call.get("args") or {}
        for name, value in args.items():
            if not SQL_NAMES.match(name):
                continue
            text = str(value).strip()
            if not (SQL_VALUE.match(text) or URL_VALUE.match(text)):
                continue
            key = (str(call.get("function", "?")), text[:200])
            seen.setdefault(key, []).append(call)
    out: list[Flag] = []
    for (function, text), calls in seen.items():
        if len(calls) < min_repeats:
            continue
        first = calls[0]
        preview = text if len(text) <= 60 else text[:59] + "…"
        out.append(
            Flag(
                str(first.get("file", "?")),
                int(first.get("line", 0) or 0),
                "n_plus_one",
                f"{function} ran {len(calls)} times on the same target in this test "
                f"({preview}): N+1 access pattern",
            )
        )
    out.sort(key=lambda f: (f.file, f.line))
    return out


FAILED_NODE = re.compile(r"^(?:FAILED|ERROR) ((?:[\w./-]+\.py)(?:::[\w:.\[\]-]+)?)", re.MULTILINE)


def failed_node_ids(output: str, limit: int = 24) -> list[str]:
    """Pytest node ids of the failing tests in one run's output, for the prove-the-fix rerun.

    A whole-file collection error ("ERROR tests/test_x.py", no ``::`` part) is kept as the bare
    file: rerunning the file is the honest thing to do with it.
    """
    return list(dict.fromkeys(FAILED_NODE.findall(output)))[:limit]
