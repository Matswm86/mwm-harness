"""Secret stripping for tool output, before the model, the panel or the transcript see it.

The text rules are a Python port of Shrike's ``packages/redact`` (github.com/SumantxD/shrike, commit
012a082, MIT License, Copyright (c) 2026 Sumant Tirkey). Shrike's order is kept: connection-URL
passwords and Bearer/Basic tokens, then values whose key names a secret (quoted, then unquoted),
then known key shapes, then high-entropy values. Added here: the literal values of secret-named
entries in the harness's ``secrets.env``, the tab folder's ``.env*`` files and the workspace's
``projects/*/.env``, plus key shapes for Groq and Hugging Face.

Replacement is ``[redacted:<key or shape>]``, so the text stays readable and the redaction can
be counted. Nothing here logs or returns a secret value.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from .config import load_secrets, workspace_root

FLAGS = re.ASCII  # JavaScript's \w, \d, \s and \b are ASCII-only; keep the same meaning
KEY = (
    # "auth" but not "author"; "ssn" but not "className" (Shrike). Changed for source code: "password",
    # "passwd", "passphrase" and a whole-word "pass", never "first_pass"/"passage"/"bypass"; no plural
    # "tokens"/"sessions"/"secrets" and no "tokenize".
    r"pass(?:word|phrase)|passwd|(?<![A-Za-z_])pass(?![A-Za-z_])|secret(?!s)|token(?!s|iz)|api[-_]?key"
    r"|auth(?!or(?!i[zs]))|session(?!s)|credential|private[-_]?key|card|cvv|(?<!cla)ssn"
)
KEY_RE = re.compile(KEY, re.I | FLAGS)
NUMERIC_KEY_RE = re.compile(r"card|cvv|ssn", re.I | FLAGS)
IDENT = rf"[\w$.-]{{0,40}}(?:{KEY})[\w$.-]{{0,40}}"  # bounded so long identifier runs stay linear

SHAPES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"-----BEGIN ([A-Z0-9 ]{1,40})-----[\s\S]{0,10000}?-----END \1-----", FLAGS),
        "pem",
    ),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", FLAGS), "aws_key"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}", FLAGS), "anthropic_key"),
    (
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}", FLAGS),
        "openai_key",
    ),  # also Alibaba DashScope sk-<hex>
    (re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{10,}", FLAGS), "stripe_key"),
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})", FLAGS),
        "github_token",
    ),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}", FLAGS), "slack_token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}", FLAGS), "google_key"),
    (re.compile(r"\beyJ[\w-]{5,}\.[\w-]{5,}\.[\w-]{5,}", FLAGS), "jwt"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}", FLAGS), "groq_key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}", FLAGS), "huggingface_token"),
]


def _quoted(q: str) -> re.Pattern[str]:
    opener = "['\"]" if q == "`" else re.escape(q)
    return re.compile(
        rf"(?<![\w$.-])({opener}?)({IDENT})\1([ \t]*[:=][ \t]*){q}((?:[^{q}\\\n]|\\.)*){q}",
        re.I | FLAGS,
    )


# Shrike also treats backticks as quotes; in markdown and tool output they mark code spans
# (commit hashes, file names), so they are left out here.
QUOTED = [_quoted(q) for q in ('"', "'")]
UNQUOTED = re.compile(
    rf"(?<![\w$.-])([\"']?)({IDENT})\1([ \t]*[:=][ \t]*)(?![\"'`\[{{(]|(?:Bearer|Basic)\s)([A-Za-z0-9_\-+/.~%]+)",
    re.I | FLAGS,
)
LITERAL = re.compile(r"^(?:true|false|null|undefined|NaN|None|True|False|-?\d+(?:\.\d+)?)$")
CONNECTION = re.compile(r"\b([a-z][a-z0-9+.-]{0,30}://)([^:/\s@'\"]+):([^@\s'\"/]+)@", re.I | FLAGS)
AUTH_HEADER = re.compile(r"\b(Bearer|Basic)\s+(?!\[redacted)[A-Za-z0-9._~+/=-]{8,}", FLAGS)
QUOTED_RUN = re.compile(r"([\"'])([A-Za-z0-9+/_=.-]{24,512})\1", FLAGS)
VALUE_RUN = re.compile(r"([=:]\s*)([A-Za-z0-9+/_=.-]{24,512})(?=$|[\s&;,)}\]])", FLAGS)
URL_HOST_BEFORE = re.compile(r"//[^\s/?#@]+$", FLAGS)
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
TOKEN_RUN = re.compile(r"^[A-Za-z0-9_-]{32,512}$", FLAGS)
LABEL = re.compile(r"\[redacted:([^\]]+)\]")
LITERAL_MIN = 8  # shorter .env values ("dev", "5432") would mask ordinary words everywhere


def label(name: str) -> str:
    return f"[redacted:{name}]"


def entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values()) if s else 0.0


WORD = re.compile(r"[A-Z]?[a-z]+\d*|[A-Z]+\d*|\d+[A-Za-z]{0,3}|[A-Za-z]\d+[A-Za-z]?", FLAGS)


def wordy(v: str) -> bool:
    """A name built from words: ``neo4j_euler_drift_2``, ``Qwen3-30B-A3B-GGUF``, ``core/cip/cell_v2.py``.

    Random tokens that happen to hold ``-`` or ``_`` have mixed-case chunks (``q8Zr3L``), so they fail.
    """
    parts = [p for p in re.split(r"[-_./]", v) if p]
    return len(parts) >= 3 and sum(not WORD.fullmatch(p) for p in parts) <= 1


def looks_secret(v: str) -> bool:
    """Digit + mixed case + entropy keeps words, paths, SQL and hex ids out (Shrike's rule)."""
    return (
        not wordy(v)
        and len(v) >= 24
        and not v.startswith("/")
        and any(c.isdigit() for c in v)
        and any(c.islower() for c in v)
        and any(c.isupper() for c in v)
        and entropy(v) > 4
    )


def looks_token(v: str) -> bool:
    """A quoted letters-and-digits run (hex or base64url session id); UUIDs stay visible as record ids."""
    return (
        not wordy(v)
        and bool(TOKEN_RUN.match(v))
        and any(c.isdigit() for c in v)
        and any(c.isalpha() for c in v)
        and not UUID.match(v)
        and entropy(v) > 3
    )


def _connection(m: re.Match[str]) -> str:
    scheme, user, password = m.groups()
    return (
        m.group(0) if password.startswith("[redacted") else f"{scheme}{user}:{label('password')}@"
    )


POINTER = re.compile(
    r"[-_](?:file|path|dir|env|name|filter|count|len|limit|ttl|seconds|timeout|budget|cap|max|min|url)$",
    re.I,
)


def names_a_secret(key: str) -> bool:
    """For a dotted key (``session.settings.mode``) only the last part counts; ``TOKEN_FILE`` names a place."""
    return not POINTER.search(key) and (
        "." not in key or bool(KEY_RE.search(key.rsplit(".", 1)[-1]))
    )


def _quoted_value(m: re.Match[str]) -> str:
    q, key, sep, value = m.groups()
    if not names_a_secret(key) or re.fullmatch(r"[A-Z][A-Z_]*[A-Z]", value):  # an env var NAME
        return m.group(0)
    if key.lower() == "session" and re.fullmatch(r"[A-Za-z][A-Za-z ]*", value):
        return m.group(0)  # "session": "London" is a trading session, not a session id
    if (
        not value or value.startswith("[redacted") or value.lower() == key.lower()
    ):  # {"pass": "pass"}
        return m.group(0)
    quote = m.group(0)[-1]
    return f"{q}{key}{q}{sep}{quote}{label(key)}{quote}"


PY_NAMES = frozenset(
    [
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "object",
        "type",
        "None",
        "self",
        "cls",
        "Any",
        "Path",
        "Optional",
        "Callable",
        "Iterable",
        "Iterator",
        "Sequence",
        "Mapping",
    ]
)
CODE_AFTER = re.compile(
    r"[(\[{\"'`]|\s*(?:==|!=|<=|>=|[<>+*/%|&-]|\b(?:or|and|if|else|for|in|is|not)\b)", FLAGS
)


def looks_like_code(key: str, sep: str, value: str, after: str) -> bool:
    """An unquoted value that is a Python name, not a secret: a type, a call, an attribute, an expression.

    Only digit-free identifier values qualify, so ``token=abc123`` or ``password=p%40ss`` never do.
    """
    if re.match(r"[(\[.]", after) or re.fullmatch(r"%[-+ #0-9.]*[sdifr]", value):
        return True  # a call, a subscript or attribute, a %-format placeholder
    if sep.endswith(" ") and ":" in sep and re.match(r" +[A-Za-z(`]", after):
        return True  # "this session: c2acb3e study + dataset" is prose, not a key and its value
    if value.startswith(("/", "~/")):
        return True  # a path
    if re.fullmatch(r"[a-z_][a-z0-9_]*", value) and re.match(r"\s*[,)}]", after):
        return True  # {"token": api_key, ...} passes a variable
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_.]*", value):
        return False
    last = re.split(r"[.]", key)[-1].lstrip("_").lower()
    return (
        "." in value  # attribute access
        or value in PY_NAMES  # a type annotation
        or value.lower() == last  # secrets=secrets, self._token = token
        or bool(re.fullmatch(r"[A-Z][A-Z_]+", value) and len(value) > 1)  # a constant
        or bool(
            re.fullmatch(r"\s+=\s+", sep)
        )  # "x = y" with spaces is an assignment; .env has none
        or bool(CODE_AFTER.match(after))
        or bool(re.match(r" [A-Za-z]", after))  # "session: one turn ..." is prose
    )


def _unquoted_value(m: re.Match[str]) -> str:
    q, key, sep, value = m.groups()
    if not names_a_secret(key) or looks_like_code(key, sep, value, m.string[m.end() : m.end() + 8]):
        return m.group(0)
    if LITERAL.match(value) and not (NUMERIC_KEY_RE.search(key) and re.fullmatch(r"\d{3,}", value)):
        return m.group(0)
    return f"{q}{key}{q}{sep}" + (f'"{label(key)}"' if q == '"' else label(key))


def _quoted_run(m: re.Match[str]) -> str:
    q, v = m.groups()
    return f"{q}{label('secret')}{q}" if looks_secret(v) or looks_token(v) else m.group(0)


def _value_run(m: re.Match[str]) -> str:
    sep, v = m.groups()
    before = m.string[max(0, m.start() - 260) : m.start()]
    if looks_secret(v) and not v.startswith("//") and not URL_HOST_BEFORE.search(before):
        return f"{sep}{label('secret')}"  # a URL's scheme or port colon is not a key
    return m.group(0)


def redact_text(s: str, literals: Iterable[tuple[str, str]] = ()) -> str:
    """Shrike's rules in Shrike's order; ``literals`` are (name, value) pairs masked first."""
    if not s:
        return s
    out = s
    for name, value in literals:
        if value in out:
            out = out.replace(value, label(name))
    out = CONNECTION.sub(_connection, out)
    out = AUTH_HEADER.sub(lambda m: f"{m.group(1)} {label('token')}", out)
    for pattern in QUOTED:
        out = pattern.sub(_quoted_value, out)
    out = UNQUOTED.sub(_unquoted_value, out)
    for pattern, name in SHAPES:
        out = pattern.sub(label(name), out)
    out = QUOTED_RUN.sub(_quoted_run, out)
    return VALUE_RUN.sub(_value_run, out)


def count_labels(before: str, after: str) -> Counter[str]:
    """How many of each ``[redacted:x]`` the redaction added (labels already in the input are not counted)."""
    return Counter(LABEL.findall(after)) - Counter(LABEL.findall(before))


def _env_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return pairs
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pairs.append((key.strip(), value.strip().strip("'\"")))
    return pairs


def secret_literals(
    cwd: Path, extra_files: Iterable[Path] = (), secrets: dict[str, str] | None = None
) -> list[tuple[str, str]]:
    """(name, value) of every secret-looking entry the harness can find, longest value first.

    An entry counts when its name matches KEY (``*_API_KEY``, ``*_TOKEN``, ``*PASSWORD*``...) or its
    value looks like a key by the entropy rules. Values under LITERAL_MIN characters are skipped.
    """
    files = [
        *sorted(cwd.glob(".env*")),
        *sorted((workspace_root() / "projects").glob("*/.env")),
        *extra_files,
    ]
    pairs = list((secrets if secrets is not None else load_secrets()).items())
    for path in files:
        if path.is_file():
            pairs += _env_pairs(path)
    found: dict[str, str] = {}
    for name, value in pairs:
        if len(value) < LITERAL_MIN or value in found:
            continue
        if KEY_RE.search(name) or looks_secret(value) or looks_token(value):
            found[value] = name.lower()
    return sorted(((n, v) for v, n in found.items()), key=lambda p: -len(p[1]))


class Redactor:
    """One per session: the literal list is read once, at session start."""

    def __init__(
        self,
        cwd: Path,
        extra_patterns: Iterable[str] = (),
        literals: list[tuple[str, str]] | None = None,
    ):
        self.literals = literals if literals is not None else secret_literals(cwd)
        self.extra = [re.compile(p) for p in extra_patterns]
        self.total = 0

    def __call__(self, text: str) -> tuple[str, Counter[str]]:
        out = redact_text(text, self.literals)
        for pattern in self.extra:
            out = pattern.sub(label("pattern"), out)
        counts = count_labels(text, out)
        self.total += sum(counts.values())
        return out, counts


def describe(counts: Counter[str]) -> str:
    """Notice text: names and numbers only, never a value."""
    parts = ", ".join(f"{name}: {n}" for name, n in counts.most_common())
    return f"redacted {sum(counts.values())} secret{'s' if sum(counts.values()) != 1 else ''} ({parts})"


WITHHELD = "[redaction failed: tool output withheld]"
RedactFn = Callable[[str], tuple[str, Counter[str]]]
