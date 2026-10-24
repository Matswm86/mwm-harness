"""Item 25: two reading levels over one set of data, chosen by the reader and never by a model.

The harness serves two kinds of reader on the same work. Mats reads a change as "what does this
do, what could break, how do I check it myself"; an engineer reads the raw diff, the check chain,
the coverage and the trace. Today the Review Desk's payload holds both already (a walkthrough line
per file, the diff, the caller map, the comments) and shows exactly one arrangement of it.

So this module adds no content and calls no model. :func:`views` takes the review payload plus the
evidence the turn already produced (check results, the last run-it coverage, the spec's
criterion-to-test map) and returns two *rendering plans* for the same bytes: which sections to
show, whether to fold the diff, and for the Beginner plan the three plain lines per file, each one
traceable to a field it was derived from. A front-end switches level by picking the other plan, so
toggling re-renders and re-runs nothing: no provider request, no test, no check.

Nothing here guesses. A file with no walkthrough line says it has none rather than getting an
invented summary; a file no check or test touched says so, which is the same admission item 22
makes on the assurance card. Both views are produced in one pass, so they can never disagree about
which files were read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mwm_harness.review import SENSITIVE

LEVELS = ("beginner", "engineer")
DEFAULT_LEVEL = "engineer"  # what every turn looked like before item 25 existed

BEGINNER_SECTIONS = ("does", "breaks", "check")
ENGINEER_SECTIONS = ("diff", "callers", "checks", "coverage", "trace", "comments")

MAX_BREAKS = 4  # lines per file in the "what could break" block
MAX_CHECK_LINES = 3  # lines per file in the "how to check it yourself" block
MAX_NAMES = 4  # caller names written out before the count takes over
MAX_TEXT = 300  # characters of any derived line (a reviewer's text is capped by review.py)
MISSING_LINES = 8  # never-run line numbers named per file before the count takes over


def parse_level(value: object) -> tuple[str, str]:
    """The level a setting names, and why it was not taken as written ("" when it was).

    Returns ``("", reason)`` for a value that names no level: the caller reports it instead of
    quietly reading the request as the default, which is how a typo becomes an invisible setting.
    """
    text = str(value if value is not None else "").strip().lower()
    if text in ("beginner", "b", "simple", "plain"):
        return "beginner", ""
    if text in ("engineer", "e", "raw", "full", "technical"):
        return "engineer", ""
    return DEFAULT_LEVEL, (
        f"{text or 'an empty value'} names no reading level: use "
        + " or ".join(LEVELS)
        + f", so {DEFAULT_LEVEL} is in force"
    )


def level_or_default(value: object) -> str:
    """The level a setting names, ignoring an unknown value (the caller has already warned)."""
    return parse_level(value)[0]


def pick(review: Mapping[str, Any], level: object) -> dict[str, Any] | None:
    """The rendering plan a front-end should follow for one review payload.

    ``None`` means this payload carries no views, which happens to a session restored from a
    transcript written before item 25: the caller then draws the payload directly, as it always
    did. An unknown level falls back to :data:`DEFAULT_LEVEL` rather than drawing nothing.
    """
    views = review.get("views")
    if not isinstance(views, Mapping):
        return None
    wanted = level_or_default(level)
    chosen = views.get(wanted)
    if not isinstance(chosen, Mapping):
        return None
    return dict(chosen)


# ------------------------------------------------------------------ derivation


def _short(label: str) -> str:
    """A 3-part label drops its module: the file this line is about is already named."""
    parts = [part for part in str(label).split(".") if part]
    if len(parts) > 1:
        return ".".join(parts[-2:])
    return parts[0] if parts else str(label)


def _clip(text: str) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= MAX_TEXT else flat[: MAX_TEXT - 1].rstrip() + "…"


def _names(values: Sequence[str]) -> str:
    shown = ", ".join(_short(value) for value in values[:MAX_NAMES])
    if len(values) > MAX_NAMES:
        shown += f" +{len(values) - MAX_NAMES} more"
    return shown


def _matched_word(path: str) -> str:
    """Which word in the path makes the harness watch this file, for an honest warning."""
    found = SENSITIVE.search(str(path))
    return found.group(0) if found else ""


def _comment_lines(comments: Sequence[Mapping[str, Any]], path: str) -> tuple[list[str], int]:
    """The reviewer's own critical words about one file, and how many suggestions it left."""
    breaks: list[str] = []
    suggestions = 0
    for comment in comments:
        if str(comment.get("file") or "") != path:
            continue
        severity = str(comment.get("severity") or "note")
        if severity == "critical":
            text = _clip(
                f"the reviewer calls this a problem at line {comment.get('line')}: "
                f"{comment.get('text')}"
            )
            fix = str(comment.get("fix") or "").strip()
            breaks.append(f"{text} (fix it: {_clip(fix)})" if fix else text)
        elif severity == "suggestion":
            suggestions += 1
    return breaks, suggestions


def _break_lines(
    change: Mapping[str, Any],
    critical: Sequence[str],
    suggestions: int,
) -> list[str]:
    """What could break, from the fields this change actually carries.

    Every line here names its source (a caller list, a diff count, an unresolved call, a path
    word), so a reader can go and check it. A file that gives no signal says so instead of being
    padded with a generic warning.
    """
    path = str(change.get("path") or "")
    callers = [str(item) for item in change.get("callers") or ()]
    unresolved = [str(item) for item in change.get("unresolved") or ()]
    added = int(change.get("added") or 0)
    removed = int(change.get("removed") or 0)
    rows: list[str] = []
    rows.extend(list(critical)[:MAX_BREAKS])
    if callers:
        rows.append(
            f"{len(callers)} function(s) reach this file ({_names(callers)}): "
            "change its behavior and those change too"
        )
    word = _matched_word(path)
    if word:
        rows.append(
            f"its path says it is watched code ({word}): a mistake here can change what gets "
            "approved, what gets hidden, or what runs"
        )
    if removed > added:
        rows.append(
            f"it removes more than it adds ({removed} lines out, {added} in): "
            "whatever used the removed lines is the thing that breaks"
        )
    if unresolved:
        rows.append(
            f"{len(unresolved)} call(s) here are only decided while the program runs "
            f"({_names(unresolved)}): a rename can pass every check and still break at run time"
        )
    if change.get("new") and not callers:
        rows.append(
            "nothing in the repo calls this new file yet, so a mistake here only shows up "
            "once something starts using it"
        )
    if change.get("truncated"):
        rows.append("its diff is longer than shown, so part of the change is not visible here")
    if suggestions:
        rows.append(f"{suggestions} suggestion(s) from the reviewer are listed under the file")
    if not rows:
        return [
            "nothing in this file's own data points at a break: no critical comment, no caller "
            "in the map, more added than removed"
        ]
    head = rows[:MAX_BREAKS]
    if len(rows) > len(head):
        head = [*head, f"+{len(rows) - len(head)} more signal(s) in the engineer view"]
    return head


def _check_lines(
    change: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    runit: Mapping[str, Any] | None,
) -> list[str]:
    """How to check this file yourself, naming only results that already exist for it.

    The command is never invented: a check is named by the name the chain gave it, coverage by the
    numbers the last run-it measured. When neither touches the file the line admits it, which is
    the reader's cue to run one, not a pass.
    """
    path = str(change.get("path") or "")
    rows: list[str] = []
    for result in checks:
        files = [str(item) for item in result.get("files") or ()]
        if path and path not in files and not any(item.endswith(path) for item in files):
            continue
        status = str(result.get("status") or "")
        name = str(result.get("check") or "a check")
        if status == "pass":
            rows.append(f"{name} passes on it: run the repo's {name} to see the same result")
        elif status == "fail":
            findings = result.get("findings") or ()
            first = ""
            if findings:
                head = findings[0] if isinstance(findings[0], Mapping) else {}
                first = str(head.get("msg") or head.get("message") or "").strip()
            rows.append(
                f"{name} FAILS on it"
                + (f": {_clip(first)}" if first else "")
                + f" ({len(findings)} finding(s))"
            )
        elif status in ("not_run", "timeout"):
            rows.append(f"{name} did not answer here: {_clip(str(result.get('reason') or status))}")
    measured = (runit or {}).get("files") or {}
    states = measured.get(path)
    if states is None:
        states = next(
            (value for key, value in measured.items() if str(key).endswith(path)),
            None,
        )
    if isinstance(states, Sequence) and states:
        hit = sum(
            1 for state in states if isinstance(state, Mapping) and state.get("state") == "hit"
        )
        miss = sum(
            1 for state in states if isinstance(state, Mapping) and state.get("state") == "miss"
        )
        line = f"a test run touched {hit} of its changed lines and never touched {miss}"
        if miss:
            misses = [
                state.get("line")
                for state in states
                if isinstance(state, Mapping) and state.get("state") == "miss"
            ]
            shown = ", ".join(str(item) for item in misses[:MISSING_LINES])
            if len(misses) > MISSING_LINES:
                shown += f" +{len(misses) - MISSING_LINES} more"
            line += f" (never run: {shown}); read those lines, they are the untested part"
        else:
            line += "; /runit re-measures it"
        rows.append(line)
    elif runit and runit.get("status") != "ran":
        rows.append(
            "no coverage for it yet: run-it did not run "
            f"({_clip(str(runit.get('reason') or runit.get('status')))}), so /runit is the check"
        )
    if not rows:
        return [
            "no check or test result names this file from this turn, so there is nothing here to "
            "re-check: /runit runs the tests and measures which of its lines they touch"
        ]
    return rows[:MAX_CHECK_LINES] if len(rows) > MAX_CHECK_LINES else rows


def _does_line(change: Mapping[str, Any]) -> str:
    """What this file does, in the reviewer's own words when it wrote any."""
    walkthrough = _clip(str(change.get("walkthrough") or ""))
    if walkthrough:
        return walkthrough
    added = int(change.get("added") or 0)
    removed = int(change.get("removed") or 0)
    kind = "a new file" if change.get("new") else "a change"
    return (
        f"no walkthrough line for this file: the reviewer wrote none for it, so read the diff "
        f"({kind}: {added} line(s) in, {removed} out)"
    )


def _headline(review: Mapping[str, Any], level: str) -> str:
    """The one line a reader sees first at this level."""
    files = review.get("files") or ()
    if str(review.get("status") or "") != "reviewed":
        reason = _clip(str(review.get("reason") or review.get("status") or "no review"))
        return f"the review desk did not read this turn: {reason}"
    if level == "engineer":
        added = sum(int(item.get("added") or 0) for item in files)
        removed = sum(int(item.get("removed") or 0) for item in files)
        seconds = float(review.get("seconds") or 0.0)
        return (
            f"{review.get('model')} read {len(files)} file(s), "
            f"+{added}/-{removed} in {seconds:.0f}s, "
            f"{len(review.get('comments') or ())} comment(s)"
        )
    summary = _clip(str(review.get("summary") or ""))
    if summary:
        return summary
    return (
        f"the reviewer looked at {len(files)} file(s) and wrote "
        f"{len(review.get('comments') or ())} note(s) about them"
    )


def _coverage_notes(coverage: Mapping[str, Sequence[str]] | None) -> list[str]:
    """Item 23's criterion-to-test map, phrased as something the reader can go and run."""
    if not coverage:
        return []
    ids = list(coverage)
    named = [cid for cid in ids if coverage.get(cid)]
    gap = [cid for cid in ids if not coverage.get(cid)]
    rows: list[str] = []
    if named:
        first = ", ".join(f"{cid} in {coverage[cid][0]}" for cid in named[:MAX_NAMES])
        more = f" (+{len(named) - MAX_NAMES} more)" if len(named) > MAX_NAMES else ""
        rows.append(
            f"{len(named)}/{len(ids)} acceptance criteria are quoted by a test: {first}{more}"
        )
    if gap:
        extra = f" (+{len(gap) - MAX_NAMES} more)" if len(gap) > MAX_NAMES else ""
        rows.append(
            f"{len(gap)} criterion(s) have no test naming them: {', '.join(gap[:MAX_NAMES])}"
            f"{extra}. Run /spec <slug> to reread what was promised."
        )
    return rows


def _notes(
    review: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    runit: Mapping[str, Any] | None,
    coverage: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    """Turn-wide lines both levels need, kept out of the per-file blocks."""
    rows: list[str] = []
    dropped = int(review.get("dropped") or 0)
    if dropped:
        rows.append(
            f"{dropped} comment(s) were dropped: they pointed at a file or line the diff does not touch"
        )
    if review.get("learnings"):
        rows.append(f"{review['learnings']} learning(s) saved for later reviews of this repo")
    failed = [str(item.get("check") or "?") for item in checks if item.get("status") == "fail"]
    if failed:
        rows.append(f"the check chain FAILED: {', '.join(failed[:MAX_NAMES])}")
    skipped = [
        str(item.get("check") or "?")
        for item in checks
        if item.get("status") in ("not_run", "timeout")
    ]
    if skipped:
        rows.append(f"the check chain did not answer: {', '.join(skipped[:MAX_NAMES])}")
    if runit:
        if runit.get("status") == "ran":
            verdict = "tests pass" if runit.get("tests_passed") else "TESTS FAIL"
            rows.append(f"the tests as a whole: {verdict} ({_clip(str(runit.get('summary')))})")
        else:
            rows.append(
                "no test ran during this turn: "
                f"{_clip(str(runit.get('reason') or runit.get('status')))}"
            )
    rows.extend(_coverage_notes(coverage))
    return [row for row in rows if row]


def _beginner_files(
    changes: Sequence[Mapping[str, Any]],
    comments: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    runit: Mapping[str, Any] | None,
) -> list[dict]:
    rows: list[dict[str, Any]] = []
    for change in changes:
        path = str(change.get("path") or "")
        critical, suggestions = _comment_lines(comments, path)
        rows.append(
            {
                "path": path,
                "does": _does_line(change),
                "breaks": _break_lines(change, critical, suggestions),
                "check": _check_lines(change, checks, runit),
            }
        )
    return rows


def _engineer_files(changes: Sequence[Mapping[str, Any]]) -> list[dict]:
    rows: list[dict[str, Any]] = []
    for change in changes:
        rows.append(
            {
                "path": str(change.get("path") or ""),
                "added": int(change.get("added") or 0),
                "removed": int(change.get("removed") or 0),
                "new": bool(change.get("new")),
                "truncated": bool(change.get("truncated")),
                "sensitive": bool(_matched_word(str(change.get("path") or ""))),
                "callers": [str(item) for item in change.get("callers") or ()],
                "unresolved": [str(item) for item in change.get("unresolved") or ()],
            }
        )
    return rows


def views(
    review: Mapping[str, Any],
    *,
    checks: Sequence[Mapping[str, Any]] = (),
    runit: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Both reading plans for one review payload, derived in a single pass.

    ``review`` is a :meth:`mwm_harness.review.Review.as_dict` payload (or the same shape read back
    from a transcript). ``checks`` is the turn's check-chain payload, ``runit`` the last run-it
    result and ``coverage`` the spec's criterion-to-test map, all already produced by the session:
    nothing here runs a command or asks a model. Neither plan copies the diff text, only whether it
    should be folded, so a turn's event log stays one size whichever level a reader picks.
    """
    changes = [item for item in (review.get("files") or ()) if isinstance(item, Mapping)]
    comments = [item for item in (review.get("comments") or ()) if isinstance(item, Mapping)]
    notes = _notes(review, checks, runit, coverage)
    beginner = {
        "level": "beginner",
        "headline": _headline(review, "beginner"),
        "fold_diff": True,
        "sections": list(BEGINNER_SECTIONS),
        "files": _beginner_files(changes, comments, checks, runit),
        "notes": notes,
    }
    engineer = {
        "level": "engineer",
        "headline": _headline(review, "engineer"),
        "fold_diff": False,
        "sections": list(ENGINEER_SECTIONS),
        "files": _engineer_files(changes),
        "notes": notes,
    }
    return {"beginner": beginner, "engineer": engineer}
