"""The typed event bus: the only way the core talks to a front-end.

The terminal REPL and the browser panel both subscribe here. The core never
prints. The one exception to "events only" is an approval, which needs an
answer back; that goes through the ``Approver`` protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol


def now_stamp() -> str:
    """UTC time to the millisecond, the stamp every event and message carries."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class Event:
    ts: str = field(default_factory=now_stamp, kw_only=True)


@dataclass
class TurnStarted(Event):
    prompt: str


@dataclass
class TextDelta(Event):
    text: str


@dataclass
class ReasoningDelta(Event):
    text: str


@dataclass
class ToolArgsDelta(Event):
    """A piece of a tool call's JSON arguments, sent while the model is still writing them.

    ``index`` names the call within this model reply (parallel calls interleave);
    ``call_id`` and ``name`` are empty until the provider has sent them.
    """

    index: int
    call_id: str
    name: str
    fragment: str


@dataclass
class ToolStarted(Event):
    tool_use_id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolFinished(Event):
    tool_use_id: str
    name: str
    content: str
    is_error: bool


@dataclass
class HookContext(Event):
    """Text a hook injected into the model's context."""

    event: str
    text: str


@dataclass
class HookBlocked(Event):
    event: str
    reason: str


@dataclass
class Notice(Event):
    """Something the person should see that is not model output."""

    level: str  # info | warn | error
    text: str


@dataclass
class UsageUpdated(Event):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    context_window: int
    soft_budget: int

    @property
    def context_fraction(self) -> float:
        """Share of the window the next request starts with: last prompt plus last answer."""
        used = self.prompt_tokens + self.completion_tokens
        return used / self.context_window if self.context_window else 0.0


@dataclass
class TodosUpdated(Event):
    todos: list[dict[str, Any]]


@dataclass
class SubagentStarted(Event):
    agent: str
    model: str
    prompt: str


@dataclass
class SubagentFinished(Event):
    agent: str
    reason: str
    tool_calls: int


@dataclass
class Compacted(Event):
    trigger: str  # manual | auto
    summary: str
    messages_before: int


@dataclass
class ToolResultsPruned(Event):
    """Step 1 of compaction: old tool output replaced by one-line stubs.

    No model call was made. ``tokens_saved`` is the harness's own character estimate
    (``compaction.CHARS_PER_TOKEN``), never an endpoint figure, and it is what decides
    whether step 2 runs at all.
    """

    trigger: str  # turn | auto
    stubbed: int
    chars_saved: int
    tokens_saved: int


@dataclass
class FilesTouched(Event):
    """Project files the tools have read or written this session, newest last."""

    paths: list[str]


@dataclass
class ChecksStarted(Event):
    files: list[str]


@dataclass
class ChecksFinished(Event):
    """Compile, lint and type-check results for the files a turn changed (``checks.py``)."""

    root: str
    results: list[dict[str, Any]]


@dataclass
class RunItFinished(Event):
    """Changed-line coverage from ``runit.changed_coverage``: which new lines a test ran."""

    root: str
    result: dict[str, Any]


@dataclass
class TraceFinished(Event):
    """The calls one test made into the repo, with argument values (``runit.trace_test``)."""

    root: str
    result: dict[str, Any]


@dataclass
class ReviewStarted(Event):
    """The Review Desk's second reader is reading this turn's diff (``review.py``)."""

    files: list[str]
    model: str


@dataclass
class ReviewFinished(Event):
    """One review: the change stack, a walkthrough line per file and the reviewer's comments.

    ``review["status"]`` is ``reviewed``, ``skipped`` (with the reason) or ``failed``; a skip
    is never presented as a review. Comments are advice: nothing here was applied to a file.
    """

    root: str
    review: dict[str, Any]


@dataclass
class DiagnoseStarted(Event):
    """The failure diagnosis is reading one failing run (``diagnose.py``)."""

    source: str
    model: str
    round: int


@dataclass
class DiagnoseFinished(Event):
    """One diagnosis: where the failure points, and what the critic says is wrong.

    ``diagnosis["status"]`` is ``diagnosed`` or ``skipped`` (with the reason: nothing inside the
    repo to point at, or no failing run to read). The suspected files are advice: no file was
    opened, edited or reverted because of them.
    """

    root: str
    diagnosis: dict[str, Any]


@dataclass
class ReviewComment(Event):
    """A reviewer comment the person acted on: sent on as their next turn, or kept for later."""

    file: str
    line: int
    text: str
    remember: bool = False


@dataclass
class BugFlagsFinished(Event):
    """Item 30: the harness's own looks over the files a turn changed (``bugflag.py``).

    Flags are advice: nothing was edited, blocked or reverted because of them.
    """

    root: str
    report: dict[str, Any]


@dataclass
class FixProved(Event):
    """Item 30 (c): the tests that failed were rerun alone after an editing turn.

    ``result["proved"]`` is True only when every previously failing node id passed alone.
    """

    root: str
    result: dict[str, Any]


@dataclass
class AssuranceFinished(Event):
    """Item 22: the assurance card for the turn's changes, one row per quality.

    ``card["done_allowed"]`` is the gate: a FAIL or an unfilled row holds the turn back from
    being presented as done. Rows the model filled carry ``model_written``.
    """

    root: str
    card: dict[str, Any]


@dataclass
class LessonsFinished(Event):
    """Item 5: the session-end lessons pass wrote candidate lines to the review file.

    ``result["status"]`` is recorded, quiet, skipped or failed, and a skip or failure always
    carries ``reason``: a lessons pass that silently did nothing would look exactly like one
    that found nothing to say. Candidates only: nothing became a rule because of this event.
    """

    result: dict[str, Any]


@dataclass
class SpecStarted(Event):
    """Item 23: one draft call for one step of the spec walk.

    ``step`` is ``requirements``, ``design`` or ``tasks``. Nothing is on disk yet when this fires:
    the draft still has to pass its validator and the person's approval card.
    """

    slug: str
    step: str
    model: str


@dataclass
class SpecFinished(Event):
    """Item 23: the outcome of one ``/spec`` walk, one step, or one status read.

    ``spec["steps"]`` carries each step with its status and, for a refused one, the validator
    errors that named what was missing. ``spec["uncovered"]`` lists the acceptance criteria no
    test quotes, which is what the assurance card's correct row fails on.
    """

    root: str
    spec: dict[str, Any]


@dataclass
class AdvisorStarted(Event):
    """Item 24: the advisor on a timer is reading a digest of this session.

    Fires at a tool-call boundary, never mid-call. ``minutes`` and ``calls`` say which
    trigger landed; the note it produces is advice and gates nothing.
    """

    model: str
    minutes: int
    calls: int


@dataclass
class AdvisorFinished(Event):
    """Item 24: one advisor pass. ``advice["status"]`` is advised, quiet, skipped or failed.

    ``advice["note"]`` is the labelled advice the working model reads on its next request;
    ``prompt_tokens`` and ``completion_tokens`` are the advisor's own cost, shown because a
    background timer that bills invisibly is the one surprise this feature must not hide.
    """

    advice: dict[str, Any]


@dataclass
class GoalStarted(Event):
    """Item 3: one /goal loop took hold: a person's end state bound to one check command.

    ``model`` is the checker's id and ``baseline_exit`` the check's exit code on the untouched
    repo. A check that passed there never starts a loop (item 7), so a start always carries a
    failing or timed-out baseline; ``None`` here means the baseline itself timed out.
    """

    goal: str
    check: str
    model: str
    max_turns: int
    baseline_exit: int | None


@dataclass
class GoalAttempt(Event):
    """One worker turn of the goal loop and what the check said after it.

    ``verdict`` is "" when the check itself failed and no checker was called at all; otherwise
    it is the checker's word (done | not-done) with its sentence in ``reason``. ``exit_code``
    None means the check timed out, which is a failure, not a pass.
    """

    turn: int
    exit_code: int | None
    verdict: str
    reason: str


@dataclass
class GoalFinished(Event):
    """One /goal run ended: done, capped, spend, brake, cancelled, checker_failed or off.

    ``report`` carries attempts, model requests spent, the last failure tail and the state
    file when the loop can still be picked up. Only a done loop deletes its file; every other
    status leaves it so a new session in this project can say what is unfinished.
    """

    root: str
    report: dict[str, Any]


@dataclass
class PlanProposed(Event):
    plan: str


@dataclass
class PlanResolved(Event):
    approved: bool


@dataclass
class ModeChanged(Event):
    mode: str


@dataclass
class EffortChanged(Event):
    effort: str


@dataclass
class ModelChanged(Event):
    model: str


@dataclass
class TurnEnded(Event):
    reason: str  # done | cancelled | error | stop_blocks_exhausted
    text: str = ""


Listener = Callable[[Event], None]


@dataclass
class EventBus:
    _listeners: list[Listener] = field(default_factory=list)

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, event: Event) -> None:
        for listener in self._listeners:
            listener(event)


class Approver(Protocol):
    async def ask(self, tool_name: str, tool_input: dict[str, Any], reason: str) -> bool: ...


class DenyAll:
    """The approver for headless runs: anything that needs a person is refused."""

    async def ask(self, tool_name: str, tool_input: dict[str, Any], reason: str) -> bool:
        return False
