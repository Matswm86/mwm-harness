"""The typed event bus: the only way the core talks to a front-end.

The terminal REPL and the browser panel both subscribe here. The core never
prints. The one exception to "events only" is an approval, which needs an
answer back; that goes through the ``Approver`` protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Event:
    pass


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
class FilesTouched(Event):
    """Project files the tools have read or written this session, newest last."""

    paths: list[str]


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
