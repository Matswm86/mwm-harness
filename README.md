# MWM Harness

A personal agent harness for Qwen, Kimi and GLM models: a terminal prompt with
slash commands first, a local browser window with panels (files, code and diffs,
tasks, plan, token meter) later. One core library, thin front-ends.

> **Work in progress.** Phases 1 to 3 are built and tested against a scripted
> model. No run against a live model has happened yet, so nothing here is stable
> and everything may change without notice.

## Roadmap

Each phase closes on a test, not on a date.

| # | Build | Closed when |
|---|---|---|
| 0 | Spike: how each model streams tool calls on one OpenAI-compatible endpoint | 3 models each stream 2 parallel tool calls; token usage logged in `COMPAT.md` |
| 1 | Message model, session transcript, provider adapters | A recorded session replays cleanly through the transcript checks |
| 2 | Agent loop, file and shell tools, permissions, sandbox, cancel, plain terminal prompt | A model fixes a bug in a scratch repo behind approval prompts; Esc kills a long command |
| 3 | Hooks engine (scripts that run before and after tools and at end of turn), rules and memory injection | A hook can block a reply and the loop recovers |
| 4 | MCP client, web fetch and search, skills, slash-command files | Configured MCP servers list their tools and answer a call |
| 5 | Browser panels: chat, status bar with live context and token meter, tasks, plan, approval dialogs | Tasks tick live; context % equals the API usage numbers |
| 6 | File tree, open-files list, code viewer, diff before approval | Rejecting a diff leaves the file byte-identical |
| 7 | Subagents on a per-agent model, compaction, resume | A subagent on another model returns; a long session compacts and continues |

## State

Phases 1, 2 and 3 are built; phase 0 (the live spike) still waits on an API key.
Every test below runs without a network: a scripted provider plays the model.

- `messages.py`, `transcript.py`: history and the session file use Anthropic-style
  content blocks (`text`, `tool_use`, `tool_result`), one JSON object per line.
  Conversion to the OpenAI wire format happens only for the outgoing request.
  `--resume` rebuilds a session from that file and repairs unanswered tool calls.
- `providers.py`, `streaming.py`: streaming client for any OpenAI-compatible
  `/chat/completions` endpoint; tool-call fragments are reassembled by index.
- `loop.py`: the turn state machine. Cancel keeps the partial answer and kills
  the running command's whole process group.
- `tools/`: `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `TodoWrite`. Names and
  argument names match Claude Code's, so existing hook matchers keep working.
- `permissions.py`: three modes (`default`, `acceptEdits`, `bypassPermissions`)
  and a hard deny list that no mode and no hook can lift (force-push, skipping
  commit hooks, recursive delete of `/` or home, vector-database deletes, broker
  orders, a hand-curated notes folder).
- `sandbox.py`: shell commands run inside bubblewrap when it is available: the
  filesystem is read-only outside the project and the session scratch folder.
- `hooks.py`: reads Claude-Code-style `settings.json` files and runs their hook
  commands with the same stdin payload and the same answers (exit 2, `decision:
  block`, `additionalContext`, `updatedInput`, `permissionDecision`, PermissionRequest
  `behavior`, `async`). Eight events: SessionStart, UserPromptSubmit, PreToolUse,
  PostToolUse, PermissionRequest, Stop, PostCompact, SessionEnd. A Stop hook that
  blocks sends its reason back to the model; `stop_hook_active` is set on the retry.
  A PreToolUse `allow` does NOT skip the approval prompt unless
  `hooks_may_approve = true` is set, because a command-rewriting hook answers
  `allow` for everything it rewrites.
- `context.py`: system prompt from the rules files it finds (`CLAUDE.md`,
  `RULES.md`, `AGENTS.md`, with `@file.md` imports) plus the workspace memory
  index; the context meter reads the endpoint's own usage numbers.
- `repl/terminal.py`, `cli.py`: the `mwm` command. Slash commands: `/help /model
  /models /context /usage /mode /permissions /tasks /hooks /memory /resume /clear
  /quit`. `mwm -p "question"` runs one headless turn.
- `spike/real_hooks_check.py`: runs a scripted turn through the hooks installed
  on this machine and reports which of them fired and blocked.

Not built yet: MCP client, web tools, skills, subagents, compaction, the browser
panels (phases 4 to 7).

## Use

```bash
.venv/bin/mwm                     # interactive, in the current directory
.venv/bin/mwm --mode acceptEdits  # file edits inside the project run without asking
.venv/bin/mwm --resume last       # continue the newest session of this directory
.venv/bin/mwm -p "summarise README.md"
```

Optional files in `~/.config/mwm-harness/`: `settings.toml` (default model,
permission mode, sandbox, extra writable paths, hook settings files),
`models.toml` (more models or another endpoint), `deny.toml` (extra deny rules),
`hooks.json` (hooks in the `settings.json` format).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -q
```

## API key

The key never goes in this repo. Put it in `~/.config/mwm-harness/secrets.env`:

```bash
mkdir -p ~/.config/mwm-harness
printf 'MWM_HARNESS_API_KEY=%s\n' 'PASTE-KEY-HERE' > ~/.config/mwm-harness/secrets.env
chmod 600 ~/.config/mwm-harness/secrets.env
```

## Run the spike

```bash
.venv/bin/python spike/spike_toolcalls.py
```

It exits 2 when the key is missing or is a hostname, 1 when any model request
fails, 0 when every model answered. Raw chunks land in `spike/out/` (ignored by
git).
