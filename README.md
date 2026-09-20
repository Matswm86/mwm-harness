# MWM Harness

A personal agent harness for Qwen, Kimi and GLM models: a terminal prompt with
slash commands first, a local browser window with panels (files, code and diffs,
tasks, plan, token meter) later. One core library, thin front-ends.

> **Work in progress.** Phases 1 to 6 are built and tested against a scripted
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

Phases 1 to 6 are built; phase 0 (the live spike) still waits on an API key.
Every test runs without a network: a scripted provider plays the model, a fake
stdio server plays MCP, a mocked transport plays the web.

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
- `mcp_client.py`: reads `.mcp.json` as Claude Code writes it (project, workspace,
  then `mcp.json` in the config folder) and offers each server tool to the model
  as `mcp__server__tool`. Two transports, no SDK dependency: stdio (one JSON-RPC
  message per line) and streamable HTTP (JSON or event-stream answers, session
  header). A server starts on its first call; the tool listing comes from a cache
  keyed on the server's configuration and its script's modification time. The
  per-server `timeout` of `.mcp.json` is honoured, a timeout or a cancel sends
  `notifications/cancelled`. Image results are saved to the scratch folder and
  named in the result. A tool marked `readOnlyHint` runs unasked; every other MCP
  tool asks, unless a pattern in `mcp_allow` covers it. The deny list still wins.
- `tools/web.py`: `WebFetch` (HTML to text with headings, lists and link targets;
  5 MB cap; each redirect hop re-checked; loopback and private addresses refused
  unless `web_allow_private = true`) and `WebSearch` (DuckDuckGo's HTML page, or
  the Brave API when `BRAVE_API_KEY` is set).
- `skills.py`: `SKILL.md` folders from `.claude/skills` (project, workspace, home)
  and from `skill_dirs`; the system prompt lists names and descriptions, the
  `Skill` tool loads a body. Command files (`.claude/commands/**/*.md`) become
  slash commands with `$ARGUMENTS` and `$1`..`$9` filled in.
- `repl/terminal.py`, `cli.py`: the `mwm` command. Slash commands: `/help /model
  /models /context /usage /mode /permissions /tasks /hooks /memory /mcp /skills
  /commands /plan /files /open /resume /clear /quit`, plus one per command file and per skill.
  `mwm -p "question"` runs one headless turn; `--no-mcp` and `--no-hooks` exist.
- `web/server.py`, `web/static/index.html`: the browser panel, `mwm --web`. One
  page, one websocket, the same `Session` as the terminal: chat with streamed
  text, tool cards, a status bar (model, permission mode, context share, tokens
  in, out and cached, request count), the task list ticking live, the last plan,
  and approval dialogs (`y`, `a`, `n`; Esc stops a turn). Slash commands work in
  the message box. The server binds to 127.0.0.1, refuses any `Host` header but
  its own, refuses a websocket from another `Origin`, and needs a per-launch
  token that travels in the address fragment, so it never reaches a log or a
  `Referer`. A page that opens late gets the history and any open approval.
  The API does not expose the plan's credit balance, so the bar counts requests.
- `preview.py` and the panel's file views: a file tree of the project, tabs of
  open files, a list of files the tools touched, and a read-only code viewer
  (Monaco 0.56.0 from jsdelivr; without a network the page falls back to plain
  text). Before a `Write` or `Edit` is approved, its result is computed without
  touching the disk and shown as a side-by-side diff; the terminal prompt prints
  the same change as a unified diff. Browsing answers only for paths inside the
  project: `..`, absolute paths and symlinks that lead out are refused.
- Plan mode (`/plan`, `--mode plan`): only reading tools run; the model hands in
  its plan with `ExitPlanMode`, and your yes returns to the mode you came from.
- `spike/panel_demo.py`: the panel with a scripted model, for a look without a key.
- `spike/real_mcp_check.py`: starts every configured MCP server, lists its tools,
  and sends one `search_knowledge` call through a real session.
- `spike/real_hooks_check.py`: runs a scripted turn through the hooks installed
  on this machine and reports which of them fired and blocked.

Not built yet: subagents and compaction (phase 7).

## Use

```bash
.venv/bin/mwm                     # interactive, in the current directory
.venv/bin/mwm --mode acceptEdits  # file edits inside the project run without asking
.venv/bin/mwm --resume last       # continue the newest session of this directory
.venv/bin/mwm --web               # browser panel on 127.0.0.1:8765 (needs the web extra)
.venv/bin/mwm -p "summarise README.md"
```

Optional files in `~/.config/mwm-harness/`: `settings.toml` (default model,
permission mode, sandbox, extra writable paths, hook settings files, `mcp_allow`,
`mcp_files`, `skill_dirs`, `web_allow_private`),
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
