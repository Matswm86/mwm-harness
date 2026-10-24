# MWM Harness

A personal agent harness for Qwen, Kimi and GLM models on any OpenAI-compatible
endpoint, in the mould of Claude Code and Qwen Code: a coding agent on models of
your choosing, behind permissions, a sandbox and hooks you control. One core
library, two thin front-ends on the same session. `mwm --open` shows the
browser panel (files, code and diffs, tasks, plan, checks, token meter); `mwm`
alone is a terminal prompt with the same slash commands. Work in progress: run
against local models only so far.

## What it does

- Runs a coding-agent loop with file, shell, web fetch and web search tools,
  behind four permission modes, a hard deny list and a bubblewrap shell sandbox
  that has no network unless a call asks for it.
- Reuses what you already have for Claude Code: hook scripts, `.mcp.json` MCP
  servers, `SKILL.md` skills, command files and subagent files.
- Strips secrets (keys, tokens, `.env` values) from every tool result before
  the model, the panel or the transcript see it.
- Shows the work in a local browser panel: the code as the model writes it, a
  diff before each approval, lint and type-check marks after each edit, a
  view of which changed lines a test ran, tabs, voice input and a STOP ALL
  button.
- Puts back the files of the last turn with `/undo`, and says which harness
  layer failed in a session with `/postmortem`.
- Compacts and resumes long sessions and runs subagents on a per-agent model.
- Ships reasoning playbooks, a critic agent and an eval of 52 cases
  (`mwm-eval`).

## How it works

A Python core library (3.12 or newer) holds the message model, the turn loop,
the tools, permissions, hooks and the MCP client. The terminal prompt and the
browser panel are thin front-ends that listen to one event bus and call the
same session. A model is reached through an OpenAI-compatible streaming
`/chat/completions` endpoint with `httpx`; history is kept as Anthropic-style
content blocks in a JSON-lines transcript that `--resume` rebuilds from. Shell
commands run inside bubblewrap when it is installed. The panel is a FastAPI
and uvicorn server with one page and one websocket bound to 127.0.0.1, with
Monaco for the code view and three.js for the scene. Every test runs offline:
a scripted provider plays the model and a fake stdio server plays MCP.

## Status

Version 0.1.0, work in progress, nothing stable. Phases 1 to 7 of the roadmap
below are built. The first runs against a live model happened on a local 4B
model behind Ollama: it fixed a bug through Read, Edit and Bash behind approval
prompts. The three hosted target models have not been run yet, and the
terminal front-end is still the plain prompt that phase 8 is to rebuild.

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
| 8 | Terminal front-end to the standard of Claude Code and Qwen Code: answers streamed as rendered markdown, a spinner with time and tokens, a framed input with history and slash completion, tool cards, coloured diff approvals, a status line | A bug-fix task reads like it does in those tools; every slash command works; the core and its tests are untouched |

Where each phase stands: 1 to 6 are closed. 0 is half closed: six local models
passed the parallel tool-call spike (`COMPAT.md`), the hosted models wait on a
key. 2 closed on a local model. 7 is built, its closing tests need a hosted
model. 8 has not started. The panel renders only paragraphs, fenced code,
inline code and bold in answers: no headings, lists or links yet.

## State

Every test runs without a network: a scripted provider plays the model, a fake
stdio server plays MCP, a mocked transport plays the web.

- `messages.py`, `transcript.py`: history and the session file use Anthropic-style
  content blocks (`text`, `tool_use`, `tool_result`), one JSON object per line.
  Conversion to the OpenAI wire format happens only for the outgoing request.
  `--resume` rebuilds a session from that file and repairs unanswered tool calls.
- `providers.py`, `streaming.py`: streaming client for any OpenAI-compatible
  `/chat/completions` endpoint; tool-call fragments are reassembled by index.
- `loop.py`: the turn state machine. Cancel keeps the partial answer and kills
  the running command's whole process group. Before each request it stat's the
  files this session read or wrote, and any file whose mtime moved goes into one
  reminder at the tail of that request ("these files changed on disk since you
  last read them"). The reminder is request-only: it is never appended to the
  history or written to the transcript, so it cannot disturb the cached prefix or
  a `/resume`, and a session's own write re-stamps the file so the writer is never
  reminded about itself. Each session keeps its own stamps, which is what makes
  two panel tabs on one folder warn each other; a subagent inherits a copy of its
  parent's stamps, so a child's read cannot silence the parent's warning. A file
  that vanished is named once as `(deleted)`.
- `tools/`: `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `TodoWrite`. Names and
  argument names match Claude Code's, so existing hook matchers keep working.
- `permissions.py`: four modes (`default`, `acceptEdits`, `bypassPermissions`,
  `plan`) and a hard deny list that no mode and no hook can lift (force-push, skipping
  commit hooks by flag or by `core.hooksPath`, recursive delete of `/` or home
  with `rm` or `find`, a decoded command piped into a shell, vector-database
  deletes, broker orders, a hand-curated notes folder). Broker and delete tools
  of MCP servers are matched on what the tool does, so renaming a server does
  not lift the rule. Reading is free inside the project and its read roots
  (the workspace, the scratch folder, `extra_writable`, `extra_readable`);
  anywhere else the person is asked. Key material (`~/.ssh`, `secrets.env`,
  `.aws`, `.gnupg`, `.netrc`, git and gh credentials) is denied for Read, Edit
  and Bash in every mode.
- `sandbox.py`: shell commands run inside bubblewrap when it is available: the
  filesystem is read-only outside the project and the session scratch folder,
  and an empty folder is mounted over `~/.ssh`, `~/.gnupg`, `~/.aws` and the
  harness's own config. `/tmp` gets its own writable tmpfs (mode 1777, empty at
  start), because a read-only `/tmp` breaks every tool that calls `mktemp`,
  the `detect-secrets` pre-commit hook included; host `/tmp` contents stay
  invisible. Every sandboxed run also gets resource caps (plan item 36): the
  command tree lives in a transient user systemd scope with `TasksMax` and
  `MemoryMax`, `/tmp` is mounted with a size, one file a command may grow is
  capped by `RLIMIT_FSIZE`, and the session's own Write and Edit calls share a
  total disk budget. A run stopped by a cap reports which cap in the tool
  result instead of a bare crash; without a user bus the scope caps cannot be
  armed and the session says so at start. `sandbox = "off"` drops the four
  sandbox caps; the session-write budget stays. Each command also gets an empty network namespace (no
  internet, no host services) unless the call sets `network: true`. What that
  flag costs depends on `shell_network`: `ask` asks the person, except in
  `bypassPermissions`, which is by definition the mode where nobody is watching
  (a headless run's approver refuses everything, so an ask there was a silent
  failure, not a pause); `allow` falls through to the normal permission
  decision; `deny` refuses in every mode. A command's
  environment carries no variable whose name looks like a secret
  (`bash_env_keep` exempts one); MCP servers keep their own keys and lose the
  model keys. When bubblewrap is missing the session says so at start instead
  of running bare in silence.
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
  A model whose context window is under `defer_mcp_tools_below` (65,536 tokens by
  default, so every local entry of `models.toml`) gets only the MCP tool names,
  listed in the description of the `ToolSearch` tool, and loads a definition with
  it when it wants to call one; a loaded tool stays in the request for the rest
  of the session. `0` sends every definition to every model.
- `tools/web.py`: `WebFetch` (HTML to text with headings, lists and link targets;
  5 MB cap; each redirect hop re-checked; loopback and private addresses refused
  unless `web_allow_private = true`) and `WebSearch` (DuckDuckGo's HTML page, or
  the Brave API when `BRAVE_API_KEY` is set).
- `skills.py`: `SKILL.md` folders from `.claude/skills` (project, workspace, home)
  and from `skill_dirs`; the system prompt lists names and descriptions, the
  `Skill` tool loads a body. Command files (`.claude/commands/**/*.md`) become
  slash commands with `$ARGUMENTS` and `$1`..`$9` filled in.
- `repl/terminal.py`, `cli.py`: the `mwm` command. Slash commands: `/help /model
  /models /context /usage /mode /think /permissions /tasks /hooks /memory /mcp /skills
  /commands /plan /files /open /agents /jev /postmortem /undo /hw /check /runit
  /trace /diagnose /review /review-send /review-remember /compact /init /resume /tab
  /clear /quit`, plus one per command file and per skill.
  A model's reasoning text (the `reasoning_content` or `reasoning` field of the
  stream) prints dimmed as it arrives; `/think off` hides it. Every printed
  block carries its local time, and every event and message a UTC stamp.
  `mwm -p "question"` runs one headless turn; `--no-mcp` and `--no-hooks` exist.
- `web/server.py`, `web/static/index.html`: the browser panel, `mwm --web`, in
  the dark command-center layout. One page, one websocket, the same `Session` as
  the terminal. A status strip (model, permission mode, context share, tokens in,
  out and cached, request count, shell sandbox, CPU and GPU load and temperature,
  Oslo and New York clocks, the trading session, Topstep's 16:10 ET flatten
  countdown and a STOP ALL button); a left rail with agents, finished-turn
  results, check results, the task list, the last plan, MCP servers, this
  machine's sensors, touched files, the file tree, the session and the newest
  handoffs; a three.js scene in the middle (the agent as a sun, MCP servers and
  subagents in orbit, a pulse per MCP call, the file being written as a
  satellite, the Oslo trading day as the outer ring); a code pane under it that
  types each Edit and Write as the model writes it (the call's arguments
  stream in) and keeps a result card per finished turn; the chat on the right
  with approval cards (`y`, `a`, `n`; Esc stops a turn). Above the chat, an MNQ
  candle chart (5 or 1 minute, Oslo time, refreshed every 20 s) from TopstepX's
  read-only bars endpoint, with Yahoo NQ=F as a fallback marked delayed;
  credentials come from `PROJECT_X_USERNAME` / `PROJECT_X_API_KEY` or the file
  in the `chart_env_file` setting and never reach the page. Background jobs, the
  other machines (docker, timers, VPS) and MASTER_TODO have a place in the
  layout and no data source yet.
  Voice: the mic button (Alt+M) records in the browser and the panel server
  transcribes on this machine with faster-whisper (`pip install -e .[voice]`;
  the medium model on a CUDA GPU, else the small one on the CPU; settings
  `voice_model`, `voice_device`, `voice_language`). The text lands in the message
  box for checking; "send what I say" sends it at once, "read answers aloud" uses
  the browser's speech voice. "slash plan" at the start becomes `/plan`.
  Mods: F1-F9 and a button bar run one action each from `mods.toml` (config folder,
  else `mwm_harness/mods.default.toml`): `shell` through the Bash tool and its
  approvals (`confirm = true` always asks), `agent` on the last answer or the
  uncommitted diff (a result card, not part of the conversation), `command`
  (`/compact`, a command file) or `prompt`. Templates `{tab.changed}`,
  `{tab.repo}`, `{prompt}`; a mod that cannot run says why on its greyed button.
  Tabs: the new-tab dialog takes a folder, a model and "isolated" (a git worktree
  on its own branch under `worktree_dir`, left on disk when the tab closes);
  double-click renames; Ctrl+Shift+T reopens the last closed tab from its
  transcript; a tab switched to mid-turn replays the running turn; tabs on the
  same local model queue with a note; `/tab` works in the panel and the terminal. Slash commands work in
  the message box. The server binds to 127.0.0.1, refuses any `Host` header but
  its own, refuses a websocket from another `Origin`, and needs a per-launch
  token that travels in the address fragment, so it never reaches a log or a
  `Referer`. A page that opens late gets the history and any open approval.
  The API does not expose the plan's credit balance, so the bar counts requests.
  The `+` button (Ctrl+T) opens another session in the same window, up to
  `max_tabs` (6). Each tab has its own transcript, model, permission mode, task
  list and approvals, a mark shows which tab is busy or waiting for an answer,
  and the tabs share the MCP server processes of the first one. The last tab
  cannot be closed.
- `preview.py` and the panel's file views: a file tree of the project, tabs of
  open files, a list of files the tools touched, and a code viewer
  (Monaco 0.56.0 from jsdelivr, or from the machine itself after
  `mwm --vendor-monaco`, which downloads it, three.js 0.149.0 and the three
  fonts once, checks each npm tarball against a pinned sha512 and lets the panel
  drop the CDN from its content security policy; with neither, the page falls
  back to plain text and system fonts). Before a `Write` or `Edit` is approved, its result is computed without
  touching the disk and shown as a diff in the code pane; the terminal prompt prints
  the same change as a unified diff. A file opened from the tree is also
  editable: Ctrl+S saves it, and that save asks for approval the same way the
  agent's own `Write` does, so the hard deny list and plan mode cover it and
  `acceptEdits` skips the question as it does for the agent. Answering "always"
  there stops the asking about your saves and does not let the agent's writes
  through. A save is not a read, so the session's next request names the file as
  changed on disk. An unsaved edit marks its tab, closing one asks first, and a
  file that changes underneath an unsaved edit keeps your buffer and says so.
  Browsing and saving answer only for paths inside
  the project: `..`, absolute paths and symlinks that lead out are refused.
- `agents.py`: subagents. `agents/<name>.md` files (frontmatter `name`,
  `description`, `tools`, `model`, then the agent's system prompt) load from
  `.claude/agents` (project, workspace, home). The `Task` tool runs one as a child
  session: own history and transcript, only the tools its file lists (patterns
  such as `mcp__server__*` work), never `Task` itself, the parent's permission
  rules, approver and tool hooks, no Stop hooks. Only the final report returns.
  `agent_models` in `settings.toml` maps a file's `model:` word to a configured
  model (`opus = "qwen3.8-max"`); an unmapped word runs on the parent's model and
  says so. Stopping the parent turn stops the agent and kills its command.
- Compaction: at the model's soft budget (or on `/compact [focus]`) the model
  writes a summary that replaces the history; the `PostCompact` hooks fire with
  the summary; `--resume` starts from the last summary. A failed summary keeps
  the history. `auto_compact = false` switches the automatic part off.
- Large tool results (ObservationPack, from SoL-Pi, arXiv 2609.20519): a result
  over 10 KiB goes to the model in full on the next two requests; later requests
  carry its first and last 512 characters and the path of a scratch file with
  the exact text, which the model reads back with `Read`. The history and the
  transcript keep the full text. `observation_pack_limit = 0` switches it off.
- Plan mode (`/plan`, `--mode plan`): only reading tools run; the model hands in
  its plan with `ExitPlanMode`, and your yes returns to the mode you came from.
- `spike/panel_demo.py`: the panel with a scripted model, for a look without a key.
- `spike/real_agents_check.py`: loads the agent, skill and command files of this
  machine and lists tool names no harness tool or MCP server provides.
- `spike/real_mcp_check.py`: starts every configured MCP server, lists its tools,
  and sends one `search_knowledge` call through a real session.
- `spike/real_hooks_check.py`: runs a scripted turn through the hooks installed
  on this machine and reports which of them fired and blocked.

- `jev.py`, the `Judge` tool, `mwm-jev`: a second opinion from Jev, a model that
  returns typed decisions (a yes/no probability, one option of a closed set, a
  level on a scale) and no text. It is one extra vote beside other evidence,
  never the decider. Every call goes to an append-only log; the real result is
  recorded later and `mwm-jev report` gives the hit rate per domain and label,
  with a Brier score for yes/no questions. The tool asks for approval because
  the text it judges leaves the machine. Key: `TYPESAFE_API_KEY`. Which model
  answers (`model = auto`, the default) was set by a bake-off on 2026-10-10: the
  routing question (one choice over five knowledge-base collections) goes to a
  local fine-tuned 4B model in Ollama (`MWM_JEV_ROUTE_MODEL` names another; an
  empty value sends routing to Jev); every other question goes to Jev. When
  Jev's confidence on a passage-relevance or an answer-check question is under a
  measured cut-off, Sonnet answers instead, through the workspace's `claude -p`
  wrapper; `MWM_JEV_ESCALATE=0` turns that off. A call that fails on the local
  model or on Jev is logged and retried on Cloudflare's Clef (Workers AI) when
  `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` are set (a failed local call
  with no Clef goes to Jev); a caller can also ask for Clef by name.
- Loop guards found by the first live runs: a turn ends after 60 model requests
  (`max_requests_per_turn`); compaction keeps the prompt size right after a
  summary as a floor and waits for half a budget of growth, because a prompt
  whose fixed part is over the budget used to compact after every tool call; an
  `Edit` that cannot apply returns its error without asking the person first.
- Two more brakes on a model that loops, both off at 0: the 3rd identical tool
  call in a row is not run and the turn ends (`max_identical_calls`, default 3);
  an `Edit` or `Write` is refused after 4 edit-then-`Bash` cycles in a row whose
  run failed with no `Read`, `Grep`, `Glob`, web or `Task` call in between
  (`max_blind_fix_cycles`, default 4). A passing run or any such look resets the
  count.
- `redact.py`: secret stripping. Every tool result and every hook note is
  scanned before the model, the panel or the transcript see it. Passwords in
  connection URLs, bearer tokens, values whose key names a secret, known key
  shapes (cloud, Git hosting, chat, payment and model-provider keys, PEM blocks,
  JWTs) and high-entropy values become `[redacted:<key or shape>]`; the values
  of secret-named entries in `secrets.env`, in the project's `.env*` files and in
  the workspace's per-project `.env` files are masked wherever they appear. The
  text rules are a Python port of the redaction package of Shrike (MIT). When
  the redactor itself fails the output is withheld, not passed on.
  `redact = false` switches it off, `redact_extra_patterns` adds regexes.
- `checks.py`, `runit.py`, `trace_plugin.py`: after a turn that wrote files, the
  check chain of `checks.default.toml` (py_compile, ruff, mypy, `node --check`,
  `bash -n`, shellcheck, gdlint) runs on those files with no model call. A
  repo's `.mwm/checks.toml`, else the config folder's `checks.toml`, replaces the
  chain; a missing tool is reported as not run, never as a pass;
  `checks_after_turn = false` switches the automatic run off and `/check` runs
  it by hand. `/runit [pytest args]` runs the tests under coverage with one
  context per test and marks each changed line hit (with the tests that ran it)
  or miss; `/trace TEST` runs one test and lists the calls it made into the repo
  with their argument values. The panel shows the same in its Checks rail and
  the code pane (the "run it" button).
- `review.py`, `codemap.py`: the Review Desk. After a turn that wrote files, a
  second reader on another model (`review_model`, default the `critic` entry of
  `models.toml`) reads that turn's diff and answers in strict JSON: one
  plain-English line per changed file, plus comments with a severity, a line and
  a fix. It advises: nothing it says is applied to a file, and a comment the
  person acts on becomes their own next turn (`/review-send N`) or a standing
  note in `.mwm/review-learnings.md` (`/review-remember N`), whose 60 newest
  lines every later review of that repo is told. In a git repo that file goes
  into `.git/info/exclude`, so a public repo never publishes it. A review is
  skipped, and says so, when another reviewer holds the workspace lock, when the
  diff is empty or withheld, or when a headless run would have to call a
  subscription-plan endpoint; a skip is never shown as a review. The panel shows
  all of it in the Review tab beside Code and Result. `codemap.py` answers "who
  calls this": a static map of every function in the tree, built with the
  standard library's `ast`, so each changed file also lists the repo functions
  that call the ones its diff touched. A call whose target only exists at run
  time (`getattr`, a dict entry, a local holding a callable, a method the class
  does not define) is listed as unresolved rather than guessed. The same map, one
  line per function, goes into the system prompt of a model whose window is under
  `code_map_below` (65,536 by default), so a small local model can name the file
  worth reading instead of reading whole files; it is cut at 24,000 characters.
- `diagnose.py`: failure diagnosis. When the check chain or a test run fails after a turn that
  wrote files, one call turns that failure into a short report: where it points, why, and the order
  to check in. The location is found without a model: Python traceback frames, `pytest` short-test
  summary rows and check findings are read for their file and line, newest frame first, and only
  positions inside this repo are kept (a standard-library frame is dropped). A critic model
  (`diagnose_model`, defaulting to `review_model`) then reads that location plus the turn's diff and
  answers in strict JSON with `root_cause`, `suspected_files` and up to three `guidance` steps. A
  file the critic names that neither the failure nor the diff mentions is dropped and counted, never
  shown as a suspect. The report goes to the next request as a meta message, and an edit to a file
  outside the suspected list in that turn shows one notice: advice, never a block. `/diagnose` reads
  the last failing run, `/diagnose <pasted output>` reads what you paste, and the panel's run-it
  button diagnoses a failing suite on its own. `diagnose_rounds` (3) caps how many diagnoses one
  task gets before the harness stops and asks you instead; `diagnose_after_fail = false` switches
  the automatic pass off. With no critic configured it still ships the location and says the reason
  is missing; a critic that answers nothing usable is a line in the report, not a failed turn, and a
  headless run never calls a subscription-plan endpoint for it. Diffs that failed redaction are not
  sent, so a diagnosis can leak a secret no more than the check chain can.
- `undo.py`: `/undo` (the panel's "undo turn" button) puts back the files the
  newest file-changing turn wrote with `Write` or `Edit`, from a copy taken
  before each write. It keeps the last 10 turns, leaves a file alone that
  changed again after the turn and says so, and names the turn's `Bash` calls,
  whose changes it cannot undo. Emergency stop: the panel's STOP ALL button
  (Alt+Shift+X, or `/stop all`) cancels every tab's turn, kills their shell
  commands, refuses open approvals and cancels running test runs.
- `hardware.py`: `/hw` and the panel's "This machine" rail panel and CPU and GPU
  chips read CPU load and temperature (`/proc/stat`, hwmon), memory, fans, disk
  and the NVIDIA GPU (`nvidia-smi`) of the machine the harness runs on. Linux
  only, no extra packages, read-only; a source that is missing shows as unknown
  with the reason, never as a guess.
- `memory_panel.py`: the panel's Handoffs rail panel lists the newest
  `handoff-*.md` files of the workspace's `memory/` folder with their top
  heading, searches memory through the workspace's `memory-archive` MCP server
  and opens a `.md` file inside that folder read-only. Nothing writes to memory.
- `postmortem.py`: `/postmortem [draft] [ID]` reads a session transcript, sorts
  every failure it records (tool errors, hook and permission blocks, Stop-gate
  feedback, cancelled turns, compactions, slow tools) into the harness layer it
  points at, and writes a report naming the kind of fix that keeps it from
  coming back. `draft` hands the report to the model to write one fix, and every
  file it writes goes through the normal approval. The panel shows how long each
  tool call took and marks those over 30 s.

Run against a live model so far: the tool-call spike and a bug-fix task, on
local models only. Phase 7 closes when a subagent on another model returns and
a long session compacts at its budget and continues.

## Reviews

The code was written by a model, so it gets reviewed like code from a stranger.
A security review of the permission, sandbox, web and panel code found five
working bypasses (any file readable without a prompt, API keys in every shell
command's environment, two deny rules lifted by renaming an MCP server, `rm`
with long options, a silent no-sandbox fallback). All five are fixed and each
is a test in `tests/test_security_review.py`. Since 2026-10-10 `WebFetch`
resolves a host once and connects to the address it checked (no DNS-rebinding
window), and shell commands run without network unless the call sets
`network: true`, which asks first unless the session runs in
`bypassPermissions` (`shell_network` in settings). Still
open: about half the code (`cli.py`, the terminal front-end, `skills.py`, the MCP
protocol handling) has had no review yet.

## Use

```bash
.venv/bin/mwm                     # interactive, in the current directory
.venv/bin/mwm --mode acceptEdits  # file edits inside the project run without asking
.venv/bin/mwm -p "task" --yolo    # one unattended turn: no prompt of any kind
.venv/bin/mwm --resume last       # continue the newest session of this directory
.venv/bin/mwm --web               # browser panel on 127.0.0.1:8765 (needs the web extra)
.venv/bin/mwm --open              # the same, shown in its own browser window
.venv/bin/mwm --vendor-monaco     # once: fetch the code viewer, 3D library and fonts
.venv/bin/mwm -p "summarise README.md"
.venv/bin/mwm-jev report          # the judge's decision log and hit rate
```

`--yolo` is the one switch for a run nobody is watching: it sets
`bypassPermissions` and turns an `ask` network policy into `allow`, so a
command that needs the internet runs instead of stopping the job. An explicit
`shell_network = "deny"` stays a denial, and the hard deny list in
`permissions.py` applies in every mode, `--yolo` included. The same two things
can be set once for every session in `settings.toml` (`permission_mode`,
`shell_network`), and the panel's mode dropdown reaches the same modes.

Exit codes of `mwm`: 2 = no API key, 3 = any other configuration mistake
(unknown model, unknown agent).

### A local model

`models.toml` carries six entries for Ollama's OpenAI-compatible endpoint: 16k
variants of qwen3-vl 4B, gemma4 e4b, Ornith-1.5-35B-A3B and Spark-X2.5-4B, and
32k variants of the last two. Any non-empty `OLLAMA_API_KEY` will do. Two things
to know: Ollama cuts a prompt that is longer than the model's context length
without saying so, and its `/v1` endpoint cannot raise that length per request,
so build a variant once:

```bash
printf 'FROM qwen3-vl:4b-instruct\nPARAMETER num_ctx 16384\n' > Modelfile
ollama create qwen3-vl-4b-instruct-16k -f Modelfile
OLLAMA_API_KEY=ollama .venv/bin/mwm --model qwen3-vl-4b-instruct-16k --no-mcp --no-hooks
```

A 4B model on a laptop GPU is slow and loops on wrong edits; it is a way to
exercise the loop, not a way to work.

Optional files in `~/.config/mwm-harness/`: `settings.toml` (default model,
permission mode, sandbox, `sandbox_tasks_max`, `sandbox_memory_max_mb`,
`sandbox_file_max_mb`, `sandbox_tmpfs_max_mb`, `session_write_cap_mb`,
extra writable and readable paths, `bash_env_keep`,
`max_requests_per_turn`, hook settings files, `mcp_allow`, `mcp_files`,
`skill_dirs`, `skill_exclude`, `web_allow_private`, `agent_models`),
`models.toml` (more models or another endpoint), `deny.toml` (extra deny rules),
`hooks.json` (hooks in the `settings.json` format), `mods.toml` (the panel's F1-F9
actions), `checks.toml` (the check chain).

## Reasoning playbooks, the critic, and the eval

A smaller model writes and debugs well and still misses what a careful reviewer
catches: a claim nobody ran, a search that came back empty for the wrong reason,
two changes credited to one. The harness carries that part as procedure.

- Four playbooks ship as skills and need no setup: `reason-predelivery` (grade
  every claim as ran / read / recalled / guessed before answering), `reason-debug`
  (reproduce, three hypotheses with a disproving check each, stop after two failed
  fixes), `reason-audit` (inventory and blast radius before judgement, findings
  with evidence, coverage count), `reason-challenge` (argue the other side, name
  the kill test, state confidence as a size). A skill of the same name in your
  own skill folders replaces the shipped one.
- The `critic` agent is a read-only second reader with a fixed five-question
  checklist and a `PASS | REVISE | BLOCK` verdict. It asks for the model tier
  `critic`; map it to another model family than the writer in `settings.toml`
  (`[agent_models]` `critic = "..."`), because a model that reviews its own work
  shares its blind spots. Unmapped, it runs on the parent model with a warning.
- `evals/cases.toml` holds 52 situations, each an ordinary request with one trap
  in its facts, all taken from real incidents and rewritten without names. Run:

```bash
.venv/bin/mwm-eval --model qwen3.8-max --arms bare,playbooks
.venv/bin/mwm-eval --model qwen3.8-max --arms critic --critic kimi-k2.7-code
.venv/bin/mwm-eval --model "cmd:some-cli -p" --judge glm-5.2   # any CLI that reads stdin
.venv/bin/mwm-eval --compare evals/out/*.jsonl
```

  Without `--judge` a regex grader runs: repeatable and offline, and crude (it
  can miss a correct reply worded unusually). With 52 cases the 95 % interval on
  a pass rate is about 13 points either side, so only large gaps mean anything.
  Single turn, no tools: it measures noticing, not agent behaviour. A model's
  reply is cut at 2000 tokens (`--max-tokens`, 0 = no cap; a `cmd:` model is not
  capped), and `--resume FILE` continues an interrupted run from its result file.

## Bug flags, prove-the-fix, and the assurance card

After any turn where the model touched files, the harness takes three looks of its
own. They are looks: the flags advise, the proof reruns, the card holds `DONE`,
nothing here edits code.

- **Bug flags** (`/flags`, `bug_flags` in settings, on by default). Four AST looks
  at the Python files this turn changed: `swallowed_exception` (an `except` that is
  only `pass`, or catches `Exception` with no raise and no log), `silent_failure`
  (an `except` returning `None`/`[]`/`{}`/`0` unlogged, the shape the workspace
  no-silent-failures rule bans), `query_in_loop` (a DB or HTTP call written inside
  a `for`/`while` body), and `parse_error` (a file that does not parse is flagged,
  never skipped). `n_plus_one` is the dynamic half: a run-it trace of one test that
  enters the same repo function 3+ times with the same SQL or URL argument. The
  turn-end pass emits a `BugFlagsFinished` event; the panel lists the hits under
  the turn's output.
- **Prove-the-fix** (`/fix`, `prove_fix_after_turn`). When a pytest run fails
  during a turn, the harness records the failing node ids (including bare
  `ERROR path.py` collection errors). If the model then edits files, the turn-end
  pass reruns exactly those tests, alone, in the repo's own Python. `proved` is
  true only when the rerun exits 0 and every recorded node is accounted for: a
  test that vanished (deleted or renamed) comes back `missing`, never silently
  `passed`, because deleting a test proves nothing. A passing test run the model
  itself claims, without a rerun on record, is stored as `by: "model"` and is not
  proof. The card's `correct` row counts a proved fix as green coverage.
- **Assurance card** (`/assurance`, `assurance_after_turn`). Six rows answer one
  question before the model may say it is done: *safe, correct, reliable,
  performant, scalable, fit for operation*. Three rows come only from code
  evidence: `safe` (secret scan, dependency audit when a deps file changed, a
  PyPI existence lookup that FAILs and names any package that does not exist,
  semgrep or bandit when installed, an OWASP LLM checklist note when the diff
  touches a model gateway), `correct` (test chain status, failing checks, a
  missing prove-the-fix, run-it line coverage), `reliable` (a silent-failure or
  swallowed-exception flag FAILs it; advice kinds only show as evidence). The
  model may fill only `performant`, `scalable` and `fit for operation`, as `na`
  or `pass` with a reason, and those rows are labelled model-written; a
  `ValueError` refuses the model touching the other three. `NOT RUN` with a
  reason is honest and does not hold; a `FAIL` or a still-empty row holds `DONE`
  (`done_allowed` false; the panel's chip reads HELD and its detail shows every
  row's status and who wrote it). Scans that need a binary use it only if
  installed (`assurance_scans` off skips them); the row then says NOT RUN and
  names what is missing. One deviation from the plan is honest and permanent:
  the PyPI lookup checks existence and reports the package's age, because the
  public JSON API no longer exposes download counts, so there is no popularity
  floor to check.

Tests: `tests/test_bugflag.py`, `tests/test_assurance.py` (37, all offline).

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

## License

MIT, see [LICENSE](LICENSE). Anyone may use, change and share the code, including in commercial products, as long as the copyright notice stays with it.
