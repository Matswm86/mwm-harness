"""Model registry, secrets and harness settings.

``models.toml`` ships with the repo; a file of the same name in the config
directory overrides or adds entries, so a model can point at another endpoint
or key without a code change.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_MODELS = Path(__file__).resolve().parent.parent / "models.toml"

SUBSCRIPTION_ENDPOINT_MARKER = "token-plan."

# Reasoning effort tiers, lowest first. The qwen3.8 family's ladder stops at xhigh,
# so a configured max is clamped there rather than sent and refused.
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(Exception):
    """Something the person running the harness has to fix. ``mwm`` exits with ``code``."""

    code = 3


class MissingKey(ConfigError):
    """No usable API key. Exit 2, so a script can tell it from a mistyped model or agent."""

    code = 2


def config_dir() -> Path:
    return Path(os.environ.get("MWM_HARNESS_CONFIG", Path.home() / ".config" / "mwm-harness"))


def workspace_root() -> Path:
    """The workspace that owns the rules, memory index and project hooks."""
    return Path(os.environ.get("MWM_WORKSPACE", Path.home() / "MWM"))


@dataclass
class ModelSpec:
    id: str
    base_url: str
    key_env: str
    # The model name sent on the wire. Empty means "use id". Needed when two
    # registry entries share one upstream model but differ in endpoint and key,
    # e.g. qwen3.8-flash on the Token Plan (interactive only) and the same model
    # on pay-as-you-go (script-permitted). TOML keys must be unique, so the
    # registry id carries a prefix the API would reject.
    api_model: str = ""
    family: str = ""
    context_window: int = 128_000
    soft_budget: int = 100_000
    thinking_field: str = ""
    echo_reasoning: bool = False
    parallel_tool_calls: bool = True
    # True only when the endpoint was measured taking an ``image_url`` part AND
    # reading the picture correctly (spike/spike_vision.py, recorded in COMPAT.md).
    # A text-only model that is sent an image fails the whole request, so images
    # are replaced by a note naming the file unless this is set.
    vision: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)


def load_models(paths: list[Path] | None = None) -> dict[str, ModelSpec]:
    paths = paths or [REPO_MODELS, config_dir() / "models.toml"]
    defaults: dict[str, Any] = {}
    raw_models: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        defaults.update(data.get("defaults") or {})
        for model_id, entry in (data.get("models") or {}).items():
            raw_models.setdefault(model_id, {}).update(entry)
    known = set(ModelSpec.__dataclass_fields__)
    models: dict[str, ModelSpec] = {}
    for model_id, entry in raw_models.items():
        merged = {**defaults, **entry}
        unknown = set(merged) - known
        if unknown:
            raise ConfigError(f"models.toml: unknown keys for {model_id}: {sorted(unknown)}")
        if not merged.get("base_url") or not merged.get("key_env"):
            raise ConfigError(f"models.toml: {model_id} needs base_url and key_env")
        models[model_id] = ModelSpec(id=model_id, **merged)
    if not models:
        raise ConfigError("no models configured")
    return models


def load_secrets(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE lines from secrets.env. The environment wins over the file."""
    path = path or config_dir() / "secrets.env"
    secrets: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip().strip("'\"")
    return secrets


def api_key_for(spec: ModelSpec, secrets: dict[str, str] | None = None) -> str:
    key = os.environ.get(spec.key_env) or (secrets if secrets is not None else load_secrets()).get(
        spec.key_env, ""
    )
    if not key:
        raise MissingKey(
            f"no API key: set {spec.key_env} in the environment or in "
            f"{config_dir() / 'secrets.env'} (mode 600)"
        )
    if key.endswith(".com") or "://" in key:
        raise MissingKey(f"{spec.key_env} holds a hostname or URL, not an API key")
    return key


def is_subscription_endpoint(base_url: str) -> bool:
    """True when ``base_url`` is a subscription plan endpoint, not a pay-as-you-go one.

    The Token Plan is licensed for interactive use inside a coding or agent tool
    only. Its terms ban automation scripts, custom application backends and
    non-interactive batch calls, and name subscription suspension or key banning
    as the penalty. A headless ``mwm -p`` turn is a script call, so it runs on a
    pay-as-you-go or local model; the interactive REPL and the agents it spawns
    stay on the plan, because the terms treat calls initiated interactively
    within a tool as normal use.
    """
    return SUBSCRIPTION_ENDPOINT_MARKER in urlparse(base_url).netloc.lower()


@dataclass
class Settings:
    """Harness behaviour. Read from ``settings.toml`` in the config directory."""

    default_model: str = "qwen3.8-max"
    # How hard reasoning models think before answering, sent as reasoning_effort on
    # the qwen3.8 family and ignored elsewhere. "" leaves the endpoint's own default.
    # /effort changes it for one session; thinking tokens are output tokens, so this
    # is the main per-turn cost dial.
    reasoning_effort: str = "medium"
    permission_mode: str = "default"
    sandbox: str = "auto"  # auto | bwrap | off
    # Resource caps for sandboxed shell commands (plan item 36). An unattended --yolo run
    # has no human to notice a job filling the disk or forking without end, and wall clock
    # and output size were the only caps. 0 switches a cap off. sandbox = "off" drops the
    # four caps below (they live in the sandbox); the session-write cap is a tool-layer
    # guard and stays on. The process and memory caps need a user systemd scope (probed
    # at session start, reported when absent); the /tmp and per-file caps are enforced by
    # bwrap and rlimits alone.
    sandbox_tasks_max: int = 512  # processes one Bash call may run at once (cgroup pids.max)
    sandbox_memory_max_mb: int = 16384  # RAM one Bash call may hold (cgroup memory.max)
    sandbox_file_max_mb: int = 2048  # RLIMIT_FSIZE: one file a sandboxed command may grow
    sandbox_tmpfs_max_mb: int = 2048  # size of the private /tmp tmpfs
    # Total bytes a session's Write and Edit calls may put on disk. Bash-written bytes are
    # covered per-file and in /tmp by the caps above, but no cheap and honest counter exists
    # for many small files written by a shell loop, so the session cap covers what the
    # tools themselves write. 0 switches it off.
    session_write_cap_mb: int = 4096
    # Bash calls with network: true. ask = always ask the person, in every mode (default);
    # allow = the normal permission decision; deny = refused. Without it, no network.
    shell_network: str = "ask"
    # Names from sandbox.PRIVATE_DIRS to leave visible inside the sandbox instead of
    # hiding under an empty tmpfs. They stay read-only, because the root bind is. Name
    # one only when a sandboxed command must authenticate with it: ".config/gh" lets
    # git push reach GitHub through gh's credential helper. The secret-files deny rule
    # still refuses a command that names the folder or prints the token.
    sandbox_private_readable: list[str] = field(default_factory=list)
    extra_writable: list[str] = field(default_factory=list)
    # Folders outside the project that reading tools may open without asking. The
    # workspace root, the scratch folder and extra_writable are always included.
    extra_readable: list[str] = field(default_factory=list)
    # Environment variables a Bash command may see although their name looks like a secret.
    bash_env_keep: list[str] = field(default_factory=list)
    hook_settings: list[str] = field(default_factory=list)
    hook_timeout: float = 60.0
    max_stop_blocks: int = 3
    # Model requests one turn may make before the harness ends it. A weak model
    # can repeat one tool call forever; 0 switches the cap off.
    max_requests_per_turn: int = 60
    # The same tool call (name and input) this many times in a row ends the turn
    # without running it. Seen live 09-20: a 4B model sent one stale Edit 14 times
    # in a row. 0 switches the brake off.
    max_identical_calls: int = 3
    # An Edit or Write is refused after this many edit-then-Bash cycles in a row whose
    # Bash run failed with no Read, Grep, Glob, web or Task call in between. It catches
    # a fix loop that alternates different edits and failing runs, which
    # max_identical_calls cannot see. A successful Bash run or any look resets it. 0 = off.
    max_blind_fix_cycles: int = 4
    hooks_may_approve: bool = False  # a PreToolUse "allow" skips the approval prompt
    tool_output_cap: int = 30_000
    mcp_enabled: bool = True
    mcp_files: list[str] = field(default_factory=list)  # default: .mcp.json of cwd and workspace
    mcp_allow: list[str] = field(default_factory=list)  # tool-name patterns that run unasked
    # Models with a context window below this get MCP tool names only, plus ToolSearch to
    # load a definition when needed. 0 sends every definition to every model.
    defer_mcp_tools_below: int = 65_536
    # A model whose window is under this also gets a one-line-per-function map of the repo in
    # its system prompt, so it can name the file worth reading instead of reading whole files.
    # The map is cut at codemap.MAX_TEXT_CHARS. 0 sends no map to any model.
    code_map_below: int = 65_536
    max_tabs: int = 6  # sessions the browser panel holds open at once
    worktree_dir: str = "~/.local/share/mwm-harness/worktrees"  # where isolated panel tabs live
    chart_env_file: str = ""  # .env with PROJECT_X_USERNAME/PROJECT_X_API_KEY for the panel chart
    voice_model: str = "medium"  # faster-whisper model for the panel mic button
    voice_device: str = "auto"  # auto (CUDA, else the small model on CPU) | cuda | cpu
    voice_language: str = ""  # "" detects the language per clip; "no" or "en" forces one
    web_allow_private: bool = False  # WebFetch may reach loopback and private addresses
    skill_dirs: list[str] = field(default_factory=list)  # "folder" or "prefix=folder"
    # Skill names never offered to the model, e.g. skills whose license bars use outside
    # the product they ship with. A name matches with or without its "prefix:".
    skill_exclude: list[str] = field(default_factory=list)
    agent_models: dict[str, str] = field(default_factory=dict)  # e.g. opus = "qwen3.8-max"
    # Run the check chain (checks.default.toml) on the files a turn changed, before it ends.
    checks_after_turn: bool = True
    # Review Desk: after a turn that wrote files, a second reader on another model reads its diff
    # and leaves comments. It never edits anything; the person sends or remembers each comment.
    review_after_turn: bool = True
    review_model: str = (
        "critic"  # a models.toml id or an agent_models key; "" = this session's model
    )
    # A folder of PID-stamped ".active-<pid>" locks. A live lock means another reviewer already
    # runs, so this one is skipped and says so; a skip is never reported as a review.
    review_lock_dir: str = ""
    # Failure diagnosis (plan item 31): after a turn whose checks or test run failed, one call
    # names where the failure points in this repo and, with a critic model, why. The answer goes
    # to the next request as a meta note; nothing it says is applied to a file.
    diagnose_after_fail: bool = True
    # "" = the same model the Review Desk uses (review_model), so a critic from another family
    # reads the failure. A model id or an agent_models key overrides that.
    diagnose_model: str = ""
    # Diagnoses one task may ask for before the harness stops and names the failure to the
    # person instead of spending more requests on it. 0 means no cap.
    diagnose_rounds: int = 3
    # Bug flags (plan item 30): after a turn that wrote Python, the harness's own AST looks
    # (swallowed exception, silent failure, query in a loop) run over the changed files. They
    # are advice and feed the assurance card's reliable row; nothing is blocked by a flag.
    bug_flags: bool = True
    # Prove-the-fix (plan item 30 (c)): when a turn edited files after tests failed, exactly
    # those failing node ids are rerun alone and the PASS/FAIL lands beside the turn.
    prove_fix_after_turn: bool = True
    # Assurance card (plan item 22): after a turn that wrote files, the six quality rows are
    # filled from the evidence this session has (checks, run-it, bug flags, scans). Building
    # it runs no tests and no model: rows with no evidence say NOT RUN, never pass.
    assurance_after_turn: bool = True
    # The external scans the safe row may run when a turn touched dependencies or code:
    # detect-secrets/bandit/pip-audit/PyPI existence of newly added requirement names. A slow
    # box or an offline session can switch them off; the row then says NOT RUN honestly.
    assurance_scans: bool = True
    auto_compact: bool = True  # summarise the history when the soft budget is reached
    # Tool results over this many characters are sent in full for two requests, then as an
    # excerpt plus the path of a scratch file holding the exact text. 0 switches it off.
    observation_pack_limit: int = 10 * 1024
    # Strip secrets (keys, tokens, passwords, .env values) from every tool result and hook note
    # before the model, the panel or the transcript see it. Extra regexes are masked too.
    redact: bool = True
    redact_extra_patterns: list[str] = field(default_factory=list)


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_dir() / "settings.toml"
    settings = Settings()
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if key not in Settings.__dataclass_fields__:
                raise ConfigError(f"settings.toml: unknown key {key}")
            setattr(settings, key, value)
    if settings.shell_network not in ("ask", "allow", "deny"):
        raise ConfigError(
            f"settings.toml: shell_network must be ask, allow or deny, got {settings.shell_network!r}"
        )
    if settings.reasoning_effort not in ("", *EFFORTS):
        raise ConfigError(
            f"settings.toml: reasoning_effort must be one of {', '.join(EFFORTS)} or empty, "
            f"got {settings.reasoning_effort!r}"
        )
    return settings
