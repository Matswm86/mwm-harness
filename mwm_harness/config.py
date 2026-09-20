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

REPO_MODELS = Path(__file__).resolve().parent.parent / "models.toml"


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
    family: str = ""
    context_window: int = 128_000
    soft_budget: int = 100_000
    thinking_field: str = ""
    echo_reasoning: bool = False
    parallel_tool_calls: bool = True
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


@dataclass
class Settings:
    """Harness behaviour. Read from ``settings.toml`` in the config directory."""

    default_model: str = "qwen3.8-max"
    permission_mode: str = "default"
    sandbox: str = "auto"  # auto | bwrap | off
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
    hooks_may_approve: bool = False  # a PreToolUse "allow" skips the approval prompt
    tool_output_cap: int = 30_000
    mcp_enabled: bool = True
    mcp_files: list[str] = field(default_factory=list)  # default: .mcp.json of cwd and workspace
    mcp_allow: list[str] = field(default_factory=list)  # tool-name patterns that run unasked
    web_allow_private: bool = False  # WebFetch may reach loopback and private addresses
    skill_dirs: list[str] = field(default_factory=list)  # "folder" or "prefix=folder"
    agent_models: dict[str, str] = field(default_factory=dict)  # e.g. opus = "qwen3.8-max"
    auto_compact: bool = True  # summarise the history when the soft budget is reached


def load_settings(path: Path | None = None) -> Settings:
    path = path or config_dir() / "settings.toml"
    settings = Settings()
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if key not in Settings.__dataclass_fields__:
                raise ConfigError(f"settings.toml: unknown key {key}")
            setattr(settings, key, value)
    return settings
