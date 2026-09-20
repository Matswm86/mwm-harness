"""The ``mwm`` command.

mwm                        interactive session in the current directory
mwm -p "question"          one headless turn; prints the answer, exit 0 on success
mwm --resume last          continue the newest session of this directory
mwm -p "task" --agent NAME run one subagent headless and print its report (for scripts and hooks)
mwm --web                  the same session in a local browser window (127.0.0.1 only)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mwm_harness import __version__
from mwm_harness.config import (
    ConfigError,
    api_key_for,
    config_dir,
    load_models,
    load_secrets,
    load_settings,
)
from mwm_harness.loop import Session
from mwm_harness.mcp_client import McpError, McpManager, default_mcp_files, load_server_configs
from mwm_harness.permissions import MODES
from mwm_harness.providers import OpenAICompatProvider
from mwm_harness.repl.terminal import Printer, TerminalApprover, repl, resolve_session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mwm", description="MWM Harness")
    parser.add_argument("-p", "--print", dest="prompt", help="run one headless turn and exit")
    parser.add_argument("--model", help="model id from models.toml")
    parser.add_argument("--mode", choices=MODES, help="permission mode")
    parser.add_argument("--resume", metavar="ID", help="session id prefix, or 'last'")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="project directory")
    parser.add_argument("--no-hooks", action="store_true", help="run without any hooks")
    parser.add_argument("--no-mcp", action="store_true", help="run without MCP servers")
    parser.add_argument(
        "--agent", metavar="NAME", help="with -p: run this subagent, print its report"
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PATTERN",
        help="tool-name pattern that runs unasked (repeatable); the deny list still applies",
    )
    parser.add_argument("--web", action="store_true", help="serve the browser panel")
    parser.add_argument("--port", type=int, default=8765, help="panel port (default 8765)")
    parser.add_argument(
        "--open",
        action="store_true",
        help="start the panel and show it in its own browser window (implies --web)",
    )
    parser.add_argument(
        "--vendor-monaco",
        action="store_true",
        help="download the panel's code viewer once (sha512-checked) for offline use",
    )
    parser.add_argument("--version", action="version", version=f"mwm-harness {__version__}")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.mode:
        settings.permission_mode = args.mode
    settings.mcp_allow = [*settings.mcp_allow, *args.allow]
    if args.prompt == "-":
        args.prompt = sys.stdin.read().strip()  # a hook pipes its prompt in
    if args.agent and not args.prompt:
        raise ConfigError("--agent needs -p with the task for the agent")
    models = load_models()
    model_id = args.model or settings.default_model
    if model_id not in models:
        raise ConfigError(f"unknown model {model_id}; configured: {', '.join(models)}")
    secrets = load_secrets()
    api_key_for(models[model_id], secrets)  # fail now, not after the first prompt

    resume_from = None
    if args.resume:
        resume_from = resolve_session(args.cwd, args.resume)
        if resume_from is None:
            raise ConfigError(f"no session matches {args.resume!r} for {args.cwd}")

    mcp = None
    if settings.mcp_enabled and not args.no_mcp:
        files = [Path(p).expanduser() for p in settings.mcp_files]
        try:
            configs = load_server_configs(files or default_mcp_files(args.cwd.resolve()))
        except McpError as exc:
            raise ConfigError(str(exc)) from exc
        if configs:
            mcp = McpManager(configs, log_dir=config_dir() / "logs")

    holder: list[Session] = []
    session = Session(
        cwd=args.cwd,
        model=models[model_id],
        provider=OpenAICompatProvider(secrets),
        settings=settings,
        approver=None if args.prompt else TerminalApprover(lambda: holder[0]),
        hook_settings=[] if args.no_hooks else None,
        resume_from=resume_from,
        mcp=mcp,
        models=models,
    )
    holder.append(session)

    if args.web or args.open:
        try:
            from mwm_harness.web.server import serve
        except ImportError as exc:
            raise ConfigError(
                f"the panel needs the web extra: pip install 'mwm-harness[web]' ({exc})"
            ) from exc
        await serve(session, models, args.port, open_window=args.open)
        return 0

    if not args.prompt:
        await repl(session, models)
        return 0

    # Headless: only errors and hook blocks go to stderr, the answer goes to stdout.
    quiet = Printer(color=False)
    session.bus.subscribe(lambda event: quiet(event) if _is_problem(event) else None)
    if args.agent:
        # No session hooks here: a Stop gate may be the caller, and must not start itself again.
        if args.agent not in session.agents:
            raise ConfigError(f"unknown agent {args.agent}; known: {', '.join(session.agents)}")
        await session.connect_mcp()
        try:
            report = await session.run_agent(args.agent, args.prompt)
        finally:
            if session.mcp is not None:
                await session.mcp.close()
        print(report.content)
        return 1 if report.is_error else 0
    await session.start()
    ended = await session.send(args.prompt)
    await session.close()
    print(ended.text)
    return 0 if ended.reason == "done" else 1


def _is_problem(event: object) -> bool:
    from mwm_harness import events as ev

    return isinstance(event, (ev.Notice, ev.HookBlocked))


def vendor_monaco_command() -> int:
    from mwm_harness.web.vendor import VendorError, vendor_monaco

    try:
        root, count = vendor_monaco()
    except (VendorError, OSError) as exc:
        print(f"mwm: {exc}", file=sys.stderr)
        return 1
    print(f"code viewer ready: {count} files in {root}; the panel now runs with no network")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.vendor_monaco:
        return vendor_monaco_command()
    try:
        return asyncio.run(run(args))
    except ConfigError as exc:
        print(f"mwm: {exc}", file=sys.stderr)
        return exc.code  # 2 = no API key, 3 = any other configuration mistake
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
