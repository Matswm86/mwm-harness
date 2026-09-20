"""The ``mwm`` command.

mwm                        interactive session in the current directory
mwm -p "question"          one headless turn; prints the answer, exit 0 on success
mwm --resume last          continue the newest session of this directory
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mwm_harness import __version__
from mwm_harness.config import ConfigError, api_key_for, load_models, load_secrets, load_settings
from mwm_harness.loop import Session
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
    parser.add_argument("--version", action="version", version=f"mwm-harness {__version__}")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.mode:
        settings.permission_mode = args.mode
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

    holder: list[Session] = []
    session = Session(
        cwd=args.cwd,
        model=models[model_id],
        provider=OpenAICompatProvider(secrets),
        settings=settings,
        approver=None if args.prompt else TerminalApprover(lambda: holder[0]),
        hook_settings=[] if args.no_hooks else None,
        resume_from=resume_from,
    )
    holder.append(session)

    if not args.prompt:
        await repl(session, models)
        return 0

    # Headless: only errors and hook blocks go to stderr, the answer goes to stdout.
    quiet = Printer(color=False)
    session.bus.subscribe(lambda event: quiet(event) if _is_problem(event) else None)
    await session.start()
    ended = await session.send(args.prompt)
    await session.close()
    print(ended.text)
    return 0 if ended.reason == "done" else 1


def _is_problem(event: object) -> bool:
    from mwm_harness import events as ev

    return isinstance(event, (ev.Notice, ev.HookBlocked))


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(run(parse_args(argv)))
    except ConfigError as exc:
        print(f"mwm: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
