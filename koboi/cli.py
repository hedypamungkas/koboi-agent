"""koboi/cli -- Console-script entry point (argparse, graceful dep handling).

The ``koboi`` console script is a single argparse dispatcher that routes every
subcommand to a handler in :mod:`koboi.cli_commands` (core, stdlib-only I/O) or
:mod:`koboi.server.app` / :mod:`koboi.tui.app` (lazy-imported, gated on the
``[api]`` / ``[tui]`` extras). On a bare ``pip install koboi-agent`` (no extras)
the non-interactive commands work out of the box: ``--help``, ``validate``,
``run`` (incl. ``--print``), ``chat --print``, ``sessions``, ``keys``. Only
``serve`` (needs ``[api]``) and interactive ``chat`` (needs ``[tui]``) require
extras; both fail with a clear install hint instead of a traceback.
"""

from __future__ import annotations

import sys


def _run_serve(args) -> None:
    """``koboi serve <config> [--host H] [--port P]`` -> HTTP/SSE server (api extra)."""
    try:
        from koboi.server.app import serve_app
    except ImportError:
        print(
            "Error: API dependencies (fastapi, uvicorn) are not installed.\n"
            "\n"
            "Install them with:\n"
            "    pip install koboi-agent[api]\n",
            file=sys.stderr,
        )
        sys.exit(1)

    serve_app(args.config, host=args.host, port=args.port)


def _run_mcp_serve(args) -> None:
    """``koboi mcp-serve <config>`` -> expose this agent's tools as a stdio MCP server (G9).

    Core-only (no [api] extra). Default exposure is SAFE-only; ``--allow`` adds a named
    MODERATE tool; ``--allow-all`` exposes everything (incl. DESTRUCTIVE, dangerous).
    """
    from koboi.mcp.tool_server import serve_koboi_tools

    serve_koboi_tools(args.config_path, allow=args.allow, allow_all=args.allow_all)


def _run_keys(args) -> None:
    """``koboi keys create|list|revoke|rotate`` -- API key management (M3)."""
    from koboi.server.keys_cli import DEFAULT_KEYS_FILE, create_key, list_keys, revoke_key, rotate_key

    file_path = args.file or DEFAULT_KEYS_FILE

    if args.keys_command == "create":
        plaintext = create_key(file_path, args.label)
        print(f"Created key (shown once):\n  {plaintext}")
        print(f"  Stored in: {file_path}")
    elif args.keys_command == "list":
        keys = list_keys(file_path)
        if not keys:
            print("No keys found.")
        for k in keys:
            status = "REVOKED" if k["revoked"] else "active"
            print(f"  {k['id']}  {status}  {k.get('label', '')}")
    elif args.keys_command == "revoke":
        if revoke_key(args.key_id, file_path):
            print(f"Revoked: {args.key_id}")
        else:
            print(f"Key not found: {args.key_id}", file=sys.stderr)
            sys.exit(1)
    elif args.keys_command == "rotate":
        new = rotate_key(args.key_id, file_path, args.label)
        if new:
            print(f"Rotated {args.key_id}. New key (shown once):\n  {new}")
        else:
            print(f"Key not found: {args.key_id}", file=sys.stderr)
            sys.exit(1)


def _run_chat(args) -> int:
    """``chat --print`` runs core-only; interactive ``chat`` needs the [tui] extra."""
    from koboi import cli_commands

    if args.print_mode:
        return cli_commands.cmd_chat_print(args.config_path, verbose=args.verbose)

    try:
        from koboi.tui.app import run_chat_interactive
    except ImportError:
        print(
            "Error: TUI dependencies (rich, textual) are not installed.\n"
            "\n"
            "Install them with:\n"
            "    pip install koboi-agent[tui]\n"
            "\n"
            "Or use `koboi chat --print` for pipe-friendly JSON-line output (no extras).\n",
            file=sys.stderr,
        )
        return 1
    return run_chat_interactive(args.config_path, verbose=args.verbose, no_tui=args.no_tui, no_stream=args.no_stream)


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="koboi",
        description="Koboi Agent — universal configurable AI agent framework.",
    )
    sub = parser.add_subparsers(dest="command")

    # serve (needs [api])
    p = sub.add_parser("serve", help="Run the HTTP/SSE server (needs [api] extra)")
    p.add_argument("config", help="agent config YAML path")
    p.add_argument("--host", default=None, help="bind host (overrides server.host; default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="bind port (overrides server.port; default 8000)")

    # keys (core; keys_cli is pure-stdlib). ``--file`` is accepted on the parent
    # (``keys --file X list``) AND on each subcommand (``keys list --file X``) so
    # both natural orderings work.
    p = sub.add_parser("keys", help="Manage API keys")
    p.add_argument("--file", default=None, help="keys file (default: ~/.koboi/keys.json)")
    keys_sub = p.add_subparsers(dest="keys_command", required=True)
    p_create = keys_sub.add_parser("create", help="Create a new API key")
    p_create.add_argument("--file", default=argparse.SUPPRESS, help="keys file (override parent --file)")
    p_create.add_argument("--label", default="", help="Label for the key")
    p_list = keys_sub.add_parser("list", help="List all keys")
    p_list.add_argument("--file", default=argparse.SUPPRESS, help="keys file (override parent --file)")
    p_revoke = keys_sub.add_parser("revoke", help="Revoke a key")
    p_revoke.add_argument("--file", default=argparse.SUPPRESS, help="keys file (override parent --file)")
    p_revoke.add_argument("key_id", help="Key ID to revoke")
    p_rotate = keys_sub.add_parser("rotate", help="Rotate a key (revoke old + create new)")
    p_rotate.add_argument("--file", default=argparse.SUPPRESS, help="keys file (override parent --file)")
    p_rotate.add_argument("key_id", help="Key ID to rotate")
    p_rotate.add_argument("--label", default="", help="Label for the new key")

    # validate (core)
    p = sub.add_parser("validate", help="Validate a YAML config without running the agent")
    p.add_argument("config_path")

    # run (core; --print is pipe-friendly JSON lines)
    p = sub.add_parser("run", help="Run a single agent query (non-interactive or one-shot)")
    p.add_argument("config_path")
    p.add_argument("--message", "-m", default=None, help="Message to send (prompted if omitted; stdin in --print)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show debug output")
    p.add_argument("--print", dest="print_mode", action="store_true", help="Stream JSON lines (pipe-friendly)")
    p.add_argument("--resume", dest="resume_session", default=None, help="Resume an interrupted session by ID")

    # mcp-serve (core-only stdio; exposes koboi tools to external MCP clients)
    p = sub.add_parser(
        "mcp-serve",
        help="Expose this agent's tools as an MCP server over stdio (for Claude Desktop/Cursor/etc.)",
    )
    p.add_argument("config_path")
    p.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="TOOL",
        help="Expose a named (MODERATE) tool in addition to the SAFE-only default (repeatable)",
    )
    p.add_argument(
        "--allow-all",
        action="store_true",
        help="Expose EVERY tool including DESTRUCTIVE (dangerous: bypasses approval)",
    )

    # chat (interactive needs [tui]; --print is core)
    p = sub.add_parser("chat", help="Interactive chat (needs [tui] extra unless --print)")
    p.add_argument("config_path")
    p.add_argument("--verbose", "-v", action="store_true", help="Show debug output")
    p.add_argument("--no-stream", dest="no_stream", action="store_true", help="Disable streaming output")
    p.add_argument("--no-tui", dest="no_tui", action="store_true", help="Legacy Rich interface instead of Textual TUI")
    p.add_argument("--print", dest="print_mode", action="store_true", help="JSON-line output (pipe-friendly, no TUI)")

    # sessions (core)
    p = sub.add_parser("sessions", help="List persisted sessions for an agent's database")
    p.add_argument("config_path")
    p.add_argument("--limit", type=int, default=50, help="Max sessions to list")
    p.add_argument("--delete", default=None, metavar="SESSION_ID", help="Delete a session's persisted rows")

    # eval (core)
    p = sub.add_parser("eval", help="Run evaluation suite against an agent config")
    p.add_argument("config_path")
    p.add_argument("--cases", default=None, help="Eval cases YAML file")

    # eval-test (core; eve-style t evals)
    p = sub.add_parser("eval-test", help="Run eve-style t eval tests (*.eval.py files)")
    p.add_argument("path", help="Directory or file with *.eval.py tests")
    p.add_argument("--config", "-c", default=None, help="Agent YAML config for live runs")
    p.add_argument(
        "--mock",
        dest="mock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force mock/live mode (default: from module)",
    )
    p.add_argument("--strict", action="store_true", help="Exit non-zero on any gate failure")
    p.add_argument("--threshold", type=float, default=0.6, help="Soft-score pass threshold")
    p.add_argument(
        "--parallel",
        dest="parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run tests concurrently",
    )
    p.add_argument("--max-concurrency", type=int, default=5, help="Max parallel tests")
    p.add_argument("--tags", default=None, help="Comma-separated tag filter (any-of)")

    # diagnostics (core)
    p = sub.add_parser("diagnostics", help="Export session diagnostics as a ZIP bundle")
    p.add_argument("config_path")
    p.add_argument("--output", "-o", default=None, help="Output file path (default: diagnostics_<ts>.zip)")

    # graph (core) -- render the orchestration agent DAG (Mermaid/JSON)
    p = sub.add_parser("graph", help="Render the orchestration agent DAG (Mermaid or JSON)")
    p.add_argument("config_path")
    p.add_argument("--format", choices=["mermaid", "json"], default="mermaid")

    # init-zsh (core)
    p = sub.add_parser("init-zsh", help="Install the ZSH plugin for :koboi prefix command")
    p.add_argument("--target", default=None, help="Custom plugin install directory")

    return parser


def main() -> None:
    """Entry point for the ``koboi`` console script."""
    # Load .env once for every command (previously only the TUI group did this;
    # config ${VAR:default} interpolation reads os.environ, so all commands need it).
    from dotenv import load_dotenv

    load_dotenv()

    from koboi import cli_commands

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "serve":
        _run_serve(args)
        return
    if args.command == "keys":
        _run_keys(args)
        return
    if args.command == "chat":
        sys.exit(_run_chat(args))

    # Core (no-extra) commands -- all route to cli_commands and exit with its code.
    if args.command == "validate":
        sys.exit(cli_commands.cmd_validate(args.config_path))
    if args.command == "run":
        sys.exit(
            cli_commands.cmd_run(args.config_path, args.message, args.verbose, args.print_mode, args.resume_session)
        )
    if args.command == "mcp-serve":
        _run_mcp_serve(args)
        return
    if args.command == "sessions":
        sys.exit(cli_commands.cmd_sessions(args.config_path, args.limit, delete=args.delete))
    if args.command == "eval":
        sys.exit(cli_commands.cmd_eval(args.config_path, args.cases))
    if args.command == "eval-test":
        sys.exit(
            cli_commands.cmd_eval_test(
                args.path,
                args.config,
                args.mock,
                args.strict,
                args.threshold,
                args.parallel,
                args.max_concurrency,
                args.tags,
            )
        )
    if args.command == "diagnostics":
        sys.exit(cli_commands.cmd_diagnostics(args.config_path, args.output))
    if args.command == "graph":
        sys.exit(cli_commands.cmd_graph(args.config_path, args.format))
    if args.command == "init-zsh":
        sys.exit(cli_commands.cmd_init_zsh(args.target))

    parser.print_help()


if __name__ == "__main__":
    main()
