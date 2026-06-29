"""koboi/cli -- Console-script entry point with graceful dep handling."""

from __future__ import annotations

import sys


def _run_serve() -> None:
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

    import argparse

    parser = argparse.ArgumentParser(prog="koboi serve", description="Run the koboi HTTP/SSE server")
    parser.add_argument("config", help="agent config YAML path")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    args = parser.parse_args(sys.argv[2:])
    serve_app(args.config, host=args.host, port=args.port)


def _run_keys() -> None:
    """``koboi keys create|list|revoke|rotate`` -- API key management (M3)."""
    import argparse

    from koboi.server.keys_cli import DEFAULT_KEYS_FILE, create_key, list_keys, revoke_key, rotate_key

    parser = argparse.ArgumentParser(prog="koboi keys", description="Manage API keys")
    parser.add_argument("--file", default=DEFAULT_KEYS_FILE, help=f"keys file (default: {DEFAULT_KEYS_FILE})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new API key")
    p_create.add_argument("--label", default="", help="Label for the key")

    sub.add_parser("list", help="List all keys")

    p_revoke = sub.add_parser("revoke", help="Revoke a key")
    p_revoke.add_argument("key_id", help="Key ID to revoke")

    p_rotate = sub.add_parser("rotate", help="Rotate a key (revoke old + create new)")
    p_rotate.add_argument("key_id", help="Key ID to rotate")
    p_rotate.add_argument("--label", default="", help="Label for the new key")

    args = parser.parse_args(sys.argv[2:])

    if args.command == "create":
        plaintext = create_key(args.file, args.label)
        print(f"Created key (shown once):\n  {plaintext}")
        print(f"  Stored in: {args.file}")
    elif args.command == "list":
        keys = list_keys(args.file)
        if not keys:
            print("No keys found.")
        for k in keys:
            status = "REVOKED" if k["revoked"] else "active"
            print(f"  {k['id']}  {status}  {k.get('label', '')}")
    elif args.command == "revoke":
        if revoke_key(args.key_id, args.file):
            print(f"Revoked: {args.key_id}")
        else:
            print(f"Key not found: {args.key_id}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "rotate":
        new = rotate_key(args.key_id, args.file, args.label)
        if new:
            print(f"Rotated {args.key_id}. New key (shown once):\n  {new}")
        else:
            print(f"Key not found: {args.key_id}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """Entry point for the ``koboi`` console script."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _run_serve()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "keys":
        _run_keys()
        return
    try:
        from koboi.tui.app import main as tui_main
    except ImportError:
        print(
            "Error: TUI dependencies (click, rich, textual) are not installed.\n"
            "\n"
            "Install them with:\n"
            "    pip install koboi-agent[tui]\n"
            "\n"
            "Or install everything:\n"
            "    pip install koboi-agent[all]\n",
            file=sys.stderr,
        )
        sys.exit(1)

    tui_main()


if __name__ == "__main__":
    main()
