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


def main() -> None:
    """Entry point for the ``koboi`` console script."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _run_serve()
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
