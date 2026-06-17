"""koboi/cli -- Console-script entry point with graceful TUI-dep handling."""
from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``koboi`` console script."""
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
