"""Allow running koboi as `python -m koboi`."""
from __future__ import annotations

try:
    from koboi.tui.app import main
except ImportError:
    import sys
    print(
        "Error: TUI dependencies not installed.\n"
        "Install with: pip install koboi-agent[tui]",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    main()
