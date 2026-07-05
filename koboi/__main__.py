"""Allow running koboi as `python -m koboi`.

Routes through :func:`koboi.cli.main` so ``python -m koboi`` behaves identically
to the ``koboi`` console script (and works on a bare install for all no-TUI
commands).
"""

from __future__ import annotations

from koboi.cli import main

if __name__ == "__main__":
    main()
