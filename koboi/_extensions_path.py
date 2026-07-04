"""koboi/_extensions_path -- add ``KOBOI_EXTENSIONS_DIR`` to ``sys.path`` at import.

Lets a mounted/derived extensions directory be importable by the custom-module
loaders (``tools.custom``, ``rag.custom_modules``, ``context.custom_modules`` -- all
use :func:`importlib.import_module`). Imported eagerly by ``koboi/__init__.py`` so
the path is set before any custom module loads. No-op when the env var is unset
(harmless for pip / non-container use).
"""

from __future__ import annotations

import os
import sys

_ext_dir = os.environ.get("KOBOI_EXTENSIONS_DIR", "")
if _ext_dir and os.path.isdir(_ext_dir) and _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)
