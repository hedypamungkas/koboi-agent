"""F2: KOBOI_EXTENSIONS_DIR adds the dir to sys.path so the custom-module loaders
(``tools.custom``, ``rag.custom_modules``, ``context.custom_modules`` — all
``importlib.import_module``-based) can import modules from it. CI-runnable (no Docker).
"""

from __future__ import annotations

import importlib
import sys

import koboi._extensions_path


def test_extensions_dir_added_to_sys_path(tmp_path, monkeypatch):
    monkeypatch.setenv("KOBOI_EXTENSIONS_DIR", str(tmp_path))
    importlib.reload(koboi._extensions_path)  # re-run with the env set
    try:
        assert str(tmp_path) in sys.path
    finally:
        sys.path[:] = [p for p in sys.path if p != str(tmp_path)]


def test_extensions_dir_noop_when_unset(monkeypatch):
    # No env / nonexistent dir -> sys.path unchanged (harmless for pip/non-container use).
    monkeypatch.delenv("KOBOI_EXTENSIONS_DIR", raising=False)
    before = list(sys.path)
    importlib.reload(koboi._extensions_path)
    assert sys.path == before


def test_extension_module_importable_and_tool_recognized(tmp_path, monkeypatch):
    # A custom @tool module placed in the extensions dir is importable + the @tool
    # decorator is applied (sets _tool_def) -- i.e. koboi recognizes it on load.
    (tmp_path / "ext_probe.py").write_text(
        "from koboi.tools.registry import tool\n"
        "@tool(name='ext_probe', description='probe',\n"
        "      parameters={'type': 'object', 'properties': {}, 'required': []})\n"
        "def ext_probe() -> str:\n"
        "    return 'ok'\n"
    )
    monkeypatch.setenv("KOBOI_EXTENSIONS_DIR", str(tmp_path))
    importlib.reload(koboi._extensions_path)
    try:
        assert str(tmp_path) in sys.path
        mod = importlib.import_module("ext_probe")
        assert hasattr(mod.ext_probe, "_tool_def"), "@tool decorator did not apply"
    finally:
        sys.modules.pop("ext_probe", None)
        sys.path[:] = [p for p in sys.path if p != str(tmp_path)]
