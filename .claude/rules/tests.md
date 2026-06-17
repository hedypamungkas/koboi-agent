---
globs: ["tests/**/*.py"]
---

# Test conventions

- All tests use pytest with `asyncio_mode="auto"` (no `@pytest.mark.asyncio` needed)
- Use `MockClient` from `conftest.py` for LLM responses
- Use `make_mock_response()` and `make_mock_tool_call()` helpers from `conftest.py`
- Test classes: `TestXxx` with methods `test_xxx`
- Fixtures available: `mock_client`, `tool_registry`, `memory`, `simple_config` (from `conftest.py`)
- Use `tmp_path` fixture for file-based tests
- Import from `koboi.*` not from relative paths
