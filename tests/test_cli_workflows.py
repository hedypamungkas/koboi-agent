"""Tests for the workflow CLI surface (S4): export/import/workflows/run --workflow
and the non-lossy `koboi graph --format json`."""

import json

import pytest

from koboi import cli_commands
from koboi.workflows.store import FileWorkflowStore

CONFIG = """
agent:
  name: conditional-orchestrator
  system_prompt: "You orchestrate."
orchestration:
  enabled: true
  execution:
    mode: dag
    full_graph: true
  router:
    type: keyword
  determinism:
    temperature: 0.0
    seed: 42
  agents:
    - name: classify
      system_prompt: "Classify sentiment."
      keywords: [classify]
      conditionals:
        - to: praise
          when: {contains: "POS"}
    - name: praise
      system_prompt: "Praise."
      depends_on: [classify]
llm:
  provider: openai
  model: ${OPENAI_MODEL:gpt-4o-mini}
  api_key: ${OPENAI_API_KEY:}
"""


@pytest.fixture
def wf_dir(tmp_path, monkeypatch):
    d = tmp_path / "wfs"
    monkeypatch.setenv("KOBOI_WORKFLOWS_DIR", str(d))
    return d


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(CONFIG, encoding="utf-8")
    return p


class TestExport:
    def test_export_stdout_yaml_preserves_env_template(self, cfg, capsys):
        rc = cli_commands.cmd_export_workflow(str(cfg), fmt="yaml")
        assert rc == 0
        out = capsys.readouterr().out
        assert out.lstrip().startswith("workflow:")
        assert "${OPENAI_API_KEY:}" in out  # placeholder kept (re-runnable)

    def test_export_save_to_store(self, cfg, wf_dir, capsys):
        rc = cli_commands.cmd_export_workflow(str(cfg), save=True, name="myflow")
        assert rc == 0
        store = FileWorkflowStore()
        assert store.exists("myflow")
        assert "workflow:" in store.load("myflow")

    def test_export_output_file(self, cfg, tmp_path):
        out = tmp_path / "bundle.yaml"
        rc = cli_commands.cmd_export_workflow(str(cfg), output=str(out))
        assert rc == 0
        assert out.exists()
        assert "${OPENAI_API_KEY:}" in out.read_text(encoding="utf-8")


class TestImportListShowDelete:
    def test_full_lifecycle(self, wf_dir, cfg, capsys):
        bundle_path = wf_dir.parent / "bundle.yaml"
        assert cli_commands.cmd_export_workflow(str(cfg), output=str(bundle_path)) == 0
        capsys.readouterr()

        assert cli_commands.cmd_import_workflow(str(bundle_path), name="imported") == 0
        store = FileWorkflowStore()
        assert store.exists("imported")

        assert cli_commands.cmd_workflows("list") == 0
        assert "imported" in capsys.readouterr().out

        assert cli_commands.cmd_workflows("show", name="imported") == 0
        assert "workflow:" in capsys.readouterr().out

        assert cli_commands.cmd_workflows("delete", name="imported") == 0
        assert not store.exists("imported")

    def test_show_missing_returns_error(self, wf_dir):
        assert cli_commands.cmd_workflows("show", name="nope") == 1

    def test_import_redacts_concrete_secret(self, wf_dir, tmp_path, capsys):
        # A bundle carrying a concrete api_key must be redacted on import.
        bundle = tmp_path / "leaky.yaml"
        bundle.write_text(
            "workflow:\n  name: leaky\n  schema_version: '1.0'\n"
            "agent:\n  name: x\n"
            "llm:\n  provider: openai\n  model: m\n"
            "  api_key: sk-live-supersecretkey1234567890abcd\n",
            encoding="utf-8",
        )
        assert cli_commands.cmd_import_workflow(str(bundle)) == 0
        stored = FileWorkflowStore().load("leaky")
        assert "sk-live" not in stored


class TestGraphJsonNonLossy:
    def test_includes_conditionals_and_agents_and_legacy_keys(self, cfg, capsys):
        rc = cli_commands.cmd_graph(str(cfg), fmt="json")
        assert rc == 0
        snap = json.loads(capsys.readouterr().out)
        # backward-compat keys preserved
        assert snap["nodes"] == ["classify", "praise"]
        assert {"from": "classify", "to": "praise"} in snap["edges"]
        # non-lossy additions
        assert snap["execution_mode"] == "dag"
        assert snap["router"] == {"type": "keyword"}
        assert any(c["from"] == "classify" and c["to"] == "praise" for c in snap["conditionals"])
        assert snap["agents"][0]["name"] == "classify"


class TestRunWorkflow:
    def test_run_workflow_loads_bundle_and_runs(self, cfg, wf_dir, monkeypatch, capsys):
        cli_commands.cmd_export_workflow(str(cfg), save=True, name="rw")
        capsys.readouterr()
        from koboi.facade import KoboiAgent

        captured: dict = {}

        class _StubAgent:
            async def run(self, message):
                captured["ran"] = message
                return "STUB-RESULT"

        def _fake_from_config_string(bundle, verbose=False):
            captured["bundle"] = bundle
            return _StubAgent()

        monkeypatch.setattr(KoboiAgent, "from_config_string", _fake_from_config_string)
        rc = cli_commands.cmd_run(
            "dummy", "hello world", False, False, None, workflow_name="rw"
        )
        assert rc == 0
        assert "workflow:" in captured["bundle"]
        assert captured["ran"] == "hello world"

    def test_run_workflow_input_json_message(self, cfg, wf_dir, monkeypatch, capsys):
        cli_commands.cmd_export_workflow(str(cfg), save=True, name="rw2")
        capsys.readouterr()
        from koboi.facade import KoboiAgent

        captured: dict = {}

        class _StubAgent:
            async def run(self, message):
                captured["ran"] = message
                return "OK"

        monkeypatch.setattr(
            KoboiAgent, "from_config_string", lambda bundle, verbose=False: _StubAgent()
        )
        rc = cli_commands.cmd_run(
            "dummy", None, False, False, None, workflow_name="rw2", input_json='{"message": "from json"}'
        )
        assert rc == 0
        assert captured["ran"] == "from json"
