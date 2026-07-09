"""tests/test_cli_graph.py -- #4 ``koboi graph`` Mermaid/JSON rendering."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from koboi.cli_commands import cmd_graph

DEMO = Path(__file__).resolve().parent.parent / "configs" / "dag_demo.yaml"


def test_cmd_graph_mermaid(capsys):
    rc = cmd_graph(str(DEMO), "mermaid")

    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("graph TD")
    # all four agents present as nodes
    for name in ("research", "draft", "review", "publish"):
        assert name in out
    # depends_on rendered as edges
    assert "research --> draft" in out
    assert "draft --> review" in out
    assert "review --> publish" in out
    assert "draft --> publish" in out


def test_cmd_graph_json(capsys):
    import json

    rc = cmd_graph(str(DEMO), "json")

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["nodes"]) == {"research", "draft", "review", "publish"}
    edge_pairs = {(e["from"], e["to"]) for e in data["edges"]}
    assert ("research", "draft") in edge_pairs
    assert ("draft", "review") in edge_pairs
    assert ("review", "publish") in edge_pairs
    assert ("draft", "publish") in edge_pairs


def test_cmd_graph_missing_agents(tmp_path, capsys):
    cfg = tmp_path / "no_agents.yaml"
    cfg.write_text("agent:\n  name: x\nllm:\n  provider: openai\n  model: m\n")
    rc = cmd_graph(str(cfg), "mermaid")

    assert rc == 1
    assert "No orchestration agents" in capsys.readouterr().err


def test_graph_cli_help():
    result = subprocess.run([sys.executable, "-m", "koboi.cli", "graph", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DAG" in result.stdout or "graph" in result.stdout
