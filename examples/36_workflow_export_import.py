"""Example 36 -- Workflow export/import (v1).

Demonstrates the self-contained workflow bundle: export a config to a portable,
secret-redacted YAML bundle (with a ``workflow:`` envelope + provenance), then
import it back and re-load it as a valid Config. No API key needed (serialization
only).

CLI equivalent:
    koboi export configs/workflow_export_demo.yaml --output support_bot.yaml
    koboi import support_bot.yaml --name support-bot
    koboi run --workflow support-bot -m "hi"

Run: python examples/36_workflow_export_import.py
"""

import os
import tempfile
from pathlib import Path

from koboi.config import Config
from koboi.workflows import WorkflowDefinition, build_from_config_path
from koboi.workflows.store import FileWorkflowStore

CONFIG_YAML = """\
agent:
  name: support-bot
  system_prompt: "You are a helpful support assistant."
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY:}
orchestration:
  enabled: true
  execution: {mode: dag}
  determinism: {temperature: 0.0, seed: 42}
  agents:
    - name: triage
      system_prompt: "Classify the issue."
    - name: resolve
      system_prompt: "Resolve it."
      depends_on: [triage]
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "support.yaml"
        cfg_path.write_text(CONFIG_YAML)

        # 1. EXPORT -- build a WorkflowDefinition from the un-interpolated source.
        #    Secrets are redacted (concrete -> masked); ${VAR} templates are KEPT
        #    so the bundle is share-safe AND re-runnable on another machine.
        wd = build_from_config_path(cfg_path, name="support-bot", description="Triage + resolve demo")
        bundle = wd.to_bundle_yaml()
        print("=" * 70)
        print("EXPORTED BUNDLE (workflow: envelope + redacted config body)")
        print("=" * 70)
        print(bundle)
        assert "${OPENAI_API_KEY:}" in bundle, "env template must survive export"
        assert wd.provenance.koboi_version is not None

        # 2. IMPORT -- round-trip the bundle back into a WorkflowDefinition.
        wd2 = WorkflowDefinition.from_bundle_yaml(bundle)
        print("=" * 70)
        print(f"IMPORTED: name={wd2.name!r}  provenance.koboi_version={wd2.provenance.koboi_version}")
        print(f"          determinism={wd2.determinism}")
        assert wd2.name == "support-bot"

        # 3. STORE -- save to the project workflow store + list + reload.
        os.environ["KOBOI_WORKFLOWS_DIR"] = str(Path(d) / "wfs")
        store = FileWorkflowStore()
        path = store.save("support-bot", bundle)
        print("=" * 70)
        print(f"SAVED to {path}")
        print(f"WORKFLOWS in store: {[w['name'] for w in store.list()]}")

        # 4. RE-RUNNABLE -- the stored bundle loads as a valid Config.
        cfg = Config.from_string(store.load("support-bot"))
        print("=" * 70)
        print(f"RE-LOADED as Config: agent={cfg.raw['agent']['name']!r}")


if __name__ == "__main__":
    main()
