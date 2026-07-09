"""Example 33: Custom command hooks -- forward every LLM response to a channel.

Demonstrates the ``hooks:`` YAML section: an EXTERNAL script (no Python in the
agent) is triggered on a lifecycle event. Here, every LLM response is forwarded
to a file via ``examples/_command_hook_forwarder.py`` -- a stand-in for a
WhatsApp / Telegram / Slack webhook.

This example is **self-contained** (mock LLM, no API key), so it runs anywhere::

    python examples/33_command_hook_messaging.py

The declarative YAML equivalent for a *real* agent is in
``configs/command_hook_notify.yaml``::

    hooks:
      allow_exec: true                       # default-deny gate; must opt in
      on_event:
        - name: forward-to-whatsapp
          command: ["uvx", "my-wa-forwarder"]   # or ["uv", "run", "forwarder.py"]
          events: ["post_output"]
          fire_and_forget: true               # observe/side-effect, zero SSE latency

Protocol: koboi sends a JSON HookContext on stdin; the script reads it and acts.
When ``fire_and_forget: false`` the script can also return JSON mutations
(abort / inject / modified_tool_result). See ``docs/custom-hooks.md``.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from conftest import setup_example
from koboi.config import Config
from koboi.events import CompleteEvent, TextDeltaEvent
from koboi.facade import _build_command_hooks
from koboi.hooks.chain import HookChain
from koboi.hooks.command_hook import CommandHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.sandbox.passthrough import PassthroughBackend
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, TokenUsage


class _MockClient:
    """Minimal mock LLM so the example needs no API key."""

    _model = "mock"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages, tools=None):
        return AgentResponse(
            content="Hello! This response is forwarded to the channel by an external command hook.",
            tool_calls=[],
            usage=TokenUsage(1, 1),
        )

    async def complete_stream(self, messages, tools=None):
        resp = await self.complete(messages, tools)
        if resp.content:
            yield TextDeltaEvent(content=resp.content)
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


async def _run(out_path: Path) -> None:
    forwarder = Path(__file__).parent / "_command_hook_forwarder.py"
    config = Config(
        {
            "hooks": {
                "allow_exec": True,
                "command_timeout": 10,
                "on_event": [
                    {
                        "name": "forward-response",
                        "command": [sys.executable, str(forwarder), str(out_path)],
                        "events": ["post_output"],
                        "fire_and_forget": True,
                    }
                ],
            }
        }
    )
    chain = HookChain()
    _build_command_hooks(config, PassthroughBackend(), chain)

    agent = AgentCore(
        client=_MockClient(),
        memory=ConversationMemory(),
        tools=ToolRegistry(),
        max_iterations=3,
        hook_chain=chain,
    )
    result = await agent.run("Say hello.")

    # fire-and-forget hooks run off-loop; let the background task finish before reading.
    hook = chain.find_hook(lambda h: isinstance(h, CommandHook))
    if hook and hook._bg_tasks:
        await asyncio.wait_for(asyncio.gather(*hook._bg_tasks, return_exceptions=True), timeout=5)

    print(f"\nAgent response : {result.content}")
    print(f"Forwarded file : {out_path}")
    print(f"File contents  : {out_path.read_text().strip()}")


def main() -> None:
    setup_example(
        "Custom Command Hooks",
        "Forward every LLM response to a channel -- no code in the agent",
    )
    out = Path(tempfile.gettempdir()) / "koboi_command_hook_forward.txt"
    if out.exists():
        out.unlink()
    asyncio.run(_run(out))


if __name__ == "__main__":
    main()
