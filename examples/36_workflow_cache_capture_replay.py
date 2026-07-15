"""Example 36 -- Cache + Capture + offline Replay (v2 + v3) -- the wedge.

The wedge: run an agent once in cache mode (memoize its responses), CAPTURE the
run into a portable bundle (+ cache sidecar), then RE-PLAY it fully OFFLINE --
no API key, byte-identical output, ZERO live model calls. This is the
"beyond Claude Code" determinism mechanism (Claude Code's resume is session-scoped
+ dies on process exit; this is a persistent, shareable, offline-replayable artifact).

Uses a MockClient stand-in (no API key). Real-world: swap MockClient for your
provider (koboi.llm.create_client) and the flow is identical.

CLI equivalent:
    koboi run configs/geo.yaml --replay-mode cache -m "What is the capital of France?"
    koboi capture configs/geo.yaml --with-cache --save --name geo-cap
    koboi run --workflow geo-cap -m "What is the capital of France?"   # offline replay

Run: python examples/36_workflow_cache_capture_replay.py
"""

import asyncio
import os
import tempfile

from koboi.llm.cache import CachedClient, CacheMissPolicy, ResponseCache
from koboi.loop import AgentCore
from koboi.types import AgentResponse, TokenUsage
from koboi.workflows import capture_from_run
from koboi.workflows.cache_sidecar import DirectoryCacheSidecar

QUESTION = "What is the capital of France?"
ANSWER = "Paris"


class MockClient:
    """A stand-in for a real LLM provider (avoids needing an API key for the demo)."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.call_count = 0

    @property
    def model(self) -> str:
        return "demo-model"

    async def complete(self, messages, tools=None, response_format=None) -> AgentResponse:
        self.call_count += 1
        return AgentResponse(content=self._answer, usage=TokenUsage())

    async def get_embeddings(self, text: str):
        return None

    async def close(self) -> None:
        pass


async def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        cache_dir = os.path.join(d, "run_cache")

        # 1. CACHE RUN -- the model is called ONCE; the response is memoized to disk.
        mock = MockClient(ANSWER)
        core = AgentCore(client=CachedClient(mock, ResponseCache(cache_dir)))
        r1 = await core.run(QUESTION)
        print(f"[1] CACHE RUN   model calls = {mock.call_count}   answer = {r1.content!r}")

        # 2. CAPTURE -- freeze the run's response cache into a portable sidecar.
        _, entries = capture_from_run(
            config_text="agent:\n  name: geo\nllm:\n  provider: openai\n  model: demo-model\n",
            name="geo",
            with_cache=True,
            cache_dir=cache_dir,
        )
        sidecar = os.path.join(d, "sidecar")
        DirectoryCacheSidecar(sidecar).write(entries)
        print(f"[2] CAPTURE     froze {len(entries)} cached response(s) into the sidecar")

        # 3. OFFLINE REPLAY -- a FRESH model (would answer WRONG if called), wrapped
        #    in a RAISE CachedClient over the sidecar. Every response is a HIT ->
        #    ZERO live calls, byte-identical output. No API key required.
        mock2 = MockClient("WRONG ANSWER")
        core2 = AgentCore(client=CachedClient(mock2, ResponseCache(sidecar), on_miss=CacheMissPolicy.RAISE))
        r2 = await core2.run(QUESTION)
        print(f"[3] REPLAY      model calls = {mock2.call_count}   answer = {r2.content!r}")

        assert mock2.call_count == 0, "replay must not call the model"
        assert r2.content == ANSWER, "replay must be byte-identical"
        print("\n[OK] byte-identical replay, ZERO live model calls -- fully offline.")


if __name__ == "__main__":
    asyncio.run(main())
