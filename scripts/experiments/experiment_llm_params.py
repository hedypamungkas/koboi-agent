"""experiment_llm_params.py -- Post-fix regression probe for LLM-parameter wiring.

No network, no API keys. Drives the REAL production path
    Config -> facade._build_client -> RetryClient -> create_client -> factory -> Adapter
and intercepts at the HTTP egress point (HttpTransport.post). The body captured
there is exactly what would be sent to the provider, so this asserts that every
configured knob actually reaches the wire.

History: an earlier revision of this script proved the bugs (max_tokens dropped
for OpenAI/Cloudflare; sampling + reasoning params never forwarded; orchestration
llm_config dead). Those are now fixed; this version asserts the fixed state and
exits non-zero on any regression.

Run:    python3 experiment_llm_params.py
Exit 0  == all params reach the body (no regression).
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from typing import Any

# --- koboi production imports (the exact code under test) --------------------
from koboi.config import Config
from koboi.facade import _build_client
from koboi.types import AgentDef

_OPENAI_CANNED = {
    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
}
_ANTHROPIC_CANNED = {
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 2},
}


def _install_spy(client: Any) -> dict:
    """Replace HttpTransport.post with a capturing spy. Returns the capture dict."""
    captured: dict = {}
    transport = client._impl._transport  # RetryClient -> adapter -> HttpTransport

    async def spy_post(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = json.loads(json.dumps(body))
        if "/chat/completions" in path:
            return _OPENAI_CANNED
        if "/messages" in path:
            return _ANTHROPIC_CANNED
        raise AssertionError(f"unexpected path {path}")

    transport.post = spy_post
    return captured


def _install_stream_spy(client: Any) -> dict:
    captured: dict = {}
    transport = client._impl._transport

    async def spy_stream(path: str, body: dict):
        captured["body"] = json.loads(json.dumps(body))
        if "/chat/completions" in path:
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'.encode()
            yield "data: [DONE]".encode()
        elif "/messages" in path:
            yield 'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1}}}'.encode()
            yield 'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}'.encode()
            yield 'event: message_stop\ndata: {"type":"message_stop"}'.encode()
        else:
            raise AssertionError(f"unexpected path {path}")

    transport.post_stream = spy_stream
    return captured


def _build_and_capture(llm_cfg: dict) -> dict:
    config = Config.from_dict({"agent": {"name": "repro"}, "llm": llm_cfg}, validate=True)
    client = _build_client(config, logger=None)
    captured = _install_spy(client)
    asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    asyncio.run(client.close())
    return captured


def _build_and_capture_stream(llm_cfg: dict) -> dict:
    config = Config.from_dict({"agent": {"name": "repro"}, "llm": llm_cfg}, validate=True)
    client = _build_client(config, logger=None)
    captured = _install_stream_spy(client)

    async def drain() -> None:
        async for _ev in client.complete_stream([{"role": "user", "content": "hi"}]):
            pass

    asyncio.run(drain())
    asyncio.run(client.close())
    return captured


_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        {detail}")
    results.append((name, ok, detail))


def section(title: str) -> None:
    print(f"\n{_BOLD}=== {title} ==={_RESET}")


def main() -> int:
    print(
        f"{_BOLD}koboi-agent LLM parameter wiring -- post-fix regression probe{_RESET}\n"
        f"Intercepting at HttpTransport.post (the real HTTP egress point).\n"
    )

    # ---------------------------------------------------------------- EXP A
    section("EXP A: max_tokens reaches the body (OpenAI + Anthropic + stream)")
    body_oai = _build_and_capture(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-repro",
         "max_tokens": 8192, "temperature": 0.5}
    )
    body_ant = _build_and_capture(
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514",
         "api_key": "sk-ant-repro", "max_tokens": 8192, "temperature": 0.5}
    )
    report(
        "OpenAI forwards llm.max_tokens",
        body_oai["body"].get("max_tokens") == 8192,
        f'body["max_tokens"] = {body_oai["body"].get("max_tokens")!r}  (expected 8192)',
    )
    report(
        "Anthropic forwards llm.max_tokens",
        body_ant["body"].get("max_tokens") == 8192,
        f'body["max_tokens"] = {body_ant["body"].get("max_tokens")!r}  (expected 8192)',
    )
    body_oai_stream = _build_and_capture_stream(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-repro",
         "max_tokens": 8192, "temperature": 0.5}
    )
    report(
        "OpenAI forwards llm.max_tokens on the streaming/SSE path",
        body_oai_stream["body"].get("max_tokens") == 8192,
        f'streamed body keys = {sorted(body_oai_stream["body"].keys())}',
    )
    # Unset -> OpenAI must OMIT max_tokens (no force-cap); Anthropic falls back to 4096.
    body_oai_unset = _build_and_capture(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-repro"}
    )
    body_ant_unset = _build_and_capture(
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "sk-ant-repro"}
    )
    report(
        "OpenAI omits max_tokens when unset (no force-cap regression)",
        "max_tokens" not in body_oai_unset["body"],
        f'body keys = {sorted(body_oai_unset["body"].keys())}',
    )
    report(
        "Anthropic falls back to 4096 when unset (API requires the field)",
        body_ant_unset["body"].get("max_tokens") == 4096,
        f'body["max_tokens"] = {body_ant_unset["body"].get("max_tokens")!r}',
    )

    # ---------------------------------------------------------------- EXP B
    section("EXP B: temperature reaches the body (control)")
    report(
        "OpenAI forwards llm.temperature",
        body_oai["body"].get("temperature") == 0.5,
        f'body["temperature"] = {body_oai["body"].get("temperature")!r}',
    )

    # ---------------------------------------------------------------- EXP C
    section("EXP C: reasoning / thinking budget reaches the body")
    body_reason = _build_and_capture(
        {"provider": "openai", "model": "o3-mini", "api_key": "sk-repro",
         "reasoning_effort": "high", "thinking": {"budget_tokens": 2000},
         "max_completion_tokens": 8000}
    )
    report(
        "reasoning_effort reaches the body",
        body_reason["body"].get("reasoning_effort") == "high",
        f'body["reasoning_effort"] = {body_reason["body"].get("reasoning_effort")!r}',
    )
    report(
        "thinking.budget_tokens reaches the body",
        body_reason["body"].get("thinking") == {"budget_tokens": 2000},
        f'body["thinking"] = {body_reason["body"].get("thinking")!r}',
    )
    report(
        "max_completion_tokens reaches the body AND suppresses max_tokens (o-series)",
        body_reason["body"].get("max_completion_tokens") == 8000 and "max_tokens" not in body_reason["body"],
        f'max_completion_tokens={body_reason["body"].get("max_completion_tokens")!r}, '
        f'has max_tokens={"max_tokens" in body_reason["body"]}',
    )

    # ---------------------------------------------------------------- EXP D
    section("EXP D: sampling/utility params reach the body")
    body_sampler = _build_and_capture(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-repro",
         "top_p": 0.1, "frequency_penalty": 0.5, "presence_penalty": 0.5,
         "stop": ["\n"], "seed": 42, "response_format": {"type": "json_object"}}
    )
    expected = {"top_p": 0.1, "frequency_penalty": 0.5, "presence_penalty": 0.5,
                "stop": ["\n"], "seed": 42, "response_format": {"type": "json_object"}}
    missing = [k for k, v in expected.items() if body_sampler["body"].get(k) != v]
    report(
        "all sampling/utility params present and correct",
        not missing,
        f'missing/mismatched = {missing}  |  body keys = {sorted(body_sampler["body"].keys())}',
    )

    # ---------------------------------------------------------------- EXP E
    section("EXP E: orchestration per-agent llm_config drives a dedicated client")
    try:
        from koboi.orchestration.factory import AgentFactory

        config = Config.from_dict(
            {"agent": {"name": "repro"},
             "llm": {"provider": "openai", "model": "gpt-4o-mini",
                      "api_key": "sk-repro", "temperature": 0.9}},
            validate=True,
        )
        shared = _build_client(config, logger=None)

        def _agent_client_builder(agent_llm: dict):
            overrides = {k: v for k, v in agent_llm.items() if k != "max_context_tokens"}
            return _build_client(config, logger=None, llm_overrides=overrides)

        ad = AgentDef(
            name="worker",
            system_prompt="hi",
            llm_config={"temperature": 0.1, "max_tokens": 1234, "max_context_tokens": 9000},
        )
        agent = AgentFactory.create_configured_agent(
            ad, client=shared, logger=None, client_builder=_agent_client_builder
        )
        dedicated = agent.client is not shared
        report(
            "agent gets a DEDICATED client (not the shared one)",
            dedicated,
            f"agent.client is shared = {agent.client is shared}",
        )
        report(
            "per-agent temperature/max_tokens applied on the dedicated client",
            getattr(agent.client._impl, "_temperature", None) == 0.1
            and getattr(agent.client._impl, "_max_tokens", None) == 1234,
            f"_temperature={agent.client._impl._temperature!r}, "
            f"_max_tokens={agent.client._impl._max_tokens!r}",
        )
        report(
            "max_context_tokens still consumed from llm_config",
            getattr(agent, "max_context_tokens", None) == 9000,
            f"agent.max_context_tokens = {getattr(agent, 'max_context_tokens', None)!r}",
        )

        # Agent with only max_context_tokens -> reuses the shared client.
        ad2 = AgentDef(name="lite", system_prompt="hi", llm_config={"max_context_tokens": 4000})
        agent2 = AgentFactory.create_configured_agent(
            ad2, client=shared, logger=None, client_builder=_agent_client_builder
        )
        report(
            "agent with only max_context_tokens reuses the shared client",
            agent2.client is shared,
            f"agent2.client is shared = {agent2.client is shared}",
        )
        asyncio.run(shared.close())
        if dedicated:
            asyncio.run(agent.client.close())
    except Exception:
        traceback.print_exc()
        report("EXP E ran without crashing", False, "exception above")

    # ---------------------------------------------------------------- SUMMARY
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"  {passed}/{len(results)} checks passed")
    all_ok = all(ok for _, ok, _ in results)
    print(f"\n  {_PASS if all_ok else _FAIL} overall: "
          f"{'all params reach the body -- no regression' if all_ok else 'REGRESSION -- a param is being dropped'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
