#!/usr/bin/env python3
"""A2A multi-container smoke: drive agent A (on container X) to delegate to agent C (on
container Y) over the real docker network, with a real LLM. Asserts C's answer comes back.

Prerequisites: Docker running + OPENAI_API_KEY/OPENAI_BASE_URL exported (Surplus gateway).
Run from the repo root AFTER `docker compose -f docker-compose.a2a.yml up -d --wait`, or let
this script bring the stack up/down itself:

    OPENAI_API_KEY=... OPENAI_BASE_URL=... python scripts/a2a_docker_smoke.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parent.parent
_COMPOSE = _REPO / "docker-compose.a2a.yml"
_X_BASE = "http://localhost:8080"
_X_URL = _X_BASE + "/v1/chat/stream"
_X_KEY = "koboi_smoke_x"  # matches KOBOI_API_KEYS in docker-compose.a2a.yml (service x)
_PROMPT = (
    "What is the capital of France? You MUST ask peer C via the call_peer_agent tool "
    '(calls=[{"peer": "C", "message": "..."}]), then report C\'s answer.'
)


def _compose(*args: str) -> None:
    subprocess.run(["docker", "compose", "-f", str(_COMPOSE), *args], check=True)


def _wait_ready(base: str, timeout: float = 120.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5) as c:
                if c.get(f"{base}/healthz").status_code == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1)
    raise RuntimeError(f"A did not become ready: {last}")


def _stream_answer(url: str, message: str, key: str) -> str:
    with httpx.Client(timeout=180) as c:
        with c.stream(
            "POST",
            url,
            json={"message": message},
            headers={"Authorization": f"Bearer {key}", "X-Session-Id": "docker-a2a-smoke"},
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "complete":
                    return str(evt.get("content", ""))
                if evt.get("type") == "error":
                    raise RuntimeError(f"stream error: {evt}")
    return ""


def main() -> int:
    managed = "--no-up" not in sys.argv  # if --no-up, assume the stack is already running
    try:
        if managed:
            print(">> bringing up the A2A stack (builds on first run)...")
            _compose("up", "-d", "--wait")
        _wait_ready(_X_BASE)
        print(">> driving A to delegate to C across containers...")
        answer = _stream_answer(_X_URL, _PROMPT, _X_KEY)
        print(f">> A's final answer: {answer!r}")
        assert "paris" in answer.lower(), f"delegation did not propagate C's answer: {answer!r}"
        print("SMOKE PASSED: A delegated to C over the docker network and returned the answer.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if managed:
            print(">> tearing down...")
            with subprocess.Popen(  # noqa: S603
                ["docker", "compose", "-f", str(_COMPOSE), "down"],
                stdout=subprocess.DEVNULL,
            ) as p:
                p.wait()


if __name__ == "__main__":
    raise SystemExit(main())
