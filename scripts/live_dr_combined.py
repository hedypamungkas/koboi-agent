#!/usr/bin/env python3
"""Combined Deep Research run: cited report + auto image + speech + video (end-to-end live).

Builds a real KoboiAgent with deep_research + media + Firecrawl web search, runs a research query,
and produces a multimedia briefing: a cited text report PLUS an auto-generated explanatory image,
a voiceover (TTS), and a summary video — all from the report content.

Env (from .env): OPENAI_API_KEY (= Surplus key), OPENAI_BASE_URL (= Surplus),
FIRECRAWL_API_KEY. SURPLUS_API_KEY aliased ← OPENAI_API_KEY.

Run: PYTHONPATH=. python scripts/live_dr_combined.py
Output: dr_combined_output/ (report.md + artifacts/).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    for _p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if (_p / ".env").exists():
            load_dotenv(str(_p / ".env"))
            break
except ImportError:
    pass

_KEY = os.environ.get("OPENAI_API_KEY", "")
_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.surplusintelligence.ai/v1")
os.environ.setdefault("SURPLUS_API_KEY", _KEY)
_FC = os.environ.get("FIRECRAWL_API_KEY", "")

if not _KEY:
    print("ERROR: OPENAI_API_KEY (= Surplus key) not found"); sys.exit(1)
if not _FC:
    print("WARNING: FIRECRAWL_API_KEY not found — web search will use mock (no live web)")

_QUERY = "Research the latest breakthroughs in solid-state battery technology in 2026."

_CONFIG = {
    "agent": {"name": "dr-media-combined", "system_prompt": "You plan and run iterative, cited web research.", "mode": "act", "max_iterations": 20},
    "llm": {"provider": "openai", "model": "gpt-5.4", "api_key": _KEY, "base_url": _BASE},
    "orchestration": {"enabled": True, "execution": {"mode": "deep_research"}},
    "research": {
        "max_depth": 1,
        "coverage_threshold": 0.5,
        "capabilities": ["web", "image"],
        "media": {"enabled": True, "kinds": ["image", "speech", "video"], "max_items": 1},
    },
    "websearch": {
        "search": {"provider": "firecrawl" if _FC else "mock", **({"firecrawl": {"api_key": _FC}} if _FC else {})},
        "fetch": {"provider": "firecrawl" if _FC else "httpx", **({"firecrawl": {"api_key": _FC}} if _FC else {})},
    },
    "media": {
        "enabled": True,
        "image": {"provider": "surplus", "model": "venice-z-image-turbo", "surplus": {"api_key": _KEY, "base_url": _BASE}},
        "video": {"provider": "surplus", "model": "kling-v3-4k-text-to-video", "surplus": {"api_key": _KEY, "base_url": _BASE}},
        "speech": {"provider": "surplus", "model": "venice-elevenlabs-tts-turbo-v2-5", "surplus": {"api_key": _KEY, "base_url": _BASE}},
        "storage": {"backend": "local", "dir": "./dr_combined_output/artifacts"},
    },
    "memory": {"backend": "sqlite", "db_path": "./dr_combined_output/dr.db"},
    "sandbox": {"backend": "restricted", "workdir": "./dr_combined_output/workspace"},
}


async def main():
    from koboi.config import Config
    from koboi.facade import KoboiAgent

    out_dir = Path("dr_combined_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Combined DR Run ===")
    print(f"Query: {_QUERY}")
    print(f"Models: image=venice-z-image-turbo, speech=venice-elevenlabs-tts-turbo-v2-5, video=kling-v3-4k-text-to-video")
    print(f"Web: {'firecrawl (live)' if _FC else 'mock (offline)'}")
    print()

    cfg = Config.from_dict(_CONFIG, validate=True)
    agent = KoboiAgent.from_dict(_CONFIG)

    t0 = time.time()
    print("Running DR (this takes several minutes — DR + video gen)...")
    result = await agent.run(_QUERY)
    elapsed = round(time.time() - t0, 1)

    # Save the report
    report = result.content or "(empty)"
    (out_dir / "report.md").write_text(report)

    # Media artifacts from metadata
    media_artifacts = (result.metadata or {}).get("media_artifacts", [])
    research_sources = (result.metadata or {}).get("research_sources", [])

    print(f"\n=== DONE ({elapsed}s) ===")
    print(f"Report: {len(report)} chars")
    print(f"Research sources: {len(research_sources)}")
    print(f"Media artifacts: {len(media_artifacts)}")
    for art in media_artifacts:
        print(f"  - {art.get('kind')}: {art.get('local_path', art.get('prompt', '')[:60])}")
    print(f"\nReport saved: {out_dir / 'report.md'}")
    if media_artifacts:
        print(f"Artifacts: {out_dir / 'artifacts'}/")
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
