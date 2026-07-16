#!/usr/bin/env python3
"""Focused re-run of the 4 failed/empty models from the first live smoke.

Only re-tests: gpt-image-2 (timeout fix), veo3 (simplified prompt+aspect_ratio),
seedance-2-0-i2v (public image URL instead of data URI), tts-xai-v1 (retry).
Saves to a separate subfolder so it doesn't overwrite the first run.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from datetime import datetime
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
if not _KEY:
    print("ERROR: no OPENAI_API_KEY"); sys.exit(1)

# Public image URL for the i2v test (reliable, CDN-hosted; data URI was rejected by gateway).
_I2V_SOURCE_URL = "https://picsum.photos/id/1018/1024/576"  # mountain landscape

_SURPLUS = {"api_key": _KEY, "base_url": _BASE}


def _conf(storage_dir: str) -> dict:
    return {
        "enabled": True,
        "image": {"provider": "surplus", "model": "venice-gpt-image-2", "surplus": _SURPLUS},
        "video": {"provider": "surplus", "model": "veo3-1-full-text-to-video", "surplus": _SURPLUS},
        "speech": {"provider": "surplus", "model": "tts-xai-v1", "surplus": _SURPLUS},
        "storage": {"backend": "local", "dir": storage_dir},
    }


async def main():
    from koboi.media import build_media
    from koboi.media.types import MediaRequest

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("live_smoke_outputs") / f"rerun_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== focused re-run → {run_dir} ===")

    backend = build_media(_conf(str(run_dir / "_art")))
    results: list = []

    async def _try(label, coro_factory, subdir, ext):
        t0 = time.time()
        print(f"\n--- {label} ---")
        try:
            result = await coro_factory()
            out = run_dir / subdir / f"{label}.{ext}"
            out.parent.mkdir(parents=True, exist_ok=True)
            if result.local_path:
                shutil.copy2(result.local_path, out)
            elif result.data:
                out.write_bytes(result.data)
            status = "ok" if result.status == "ok" and out.exists() and out.stat().st_size > 0 else f"empty({result.status})"
            sz = out.stat().st_size if out.exists() else 0
            print(f"  {label}: {status} ({sz} bytes, {round(time.time()-t0,1)}s)")
            results.append({"model": label, "status": status, "size": sz, "latency": round(time.time()-t0,1)})
        except Exception as e:
            print(f"  {label}: FAILED — {e}")
            results.append({"model": label, "status": f"failed: {e}", "size": 0, "latency": round(time.time()-t0,1)})

    # 1. gpt-image-2 (timeout fix: 300s)
    await _try("venice-gpt-image-2",
               lambda: backend.generate_image(MediaRequest(modality="image", prompt="A serene mountain lake at sunrise with mist rising, photorealistic, golden hour", model="venice-gpt-image-2")),
               "image", "png")

    # 2. veo3 (simplified prompt + aspect_ratio)
    await _try("veo3-1-full-text-to-video",
               lambda: backend.generate_video(MediaRequest(modality="video", prompt="Drone shot over mountains at dawn", model="veo3-1-full-text-to-video", aspect_ratio="16:9", duration_seconds=5)),
               "video_t2v", "mp4")

    # 3. i2v (public image URL instead of data URI)
    await _try("seedance-2-0-image-to-video",
               lambda: backend.generate_video(MediaRequest(modality="video", prompt="Slow zoom in on the landscape", model="seedance-2-0-image-to-video", input_images=[_I2V_SOURCE_URL], duration_seconds=5)),
               "video_i2v", "mp4")

    # 4. tts-xai-v1 (retry)
    await _try("tts-xai-v1",
               lambda: backend.generate_speech(MediaRequest(modality="speech", prompt="Welcome to the future of AI-powered media generation.", model="tts-xai-v1")),
               "tts", "mp3")

    await backend.close()

    # manifest
    md = ["# Focused Re-run (Failed Models)", "", "| Model | Status | Size | Latency |", "|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['model']} | {r['status']} | {r['size']} | {r['latency']}s |")
    (run_dir / "MANIFEST.md").write_text("\n".join(md) + "\n")
    print(f"\n=== done → {run_dir}/MANIFEST.md ===")


if __name__ == "__main__":
    asyncio.run(main())
