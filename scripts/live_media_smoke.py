#!/usr/bin/env python3
"""Tier B: live all-models media smoke — generate one output per listed Surplus model + a DR run.

Saves labeled artifacts + a MANIFEST.md comparison table to ``live_smoke_outputs/run_<ts>/``.
Exercises the real gateway (real $). Run: ``python scripts/live_media_smoke.py``.

Env (from .env via load_dotenv): OPENAI_API_KEY (= Surplus inf_ key), OPENAI_BASE_URL (= Surplus),
FIRECRAWL_API_KEY (for the DR web search). SURPLUS_API_KEY is aliased ← OPENAI_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# --- setup: load .env + alias the key ---
try:
    from dotenv import load_dotenv

    # Walk all parents to find .env (script is in a worktree; .env is at the repo root).
    for _candidate in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        _env = _candidate / ".env"
        if _env.exists():
            load_dotenv(str(_env))
            break
except ImportError:
    pass

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.surplusintelligence.ai/v1")
os.environ.setdefault("SURPLUS_API_KEY", _OPENAI_KEY)
_FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")

if not _OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY (= Surplus key) not found in env/.env", file=sys.stderr)
    sys.exit(1)

# --- models + prompts ---
IMAGE_MODELS = ["venice-z-image-turbo", "venice-nano-banana-pro", "venice-gpt-image-2"]
VIDEO_T2V_MODELS = ["kling-v3-4k-text-to-video", "veo3-1-full-text-to-video", "seedance-1-5-pro-text-to-video"]
VIDEO_I2V_MODEL = "seedance-2-0-image-to-video"
TTS_MODELS = ["tts-xai-v1", "tts-gemini-3-1-flash", "venice-elevenlabs-tts-turbo-v2-5"]
STT_MODEL = "venice-whisper-large-v3"

IMAGE_PROMPT = "A serene mountain lake at sunrise with mist rising off the water, photorealistic, golden hour lighting"
VIDEO_PROMPT = "A cinematic drone shot flying over a serene mountain lake at sunrise, mist rising, golden hour"
I2V_PROMPT = "Slowly zoom in on the mountain lake scene, water rippling gently, mist swirling"
TTS_TEXT = "Welcome to the future of AI-powered media generation. This is a test of speech synthesis quality."
DR_QUERY = "Research the latest breakthroughs in solid-state battery technology in 2026"

_LLM_MODEL = os.environ.get("SMOKE_LLM_MODEL", "gpt-5.4")


def _media_conf(storage_dir: str) -> dict:
    """Build a media config with the surplus provider for all modalities."""
    surplus = {"api_key": _OPENAI_KEY, "base_url": _BASE_URL}
    return {
        "enabled": True,
        "image": {"provider": "surplus", "model": IMAGE_MODELS[0], "surplus": surplus},
        "video": {"provider": "surplus", "model": VIDEO_T2V_MODELS[0], "surplus": surplus},
        "music": {"provider": "surplus", "model": "venice-minimax-music-v26", "surplus": surplus},
        "speech": {"provider": "surplus", "model": TTS_MODELS[0], "surplus": surplus},
        "transcription": {"provider": "surplus", "model": STT_MODEL, "surplus": surplus},
        "storage": {"backend": "local", "dir": storage_dir},
    }


def _record(manifest: list, modality: str, model: str, prompt: str, params: dict, result, error: str, latency: float, out_file: str | None):
    """Append a row to the manifest."""
    cost = None
    billing_unit = None
    if result is not None:
        cost = float(result.cost_usd) if result.cost_usd is not None else None
        billing_unit = result.billing_unit.value if result.billing_unit else None
    manifest.append({
        "modality": modality, "model": model, "prompt": prompt[:80], "params": params,
        "cost_usd": cost, "billing_unit": billing_unit, "latency_s": round(latency, 1),
        "status": "ok" if error is None else "failed", "error": error, "file": out_file,
    })


async def _gen_image(backend, model, prompt, run_dir, manifest, response_format=None):
    from koboi.media.types import MediaRequest

    t0 = time.time()
    req = MediaRequest(modality="image", prompt=prompt, model=model, response_format=response_format)
    try:
        result = await backend.generate_image(req)
        ext = ".png"
        out = run_dir / "image" / f"{model}{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if result.local_path:
            shutil.copy2(result.local_path, out)
        elif result.data:
            out.write_bytes(result.data)  # fallback: materialization may have failed
        _record(manifest, "image", model, prompt, {"response_format": response_format}, result, None, time.time() - t0, str(out))
        print(f"  [image] {model}: OK ({round(time.time()-t0,1)}s) -> {out.name}")
        return result
    except Exception as e:
        _record(manifest, "image", model, prompt, {"response_format": response_format}, None, str(e), time.time() - t0, None)
        print(f"  [image] {model}: FAILED — {e}")
        return None


async def _gen_video(backend, model, prompt, run_dir, manifest, duration=5, input_images=None):
    from koboi.media.types import MediaRequest

    t0 = time.time()
    req = MediaRequest(modality="video", prompt=prompt, model=model, duration_seconds=duration, input_images=input_images)
    subdir = "video_i2v" if input_images else "video_t2v"
    try:
        result = await backend.generate_video(req)
        out = run_dir / subdir / f"{model}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        if result.local_path:
            shutil.copy2(result.local_path, out)
        elif result.data:
            out.write_bytes(result.data)  # fallback: materialization may have failed
        _record(manifest, subdir, model, prompt, {"duration": duration, "input_images": input_images}, result, None, time.time() - t0, str(out))
        print(f"  [{subdir}] {model}: OK ({round(time.time()-t0,1)}s) -> {out.name}")
    except Exception as e:
        _record(manifest, subdir, model, prompt, {"duration": duration}, None, str(e), time.time() - t0, None)
        print(f"  [{subdir}] {model}: FAILED — {e}")


async def _gen_speech(backend, model, text, run_dir, manifest):
    from koboi.media.types import MediaRequest

    t0 = time.time()
    req = MediaRequest(modality="speech", prompt=text, model=model)
    try:
        result = await backend.generate_speech(req)
        out = run_dir / "tts" / f"{model}.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        if result.local_path:
            shutil.copy2(result.local_path, out)
        elif result.data:
            out.write_bytes(result.data)  # fallback: materialization may have failed
        _record(manifest, "speech", model, text, {}, result, None, time.time() - t0, str(out))
        print(f"  [tts] {model}: OK ({round(time.time()-t0,1)}s) -> {out.name}")
        return result
    except Exception as e:
        _record(manifest, "speech", model, text, {}, None, str(e), time.time() - t0, None)
        print(f"  [tts] {model}: FAILED — {e}")
        return None


async def _gen_stt(backend, model, audio_path, run_dir, manifest):
    t0 = time.time()
    try:
        audio = Path(audio_path).read_bytes()
        text = await backend.transcribe(audio, model=model)
        out = run_dir / "stt" / f"{model}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        manifest.append({"modality": "stt", "model": model, "prompt": "(audio input)", "params": {},
                         "cost_usd": None, "billing_unit": None, "latency_s": round(time.time()-t0, 1),
                         "status": "ok", "error": None, "file": str(out), "transcribed_text": text})
        print(f"  [stt] {model}: OK ({round(time.time()-t0,1)}s)")
    except Exception as e:
        manifest.append({"modality": "stt", "model": model, "prompt": "(audio input)", "params": {},
                         "cost_usd": None, "billing_unit": None, "latency_s": round(time.time()-t0, 1),
                         "status": "failed", "error": str(e), "file": None})
        print(f"  [stt] {model}: FAILED — {e}")


def _write_manifest(run_dir: Path, manifest: list):
    md = ["# Live Media Smoke — Model Comparison", "", f"Run: {run_dir.name}", "",
          "| Modality | Model | Cost $ | Latency s | Status | File |",
          "|---|---|---|---|---|---|"]
    for m in manifest:
        cost = f"{m['cost_usd']:.4f}" if m["cost_usd"] is not None else "—"
        fname = Path(m["file"]).name if m["file"] else "—"
        md.append(f"| {m['modality']} | {m['model']} | {cost} | {m['latency_s']} | {m['status']} | {fname} |")
    (run_dir / "MANIFEST.md").write_text("\n".join(md) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


async def main():
    from koboi.media import build_media

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("live_smoke_outputs") / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== live media smoke → {run_dir} ===")

    backend = build_media(_media_conf(str(run_dir / "_artifacts")))
    if backend is None:
        print("ERROR: build_media returned None (media not enabled?)", file=sys.stderr)
        sys.exit(1)

    manifest: list = []

    # 1. Image (3 models) — b64_json (validated); first image's bytes feed the i2v step.
    print("\n--- IMAGE ---")
    first_image = await _gen_image(backend, IMAGE_MODELS[0], IMAGE_PROMPT, run_dir, manifest)
    for model in IMAGE_MODELS[1:]:
        await _gen_image(backend, model, IMAGE_PROMPT, run_dir, manifest)

    # 2. Video text-to-video (3 models, 5s each — minutes each).
    print("\n--- VIDEO (text-to-video) — this takes minutes per model ---")
    for model in VIDEO_T2V_MODELS:
        await _gen_video(backend, model, VIDEO_PROMPT, run_dir, manifest, duration=5)

    # 3. Video image-to-video (from the first generated image as a base64 data URI).
    print("\n--- VIDEO (image-to-video) ---")
    import base64

    if first_image and first_image.data:
        data_uri = f"data:image/png;base64,{base64.b64encode(first_image.data).decode()}"
        await _gen_video(backend, VIDEO_I2V_MODEL, I2V_PROMPT, run_dir, manifest, input_images=[data_uri])
    else:
        print("  [i2v] skipped — no source image data (first image gen failed)")

    # 4. TTS (3 models).
    print("\n--- SPEECH (TTS) ---")
    first_tts = await _gen_speech(backend, TTS_MODELS[0], TTS_TEXT, run_dir, manifest)
    for model in TTS_MODELS[1:]:
        await _gen_speech(backend, model, TTS_TEXT, run_dir, manifest)

    # 5. STT (transcribe the first TTS clip).
    print("\n--- STT ---")
    tts_path = None
    if first_tts and first_tts.local_path:
        tts_path = first_tts.local_path
    elif first_tts and first_tts.data:
        tts_path = run_dir / "tts" / f"_stt_source.mp3"
        tts_path.parent.mkdir(parents=True, exist_ok=True)
        tts_path.write_bytes(first_tts.data)
    if tts_path:
        await _gen_stt(backend, STT_MODEL, tts_path, run_dir, manifest)
    else:
        print("  [stt] skipped — no TTS audio to transcribe")

    _write_manifest(run_dir, manifest)
    print(f"\n=== per-model matrix done → {run_dir}/MANIFEST.md ===")
    print(f"    (DR combined run is separate — run with: python scripts/live_media_smoke.py --dr)")


if __name__ == "__main__":
    asyncio.run(main())
