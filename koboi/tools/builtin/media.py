"""koboi/tools/builtin/media -- image generation tool (thin delegating wrapper).

``generate_image`` delegates to a ``MediaBackend`` injected via the tool registry's dep
store (``media_provider``); the backend abstracts the gateway (surplus), budget metering
(``CountingImageProvider``), and artifact materialization (``MediaStore``). Mirrors
``koboi.tools.builtin.web.web_search`` (delegating wrapper over a registry provider).

``risk_level=MODERATE`` (billed side-effect) and ``idempotent=False`` so the resume path
never silently double-fires a billed generation (``types.py`` ToolDefinition).
"""

from __future__ import annotations

from pathlib import Path

import httpx

from koboi.media.types import MediaRequest, MediaResult
from koboi.tools.registry import tool
from koboi.types import RiskLevel


@tool(
    name="generate_image",
    group="media",
    description=(
        "Generate an image from a text prompt and save it locally. "
        "REQUIRED parameter: 'prompt'. Optional: 'size' (e.g. '1024x1024'), "
        "'n' (number of images, default 1). Returns the saved file path, dimensions, "
        "and cost. Does NOT accept other parameters."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Image generation prompt"},
            "size": {"type": "string", "description": "Image size, e.g. '1024x1024'"},
            "n": {"type": "integer", "description": "Number of images (default 1)"},
        },
        "required": ["prompt"],
    },
    risk_level=RiskLevel.MODERATE,
    deps=["media_provider"],
    idempotent=False,
)
async def generate_image(
    prompt: str,
    size: str | None = None,
    n: int = 1,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured (enable media.image in config)"

    model = None
    if _tool_config:
        model = _tool_config.get("image_model")
    req = MediaRequest(
        modality="image",
        prompt=prompt,
        size=size,
        n=max(1, int(n)),
        model=model,
    )
    try:
        result = await backend.generate_image(req)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: image generation failed — {e}"
    return _format_result(result, "Image")


@tool(
    name="generate_video",
    group="media",
    description=(
        "Generate a short video from a text prompt and save it locally. "
        "REQUIRED parameter: 'prompt'. Optional: 'aspect_ratio' (e.g. '16:9'), "
        "'duration_seconds'. Returns the saved file path, duration, and cost. "
        "Video generation is slow (minutes) and billed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Video generation prompt"},
            "aspect_ratio": {"type": "string", "description": "Aspect ratio, e.g. '16:9', '9:16'"},
            "duration_seconds": {"type": "number", "description": "Desired video length in seconds"},
        },
        "required": ["prompt"],
    },
    risk_level=RiskLevel.DESTRUCTIVE,
    deps=["media_provider"],
    idempotent=False,
    timeout=1800.0,
)
async def generate_video(
    prompt: str,
    aspect_ratio: str | None = None,
    duration_seconds: float | None = None,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured (enable media.video in config)"

    model = _tool_config.get("video_model") if _tool_config else None
    req = MediaRequest(
        modality="video",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        model=model,
    )
    try:
        result = await backend.generate_video(req)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: video generation failed — {e}"
    return _format_result(result, "Video")


@tool(
    name="generate_music",
    group="media",
    description=(
        "Generate a music/SFX clip from a text prompt and save it locally. "
        "REQUIRED parameter: 'prompt'. Optional: 'duration_seconds', 'lyrics_prompt', "
        "'voice', 'force_instrumental'. Returns the saved file path, duration, and cost."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Music generation prompt (style/mood)"},
            "duration_seconds": {"type": "number", "description": "Desired length in seconds"},
            "lyrics_prompt": {"type": "string", "description": "Optional lyrics text"},
            "voice": {"type": "string", "description": "Voice/vocalist id (optional)"},
            "force_instrumental": {"type": "boolean", "description": "Force instrumental (no vocals)"},
        },
        "required": ["prompt"],
    },
    risk_level=RiskLevel.MODERATE,
    deps=["media_provider"],
    idempotent=False,
    timeout=600.0,
)
async def generate_music(
    prompt: str,
    duration_seconds: float | None = None,
    lyrics_prompt: str | None = None,
    voice: str | None = None,
    force_instrumental: bool | None = None,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured (enable media.music in config)"

    model = _tool_config.get("music_model") if _tool_config else None
    req = MediaRequest(
        modality="music",
        prompt=prompt,
        duration_seconds=duration_seconds,
        lyrics_prompt=lyrics_prompt,
        voice=voice,
        force_instrumental=force_instrumental,
        model=model,
    )
    try:
        result = await backend.generate_music(req)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: music generation failed — {e}"
    return _format_result(result, "Music")


@tool(
    name="generate_speech",
    group="media",
    description=(
        "Synthesize speech audio from text and save it locally. REQUIRED parameter: 'prompt' "
        "(the text to speak). Optional: 'voice', 'response_format' (e.g. 'mp3', 'wav'), 'speed'. "
        "Returns the saved file path and character count."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The text to synthesize"},
            "voice": {"type": "string", "description": "Voice id (e.g. 'alloy')"},
            "response_format": {"type": "string", "description": "Audio format: 'mp3', 'wav', 'opus'"},
            "speed": {"type": "number", "description": "Speed multiplier (e.g. 1.0)"},
        },
        "required": ["prompt"],
    },
    risk_level=RiskLevel.MODERATE,
    deps=["media_provider"],
    idempotent=False,
    timeout=120.0,
)
async def generate_speech(
    prompt: str,
    voice: str | None = None,
    response_format: str | None = None,
    speed: float | None = None,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured (enable media.speech in config)"

    model = _tool_config.get("speech_model") if _tool_config else None
    req = MediaRequest(
        modality="speech",
        prompt=prompt,
        voice=voice,
        response_format=response_format,
        speed=speed,
        model=model,
    )
    try:
        result = await backend.generate_speech(req)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: speech generation failed — {e}"
    return _format_result(result, "Speech")


@tool(
    name="transcribe_audio",
    group="media",
    description=(
        "Transcribe speech from an audio file or URL to text. REQUIRED: exactly one of 'file_path' "
        "or 'url'. Optional: 'language_code'. Returns the transcribed text. Read-only (idempotent)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to a local audio file"},
            "url": {"type": "string", "description": "URL of an audio file"},
            "language_code": {"type": "string", "description": "ISO language hint, e.g. 'en', 'id'"},
        },
        "required": [],
    },
    risk_level=RiskLevel.MODERATE,
    deps=["media_provider"],
    idempotent=True,
)
async def transcribe_audio(
    file_path: str | None = None,
    url: str | None = None,
    language_code: str | None = None,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured (enable media.transcription in config)"
    if url:
        try:
            audio = await _fetch_audio_bytes(url)
        except Exception as e:  # noqa: BLE001 - boundary
            return f"Error: failed to fetch audio from {url} — {e}"
    elif file_path:
        try:
            audio = Path(file_path).read_bytes()
        except Exception as e:  # noqa: BLE001 - boundary
            return f"Error: failed to read {file_path} — {e}"
    else:
        return "Error: provide 'file_path' or 'url'"
    try:
        text = await backend.transcribe(audio, language_code=language_code)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: transcription failed — {e}"
    return text or "(no speech transcribed)"


async def _fetch_audio_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


@tool(
    name="submit_media_job",
    group="media",
    description=(
        "Submit a media generation job and return a job_id. Video/music run asynchronously (poll "
        "with check_media_job); image/speech complete immediately. REQUIRED: 'modality', 'prompt'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "modality": {"type": "string", "description": "image|video|music|speech"},
            "prompt": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "aspect_ratio": {"type": "string"},
            "voice": {"type": "string"},
        },
        "required": ["modality", "prompt"],
    },
    risk_level=RiskLevel.MODERATE,
    deps=["media_provider"],
    idempotent=False,
)
async def submit_media_job(
    modality: str,
    prompt: str,
    duration_seconds: float | None = None,
    aspect_ratio: str | None = None,
    voice: str | None = None,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured"
    req = MediaRequest(
        modality=modality,
        prompt=prompt,
        duration_seconds=duration_seconds,
        aspect_ratio=aspect_ratio,
        voice=voice,
    )
    try:
        job = await backend.submit_media_job(req)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: submit failed — {e}"
    return f"Media job submitted: id={job.job_id} kind={job.kind} status={job.status}"


@tool(
    name="check_media_job",
    group="media",
    description=(
        "Check a submitted media job's status (poll once). Returns the status + the saved artifact "
        "path once succeeded. REQUIRED: 'job_id'."
    ),
    parameters={
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
    risk_level=RiskLevel.SAFE,
    deps=["media_provider"],
    idempotent=True,
)
async def check_media_job(
    job_id: str,
    _deps: dict | None = None,
    _tool_config: dict | None = None,
) -> str:
    backend = (_deps or {}).get("media_provider")
    if backend is None:
        return "Error: media not configured"
    try:
        job = await backend.check_media_job(job_id)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: check failed — {e}"
    if job is None:
        return f"No media job with id={job_id}"
    path = job.result.local_path if (job.result and job.result.local_path) else None
    if job.status == "succeeded" and path:
        return f"Media job {job_id}: succeeded -> {path}"
    return f"Media job {job_id}: {job.status}"


def _format_result(result: MediaResult, label: str) -> str:
    """Render a MediaResult as the tool output string (image/video/music)."""
    if result.status != "ok":
        reason = result.rejection_reason or result.status
        suffix = " (content filtered)" if result.safety_blocked else ""
        return f"{label} generation {result.status}: {reason}{suffix}"

    parts: list[str] = [result.content_type or label.lower()]
    if result.width and result.height:
        parts.append(f"{result.width}x{result.height}")
    if result.duration_seconds is not None:
        parts.append(f"{result.duration_seconds:.1f}s")
    if result.cost_usd is not None:
        unit = f"/{result.billing_unit.value}" if result.billing_unit else ""
        parts.append(f"${result.cost_usd:.4f}{unit}")
    if result.model:
        parts.append(f"model={result.model}")

    location = result.local_path or result.url or "(no artifact)"
    return f"{label} saved: {location} ({', '.join(parts)})"
