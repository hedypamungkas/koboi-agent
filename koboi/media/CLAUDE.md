# koboi/media/ -- Multimodal generation (image/video/music/speech + STT)

## What this is
A provider-abstract multimodal generation layer: image, video, music, speech (TTS), and
transcription (STT). The default backend is the **Surplus Intelligence** gateway (OpenAI-compatible);
`mock` providers run offline. Opt-in via the top-level `media:` config section; inert otherwise. The
`MediaBackend` is injected as the `media_provider` tool dep (mirrors the `sandbox`/`fetch_provider` seams).

## Key files
```
types.py            MediaUnit (billing enum) + MediaRequest/MediaResult/MediaBudget dataclasses
base.py             One ABC per capability: BaseImage/Video/Music/Speech/TranscriptionProvider
backend.py          MediaBackend (per-modality dispatcher + artifact materialization) + build_media(conf)
registry.py         5 registries + @register_{image,video,music,speech,transcription}_provider + build_*_provider
providers/surplus.py  Surplus*Provider (registered "surplus"; default base_url https://api.surplusintelligence.ai/v1)
providers/mock.py     Mock*Provider (registered "mock"; offline deterministic default)
store.py            MediaStore -- durable artifact store: backend local|r2|s3 (r2/s3 need [media-cloud])
budget.py           Counting*Provider -- fail-soft budget-metering wrappers (per-modality caps)
model_profile.py    ModelProfile + register_model_profile + ~22 built-in profiles (validate/auto-correct before a billed call)
async_job.py        MediaJob + run_async_job (shared submit -> poll -> fetch loop for video/music)
```

## Capabilities + tools (koboi/tools/builtin/media.py)
All 7 tools are `group="media"`, `deps=["media_provider"]`:
- `generate_image` (MODERATE), `generate_video` (DESTRUCTIVE; async), `generate_music` (MODERATE; async),
  `generate_speech` (MODERATE), `transcribe_audio` (MODERATE) -- the blocking facades.
- `submit_media_job` (MODERATE) + `check_media_job` (SAFE) -- the non-blocking async surface (video/music).

## Config (`media:` section; `MediaConfig` in config_models.py)
```yaml
media:
  enabled: true
  image:    { provider: surplus, surplus: { api_key: ${SURPLUS_API_KEY:}, model: venice-z-image-turbo } }
  video:    { provider: surplus, surplus: { api_key: ${SURPLUS_API_KEY:} } }
  budget:   { max_cost_usd: 5.0, max_images: 50, max_video_seconds: 120 }
  storage:  { backend: r2, bucket: ${R2_BUCKET:}, endpoint_url: ${R2_ENDPOINT:},
              access_key_id: ${R2_KEY:}, secret_access_key: ${R2_SECRET:} }
  profiles: [...]                     # ModelProfile overrides
  custom_modules: [mypkg.media_providers]   # @register_* providers
```
Storage `r2`/`s3` need the `[media-cloud]` extra (`boto3`).

## Extension API -- add a provider
Decorator-based (mirrors `websearch.registry`): `@register_image_provider("name", "desc")` on a
`BaseImageProvider` subclass (or the per-modality variant). Load via `media.custom_modules`.

## Surfaces
- **Agent tools**: the 7 above (present only when `media.enabled` + a provider configured).
- **REST**: `POST /v1/media/generate` (sync), `POST /v1/media/jobs` (async, 202), `GET /v1/media/jobs/{job_id}`.
- **Deep Research**: `research.capabilities` (tokens image/video/music/speech) + `research.media` drive an
  auto multimedia briefing generated post-synthesis (`orchestration/research.py`).
- **TUI**: **F3** opens the Media Gallery (collected artifact paths/metadata).
- **Events**: `MediaGeneratedEvent` (`events.py`) is emitted when a media tool fires.

## Conventions
- Per-modality registries, NOT one -- image/video/music/speech/transcription are independent.
- Video/music are async (submit -> poll -> fetch); the blocking `generate_video`/`generate_music` facades poll internally.
- `ModelProfile` validates a request (sizes/durations/voices) before the billed call and auto-corrects where it can.
- Budget is fail-soft: an over-budget call is rejected with a clear message, never a crash.

## Gotchas
- **Inert without `media.enabled`** -- the facade only builds a `MediaBackend` then; without it the tools
  are absent and the `media_provider` dep is unset.
- **`generate_video` is DESTRUCTIVE** (cost/time) -- `timeout=1800s`; gated by approval + budget.
- **R2/S3 need `[media-cloud]`** -- `boto3`; `backend: local` (default) writes to `storage.dir`.
- **Surplus is OpenAI-compatible** -- `auth_mode: bearer`; a custom gateway reuses the Surplus adapter with a different `base_url`.
