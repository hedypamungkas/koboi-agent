# Multimodal Generation Strategy — `koboi/media/`

> Status: **SHIPPED** (PR #43, W0–W5c; 2026-07-15). See `koboi/media/` + `koboi/media/CLAUDE.md`.
> Scope: how koboi-agent should integrate **image / video / audio generation** as a first-class
> platform capability — routed through the **Surplus Intelligence** gateway, but behind a
> provider-agnostic abstraction — and how the Deep Research capability (PR #37) and every other
> agent can consume it.
> Related: `docs/deep-research-plan.md` (PR #37), `docs/agentic-vs-autonomous-strategy.md`.

---

## TL;DR

1. **Build `koboi/media/` by cloning the `koboi/web/` package that PR #37 introduces.** Same
   shape: one ABC per capability, decorator-based registry mirroring `koboi/rag/registry.py`,
   a `mock` provider (offline default) + a `surplus` provider (the gateway). This is the lowest-risk
   path because the structural twin already exists and every seam (config, dep-injection, facade,
   orchestration, events, persistence, eval) is already proven by web/RAG.

2. **Expose generation primarily as deps-injected tools** (`generate_image` / `generate_video` /
   `generate_audio`, `deps=["media_provider"]`, `risk_level=MODERATE`, `idempotent=False`). This is
   the **agentic primitive**: any agent, in any mode (single-agent chat, orchestration node,
   subagent, MCP-exposed), can decide to generate. It flows through the existing
   risk/approval/audit/mode-block/journal pipeline for free. Deep Research is *one* consumer, not
   the only one — that is the platform answer.

3. **Normalize the calling/billing chaos inside one type pair.** Surplus exposes image/TTS/STT as
   **synchronous** (request → bytes) and video/music as **async jobs** (submit → poll → fetch), billed
   in **five different units** (per-image / per-megapixel / per-M-token / per-job / per-second /
   per-char). A single `await generate(req) -> MediaResult` facade hides the sync/async split and
   always returns a normalized `cost_usd` + `billing_unit` + `billing_quantity` + `url_expires_at`.
   An advanced `submit()/poll()/cancel()/fetch_artifact()` surface is exposed for callers that want
   async control.

4. **Cost control mirrors `ResearchBudget` + `CountingSearchProvider`.** A `MediaBudget`
   (per-modality caps **and** a USD ceiling) is enforced by a `CountingMediaProvider` proxy that
   fail-softs to an empty result on exhaustion — identical to the web/counting.py pattern.

5. **Auth is pluggable** (reuse `koboi/llm/auth.py` `BearerAuth`/`AuthStrategy`). Crucially, Surplus
   supports **x402 / MPP pay-per-request (USDC)** natively — this is the wedge for koboi's
   *trustworthy unattended autonomy* positioning (agents that pay-per-inference without a standing
   key).

6. **Phase it:** W0 = package + types + `mock` + `surplus` image-only (sync, easiest) + tools →
   W1 = video + music (async job envelope) → W2 = audio TTS/STT → W3 = Deep Research integration →
   W4 = orchestration media nodes → W5 = TUI/server/eval.

---

## 1. The gateway: what Surplus Intelligence actually is

Source: live `GET /v1/models` (345 models) + the four API-reference doc pages, fetched 2026-07-13.

Surplus Intelligence is an **OpenRouter-clone inference marketplace** ("open order book for AI
models") that resells many providers behind one OpenAI-compatible API and settles USDC to the
winning seller. It is **not** media-first, but it does expose all media modalities as first-class
endpoints.

**Base URL:** `https://api.surplusintelligence.ai/v1` (OpenAI-compatible).

| Endpoint | Style | Modality |
|---|---|---|
| `POST /v1/images/generations` | **sync** | image |
| `POST /v1/images/edits` | **sync** | image (i2i) |
| `POST /v1/video/generations` | **async job** (202 + `poll_url`/`cancel_url`/`job_token`) | video |
| `POST /v1/music/generations` | **async job** | music / SFX |
| `POST /v1/audio/speech` | **sync** (OpenAI shape) | TTS |
| `POST /v1/audio/transcriptions` | **sync** (Whisper shape) | STT |
| `GET /v1/media/artifacts/:jobId/:index` | proxy download | any async artifact |
| `GET /v1/models` | catalog | all |

**Confirmed model brands present:** `venice-z-image-turbo` (Z-Image Turbo), `gpt-5-image` /
`gpt-5.4-image-2` (GPT-image, **token-metered**), `gemini-2.5-flash-image`, `flux.2-flex`,
`seedream-4.5`, `kling-v3/o3/2.6` (Kling, per-job), `veo3-*`, `sora-2-*`, `happyhorse-1-*` (Happy
Horse, per-second), `runway-gen4-5`, `seedance-*`, `wan-2-*`, `ltx-2-*`, ElevenLabs (TTS + music +
SFX), MiniMax (music v26 + speech-02). All brands the user named are present.

**Billing units (the core normalization problem):**

| Modality | Dominant unit on Surplus | Notes |
|---|---|---|
| Image (most) | **per-image** or **per-megapixel** | driven by `size`/`resolution` |
| Image (gpt-5-image) | **per-million-tokens** | served on the chat wire; token-metered |
| Video (Kling/Veo/Sora) | **per-job** | `duration_seconds` + resolution tier influence the quote |
| Video (Happy Horse/MiniMax) | **per-second** | via `duration_seconds` |
| Music / SFX | **per-job** | USDC microdollars |
| TTS | **per-char** (OpenAI convention; SI per-char price unconfirmed) | sync |
| STT | **per-minute** (inferred) | sync |

All media is priced in **USDC microdollars** (`estimated_cost_usdc` / `max_cost_usdc`); LLMs are
per-token. Sellers set either `price_input_per_1m`/`price_output_per_1m` (microdollars) or
`cost_multiplier` (fraction of reference price); a fee multiplier (basis points) is applied. Buyers
cap spend via header `X-Max-Price-Per-1M` or body `max_price_per_1m`, and can enforce a minimum
discount with a `/min{N}/` path prefix.

**Auth — four modes, pluggable:**

| Method | Header | Use case |
|---|---|---|
| API key | `Authorization: Bearer inf_xxx` | programmatic |
| Session cookie | (web UI) | humans |
| **x402** (USDC on Base) | `PAYMENT-SIGNATURE: <sig>` | **autonomous agents** — pay-per-request |
| **MPP** (Tempo) | `Authorization: Payment <cred>` | autonomous agents |

No auth header → **HTTP 402** advertising x402 + MPP. The pay-per-request crypto settlement is native
and is a strategic fit for unattended koboi agents (see §5, §10).

**Gotchas that shape the abstraction:**
- **Artifacts are short-lived:** presigned S3 URLs expire in **15 minutes**; the proxy
  `/v1/media/artifacts/...` lives **3 hours**; unfinished jobs expire after **30 minutes**. A
  generated URI is **not durable** — the platform must materialize bytes (to local FS / R2 / etc.)
  before the window closes, or persist a re-fetchable reference.
- **`job_token`** authorizes poll/cancel *without* the API key — enables a two-tier auth model
  (submitter key vs. poller token).
- **Content safety** is per-seller (`top_provider.is_moderated`); rejections surface as `failed`
  job status, HTTP 4xx, or a safety-substituted artifact. The abstraction must distinguish
  `safety_blocked` from hard errors.

---

## 2. Why a provider-agnostic abstraction (not a hardcoded Surplus client)

Two reasons force the abstraction:

1. **The unit/execution heterogeneity in §1.** Even *within* Surplus, image is sync-per-image,
   video is async-per-job, TTS is sync-per-char. Hardcoding this into each tool means duplicating
   the sync/async, polling, cost-normalization, and URL-materialization logic across image/video/
   audio tools and again across any future provider. One `MediaBackend` ABC + one normalized type
   pair absorbs it once.
2. **Platform, not a Surplus plugin.** The user's explicit constraint: Surplus is *the* gateway
   today, but the integration must be **provider-agnostic** so a future direct-provider (e.g. a
   self-hosted ComfyUI/SD endpoint, a Cloudflare Workers-AI image model, an ElevenLabs-direct
   account) registers the same way. This is exactly the `koboi/web/` design: `brave`/`firecrawl`/
   `ddg` are interchangeable search providers behind one `BaseSearchProvider` ABC; `surplus` is the
   first `BaseImageProvider`/`BaseVideoProvider`/`BaseAudioProvider` behind one ABC each.

The design lesson from PR #37's `web/` package is that **the gateway is just another provider in the
registry** — never named in tool code, never imported by the facade directly.

---

## 3. Proposed architecture: `koboi/media/`

Mirror `koboi/web/` (PR #37) and `koboi/rag/registry.py` exactly. The web package is the closest
structural analog: same ABC-per-capability, same decorator registry, same per-provider YAML sub-dict,
same thin-delegating tools, same counting-proxy budget enforcement.

### 3.1 Package layout

```
koboi/media/
  __init__.py        # imports providers/* so @register_* fire on import (idempotent)
  base.py            # BaseImageProvider / BaseVideoProvider / BaseAudioProvider ABCs
  types.py           # MediaRequest, MediaResult, MediaUnit, MediaJob, MediaBudget
  registry.py        # ProviderRegistry + @register_image/video/audio_provider + build_media + load_custom_components
  budget.py          # MediaBudget + CountingMediaProvider proxy (mirror web/providers/counting.py)
  store.py           # MediaStore — artifact manifest + URI materialization (the SourceStore analog)
  providers/
    __init__.py
    mock.py           # offline default (deterministic placeholder bytes) — mirrors web/providers/mock.py
    surplus.py        # the gateway backend; implements all three ABCs (mirrors Firecrawl implementing search+fetch)
koboi/tools/builtin/
  media.py            # generate_image / generate_video / generate_audio / transcribe_audio / generate_music
```

### 3.2 The ABCs — `koboi/media/base.py` (mirror `web/base.py:21-41`)

One ABC per capability. A provider may implement several (Surplus implements all three — same
precedent as Firecrawl registering both search + fetch at `web/providers/firecrawl.py:41,104`).

```python
"""koboi/media/base.py -- generation-provider ABCs (image/video/audio)."""
from abc import ABC, abstractmethod
from .types import MediaRequest, MediaResult


class BaseImageProvider(ABC):
    """text/image -> image (synchronous on Surplus; may be async elsewhere)."""

    @abstractmethod
    async def generate_image(self, req: MediaRequest) -> MediaResult: ...


class BaseVideoProvider(ABC):
    """text/image -> video (async job on Surplus; submit/poll/fetch)."""

    @abstractmethod
    async def generate_video(self, req: MediaRequest) -> MediaResult: ...

    # Advanced surface for callers that want async control (default = block until terminal).
    async def submit_video(self, req: MediaRequest) -> "MediaJob": ...
    async def poll_video(self, job: "MediaJob") -> "MediaJob": ...
    async def cancel_video(self, job: "MediaJob") -> None: ...


class BaseAudioProvider(ABC):
    """text -> speech (TTS, sync) / audio bytes -> text (STT, sync) / text -> music (async)."""

    @abstractmethod
    async def synthesize_speech(self, req: MediaRequest) -> MediaResult: ...   # TTS

    async def generate_music(self, req: MediaRequest) -> MediaResult: ...       # async (default: raise)
    async def transcribe(self, audio: bytes, **opts) -> str: ...                 # STT (default: raise)
```

A security/contract module docstring (like `web/base.py:1-12`'s SSRF contract) should document the
**provenance + content-safety contract** every provider must honor (no prompt injection into the
returned artifact metadata; respect `safety_blocked`; never return a URI without an expiry).

### 3.3 The registry — `koboi/media/registry.py` (mirror `web/registry.py` + `rag/registry.py`)

Verbatim twin of the web registry. Three module-level registries, three decorators, the same
`_SECRET_KEYS` + `_redact` + `_merged_provider_conf` + `_resolve_kwargs` + `_build_provider`
helpers.

```python
"""koboi/media/registry.py -- decorator-based generation-provider registries."""
import importlib

_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "x_payment_signature"})


class ProviderEntry:
    __slots__ = ("cls", "parameters", "description", "config_aliases", "inject")
    # ... identical to web/registry.py:38-55 ...


class ProviderRegistry:
    # ... identical to web/registry.py:58-99 ...


image_provider_registry = ProviderRegistry("image")
video_provider_registry = ProviderRegistry("video")
audio_provider_registry = ProviderRegistry("audio")


def register_image_provider(name, description="", *, config_aliases=None, inject=None):
    def decorator(cls):
        image_provider_registry.register(name, cls, description=description,
                                         config_aliases=config_aliases, inject=inject)
        return cls
    return decorator


register_video_provider = ...   # same shape, video_provider_registry
register_audio_provider = ...   # same shape, audio_provider_registry


def build_image_provider(media_conf: dict | None) -> BaseImageProvider | None:
    return _build_provider(image_provider_registry, "image",
                           (media_conf or {}).get("image", {}) or {}, fallback_name="mock")


build_video_provider = ...   # fallback "mock"
build_audio_provider = ...   # fallback "mock"


def load_custom_components(custom_modules: list[str]) -> None:
    """Import modules so @register_* decorators fire on import (mirror web/registry.py:296-309)."""
    for module_path in custom_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning("Failed to import custom media module '%s': %s", module_path, e)
```

### 3.4 Config — a `media:` section (mirror `WebConfig` at `config_models.py:156-171`)

Pydantic-cosmetic (runtime reads it as a plain dict via `config.get("media", ...)`), exactly like
`WebConfig`/`RagConfig`:

```python
class MediaConfig(BaseModel):
    model_config = {"extra": "ignore"}
    enabled: bool = False
    image: dict = Field(default_factory=dict)   # {provider: surplus, surplus: {api_key, ...}}
    video: dict = Field(default_factory=dict)
    audio: dict = Field(default_factory=dict)
    budget: dict = Field(default_factory=dict)  # {max_cost_usd, max_images, max_video_seconds, ...}
    storage: dict = Field(default_factory=dict) # {backend: local|r2, ...} — where materialized bytes go
    custom_modules: list[str] = Field(default_factory=list)
```

Example YAML:

```yaml
media:
  enabled: true
  image:
    provider: surplus
    model: venice-z-image-turbo        # per-image
    surplus:
      base_url: ${SURPLUS_BASE_URL:https://api.surplusintelligence.ai/v1}
      api_key: ${SURPLUS_API_KEY:}
      auth_mode: bearer                # bearer | x402 | mpp
  video:
    provider: surplus
    model: kling-v3                    # per-job
    surplus: { api_key: ${SURPLUS_API_KEY:} }
  audio:
    provider: surplus
    tts_model: venice-kokoro-tts       # sync
    music_model: venice-minimax-music-v26  # async
    surplus: { api_key: ${SURPLUS_API_KEY:} }
  budget:
    max_cost_usd: 5.00
    max_images: 20
    max_video_seconds: 60
    max_audio_seconds: 120
  storage:
    backend: local                     # local | r2 | memory
    dir: ./media_artifacts
  custom_modules:
    - mycorp.media_providers.comfyui
```

### 3.5 Facade wiring — a `_build_media()` peer of `_build_rag` (mirror §6 of the main-branch audit)

Add a module-level `_build_media(config, logger)` near `_build_rag` (`facade.py:797`), an
`AgentAssembler.build_media()` method, a call in `build()` between `build_tools()` and `build_hooks()`,
then **one** `set_dep` seam:

```python
self.tools.set_dep("media_provider", self.media)   # mirror set_dep("sandbox", ...) at facade.py:1108
```

Close any HTTP transport in `KoboiAgent.close()` (duck-typed `hasattr(p, "close")`, mirroring the
RAG rerank close at `facade.py:253-260`). Orchestration gets it for free if the same `set_dep` line
is added to `_build_tools_from_config` (`orchestration/factory.py:293-357`).

---

## 4. The normalized type pair (absorbs §1's chaos)

```python
"""koboi/media/types.py -- normalized generation request/result."""

class MediaUnit(str, Enum):
    IMAGE = "image"          # per-image
    MEGAPIXEL = "megapixel"  # per-megapixel
    TOKEN = "token"          # per-million-tokens (gpt-5-image)
    JOB = "job"              # per-job (video, music)
    SECOND = "second"        # per-second (Happy Horse video, billed audio duration)
    CHAR = "char"            # per-char (TTS)
    MINUTE = "minute"        # per-minute (STT)


@dataclass
class MediaRequest:
    modality: str                   # "image" | "video" | "speech" | "music" | "transcription"
    prompt: str
    model: str | None = None        # overrides config default
    # per-modality params (modality-specific; validated against /v1/models supported_parameters)
    n: int | None = None            # image count
    size: str | None = None         # image
    resolution: str | None = None   # image/video (1K/2K/4K, 480p-4k)
    quality: str | None = None      # image (low/med/high)
    aspect_ratio: str | None = None # video (16:9 ...)
    duration_seconds: float | None = None  # video/music
    voice: str | None = None        # speech/music
    language_code: str | None = None
    lyrics_prompt: str | None = None
    force_instrumental: bool | None = None
    input_images: list[str] | None = None  # i2v / image-edit (URLs or data URIs)
    response_format: str | None = None     # "b64_json" | "url" (sync image only)
    # universal
    idempotency_key: str | None = None
    webhook_url: str | None = None          # async only
    metadata: dict = field(default_factory=dict)


@dataclass
class MediaResult:
    request_id: str
    modality: str
    status: str                     # "ok" | "rejected" | "failed"
    data: bytes | None = None       # materialized bytes (sync, or after fetch_artifact)
    url: str | None = None          # source URI (NOT durable — see url_expires_at)
    url_expires_at: float | None = None
    content_type: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    # normalized cost (the key unification)
    cost_usd: Decimal | None = None
    billing_unit: MediaUnit | None = None
    billing_quantity: float | None = None
    raw_usage: dict = field(default_factory=dict)
    # async-job surface
    job_id: str | None = None
    job_token: str | None = None
    # safety
    safety_blocked: bool = False
    rejection_reason: str | None = None
    raw: dict = field(default_factory=dict)

    async def fetch_bytes(self) -> bytes: ...   # follows url; caller MUST call before url_expires_at
```

The two non-obvious fields that save real bugs: **`billing_unit`** (a budget guard must not compare
tokens against dollars) and **`url_expires_at`** (forces callers to materialize bytes before the
15-min/3-hour window closes).

### 4.1 The sync/async facade

The public surface is **always-blocking** `await provider.generate_image(req) -> MediaResult`. Sync
backends (image/TTS/STT) return immediately; async backends (video/music) internally submit, poll
until terminal, fetch the artifact, and return — absorbing the job envelope. Callers that want async
control (e.g. a server streaming progress) use `submit_video`/`poll_video`/`cancel_video` directly.
This is the recommendation from the external research: one uniform facade + an opt-in async surface.

---

## 5. How generation is exposed — "not always tools" (the platform answer)

The user's explicit ask: *integration must be more agentic than just "deep research invokes a tool";*
*research which exposure is appropriate.* The answer is a **"who decides to generate?"** matrix.
Generation is one capability; the four exposure tiers differ only in *who* triggers it. All four
share the same `media_provider` — they are compositions, not separate systems.

| Tier | Who decides | Surface | When it's right |
|---|---|---|---|
| **1. Tool** | **the model** | `generate_image`/`generate_video`/`generate_audio` deps-injected tools | the **default / primary**. Any agent, any mode, decides mid-turn. Most agentic. |
| **2. Programmatic API** | a developer's code | `agent.media.generate(req)` direct call | batch/cron/server endpoints where you don't want the LLM in the loop (or its decision cost). E.g. a daily auto-generated video briefing. |
| **3. Orchestration node** | a workflow graph | a "media" `AgentDef` node in a DAG/workflow_graph | structured pipelines where media is a deterministic step with downstream consumers (research → diagram node → voiceover node → assembly node). |
| **4. Execution-mode step** | a mode/policy | `execution.mode: deep_research` with `research.capabilities: [web, image, video, audio]` auto-producing a multimedia briefing | the named use case: a cited report *plus* an explanatory figure, a 10s summary video, a voiceover. |

**Recommendation: build Tier 1 + Tier 2 as the core (they share one provider); Tier 3 and Tier 4 are
thin compositions on top.** Tier 1 is the agentic primitive — it makes generation available across
the entire platform (single-agent chat, orchestration subagents, MCP-exposed tools, subagents) and
flows through the existing risk/approval/audit/mode-block/journal pipeline for **zero** loop/pipeline
changes. Tiers 3–4 reuse the same provider and tools; they just decide *when* to call them.

This is the platform framing: **Surplus-as-generation-gateway is not a Deep Research feature — it is
an agent capability.** Deep Research is the most compelling *first consumer*, not the only one.

### 5.1 Tier 1 — the tools (`koboi/tools/builtin/media.py`, mirror `web.py:88-116`)

```python
"""koboi/tools/builtin/media.py -- generation tools (thin delegating wrappers)."""
from koboi.tools.registry import tool
from koboi.types import RiskLevel


@tool(
    name="generate_image",
    group="media",
    description="Generate an image from a prompt. Returns a local artifact URI + dimensions + cost.",
    parameters={"type": "object",
                "properties": {"prompt": {"type": "string"}, "size": {"type": "string"},
                               "n": {"type": "integer"}},
                "required": ["prompt"]},
    risk_level=RiskLevel.MODERATE,        # billed side-effect
    deps=["media_provider"],
    idempotent=False,                     # never silently double-fire on crash-resume
)
async def generate_image(prompt: str, size: str | None = None, n: int = 1,
                         _deps: dict | None = None, _tool_config: dict | None = None) -> str:
    from koboi.media.types import MediaRequest
    provider = (_deps or {}).get("media_provider")
    if provider is None:
        return "Error: no media_provider configured (enable media.image)"
    req = MediaRequest(modality="image", prompt=prompt, size=size, n=n,
                       model=_tool_config.get("image_model"))
    try:
        result = await provider.generate_image(req)
    except Exception as e:
        return f"Error: image generation failed — {e}"
    return _format_media_result(result)    # "Image saved: ./media_artifacts/<id>.png (1024x1024, $0.012)"
```

`generate_video` and `generate_audio` (TTS) are identical in shape; `generate_video` is
`risk_level=DESTRUCTIVE` (cost/latency) and `generate_audio` is `MODERATE`. `transcribe_audio`
(STT, input→text) and `generate_music` (async) are optional siblings. The body is near-trivial
because the provider does the work — exactly the web_search precedent.

**Why `idempotent=False` matters:** per `types.py:28-32`, `_repair_interrupted_turn` skips
re-execution for non-idempotent tools on crash-resume — generation is a *billed* side effect that
must not silently double-fire. This is already the convention; we just set the flag.

### 5.2 Tier 2 — programmatic API

A thin method on `KoboiAgent` (the facade already exposes `mcp_status()`, `close()`, etc.):

```python
async def media_generate(self, req: MediaRequest) -> MediaResult:
    if self._media is None:
        raise AgentError("media not configured (enable media:)")
    return await self._media.dispatch(req)   # routes to generate_image/video/speech by modality
```

For server consumers, a `POST /v1/media/generate` endpoint (same shape as `/v1/chat/stream`) that
returns the artifact URI — the natural home for batch/cron and the `koboi-use-cases` sector apps.

---

## 6. The Deep Research use case (the named scenario)

"Deep research on top of dynamic workflow that can invoke generation." PR #37's `_run_deep_research`
(`orchestrator.py:830-1069`) is plan → per-node search/fetch waves → CoverageEvaluator → drill →
cited text synthesis. Three concrete insertion points (file:line from the PR #37 worktree):

### Insertion A — post-synthesis media step (loosest; the "multimedia briefing")

Between `_synthesize_research` (`orchestrator.py:1028`) and the terminal event, add a
`_generate_research_media(query, ctx, report)` step guarded by `research.media: {enabled, kinds,
when}`. It runs one LLM call (or heuristic) to pick the single best supporting image / 10s video /
voiceover, calls the media provider (wrapped in `CountingMediaProvider`), appends
`![Figure 1: ...](uri)` to the report, and adds a `media_artifacts` list to the terminal
`OrchestrationCompleteEvent.metadata` (`orchestrator.py:1062-1068`). Best ratio of payoff to coupling.

### Insertion B — per-node generation tool (the building block)

Extend the per-node tool bundle so any research node can generate inline:
- Add the media tools to `RESEARCH_TOOLS_CONFIG` (`research.py:28-29`) or accept a
  `research.media.tools` override → `{"builtin": ["web_search","web_fetch","generate_image"]}`.
- Inject the dep at `factory.py:345-351` (the "W4" block) — add `if media_provider is not None:
  registry.set_dep("media_provider", media_provider)`.
- Thread `media_provider` through `AgentFactory.create_configured_agent` (`factory.py:199-261`)
  exactly like `search_provider`/`fetch_provider`.
- Build + counting-wrap it once at `orchestrator.py:875-876`.
- Extend `RESEARCH_NODE_PREAMBLE` (`research.py:34-45`) with a generation clause (the known tool-nudge
  gap means nodes under-invoke tools without a system-prompt push).
- Emit `ImageGeneratedEvent` in the per-node switch at `orchestrator.py:509-517`.

### Insertion C — `research.capabilities` flag (cleanest config surface)

A `research.capabilities: list[str]` knob (default `["web"]`; opt-in `["web","image","video","audio"]`)
drives: which tools enter the bundle, which providers get built + counting-wrapped, which events
fire, and which `MediaBudget` counters apply. One-line opt-in: "briefing with an explanatory image".

**Recommendation: ship B + C together (capabilities flag drives the per-node tools), then A as a
follow-on for the polished multimedia briefing.** B is the highest-fidelity mirror of the existing
pattern — every seam already exists. A is the user-facing payoff. C is the config surface that gates
both.

### 6.1 Budget + cost control (mirror `ResearchBudget` + `CountingSearchProvider`)

Generation is expensive and billed heterogeneously, so budget enforcement is non-negotiable. Mirror
the web/counting.py pattern exactly: a `MediaBudget` dataclass + a `CountingMediaProvider` proxy that
fail-softs on exhaustion.

```python
"""koboi/media/budget.py -- per-modality + USD cost caps, enforced at the provider proxy."""

@dataclass
class MediaBudget:
    max_cost_usd: float = 5.0
    max_images: int = 20
    max_video_seconds: float = 60.0
    max_audio_seconds: float = 120.0
    used_cost_usd: Decimal = Decimal("0")
    used_images: int = 0
    used_video_seconds: float = 0.0
    used_audio_seconds: float = 0.0

    def remaining(self, modality: str, qty: float = 1.0, est_cost: float = 0.0) -> bool:
        if self.used_cost_usd + est_cost > self.max_cost_usd: return False
        if modality == "image" and self.used_images >= self.max_images: return False
        if modality == "video" and self.used_video_seconds + qty > self.max_video_seconds: return False
        if modality in ("speech", "music") and self.used_audio_seconds + qty > self.max_audio_seconds: return False
        return True

    def record(self, result: MediaResult) -> None:
        if result.cost_usd: self.used_cost_usd += result.cost_usd
        # ... accrue per modality using result.billing_quantity ...
```

`CountingMediaProvider` wraps the real provider and short-circuits to a `status="rejected"` result
when `remaining()` is False — never crashes the node (same intent as
`web/providers/counting.py:1-8`). The USD ceiling is the single most important cap because it is
**unit-agnostic** — it bounds spend regardless of whether the model bills per-image or per-second.
For Deep Research, `MediaBudget` extends `ResearchBudget` (or composes it) and round-trips through
`ResearchContext.to_json/from_json`.

---

## 7. Platform concerns (events, persistence, TUI, server, eval, sandbox, safety)

- **Events** (`events.py`, mirror `SearchEvent` at `117-149`): `ImageGeneratedEvent`,
  `VideoGeneratedEvent`, `AudioGeneratedEvent` carrying `{model, uri, content_type,
  duration_seconds|width/height, cost_usd, billing_unit}`. Add to the `StreamEvent` union + the
  per-node emission switch (`orchestrator.py:509-517`).
- **Persistence / URI materialization** (the SourceStore analog): a `MediaStore` writes a manifest
  `{artifact_id, modality, model, local_path|remote_uri, content_hash, cost_usd, provenance}`. **Because
  gateway URIs expire in 15 min / 3 h, the platform must materialize bytes** (to local FS / R2 / S3)
  on generation, *before* persisting the reference — never store a bare Surplus URL as durable state.
  `media.storage.backend: local|r2|memory` controls this; this connects to the existing
  "koboi-as-consumer, docs in R2/S3" memory. A `media_artifacts` SQLite table (mirror
  `research_context`) survives restart.
- **TUI**: render image inline (Textual image widget), audio as a waveform + play, video as a poster
  frame + open. A new f-screen or a media panel in the chat view.
- **Server/SSE**: stream `MediaGeneratedEvent` over the existing SSE channel; a `POST /v1/media/generate`
  REST endpoint (Tier 2); `Idempotency-Key` dedup like `/v1/jobs`. Long video jobs map naturally to
  the async `jobs` subsystem (webhook on terminal status).
- **Eval**: a `MediaProvenanceScorer` (mirror `DeepResearchFaithfulnessScorer`) reads
  `metadata['media_artifacts']` and checks (a) every cited `[fig:n]` resolves to a stored artifact,
  (b) cost is within budget, (c) no `safety_blocked` slipped through.
- **Sandbox/egress**: generation needs **network egress** to Surplus. Per the sandbox model
  (`sandbox.network_isolation`), media calls must run where egress is allowed (the provider is built
  in the facade, not in a restricted subprocess — so this is fine by default; just document it).
  `restricted` seccomp hard-deny would block generation — surface a clear config error.
- **Content-safety policy**: respect `safety_blocked`; add a guardrail option to *pre-screen* prompts
  (cheap text guardrail) before the billed call, and to *post-screen* artifacts for provenance
  watermarking (C2PA) where the provider supports it.

---

## 8. Phased plan

| Wave | Scope | Outcome | Gates |
|---|---|---|---|
| **W0** | `koboi/media/` package: `base.py`, `types.py`, `registry.py`, `store.py`; `providers/mock.py` (deterministic placeholder bytes); `providers/surplus.py` **image-only** (sync `/v1/images/generations`); `tools/builtin/media.py` `generate_image`; `_build_media` facade seam; `MediaConfig`. | Any agent can `generate_image`; full pipeline (risk/approval/audit/mode/journal) exercised end-to-end on the cheapest modality. | pytest + cov≥80 + ruff/mypy/bandit clean; `mock` provider offline tests. |
| **W1** | `surplus.py` **video + music** (async job envelope: submit→poll→cancel→fetch; `MediaJob`); `generate_video` (DESTRUCTIVE) + `generate_music`; `CountingMediaProvider` + `MediaBudget`; URI materialization to `storage.backend`. | The hardest modality (async, per-job/per-second, expiring URIs) is solved once. | Live smoke against Surplus (key-gated); budget-exhaustion fail-soft test. |
| **W2** | `surplus.py` **audio**: TTS (sync `/v1/audio/speech`, optional streaming) + STT (`/v1/audio/transcriptions`); `generate_audio` + `transcribe_audio`. | All three named modalities complete. | Streaming TTS chunk test (mock). |
| **W3** | Deep Research integration: Insertion B + C (`research.capabilities`, per-node tools, `MediaBudget` merged with `ResearchBudget`, `RESEARCH_NODE_PREAMBLE` media clause, `MediaGeneratedEvent` emission). | "Briefing with an explanatory image" works end-to-end. | Deep-research eval gains a media-provenance scorer. |
| **W4** | Insertion A (post-synthesis multimedia step) + orchestration media nodes (`workflow_graph` media `AgentDef`). | Cited report + figure + 10s video + voiceover. | New `configs/multimedia_research_demo.yaml`. |
| **W5** | TUI media panel; `POST /v1/media/generate` + SSE; x402/MPP auth strategies; docs-align (CLAUDE.md + configs/CLAUDE.md). | Platform-complete; unattended pay-per-request works. | Server e2e; x402 auth unit test. |

Each wave is **additive / opt-in** (`media.enabled: false` default → zero behavior change), matching
the project's back-compat discipline. W0 alone is a shippable, useful increment.

---

## 9. Open decisions (need your call)

1. **Provider naming / gateway-neutrality.** Proposal: package = `koboi/media/`, first provider =
   `surplus`. Confirm this keeps the abstraction gateway-neutral (future ComfyUI / Workers-AI /
   ElevenLabs-direct register identically), vs. naming the package around Surplus.
2. **Artifact storage backend default.** `local` (FS under `media.storage.dir`) vs `r2`/`s3`. Given
   the "koboi-as-consumer, docs in R2/S3" memory, R2 may be the better default for server deploys —
   but local is simpler for W0. Proposal: `local` default, `r2` opt-in (reuse the existing R2/S3
   HTTP-fetch seam).
3. **Auth default + x402 timing.** Bearer for W0–W4; surface x402/MPP in W5 as the unattended-autonomy
   wedge — or pull it forward if autonomous payment is a near-term goal.
4. **Risk levels.** Image/TTS = `MODERATE`; video/music = `DESTRUCTIVE` (cost+latency). Confirm, or
   make all generation `DESTRUCTIVE` (stricter) given it is billed.
5. **Tier-2 REST surface.** Ship `POST /v1/media/generate` in W0 or defer to W5? Proposal: defer —
   keep W0 tool-only (the agentic surface), add REST when server consumers materialize.
6. **Scope of "audio".** TTS + STT + music, or TTS-only for v1? Proposal: TTS in W2, music with W1
   (shares the async envelope), STT optional later (it is *input*, not generation).
7. **Deep Research coupling.** Insertion B+C in W3 then A in W4 (recommended), or A-first for a
   faster demo?

---

## 10. Strategic fit (one paragraph)

This capability extends two of koboi's existing wedges at once. (1) **Trustworthy unattended
autonomy** — Surplus's native x402/MPP pay-per-request means an autonomous koboi agent can generate
media without a standing API key, settling each inference in USDC; combined with the existing
sandbox/journal/resume + `MediaBudget` USD ceiling, unattended generation becomes safe-by-design.
(2) **Platform, not a feature** — by exposing generation as deps-injected tools (Tier 1) shared by
single-agent chat, orchestration, subagents, MCP, and Deep Research, media becomes an *agent
capability* rather than a Deep-Research-only addon, matching the open-core/enterprise-separate-repo
strategy (the abstraction lives in OSS Apache-2.0 core; a future enterprise media-gateway tier could
add SSO/RBAC/audit on top of the same `BaseImageProvider` seam).
