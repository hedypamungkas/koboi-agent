"""koboi/server/schema -- Pydantic request/response models + error envelope.

Pure Pydantic v2 (no FastAPI import) so it unit-tests without the ``api`` extra.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    """POST /v1/chat/stream body.

    Accepts EITHER ``{"message": "..."}`` OR ``{"messages": [{"role","content"}]}``
    (the last user-role message wins). ``user_message()`` resolves the effective
    prompt and raises ``ValueError`` if no non-empty user message is present.
    """

    model_config = {"extra": "ignore"}

    message: str | None = Field(default=None, max_length=65536)  # H6: bound body size
    messages: list[dict[str, Any]] | None = Field(default=None, max_length=50)  # H6: bound turn count

    # G2: per-request mode + iteration cap. mode + the cap are enforced in the
    # route handler (400 invalid_mode / clamped to server.limits.max_iterations_cap);
    # the ge=1 floor is Pydantic (422). None = config default (config-only path
    # unchanged).
    mode: str | None = Field(default=None)
    max_iterations: int | None = Field(default=None, ge=1)

    def user_message(self) -> str:
        if isinstance(self.message, str) and self.message.strip():
            return self.message
        if isinstance(self.messages, list):
            for entry in reversed(self.messages):
                if isinstance(entry, dict) and entry.get("role") == "user":
                    content = entry.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content
        raise ValueError("missing non-empty 'message' or user-role 'messages' entry")


class CreateSessionResponse(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


class SessionDeletedResponse(BaseModel):
    session_id: str
    evicted: bool


class SessionListItem(BaseModel):
    session_id: str
    title: str | None = None
    owner: str | None = None
    message_count: int = 0
    model: str | None = None
    agent_name: str | None = None
    first_message: str | None = None
    updated_at: float | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]


class SessionForkResponse(BaseModel):
    session_id: str
    source_session_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    retriable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ReadyzCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class ReadyzResponse(BaseModel):
    status: Literal["ok", "down"]
    checks: list[ReadyzCheck]


class ApproveRequest(BaseModel):
    """POST /v1/sessions/:id/approve body."""

    model_config = {"extra": "ignore"}

    approval_id: str
    decision: Literal["approve", "deny"] = "approve"
    scope: Literal["once", "always"] = "once"


class ApproveResponse(BaseModel):
    approval_id: str
    resolved: bool


class TransferRequest(BaseModel):
    """POST /v1/sessions/:id/transfer body (B1 handover takeover).

    ``operator`` is the new owner (defaults to the caller's api-key id).
    ``handover_id`` is for audit correlation with the ``HandoverEvent``; it is NOT
    required (there is no Future to resolve -- the bot's run already ended).
    """

    model_config = {"extra": "ignore"}

    operator: str | None = None
    handover_id: str | None = None


class TransferResponse(BaseModel):
    session_id: str
    transferred: bool
    owner: str


class JobSubmitRequest(BaseModel):
    """POST /v1/jobs body."""

    model_config = {"extra": "ignore"}

    message: str = Field(max_length=65536)  # H6: bound body size
    session_id: str | None = None

    # G2: per-request mode + iteration cap; see ChatStreamRequest. Jobs always
    # reject yolo (allow_yolo=False) regardless of server.allowed_modes — an
    # autonomous (no-HITL) run must not drop the approval gate + rate limiter.
    mode: str | None = Field(default=None)
    max_iterations: int | None = Field(default=None, ge=1)
    # Workflow export/import (v1): run a stored workflow bundle as an autonomous
    # job instead of the server-level config. Owner-scoped; validated at submit.
    workflow_ref: str | None = Field(default=None)
    # v2/v3: run the job in cache/replay mode (file-backed response cache) so the
    # completed run can be captured into a byte-deterministic bundle. Plain jobs
    # build a fresh per-job agent (_execute_plain_cache_job); workflow_ref jobs
    # hydrate the captured sidecar.
    replay_mode: str | None = Field(default=None)


class WorkflowCreateRequest(BaseModel):
    """POST /v1/workflows body."""

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    bundle: str = Field(min_length=1)  # the workflow bundle YAML text


class WorkflowListItem(BaseModel):
    name: str
    description: str | None = None
    created_at: float
    updated_at: float


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowListItem]


class WorkflowResponse(BaseModel):
    name: str
    description: str | None = None
    owner: str
    created_at: float
    updated_at: float


class CaptureRequest(BaseModel):
    """POST /v1/jobs/{id}/capture body."""

    model_config = {"extra": "ignore"}

    name: str | None = None
    with_cache: bool = False
    redact_cache: bool = False


class CaptureResponse(BaseModel):
    name: str
    description: str | None = None
    cache_entries: int = 0
    cache_redacted: bool = False
    created_at: float
    updated_at: float


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    session_id: str
    result: dict[str, Any] | None = None
    error: str | None = None
    error_class: str | None = None
    retriable: bool = False


# --- G6: MCP server management (/v1/sessions/{id}/mcp/servers) ---


class McpServerCreateRequest(BaseModel):
    """POST body to attach an MCP server to a session at runtime."""

    model_config = {"extra": "ignore"}

    transport: Literal["stdio", "streamable-http"] = "stdio"
    # stdio
    command: str = ""
    args: list[str] = Field(default_factory=list)
    # streamable-http
    url: str = ""
    auth: dict[str, Any] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # shared
    timeout: float = 30.0
    group: str | None = None
    risk_level: str = "safe"  # safe | moderate | destructive


class McpServerResponse(BaseModel):
    model_config = {"extra": "ignore"}

    id: str
    name: str = ""
    transport: str = "stdio"
    connected: bool = False
    tool_names: list[str] = Field(default_factory=list)
    server_info: dict[str, Any] = Field(default_factory=dict)


class McpServerListResponse(BaseModel):
    model_config = {"extra": "ignore"}

    servers: list[McpServerResponse] = Field(default_factory=list)


class MediaGenerateRequest(BaseModel):
    """POST /v1/media/generate body."""
    model_config = {"extra": "ignore"}
    modality: str = Field(default="image")
    prompt: str = Field(max_length=65536)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=20)
    size: str | None = None
    quality: str | None = None
    response_format: str | None = None
    aspect_ratio: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    audio: bool | None = None
    voice: str | None = None
    language_code: str | None = None
    lyrics_prompt: str | None = Field(default=None, max_length=65536)
    webhook_url: str | None = None
    idempotency_key: str | None = None
    session_id: str | None = None


class MediaGenerateResponse(BaseModel):
    """POST /v1/media/generate response."""
    model_config = {"extra": "ignore"}
    request_id: str
    modality: str
    status: str
    local_path: str | None = None
    url: str | None = None
    url_expires_at: float | None = None
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    cost_usd: float | None = None
    billing_unit: str | None = None
    billing_quantity: float | None = None
    safety_blocked: bool = False
    rejection_reason: str | None = None
    model: str | None = None


class MediaJobResponse(BaseModel):
    """GET /v1/media/jobs/{id} response."""
    model_config = {"extra": "ignore"}
    job_id: str
    status: str
    result: MediaGenerateResponse | None = None
