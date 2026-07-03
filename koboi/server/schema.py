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


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    session_id: str
    result: dict[str, Any] | None = None
    error: str | None = None
    error_class: str | None = None
    retriable: bool = False
