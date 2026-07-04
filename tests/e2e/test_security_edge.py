"""Security & HTTP edge-case suite (category: security) — 20 fast, deterministic
cases. No LLM calls, no throttling. Each case asserts an exact status code.

Grounded in the actual route + middleware behavior:
  * auth (auth.py): OPEN_PATHS bypass; else ``Bearer <token>`` validated, 401 on
    missing/malformed/invalid.
  * chat/stream (app.py): ``user_message()`` ValueError → 400; bad X-Session-Id → 400.
  * session/job lookups: valid-format-but-missing id → 404; cancel a terminal job → 409.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import API_KEY

pytestmark = pytest.mark.e2e


def _auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _open() -> dict:
    return {"Content-Type": "application/json"}


# Each case: (id, method, path, json_body, extra_headers, expected_status)
CASES: list[tuple[str, str, str, dict | None, dict, int]] = [
    # --- Auth (6) ---
    ("auth_no_header", "POST", "/v1/sessions", None, {"Content-Type": "application/json"}, 401),
    ("auth_wrong_key", "POST", "/v1/sessions", None, {"Content-Type": "application/json", "Authorization": "Bearer wrong_key_xyz"}, 401),
    ("auth_bearer_no_space", "POST", "/v1/sessions", None, {"Content-Type": "application/json", "Authorization": "Bearer"}, 401),
    ("auth_not_bearer_scheme", "POST", "/v1/sessions", None, {"Content-Type": "application/json", "Authorization": "Basic abc123"}, 401),
    ("auth_lowercase_scheme", "POST", "/v1/sessions", None, {"Content-Type": "application/json", "Authorization": f"bearer {API_KEY or 'x'}"}, 401),
    ("auth_healthz_open", "GET", "/healthz", None, {}, 200),
    # --- Path traversal via X-Session-Id (5) → is_safe_session_id rejects → 400 ---
    ("path_traversal_dotdot", "POST", "/v1/chat/stream", {"message": "hi"}, {"X-Session-Id": "../etc/passwd", **_auth()}, 400),
    ("path_traversal_abs", "POST", "/v1/chat/stream", {"message": "hi"}, {"X-Session-Id": "/etc/passwd", **_auth()}, 400),
    ("path_traversal_slash", "POST", "/v1/chat/stream", {"message": "hi"}, {"X-Session-Id": "a/b", **_auth()}, 400),
    ("path_traversal_space", "POST", "/v1/chat/stream", {"message": "hi"}, {"X-Session-Id": "bad session", **_auth()}, 400),
    ("path_traversal_dotdot_only", "POST", "/v1/chat/stream", {"message": "hi"}, {"X-Session-Id": "..", **_auth()}, 400),
    # --- Body validation on /chat/stream (4) → 400 ---
    ("body_empty_object", "POST", "/v1/chat/stream", {}, _auth(), 400),
    ("body_empty_message", "POST", "/v1/chat/stream", {"message": ""}, _auth(), 400),
    ("body_empty_messages", "POST", "/v1/chat/stream", {"messages": []}, _auth(), 400),
    ("body_messages_no_user_role", "POST", "/v1/chat/stream", {"messages": [{"role": "system", "content": "x"}]}, _auth(), 400),
    # --- HTTP edge: valid-format-but-missing ids → 404 (5) ---
    ("get_unknown_session", "GET", "/v1/sessions/nonexistent-sess-1", None, _auth(), 404),
    ("delete_unknown_session", "DELETE", "/v1/sessions/nonexistent-sess-2", None, _auth(), 404),
    ("approve_unknown_session", "POST", "/v1/sessions/nonexistent-sess-3/approve", {"approval_id": "nonexistent-approval", "decision": "approve"}, _auth(), 404),
    ("get_unknown_job", "GET", "/v1/jobs/nonexistent-job-1", None, _auth(), 404),
    ("cancel_unknown_job", "POST", "/v1/jobs/nonexistent-job-2/cancel", None, _auth(), 404),
]


@pytest.mark.parametrize(
    "method,path,body,headers,expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
async def test_security_edge(client, method, path, body, headers, expected):
    if not API_KEY and expected == 401:
        pytest.skip("KOBOI_API_KEY not set — cannot test 401 (server is in dev-allow mode)")
    # Use the client fixture's base_url; headers override defaults.
    full_headers = {"Content-Type": "application/json"}
    full_headers.update(headers)
    r = await client.request(method, path, json=body, headers=full_headers)
    assert r.status_code == expected, (
        f"{method} {path}: expected {expected}, got {r.status_code}. body={r.text[:300]}"
    )
