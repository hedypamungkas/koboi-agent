"""koboi/mcp/auth.py -- AuthStrategy implementations for MCP HTTP transports (G1).

Reuses the ``AuthStrategy`` ABC from ``koboi/llm/auth.py`` so MCP auth composes
the same way LLM-provider auth does. Adds an OAuth2 client-credentials / refresh
strategy with an in-memory token cache so enterprise MCP servers (Gmail, Drive,
Slack, GitHub, ...) whose short-lived tokens expire (~1h) stay usable without a
manual token rotation + agent restart.
"""

from __future__ import annotations

import time

import httpx

from koboi.llm.auth import AuthStrategy, BearerAuth


class OAuthClientCredentialsAuth(AuthStrategy):
    """OAuth2 token acquisition + refresh for ``StreamableHTTPMCPClient``.

    Supports two grant types:
      * ``client_credentials`` (default) -- machine-to-machine; no user.
      * ``refresh_token`` -- when a ``refresh_token`` is provided, refresh uses it
        (and an initial ``access_token`` may be seeded to avoid a first-call fetch).

    The token is cached in memory with its expiry (``expires_in`` seconds from the
    token endpoint, minus a 30s safety margin). ``apply()`` lazily ensures a live
    token; ``refresh(force=True)`` forces re-acquisition and is called on HTTP 401.
    """

    EXPIRY_SAFETY_SECONDS = 30.0

    def __init__(
        self,
        token_endpoint: str,
        client_id: str,
        client_secret: str = "",
        scopes: str = "",
        refresh_token: str = "",
        access_token: str = "",
        expires_in: float | None = None,
        timeout: float = 30.0,
    ):
        if not token_endpoint or not client_id:
            raise ValueError("OAuth requires token_endpoint and client_id")
        self._token_endpoint = token_endpoint
        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes = scopes
        self._refresh_token = refresh_token
        self._timeout = timeout
        self._access_token = access_token or ""
        # Absolute expiry epoch seconds; 0 == unknown/expired.
        self._expires_at = (time.monotonic() + expires_in - self.EXPIRY_SAFETY_SECONDS) if expires_in else 0.0

    # --- AuthStrategy ---

    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        self._ensure_token()
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    # --- token lifecycle ---

    def _ensure_token(self) -> None:
        # No token at all -> must fetch. Known-expired token -> refresh. A seeded token
        # with unknown lifetime (expires_at == 0.0) is used optimistically until a 401.
        if not self._access_token:
            self.refresh(force=False)
        elif self._expired():
            self.refresh(force=False)

    def _expired(self) -> bool:
        # Only "expired" when we have a known expiry that has passed.
        return self._expires_at != 0.0 and time.monotonic() >= self._expires_at

    def refresh(self, force: bool = True) -> None:
        """Re-acquire a token. With ``refresh_token`` set, uses the refresh grant;
        otherwise client_credentials. ``force`` is honored for 401 recovery."""
        data: dict[str, str] = {"client_id": self._client_id}
        if self._client_secret:
            data["client_secret"] = self._client_secret
        if self._scopes:
            data["scope"] = self._scopes
        if self._refresh_token:
            data["grant_type"] = "refresh_token"
            data["refresh_token"] = self._refresh_token
        else:
            data["grant_type"] = "client_credentials"

        resp = httpx.post(self._token_endpoint, data=data, timeout=self._timeout)
        if resp.status_code >= 400:
            raise OAuthError(f"token endpoint returned HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise OAuthError(f"token endpoint response missing access_token: {payload}")
        self._access_token = token
        # New refresh_token may be rotated by some providers.
        new_refresh = payload.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
        expires_in = payload.get("expires_in")
        self._expires_at = time.monotonic() + float(expires_in) - self.EXPIRY_SAFETY_SECONDS if expires_in else 0.0

    @property
    def supports_refresh(self) -> bool:
        """Whether this strategy can attempt recovery on a 401 (always true for OAuth)."""
        return True


class OAuthError(Exception):
    """Failure acquiring/refreshing an OAuth token."""


def build_mcp_auth(auth_config: dict | None) -> AuthStrategy | None:
    """Build an ``AuthStrategy`` from an MCP ``auth`` config dict (G1).

    - type ``none`` (or unset) -> ``None`` (no auth header)
    - type ``bearer``          -> static ``BearerAuth``
    - type ``oauth``           -> ``OAuthClientCredentialsAuth`` (client_credentials/refresh)
    """
    cfg = auth_config or {}
    auth_type = (cfg.get("type") or "none").lower()
    if auth_type == "bearer":
        token = cfg.get("token", "")
        return BearerAuth(token) if token else None
    if auth_type == "oauth":
        return OAuthClientCredentialsAuth(
            token_endpoint=cfg.get("token_endpoint", ""),
            client_id=cfg.get("client_id", ""),
            client_secret=cfg.get("client_secret", ""),
            scopes=cfg.get("scopes", ""),
            refresh_token=cfg.get("refresh_token", ""),
            access_token=cfg.get("access_token", ""),
            expires_in=cfg.get("expires_in"),
            timeout=float(cfg.get("timeout", 30.0)),
        )
    return None
