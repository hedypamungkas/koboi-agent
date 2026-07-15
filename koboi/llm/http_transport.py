"""koboi/llm/http_transport.py -- Shared async HTTP transport with error mapping."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from koboi.llm.auth import AuthStrategy
from koboi.llm.base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
)

_logger = logging.getLogger(__name__)

_DEFAULT_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 529})
_MAX_RETRIES = 2


class HttpTransport:
    def __init__(
        self,
        base_url: str,
        auth: AuthStrategy,
        default_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        max_retries: int = _MAX_RETRIES,
    ):
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._default_headers = {
            "Content-Type": "application/json",
            **(default_headers or {}),
        }
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def base_url(self) -> str:
        """Provider base URL (for telemetry / result attribution)."""
        return self._base_url

    async def post(self, path: str, body: dict) -> dict:
        url = f"{self._base_url}{path}"
        headers = self._auth.apply({**self._default_headers})

        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(url, json=body, headers=headers)
            except httpx.ConnectError as e:
                raise LLMConnectionError(f"Connection failed to {self._base_url}: {e}") from e
            except httpx.TimeoutException as e:
                raise LLMConnectionError(f"Request timed out after {self._timeout}s: {e}") from e

            if response.status_code < 400:
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    raise LLMResponseParseError(f"Invalid JSON in response: {e}") from e

            detail = self._extract_error_detail(response)
            retry_after = response.headers.get("retry-after")

            if response.status_code in _DEFAULT_RETRYABLE_STATUS and attempt < self._max_retries:
                last_error = f"HTTP {response.status_code}: {detail}"
                wait = float(retry_after) if retry_after else (2**attempt)
                _logger.warning(
                    "Retrying %s (status %d, attempt %d/%d, wait %.1fs): %s",
                    path,
                    response.status_code,
                    attempt + 1,
                    self._max_retries,
                    wait,
                    detail,
                )
                await asyncio.sleep(wait)
                continue

            self._raise_for_status(response.status_code, detail, retry_after)

        raise LLMServerError(f"Max retries exceeded: {last_error}")

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Single request (no retry) with auth + connect/timeout error mapping.

        Shared by ``get``/``delete``/``get_bytes`` for async-job poll/cancel/artifact-fetch.
        """
        headers = self._auth.apply({**self._default_headers})
        try:
            return await self._client.request(method, url, headers=headers, **kwargs)
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Connection failed to {url}: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Request timed out after {self._timeout}s: {e}") from e

    async def get(self, path: str, params: dict | None = None) -> dict:
        """GET ``{base_url}{path}`` (retry-less) -> parsed JSON. Used for async-job polling."""
        response = await self._request("GET", f"{self._base_url}{path}", params=params)
        return self._read_json(response)

    async def delete(self, path: str) -> dict:
        """DELETE ``{base_url}{path}`` (retry-less) -> parsed JSON. Used for async-job cancel."""
        response = await self._request("DELETE", f"{self._base_url}{path}")
        return self._read_json(response)

    async def get_bytes(self, url: str) -> bytes:
        """GET an absolute URL (follow redirects) -> raw bytes. Used for artifact download.

        Accepts a full URL (presigned S3 or the gateway proxy), so the caller controls the host.
        """
        response = await self._request("GET", url, follow_redirects=True)
        if response.status_code >= 400:
            self._raise_for_status(
                response.status_code,
                self._extract_error_detail(response),
                response.headers.get("retry-after"),
            )
        return response.content

    async def post_bytes(self, path: str, body: dict) -> bytes:
        """POST a JSON body -> raw bytes (e.g. TTS ``/audio/speech`` binary response)."""
        response = await self._request("POST", f"{self._base_url}{path}", json=body)
        if response.status_code >= 400:
            self._raise_for_status(
                response.status_code,
                self._extract_error_detail(response),
                response.headers.get("retry-after"),
            )
        return response.content

    async def post_form(self, path: str, files: dict, data: dict | None = None) -> dict:
        """POST a multipart form -> parsed JSON (e.g. STT ``/audio/transcriptions``).

        Sends auth-only headers (httpx sets the ``Content-Type`` multipart boundary; a JSON
        Content-Type here would corrupt the multipart body).
        """
        url = f"{self._base_url}{path}"
        headers = self._auth.apply({})
        try:
            response = await self._client.request("POST", url, headers=headers, files=files, data=data or {})
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Connection failed to {url}: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Request timed out after {self._timeout}s: {e}") from e
        return self._read_json(response)

    def _read_json(self, response: httpx.Response) -> dict:
        """Parse a JSON response, or raise an ``LLMError`` mapped from the HTTP status."""
        if response.status_code < 400:
            try:
                return response.json()
            except (json.JSONDecodeError, ValueError) as e:
                raise LLMResponseParseError(f"Invalid JSON in response: {e}") from e
        detail = self._extract_error_detail(response)
        self._raise_for_status(response.status_code, detail, response.headers.get("retry-after"))
        raise LLMResponseParseError(
            f"HTTP {response.status_code}: {detail}"
        )  # pragma: no cover - _raise_for_status always raises

    async def post_stream(self, path: str, body: dict) -> AsyncIterator[bytes]:
        """Stream SSE lines from a POST request. Yields raw bytes per line."""
        body["stream"] = True
        url = f"{self._base_url}{path}"
        headers = self._auth.apply({**self._default_headers})

        try:
            async with self._client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code >= 400:
                    detail = await response.aread()
                    self._raise_for_status(response.status_code, detail.decode()[:500])

                async for line in response.aiter_lines():
                    if line.strip():
                        yield line.encode()
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Stream connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Stream timed out: {e}") from e

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
            error = data.get("error", data)
            if isinstance(error, dict):
                return error.get("message", str(data))
            return str(error)
        except (json.JSONDecodeError, ValueError):
            return response.text[:500]

    @staticmethod
    def _raise_for_status(status: int, detail: str, retry_after: str | None = None) -> None:
        if status in (401, 403):
            raise LLMAuthenticationError(f"Authentication failed (HTTP {status}): {detail}")
        if status == 429:
            raise LLMRateLimitError(
                f"Rate limit exceeded: {detail}",
                retry_after=float(retry_after) if retry_after else None,
            )
        if status == 400:
            raise LLMInvalidRequestError(f"Bad request: {detail}")
        if status >= 500:
            raise LLMServerError(f"Server error (HTTP {status}): {detail}")
        raise LLMInvalidRequestError(f"HTTP {status}: {detail}")

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpTransport:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
