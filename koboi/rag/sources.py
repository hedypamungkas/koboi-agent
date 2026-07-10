"""koboi/rag/sources.py -- remote document fetching (HTTP / S3-compatible) + cache.

Cloudflare R2 is S3-compatible (use ``endpoint_url``). HTTP uses ``httpx`` (already a
hard dependency -- zero new deps); S3 uses ``boto3`` (optional ``[rag-cloud]`` extra).
Secrets stay out of YAML via the existing ``${VAR}`` interpolation; they are never
included in cache-key material or logged.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

# Reuse the web tool's SSRF guard + retry policy so remote RAG ingestion can't reach
# cloud metadata services and shares the same backoff semantics as web_fetch.
from koboi.tools.builtin.web import (  # noqa: E402
    MAX_RETRIES,
    MAX_TIMEOUT,
    RETRYABLE_STATUS,
    _check_url_ssrf,
)

_SECRET_KEYS = frozenset({"access_key_id", "secret_access_key", "headers", "token"})


def fetch_http(url: str, *, headers: dict | None = None, timeout: int | None = None) -> bytes:
    """Fetch a single HTTP(S) document. Raises on non-retryable failure."""
    import httpx  # hard dependency (pyproject)

    _check_url_ssrf(url)
    timeout_s = max(1, min(int(timeout or MAX_TIMEOUT), MAX_TIMEOUT))
    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                    continue
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPError as exc:
                last_exc = exc
    raise last_exc or RuntimeError(f"HTTP fetch failed for {url}")


def name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] if path else ""
    return name or "document"


def source_key(spec: dict) -> str:
    """Stable cache key for a source spec (secrets excluded from the material)."""
    safe = {k: v for k, v in spec.items() if k not in _SECRET_KEYS}
    blob = repr(sorted(safe.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def fetch_http_entry(entry: dict, doc_cache: DocumentCache | None) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for an HTTP ``documents[]`` entry, with cache."""
    url = entry.get("url", "")
    if not url:
        return
    key = source_key({"source": "http", "url": url})
    if doc_cache:
        hit = doc_cache.get(key)
        if hit is not None:
            yield hit
            return
    try:
        data = fetch_http(url, headers=entry.get("headers"), timeout=entry.get("timeout"))
    except Exception as exc:  # network / SSRF / status -> skip this entry, keep building
        _logger.warning("HTTP fetch failed for %s: %s", url, exc)
        return
    name = name_from_url(url)
    if doc_cache:
        doc_cache.put(key, name, data)
    yield name, data


def fetch_s3_entry(entry: dict, doc_cache: DocumentCache | None) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for objects under an S3/R2 prefix, with per-object cache."""
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError:
        _logger.error("RAG source 's3' skipped: install the optional extra -- pip install 'koboi-agent[rag-cloud]'")
        return
    bucket = entry.get("bucket")
    prefix = entry.get("key", "")
    if not bucket:
        _logger.warning("s3 document source missing 'bucket'; skipping")
        return
    try:
        client = boto3.client(
            "s3",
            endpoint_url=entry.get("endpoint_url") or None,
            region_name=entry.get("region", "auto"),
            aws_access_key_id=entry.get("access_key_id"),
            aws_secret_access_key=entry.get("secret_access_key"),
        )
        paginator = client.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                okey = obj["Key"]
                if okey.endswith("/"):  # folder marker
                    continue
                found = True
                name = okey.rsplit("/", 1)[-1] or okey
                ckey = source_key({"source": "s3", "bucket": bucket, "key": okey})
                cached = doc_cache.get(ckey) if doc_cache else None
                if cached is not None:
                    yield cached
                    continue
                data = client.get_object(Bucket=bucket, Key=okey)["Body"].read()
                if doc_cache:
                    doc_cache.put(ckey, name, data)
                yield name, data
        if not found:
            _logger.warning("s3 bucket=%s prefix=%r returned no objects", bucket, prefix)
    except Exception as exc:  # credentials / endpoint / network -> skip, keep building
        _logger.warning("s3 fetch failed (bucket=%s): %s", bucket, exc)


class DocumentCache:
    """Opt-in on-disk cache for fetched remote documents (bytes + original name).

    Per-session agent builds (``koboi/server/pool.py``) re-run ``_load_documents``; this
    cache prevents re-fetching the corpus over the network on every session. Keyed by a
    content-spec hash; invalidation is manual (delete the directory).
    """

    def __init__(self, dir_path: str) -> None:
        self.dir = Path(dir_path)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.dir / key, self.dir / f"{key}.name"

    def get(self, key: str) -> tuple[str, bytes] | None:
        data_f, name_f = self._paths(key)
        if data_f.exists() and name_f.exists():
            return name_f.read_text(), data_f.read_bytes()
        return None

    def put(self, key: str, name: str, data: bytes) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data_f, name_f = self._paths(key)
        data_f.write_bytes(data)
        name_f.write_text(name)
