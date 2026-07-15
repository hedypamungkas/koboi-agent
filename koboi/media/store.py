"""koboi/media/store.py -- materialize generated artifacts to durable storage.

Backends: ``local`` (filesystem, W0 default) and ``r2``/``s3`` (S3-compatible object storage, W5b;
Cloudflare R2 via ``endpoint_url``). Gateway artifact URIs are short-lived (Surplus: 15 min direct
S3 / 3 h proxy), so the backend materializes bytes before returning a result. Content-hash dedup
avoids re-saving identical bytes (``put_object`` is idempotent on Key, so dedup is free for cloud).
The cloud backend reuses the boto3 ``[media-cloud]`` extra + the client shape proven by
``koboi.rag.sources.fetch_s3_entry``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import httpx

from koboi.media.types import MediaResult

_logger = logging.getLogger(__name__)

_EXT_BY_CONTENT_TYPE: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
}

_CLOUD_BACKENDS = frozenset({"r2", "s3"})


class _S3Writer:
    """S3/R2 object writer (boto3 ``put_object``). Mirrors ``rag.sources.fetch_s3_entry``'s client."""

    def __init__(self, conf: dict) -> None:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise NotImplementedError(
                "R2/S3 media storage requires the [media-cloud] extra -- pip install 'koboi-agent[media-cloud]'"
            ) from e
        bucket = conf.get("bucket")
        if not bucket:
            raise ValueError("media.storage.bucket is required for the r2/s3 backend")
        self._bucket = bucket
        self._prefix = (conf.get("key_prefix") or "media/").rstrip("/") + "/"
        self._client = boto3.client(
            "s3",
            endpoint_url=conf.get("endpoint_url") or None,
            region_name=conf.get("region", "auto"),
            aws_access_key_id=conf.get("access_key_id"),
            aws_secret_access_key=conf.get("secret_access_key"),
        )

    async def put(self, key: str, data: bytes, content_type: str | None) -> str:
        full_key = f"{self._prefix}{key}"
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=full_key,
            Body=data,
            ContentType=content_type or "application/octet-stream",
        )
        return f"s3://{self._bucket}/{full_key}"

    async def close(self) -> None:
        """boto3 clients hold no awaitable resource."""
        return None


class MediaStore:
    """Durable artifact store. ``backend`` in {local, r2, s3}."""

    def __init__(
        self,
        backend: str = "local",
        dir: str = "./media_artifacts",
        *,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        key_prefix: str | None = None,
    ) -> None:
        self._backend = backend
        self._dir = Path(dir)
        self._writer: _S3Writer | None = None
        if backend == "local":
            pass
        elif backend in _CLOUD_BACKENDS:
            self._writer = _S3Writer(
                {
                    "bucket": bucket,
                    "endpoint_url": endpoint_url,
                    "region": region,
                    "access_key_id": access_key_id,
                    "secret_access_key": secret_access_key,
                    "key_prefix": key_prefix,
                }
            )
        else:
            raise NotImplementedError(f"media storage backend '{backend}' not implemented (use local|r2|s3)")

    async def save(self, result: MediaResult) -> str:
        """Persist ``result.data`` (or fetch ``result.url``) -> durable location (path or ``s3://`` URI)."""
        data = result.data
        if data is None:
            if not result.url:
                raise ValueError("MediaResult has neither data nor a url to store")
            data = await self._fetch(result.url)
        digest = hashlib.sha256(data).hexdigest()[:32]
        ext = _EXT_BY_CONTENT_TYPE.get(result.content_type or "", ".bin")
        key = f"{digest}{ext}"
        if self._writer is not None:
            return await self._writer.put(key, data, result.content_type)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / key
        if not path.exists():
            path.write_bytes(data)
        return str(path)

    async def _fetch(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def close(self) -> None:
        if self._writer is not None:
            await self._writer.close()
        return None
