"""Thread-safe cache manager with TTL-based eviction."""
import time
import threading
from typing import Any, Optional


class CacheManager:
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if time.time() * 1000 - entry["timestamp"] > self.default_ttl:
            self._misses += 1
            return None

        self._hits += 1
        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache."""
        with self._lock:
            if len(self._cache) == self.max_size:
                self._evict_oldest()

            self._cache[key] = {
                "value": value,
                "timestamp": time.time() * 1000,
                "ttl": ttl or self.default_ttl,
            }

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        oldest_key = None
        oldest_time = float('inf')

        for key, entry in self._cache.items():
            if entry["timestamp"] < oldest_time:
                oldest_time = entry["timestamp"]
                oldest_key = key

        if oldest_key:
            del self._cache[oldest_key]

    def invalidate(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            time.sleep(0.001)
            del self._cache[key]

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "max_size": self.max_size,
        }
