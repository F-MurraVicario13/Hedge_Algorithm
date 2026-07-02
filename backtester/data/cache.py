"""
Simple disk cache for the public Polymarket APIs. Every raw response gets
written to disk keyed off the request; repeat runs read from disk instead of
re-hitting the API. Only cache misses go through the rate limiter.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_CACHE_DIR = Path("data_cache")


class DiskCache:
    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, min_interval: float = 0.25):
        """
        cache_dir    : where raw JSON responses are stored
        min_interval : minimum seconds between actual network fetches (politeness throttle)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last_request_ts = 0.0

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        safe_prefix = "".join(c if c.isalnum() else "_" for c in key)[:60]
        return self.cache_dir / f"{safe_prefix}_{digest}.json"

    def get_json(self, key: str, fetch_fn: Callable[[], Any]) -> Any:
        """Return cached JSON for `key`, else call fetch_fn() (rate-limited) and cache it."""
        path = self._path_for(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        self._throttle()
        data = fetch_fn()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def has_json(self, key: str) -> bool:
        """Return True when the cache already contains a JSON blob for `key`."""
        return self._path_for(key).exists()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()
