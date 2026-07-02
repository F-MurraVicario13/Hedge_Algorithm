"""
Client for the public, read-only Polymarket Gamma API (markets/events/search).
Ported from polymarket_hedge.py and extended with bulk pagination over closed
(resolved) markets, since the backtester needs many historical markets, not
just one live one.

Nothing here places, cancels, or manages orders -- it only issues GET requests.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from typing import Any, Iterator, Optional

from backtester.data.cache import DiskCache

GAMMA = "https://gamma-api.polymarket.com"
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (backtester; read-only)",
}


def _request(path: str, params: dict) -> Any:
    """Issue the actual HTTP GET. Isolated so tests can monkeypatch just this."""
    url = f"{GAMMA}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"Gamma request failed for {url} with HTTP {e.code} {e.reason}"
            + (f"\nResponse body:\n{body}" if body else "")
        ) from e


def _get(cache: DiskCache, path: str, params: dict) -> Any:
    key = f"gamma_{path.strip('/')}_{urllib.parse.urlencode(sorted(params.items()))}"
    return cache.get_json(key, lambda: _request(path, params))


def search_events(cache: DiskCache, keyword: str, limit: int = 8) -> list[tuple[str, str]]:
    """Find events (games/matchups) by keyword via the public-search endpoint."""
    data = _get(cache, "/public-search", {"q": keyword, "limit_per_type": limit})
    return [(e.get("title", "?"), e.get("slug", "")) for e in data.get("events", [])[:limit]]


def _parse_prices(raw) -> list[float]:
    prices = json.loads(raw) if isinstance(raw, str) else raw
    return [float(p) for p in prices] if prices else []


def _parse_clob_token_ids(raw) -> list[str]:
    if not raw:
        return []
    ids = json.loads(raw) if isinstance(raw, str) else raw
    return [str(i) for i in ids]


def parse_market(m: dict) -> dict:
    """Normalize one raw Gamma market record into the fields the rest of the pipeline needs."""
    outcomes = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes", [])
    return {
        "id": m.get("id"),
        "slug": m.get("slug"),
        "question": m.get("question"),
        "outcomes": outcomes,
        "outcome_prices": _parse_prices(m.get("outcomePrices")),
        "clob_token_ids": _parse_clob_token_ids(m.get("clobTokenIds")),
        "closed": bool(m.get("closed")),
        "spread": float(m.get("spread") or 0),
        "liquidity": float(m.get("liquidity") or 0),
        "volume24hr": float(m.get("volume24hr") or 0),
        "end_date": m.get("endDate"),
        "tags": m.get("tags") or [],
    }


def fetch_event_markets(cache: DiskCache, slug: str) -> list[dict]:
    """Given an event slug, return its child markets, parsed."""
    rows = _get(cache, "/events", {"slug": slug})
    if not rows:
        return []
    return [parse_market(m) for m in rows[0].get("markets", [])]


def fetch_market(cache: DiskCache, slug: Optional[str] = None, market_id: Optional[str] = None) -> dict:
    """Return a normalized dict for one binary market."""
    if market_id:
        rows = _get(cache, "/markets", {"id": market_id})
    elif slug:
        rows = _get(cache, "/markets", {"slug": slug})
    else:
        raise ValueError("need slug or market_id")
    if not rows:
        raise LookupError("no market found for that slug/id")
    return parse_market(rows[0])


def iter_closed_markets(
    cache: DiskCache,
    extra_params: Optional[dict] = None,
    page_size: int = 100,
    max_pages: Optional[int] = None,
) -> Iterator[dict]:
    """
    Page through /markets?closed=true (plus any extra_params, e.g. a sport tag),
    yielding parsed markets. Stops when a page comes back empty or max_pages is hit.
    """
    params = {"closed": "true", "limit": page_size}
    if extra_params:
        params.update(extra_params)
    offset = 0
    pages = 0
    while max_pages is None or pages < max_pages:
        try:
            page = _get(cache, "/markets", {**params, "offset": offset})
        except RuntimeError as e:
            if "offset too large" in str(e):
                return
            raise
        if not page:
            return
        for m in page:
            yield parse_market(m)
        if len(page) < page_size:
            return
        offset += page_size
        pages += 1
