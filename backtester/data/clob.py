"""
Client for Polymarket's public CLOB price-history endpoint. Read-only.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Optional

from backtester.data.cache import DiskCache

CLOB = "https://clob.polymarket.com"
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (backtester; read-only)",
}


def _request(params: dict) -> Any:
    """Issue the actual HTTP GET. Isolated so tests can monkeypatch just this."""
    url = f"{CLOB}/prices-history?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch_price_history(
    cache: DiskCache,
    clob_token_id: str,
    interval: str = "max",
    fidelity: Optional[int] = None,
) -> list[tuple[int, float]]:
    """
    Return the price series for one CLOB token as a list of (unix_seconds, price),
    sorted ascending by time. `fidelity` is the candle size in minutes.
    """
    params: dict = {"market": clob_token_id, "interval": interval}
    if fidelity is not None:
        params["fidelity"] = fidelity

    key = f"clob_prices_history_{urllib.parse.urlencode(sorted(params.items()))}"
    data = cache.get_json(key, lambda: _request(params))

    history = [(int(pt["t"]), float(pt["p"])) for pt in data.get("history", [])]
    history.sort(key=lambda pt: pt[0])
    for a, b in zip(history, history[1:]):
        assert a[0] <= b[0], "price history must be chronologically ordered"
    return history
