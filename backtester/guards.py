"""
No-look-ahead guard used by both signal and engine modules. A decision made
"as of" timestamp `decision_time` must never be informed by data whose
timestamp is later than that -- otherwise the backtest is telling you about a
trade you couldn't actually have made.
"""

from __future__ import annotations

from typing import Iterable, Tuple


def assert_no_lookahead(decision_time: int, points: Iterable[Tuple[int, float]]) -> None:
    """Raise if any (t, price) point used for a decision is later than decision_time."""
    for t, _p in points:
        if t > decision_time:
            raise AssertionError(
                f"look-ahead violation: decision at t={decision_time} used data from t={t}"
            )


def price_at_or_before(series: list[Tuple[int, float]], decision_time: int):
    """
    Return the last (t, p) point with t <= decision_time, or None if the series
    has no such point. This is the only sanctioned way to look up "the price at
    time t" -- it structurally cannot return a future point.
    """
    result = None
    for t, p in series:
        if t > decision_time:
            break
        result = (t, p)
    return result
