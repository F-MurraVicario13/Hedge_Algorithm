"""
Entry/exit rule detection for the "underdog insurance / mean-reversion" rule:
buy the underdog when the favorite overreacts toward THRESH, sell back on a
rebound, a stop, a time limit, or hold to settlement -- whichever comes first.

Everything here reads price series that are already known as of each decision
point; no function ever consults a timestamp later than the one it's deciding
at (see backtester.guards).

A market is modeled as exactly two complementary outcome token price series,
outcome_series[0] and outcome_series[1] (index 0 is conventionally the "Yes"
token per Gamma's clobTokenIds ordering, but the rule is symmetric -- either
side can be the "favorite" at a given moment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from backtester.guards import price_at_or_before

PricePoint = Tuple[int, float]
PriceSeries = List[PricePoint]


@dataclass(frozen=True)
class SignalParams:
    thresh: float = 0.88          # favorite entry threshold
    rebound: float = 0.05         # underdog price rise (from entry) that triggers a take-profit exit
    stop: float = 0.03            # underdog further price drop (from entry) that triggers a stop-loss exit
    time_limit_seconds: Optional[int] = None  # force-exit if still open this long after entry; None = no limit
    consistency_tolerance: float = 0.05  # how far the underdog's price may exceed (1-thresh) and still count


@dataclass(frozen=True)
class EntryEvent:
    entry_time: int
    entry_price: float            # underdog's own price at entry_time (not 1 - favorite_price)
    favorite_outcome_index: int
    underdog_outcome_index: int
    favorite_price_at_entry: float


@dataclass(frozen=True)
class ExitEvent:
    kind: str                     # "rebound" | "stop" | "time_limit" | "settlement"
    exit_time: int
    exit_price: float             # market price for live exits; resolved outcome (0.0/1.0) for settlement
    is_settlement: bool


@dataclass(frozen=True)
class RawTrade:
    """A signal-detected trade at raw (pre-cost) market prices."""
    underdog_outcome_index: int
    entry: EntryEvent
    exit: ExitEvent


def detect_entries(outcome_series: List[PriceSeries], params: SignalParams) -> List[EntryEvent]:
    """
    Scan both outcome series for an upward cross of `thresh`. Each such cross
    on side F is a candidate entry on the OTHER side (the underdog), priced at
    the underdog's own last known price at-or-before that same timestamp --
    never at a later, more favorable (or unfavorable) price.
    """
    if len(outcome_series) != 2:
        raise ValueError("only binary markets are supported")

    events: List[EntryEvent] = []
    for fav_idx in (0, 1):
        dog_idx = 1 - fav_idx
        fav_series = outcome_series[fav_idx]
        dog_series = outcome_series[dog_idx]

        prev_price = None
        for t, p in fav_series:
            if prev_price is not None and prev_price < params.thresh <= p:
                dog_point = price_at_or_before(dog_series, t)
                if dog_point is not None:
                    dog_time, dog_price = dog_point
                    if dog_price <= (1 - params.thresh) + params.consistency_tolerance:
                        events.append(EntryEvent(
                            entry_time=t,
                            entry_price=dog_price,
                            favorite_outcome_index=fav_idx,
                            underdog_outcome_index=dog_idx,
                            favorite_price_at_entry=p,
                        ))
            prev_price = p

    events.sort(key=lambda e: e.entry_time)
    return events


def detect_exit(
    entry_time: int,
    entry_price: float,
    future_series: PriceSeries,
    params: SignalParams,
    resolved_outcome: Optional[float],
) -> ExitEvent:
    """
    Walk `future_series` (points strictly after entry_time, chronological) and
    return the first triggered exit. `future_series` containing only points
    after entry_time is itself the no-look-ahead guarantee: the function
    physically cannot see anything before or at entry, and processing stops at
    the first qualifying point so nothing later than the trigger is consulted
    either.

    Priority when a single price point satisfies more than one condition:
    rebound/stop (a real price move) take precedence over time_limit (a
    passive clock expiry), since a resting position's fill would have already
    triggered on price before the clock check would matter in practice.
    """
    for t, p in future_series:
        if p >= entry_price + params.rebound:
            return ExitEvent("rebound", t, p, is_settlement=False)
        if p <= entry_price - params.stop:
            return ExitEvent("stop", t, p, is_settlement=False)
        if params.time_limit_seconds is not None and (t - entry_time) >= params.time_limit_seconds:
            return ExitEvent("time_limit", t, p, is_settlement=False)

    if resolved_outcome is None:
        raise ValueError(
            "price series ended with no exit trigger and no resolved outcome was provided; "
            "cannot settle an unresolved market"
        )
    last_time = future_series[-1][0] if future_series else entry_time
    return ExitEvent("settlement", last_time, float(resolved_outcome), is_settlement=True)


def simulate_market_trades(
    outcome_series: List[PriceSeries],
    resolved_outcome_index: int,
    params: SignalParams,
) -> List[RawTrade]:
    """
    Detect all non-overlapping trades for one market: while a position is
    open, later entry signals are ignored (no pyramiding into the same
    market). Trades are raw/pre-cost; the engine applies the cost model.
    """
    candidates = detect_entries(outcome_series, params)
    trades: List[RawTrade] = []
    busy_until: Optional[int] = None

    for cand in candidates:
        if busy_until is not None and cand.entry_time < busy_until:
            continue

        dog_series = outcome_series[cand.underdog_outcome_index]
        future = [pt for pt in dog_series if pt[0] > cand.entry_time]
        resolved_for_dog = 1.0 if cand.underdog_outcome_index == resolved_outcome_index else 0.0

        exit_event = detect_exit(cand.entry_time, cand.entry_price, future, params, resolved_for_dog)
        trades.append(RawTrade(
            underdog_outcome_index=cand.underdog_outcome_index,
            entry=cand,
            exit=exit_event,
        ))
        busy_until = exit_event.exit_time

    return trades
