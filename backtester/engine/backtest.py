"""
Turns signal-detected raw trades into priced Trade records, both gross (raw
mid-to-mid, no costs) and net (cost-model-adjusted fills), for the SAME
entry/exit timestamps -- so the only difference between the two numbers is
the cost model, making the cost drag explicit rather than confounding it with
a different set of trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from backtester.costs.model import CostModel, entry_fill_price, exit_fill_price
from backtester.settlement import close_before_settlement
from backtester.signal.rules import PriceSeries, RawTrade, SignalParams, simulate_market_trades


@dataclass(frozen=True)
class MarketData:
    market_id: str
    outcome_series: List[PriceSeries]  # exactly 2 complementary outcome price series
    resolved_outcome_index: int
    end_date: Optional[str] = None  # ISO date string; used upstream for train/test splitting


@dataclass(frozen=True)
class Trade:
    market_id: str
    underdog_outcome_index: int
    entry_time: int
    entry_mid: float
    entry_fill: float
    exit_time: int
    exit_kind: str
    exit_mid_or_outcome: float
    exit_fill: float
    is_settlement: bool
    raw_pnl: float          # per 1 contract, mid-to-mid, no costs
    net_pnl: float          # per 1 contract, cost-adjusted fills
    raw_return_pct: float   # raw_pnl / entry_mid
    net_return_pct: float   # net_pnl / entry_fill


def build_trade(market_id: str, raw: RawTrade, cost_model: CostModel) -> Trade:
    entry_mid = raw.entry.entry_price
    exit_value = raw.exit.exit_price  # a market price, or the resolved 0.0/1.0 for settlement

    entry_fill = entry_fill_price(entry_mid, cost_model)
    exit_fill = exit_fill_price(exit_value, cost_model, raw.exit.is_settlement)

    # Same settlement primitive either way: shares*(exit - entry). For a settlement
    # exit, exit_value is 0.0/1.0, which makes this identical to settlement.payoff().
    raw_pnl = close_before_settlement(1.0, entry_mid, exit_value)
    net_pnl = close_before_settlement(1.0, entry_fill, exit_fill)

    return Trade(
        market_id=market_id,
        underdog_outcome_index=raw.underdog_outcome_index,
        entry_time=raw.entry.entry_time,
        entry_mid=entry_mid,
        entry_fill=entry_fill,
        exit_time=raw.exit.exit_time,
        exit_kind=raw.exit.kind,
        exit_mid_or_outcome=exit_value,
        exit_fill=exit_fill,
        is_settlement=raw.exit.is_settlement,
        raw_pnl=raw_pnl,
        net_pnl=net_pnl,
        raw_return_pct=raw_pnl / entry_mid,
        net_return_pct=net_pnl / entry_fill,
    )


def simulate_all(
    markets: Iterable[MarketData],
    params: SignalParams,
    cost_model: CostModel,
) -> List[Trade]:
    trades: List[Trade] = []
    for market in markets:
        raw_trades = simulate_market_trades(market.outcome_series, market.resolved_outcome_index, params)
        trades.extend(build_trade(market.market_id, raw, cost_model) for raw in raw_trades)
    return trades
