"""
Trading-cost model. This is the module that keeps the backtest honest: it's
easy for a mechanical rule to look profitable on raw mid-price history and
then evaporate once you pay to actually cross the spread twice per trade.

Polymarket's public prices-history endpoint returns a single trade/mid price
series, not separate bid/ask history. So we model the ask as mid + half_spread
and the bid as mid - half_spread, plus an optional extra slippage term for
thin books where the top-of-book size wouldn't cover a real order.

Settlement (redemption at market resolution) is NOT a market sale -- a winning
contract pays exactly $1 and a losing one pays exactly $0, with no spread.
Only live-market exits (rebound / stop / time_limit) pay the spread.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    half_spread: float = 0.04     # assumed half-spread in price units (e.g. 0.04 = 4 cents), config knob
    slippage: float = 0.0         # extra price impact per side for thin top-of-book, config knob
    fill_probability: float = 1.0  # probability a resting order at the modeled price actually fills
    price_floor: float = 0.01
    price_cap: float = 0.99

    def __post_init__(self):
        if not (0.0 < self.fill_probability <= 1.0):
            raise ValueError(f"fill_probability must be in (0, 1], got {self.fill_probability}")
        if self.half_spread < 0 or self.slippage < 0:
            raise ValueError("half_spread and slippage must be non-negative")

    def buy_price(self, mid: float) -> float:
        """Price paid to buy at the ask: mid + half-spread + slippage, capped below $1."""
        return min(self.price_cap, mid + self.half_spread + self.slippage)

    def sell_price(self, mid: float) -> float:
        """Price received selling at the bid: mid - half-spread - slippage, floored above $0."""
        return max(self.price_floor, mid - self.half_spread - self.slippage)


ZERO_COST_MODEL = CostModel(half_spread=0.0, slippage=0.0, fill_probability=1.0)


def entry_fill_price(mid_price: float, model: CostModel) -> float:
    return model.buy_price(mid_price)


def exit_fill_price(mid_price_or_outcome: float, model: CostModel, is_settlement: bool) -> float:
    """Settlement redemption is at par (no spread); a live-market exit pays the bid-side cost."""
    if is_settlement:
        return mid_price_or_outcome
    return model.sell_price(mid_price_or_outcome)
