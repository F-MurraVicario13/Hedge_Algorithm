"""
Aggregate metrics, train/test splitting, parameter sweeps, and the
plain-English verdict. This is the layer that turns a pile of Trade records
into "would this have actually made money, net of costs."
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from backtester.costs.model import CostModel
from backtester.engine.backtest import MarketData, Trade, simulate_all
from backtester.signal.rules import SignalParams


@dataclass(frozen=True)
class Metrics:
    n_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    ev_per_trade_pct: float
    ev_per_signal_pct: float  # ev_per_trade_pct haircut by fill_probability


EMPTY_METRICS = Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _returns(trades: List[Trade], use_net: bool) -> List[float]:
    return [t.net_return_pct if use_net else t.raw_return_pct for t in trades]


def equity_curve(trades: List[Trade], use_net: bool) -> List[float]:
    """Cumulative return assuming a fixed $1 notional staked on each trade in entry-time order."""
    ordered = sorted(trades, key=lambda t: t.entry_time)
    curve = []
    cum = 0.0
    for r in _returns(ordered, use_net):
        cum += r
        curve.append(cum)
    return curve


def max_drawdown(curve: List[float]) -> float:
    """Largest peak-to-trough decline of the cumulative equity curve."""
    peak = 0.0
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        worst = max(worst, peak - v)
    return worst


def compute_metrics(trades: List[Trade], use_net: bool, fill_probability: float = 1.0) -> Metrics:
    if not trades:
        return EMPTY_METRICS

    returns = _returns(trades, use_net)
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    curve = equity_curve(trades, use_net)
    ev = sum(returns) / len(returns)

    return Metrics(
        n_trades=len(trades),
        win_rate=len(wins) / len(returns),
        avg_win_pct=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss_pct=(sum(losses) / len(losses)) if losses else 0.0,
        total_return_pct=curve[-1] if curve else 0.0,
        max_drawdown_pct=max_drawdown(curve),
        ev_per_trade_pct=ev,
        ev_per_signal_pct=ev * fill_probability,
    )


def split_train_test(
    markets: List[MarketData], test_fraction: float = 0.3
) -> Tuple[List[MarketData], List[MarketData]]:
    """
    Sort markets by end_date and hold out the most recent `test_fraction` as
    the test set. Params must only ever be tuned against the train half;
    the test half is for reporting final numbers once, not for search.
    """
    if not markets:
        return [], []
    dated = sorted(markets, key=lambda m: m.end_date or "")
    n_test = min(len(dated) - 1, max(1, round(len(dated) * test_fraction))) if len(dated) > 1 else 0
    split_idx = len(dated) - n_test
    return dated[:split_idx], dated[split_idx:]


def sweep(
    markets: List[MarketData],
    param_grid: List[SignalParams],
    cost_model: CostModel,
) -> List[Tuple[SignalParams, Metrics, Metrics]]:
    """Run the full simulation for each parameter combo. Returns (params, raw_metrics, net_metrics)."""
    results = []
    for params in param_grid:
        trades = simulate_all(markets, params, cost_model)
        raw_metrics = compute_metrics(trades, use_net=False)
        net_metrics = compute_metrics(trades, use_net=True, fill_probability=cost_model.fill_probability)
        results.append((params, raw_metrics, net_metrics))
    return results


def overfitting_flag(sweep_results: List[Tuple[SignalParams, Metrics, Metrics]],
                      profitable_fraction_threshold: float = 0.25) -> Optional[str]:
    """
    If only a small slice of the swept parameter grid is net-profitable while
    the best combo looks great, that's the classic overfitting tell: the rule
    isn't robust, you got lucky on one setting. Returns a warning string, or
    None if profitability looks broad-based.
    """
    with_trades = [(p, r, n) for p, r, n in sweep_results if n.n_trades > 0]
    if not with_trades:
        return "No parameter combination in the sweep produced any trades."

    profitable = [x for x in with_trades if x[2].ev_per_trade_pct > 0]
    fraction = len(profitable) / len(with_trades)
    if fraction < profitable_fraction_threshold and profitable:
        best = max(profitable, key=lambda x: x[2].ev_per_trade_pct)
        return (
            f"OVERFITTING RISK: only {len(profitable)}/{len(with_trades)} "
            f"({fraction:.0%}) of swept parameter combos were net-profitable. "
            f"Best combo {best[0]} had net EV/trade {best[2].ev_per_trade_pct:+.2%} -- "
            f"treat this as a narrow parameter island, not a robust edge."
        )
    return None


def spread_sensitivity(
    markets: List[MarketData],
    params: SignalParams,
    base_cost_model: CostModel,
    half_spreads: Tuple[float, ...] = (0.02, 0.03, 0.04, 0.05, 0.06),
) -> List[Tuple[float, Metrics]]:
    """Re-run the same trades' costing at a range of assumed half-spreads."""
    results = []
    for hs in half_spreads:
        cost_model = replace(base_cost_model, half_spread=hs)
        trades = simulate_all(markets, params, cost_model)
        metrics = compute_metrics(trades, use_net=True, fill_probability=cost_model.fill_probability)
        results.append((hs, metrics))
    return results


def verdict(raw_metrics: Metrics, net_metrics: Metrics, sensitivity: List[Tuple[float, Metrics]]) -> str:
    """Plain-English summary: does EV survive costs, and how sensitive is it to the spread assumption."""
    if net_metrics.n_trades == 0:
        return "No trades were generated by this rule/parameter combination -- nothing to verdict."

    lines = [
        f"Trades: {net_metrics.n_trades}   Win rate: {net_metrics.win_rate:.1%}",
        f"Gross EV/trade (no costs): {raw_metrics.ev_per_trade_pct:+.2%}   "
        f"Net EV/trade (with costs): {net_metrics.ev_per_trade_pct:+.2%}",
        f"Net EV/signal (fill-probability haircut applied): {net_metrics.ev_per_signal_pct:+.2%}",
        f"Max drawdown (net, $1/trade notional): {net_metrics.max_drawdown_pct:.2f}",
    ]

    if net_metrics.ev_per_trade_pct > 0:
        lines.append("VERDICT: the edge SURVIVES modeled trading costs.")
    else:
        lines.append("VERDICT: the edge DOES NOT SURVIVE modeled trading costs.")

    breakeven = next((hs for hs, m in sensitivity if m.ev_per_trade_pct <= 0), None)
    lo, hi = sensitivity[0][0], sensitivity[-1][0]
    if breakeven is not None:
        lines.append(f"Spread sensitivity: turns net-negative once half-spread reaches ~{breakeven:.2f} "
                      f"(tested {lo:.2f}-{hi:.2f}).")
    else:
        lines.append(f"Spread sensitivity: stays net-positive across the tested half-spread range "
                      f"({lo:.2f}-{hi:.2f}).")
    return "\n".join(lines)
