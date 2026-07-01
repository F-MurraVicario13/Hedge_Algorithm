"""
CLI entry point: fetches resolved markets, builds price-history datasets,
splits train/test by date, sweeps a small grid of parameters on train only,
and reports final numbers on the untouched test set -- gross and net of
modeled trading costs.

NOTE ON THE `--sport` FILTER: Gamma's tag/category taxonomy isn't something
this codebase has verified against a live response (the sandbox this was
built in has no outbound network access). `--sport` is passed through as
{"tag": <value>} as a best-effort guess; use --gamma-param key=value
(repeatable) to override or add filters once you've inspected a real
`/markets` response for the actual field name your account/region sees.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

from backtester.costs.model import CostModel
from backtester.data.cache import DEFAULT_CACHE_DIR, DiskCache
from backtester.data.clob import fetch_price_history
from backtester.data.gamma import iter_closed_markets
from backtester.engine.backtest import MarketData, simulate_all
from backtester.engine.metrics import (
    Metrics,
    compute_metrics,
    overfitting_flag,
    spread_sensitivity,
    split_train_test,
    sweep,
    verdict,
)
from backtester.settlement import resolve_binary_outcome
from backtester.signal.rules import SignalParams


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="backtest",
        description="Backtest the underdog-insurance / mean-reversion rule against Polymarket history. "
                     "Read-only: never places, cancels, or manages an order.",
    )
    ap.add_argument("--sport", help="best-effort tag filter, e.g. soccer (see module docstring)")
    ap.add_argument("--gamma-param", action="append", default=[], metavar="KEY=VALUE",
                     help="extra raw Gamma /markets query param, repeatable")
    ap.add_argument("--thresh", type=float, default=0.88, help="favorite entry threshold")
    ap.add_argument("--rebound", type=float, default=0.05, help="underdog rebound (from entry) that exits")
    ap.add_argument("--stop", type=float, default=0.03, help="underdog further drop (from entry) that exits")
    ap.add_argument("--time-limit", type=int, default=None, metavar="SECONDS",
                     help="force-exit if still open this long after entry (default: no limit)")
    ap.add_argument("--half-spread", type=float, default=0.04,
                     help="assumed half-spread when only midpoint history is available")
    ap.add_argument("--slippage", type=float, default=0.0, help="extra price impact per side, thin-book haircut")
    ap.add_argument("--fill-probability", type=float, default=1.0,
                     help="probability a resting order at the modeled price actually fills")
    ap.add_argument("--interval", default="max", help="clob prices-history interval (max|1m|1w|1d)")
    ap.add_argument("--fidelity", type=int, default=None, help="clob prices-history candle size, minutes")
    ap.add_argument("--limit", type=int, default=200, help="max number of resolved markets to pull")
    ap.add_argument("--test-fraction", type=float, default=0.3,
                     help="fraction of markets (most recent by end date) held out as the test set")
    ap.add_argument("--no-sweep", action="store_true",
                     help="skip the parameter sweep and just use the given thresh/rebound/stop")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="disk cache directory")
    return ap


def _parse_gamma_params(pairs: List[str]) -> Dict[str, str]:
    parsed = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--gamma-param expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        parsed[key] = value
    return parsed


def build_market_dataset(
    cache: DiskCache,
    sport: Optional[str],
    gamma_params: Dict[str, str],
    limit: int,
    interval: str,
    fidelity: Optional[int],
) -> Tuple[List[MarketData], Dict[str, int]]:
    """Pull resolved binary markets and their price history, skipping anything that doesn't fit cleanly."""
    extra = dict(gamma_params)
    if sport:
        extra.setdefault("tag", sport)

    markets: List[MarketData] = []
    skipped = {"not_binary": 0, "unresolved": 0, "too_short_history": 0}

    for raw in iter_closed_markets(cache, extra_params=extra, page_size=100):
        if len(markets) >= limit:
            break
        if len(raw["clob_token_ids"]) != 2 or len(raw["outcome_prices"]) != 2:
            skipped["not_binary"] += 1
            continue
        try:
            resolved_idx = resolve_binary_outcome(raw["outcome_prices"])
        except ValueError:
            skipped["unresolved"] += 1
            continue

        tok0, tok1 = raw["clob_token_ids"]
        hist0 = fetch_price_history(cache, tok0, interval=interval, fidelity=fidelity)
        hist1 = fetch_price_history(cache, tok1, interval=interval, fidelity=fidelity)
        if len(hist0) < 2 or len(hist1) < 2:
            skipped["too_short_history"] += 1
            continue

        markets.append(MarketData(
            market_id=raw.get("slug") or str(raw.get("id")),
            outcome_series=[hist0, hist1],
            resolved_outcome_index=resolved_idx,
            end_date=raw.get("end_date"),
        ))

    return markets, skipped


def build_param_grid(thresh: float, rebound: float, stop: float, time_limit: Optional[int]) -> List[SignalParams]:
    """A small sweep around the requested parameters -- not a single cherry-picked combo."""
    threshes = sorted({round(thresh + d, 4) for d in (-0.02, 0.0, 0.02) if 0.0 < thresh + d < 1.0})
    rebounds = sorted({round(rebound + d, 4) for d in (-0.02, 0.0, 0.02) if rebound + d > 0.0})
    stops = sorted({round(stop + d, 4) for d in (-0.01, 0.0, 0.01) if stop + d > 0.0})
    return [
        SignalParams(thresh=th, rebound=rb, stop=sp, time_limit_seconds=time_limit)
        for th in threshes for rb in rebounds for sp in stops
    ]


def _fmt_metrics_row(label: str, m: Metrics) -> str:
    return (f"  {label:<24} n={m.n_trades:<5} win%={m.win_rate:>6.1%}  "
            f"avg_win={m.avg_win_pct:>+7.2%}  avg_loss={m.avg_loss_pct:>+7.2%}  "
            f"total_ret={m.total_return_pct:>+7.2%}  max_dd={m.max_drawdown_pct:>6.2%}  "
            f"ev/trade={m.ev_per_trade_pct:>+7.2%}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    gamma_params = _parse_gamma_params(args.gamma_param)
    cache = DiskCache(cache_dir=args.cache_dir)

    print(f"Fetching resolved markets (sport={args.sport!r}, limit={args.limit})...", file=sys.stderr)
    markets, skipped = build_market_dataset(
        cache, args.sport, gamma_params, args.limit, args.interval, args.fidelity,
    )
    print(f"Loaded {len(markets)} usable markets. Skipped: {skipped}", file=sys.stderr)
    if not markets:
        print("No usable markets found -- nothing to backtest.", file=sys.stderr)
        return

    train, test = split_train_test(markets, test_fraction=args.test_fraction)
    print(f"Train: {len(train)} markets   Test (untouched until final report): {len(test)} markets")

    cost_model = CostModel(
        half_spread=args.half_spread, slippage=args.slippage, fill_probability=args.fill_probability,
    )

    grid = [SignalParams(thresh=args.thresh, rebound=args.rebound, stop=args.stop,
                          time_limit_seconds=args.time_limit)] if args.no_sweep else \
        build_param_grid(args.thresh, args.rebound, args.stop, args.time_limit)

    print(f"\n=== Parameter sweep on TRAIN set ({len(grid)} combos) ===")
    sweep_results = sweep(train, grid, cost_model)
    for params, raw_m, net_m in sweep_results:
        print(f"thresh={params.thresh:.2f} rebound={params.rebound:.2f} stop={params.stop:.2f}")
        print(_fmt_metrics_row("gross (no costs)", raw_m))
        print(_fmt_metrics_row("net (with costs)", net_m))

    flag = overfitting_flag(sweep_results)
    if flag:
        print(f"\n{flag}")

    profitable = [(p, r, n) for p, r, n in sweep_results if n.n_trades > 0]
    if profitable:
        best_params, _, _ = max(profitable, key=lambda x: x[2].ev_per_trade_pct)
    else:
        best_params = grid[0]
    print(f"\nBest train-set params by net EV/trade: thresh={best_params.thresh:.2f} "
          f"rebound={best_params.rebound:.2f} stop={best_params.stop:.2f}")

    print("\n=== Final report on held-out TEST set (params fixed from train) ===")
    test_trades = simulate_all(test, best_params, cost_model)
    raw_m = compute_metrics(test_trades, use_net=False)
    net_m = compute_metrics(test_trades, use_net=True, fill_probability=cost_model.fill_probability)
    print(_fmt_metrics_row("gross (no costs)", raw_m))
    print(_fmt_metrics_row("net (with costs)", net_m))

    base = args.half_spread
    half_spreads = tuple(sorted({round(max(0.0, base + d), 4) for d in (-0.02, -0.01, 0.0, 0.01, 0.02, 0.04)}))
    sensitivity = spread_sensitivity(test, best_params, cost_model, half_spreads=half_spreads)
    print("\n=== Spread sensitivity (test set) ===")
    for hs, m in sensitivity:
        print(f"  half_spread={hs:.2f}   ev/trade={m.ev_per_trade_pct:+.2%}   n={m.n_trades}")

    print(f"\n{verdict(raw_m, net_m, sensitivity)}")


if __name__ == "__main__":
    main()
