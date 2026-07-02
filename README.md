# hedge_polymarket

Backtester for a Polymarket "underdog insurance" / mean-reversion strategy: when a
favorite spikes toward some threshold (e.g. 0.88), buy the underdog on the
assumption it's oversold, then exit on a rebound, a stop, a time limit, or
settlement -- whichever comes first.

It's read-only. Nothing in here places, cancels, or manages a live order --
it only pulls historical market data from Polymarket's public Gamma/CLOB
APIs, runs the rule against it, and reports the numbers.

## How it works

1. Pull resolved (closed) binary markets from the Gamma API, optionally
   filtered by sport/tag.
2. Fetch each outcome token's price history from the CLOB API.
3. Split markets into train/test by end date (test = most recent slice).
4. Sweep a small grid of parameters (`thresh`, `rebound`, `stop`) on the
   train set only.
5. Pick the best combo by net expected value per trade, then run it once,
   untouched, on the test set.
6. Report gross vs. net-of-costs metrics, an overfitting flag, and spread
   sensitivity.

Everything is guarded against look-ahead: a decision at time `t` can only
ever see price points at or before `t` (see `backtester/guards.py`).

All API responses are cached to disk (`data_cache/` by default) so repeat
runs don't re-hit the network.

## Setup

Requires Python >= 3.9. No third-party runtime dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate      # on Windows
# source .venv/bin/activate # on macOS/Linux

pip install -e .[dev]
```

## Running a backtest

Once installed, the `backtest` command is on your PATH:

```bash
backtest
```

That runs with defaults: pulls up to 200 resolved markets, sweeps thresholds
around 0.88, and prints train + held-out test results.

Useful flags:

```bash
backtest --sport soccer --limit 500
backtest --thresh 0.90 --rebound 0.06 --stop 0.04 --no-sweep
backtest --half-spread 0.05 --slippage 0.01 --fill-probability 0.8
backtest --cache-dir ./data_cache --interval 1d
```

Run `backtest --help` for the full list of options (entry threshold, exit
rules, cost modeling, train/test split, cache location, etc).

You can also run it without installing, straight from the repo root:

```bash
python -m backtester.cli --sport basketball
```

Note on `--sport`: Gamma's tag/category taxonomy hasn't been verified
against a live response in this environment (no outbound network access
during development). It's passed through as a best-effort `{"tag": ...}`
filter -- use `--gamma-param key=value` (repeatable) to override it once
you've inspected a real `/markets` response for your account/region.

## Testing

```bash
pytest
```

Tests cover the disk cache, Gamma/CLOB clients, cost model, settlement
logic, signal detection, engine, metrics, and CLI wiring.

## Project layout

```
backtester/
  cli.py            entry point: build_parser, dataset assembly, main()
  guards.py         no-look-ahead assertions used by signal + engine
  settlement.py     resolves a market's binary outcome from settlement prices
  data/
    cache.py        disk cache + request throttle
    gamma.py        Gamma API client (markets/events/search)
    clob.py         CLOB API client (price history)
  signal/
    rules.py        entry/exit detection for the mean-reversion rule
  engine/
    backtest.py     runs the signal rule across a set of markets
    metrics.py       win rate, EV/trade, drawdown, train/test split, sweep
  costs/
    model.py        spread/slippage/fill-probability cost model
tests/               pytest suite mirroring the structure above
```
