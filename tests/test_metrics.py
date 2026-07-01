import pytest

from backtester.costs.model import CostModel, ZERO_COST_MODEL
from backtester.engine.backtest import MarketData, Trade, simulate_all
from backtester.engine.metrics import (
    compute_metrics,
    equity_curve,
    max_drawdown,
    overfitting_flag,
    spread_sensitivity,
    split_train_test,
    sweep,
    verdict,
)
from backtester.signal.rules import SignalParams


def make_trade(entry_time, raw_pnl, net_pnl, entry_mid=0.10, entry_fill=0.14, market_id="m"):
    return Trade(
        market_id=market_id,
        underdog_outcome_index=1,
        entry_time=entry_time,
        entry_mid=entry_mid,
        entry_fill=entry_fill,
        exit_time=entry_time + 100,
        exit_kind="rebound",
        exit_mid_or_outcome=entry_mid + raw_pnl,
        exit_fill=entry_fill + net_pnl,
        is_settlement=False,
        raw_pnl=raw_pnl,
        net_pnl=net_pnl,
        raw_return_pct=raw_pnl / entry_mid,
        net_return_pct=net_pnl / entry_fill,
    )


def test_equity_curve_is_cumulative_in_entry_time_order():
    trades = [make_trade(200, 0.06, 0.06), make_trade(100, -0.02, -0.02)]
    curve = equity_curve(trades, use_net=False)
    # sorted by entry_time: t=100 (-0.02/0.10=-0.2) then t=200 (0.06/0.10=0.6)
    assert curve[0] == pytest.approx(-0.2)
    assert curve[1] == pytest.approx(0.4)


def test_max_drawdown_known_sequence():
    curve = [0.1, 0.2, 0.05, 0.15]
    assert max_drawdown(curve) == pytest.approx(0.15)


def test_compute_metrics_empty_trades():
    m = compute_metrics([], use_net=False)
    assert m.n_trades == 0
    assert m.win_rate == 0.0


def test_compute_metrics_known_values():
    trades = [
        make_trade(100, raw_pnl=0.06, net_pnl=0.06, entry_mid=0.10, entry_fill=0.10),
        make_trade(200, raw_pnl=-0.04, net_pnl=-0.04, entry_mid=0.08, entry_fill=0.08),
    ]
    m = compute_metrics(trades, use_net=False)
    assert m.n_trades == 2
    assert m.win_rate == pytest.approx(0.5)
    assert m.avg_win_pct == pytest.approx(0.6)   # 0.06/0.10
    assert m.avg_loss_pct == pytest.approx(-0.5)  # -0.04/0.08
    assert m.ev_per_trade_pct == pytest.approx((0.6 + (-0.5)) / 2)


def test_split_train_test_holds_out_most_recent_by_end_date():
    markets = [
        MarketData("m1", [[], []], 0, end_date="2025-01-01"),
        MarketData("m2", [[], []], 0, end_date="2025-06-01"),
        MarketData("m3", [[], []], 0, end_date="2025-12-01"),
        MarketData("m4", [[], []], 0, end_date="2026-03-01"),
    ]
    train, test = split_train_test(markets, test_fraction=0.25)
    assert [m.market_id for m in train] == ["m1", "m2", "m3"]
    assert [m.market_id for m in test] == ["m4"]


def test_split_train_test_empty_input():
    assert split_train_test([]) == ([], [])


def _rebound_market(market_id="a"):
    return MarketData(
        market_id=market_id,
        outcome_series=[
            [(0, 0.50), (100, 0.70), (200, 0.90)],
            [(0, 0.50), (100, 0.30), (200, 0.10), (300, 0.16)],
        ],
        resolved_outcome_index=0,
    )


def test_sweep_runs_every_param_combo():
    markets = [_rebound_market()]
    grid = [
        SignalParams(thresh=0.88, rebound=0.03, stop=0.03),
        SignalParams(thresh=0.88, rebound=0.05, stop=0.03),
    ]
    results = sweep(markets, grid, ZERO_COST_MODEL)
    assert len(results) == 2
    for params, raw_m, net_m in results:
        assert raw_m.n_trades == 1


def test_overfitting_flag_warns_on_narrow_island():
    losing = compute_metrics([make_trade(100, -0.01, -0.01)], use_net=True)
    winning = compute_metrics([make_trade(100, 0.10, 0.10)], use_net=True)
    sweep_results = [
        (SignalParams(rebound=0.01), losing, losing),
        (SignalParams(rebound=0.02), losing, losing),
        (SignalParams(rebound=0.03), losing, losing),
        (SignalParams(rebound=0.04), losing, losing),
        (SignalParams(rebound=0.05), winning, winning),
    ]
    warning = overfitting_flag(sweep_results, profitable_fraction_threshold=0.25)
    assert warning is not None
    assert "OVERFITTING RISK" in warning


def test_overfitting_flag_silent_when_broadly_profitable():
    winning = compute_metrics([make_trade(100, 0.10, 0.10)], use_net=True)
    sweep_results = [
        (SignalParams(rebound=0.01), winning, winning),
        (SignalParams(rebound=0.02), winning, winning),
        (SignalParams(rebound=0.03), winning, winning),
    ]
    assert overfitting_flag(sweep_results) is None


def test_spread_sensitivity_ev_decreases_as_spread_widens():
    markets = [_rebound_market()]
    results = spread_sensitivity(
        markets, SignalParams(thresh=0.88, rebound=0.05, stop=0.03),
        CostModel(half_spread=0.0), half_spreads=(0.0, 0.02, 0.05),
    )
    evs = [m.ev_per_trade_pct for _, m in results]
    assert evs[0] > evs[1] > evs[2]


def test_verdict_reports_survival_when_net_ev_positive():
    trades = [make_trade(100, 0.06, 0.02, entry_mid=0.10, entry_fill=0.14)]
    raw_m = compute_metrics(trades, use_net=False)
    net_m = compute_metrics(trades, use_net=True)
    text = verdict(raw_m, net_m, [(0.02, net_m), (0.04, net_m)])
    assert "SURVIVES" in text


def test_verdict_reports_failure_when_net_ev_negative():
    trades = [make_trade(100, 0.06, -0.02, entry_mid=0.10, entry_fill=0.14)]
    raw_m = compute_metrics(trades, use_net=False)
    net_m = compute_metrics(trades, use_net=True)
    text = verdict(raw_m, net_m, [(0.02, net_m), (0.04, net_m)])
    assert "DOES NOT SURVIVE" in text


def test_verdict_no_trades():
    empty = compute_metrics([], use_net=True)
    assert "No trades" in verdict(empty, empty, [])
