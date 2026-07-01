"""
Hand-crafted price paths where the correct trade P&L is known independent of
any live data, so the engine's accounting (not the data pipeline) is what's
under test here.
"""

import pytest

from backtester.costs.model import CostModel, ZERO_COST_MODEL
from backtester.engine.backtest import MarketData, simulate_all
from backtester.signal.rules import SignalParams

PARAMS = SignalParams(thresh=0.88, rebound=0.05, stop=0.03, time_limit_seconds=500)


def test_rebound_exit_pnl():
    # favorite overreacts to 0.90 at t=200, underdog bought at 0.10, rebounds to
    # 0.16 at t=300 -> known correct raw profit = 0.16 - 0.10 = 0.06 / contract.
    market = MarketData(
        market_id="rebound-case",
        outcome_series=[
            [(0, 0.50), (100, 0.70), (200, 0.90)],           # favorite
            [(0, 0.50), (100, 0.30), (200, 0.10), (300, 0.16), (400, 0.05)],  # underdog
        ],
        resolved_outcome_index=0,
    )
    trades = simulate_all([market], PARAMS, ZERO_COST_MODEL)

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_kind == "rebound"
    assert t.entry_time == 200 and t.entry_mid == pytest.approx(0.10)
    assert t.exit_time == 300 and t.exit_mid_or_outcome == pytest.approx(0.16)
    assert t.raw_pnl == pytest.approx(0.06)
    assert t.net_pnl == pytest.approx(0.06)  # zero-cost model: net must equal raw


def test_rebound_exit_pnl_with_costs_can_flip_a_winner_into_a_loser():
    market = MarketData(
        market_id="rebound-case",
        outcome_series=[
            [(0, 0.50), (100, 0.70), (200, 0.90)],
            [(0, 0.50), (100, 0.30), (200, 0.10), (300, 0.16)],
        ],
        resolved_outcome_index=0,
    )
    cost_model = CostModel(half_spread=0.04, slippage=0.0)
    trades = simulate_all([market], PARAMS, cost_model)

    t = trades[0]
    # entry_fill = 0.10 + 0.04 = 0.14 ; exit_fill = 0.16 - 0.04 = 0.12
    assert t.entry_fill == pytest.approx(0.14)
    assert t.exit_fill == pytest.approx(0.12)
    assert t.net_pnl == pytest.approx(-0.02)
    assert t.raw_pnl == pytest.approx(0.06)  # gross was a winner; costs made it a loser


def test_stop_exit_pnl():
    market = MarketData(
        market_id="stop-case",
        outcome_series=[
            [(0, 0.60), (100, 0.92)],
            [(0, 0.40), (100, 0.08), (200, 0.06), (300, 0.04)],
        ],
        resolved_outcome_index=0,
    )
    trades = simulate_all([market], PARAMS, ZERO_COST_MODEL)

    t = trades[0]
    assert t.exit_kind == "stop"
    assert t.entry_mid == pytest.approx(0.08)
    assert t.exit_mid_or_outcome == pytest.approx(0.04)
    assert t.raw_pnl == pytest.approx(-0.04)


def test_time_limit_exit_pnl():
    market = MarketData(
        market_id="time-limit-case",
        outcome_series=[
            [(0, 0.50), (100, 0.95)],
            [(0, 0.50), (100, 0.05), (300, 0.06), (700, 0.065)],
        ],
        resolved_outcome_index=0,
    )
    trades = simulate_all([market], PARAMS, ZERO_COST_MODEL)

    t = trades[0]
    assert t.exit_kind == "time_limit"
    assert t.exit_time == 700
    assert t.raw_pnl == pytest.approx(0.015)


def test_settlement_exit_survivorship_case_underdog_goes_to_zero():
    # underdog never rebounds or stops out within the series; the market
    # resolves against it. Survivorship guard: this trade must be included
    # in the results, not silently dropped for "never bouncing back."
    market = MarketData(
        market_id="never-bounces",
        outcome_series=[
            [(0, 0.55), (100, 0.95)],
            [(0, 0.45), (100, 0.05), (200, 0.04), (300, 0.03)],
        ],
        resolved_outcome_index=0,  # favorite (index 0) wins -> underdog resolves to 0.0
    )
    trades = simulate_all([market], PARAMS, ZERO_COST_MODEL)

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_kind == "settlement"
    assert t.is_settlement is True
    assert t.exit_mid_or_outcome == pytest.approx(0.0)
    assert t.raw_pnl == pytest.approx(-0.05)  # lost the full stake


def test_settlement_exit_pays_no_spread_but_entry_still_costs():
    market = MarketData(
        market_id="never-bounces",
        outcome_series=[
            [(0, 0.55), (100, 0.95)],
            [(0, 0.45), (100, 0.05), (200, 0.04), (300, 0.03)],
        ],
        resolved_outcome_index=0,
    )
    cost_model = CostModel(half_spread=0.04, slippage=0.0)
    trades = simulate_all([market], PARAMS, cost_model)

    t = trades[0]
    assert t.entry_fill == pytest.approx(0.09)  # 0.05 + 0.04 spread paid to get in
    assert t.exit_fill == pytest.approx(0.0)    # redemption at par, no spread
    assert t.net_pnl == pytest.approx(-0.09)


def test_multiple_markets_combine_into_one_trade_list():
    market_a = MarketData(
        market_id="a",
        outcome_series=[
            [(0, 0.50), (100, 0.70), (200, 0.90)],
            [(0, 0.50), (100, 0.30), (200, 0.10), (300, 0.16)],
        ],
        resolved_outcome_index=0,
    )
    market_b = MarketData(
        market_id="b",
        outcome_series=[
            [(0, 0.60), (100, 0.92)],
            [(0, 0.40), (100, 0.08), (200, 0.06), (300, 0.04)],
        ],
        resolved_outcome_index=0,
    )
    trades = simulate_all([market_a, market_b], PARAMS, ZERO_COST_MODEL)
    assert {t.market_id for t in trades} == {"a", "b"}
    assert len(trades) == 2
