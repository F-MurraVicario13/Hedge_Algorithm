import pytest

from backtester.costs.model import CostModel, ZERO_COST_MODEL, entry_fill_price, exit_fill_price


def test_buy_price_adds_half_spread_and_slippage():
    model = CostModel(half_spread=0.04, slippage=0.01)
    assert model.buy_price(0.10) == pytest.approx(0.15)


def test_sell_price_subtracts_half_spread_and_slippage():
    model = CostModel(half_spread=0.04, slippage=0.01)
    assert model.sell_price(0.20) == pytest.approx(0.15)


def test_buy_price_caps_below_price_cap():
    model = CostModel(half_spread=0.10, price_cap=0.99)
    assert model.buy_price(0.95) == pytest.approx(0.99)


def test_sell_price_floors_above_price_floor():
    model = CostModel(half_spread=0.10, price_floor=0.01)
    assert model.sell_price(0.05) == pytest.approx(0.01)


def test_zero_cost_model_is_a_no_op():
    assert ZERO_COST_MODEL.buy_price(0.33) == pytest.approx(0.33)
    assert ZERO_COST_MODEL.sell_price(0.33) == pytest.approx(0.33)


def test_invalid_fill_probability_rejected():
    with pytest.raises(ValueError):
        CostModel(fill_probability=0.0)
    with pytest.raises(ValueError):
        CostModel(fill_probability=1.5)


def test_negative_half_spread_or_slippage_rejected():
    with pytest.raises(ValueError):
        CostModel(half_spread=-0.01)
    with pytest.raises(ValueError):
        CostModel(slippage=-0.01)


def test_entry_fill_price_uses_buy_price():
    model = CostModel(half_spread=0.04, slippage=0.0)
    assert entry_fill_price(0.10, model) == pytest.approx(0.14)


def test_exit_fill_price_live_market_pays_spread():
    model = CostModel(half_spread=0.04, slippage=0.0)
    assert exit_fill_price(0.16, model, is_settlement=False) == pytest.approx(0.12)


def test_exit_fill_price_settlement_has_no_spread():
    model = CostModel(half_spread=0.04, slippage=0.0)
    # a resolved-YES redemption pays exactly $1, no bid-side haircut
    assert exit_fill_price(1.0, model, is_settlement=True) == pytest.approx(1.0)
    assert exit_fill_price(0.0, model, is_settlement=True) == pytest.approx(0.0)
