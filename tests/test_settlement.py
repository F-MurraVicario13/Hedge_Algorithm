import pytest

from backtester.settlement import (
    close_before_settlement,
    hedge_report,
    payoff,
    resolve_binary_outcome,
)


def test_payoff_win():
    # 10 contracts bought at 0.20, resolves YES -> pay $1 each
    assert payoff(10, 0.20, 1.0) == pytest.approx(8.0)


def test_payoff_loss():
    # same position, resolves NO -> pay $0 each, lose the stake
    assert payoff(10, 0.20, 0.0) == pytest.approx(-2.0)


def test_payoff_rejects_unresolved_outcome():
    with pytest.raises(ValueError):
        payoff(10, 0.20, 0.5)


def test_close_before_settlement():
    # bought at 0.20, sold at 0.35 before the market resolved
    assert close_before_settlement(10, 0.20, 0.35) == pytest.approx(1.5)


def test_resolve_binary_outcome_yes_wins():
    assert resolve_binary_outcome(["0.999999", "0.000001"]) == 0


def test_resolve_binary_outcome_no_wins():
    assert resolve_binary_outcome(["0.000001", "0.999999"]) == 1


def test_resolve_binary_outcome_rejects_unresolved_market():
    # exact ties remain ambiguous and should still be rejected
    with pytest.raises(ValueError):
        resolve_binary_outcome(["0.5", "0.5"])


class TestHedgeReportKnownAnswer:
    """
    fav_shares=100 @ 0.90 (cost $90, unhedged gain-if-fav-wins = $10)
    underdog_ask=0.12  ->  arb_number = 0.90 + 0.12 = 1.02 (> 1, no free arb)

    Hand-computed exact fractions:
      n_full       = 90 / 0.88        = 1125/11  = 102.272727...
      n_maxprofit  = 10 / 0.12        = 250/3    = 83.333333...
      fav_wins@n_full   = 10 - (1125/11)*0.12   = -25/11   = -2.272727...
      dog_wins@n_full   = (1125/11)*0.88 - 90   = 0.0        (principal fully protected)
      fav_wins@n_maxprofit  = 10 - (250/3)*0.12 = 0.0        (break-even by design)
      dog_wins@n_maxprofit  = (250/3)*0.88 - 90 = -50/3    = -16.666666...
    """

    def setup_method(self):
        self.rep = hedge_report(fav_shares=100, fav_entry=0.90, underdog_ask=0.12, verbose=False)

    def test_basic_figures(self):
        assert self.rep["fav_cost"] == pytest.approx(90.0)
        assert self.rep["fav_win_gain_unhedged"] == pytest.approx(10.0)
        assert self.rep["arb_number"] == pytest.approx(1.02)
        assert self.rep["can_fully_insure_and_profit"] is False

    def test_full_insurance_leg(self):
        f = self.rep["full_insurance"]
        assert f["shares"] == pytest.approx(1125 / 11)
        assert f["cost"] == pytest.approx(135 / 11)
        assert f["fav_wins"] == pytest.approx(-25 / 11)
        assert f["underdog_wins"] == pytest.approx(0.0, abs=1e-9)

    def test_max_profitable_insurance_leg(self):
        mp = self.rep["max_profitable_insurance"]
        assert mp["shares"] == pytest.approx(250 / 3)
        assert mp["cost"] == pytest.approx(10.0)
        assert mp["fav_wins"] == pytest.approx(0.0, abs=1e-9)
        assert mp["underdog_wins"] == pytest.approx(-50 / 3)


def test_hedge_report_arb_case_is_free_money():
    # fav_entry + underdog_ask < 1.0 -> full insurance can protect both sides profitably
    rep = hedge_report(fav_shares=100, fav_entry=0.60, underdog_ask=0.30, verbose=False)
    assert rep["arb_number"] == pytest.approx(0.90)
    assert rep["can_fully_insure_and_profit"] is True
    f = rep["full_insurance"]
    # both outcomes should be non-negative in true arb territory
    assert f["fav_wins"] >= 0
    assert f["underdog_wins"] >= 0
