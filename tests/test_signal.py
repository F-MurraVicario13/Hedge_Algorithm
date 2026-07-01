import pytest

from backtester.signal.rules import (
    SignalParams,
    detect_entries,
    detect_exit,
    simulate_market_trades,
)


def test_detect_entries_finds_upward_cross_and_prices_underdog_at_same_time():
    fav = [(0, 0.50), (100, 0.70), (200, 0.90)]
    dog = [(0, 0.50), (100, 0.30), (200, 0.10)]
    params = SignalParams(thresh=0.88)

    events = detect_entries([fav, dog], params)

    assert len(events) == 1
    e = events[0]
    assert e.entry_time == 200
    assert e.entry_price == pytest.approx(0.10)
    assert e.favorite_outcome_index == 0
    assert e.underdog_outcome_index == 1


def test_detect_entries_is_symmetric_either_side_can_be_favorite():
    # this time outcome index 1 is the one that crosses thresh
    a = [(0, 0.50), (100, 0.10)]
    b = [(0, 0.50), (100, 0.90)]
    params = SignalParams(thresh=0.88)

    events = detect_entries([a, b], params)

    assert len(events) == 1
    assert events[0].favorite_outcome_index == 1
    assert events[0].underdog_outcome_index == 0
    assert events[0].entry_price == pytest.approx(0.10)


def test_detect_entries_rejects_inconsistent_underdog_price():
    # favorite crosses thresh, but the "underdog" side is nowhere near a discount
    # (e.g. bad data, or a 3+ outcome market misread as binary) -- must not fire
    fav = [(0, 0.50), (100, 0.90)]
    dog = [(0, 0.50), (100, 0.60)]  # way above (1 - 0.88) + 0.05 tolerance
    params = SignalParams(thresh=0.88, consistency_tolerance=0.05)

    events = detect_entries([fav, dog], params)
    assert events == []


def test_detect_entries_no_lookahead_uses_price_at_decision_time_not_later():
    fav = [(0, 0.50), (100, 0.90)]
    # at t=100 (decision time) the dog is NOT at a discount; it only crashes later.
    # A look-ahead bug would peek at t=150 and wrongly fire an entry.
    dog = [(0, 0.50), (100, 0.30), (150, 0.05)]
    params = SignalParams(thresh=0.88, consistency_tolerance=0.05)

    events = detect_entries([fav, dog], params)
    assert events == []  # must NOT fire just because the dog crashes shortly after


def test_detect_exit_rebound():
    entry_time, entry_price = 200, 0.10
    future = [(300, 0.16)]
    exit_event = detect_exit(entry_time, entry_price, future, SignalParams(rebound=0.05, stop=0.03), None)
    assert exit_event.kind == "rebound"
    assert exit_event.exit_time == 300
    assert exit_event.exit_price == pytest.approx(0.16)
    assert exit_event.is_settlement is False


def test_detect_exit_stop():
    entry_time, entry_price = 100, 0.08
    future = [(200, 0.06), (300, 0.04)]
    exit_event = detect_exit(entry_time, entry_price, future, SignalParams(rebound=0.05, stop=0.03), None)
    assert exit_event.kind == "stop"
    assert exit_event.exit_time == 300
    assert exit_event.exit_price == pytest.approx(0.04)


def test_detect_exit_stops_at_first_trigger_not_a_later_more_extreme_price():
    # no-look-ahead for exits: must fire at the FIRST qualifying point, never
    # scan ahead for a "better" price before deciding.
    entry_time, entry_price = 0, 0.10
    future = [(10, 0.08), (20, 0.065), (30, 0.03)]  # stop threshold is 0.10-0.03=0.07
    exit_event = detect_exit(entry_time, entry_price, future, SignalParams(rebound=0.05, stop=0.03), None)
    assert exit_event.exit_time == 20
    assert exit_event.exit_price == pytest.approx(0.065)


def test_detect_exit_time_limit():
    entry_time, entry_price = 100, 0.05
    future = [(300, 0.06), (700, 0.065)]  # neither rebound (>=0.10) nor stop (<=0.02) trigger
    params = SignalParams(rebound=0.05, stop=0.03, time_limit_seconds=500)
    exit_event = detect_exit(entry_time, entry_price, future, params, None)
    assert exit_event.kind == "time_limit"
    assert exit_event.exit_time == 700
    assert exit_event.exit_price == pytest.approx(0.065)


def test_detect_exit_settlement_when_series_ends_with_no_trigger():
    entry_time, entry_price = 100, 0.05
    future = [(200, 0.04), (300, 0.03)]  # never rebounds or stops out
    exit_event = detect_exit(entry_time, entry_price, future, SignalParams(rebound=0.05, stop=0.03), resolved_outcome=0.0)
    assert exit_event.kind == "settlement"
    assert exit_event.exit_time == 300
    assert exit_event.exit_price == pytest.approx(0.0)
    assert exit_event.is_settlement is True


def test_detect_exit_raises_if_unresolved_and_no_trigger():
    with pytest.raises(ValueError):
        detect_exit(0, 0.05, [(100, 0.06)], SignalParams(rebound=0.05, stop=0.03), resolved_outcome=None)


def test_simulate_market_trades_blocks_pyramiding():
    # favorite crosses thresh twice; the second cross (t=200) happens while the
    # first trade is still open (it doesn't exit until t=400) and must be
    # ignored -- no overlapping positions in the same market.
    fav = [(0, 0.5), (100, 0.90), (150, 0.80), (200, 0.91)]
    dog = [(0, 0.5), (100, 0.10), (150, 0.12), (200, 0.11), (400, 0.16)]
    params = SignalParams(thresh=0.88, rebound=0.05, stop=0.03)

    trades = simulate_market_trades([fav, dog], resolved_outcome_index=0, params=params)

    assert len(trades) == 1
    assert trades[0].entry.entry_time == 100
    assert trades[0].exit.exit_time == 400
