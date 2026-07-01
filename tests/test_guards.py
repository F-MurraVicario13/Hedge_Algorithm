import pytest

from backtester.guards import assert_no_lookahead, price_at_or_before


def test_price_at_or_before_returns_last_qualifying_point():
    series = [(100, 0.1), (200, 0.2), (300, 0.3)]
    assert price_at_or_before(series, 250) == (200, 0.2)
    assert price_at_or_before(series, 300) == (300, 0.3)
    assert price_at_or_before(series, 100) == (100, 0.1)


def test_price_at_or_before_returns_none_when_nothing_qualifies():
    series = [(100, 0.1), (200, 0.2)]
    assert price_at_or_before(series, 50) is None


def test_price_at_or_before_never_returns_a_future_point():
    series = [(100, 0.1), (200, 0.2), (300, 0.9)]
    # even though 0.9 might look like a tempting "answer", it's after t=250 and must not be returned
    result = price_at_or_before(series, 250)
    assert result[0] <= 250


def test_assert_no_lookahead_raises_on_future_point():
    with pytest.raises(AssertionError):
        assert_no_lookahead(100, [(50, 0.1), (150, 0.2)])


def test_assert_no_lookahead_passes_when_all_points_are_past_or_present():
    assert_no_lookahead(100, [(50, 0.1), (100, 0.2)])  # should not raise
