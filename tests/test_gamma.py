import json

import pytest

from backtester.data import gamma
from backtester.data.cache import DiskCache


def make_raw_market(**overrides):
    raw = {
        "id": "123",
        "slug": "team-a-vs-team-b",
        "question": "Will Team A win?",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "clobTokenIds": json.dumps(["tok-yes", "tok-no"]),
        "closed": True,
        "spread": "0.02",
        "liquidity": "1000.5",
        "volume24hr": "500.25",
        "endDate": "2026-01-01T00:00:00Z",
        "tags": ["soccer"],
    }
    raw.update(overrides)
    return raw


def test_parse_market_decodes_json_string_fields():
    parsed = gamma.parse_market(make_raw_market())
    assert parsed["outcomes"] == ["Yes", "No"]
    assert parsed["outcome_prices"] == [1.0, 0.0]
    assert parsed["clob_token_ids"] == ["tok-yes", "tok-no"]
    assert parsed["closed"] is True
    assert parsed["spread"] == pytest.approx(0.02)
    assert parsed["liquidity"] == pytest.approx(1000.5)
    assert parsed["volume24hr"] == pytest.approx(500.25)


def test_parse_market_handles_missing_optional_fields():
    raw = make_raw_market(clobTokenIds=None, outcomePrices=None)
    parsed = gamma.parse_market(raw)
    assert parsed["clob_token_ids"] == []
    assert parsed["outcome_prices"] == []


def test_fetch_market_by_slug(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    captured = {}

    def fake_request(path, params):
        captured["path"] = path
        captured["params"] = params
        return [make_raw_market()]

    monkeypatch.setattr(gamma, "_request", fake_request)

    market = gamma.fetch_market(cache, slug="team-a-vs-team-b")

    assert captured["path"] == "/markets"
    assert captured["params"] == {"slug": "team-a-vs-team-b"}
    assert market["question"] == "Will Team A win?"


def test_fetch_market_raises_lookup_error_when_empty(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    monkeypatch.setattr(gamma, "_request", lambda path, params: [])

    with pytest.raises(LookupError):
        gamma.fetch_market(cache, slug="nonexistent")


def test_search_events_returns_title_slug_pairs(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    monkeypatch.setattr(
        gamma, "_request",
        lambda path, params: {"events": [{"title": "Team A vs Team B", "slug": "a-vs-b"}]},
    )

    results = gamma.search_events(cache, "team a")
    assert results == [("Team A vs Team B", "a-vs-b")]


def test_fetch_event_markets(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    monkeypatch.setattr(
        gamma, "_request",
        lambda path, params: [{"markets": [make_raw_market(), make_raw_market(slug="other-market")]}],
    )

    markets = gamma.fetch_event_markets(cache, "some-event")
    assert len(markets) == 2
    assert markets[1]["slug"] == "other-market"


def test_iter_closed_markets_paginates_until_short_page(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    pages = {
        0: [make_raw_market(slug=f"m{i}") for i in range(3)],
        3: [make_raw_market(slug="m3")],  # short page -> stop
    }

    def fake_request(path, params):
        assert path == "/markets"
        return pages[params["offset"]]

    monkeypatch.setattr(gamma, "_request", fake_request)

    results = list(gamma.iter_closed_markets(cache, page_size=3))
    assert [m["slug"] for m in results] == ["m0", "m1", "m2", "m3"]


def test_iter_closed_markets_stops_on_empty_page(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    monkeypatch.setattr(gamma, "_request", lambda path, params: [])

    results = list(gamma.iter_closed_markets(cache, page_size=100))
    assert results == []
