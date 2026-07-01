import pytest

from backtester.data import clob
from backtester.data.cache import DiskCache


def test_fetch_price_history_parses_and_sorts(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)

    unsorted_history = {
        "history": [
            {"t": 300, "p": 0.55},
            {"t": 100, "p": 0.40},
            {"t": 200, "p": 0.50},
        ]
    }

    captured = {}

    def fake_request(params):
        captured["params"] = params
        return unsorted_history

    monkeypatch.setattr(clob, "_request", fake_request)

    history = clob.fetch_price_history(cache, "tok-yes", interval="max", fidelity=60)

    assert captured["params"] == {"market": "tok-yes", "interval": "max", "fidelity": 60}
    assert history == [(100, 0.40), (200, 0.50), (300, 0.55)]


def test_fetch_price_history_empty(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    monkeypatch.setattr(clob, "_request", lambda params: {"history": []})

    assert clob.fetch_price_history(cache, "tok-yes") == []


def test_fetch_price_history_uses_cache(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    calls = []

    def fake_request(params):
        calls.append(params)
        return {"history": [{"t": 1, "p": 0.5}]}

    monkeypatch.setattr(clob, "_request", fake_request)

    clob.fetch_price_history(cache, "tok-yes")
    clob.fetch_price_history(cache, "tok-yes")

    assert len(calls) == 1
