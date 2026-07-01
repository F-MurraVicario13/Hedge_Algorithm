import pytest

from backtester import cli
from backtester.data.cache import DiskCache
from backtester.data import clob as clob_mod
from backtester.engine.backtest import MarketData


def test_parse_gamma_params():
    assert cli._parse_gamma_params(["tag=soccer", "closed=true"]) == {"tag": "soccer", "closed": "true"}


def test_parse_gamma_params_rejects_missing_equals():
    with pytest.raises(ValueError):
        cli._parse_gamma_params(["not-a-kv-pair"])


def test_build_param_grid_filters_non_positive_and_dedupes():
    grid = cli.build_param_grid(thresh=0.88, rebound=0.01, stop=0.005, time_limit=None)
    # rebound=0.01-0.02 would go negative and must be dropped, same for stop=0.005-0.01
    assert all(p.rebound > 0 for p in grid)
    assert all(p.stop > 0 for p in grid)
    assert all(0.0 < p.thresh < 1.0 for p in grid)


def _fake_raw_market(slug, tok0, tok1, outcome_prices, end_date="2026-01-01"):
    return {
        "id": slug, "slug": slug, "question": slug,
        "outcomes": ["Yes", "No"],
        "outcome_prices": outcome_prices,
        "clob_token_ids": [tok0, tok1],
        "closed": True, "spread": 0.02, "liquidity": 100.0, "volume24hr": 50.0,
        "end_date": end_date, "tags": [],
    }


def test_build_market_dataset_skips_non_binary_and_unresolved(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)

    raw_markets = [
        _fake_raw_market("good", "tok-a", "tok-b", [1.0, 0.0]),
        {**_fake_raw_market("not-binary", "tok-c", "tok-d", [1.0, 0.0]), "clob_token_ids": ["only-one"]},
        _fake_raw_market("unresolved", "tok-e", "tok-f", [0.5, 0.5]),
    ]
    monkeypatch.setattr(cli, "iter_closed_markets",
                         lambda cache, extra_params=None, page_size=100, max_pages=None: iter(raw_markets))
    monkeypatch.setattr(clob_mod, "_request", lambda params: {
        "history": [{"t": 0, "p": 0.5}, {"t": 100, "p": 0.6}]
    })

    markets, skipped = cli.build_market_dataset(cache, sport=None, gamma_params={}, limit=10,
                                                 interval="max", fidelity=None)

    assert [m.market_id for m in markets] == ["good"]
    assert skipped["not_binary"] == 1
    assert skipped["unresolved"] == 1


def test_build_market_dataset_respects_limit(tmp_path, monkeypatch):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    raw_markets = [_fake_raw_market(f"m{i}", f"a{i}", f"b{i}", [1.0, 0.0]) for i in range(5)]
    monkeypatch.setattr(cli, "iter_closed_markets",
                         lambda cache, extra_params=None, page_size=100, max_pages=None: iter(raw_markets))
    monkeypatch.setattr(clob_mod, "_request", lambda params: {
        "history": [{"t": 0, "p": 0.5}, {"t": 100, "p": 0.6}]
    })

    markets, _ = cli.build_market_dataset(cache, sport=None, gamma_params={}, limit=2,
                                           interval="max", fidelity=None)
    assert len(markets) == 2


def test_main_end_to_end_with_synthetic_dataset(monkeypatch, capsys):
    """
    Smoke-tests the full CLI pipeline (sweep -> best params -> test report ->
    sensitivity -> verdict) without touching the network, by monkeypatching
    build_market_dataset to return a hand-built synthetic universe.
    """
    def make_market(i, end_date):
        return MarketData(
            market_id=f"market-{i}",
            outcome_series=[
                [(0, 0.50), (100, 0.70), (200, 0.90)],
                [(0, 0.50), (100, 0.30), (200, 0.10), (300, 0.16)],
            ],
            resolved_outcome_index=0,
            end_date=end_date,
        )

    synthetic_markets = [make_market(i, f"2025-{(i % 12) + 1:02d}-01") for i in range(10)]

    monkeypatch.setattr(
        cli, "build_market_dataset",
        lambda cache, sport, gamma_params, limit, interval, fidelity: (synthetic_markets, {"not_binary": 0}),
    )

    cli.main(["--thresh", "0.88", "--rebound", "0.05", "--stop", "0.03", "--half-spread", "0.04"])

    out = capsys.readouterr().out
    assert "Parameter sweep on TRAIN set" in out
    assert "Final report on held-out TEST set" in out
    assert "Spread sensitivity" in out
    assert "VERDICT" in out


def test_main_reports_when_no_markets_found(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "build_market_dataset",
        lambda cache, sport, gamma_params, limit, interval, fidelity: ([], {"not_binary": 5}),
    )
    cli.main([])
    err = capsys.readouterr().err
    assert "No usable markets found" in err
