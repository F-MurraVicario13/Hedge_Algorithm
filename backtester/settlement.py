"""
Settlement math shared by the live hedge calculator (polymarket_hedge.py) and
the backtest engine. Every Polymarket binary contract pays $1 if its outcome
resolves YES and $0 if it resolves NO -- every P&L figure in this codebase is
derived from that single identity (`payoff`, below). Nothing computes P&L any
other way.
"""

from __future__ import annotations


def payoff(shares: float, price: float, outcome: float) -> float:
    """
    P&L for holding `shares` contracts bought at `price` through settlement,
    where `outcome` is the resolved result for that side: 1.0 if it won,
    0.0 if it lost. This is the entire settlement rule -- everything else
    (hedge_report, the backtest engine's mark-to-settlement path) composes it.
    """
    if outcome not in (0.0, 1.0):
        raise ValueError(f"outcome must be a resolved 0.0/1.0, got {outcome!r}")
    return shares * (outcome - price)


def close_before_settlement(shares: float, entry_price: float, exit_price: float) -> float:
    """P&L for selling a position before the market resolves (no settlement involved)."""
    return shares * (exit_price - entry_price)


def resolve_binary_outcome(outcome_prices) -> int:
    """
    Given a closed market's outcomePrices (e.g. ["1", "0"] or ["0", "1"]),
    return the index of the winning outcome. Raises if the prices aren't a
    clean 0/1 pair -- callers must never treat an unresolved or ambiguous
    market as ground truth (that would be a look-ahead / survivorship bug).
    """
    prices = [float(p) for p in outcome_prices]
    if sorted(prices) != [0.0, 1.0]:
        raise ValueError(f"market is not cleanly resolved: outcomePrices={outcome_prices!r}")
    return prices.index(1.0)


def hedge_report(fav_shares, fav_entry, underdog_ask, verbose=True) -> dict:
    """
    Ported from polymarket_hedge.py, rewritten in terms of `payoff` so the
    live calculator's two-legged accounting and the backtest engine's
    single-legged accounting share one settlement primitive instead of two
    parallel implementations of "contracts pay $1 or $0".

    fav_shares    : # of favorite contracts already held
    fav_entry     : price paid for the favorite (cost basis)
    underdog_ask  : price to buy the underdog now (the ASK, not the mid)
    """
    fav_cost = fav_shares * fav_entry
    fav_win_gain = payoff(fav_shares, fav_entry, 1.0)  # profit if you did nothing and fav wins

    # Shares that FULLY protect principal if the favorite loses:
    n_full = fav_cost / (1 - underdog_ask)
    cost_full = n_full * underdog_ask

    # Max underdog shares before the FAVORITE-WINS outcome turns negative:
    n_maxprofit = fav_win_gain / underdog_ask
    cost_maxprofit = n_maxprofit * underdog_ask  # == fav_win_gain

    can_do_both = n_full <= n_maxprofit
    arb_number = fav_entry + underdog_ask  # < 1.0 == free arb territory

    def outcomes_at(n_u):
        fav_wins = payoff(fav_shares, fav_entry, 1.0) + payoff(n_u, underdog_ask, 0.0)
        dog_wins = payoff(fav_shares, fav_entry, 0.0) + payoff(n_u, underdog_ask, 1.0)
        return fav_wins, dog_wins

    rep = {
        "fav_cost": fav_cost,
        "fav_win_gain_unhedged": fav_win_gain,
        "arb_number": arb_number,
        "can_fully_insure_and_profit": can_do_both,
        "full_insurance": {"shares": n_full, "cost": cost_full,
                            "fav_wins": outcomes_at(n_full)[0],
                            "underdog_wins": outcomes_at(n_full)[1]},
        "max_profitable_insurance": {"shares": n_maxprofit, "cost": cost_maxprofit,
                                      "fav_wins": outcomes_at(n_maxprofit)[0],
                                      "underdog_wins": outcomes_at(n_maxprofit)[1]},
    }

    if verbose:
        print(f"\nFavorite: {fav_shares:g} contracts @ {fav_entry:.3f}  (cost ${fav_cost:,.2f})")
        print(f"Underdog buy price (ask): {underdog_ask:.3f}")
        print(f"Unhedged profit if favorite wins: ${fav_win_gain:,.2f}")
        print("-" * 62)
        print(f"ARB CHECK  fav_entry + underdog_ask = {arb_number:.3f}", end="  ")
        if arb_number < 1:
            print("< 1.00  -> you can protect BOTH sides for a locked-in gain.")
            print("           This is arbitrage; the right move is HOLD to settlement.")
        elif abs(arb_number - 1) < 1e-9:
            print("= 1.00  -> full insurance is possible but drives profit to zero.")
        else:
            print("> 1.00  -> full insurance costs MORE than your gain; you must")
            print("           choose how much principal to protect (partial only).")
        print("-" * 62)
        f = rep["full_insurance"]
        print("FULL DOWNSIDE PROTECTION")
        print(f"  buy {f['shares']:.1f} underdog @ {underdog_ask:.3f}  = ${f['cost']:,.2f}")
        print(f"  if favorite wins:  ${f['fav_wins']:+,.2f}")
        print(f"  if underdog wins:  ${f['underdog_wins']:+,.2f}  (principal protected)")
        mp = rep["max_profitable_insurance"]
        print("MOST INSURANCE THAT STILL PROFITS IF FAVORITE WINS")
        print(f"  buy {mp['shares']:.1f} underdog @ {underdog_ask:.3f}  = ${mp['cost']:,.2f}")
        print(f"  if favorite wins:  ${mp['fav_wins']:+,.2f}  (break-even by design)")
        print(f"  if underdog wins:  ${mp['underdog_wins']:+,.2f}")
        print("-" * 62)
        print("PARTIAL COVERAGE (dial between the two above):")
        for k in (0.25, 0.5, 0.75, 1.0):
            n = k * n_full
            fw, uw = outcomes_at(n)
            print(f"  cover {int(k * 100):>3}% principal: buy {n:6.1f} @ {underdog_ask:.3f} "
                  f"(${n * underdog_ask:7.2f})  fav:${fw:+8.2f}  dog:${uw:+8.2f}")
    return rep
