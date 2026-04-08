"""
Bet Sizing Module — Kalshi Tennis Prediction Markets
Implements Edge-Proportional Cap sizing (Option 3) as specified in bet_sizing_strategy.md.
"""

from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────────────

EDGE_THRESHOLD = 0.04  # minimum edge to place a bet (mirrors live.py)
KELLY_FRACTION = 0.125  # ⅛ Kelly

DAILY_CAP_BY_ROUND: dict[str, float] = {
    "R128": 0.20,
    "R64":  0.20,
    "R32":  0.18,
    "R16":  0.15,
    "QF":   0.12,
    "SF":   0.12,
    "F":    0.12,
}

# ── Core sizing function ───────────────────────────────────────────────────────

def size_bet(
    p_model: float,
    p_market: float,
    bankroll: float,
    current_exposure: float = 0.0,
    round_stage: str = "R32",
    kelly_fraction: float = KELLY_FRACTION,
    edge_threshold: float = EDGE_THRESHOLD,
) -> dict:
    """
    Compute bet size for a single match.

    Parameters
    ----------
    p_model : float
        Calibrated composite win probability from predict().
    p_market : float
        Kalshi YES price (implied probability) for the favorite.
    bankroll : float
        Session-start bankroll snapshot.
    current_exposure : float
        Sum of stake fractions already deployed today (0.0 to 1.0).
    round_stage : str
        Tournament round; selects daily cap tier.
    kelly_fraction : float
        Fractional Kelly multiplier (default 0.125 = ⅛ Kelly).
    edge_threshold : float
        Minimum edge required to place a bet (default 0.04).

    Returns
    -------
    dict with keys:
        stake           – dollar amount to bet (0.0 if no bet)
        stake_pct       – fraction of bankroll
        edge            – p_model - p_market
        kelly_raw       – full Kelly fraction
        kelly_frac      – fractional Kelly value
        edge_cap        – per-match cap from edge formula
        daily_cap       – applicable daily cap for this round
        remaining_capacity – daily cap headroom
        signal          – "BET", "NO_EDGE", or "CAP_BOUND"
    """
    edge = p_model - p_market
    daily_cap = DAILY_CAP_BY_ROUND.get(round_stage.upper(), 0.18)
    remaining_capacity = max(0.0, daily_cap - current_exposure)

    # Layer 0: edge filter
    if edge <= edge_threshold:
        return {
            "stake": 0.0,
            "stake_pct": 0.0,
            "edge": edge,
            "kelly_raw": 0.0,
            "kelly_frac": 0.0,
            "edge_cap": 0.0,
            "daily_cap": daily_cap,
            "remaining_capacity": remaining_capacity,
            "signal": "NO_EDGE",
        }

    # Layer 1: Kelly criterion
    kelly_raw = edge / (1 - p_market)
    kelly_frac = kelly_raw * kelly_fraction

    # Layer 2: edge-proportional per-match cap
    edge_cap = min(0.05, 0.005 + edge * 0.5)
    stake_pct = min(kelly_frac, edge_cap)

    # Layer 3: daily exposure cap
    stake_pct = min(stake_pct, remaining_capacity)

    signal = "BET" if stake_pct > 0 else "CAP_BOUND"
    stake_dollars = round(stake_pct * bankroll, 2)

    return {
        "stake": stake_dollars,
        "stake_pct": stake_pct,
        "edge": edge,
        "kelly_raw": kelly_raw,
        "kelly_frac": kelly_frac,
        "edge_cap": edge_cap,
        "daily_cap": daily_cap,
        "remaining_capacity": remaining_capacity,
        "signal": signal,
    }


# ── Batch sizing for a full day ────────────────────────────────────────────────

def size_day(
    matches: list[dict],
    bankroll: float,
    round_stage: str,
) -> list[dict]:
    """
    Size all actionable bets for a trading day, respecting the daily cap.

    Parameters
    ----------
    matches : list of dict
        Each dict must contain at minimum: "p_model", "p_market".
        Optional key "match_id" is preserved in output.
    bankroll : float
        Session-start bankroll.
    round_stage : str
        Tournament round for daily cap selection.

    Returns
    -------
    list of dict
        Each input match dict augmented with sizing output from size_bet().
        Sorted by edge descending. Non-actionable matches included with stake=0.
    """
    # Compute edge for all matches and sort by edge descending (priority rule)
    annotated = []
    for m in matches:
        edge = m["p_model"] - m["p_market"]
        annotated.append({**m, "_edge": edge})
    annotated.sort(key=lambda x: x["_edge"], reverse=True)

    current_exposure = 0.0
    results = []

    for m in annotated:
        sizing = size_bet(
            p_model=m["p_model"],
            p_market=m["p_market"],
            bankroll=bankroll,
            current_exposure=current_exposure,
            round_stage=round_stage,
        )
        if sizing["signal"] == "BET":
            current_exposure += sizing["stake_pct"]

        result = {k: v for k, v in m.items() if k != "_edge"}
        result.update(sizing)
        results.append(result)

    return results


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Reproduces the worked example from bet_sizing_strategy.md (Section 6)
    BANKROLL = 1000.0
    ROUND = "R32"

    example_matches = [
        {"match_id": "Fils",      "p_market": 0.64, "p_model": 0.70},
        {"match_id": "Zverev",    "p_market": 0.72, "p_model": 0.77},
        {"match_id": "Paul",      "p_market": 0.71, "p_model": 0.76},
        {"match_id": "Khachanov", "p_market": 0.71, "p_model": 0.76},
        {"match_id": "Fritz",     "p_market": 0.71, "p_model": 0.76},
        {"match_id": "Chung",     "p_market": 0.72, "p_model": 0.77},
        {"match_id": "Boulais",   "p_market": 0.74, "p_model": 0.79},
    ]

    results = size_day(example_matches, bankroll=BANKROLL, round_stage=ROUND)

    print(f"{'Match':<12} {'p_mkt':>6} {'p_mdl':>6} {'Edge':>6} "
          f"{'Kelly%':>8} {'EdgeCap':>8} {'Stk%':>6} {'Stake':>7} {'Cumul':>7} {'Signal'}")
    print("-" * 90)

    cumulative = 0.0
    for r in results:
        cumulative += r["stake_pct"]
        print(
            f"{r['match_id']:<12} "
            f"{r['p_market']:>6.2f} "
            f"{r['p_model']:>6.2f} "
            f"{r['edge']:>6.2%} "
            f"{r['kelly_frac']:>8.2%} "
            f"{r['edge_cap']:>8.2%} "
            f"{r['stake_pct']:>6.2%} "
            f"${r['stake']:>6.2f} "
            f"{cumulative:>7.2%} "
            f"{r['signal']}"
        )
    print(f"\nTotal deployed: ${sum(r['stake'] for r in results):.2f} "
          f"({sum(r['stake_pct'] for r in results):.1%} of bankroll)")
