# Bet Sizing Strategy — Specification

This document specifies the bet sizing logic for the Kalshi tennis prediction market system. It should be integrated into `live.py` (or a new `sizing.py` module called by `live.py`).

---

## 1. Core Concepts

### Bankroll
- **Definition**: Total capital allocated to Kalshi trading. Calculated as: Kalshi cash balance + expected value of unsettled winning positions.
- **Session snapshot**: Bankroll is calculated **once** at the start of each trading session (i.e., each day). All sizing that day references this single number. Do not recalculate mid-day after wins/losses.

### Payout Structure
- Kalshi binary contracts pay $1 per contract on a win.
- Buying a YES contract at price `p_market` costs `p_market` per contract.
- Profit on win: `(1 - p_market)` per contract. Loss on loss: `p_market` per contract.
- Net odds (profit per dollar risked): `b = (1 - p_market) / p_market`.

### Edge
- `edge = p_model - p_market`, where `p_model` is the calibrated composite probability from `predict()` and `p_market` is the Kalshi YES price for the favorite.
- A bet is only placed if `edge > EDGE_THRESHOLD` (currently 0.04).

---

## 2. Sizing Formula (Option 3: Edge-Proportional Cap)

Three layers applied in sequence:

### Layer 1: Kelly Criterion
Raw Kelly fraction for Kalshi binary structure:

```
kelly_raw = edge / (1 - p_market)
```

Apply **⅛ Kelly** (fractional Kelly = 0.125):

```
kelly_frac = kelly_raw * 0.125
```

Rationale: ¼ Kelly causes the per-match cap to bind on virtually every bet in the 62–78% market price range with 4–7pp edges, collapsing sizing to flat-betting. ⅛ Kelly keeps the intermediate calculation meaningful and allows the edge-proportional cap to shape position sizes.

### Layer 2: Edge-Proportional Per-Match Cap
The maximum fraction of bankroll risked on any single match scales with edge size:

```
edge_cap = min(0.05, 0.005 + edge * 0.5)
```

This maps:
- 4pp edge → 2.5% cap
- 5pp edge → 3.0% cap
- 6pp edge → 3.5% cap
- 7pp edge → 4.0% cap
- 8pp edge → 4.5% cap
- 10pp+ edge → 5.0% cap (hard ceiling)

The per-match stake fraction is:

```
stake_pct = min(kelly_frac, edge_cap)
```

### Layer 3: Daily Exposure Cap (Tiered by Round)
Total capital deployed across all open positions in a single day must not exceed a cap that varies by tournament round:

| Round        | Daily Cap | Rationale                                                       |
|-------------|-----------|----------------------------------------------------------------|
| R64 / R32   | 18–20%    | Most matches, strongest favorite-longshot bias, low correlation |
| R16         | 15%       | Still busy but edges slightly thinner                          |
| QF onward   | 12%       | Fewer matches, tighter pricing, more public attention          |

Implementation:

```
remaining_capacity = max(0, daily_cap - current_exposure)
stake_pct = min(stake_pct, remaining_capacity)
```

Where `current_exposure` is the sum of `stake_pct` values for all bets already placed that day.

**Priority rule**: When the daily cap would prevent placing all actionable bets, rank bets by edge (descending) and fill from the top until the cap binds. Skip remaining bets.

### Final Stake

```
stake_dollars = round(stake_pct * bankroll, 2)
```

---

## 3. Function Signature

```python
def size_bet(
    p_model: float,
    p_market: float,
    bankroll: float,
    current_exposure: float = 0.0,
    round_stage: str = "R32",      # one of: "R128", "R64", "R32", "R16", "QF", "SF", "F"
    kelly_fraction: float = 0.125,
    edge_threshold: float = 0.04,
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
        Tournament round, used to select the daily cap tier.
    kelly_fraction : float
        Fractional Kelly multiplier (default 0.125 = ⅛ Kelly).
    edge_threshold : float
        Minimum edge required to place a bet (default 0.04).

    Returns
    -------
    dict with keys:
        - "stake": float, dollar amount to bet (0.0 if no bet)
        - "stake_pct": float, fraction of bankroll
        - "edge": float, p_model - p_market
        - "kelly_raw": float, full Kelly fraction
        - "kelly_frac": float, fractional Kelly value
        - "edge_cap": float, per-match cap from edge formula
        - "daily_cap": float, applicable daily cap for this round
        - "remaining_capacity": float, daily cap headroom
        - "signal": str, one of "BET", "NO_EDGE", "CAP_BOUND"
    """
```

---

## 4. Daily Cap Mapping

```python
DAILY_CAP_BY_ROUND = {
    "R128": 0.20,
    "R64":  0.20,
    "R32":  0.18,
    "R16":  0.15,
    "QF":   0.12,
    "SF":   0.12,
    "F":    0.12,
}
```

---

## 5. Batch Sizing for a Full Day

When evaluating multiple matches in a single session:

1. Snapshot bankroll once.
2. Run `predict()` on all matches to get `p_model` for each.
3. Compute edge for each; filter to those exceeding `EDGE_THRESHOLD`.
4. **Sort by edge descending** (largest edge = highest priority).
5. Iterate through sorted list, calling `size_bet()` with cumulative `current_exposure`.
6. Stop or skip when `remaining_capacity` hits zero.

```python
def size_day(
    matches: list[dict],   # each has "p_model", "p_market", "match_id", etc.
    bankroll: float,
    round_stage: str,
) -> list[dict]:
    """
    Size all actionable bets for a trading day, respecting the daily cap.

    Parameters
    ----------
    matches : list of dict
        Each dict must contain at minimum: "p_model", "p_market".
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
```

---

## 6. Worked Example

**Setup**: $1,000 bankroll, R32 day (daily cap = 18%), 7 matches evaluated.

| Match       | p_mkt | p_model | Edge  | Kelly_raw | ⅛ Kelly | Edge Cap | Stake% | Stake  | Cumulative |
|-------------|-------|---------|-------|-----------|---------|----------|--------|--------|------------|
| Fils        | 0.64  | 0.70    | 6.0pp | 16.7%     | 2.1%    | 3.5%     | 2.1%   | $21    | 2.1%       |
| Zverev      | 0.72  | 0.77    | 5.0pp | 17.9%     | 2.2%    | 3.0%     | 2.2%   | $22    | 4.3%       |
| Paul        | 0.71  | 0.76    | 5.0pp | 17.2%     | 2.2%    | 3.0%     | 2.2%   | $22    | 6.5%       |
| Khachanov   | 0.71  | 0.76    | 5.0pp | 17.2%     | 2.2%    | 3.0%     | 2.2%   | $22    | 8.6%       |
| Fritz       | 0.71  | 0.76    | 5.0pp | 17.2%     | 2.2%    | 3.0%     | 2.2%   | $22    | 10.8%      |
| Chung       | 0.72  | 0.77    | 5.0pp | 17.9%     | 2.2%    | 3.0%     | 2.2%   | $22    | 13.0%      |
| Boulais     | 0.74  | 0.79    | 5.0pp | 19.2%     | 2.4%    | 3.0%     | 2.4%   | $24    | 15.4%      |

All 7 bets fit under the 18% R32 cap. Total deployed: $155 (15.4%).

If there were 3 more 5pp-edge bets, the 8th and 9th would partially fill, and the 10th would be skipped.

---

## 7. Integration Notes

- The `EDGE_THRESHOLD` constant (0.04) already exists in `live.py` — reuse it.
- `size_bet()` should be called after `predict()` returns a result dict, using `result["comp_p1"]` as `p_model`.
- The `display()` function in `live.py` should be extended to show sizing output (stake, edge cap, daily headroom) alongside the existing prediction output.
- For batch/automated mode: `size_day()` wraps the per-match logic and enforces priority ordering.
- Bankroll input: for CLI mode, prompt once at session start. For automated mode, read from a config or Kalshi API balance endpoint.
