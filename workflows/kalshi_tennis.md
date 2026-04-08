# Workflow: Kalshi Tennis Market Integration

## Objective
Link Kalshi prediction markets to upcoming ATP/WTA tournaments and feed implied probabilities
into the edge analysis model.

## Kalshi API Overview

Base URL: `https://trading-api.kalshi.com/trade-api/v2`

### Authentication
- **Unauthenticated**: Can read markets, prices, volume. Sufficient for monitoring.
- **Authenticated**: Adds trade history, portfolio, order placement.
  Add to `.env`: `KALSHI_API_KEY=your_key_here`

### Tennis Market Structure
- **Series ticker**: `TEN` (all tennis)
- **Event tickers**: Follow pattern `TEN-<TOURNAMENT>-<YEAR>`
  - Examples: `TEN-WIMB-26`, `TEN-RG-26`, `TEN-USO-26`, `TEN-AO-26`
- **Match markets**: Deeper — e.g. `TEN-WIMB26-DJOK-NADA`
- **Prices**: In cents (0–100). Implied prob = `(yes_bid + yes_ask) / 200`

### Known Market Naming Patterns
| Tournament | Ticker prefix |
|---|---|
| Australian Open | `TEN-AO` |
| Roland Garros | `TEN-RG` |
| Wimbledon | `TEN-WIMB` |
| US Open | `TEN-USO` |
| Indian Wells | `TEN-IW` |
| Miami Open | `TEN-MIA` |
| Madrid Open | `TEN-MAD` |
| Italian Open | `TEN-ROM` |
| Cincinnati | `TEN-CIN` |
| Canadian Open | `TEN-CAN` |

These patterns may change — always verify in the dashboard by searching the tournament name.

## Step-by-Step: Using Tab 2 → Tab 3

1. Go to **Kalshi Markets** tab
2. Search for a player name (e.g. "Sinner") or tournament ("Wimbledon")
3. Click the matching market row to expand details
4. Verify the implied probability looks reasonable (cross-reference with sportsbooks)
5. Click **"Use this market in Edge Analysis"**
6. Switch to **Edge Analysis** tab
7. Select the same two players, surface, and best-of format
8. Check **"Include market odds"** — it will be pre-filled with Kalshi's implied prob
9. Submit → read the edge signal

## Interpreting the Edge Signal

| Signal | Meaning | Action |
|---|---|---|
| VALUE — model favors P1 | Model win% > market win% by >4% | Consider backing P1 |
| VALUE — model favors P2 | Model win% < market win% by >4% | Consider backing P2 |
| Model and market roughly agree | Edge within ±4% | No actionable edge |

**Important**: The model is calibrated for ATP 2024+ and WTA 2023–2024 out-of-sample data.
It does not account for injuries, withdrawals, or very recent form changes not in the Sackmann dataset.
Always verify with additional sources before acting.

## Getting a Kalshi API Key

1. Sign up at https://kalshi.com
2. Go to Account → API Keys → Generate new key
3. Add to `.env`:
   ```
   KALSHI_API_KEY=your_key_here
   ```
4. Restart the dashboard — it will auto-pick up the key

## Edge Cases

- **No markets for a match**: Kalshi doesn't always have markets for every match. Tournament-level
  winner markets are more common than individual match markets.
- **Market already settled**: Closed markets show `status=settled`. Switch the Status filter to "closed"
  to see historical prices.
- **Price is 0 or 100**: Market is highly one-sided (near certainty). Check if there's been a
  withdrawal/retirement.
- **Volume is very low**: Thin markets — implied probability may not be reliable. Use sportsbook odds instead.
