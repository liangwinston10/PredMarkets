"""
Kalshi Tennis Market Fetcher
Reads open tennis prediction markets from Kalshi's public REST API.
No API key required for public market data.
Add KALSHI_API_KEY to .env for authenticated endpoints.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
API_KEY  = os.getenv("KALSHI_API_KEY")

# Tennis series tickers on the new Kalshi API
# Match-level markets (most useful for edge analysis)
TENNIS_MATCH_SERIES = [
    "KXATPMATCH",            # ATP individual match winner
    "KXWTAMATCH",            # WTA individual match winner
    "KXCHALLENGERMATCH",     # ATP Challenger match winner
    "KXATPCHALLENGERMATCH",  # ATP Challenger (alt ticker)
]

# Tournament winner markets (active + upcoming)
TENNIS_TOURNAMENT_SERIES = [
    "KXATPMC",               # ATP Monte Carlo (currently active)
    "KXTENNISGRANDSLAM",     # Tennis Grand Slam winner
    "KXATPGRANDSLAM",        # ATP Grand Slam winner
    "KXWTAGRANDSLAM",        # WTA Grand Slam winner
    "KXATPMAD",              # ATP Madrid
    "KXWTAMAD",              # WTA Madrid
    "KXATPIT",               # ATP Italian Open
    "KXWTAIT",               # WTA Italian Open
    "KXATPIWO",              # ATP Indian Wells
    "KXWTAIWO",              # WTA Indian Wells
    "KXWMENSINGLES",         # Wimbledon men's singles
    "KXWWOMENSINGLES",       # Wimbledon women's singles
    "KXATPFINALS",           # ATP Finals
    "KXWTAFINALS",           # WTA Finals
]

TENNIS_SERIES = TENNIS_MATCH_SERIES + TENNIS_TOURNAMENT_SERIES


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _fetch_series(series_ticker: str, status: str, limit: int) -> tuple[list, str | None]:
    """Fetch markets for a single series ticker."""
    params = {
        "series_ticker": series_ticker,
        "status":        status,
        "limit":         limit,
    }
    try:
        resp = requests.get(
            f"{BASE_URL}/markets",
            headers=_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("markets", []), None
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code in (401, 403):
            return [], "Kalshi API requires authentication. Add KALSHI_API_KEY to .env."
        if code == 404:
            return [], None  # Series doesn't exist right now — not an error
        return [], f"Kalshi API error {code}: {e}"
    except requests.exceptions.ConnectionError:
        return [], "Could not connect to Kalshi API. Check your internet connection."
    except requests.exceptions.Timeout:
        return [], "Kalshi API request timed out."
    except Exception as e:
        return [], f"Unexpected error: {e}"


def get_tennis_markets(status: str = "open", limit: int = 200) -> tuple[list, str | None]:
    """
    Fetch all Kalshi tennis markets across known tennis series.
    Returns (markets_list, error_message). error_message is None on success.
    """
    all_markets = []
    seen_tickers = set()
    last_err = None

    for series in TENNIS_SERIES:
        mkts, err = _fetch_series(series, status, limit)
        if err and "authentication" in err.lower():
            return [], err  # Hard auth failure — stop immediately
        if err:
            last_err = err
            continue
        for m in mkts:
            t = m.get("ticker", "")
            if t and t not in seen_tickers:
                seen_tickers.add(t)
                all_markets.append(m)

    if not all_markets and last_err:
        return [], last_err

    return all_markets, None


def get_market(ticker: str) -> tuple[dict, str | None]:
    """Fetch a single market by ticker. Returns (market_dict, error_message)."""
    try:
        resp = requests.get(
            f"{BASE_URL}/markets/{ticker}",
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("market", {}), None
    except Exception as e:
        return {}, str(e)


def _to_float(v) -> float | None:
    """Safely convert string or numeric price to float."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def parse_implied_prob(market: dict) -> float | None:
    """
    Extract implied probability (0–1) for the Yes side.
    New Kalshi API: prices are in dollars (0.00–1.00) returned as strings.
    Uses bid-ask midpoint if available, otherwise last_price_dollars.
    """
    yes_bid = _to_float(market.get("yes_bid_dollars"))
    yes_ask = _to_float(market.get("yes_ask_dollars"))
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0
    last = _to_float(market.get("last_price_dollars"))
    return last


def parse_volume(market: dict) -> float:
    """Return total volume (volume_fp on new API, returned as string)."""
    v = _to_float(market.get("volume_fp")) or _to_float(market.get("volume")) or 0.0
    return v


def search_player_markets(player_name: str, status: str = "open") -> tuple[list, str | None]:
    """
    Search tennis markets for a player name substring.
    Returns (matching_markets, error_message).
    """
    all_markets, err = get_tennis_markets(status=status, limit=500)
    if err:
        return [], err
    name_lower = player_name.strip().lower()
    if not name_lower:
        return all_markets, None
    matches = [
        m for m in all_markets
        if name_lower in m.get("title", "").lower()
        or name_lower in m.get("ticker", "").lower()
        or name_lower in m.get("yes_sub_title", "").lower()
        or name_lower in m.get("no_sub_title", "").lower()
        or name_lower in str(m.get("custom_strike", "")).lower()
    ]
    return matches, None


def enrich_market(market: dict) -> dict:
    """Add computed fields: implied_prob, implied_prob_pct, volume_display."""
    market = dict(market)
    prob = parse_implied_prob(market)
    market["implied_prob"]     = prob
    market["implied_prob_pct"] = f"{prob*100:.1f}%" if prob is not None else "—"
    vol = parse_volume(market)
    market["volume_display"]   = f"{vol:,.0f}"
    return market


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    auth_status = "authenticated" if API_KEY else "unauthenticated"
    print(f"Fetching Kalshi tennis markets ({auth_status})...")
    markets, err = get_tennis_markets()
    if err:
        print(f"Error: {err}")
    else:
        print(f"Found {len(markets)} open tennis markets.")
        for m in markets[:10]:
            m = enrich_market(m)
            print(f"  {m.get('ticker','?'):35s}  {m.get('title','')[:45]:45s}  "
                  f"prob={m['implied_prob_pct']}  vol={m['volume_display']}")
        if len(markets) > 10:
            print(f"  ... and {len(markets)-10} more")
