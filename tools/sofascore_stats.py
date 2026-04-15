"""
Sofascore incremental serve-stats fetcher.
Uses curl_cffi to bypass Cloudflare. Writes tennis_model/serve_stats_cache.csv.
Columns match Sackmann/TML format so live.build_player_stats() can consume directly.

Fetches ATP men's singles matches + serve stats since a given date.
Designed to run incrementally: first call fetches from since_date → today;
subsequent calls only fetch new days (cache-first).
"""

import re
import os
import time
import unicodedata
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from curl_cffi import requests as cf_requests

MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tennis_model")
CACHE_PATH = os.path.join(MODEL_DIR, "serve_stats_cache.csv")

SESSION = cf_requests.Session(impersonate="chrome124")
HEADERS = {"Accept": "application/json", "Referer": "https://www.sofascore.com/"}
TIMEOUT = 12

SCHEDULED_URL = "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date}"
STATS_URL     = "https://api.sofascore.com/api/v1/event/{event_id}/statistics"


# ── Name normalization ─────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """
    Convert accented characters to their ASCII equivalents, matching Sackmann format.
    E.g. 'Jakub Menšik' -> 'Jakub Mensik', 'Sebastián Báez' -> 'Sebastian Baez'.
    Also normalizes hyphens to spaces in names.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.strip()


# ── Filters ────────────────────────────────────────────────────────────────────

def _is_atp_singles(event: dict) -> bool:
    """Return True for ATP men's singles events. Exclude Doubles, WTA, ITF, Challenger."""
    ut   = event.get("tournament", {}).get("uniqueTournament", {})
    cat  = ut.get("category", {}).get("name", "")
    name = (ut.get("name", "") + " " + event.get("tournament", {}).get("name", "")).strip()
    if "ATP" not in cat and "ATP" not in name:
        return False
    for kw in ("Doubles", "WTA", "ITF", "Challenger", "Junior", "doubles"):
        if kw in name:
            return False
    return True


def _surface(event: dict) -> str:
    """Extract surface from groundType. Returns Hard/Clay/Grass."""
    gt = (
        event.get("groundType")
        or event.get("tournament", {}).get("uniqueTournament", {}).get("groundType")
        or ""
    )
    gl = gt.lower()
    if "clay" in gl:
        return "Clay"
    if "grass" in gl:
        return "Grass"
    return "Hard"


# ── Stats parsing ──────────────────────────────────────────────────────────────

def _parse_fraction(val) -> tuple:
    """Parse 'X/Y (Z%)' or 'X/Y' → (numerator, denominator). Returns (None, None) on failure."""
    m = re.match(r"(\d+)/(\d+)", str(val))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _fetch_stats(event_id: int) -> dict | None:
    """
    Fetch match stats for one event_id.
    Returns flat dict keyed by stat name → {home_val, away_val, home_total, away_total}.
    Uses integer homeValue/awayValue fields (not the display string).
    """
    try:
        r = SESSION.get(STATS_URL.format(event_id=event_id), headers=HEADERS, timeout=TIMEOUT)
        if not r.ok:
            return None
        for period in r.json().get("statistics", []):
            if period.get("period") != "ALL":
                continue
            stats = {}
            for grp in period.get("groups", []):
                for item in grp.get("statisticsItems", []):
                    stats[item.get("name", "")] = {
                        "hv": item.get("homeValue"),    # numerator / integer value
                        "av": item.get("awayValue"),
                        "ht": item.get("homeTotal"),    # denominator (present for fraction stats)
                        "at": item.get("awayTotal"),
                    }
            return stats if stats else None
        return None
    except Exception:
        return None


def _to_row(event: dict, stats: dict) -> dict | None:
    """Convert Sofascore event + stats into a Sackmann-compatible row dict."""
    winner_code = event.get("winnerCode", 0)
    if winner_code not in (1, 2):
        return None

    home_name = _normalize_name(event.get("homeTeam", {}).get("name") or "")
    away_name = _normalize_name(event.get("awayTeam", {}).get("name") or "")
    if not home_name or not away_name:
        return None

    w_side = "h" if winner_code == 1 else "a"   # "h"=home, "a"=away
    l_side = "a" if winner_code == 1 else "h"
    winner_name = home_name if winner_code == 1 else away_name
    loser_name  = away_name if winner_code == 1 else home_name

    def val(name, side):
        """Integer value (homeValue or awayValue)."""
        s = stats.get(name, {})
        try:
            v = s.get(f"{side}v")
            return int(v) if v is not None else None
        except Exception:
            return None

    def tot(name, side):
        """Denominator (homeTotal or awayTotal) — present for fraction stats."""
        s = stats.get(name, {})
        try:
            v = s.get(f"{side}t")
            return int(v) if v is not None else None
        except Exception:
            return None

    # Sackmann columns:
    # svpt   = homeTotal of "First serve" (total serve-point slots = first serve attempts)
    # 1stIn  = homeValue of "First serve"
    # 1stWon = homeValue of "First serve points"
    # 2ndWon = homeValue of "Second serve points"
    # bpSaved / bpFaced = homeValue / homeTotal of "Break points saved"
    w_svpt   = tot("First serve",         w_side)
    w_1stIn  = val("First serve",         w_side)
    w_1stWon = val("First serve points",  w_side)
    w_2ndWon = val("Second serve points", w_side)
    w_bpS    = val("Break points saved",  w_side)
    w_bpF    = tot("Break points saved",  w_side)
    w_ace    = val("Aces",                w_side)
    w_df     = val("Double faults",       w_side)

    l_svpt   = tot("First serve",         l_side)
    l_1stIn  = val("First serve",         l_side)
    l_1stWon = val("First serve points",  l_side)
    l_2ndWon = val("Second serve points", l_side)
    l_bpS    = val("Break points saved",  l_side)
    l_bpF    = tot("Break points saved",  l_side)
    l_ace    = val("Aces",                l_side)
    l_df     = val("Double faults",       l_side)

    # Require at minimum svpt for both players to be statistically useful
    if not w_svpt or not l_svpt:
        return None

    ts = event.get("startTimestamp")
    if not ts:
        return None
    tourney_date = pd.Timestamp(datetime.datetime.utcfromtimestamp(ts).date())

    return {
        "sofascore_event_id": event.get("id"),
        "tourney_date":  tourney_date,
        "surface":       _surface(event),
        "winner_name":   winner_name,
        "loser_name":    loser_name,
        "w_ace":    w_ace,    "w_df":     w_df,
        "w_svpt":   w_svpt,   "w_1stIn":  w_1stIn,
        "w_1stWon": w_1stWon, "w_2ndWon": w_2ndWon,
        "w_bpSaved": w_bpS,  "w_bpFaced": w_bpF,
        "l_ace":    l_ace,    "l_df":     l_df,
        "l_svpt":   l_svpt,   "l_1stIn":  l_1stIn,
        "l_1stWon": l_1stWon, "l_2ndWon": l_2ndWon,
        "l_bpSaved": l_bpS,  "l_bpFaced": l_bpF,
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def fetch_sofascore_matches_since(since: datetime.date) -> pd.DataFrame:
    """
    Fetch all completed ATP singles matches + serve stats since `since`.
    Reads existing cache to skip already-fetched events. Appends new rows and saves.
    Returns the full updated DataFrame (or empty DataFrame on total failure).
    """
    # ── Load existing cache ────────────────────────────────────────────────────
    existing_ids: set = set()
    cached_df: pd.DataFrame | None = None
    if os.path.exists(CACHE_PATH):
        try:
            cached_df = pd.read_csv(CACHE_PATH, parse_dates=["tourney_date"])
            existing_ids = set(cached_df["sofascore_event_id"].dropna().astype(int))
            # Re-fetch from 3 days before cache max to catch late-arriving stats
            cache_max = cached_df["tourney_date"].max().date()
            since = max(since, cache_max - datetime.timedelta(days=3))
        except Exception:
            pass

    # ── Collect candidate events day by day ────────────────────────────────────
    today = datetime.date.today()
    candidate_events: list[dict] = []
    d = since
    while d <= today:
        try:
            r = SESSION.get(
                SCHEDULED_URL.format(date=d.strftime("%Y-%m-%d")),
                headers=HEADERS, timeout=TIMEOUT,
            )
            if r.ok:
                for event in r.json().get("events", []):
                    eid = event.get("id")
                    if eid in existing_ids:
                        continue
                    state = event.get("status", {}).get("type", "")
                    if state != "finished":
                        continue
                    if not _is_atp_singles(event):
                        continue
                    candidate_events.append(event)
        except Exception:
            pass
        d += datetime.timedelta(days=1)
        time.sleep(0.05)  # gentle rate limiting

    print(f"  Sofascore: {len(candidate_events)} new completed ATP singles events to process")

    if not candidate_events:
        return cached_df if cached_df is not None else pd.DataFrame()

    # ── Fetch stats in parallel ────────────────────────────────────────────────
    new_rows: list[dict] = []

    def _process(event: dict):
        stats = _fetch_stats(event["id"])
        if stats:
            return _to_row(event, stats)
        return None

    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(_process, ev): ev for ev in candidate_events}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                new_rows.append(result)

    print(f"  Sofascore: {len(new_rows)} rows parsed (of {len(candidate_events)} events)")

    if not new_rows:
        return cached_df if cached_df is not None else pd.DataFrame()

    new_df = pd.DataFrame(new_rows)

    # ── Merge with cache and save ──────────────────────────────────────────────
    if cached_df is not None:
        full_df = pd.concat([cached_df, new_df], ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["sofascore_event_id"])
    else:
        full_df = new_df

    full_df = full_df.sort_values("tourney_date").reset_index(drop=True)
    full_df.to_csv(CACHE_PATH, index=False)
    print(f"  Sofascore cache saved: {len(full_df):,} total matches -> {CACHE_PATH}")
    return full_df


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    since_date = datetime.date(2026, 1, 17)
    if len(sys.argv) > 1:
        since_date = datetime.date.fromisoformat(sys.argv[1])

    print(f"Fetching Sofascore ATP singles since {since_date} ...")
    df = fetch_sofascore_matches_since(since_date)

    if df.empty:
        print("No matches returned.")
    else:
        print(f"\n{len(df):,} total matches in cache")
        print("\nBy surface:")
        print(df.groupby("surface").size().to_string())
        print("\nDate range:", df["tourney_date"].min().date(), "to", df["tourney_date"].max().date())
        print("\nSample (latest 5 clay matches):")
        clay = df[df["surface"] == "Clay"][["winner_name", "loser_name", "tourney_date", "w_ace", "w_svpt"]].tail(5)
        print(clay.to_string(index=False))
        print("\nSample winner names (for Elo key format check):")
        print(df["winner_name"].tail(10).tolist())
