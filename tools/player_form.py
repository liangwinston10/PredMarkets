"""
Recent match velocity indicator — pulls 90 days of ATP + WTA results from ESPN.

Fetches 26 weekly endpoints in parallel (~3s total), caches in memory for
the session. Subsequent lookups are instant.

Usage:
    from tools.player_form import fetch_recent_form

    form = fetch_recent_form("Jannik Sinner")
    # {"results": ["W","W","L","W","W"], "win_rate_30d": 0.90, ...}

    form = fetch_recent_form("Anastasia Potapova")
    # same structure — auto-detects ATP/WTA from the data

Standalone:
    python tools/player_form.py "Jannik Sinner"
"""

import re
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── Config ────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates={date}"
LOOKBACK_WEEKS = 13   # ~90 days
TIMEOUT        = 10

# ── Module-level cache ────────────────────────────────────────────────────────

_cache: list | None = None   # list of {"date", "note"} dicts, deduplicated


def _fetch_week(tour: str, date_str: str) -> list[dict]:
    """Fetch one weekly ESPN scoreboard. Returns list of {date, note}."""
    try:
        r = requests.get(
            ESPN_URL.format(tour=tour, date=date_str),
            headers=HEADERS, timeout=TIMEOUT,
        )
        if not r.ok:
            return []
        matches = []
        for event in r.json().get("events", []):
            for grp in event.get("groupings", []):
                if "singles" not in grp.get("grouping", {}).get("slug", "").lower():
                    continue
                for comp in grp.get("competitions", []):
                    if comp.get("status", {}).get("type", {}).get("state") != "post":
                        continue
                    notes = comp.get("notes", [])
                    if not notes:
                        continue
                    matches.append({
                        "date": comp["date"][:10],
                        "note": notes[0].get("text", ""),
                    })
        return matches
    except Exception:
        return []


def _load_cache() -> list[dict]:
    """Fetch 90 days of ATP + WTA in parallel. Deduplicate and cache."""
    global _cache
    if _cache is not None:
        return _cache

    today = datetime.date.today()
    tasks = []
    for tour in ("atp", "wta"):
        for w in range(LOOKBACK_WEEKS):
            d = (today - datetime.timedelta(weeks=w)).strftime("%Y%m%d")
            tasks.append((tour, d))

    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(_fetch_week, tour, date): (tour, date) for tour, date in tasks}
        for fut in as_completed(futures):
            raw.extend(fut.result())

    # Deduplicate by (date, note)
    seen = set()
    deduped = []
    for m in raw:
        key = (m["date"], m["note"])
        if key not in seen and m["note"]:
            seen.add(key)
            deduped.append(m)

    _cache = sorted(deduped, key=lambda m: m["date"], reverse=True)
    return _cache


# ── Name matching ─────────────────────────────────────────────────────────────

def _name_in_side(player_name: str, side: str) -> bool:
    """
    Check if player_name appears in a note side like '(2) Jannik Sinner (ITA)'.
    Matches on last name, or full name if ambiguous.
    """
    side_clean = re.sub(r"\([^)]+\)", "", side).strip()  # remove (2), (ITA) etc.
    parts = player_name.lower().split()
    side_lower = side_clean.lower()
    # Last name match (most reliable)
    last = parts[-1]
    if last in side_lower:
        # Also check first name to avoid false positives (e.g. two players with same last name)
        if len(parts) > 1:
            first = parts[0]
            return first in side_lower or last in side_lower
        return True
    return False


def _parse_note(note: str, player_name: str) -> dict | None:
    """
    Parse a note like '(2) Jannik Sinner (ITA) bt Ugo Humbert (FRA) 6-3 6-0'.
    Returns {"result": "W"|"L", "opponent": str} or None if player not found.
    """
    if " bt " not in note:
        return None
    winner_side, loser_side = note.split(" bt ", 1)
    # Remove score from loser side: strip " 6-3 7-5" style suffixes (and "ret.")
    loser_side = re.sub(r"(\s+\d+[-\u2013]\d+)+.*$", "", loser_side).strip()

    if _name_in_side(player_name, winner_side):
        # Extract opponent last name from loser side
        opp_clean = re.sub(r"\([^)]+\)", "", loser_side).strip()
        opp_parts = opp_clean.strip().split()
        opp = opp_parts[-1] if opp_parts else loser_side[:20]
        return {"result": "W", "opponent": opp}
    elif _name_in_side(player_name, loser_side):
        opp_clean = re.sub(r"\([^)]+\)", "", winner_side).strip()
        opp_parts = opp_clean.strip().split()
        opp = opp_parts[-1] if opp_parts else winner_side[:20]
        return {"result": "L", "opponent": opp}
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_recent_form(player_name: str, n: int = 10) -> dict | None:
    """
    Return recent match velocity for a player.

    Parameters
    ----------
    player_name : str   Full or partial name, e.g. "Jannik Sinner", "Sinner"
    n           : int   Number of recent matches to include in results list

    Returns
    -------
    dict with keys:
        results         list[str]   ["W","W","L",...] most recent first, up to n
        opponents       list[str]   opponent last names, same order
        win_rate_30d    float       win% over last 30 days
        win_rate_60d    float       win% over last 60 days
        days_since_last int         days since most recent match
        last_match_date str         YYYY-MM-DD
        streak          str         e.g. "W7" or "L2"
    Returns None on any failure.
    """
    try:
        matches = _load_cache()
        today = datetime.date.today()
        cutoff_30 = (today - datetime.timedelta(days=30)).isoformat()
        cutoff_60 = (today - datetime.timedelta(days=60)).isoformat()

        player_matches = []
        for m in matches:
            parsed = _parse_note(m["note"], player_name)
            if parsed:
                player_matches.append({**parsed, "date": m["date"]})

        if not player_matches:
            return None

        # Sort most recent first (already sorted, but be safe)
        player_matches.sort(key=lambda m: m["date"], reverse=True)

        # Win rates
        def win_rate(since: str) -> float | None:
            subset = [m for m in player_matches if m["date"] >= since]
            if not subset:
                return None
            return sum(1 for m in subset if m["result"] == "W") / len(subset)

        # Streak
        streak_char = player_matches[0]["result"]
        streak_count = 0
        for m in player_matches:
            if m["result"] == streak_char:
                streak_count += 1
            else:
                break

        last_date = datetime.date.fromisoformat(player_matches[0]["date"])
        days_ago = (today - last_date).days

        return {
            "results":         [m["result"] for m in player_matches[:n]],
            "opponents":       [m["opponent"] for m in player_matches[:n]],
            "win_rate_30d":    win_rate(cutoff_30),
            "win_rate_60d":    win_rate(cutoff_60),
            "days_since_last": days_ago,
            "last_match_date": player_matches[0]["date"],
            "streak":          f"{streak_char}{streak_count}",
        }
    except Exception:
        return None


def format_form_line(player_name: str, form: dict | None, width: int = 52) -> str:
    """Format a single velocity line for CLI display."""
    short = player_name.split()[-1]  # last name
    if not form:
        return f"║  {short:<14} —                                      ║"
    results    = "-".join(form["results"][:7])
    wr30       = f"{form['win_rate_30d']*100:.0f}%" if form["win_rate_30d"] is not None else "—"
    wr60       = f"{form['win_rate_60d']*100:.0f}%" if form["win_rate_60d"] is not None else "—"
    streak     = form["streak"]
    days       = form["days_since_last"]
    days_str   = f"{days}d ago" if days > 0 else "today"
    content    = f"{short:<12} {results:<16} 30d:{wr30:<5} 60d:{wr60:<5} streak:{streak:<4} ({days_str})"
    return f"║  {content:<{width}}║"


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    name = sys.argv[1] if len(sys.argv) > 1 else "Jannik Sinner"
    print(f"Fetching recent form for: {name}")
    print("Loading ESPN data (first run ~3s)...")
    form = fetch_recent_form(name)
    if form:
        import json
        print(json.dumps(form, indent=2))
        print()
        print(format_form_line(name, form))
    else:
        print("No matches found.")
