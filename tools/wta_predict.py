"""
WTA Match Prediction Wrapper
Programmatic interface to tennis_model/wta/wta_live.py for use by the dashboard.
Does NOT modify wta_live.py — imports its functions directly.
WTA is always best-of-3; the best_of parameter is absent from wta_live.predict().
"""

import os
import sys
import json
import difflib
import datetime

import pandas as pd

# Resolve path to tennis_model/wta/ relative to this file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WTA_DIR   = os.path.join(REPO_ROOT, "tennis_model", "wta")
sys.path.insert(0, WTA_DIR)

import wta_live as wta  # noqa: E402

ELO_PATH = os.path.join(WTA_DIR, "player_elos.json")


def load_engine(elo_path: str = ELO_PATH) -> dict:
    """Load WTA player_elos.json, apply incremental Elo update, return engine dict."""
    elo_ratings: dict = {}
    surf_elo_by_name: dict = {}
    rank_pts_map: dict = {}
    last_updated = None

    try:
        with open(elo_path) as f:
            raw = json.load(f)
        if "overall" in raw:
            elo_ratings      = raw["overall"]
            surf_elo_by_name = raw.get("surface", {})
            rank_pts_map     = raw.get("rank_points", {})
        else:
            elo_ratings = raw
        last_updated = raw.get("last_updated")
    except FileNotFoundError:
        pass

    if last_updated:
        since = pd.Timestamp(last_updated)
        if since.date() < datetime.date.today():
            try:
                new_df = wta.fetch_matches_since(since)
                if not new_df.empty:
                    wta.apply_incremental_elo(new_df, elo_ratings, surf_elo_by_name, rank_pts_map)
            except Exception:
                pass

    return {
        "elo_ratings":      elo_ratings,
        "surf_elo_by_name": surf_elo_by_name,
        "rank_pts_map":     rank_pts_map,
        "last_updated":     last_updated,
    }


def load_stats(engine: dict) -> dict:
    """Fetch recent WTA match data and build rolling player stats."""
    df = wta.load_recent_data()

    known_players: list = []
    h2h_map: dict = {}

    if not df.empty:
        for _, row in df.iterrows():
            wn = str(row.get("winner_name", "") or "")
            ln = str(row.get("loser_name",  "") or "")
            if wn and wn not in known_players:
                known_players.append(wn)
            if ln and ln not in known_players:
                known_players.append(ln)
            if wn and ln:
                key = tuple(sorted([wn, ln]))
                if key not in h2h_map:
                    h2h_map[key] = [0, 0]
                h2h_map[key][1] += 1
                if wn == key[0]:
                    h2h_map[key][0] += 1
        get_stats, last_date = wta.build_player_stats(df)
    else:
        def get_stats(p, s):  # noqa: ANN001
            return {
                "sgw":      wta.POP_SGW,
                "bp":       wta.POP_BP,
                "form":     wta.POP_FORM,
                "ace_rate": wta.POP_ACE_RATE,
                "ss_won":   wta.POP_SS_WON,
            }
        last_date = {}

    for name in engine["elo_ratings"]:
        if name not in known_players:
            known_players.append(name)

    return {
        "known_players": known_players,
        "h2h_map":       h2h_map,
        "get_stats":     get_stats,
        "last_date":     last_date,
    }


def run_prediction(
    p1_name: str,
    p2_name: str,
    surface: str,
    market_p1: float | None,
    engine: dict,
    stats: dict,
) -> dict:
    """
    Run the WTA prediction model (always Bo3 — no best_of param).
    Returns dict with: comp_p1, elo_p1, surf_elo_p1, rank_p1,
                       ace1, ace2, rgw1, rgw2, sgw1, sgw2, fat1, fat2, elo_diff_abs
    """
    return wta.predict(
        p1_name, p2_name, surface, market_p1,
        engine["elo_ratings"],
        engine["surf_elo_by_name"],
        engine["rank_pts_map"],
        stats["get_stats"],
        stats["last_date"],
        stats["h2h_map"],
        pd.Timestamp.now(),
    )


def search_players(partial: str, known_players: list, n: int = 15) -> list:
    """Non-interactive fuzzy WTA player search."""
    if not partial:
        return known_players[:n]
    matches = difflib.get_close_matches(partial, known_players, n=n, cutoff=0.35)
    if not matches:
        pl = partial.lower()
        matches = [c for c in known_players if pl in c.lower()][:n]
    return matches


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading WTA engine...")
    eng = load_engine()
    print(f"  {len(eng['elo_ratings']):,} players in Elo file")
    print("Loading rolling stats...")
    sts = load_stats(eng)
    print(f"  {len(sts['known_players']):,} known players")

    p1, p2 = "Aryna Sabalenka", "Iga Swiatek"
    print(f"\nTest prediction: {p1} vs {p2} | Hard | Bo3")
    result = run_prediction(p1, p2, "Hard", None, eng, sts)
    p1_pct = result["comp_p1"] * 100
    p2_pct = (1 - result["comp_p1"]) * 100
    print(f"  {p1}: {p1_pct:.1f}%")
    print(f"  {p2}: {p2_pct:.1f}%")
    print(f"  Elo diff: {result['elo_diff_abs']:.0f}")
