"""
ATP Match Prediction Wrapper
Programmatic interface to tennis_model/live.py for use by the dashboard.
Does NOT modify live.py — imports its functions directly.
"""

import os
import sys
import json
import difflib
import datetime

import pandas as pd

# Resolve path to tennis_model/ relative to this file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "tennis_model")
sys.path.insert(0, MODEL_DIR)

import live  # noqa: E402 — side-effect free at import; only main() uses stdin

ELO_PATH = os.path.join(MODEL_DIR, "player_elos.json")


def load_engine(elo_path: str = ELO_PATH) -> dict:
    """Load player_elos.json, apply incremental Elo update, return engine dict."""
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
            elo_ratings = raw  # backwards-compat with flat format
        last_updated = raw.get("last_updated")
    except FileNotFoundError:
        pass  # Engine still works with default ELO_START

    # Incremental Elo update for matches since last backtest
    if last_updated:
        since = pd.Timestamp(last_updated)
        if since.date() < datetime.date.today():
            try:
                new_df = live.fetch_matches_since(since)
                if not new_df.empty:
                    live.apply_incremental_elo(new_df, elo_ratings, surf_elo_by_name, rank_pts_map)
            except Exception:
                pass  # Non-fatal — predictions still work with existing Elos

    return {
        "elo_ratings":      elo_ratings,
        "surf_elo_by_name": surf_elo_by_name,
        "rank_pts_map":     rank_pts_map,
        "last_updated":     last_updated,
    }


def load_stats(engine: dict) -> dict:
    """Fetch recent match data and build rolling player stats."""
    df = live.load_recent_data()

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
        get_stats, last_date = live.build_player_stats(df)
    else:
        def get_stats(p, s):  # noqa: ANN001
            return {
                "sgw":      live.POP_SGW,
                "bp":       live.POP_BP,
                "form":     live.POP_FORM,
                "ace_rate": live.POP_ACE_RATE,
                "ss_won":   live.POP_SS_WON,
                "df_rate":  live.POP_DF_RATE,
                "sgw_surf": live.POP_SGW,
                "ace_surf": live.POP_ACE_RATE,
            }
        last_date = {}

    # Add players from Elo file who may not appear in recent matches
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
    best_of: int,
    market_p1: float | None,
    engine: dict,
    stats: dict,
) -> dict:
    """
    Run the ATP prediction model.
    Returns dict with: comp_p1, elo_p1, surf_elo_p1, rank_p1,
                       ace1, ace2, rgw1, rgw2, sgw1, sgw2, fat1, fat2, elo_diff_abs
    """
    return live.predict(
        p1_name, p2_name, surface, best_of, market_p1,
        engine["elo_ratings"],
        engine["surf_elo_by_name"],
        engine["rank_pts_map"],
        stats["get_stats"],
        stats["last_date"],
        stats["h2h_map"],
        pd.Timestamp.now(),
    )


def search_players(partial: str, known_players: list, n: int = 15) -> list:
    """Non-interactive fuzzy player search — returns ranked candidate list."""
    if not partial:
        return known_players[:n]
    matches = difflib.get_close_matches(partial, known_players, n=n, cutoff=0.35)
    if not matches:
        pl = partial.lower()
        matches = [c for c in known_players if pl in c.lower()][:n]
    return matches


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading ATP engine...")
    eng = load_engine()
    print(f"  {len(eng['elo_ratings']):,} players in Elo file")
    print("Loading rolling stats...")
    sts = load_stats(eng)
    print(f"  {len(sts['known_players']):,} known players")

    p1, p2 = "Jannik Sinner", "Carlos Alcaraz"
    print(f"\nTest prediction: {p1} vs {p2} | Hard | Bo3")
    result = run_prediction(p1, p2, "Hard", 3, None, eng, sts)
    p1_pct = result["comp_p1"] * 100
    p2_pct = (1 - result["comp_p1"]) * 100
    print(f"  {p1}: {p1_pct:.1f}%")
    print(f"  {p2}: {p2_pct:.1f}%")
    print(f"  Elo diff: {result['elo_diff_abs']:.0f}")
