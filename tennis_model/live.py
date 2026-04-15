"""
Tennis Composite Win% Model — Live CLI
Interactive match prediction tool. Loads player_elos.json (from backtest.py)
and fetches recent stats from the Sackmann dataset.
"""

import json
import math
import io
import os
import sys
import pathlib
import datetime
import collections
import difflib
import warnings
from typing import Optional

from sizing import size_bet, DAILY_CAP_BY_ROUND
from simulation import run_simulation, sgw_to_point_prob

_tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
try:
    from tools.player_form import fetch_recent_form, format_form_line as _fmt_form, enrich_form_quality as _enrich_form_quality
    _form_available = True
except ImportError:
    _form_available = False

# Ensure UTF-8 output on Windows so box-drawing characters render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import requests
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
_THIS_YEAR = datetime.date.today().year  # always current; Sackmann updates in-season
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
TML_URL  = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

# Rolling stats cutoff: only include matches played on or before this date.
# Override this to replay predictions for any past date, e.g.:
#   CUTOFF_DATE = pd.Timestamp("2025-01-15")
CUTOFF_DATE = pd.Timestamp(datetime.date.today())

# Per-surface weights (scipy SLSQP optimized on OOS 2023-2024 test set — avoids in-sample overfitting)
SURFACE_WEIGHTS = {
    "Hard":  {"return": 0.105, "elo": 0.167, "surf_elo": 0.220, "rank": 0.115, "ace": 0.394},
    "Clay":  {"return": 0.088, "elo": 0.034, "surf_elo": 0.540, "rank": 0.300, "df_rate": 0.038},
    "Grass": {"elo": 0.195, "rank": 0.032, "ace": 0.773},
}
DEFAULT_WEIGHTS = {"return": 0.135, "elo": 0.166, "surf_elo": 0.251, "rank": 0.134, "ace": 0.313}

FORM_BLEND_WEIGHT = 0.12   # gate/display weight — form-adj shown for conflict detection, comp used for sizing
FORM_MIN_MATCHES  = 3      # use 30d window only if >= this many matches
FORM_FLOOR        = 0.10   # clip win rates to [FLOOR, 1-FLOOR] before ratio

SURFACE_ELO_PRIOR = 10  # blend surface Elo toward overall Elo (matches backtest.py)

POP_SGW      = 0.62
POP_RGW      = 0.38
POP_BP       = 0.62
POP_FORM     = 0.50
POP_ACE_RATE = 0.065
POP_SS_WON   = 0.50
POP_DF_RATE  = 0.048
ELO_START    = 1500
ELO_K        = 32

SKILL_LEN      = 100
SURF_SKILL_LEN = 40   # surface-specific window (matches backtest.py)
FORM_LEN       = 30
FORM_DECAY     = 0.9
STAT_PRIOR     = 10

EDGE_THRESHOLD = 0.04

# (lo, hi, actual_fav_win_rate, elo_bias, comp_bias) — OOS 2024+ test set (n=6,157)
ELO_DIFF_CALIB = [
    (0,   25,  0.510, +0.007, +0.005),
    (25,  50,  0.512, +0.041, +0.033),
    (50,  75,  0.580, +0.009, -0.003),
    (75,  100, 0.583, +0.040, +0.020),
    (100, 125, 0.600, +0.055, +0.027),
    (125, 150, 0.667, +0.020, -0.008),
    (150, 175, 0.653, +0.065, +0.036),
    (175, 200, 0.716, +0.030, -0.009),
    (200, 225, 0.734, +0.037, +0.002),
    (225, 250, 0.746, +0.050, +0.012),
    (250, 275, 0.789, +0.030, -0.008),
    (275, 300, 0.779, +0.061, +0.005),
    (300, 325, 0.746, +0.111, +0.072),
    (325, 350, 0.798, +0.076, +0.033),
    (350, 375, 0.872, +0.017, -0.005),
    (375, 400, 0.906, -0.003, -0.043),
    (400, 425, 0.926, -0.011, -0.045),
    (425, 450, 0.868, +0.057, +0.030),
    (450, 475, 0.905, +0.029, -0.041),
    (475, 500, 0.893, +0.050,  0.000),
    (500, 9999, 0.962, +0.001, -0.036),
]

def elo_diff_calib(elo_diff_abs: float):
    """Return (band_label, actual_rate, elo_bias, comp_bias) for the given |elo diff|."""
    for lo, hi, rate, elo_b, comp_b in ELO_DIFF_CALIB:
        if lo <= elo_diff_abs < hi:
            label = f"{lo}-{hi}" if hi < 9999 else "500+"
            return label, rate, elo_b, comp_b
    return "500+", 0.962, +0.001, -0.036


# ── Shared helpers (same as backtest.py) ─────────────────────────────────────

def elo_expected(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))


def update_elo(ratings: dict, winner: str, loser: str):
    r_w = ratings.get(winner, ELO_START)
    r_l = ratings.get(loser,  ELO_START)
    e_w = elo_expected(r_w, r_l)
    ratings[winner] = r_w + ELO_K * (1 - e_w)
    ratings[loser]  = r_l + ELO_K * (0 - (1 - e_w))


def fetch_matches_since(since: pd.Timestamp) -> pd.DataFrame:
    """Fetch all matches after `since` across all years from since.year to today."""
    frames = []
    for year in range(since.year, _THIS_YEAR + 1):
        sources = [("Sackmann", BASE_URL.format(year=year)),
                   ("TML",      TML_URL.format(year=year))]
        for _, url in sources:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
                df = df.dropna(subset=["tourney_date"])
                if len(df) > 50:
                    frames.append(df)
                    break
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values("tourney_date")
    return df[(df["tourney_date"] > since) & (df["tourney_date"] <= CUTOFF_DATE)].reset_index(drop=True)


def apply_incremental_elo(new_matches: pd.DataFrame,
                          elo_ratings: dict,
                          surf_elo_by_name: dict,
                          rank_pts_map: dict) -> int:
    """Walk new matches chronologically and update Elo/rank-points in place. Returns match count."""
    count = 0
    for _, row in new_matches.iterrows():
        wname = row.get("winner_name")
        lname = row.get("loser_name")
        if not wname or not lname or pd.isna(wname) or pd.isna(lname):
            continue
        surface = str(row.get("surface", "")).strip()
        # Rank points (use pre-match value for predictions; update after)
        for col, player in [("winner_rank_points", wname), ("loser_rank_points", lname)]:
            pts = row.get(col)
            if pts is not None and not pd.isna(pts):
                rank_pts_map[player] = float(pts)
        # Overall Elo
        update_elo(elo_ratings, wname, lname)
        # Surface Elo
        if surface in ("Hard", "Clay", "Grass"):
            surf_dict = surf_elo_by_name.setdefault(surface, {})
            # Seed from overall Elo if player unseen on this surface
            if wname not in surf_dict:
                surf_dict[wname] = elo_ratings.get(wname, ELO_START)
            if lname not in surf_dict:
                surf_dict[lname] = elo_ratings.get(lname, ELO_START)
            update_elo(surf_dict, wname, lname)
        count += 1
    return count


def compute_sgw(first_in_frac, first_won_frac, second_won_frac) -> float:
    f  = first_in_frac   if not math.isnan(first_in_frac)   else 0.60
    w1 = first_won_frac  if not math.isnan(first_won_frac)  else 0.65
    w2 = second_won_frac if not math.isnan(second_won_frac) else 0.50
    return f * w1 + (1 - f) * w2


def fatigue_multiplier(days: Optional[float]) -> float:
    if days is None or math.isnan(days):
        return 1.0
    if days <= 1:  return 0.97
    if days <= 14: return 1.0
    if days <= 30: return 0.99
    return 0.96


def bo5_adjust(p: float) -> float:
    q = 1 - p
    return p**3 * (1 + 3*q + 6*q**2)


def form_blend(comp_p1: float, form1: dict | None, form2: dict | None,
               qs1: float | None = None, qs2: float | None = None) -> float | None:
    """Nudge comp_p1 toward form ratio. Uses quality-adjusted scores when available."""
    c1_raw = (form1.get("win_rate_30d") if form1.get("n_30d", 0) >= FORM_MIN_MATCHES
              else form1.get("win_rate_60d")) if form1 else None
    c2_raw = (form2.get("win_rate_30d") if form2.get("n_30d", 0) >= FORM_MIN_MATCHES
              else form2.get("win_rate_60d")) if form2 else None
    c1 = qs1 if qs1 is not None else c1_raw
    c2 = qs2 if qs2 is not None else c2_raw
    if c1 is None or c2 is None:
        return None
    c1 = max(FORM_FLOOR, min(1 - FORM_FLOOR, c1))
    c2 = max(FORM_FLOOR, min(1 - FORM_FLOOR, c2))
    ratio = c1 / (c1 + c2)
    return max(0.001, min(0.999, (1 - FORM_BLEND_WEIGHT) * comp_p1 + FORM_BLEND_WEIGHT * ratio))


def h2h_shrink(comp_p: float, h2h_wins: int, h2h_total: int) -> float:
    if h2h_total == 0:
        return comp_p
    h2h_rate = h2h_wins / h2h_total
    blend    = min(h2h_total, 10) / 10.0 * 0.12
    return comp_p * (1 - blend) + h2h_rate * blend


# ── Rolling stats builder ─────────────────────────────────────────────────────

def _shrink(values: list, pop_avg: float) -> float:
    n = len(values)
    if n == 0:
        return pop_avg
    return (n * float(np.mean(values)) + STAT_PRIOR * pop_avg) / (n + STAT_PRIOR)


def _decayed_mean(values: list, pop_avg: float) -> float:
    n = len(values)
    if n == 0:
        return pop_avg
    weights = [FORM_DECAY ** (n - 1 - i) for i in range(n)]
    w_sum  = sum(weights)
    w_mean = sum(w * v for w, v in zip(weights, values)) / w_sum
    return (w_sum * w_mean + STAT_PRIOR * pop_avg) / (w_sum + STAT_PRIOR)


def build_player_stats(df: pd.DataFrame):
    """Build rolling stats from a DataFrame (chronological order assumed).
    Three-deque design: all-surface skill deque (SGW/BP/df_rate),
    surface-specific skill deque (SGW/ace for sim inputs), and
    surface-specific form deque (win%) with exponential decay weighting.
    """
    skill_deques:      dict = collections.defaultdict(lambda: collections.deque(maxlen=SKILL_LEN))
    surf_skill_deques: dict = collections.defaultdict(lambda: collections.deque(maxlen=SURF_SKILL_LEN))
    form_deques:       dict = collections.defaultdict(lambda: collections.deque(maxlen=FORM_LEN))
    last_date: dict = {}

    def safe(v):
        try:
            f = float(v)
            return f if not math.isnan(f) else None
        except (TypeError, ValueError):
            return None

    def update(pname, surface, date, win, row, prefix):
        svpt_v  = safe(row.get(f"{prefix}svpt"))
        f_in_v  = safe(row.get(f"{prefix}1stIn"))
        f_won_v = safe(row.get(f"{prefix}1stWon"))
        s_won_v = safe(row.get(f"{prefix}2ndWon"))
        bpsaved_v = safe(row.get(f"{prefix}bpSaved"))
        bpfaced_v = safe(row.get(f"{prefix}bpFaced"))
        ace_v     = safe(row.get(f"{prefix}ace"))
        df_v      = safe(row.get(f"{prefix}df"))

        sgw = None
        if svpt_v and svpt_v > 0 and f_in_v is not None and f_won_v is not None and s_won_v is not None:
            sgw = compute_sgw(
                f_in_v / svpt_v,
                f_won_v / f_in_v            if f_in_v > 0              else 0.65,
                s_won_v / (svpt_v - f_in_v) if svpt_v - f_in_v > 0    else 0.50,
            )

        bp = None
        if bpfaced_v and bpfaced_v > 0 and bpsaved_v is not None:
            bp = bpsaved_v / bpfaced_v

        ace_rate = (ace_v / svpt_v) if (ace_v is not None and svpt_v and svpt_v > 0) else None
        ss_second = (svpt_v - f_in_v) if (svpt_v and f_in_v is not None) else None
        ss_won = (s_won_v / ss_second) if (ss_second and ss_second > 0 and s_won_v is not None) else None
        df_rate = (df_v / svpt_v) if (df_v is not None and svpt_v and svpt_v > 0) else None

        skill_deques[pname].append({"sgw": sgw, "bp": bp, "ace_rate": ace_rate, "ss_won": ss_won, "df_rate": df_rate})
        surf_skill_deques[(pname, surface)].append({"sgw": sgw, "ace_rate": ace_rate})
        form_deques[(pname, surface)].append({"win": 1 if win else 0})
        last_date[pname] = date

    for _, row in df.iterrows():
        row = row.to_dict()
        wname = str(row.get("winner_name", "") or "")
        lname = str(row.get("loser_name",  "") or "")
        surface = str(row.get("surface", "Hard"))
        date = row.get("tourney_date")
        if not wname or not lname or date is None:
            continue
        update(wname, surface, date, True,  row, "w_")
        update(lname, surface, date, False, row, "l_")

    def get_stats(pname, surface):
        skill      = skill_deques[pname]
        surf_skill = surf_skill_deques[(pname, surface)]
        form       = list(form_deques[(pname, surface)])

        sgws      = [s["sgw"]      for s in skill if s["sgw"]           is not None]
        bps       = [s["bp"]       for s in skill if s["bp"]            is not None]
        ace_rates = [s["ace_rate"] for s in skill if s.get("ace_rate")  is not None]
        ss_wons   = [s["ss_won"]   for s in skill if s.get("ss_won")    is not None]
        df_rates  = [s["df_rate"]  for s in skill if s.get("df_rate")   is not None]
        wins      = [s["win"]      for s in form]

        all_sgw_mean = _shrink(sgws,      POP_SGW)
        all_ace_mean = _shrink(ace_rates, POP_ACE_RATE)
        surf_sgws    = [s["sgw"]      for s in surf_skill if s["sgw"]      is not None]
        surf_aces    = [s["ace_rate"] for s in surf_skill if s.get("ace_rate") is not None]

        return {
            "sgw":      all_sgw_mean,
            "bp":       _shrink(bps,       POP_BP),
            "form":     _decayed_mean(wins, POP_FORM),
            "ace_rate": all_ace_mean,
            "ss_won":   _shrink(ss_wons,   POP_SS_WON),
            "df_rate":  _shrink(df_rates,  POP_DF_RATE),
            "sgw_surf": _shrink(surf_sgws, all_sgw_mean),   # surface-specific, for simulation
            "ace_surf": _shrink(surf_aces, all_ace_mean),
        }

    return get_stats, last_date


# ── Name fuzzy matching ───────────────────────────────────────────────────────

def fuzzy_match(name: str, candidates: list[str]) -> Optional[str]:
    matches = difflib.get_close_matches(name, candidates, n=5, cutoff=0.4)
    if not matches:
        # Try case-insensitive substring
        name_l = name.lower()
        matches = [c for c in candidates if name_l in c.lower() or c.lower() in name_l]
        matches = matches[:5]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    print(f"\nMultiple matches for '{name}':")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m}")
    while True:
        choice = input("Select number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            return matches[int(choice) - 1]
        print("Invalid choice, try again.")


# ── Composite prediction ──────────────────────────────────────────────────────

def predict(p1_name, p2_name, surface, best_of, market_p1,
            elo_ratings, surf_elo_by_name, rank_pts_map, get_stats, last_date, h2h_map,
            today):
    """Return prediction dict for p1 vs p2."""
    elo1 = elo_ratings.get(p1_name, ELO_START)
    elo2 = elo_ratings.get(p2_name, ELO_START)

    surf_ratings = surf_elo_by_name.get(surface, {})
    s_elo1 = (SURFACE_ELO_PRIOR * surf_ratings.get(p1_name, ELO_START) + SURFACE_ELO_PRIOR * elo1) / (2 * SURFACE_ELO_PRIOR)
    s_elo2 = (SURFACE_ELO_PRIOR * surf_ratings.get(p2_name, ELO_START) + SURFACE_ELO_PRIOR * elo2) / (2 * SURFACE_ELO_PRIOR)
    surf_elo_p = elo_expected(s_elo1, s_elo2)

    # Rank points ratio
    rp1 = rank_pts_map.get(p1_name, 1000.0)
    rp2 = rank_pts_map.get(p2_name, 1000.0)
    rank_p = rp1 / (rp1 + rp2) if (rp1 + rp2) > 0 else 0.5

    s1 = get_stats(p1_name, surface)
    s2 = get_stats(p2_name, surface)

    elo_p = elo_expected(elo1, elo2)
    rgw1  = 1.0 - s2["sgw"]
    rgw2  = 1.0 - s1["sgw"]

    last1 = last_date.get(p1_name)
    last2 = last_date.get(p2_name)

    def days_since(ld):
        if ld is None:
            return None
        delta = today - ld
        return delta.days if hasattr(delta, "days") else None

    fat1 = fatigue_multiplier(days_since(last1))
    fat2 = fatigue_multiplier(days_since(last2))

    w = SURFACE_WEIGHTS.get(surface, DEFAULT_WEIGHTS)
    # df_rate is a negative signal: use (1 - df_rate) so higher df → lower score
    sc1 = (w.get("return",   0) * rgw1
         + w.get("elo",      0) * elo_p
         + w.get("surf_elo", 0) * surf_elo_p
         + w.get("rank",     0) * rank_p
         + w.get("ace",      0) * s1["ace_rate"]
         + w.get("df_rate",  0) * (1 - s1["df_rate"])) * fat1
    sc2 = (w.get("return",   0) * rgw2
         + w.get("elo",      0) * (1-elo_p)
         + w.get("surf_elo", 0) * (1-surf_elo_p)
         + w.get("rank",     0) * (1-rank_p)
         + w.get("ace",      0) * s2["ace_rate"]
         + w.get("df_rate",  0) * (1 - s2["df_rate"])) * fat2

    raw_p1 = sc1 / (sc1 + sc2) if (sc1 + sc2) > 0 else 0.5

    # Simulation — use surface-specific SGW for more accurate per-surface point probabilities
    p_serve_1 = (sgw_to_point_prob(s1["sgw_surf"]) + (1 - sgw_to_point_prob(s2["sgw_surf"]))) / 2
    p_serve_2 = (sgw_to_point_prob(s2["sgw_surf"]) + (1 - sgw_to_point_prob(s1["sgw_surf"]))) / 2
    sim = run_simulation(p_serve_1, p_serve_2, best_of=best_of, n_sims=10_000)

    # H2H shrinkage
    key = tuple(sorted([p1_name, p2_name]))
    h2h_entry = h2h_map.get(key, [0, 0])
    h2h_wins_p1 = h2h_entry[0] if p1_name == key[0] else h2h_entry[1] - h2h_entry[0]
    raw_p1 = h2h_shrink(raw_p1, h2h_wins_p1, h2h_entry[1])

    if best_of == 5:
        raw_p1 = bo5_adjust(raw_p1)

    raw_p1 = max(0.001, min(0.999, raw_p1))

    return {
        "comp_p1":        raw_p1,
        "elo_p1":         elo_p,
        "surf_elo_p1":    surf_elo_p,
        "rank_p1":        rank_p,
        "ace1":           s1["ace_rate"], "ace2": s2["ace_rate"],
        "rgw1": rgw1,     "rgw2": rgw2,
        "sgw1": s1["sgw"], "sgw2": s2["sgw"],
        "fat1": fat1,     "fat2": fat2,
        "elo_diff_abs":   abs(elo1 - elo2),
        "sim_p1":         sim["p_match_a"],
        "sim_set_dist":   sim["set_score_dist"],
        "sim_confidence": sim["confidence"],
        "sim_std":        sim["sim_std"],
        "sim_avg_games":  sim["avg_total_games"],
        "p_serve_1":      p_serve_1,
        "p_serve_2":      p_serve_2,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def display(p1_name, p2_name, surface, best_of, result, market_p1, sizing=None, form1=None, form2=None, form_adj_p1=None):
    W = 54
    comp1 = result["comp_p1"];  comp2 = 1 - comp1
    elo1  = result["elo_p1"];   elo2  = 1 - elo1

    lines = [
        f"╔{'═'*W}╗",
        f"║{'TENNIS WIN% PREDICTION':^{W}}║",
        f"╠{'═'*W}╣",
        f"║  {p1_name:<22} vs  {p2_name:<{W-28}}║",
        f"║  {surface} · Best of {best_of:<{W-17}}║",
        f"╠{'═'*W}╣",
        f"║  {'COMPOSITE MODEL':<22} {comp1*100:.1f}%  /  {comp2*100:.1f}%{'':<{W-40}}║",
    ]
    sim1 = result.get("sim_p1")
    if sim1 is not None:
        sim2 = 1 - sim1
        lines.append(f"║  {'Simulation (10K)':<22} {sim1*100:.1f}%  /  {sim2*100:.1f}%{'':<{W-40}}║")
    lines += [
        f"║  {'Elo only':<22} {elo1*100:.1f}%  /  {elo2*100:.1f}%{'':<{W-40}}║",
    ]
    band_label, base_rate, elo_bias, comp_bias = elo_diff_calib(result.get("elo_diff_abs", 0))
    base_str = f"Base rate ({band_label} Elo gap, OOS)  {base_rate*100:.1f}% fav wins"
    bias_str = f"  Elo bias {elo_bias*100:+.1f}%  |  Comp bias {comp_bias*100:+.1f}%"
    lines.append(f"║  {base_str:<{W-2}}║")
    lines.append(f"║{bias_str:<{W}}║")

    # Calibrated = raw - bias (applied from favourite's perspective)
    p1_is_fav = result.get("elo_p1", 0.5) >= 0.5
    sign = -1 if p1_is_fav else +1   # subtract bias from fav, add to underdog
    calib_comp1 = max(0.001, min(0.999, comp1 + sign * comp_bias))
    calib_elo1  = max(0.001, min(0.999, elo1  + sign * elo_bias))
    calib_comp2 = 1 - calib_comp1
    calib_elo2  = 1 - calib_elo1
    lines.append(f"║  {'Calibrated comp':<22} {calib_comp1*100:.1f}%  /  {calib_comp2*100:.1f}%{'':<{W-40}}║")
    lines.append(f"║  {'Calibrated Elo':<22} {calib_elo1*100:.1f}%  /  {calib_elo2*100:.1f}%{'':<{W-40}}║")
    if form_adj_p1 is not None:
        fadj2 = 1 - form_adj_p1
        delta = (form_adj_p1 - comp1) * 100
        lines.append(f"║  {'Form-adj comp':<22} {form_adj_p1*100:.1f}%  /  {fadj2*100:.1f}%  ({delta:+.1f}pp){'':<{W-50}}║")

    if market_p1 is not None:
        market2 = 1 - market_p1
        lines.append(f"║  {'Market implied':<22} {market_p1*100:.1f}%  /  {market2*100:.1f}%{'':<{W-40}}║")

    lines.append(f"╠{'═'*W}╣")

    if market_p1 is not None:
        edge = comp1 - market_p1
        fav  = p1_name if edge >= 0 else p2_name
        edge_str = f"{edge*100:+.1f}% on {fav}"
        if edge > EDGE_THRESHOLD:
            signal = f"✅ VALUE — model > market"
        elif edge < -EDGE_THRESHOLD:
            signal = f"🔁 VALUE — model favors opponent"
        else:
            signal = f"➖ Model and market roughly agree"
        lines.append(f"║  {'Edge vs market':<22} {edge_str:<{W-26}}║")
        lines.append(f"║  {'Signal':<22} {signal:<{W-26}}║")
        lines.append(f"╠{'═'*W}╣")

    if sizing is not None:
        sig = sizing["signal"]
        if sig == "BET":
            sig_str = f"BET  ${sizing['stake']:.2f}  ({sizing['stake_pct']:.1%} bankroll)"
        elif sig == "CAP_BOUND":
            sig_str = "CAP BOUND — daily limit reached"
        else:
            sig_str = f"NO EDGE  ({sizing['edge']*100:+.1f}pp)"
        lines.append(f"║  {'Sizing signal':<22} {sig_str:<{W-26}}║")
        if sig == "BET":
            cap_str = f"edge cap {sizing['edge_cap']:.1%}  |  headroom {sizing['remaining_capacity']:.1%}"
            lines.append(f"║  {'':22} {cap_str:<{W-26}}║")
        conf_sc = sizing.get("conf_scalar", 1.0)
        sim_conf = sizing.get("sim_confidence")
        if sim_conf is not None:
            conf_str = f"sim conf {sim_conf:.2f}  →  kelly scalar {conf_sc:.2f}"
            lines.append(f"║  {'Confidence':<22} {conf_str:<{W-26}}║")
        lines.append(f"╠{'═'*W}╣")

    comp_label = f"  Component breakdown:"
    lines.append(f"║{comp_label:<{W}}║")

    def comp_line(label, v1, v2, note=""):
        s = f"║    {label:<18} {v1*100:.1f}%  vs  {v2*100:.1f}%  {note}"
        return s[:W+2].ljust(W+2) + "║"

    lines.append(comp_line("Overall Elo",    result["elo_p1"],      1-result["elo_p1"]))
    lines.append(comp_line("Surface Elo",   result["surf_elo_p1"], 1-result["surf_elo_p1"]))
    lines.append(comp_line("Rank points",   result["rank_p1"],     1-result["rank_p1"]))
    lines.append(comp_line("Return quality",result["rgw1"],        result["rgw2"]))
    lines.append(comp_line("Ace rate",      result["ace1"],        result["ace2"]))

    fat_line = f"║    {'Fatigue mult':<18} {result['fat1']:.2f}× vs   {result['fat2']:.2f}×"
    lines.append(fat_line[:W+2].ljust(W+2) + "║")

    if form1 is not None or form2 is not None:
        lines.append(f"╠{'═'*W}╣")
        lines.append(f"║{'  RECENT FORM':<{W}}║")
        lines.append(_fmt_form(p1_name, form1))
        lines.append(_fmt_form(p2_name, form2))

    lines.append(f"╚{'═'*W}╝")

    print()
    for line in lines:
        print(line)
    print()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_recent_data():
    """Load the 2 most recent available years for rolling stats.
    Tries Sackmann first, falls back to TML (same columns, covers 2025+).
    Applies CUTOFF_DATE filter so only matches already played are included.
    """
    frames = []
    for year in range(_THIS_YEAR, _THIS_YEAR - 3, -1):
        sources = [("Sackmann", BASE_URL.format(year=year)),
                   ("TML",      TML_URL.format(year=year))]
        for label, url in sources:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
                df = df.dropna(subset=["tourney_date"])
                if len(df) > 50:
                    print(f"  Loaded {year} [{label}]: {len(df):,} matches")
                    frames.append(df)
                    break  # got this year, move to next
            except Exception:
                continue
        if len(frames) == 2:
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    df = df[df["tourney_date"] <= CUTOFF_DATE]
    print(f"  Using matches up to {CUTOFF_DATE.date()} ({len(df):,} total)")

    # Supplement with Sofascore cache (covers matches after TML/Sackmann cutoff)
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_stats_cache.csv")
    if os.path.exists(cache_path):
        try:
            sc = pd.read_csv(cache_path, parse_dates=["tourney_date"])
            sc = sc[sc["tourney_date"] <= CUTOFF_DATE]
            if not sc.empty:
                tml_max = df["tourney_date"].max() if not df.empty else pd.Timestamp("2000-01-01")
                sc_new = sc[sc["tourney_date"] > tml_max]
                if not sc_new.empty:
                    df = pd.concat([df, sc_new], ignore_index=True).sort_values("tourney_date").reset_index(drop=True)
                    print(f"  + Sofascore cache: {len(sc_new):,} recent matches appended")
        except Exception:
            pass

    return df


# ── Session state (bankroll + exposure tracking) ──────────────────────────────

_SESSION_DIR = pathlib.Path(".tmp")
_VALID_ROUNDS = ("R128", "R64", "R32", "R16", "QF", "SF", "F")


def _session_path() -> pathlib.Path:
    return _SESSION_DIR / f"session_{datetime.date.today()}.json"


def _load_session() -> dict | None:
    p = _session_path()
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_session(data: dict):
    _SESSION_DIR.mkdir(exist_ok=True)
    with open(_session_path(), "w") as f:
        json.dump(data, f, indent=2)


# ── Main CLI ──────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════╗")
    print("║   Tennis Composite Win% Predictor    ║")
    print("╚══════════════════════════════════════╝\n")

    # Load Elo ratings (supports both old flat format and new {"overall":…,"surface":…})
    elo_ratings      = {}
    surf_elo_by_name = {}
    try:
        with open("player_elos.json") as f:
            raw = json.load(f)
        if "overall" in raw:
            elo_ratings      = raw["overall"]
            surf_elo_by_name = raw.get("surface", {})
            rank_pts_map     = raw.get("rank_points", {})
        else:
            elo_ratings  = raw  # backwards-compat with old flat format
            rank_pts_map = {}
        last_updated = raw.get("last_updated")
        print(f"Loaded Elo ratings for {len(elo_ratings):,} players"
              + (f" + surface Elo ({', '.join(surf_elo_by_name.keys())})" if surf_elo_by_name else "")
              + (f" + rank points ({len(rank_pts_map):,})" if rank_pts_map else "")
              + (f"  [last updated: {last_updated}]" if last_updated else "") + ".")
        # Incremental Elo update: fetch only matches since last backtest run
        if last_updated:
            since = pd.Timestamp(last_updated)
            if since.date() < datetime.date.today():
                print(f"Fetching new matches since {last_updated} for Elo update...")
                new_df = fetch_matches_since(since)
                if not new_df.empty:
                    n = apply_incremental_elo(new_df, elo_ratings, surf_elo_by_name, rank_pts_map)
                    print(f"  Applied {n} new matches to Elo ratings.")
                else:
                    print("  No new matches found.")
    except FileNotFoundError:
        last_updated = None
        rank_pts_map = {}
        print("WARNING: player_elos.json not found. Run backtest.py first for accurate Elos.")
        print("Continuing with default Elo 1500 for all players.\n")

    # Load recent data for rolling stats
    print("Fetching recent match data for rolling stats...")
    df = load_recent_data()

    # Build player name list and H2H map
    known_players = []
    h2h_map = {}  # (name1, name2) sorted tuple -> [name1_wins, total]

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

        get_stats, last_date = build_player_stats(df)
    else:
        get_stats = lambda pname, surf: {"sgw": POP_SGW, "bp": POP_BP, "form": POP_FORM,
                                         "ace_rate": POP_ACE_RATE, "ss_won": POP_SS_WON}
        last_date = {}

    # Supplement known_players with names from elo_ratings
    for name in elo_ratings:
        if name not in known_players:
            known_players.append(name)
    known_players = list(dict.fromkeys(known_players))  # deduplicate, preserve order

    today = pd.Timestamp.now()

    # ── Session setup ──────────────────────────────────────────────────────────
    session = _load_session()
    if session:
        bankroll         = session["bankroll"]
        current_exposure = session["current_exposure"]
        print(f"\nResuming today's session — bankroll ${bankroll:,.2f}, "
              f"exposure {current_exposure:.1%} ({len(session['bets'])} bet(s) placed).")
    else:
        print()
        br_raw = input("Bankroll for today's session (Kalshi cash + EV of open positions, $): ").strip()
        try:
            bankroll = float(br_raw)
            if bankroll <= 0:
                raise ValueError
        except ValueError:
            bankroll = 0.0
            print("  Invalid bankroll — sizing will be skipped.")
        current_exposure = 0.0
        session = {
            "date": str(datetime.date.today()),
            "bankroll": bankroll,
            "bets": [],
            "current_exposure": 0.0,
        }
        _save_session(session)

    while True:
        print("─" * 56)
        p1_raw = input("Player 1 name: ").strip()
        p2_raw = input("Player 2 name: ").strip()

        p1_name = fuzzy_match(p1_raw, known_players)
        if p1_name is None:
            print(f"  No match found for '{p1_raw}'. Using as-is.")
            p1_name = p1_raw

        p2_name = fuzzy_match(p2_raw, known_players)
        if p2_name is None:
            print(f"  No match found for '{p2_raw}'. Using as-is.")
            p2_name = p2_raw

        surface_raw = input("Surface [Hard/Clay/Grass]: ").strip().capitalize()
        if surface_raw not in ("Hard", "Clay", "Grass"):
            surface_raw = "Hard"
            print("  Unrecognized surface, defaulting to Hard.")

        fmt_raw = input("Format [3/5]: ").strip()
        best_of = 5 if fmt_raw == "5" else 3

        market_raw = input("Market implied prob for P1 (0–1, or Enter to skip): ").strip()
        market_p1 = None
        if market_raw:
            try:
                market_p1 = float(market_raw)
                if not (0 < market_p1 < 1):
                    raise ValueError
            except ValueError:
                print("  Invalid market probability, skipping.")
                market_p1 = None

        round_raw = input(f"Round [{'/'.join(_VALID_ROUNDS)}]: ").strip().upper()
        if round_raw not in _VALID_ROUNDS:
            round_raw = "R32"
            print("  Unrecognized round, defaulting to R32.")

        result = predict(
            p1_name, p2_name, surface_raw, best_of, market_p1,
            elo_ratings, surf_elo_by_name, rank_pts_map, get_stats, last_date, h2h_map, today
        )

        sizing = None
        if bankroll > 0 and market_p1 is not None:
            sizing = size_bet(
                p_model=result["comp_p1"],
                p_market=market_p1,
                bankroll=bankroll,
                current_exposure=current_exposure,
                round_stage=round_raw,
                sim_confidence=result.get("sim_confidence"),
                surface=surface_raw,
            )

        form1 = form2 = None
        if _form_available:
            form1 = fetch_recent_form(p1_name)
            form2 = fetch_recent_form(p2_name)

        qs1 = qs2 = None
        if _form_available:
            qs1 = _enrich_form_quality(form1, elo_ratings.get(p1_name, 1500), elo_ratings)
            qs2 = _enrich_form_quality(form2, elo_ratings.get(p2_name, 1500), elo_ratings)

        form_adj = form_blend(result["comp_p1"], form1, form2, qs1=qs1, qs2=qs2)
        display(p1_name, p2_name, surface_raw, best_of, result, market_p1, sizing, form1, form2, form_adj)

        if sizing and sizing["signal"] == "BET":
            confirm = input(f"  Place this bet (${sizing['stake']:.2f} on {p1_name})? [y/n]: ").strip().lower()
            if confirm == "y":
                current_exposure += sizing["stake_pct"]
                session["current_exposure"] = current_exposure
                session["bets"].append({
                    "match": f"{p1_name} vs {p2_name}",
                    "round": round_raw,
                    "stake": sizing["stake"],
                    "stake_pct": sizing["stake_pct"],
                    "edge": round(sizing["edge"], 4),
                    "placed": True,
                })
                _save_session(session)
                print(f"  Logged. Cumulative exposure: {current_exposure:.1%} "
                      f"/ {sizing['daily_cap']:.0%} daily cap "
                      f"({sizing['remaining_capacity'] - sizing['stake_pct']:.1%} remaining).")

        again = input("Run another match? [y/n]: ").strip().lower()
        if again != "y":
            break

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
