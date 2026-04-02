"""
Tennis Composite Win% Model — Backtest (ATP Challenger Circuit)
Runs composite model against historical Challenger data (Jeff Sackmann dataset).
Outputs accuracy, Brier score, calibration table, results CSV, and player_elos.json.
"""

import json
import math
import io
import datetime
import collections
import warnings
from typing import Optional

import requests
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
YEARS    = list(range(2005, datetime.date.today().year + 1))
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_{year}.csv"

# Per-surface weights (scipy SLSQP optimized, 2005-2024 Challenger data)
SURFACE_WEIGHTS = {
    "Hard":  {"elo": 0.473, "surf_elo": 0.119, "rank": 0.022, "ace": 0.386},
    "Clay":  {"elo": 0.518, "surf_elo": 0.415, "rank": 0.039, "ss_won": 0.028},
    "Grass": {"elo": 0.359, "ace": 0.641},
}
DEFAULT_WEIGHTS = {"elo": 0.539, "surf_elo": 0.243, "rank": 0.030, "ace": 0.185}

# Population fallback averages (Challenger — similar to ATP main tour)
POP_SGW      = 0.62
POP_RGW      = 0.38
POP_BP       = 0.62
POP_FORM     = 0.50
POP_ACE_RATE = 0.060
POP_SS_WON   = 0.50

# Elo settings
ELO_K      = 32
ELO_START  = 1500

# Rolling window sizes — reduced vs ATP (sparser Challenger histories)
SKILL_LEN = 60
FORM_LEN  = 20

FORM_DECAY = 0.9
STAT_PRIOR = 10
SURFACE_ELO_PRIOR = 10

# ── Elo helpers ───────────────────────────────────────────────────────────────

def elo_expected(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))


def update_elo(ratings: dict, winner: str, loser: str):
    r_w = ratings.get(winner, ELO_START)
    r_l = ratings.get(loser, ELO_START)
    e_w = elo_expected(r_w, r_l)
    ratings[winner] = r_w + ELO_K * (1 - e_w)
    ratings[loser]  = r_l + ELO_K * (0 - (1 - e_w))


# ── Stat helpers ──────────────────────────────────────────────────────────────

def compute_sgw(first_in_pct, first_won_pct, second_won_pct) -> float:
    f  = first_in_pct  if not math.isnan(first_in_pct)  else 0.6
    w1 = first_won_pct if not math.isnan(first_won_pct) else 0.65
    w2 = second_won_pct if not math.isnan(second_won_pct) else 0.50
    return f * w1 + (1 - f) * w2


def fatigue_multiplier(days: Optional[float]) -> float:
    if days is None or math.isnan(days):
        return 1.0
    if days <= 1:
        return 0.97
    if days <= 14:
        return 1.0
    if days <= 30:
        return 0.99
    return 0.96


def bo5_adjust(p: float) -> float:
    q = 1 - p
    return p**3 * (1 + 3*q + 6*q**2)


def h2h_shrink(comp_p: float, h2h_wins: int, h2h_total: int) -> float:
    if h2h_total == 0:
        return comp_p
    h2h_rate = h2h_wins / h2h_total
    blend    = min(h2h_total, 10) / 10.0 * 0.12
    return comp_p * (1 - blend) + h2h_rate * blend


# ── Rolling stats store ───────────────────────────────────────────────────────

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
    w_sum   = sum(weights)
    w_mean  = sum(w * v for w, v in zip(weights, values)) / w_sum
    return (w_sum * w_mean + STAT_PRIOR * pop_avg) / (w_sum + STAT_PRIOR)


class PlayerStats:
    def __init__(self):
        self.skill_deques: dict = collections.defaultdict(
            lambda: collections.deque(maxlen=SKILL_LEN)
        )
        self.form_deques: dict = collections.defaultdict(
            lambda: collections.deque(maxlen=FORM_LEN)
        )
        self.last_date: dict = {}

    def get_stats(self, player_name: str, surface: str) -> dict:
        skill = self.skill_deques[player_name]
        form  = list(self.form_deques[(player_name, surface)])

        sgws     = [s["sgw"]      for s in skill if s["sgw"]      is not None]
        bps      = [s["bp"]       for s in skill if s["bp"]       is not None]
        ace_rates= [s["ace_rate"] for s in skill if s.get("ace_rate") is not None]
        ss_wons  = [s["ss_won"]   for s in skill if s.get("ss_won")   is not None]
        wins     = [s["win"]      for s in form]

        return {
            "sgw":      _shrink(sgws,      POP_SGW),
            "bp":       _shrink(bps,       POP_BP),
            "form":     _decayed_mean(wins, POP_FORM),
            "ace_rate": _shrink(ace_rates, POP_ACE_RATE),
            "ss_won":   _shrink(ss_wons,   POP_SS_WON),
        }

    def update(self, player_name: str, surface: str, match_date, win: bool, row, prefix: str):
        def safe(v):
            try:
                f = float(v)
                return f if not math.isnan(f) else None
            except (TypeError, ValueError):
                return None

        svpt_v  = safe(row.get(f"{prefix}svpt"))
        f_in_v  = safe(row.get(f"{prefix}1stIn"))
        f_won_v = safe(row.get(f"{prefix}1stWon"))
        s_won_v = safe(row.get(f"{prefix}2ndWon"))
        bpsaved_v = safe(row.get(f"{prefix}bpSaved"))
        bpfaced_v = safe(row.get(f"{prefix}bpFaced"))
        ace_v   = safe(row.get(f"{prefix}ace"))

        sgw = None
        if svpt_v and svpt_v > 0 and f_in_v is not None and f_won_v is not None and s_won_v is not None:
            sgw = compute_sgw(
                f_in_v / svpt_v,
                f_won_v / f_in_v        if f_in_v > 0              else 0.65,
                s_won_v / (svpt_v - f_in_v) if svpt_v - f_in_v > 0 else 0.50,
            )

        bp = None
        if bpfaced_v and bpfaced_v > 0 and bpsaved_v is not None:
            bp = bpsaved_v / bpfaced_v

        ace_rate = (ace_v / svpt_v) if (ace_v is not None and svpt_v and svpt_v > 0) else None

        ss_second = (svpt_v - f_in_v) if (svpt_v and f_in_v is not None) else None
        ss_won = (s_won_v / ss_second) if (ss_second and ss_second > 0 and s_won_v is not None) else None

        self.skill_deques[player_name].append({"sgw": sgw, "bp": bp, "ace_rate": ace_rate, "ss_won": ss_won})
        self.form_deques[(player_name, surface)].append({"win": 1 if win else 0})
        self.last_date[player_name] = match_date


# ── Composite score ───────────────────────────────────────────────────────────

def composite_win_prob(p1_stats, p2_stats, elo_p1, elo_p2, surf_elo_p, rank_p, surface, best_of,
                       p1_last_date, p2_last_date, match_date,
                       h2h_p1_wins, h2h_total):
    sgw1 = p1_stats["sgw"];  sgw2 = p2_stats["sgw"]
    rgw1 = 1.0 - sgw2;       rgw2 = 1.0 - sgw1
    bp1  = p1_stats["bp"];   bp2  = p2_stats["bp"]
    form1 = p1_stats["form"]; form2 = p2_stats["form"]
    ace1  = p1_stats["ace_rate"]; ace2  = p2_stats["ace_rate"]
    ss1   = p1_stats["ss_won"];   ss2   = p2_stats["ss_won"]

    elo_p = elo_expected(elo_p1, elo_p2)

    def days_since(last_date, current_date):
        if last_date is None:
            return None
        return (current_date - last_date).days

    fat1 = fatigue_multiplier(days_since(p1_last_date, match_date))
    fat2 = fatigue_multiplier(days_since(p2_last_date, match_date))

    w = SURFACE_WEIGHTS.get(surface, DEFAULT_WEIGHTS)
    s1 = (w.get("return", 0)   * rgw1
        + w.get("elo", 0)      * elo_p
        + w.get("surf_elo", 0) * surf_elo_p
        + w.get("rank", 0)     * rank_p
        + w.get("ace", 0)      * ace1
        + w.get("ss_won", 0)   * ss1)   * fat1
    s2 = (w.get("return", 0)   * rgw2
        + w.get("elo", 0)      * (1-elo_p)
        + w.get("surf_elo", 0) * (1-surf_elo_p)
        + w.get("rank", 0)     * (1-rank_p)
        + w.get("ace", 0)      * ace2
        + w.get("ss_won", 0)   * ss2)   * fat2

    raw_p1 = s1 / (s1 + s2) if (s1 + s2) > 0 else 0.5
    raw_p1 = h2h_shrink(raw_p1, h2h_p1_wins, h2h_total)
    if best_of == 5:
        raw_p1 = bo5_adjust(raw_p1)
    raw_p1 = max(0.001, min(0.999, raw_p1))
    return raw_p1, elo_p, {
        "sgw1": sgw1,  "sgw2": sgw2,
        "rgw1": rgw1,  "rgw2": rgw2,
        "bp1":  bp1,   "bp2":  bp2,
        "form1": form1, "form2": form2,
        "ace1": ace1,  "ace2": ace2,
        "ss_won1": ss1, "ss_won2": ss2,
        "fat1": fat1,  "fat2": fat2,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_year(year: int) -> Optional[pd.DataFrame]:
    url = BASE_URL.format(year=year)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["tourney_date"])
        # Keep Challenger draws only
        df = df[df["tourney_level"] == "C"].reset_index(drop=True)
        df = df.sort_values("tourney_date").reset_index(drop=True)
        if len(df) > 0:
            print(f"  Loaded {year}: {len(df):,} Challenger matches")
            return df
    except Exception:
        pass
    print(f"  WARNING: Could not load {year}")
    return None


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest():
    print("Loading Challenger data...")
    frames = []
    for y in YEARS:
        df = load_year(y)
        if df is not None:
            frames.append(df)
    if not frames:
        print("ERROR: No data loaded. Exiting.")
        return

    all_matches = pd.concat(frames, ignore_index=True).sort_values("tourney_date")
    print(f"\nTotal matches: {len(all_matches):,}\n")

    elo_ratings   = {}
    surf_elo: dict   = {"Hard": {}, "Clay": {}, "Grass": {}}
    surf_elo_n: dict = {"Hard": collections.Counter(), "Clay": collections.Counter(), "Grass": collections.Counter()}
    last_rank_pts: dict = {}
    player_stats  = PlayerStats()
    h2h: dict     = collections.defaultdict(lambda: [0, 0])
    results    = []
    components = []

    for _, row in all_matches.iterrows():
        wname = str(row.get("winner_name", "") or "")
        lname = str(row.get("loser_name",  "") or "")
        surface  = str(row.get("surface", "Hard"))
        best_of  = int(row.get("best_of", 3)) if not pd.isna(row.get("best_of", 3)) else 3
        match_date = row["tourney_date"]
        rnd   = str(row.get("round", ""))

        if not wname or not lname:
            continue

        key = tuple(sorted([wname, lname]))
        h2h_wins_w = h2h[key][0] if wname == key[0] else h2h[key][1] - h2h[key][0]
        h2h_total  = h2h[key][1]
        h2h_wins_w = max(0, min(h2h_wins_w, h2h_total))

        elo_w = elo_ratings.get(wname, ELO_START)
        elo_l = elo_ratings.get(lname, ELO_START)
        w_stats = player_stats.get_stats(wname, surface)
        l_stats = player_stats.get_stats(lname, surface)
        w_last  = player_stats.last_date.get(wname)
        l_last  = player_stats.last_date.get(lname)

        surf_key = surface if surface in surf_elo else None
        surf = surf_elo.get(surf_key, {})
        def blended_surf_elo(pname, overall_r):
            if surf_key is None:
                return overall_r
            s_r = surf.get(pname, ELO_START)
            n   = surf_elo_n[surf_key][pname]
            return (n * s_r + SURFACE_ELO_PRIOR * overall_r) / (n + SURFACE_ELO_PRIOR)

        s_elo_w = blended_surf_elo(wname, elo_w)
        s_elo_l = blended_surf_elo(lname, elo_l)
        surf_elo_p_winner = elo_expected(s_elo_w, s_elo_l)

        rp_w = last_rank_pts.get(wname, 1000.0)
        rp_l = last_rank_pts.get(lname, 1000.0)
        rank_p_winner = rp_w / (rp_w + rp_l) if (rp_w + rp_l) > 0 else 0.5

        comp_p_winner, elo_p_winner, comps = composite_win_prob(
            w_stats, l_stats, elo_w, elo_l, surf_elo_p_winner, rank_p_winner, surface, best_of,
            w_last, l_last, match_date, h2h_wins_w, h2h_total
        )

        components.append({
            "sgw1": comps["sgw1"], "sgw2": comps["sgw2"],
            "bp1":  comps["bp1"],  "bp2":  comps["bp2"],
            "form1": comps["form1"], "form2": comps["form2"],
            "ace1": comps["ace1"], "ace2":  comps["ace2"],
            "ss_won1": comps["ss_won1"], "ss_won2": comps["ss_won2"],
            "elo_p": elo_p_winner, "surf_elo_p": surf_elo_p_winner,
            "rank_p": rank_p_winner,
            "fat1": comps["fat1"], "fat2": comps["fat2"],
            "sfm": 1.0, "best_of": best_of,
            "h2h_wins_w": h2h_wins_w, "h2h_total": h2h_total,
        })

        results.append({
            "tourney_date": match_date.strftime("%Y-%m-%d"),
            "tourney_name": row.get("tourney_name", ""),
            "surface": surface,
            "best_of": best_of,
            "round": rnd,
            "winner_name": wname,
            "loser_name":  lname,
            "elo_winner": round(elo_w, 1),
            "elo_loser":  round(elo_l, 1),
            "comp_win_prob": round(comp_p_winner, 4),
            "elo_win_prob":  round(elo_p_winner,  4),
        })

        w_rp = row.get("winner_rank_points")
        l_rp = row.get("loser_rank_points")
        if w_rp is not None and not pd.isna(w_rp):
            last_rank_pts[wname] = float(w_rp)
        if l_rp is not None and not pd.isna(l_rp):
            last_rank_pts[lname] = float(l_rp)

        update_elo(elo_ratings, wname, lname)
        if surface in surf_elo:
            update_elo(surf_elo[surface], wname, lname)
            surf_elo_n[surface][wname] += 1
            surf_elo_n[surface][lname] += 1
        player_stats.update(wname, surface, match_date, win=True,  row=row.to_dict(), prefix="w_")
        player_stats.update(lname, surface, match_date, win=False, row=row.to_dict(), prefix="l_")

        h2h[key][1] += 1
        if wname == key[0]:
            h2h[key][0] += 1

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv("backtest_results.csv", index=False)
    print("Saved: backtest_results.csv")

    pd.DataFrame(components).to_csv("backtest_components.csv", index=False)
    print("Saved: backtest_components.csv")

    elo_by_name = {name: round(r, 1) for name, r in elo_ratings.items()}
    surf_elo_by_name = {
        surface: {name: round(r, 1) for name, r in ratings.items()}
        for surface, ratings in surf_elo.items()
    }
    rank_pts_by_name = {name: round(pts, 0) for name, pts in last_rank_pts.items()}
    last_match_date = all_matches["tourney_date"].max().strftime("%Y-%m-%d")
    with open("player_elos.json", "w") as f:
        json.dump({"overall": elo_by_name, "surface": surf_elo_by_name,
                   "rank_points": rank_pts_by_name,
                   "last_updated": last_match_date}, f, indent=2)
    print(f"Saved: player_elos.json  (last match: {last_match_date})")

    # ── Metrics ───────────────────────────────────────────────────────────────
    comp = results_df["comp_win_prob"].values
    elo  = results_df["elo_win_prob"].values
    coin = np.full(len(comp), 0.5)

    acc_comp = np.mean(comp > 0.5)
    acc_elo  = np.mean(elo  > 0.5)
    brier_comp = np.mean((comp - 1) ** 2)
    brier_elo  = np.mean((elo  - 1) ** 2)
    brier_coin = np.mean((coin - 1) ** 2)
    logloss_comp = -np.mean(np.log(np.clip(comp, 1e-9, 1)))
    logloss_elo  = -np.mean(np.log(np.clip(elo,  1e-9, 1)))

    print()
    print(f"{'Metric':<30} {'Composite':>12} {'Elo-only':>10} {'Coin-flip':>10}")
    print("-" * 65)
    print(f"{'Accuracy (>50% = correct)':<30} {acc_comp*100:>11.1f}% {acc_elo*100:>9.1f}% {50.0:>9.1f}%")
    print(f"{'Brier score (lower=better)':<30} {brier_comp:>12.4f} {brier_elo:>10.4f} {brier_coin:>10.4f}")
    print(f"{'Log loss  (lower=better)':<30} {logloss_comp:>12.4f} {logloss_elo:>10.4f}")

    print("\nBy surface:")
    print(f"  {'Surface':<12} {'n':>7} {'Composite':>10} {'Elo':>8}")
    for surf in ["Hard", "Clay", "Grass"]:
        sub = results_df[results_df["surface"] == surf]
        if len(sub) == 0:
            continue
        a_c = np.mean(sub["comp_win_prob"].values > 0.5)
        a_e = np.mean(sub["elo_win_prob"].values  > 0.5)
        print(f"  {surf:<12} {len(sub):>7,} {a_c*100:>9.1f}% {a_e*100:>7.1f}%")

    print("\nBy round:")
    print(f"  {'Round':<12} {'n':>7} {'Composite':>10} {'Elo':>8}")
    for rnd in ["R128", "R64", "R32", "R16", "QF", "SF", "F"]:
        sub = results_df[results_df["round"] == rnd]
        if len(sub) == 0:
            continue
        a_c = np.mean(sub["comp_win_prob"].values > 0.5)
        a_e = np.mean(sub["elo_win_prob"].values  > 0.5)
        print(f"  {rnd:<12} {len(sub):>7,} {a_c*100:>9.1f}% {a_e*100:>7.1f}%")

    cal_probs    = np.concatenate([comp, 1.0 - comp])
    cal_outcomes = np.concatenate([np.ones(len(comp)), np.zeros(len(comp))])
    print("\nCalibration table (composite):")
    print(f"  {'Bucket':<12} {'n':>7} {'Implied':>9} {'Actual':>8} {'Bias':>8}")
    edges = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        mask = (cal_probs >= lo) & (cal_probs < hi)
        n = mask.sum()
        if n == 0:
            continue
        implied = float(cal_probs[mask].mean())
        actual  = float(cal_outcomes[mask].mean())
        bias    = implied - actual
        print(f"  {lo:.2f}-{hi:.2f}   {n:>7,}    {implied:>7.3f}   {actual:>7.3f}  {bias:>+8.3f}")

    print()


if __name__ == "__main__":
    run_backtest()
