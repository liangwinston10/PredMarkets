"""
Backtest comparison: Composite model vs Simulation model vs Elo-only.

Reads existing backtest_results.csv + backtest_components.csv (from backtest.py),
applies a vectorized numpy simulation to each row using the pre-computed sgw values,
and compares Brier score, log-loss, and accuracy across all three models.

Usage:
    python backtest_sim.py
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import p_hold_game, sgw_to_point_prob

N_SIMS  = 5_000   # per match — vectorized so this is fast
BLEND_W = 0.60


def brier(probs):
    return float(np.mean((np.array(probs) - 1.0) ** 2))

def logloss(probs):
    return float(-np.mean(np.log(np.clip(probs, 1e-9, 1.0))))

def accuracy(probs):
    return float(np.mean(np.array(probs) > 0.5))


def sim_match_vectorized(p_serve_a: float, p_serve_b: float,
                          best_of: int, n: int) -> float:
    """
    Vectorized numpy simulation of n matches. Returns P(A wins).
    Uses closed-form p_hold_game() for games, random draws only for tiebreaks.
    Each of the n simulations runs as a vector operation.
    """
    sets_to_win = 2 if best_of == 3 else 3
    ph_a = p_hold_game(p_serve_a)
    ph_b = p_hold_game(p_serve_b)

    sets_a = np.zeros(n, dtype=np.int8)
    sets_b = np.zeros(n, dtype=np.int8)
    alive  = np.ones(n, dtype=bool)
    a_serves_first = np.ones(n, dtype=bool)

    while np.any(alive):
        # Simulate one set for all alive matches
        ga = np.zeros(n, dtype=np.int8)
        gb = np.zeros(n, dtype=np.int8)
        a_srv = a_serves_first.copy()

        for _ in range(13):  # max games before tiebreak check
            # A serving
            a_serving_now = alive & a_srv
            if np.any(a_serving_now):
                holds = np.random.random(n) < ph_a
                ga = np.where(a_serving_now &  holds, ga + 1, ga)
                gb = np.where(a_serving_now & ~holds, gb + 1, gb)
            # B serving
            b_serving_now = alive & ~a_srv
            if np.any(b_serving_now):
                holds = np.random.random(n) < ph_b
                gb = np.where(b_serving_now &  holds, gb + 1, gb)
                ga = np.where(b_serving_now & ~holds, ga + 1, ga)
            a_srv = ~a_srv
            # Check early set wins (6-x with 2+ lead, x<=4)
            set_done = ((ga >= 6) & (ga - gb >= 2)) | ((gb >= 6) & (gb - ga >= 2))
            if np.all(set_done[alive]):
                break

        # Tiebreak at 6-6: approximate with p^2/(p^2+q^2) closed form
        at_tb = alive & (ga == 6) & (gb == 6)
        if np.any(at_tb):
            p_tb_a = p_serve_a**2 / (p_serve_a**2 + p_serve_b**2)
            tb_a_wins = np.random.random(n) < p_tb_a
            ga = np.where(at_tb &  tb_a_wins, 7, ga)
            gb = np.where(at_tb & ~tb_a_wins, 7, gb)  # score stored as 6
            gb = np.where(at_tb &  tb_a_wins, 6, gb)
            ga = np.where(at_tb & ~tb_a_wins, 6, ga)

        set_won_a = alive & (ga > gb)
        set_won_b = alive & (gb > ga)
        sets_a = np.where(set_won_a, sets_a + 1, sets_a)
        sets_b = np.where(set_won_b, sets_b + 1, sets_b)

        # Who serves first next set — player who received in this set serves next
        # Approximate: flip if set didn't go to tiebreak (net games even), keep otherwise
        a_serves_first = np.where(alive, ~a_serves_first, a_serves_first)

        match_done = (sets_a >= sets_to_win) | (sets_b >= sets_to_win)
        alive = alive & ~match_done

    return float(np.mean(sets_a >= sets_to_win))


def main():
    results_path    = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
    components_path = os.path.join(os.path.dirname(__file__), "backtest_components.csv")

    print("Loading CSVs...")
    res  = pd.read_csv(results_path)
    comp = pd.read_csv(components_path)

    assert len(res) == len(comp), "Row count mismatch between CSVs"
    comp = comp.drop(columns=["best_of"], errors="ignore")
    df = pd.concat([res, comp], axis=1)
    print(f"  {len(df):,} matches loaded")

    # ── Run vectorized simulation ─────────────────────────────────────────────
    print(f"Running vectorized simulation ({N_SIMS} sims/match)...")

    # Use surface-specific SGW if available (from surf_skill_deques), else fall back to all-surface
    sgw1 = df["sgw1_surf"].values if "sgw1_surf" in df.columns else df["sgw1"].values
    sgw2 = df["sgw2_surf"].values if "sgw2_surf" in df.columns else df["sgw2"].values
    best_of = df["best_of"].fillna(3).astype(int).values
    comp_p  = df["comp_win_prob"].values

    src = "surface-specific SGW" if "sgw1_surf" in df.columns else "all-surface SGW (fallback)"
    print(f"  Using {src} for simulation inputs")

    pp_w = np.array([sgw_to_point_prob(v) for v in sgw1])
    pp_l = np.array([sgw_to_point_prob(v) for v in sgw2])
    p_serve_w = (pp_w + (1 - pp_l)) / 2
    p_serve_l = (pp_l + (1 - pp_w)) / 2

    sim_probs   = np.zeros(len(df))
    blend_probs = np.zeros(len(df))

    # Group by (p_serve_w, p_serve_l, best_of) rounded to 3dp to avoid
    # running 30K unique sims — many rows share nearly identical inputs
    keys = np.round(np.column_stack([p_serve_w, p_serve_l]), 3)
    bo_arr = best_of

    unique_keys = {}
    for i, (k, bo) in enumerate(zip(map(tuple, keys), bo_arr)):
        cache_key = (k[0], k[1], bo)
        if cache_key not in unique_keys:
            unique_keys[cache_key] = None

    print(f"  Unique (p_a, p_b, bo) combos: {len(unique_keys):,}  (caching results)")

    cache = {}
    for i, (k, bo) in enumerate(zip(map(tuple, keys), bo_arr)):
        cache_key = (k[0], k[1], int(bo))
        if cache_key not in cache:
            cache[cache_key] = sim_match_vectorized(k[0], k[1], int(bo), N_SIMS)
        sim_p = cache[cache_key]
        sim_probs[i]   = sim_p
        blend_probs[i] = BLEND_W * sim_p + (1 - BLEND_W) * comp_p[i]

        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} / {len(df):,}  (cache size: {len(cache):,})")

    df["sim_win_prob"]   = sim_probs
    df["blend_win_prob"] = blend_probs

    # ── Overall metrics ───────────────────────────────────────────────────────
    models = {
        "Composite": df["comp_win_prob"].values,
        "Simulation": df["sim_win_prob"].values,
        f"Blend {int(BLEND_W*100)}/{int((1-BLEND_W)*100)}": df["blend_win_prob"].values,
        "Elo-only":  df["elo_win_prob"].values,
        "Coin-flip": np.full(len(df), 0.5),
    }

    print()
    print(f"{'Model':<22} {'Accuracy':>9} {'Brier':>8} {'LogLoss':>9}")
    print("-" * 52)
    for name, probs in models.items():
        print(f"{name:<22} {accuracy(probs)*100:>8.1f}%  {brier(probs):>8.4f}  {logloss(probs):>9.4f}")

    # ── By surface ────────────────────────────────────────────────────────────
    print("\nBy surface — Accuracy:")
    print(f"  {'Surface':<10} {'n':>7}  {'Comp':>7}  {'Sim':>7}  {'Blend':>7}  {'Elo':>7}")
    for surf in ["Hard", "Clay", "Grass"]:
        sub = df[df["surface"] == surf]
        if len(sub) == 0:
            continue
        print(f"  {surf:<10} {len(sub):>7,}  "
              f"{accuracy(sub['comp_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['sim_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['blend_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['elo_win_prob'].values)*100:>6.1f}%")

    # ── By surface — Brier ────────────────────────────────────────────────────
    print("\nBy surface — Brier score (lower=better):")
    print(f"  {'Surface':<10} {'n':>7}  {'Comp':>8}  {'Sim':>8}  {'Blend':>8}")
    for surf in ["Hard", "Clay", "Grass"]:
        sub = df[df["surface"] == surf]
        if len(sub) == 0:
            continue
        print(f"  {surf:<10} {len(sub):>7,}  "
              f"{brier(sub['comp_win_prob'].values):>8.4f}  "
              f"{brier(sub['sim_win_prob'].values):>8.4f}  "
              f"{brier(sub['blend_win_prob'].values):>8.4f}")

    # ── By round ─────────────────────────────────────────────────────────────
    print("\nBy round — Accuracy:")
    print(f"  {'Round':<8} {'n':>7}  {'Comp':>7}  {'Sim':>7}  {'Blend':>7}")
    for rnd in ["R128", "R64", "R32", "R16", "QF", "SF", "F"]:
        sub = df[df["round"] == rnd]
        if len(sub) == 0:
            continue
        print(f"  {rnd:<8} {len(sub):>7,}  "
              f"{accuracy(sub['comp_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['sim_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['blend_win_prob'].values)*100:>6.1f}%")

    # ── By best_of ────────────────────────────────────────────────────────────
    print("\nBy format — Accuracy:")
    print(f"  {'Format':<8} {'n':>7}  {'Comp':>7}  {'Sim':>7}  {'Blend':>7}")
    for bo in [3, 5]:
        sub = df[df["best_of"] == bo]
        if len(sub) == 0:
            continue
        print(f"  Bo{bo:<6} {len(sub):>7,}  "
              f"{accuracy(sub['comp_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['sim_win_prob'].values)*100:>6.1f}%  "
              f"{accuracy(sub['blend_win_prob'].values)*100:>6.1f}%")

    # ── Save enriched results ─────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "backtest_sim_results.csv")
    df[["tourney_date","tourney_name","surface","best_of","round",
        "winner_name","loser_name",
        "comp_win_prob","sim_win_prob","blend_win_prob","elo_win_prob"]].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
