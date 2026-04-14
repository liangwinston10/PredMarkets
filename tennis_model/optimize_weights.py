"""
Weight optimizer for the Tennis Composite Win% Model.
Loads backtest_components.csv (produced by backtest.py) and finds optimal
per-surface weights using scipy SLSQP (handles sum=1 constraint, bounds >=0).

Components searched: return, elo, surf_elo, rank, ace, ss_won
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMPONENT_NAMES = ["return", "elo", "surf_elo", "rank", "ace", "ss_won", "df_rate", "bp_conv"]


# ── Helpers (must match backtest.py exactly) ──────────────────────────────────

def bo5_adjust(p: np.ndarray) -> np.ndarray:
    q = 1 - p
    return p**3 * (1 + 3*q + 6*q**2)


def h2h_shrink_vec(p, h2h_wins, h2h_total):
    safe_total = np.maximum(h2h_total, 1)
    h2h_rate   = h2h_wins / safe_total
    blend      = np.minimum(h2h_total, 10) / 10.0 * 0.12
    shrunk     = p * (1 - blend) + h2h_rate * blend
    return np.where(h2h_total > 0, shrunk, p)


def eval_weights(w, arrays, mask=None):
    """Return (brier, accuracy) for weight vector [wr, we, wse, wrank, wace, wss, wdf, wbpc]."""
    wr, we, wse, wrank, wace, wss, wdf, wbpc = w

    rgw1 = 1.0 - arrays["sgw2"]
    rgw2 = 1.0 - arrays["sgw1"]
    elo_p      = arrays["elo_p"]
    surf_elo_p = arrays["surf_elo_p"]
    rank_p     = arrays["rank_p"]
    ace1       = arrays["ace1"];  ace2  = arrays["ace2"]
    ss1        = arrays["ss_won1"]; ss2 = arrays["ss_won2"]
    df1        = arrays["df_rate1"]; df2 = arrays["df_rate2"]
    bpc1       = arrays["bp_conv1"]; bpc2 = arrays["bp_conv2"]
    fat1       = arrays["fat1"];  fat2  = arrays["fat2"]
    best_of    = arrays["best_of"]
    h2h_wins_w = arrays["h2h_wins_w"]
    h2h_total  = arrays["h2h_total"]

    # df_rate is negative signal: use (1 - df_rate)
    s1 = (wr*rgw1 + we*elo_p      + wse*surf_elo_p      + wrank*rank_p      + wace*ace1 + wss*ss1 + wdf*(1-df1) + wbpc*bpc1) * fat1
    s2 = (wr*rgw2 + we*(1-elo_p)  + wse*(1-surf_elo_p)  + wrank*(1-rank_p)  + wace*ace2 + wss*ss2 + wdf*(1-df2) + wbpc*bpc2) * fat2

    total = s1 + s2
    p = np.where(total > 0, s1 / total, 0.5)
    p = h2h_shrink_vec(p, h2h_wins_w, h2h_total)
    p = np.where(best_of == 5, bo5_adjust(p), p)
    p = np.clip(p, 0.001, 0.999)

    if mask is not None:
        p = p[mask]

    brier    = float(np.mean((p - 1.0) ** 2))
    accuracy = float(np.mean(p > 0.5))
    return brier, accuracy


def optimize_for_mask(arrays, mask, label):
    """Run SLSQP minimization for a given subset of matches."""
    n = mask.sum() if mask is not None else len(arrays["elo_p"])

    def objective(w):
        brier, _ = eval_weights(w, arrays, mask)
        return brier

    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}
    bounds = [(0, 1)] * len(COMPONENT_NAMES)

    # Multi-start: try several starting points to avoid local minima
    best = None
    starts = [
        [1/8]*8,
        [0.25, 0.40, 0.35, 0, 0, 0, 0, 0],
        [0.1, 0.5, 0.3, 0.1, 0, 0, 0, 0],
        [0.2, 0.3, 0.2, 0.2, 0.05, 0.05, 0, 0],
        [0, 0.5, 0.3, 0.2, 0, 0, 0, 0],
        [0.3, 0.4, 0, 0.3, 0, 0, 0, 0],
        [0.2, 0.4, 0, 0.2, 0.1, 0.1, 0, 0],
        [0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05],
        [0.15, 0.35, 0.25, 0.15, 0, 0, 0.05, 0.05],
    ]
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 1000})
        if res.success and (best is None or res.fun < best.fun):
            best = res

    return best


# ── Main ──────────────────────────────────────────────────────────────────────

OOS_YEARS = (2023, 2024)   # hold-out test set — never train on this


def main():
    print("Loading backtest_components.csv ...")
    try:
        df = pd.read_csv("backtest_components.csv")
    except FileNotFoundError:
        print("ERROR: backtest_components.csv not found. Run backtest.py first.")
        return

    try:
        res_df = pd.read_csv("backtest_results.csv")
        surfaces = res_df["surface"].values
        years    = pd.to_datetime(res_df["tourney_date"]).dt.year.values
    except FileNotFoundError:
        print("ERROR: backtest_results.csv not found.")
        return

    # Restrict to OOS years only to avoid in-sample overfitting
    oos_mask = np.isin(years, list(OOS_YEARS))
    n_oos = oos_mask.sum()
    print(f"  {len(df):,} total matches;  {n_oos:,} OOS ({OOS_YEARS[0]}-{OOS_YEARS[1]}) used for optimization.\n")

    arrays = {col: df[col].values for col in df.columns}
    arrays["best_of"]    = arrays["best_of"].astype(int)
    arrays["h2h_wins_w"] = arrays["h2h_wins_w"].astype(float)
    arrays["h2h_total"]  = arrays["h2h_total"].astype(float)

    col_width = 8 * len(COMPONENT_NAMES) + 2 * (len(COMPONENT_NAMES) - 1)
    print(f"{'Surface':<10} {'n':>6}  "
          + "  ".join(f"{c:>8}" for c in COMPONENT_NAMES)
          + f"  {'Brier':>8} {'Acc':>8}")
    print("-" * (10 + 8 + 2 + col_width + 20))

    surface_results = {}

    for label in ["Hard", "Clay", "Grass", "ALL"]:
        if label == "ALL":
            mask = oos_mask
        else:
            mask = oos_mask & (surfaces == label)

        n = mask.sum()
        if n == 0:
            continue

        result = optimize_for_mask(arrays, mask, label)
        if result is None:
            print(f"  {label}: optimization failed")
            continue

        w = result.x
        brier, acc = eval_weights(w, arrays, mask)
        surface_results[label] = dict(zip(COMPONENT_NAMES, w))

        w_str = "  ".join(f"{v:>8.3f}" for v in w)
        print(f"{label:<10} {n:>6,}  {w_str}  {brier:>8.4f} {acc*100:>7.1f}%")

    print()
    print("Suggested SURFACE_WEIGHTS:")
    for surf in ["Hard", "Clay", "Grass"]:
        if surf not in surface_results:
            continue
        w = surface_results[surf]
        # Zero out components with weight < 0.02 (noise floor)
        cleaned = {k: round(v, 3) for k, v in w.items() if v >= 0.02}
        total = sum(cleaned.values())
        cleaned = {k: round(v / total, 3) for k, v in cleaned.items()}
        print(f'    "{surf}": {cleaned},')


if __name__ == "__main__":
    main()
