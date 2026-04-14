"""
Form adjustment backtest: raw win-rate vs quality-adjusted (Elo-based) form.

Replays backtest_results.csv chronologically to compute rolling quality-form scores
(no need to re-run the full backtest), then sweeps blend weights across both methods.

Methods compared
----------------
  Raw      : existing form1/form2 from backtest_components (decayed win-rate, Bayesian-shrunk)
  Quality  : Elo-based over/under-performance — 0.5 + mean(actual - elo_expected)

Blend formula:  form_adj = (1-w)*comp + w*(c1/(c1+c2))
  c1 = form score of winner (from comp's perspective)
  c2 = form score of loser

Outputs
-------
  - Full-sample and OOS (2024+) weight sweep tables
  - Best weight per method highlighted
  - Side-by-side surface breakdown at best OOS weight

Usage:
    python backtest_form.py
"""

import os
import collections
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
FORM_LEN    = 30          # rolling window — matches backtest.py
FORM_FLOOR  = 0.10        # clamp form scores to [FORM_FLOOR, 1-FORM_FLOOR]
MIN_ENTRIES = 3           # minimum deque entries to use quality score (else fall back)
OOS_YEAR    = 2024        # hold-out test year
WEIGHTS     = np.round(np.arange(0, 0.325, 0.025), 3)   # 0% … 30%


# ── Metrics ────────────────────────────────────────────────────────────────────

def brier(probs):
    return float(np.mean((np.asarray(probs) - 1.0) ** 2))

def logloss(probs):
    return float(-np.mean(np.log(np.clip(probs, 1e-9, 1.0))))

def accuracy(probs):
    return float(np.mean(np.asarray(probs) > 0.5))


# ── Quality form replay ───────────────────────────────────────────────────────

def build_quality_form(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Process df chronologically, building per-(player,surface) rolling deques of
    (win: int, elo_diff: float).  Returns arrays form1_q, form2_q (winner, loser)
    aligned to df row order.  None → insufficient data → caller falls back to raw.
    """
    quality_deques: dict = collections.defaultdict(
        lambda: collections.deque(maxlen=FORM_LEN)
    )

    f1q = []
    f2q = []

    for _, row in df.iterrows():
        wname   = row["winner_name"]
        lname   = row["loser_name"]
        surface = str(row["surface"])
        elo_w   = float(row["elo_winner"])
        elo_l   = float(row["elo_loser"])

        def _quality(player, surf):
            entries = list(quality_deques[(player, surf)])
            diffs = []
            for e in entries:
                # elo_diff stored as player_elo - opp_elo at time of that past match
                expected = 1.0 / (1.0 + 10 ** (-e["elo_diff"] / 400.0))
                diffs.append(e["win"] - expected)
            if len(diffs) < MIN_ENTRIES:
                return None
            return max(FORM_FLOOR, min(1 - FORM_FLOOR, 0.5 + sum(diffs) / len(diffs)))

        f1q.append(_quality(wname, surface))
        f2q.append(_quality(lname, surface))

        # Update AFTER computing scores (no look-ahead)
        quality_deques[(wname, surface)].append({"win": 1, "elo_diff": elo_w - elo_l})
        quality_deques[(lname, surface)].append({"win": 0, "elo_diff": elo_l - elo_w})

    return np.array(f1q, dtype=object), np.array(f2q, dtype=object)


# ── Form blend ────────────────────────────────────────────────────────────────

def apply_form_blend(comp: np.ndarray, f1: np.ndarray, f2: np.ndarray,
                     f1_raw: np.ndarray, f2_raw: np.ndarray, w: float) -> np.ndarray:
    """
    Apply form blend at weight w.
    f1/f2 may contain None — fall back to f1_raw/f2_raw in that case.
    """
    out = np.empty(len(comp))
    for i in range(len(comp)):
        c1 = f1[i] if f1[i] is not None else f1_raw[i]
        c2 = f2[i] if f2[i] is not None else f2_raw[i]
        c1 = max(FORM_FLOOR, min(1 - FORM_FLOOR, float(c1)))
        c2 = max(FORM_FLOOR, min(1 - FORM_FLOOR, float(c2)))
        ratio = c1 / (c1 + c2)
        out[i] = (1 - w) * comp[i] + w * ratio
    return out


def sweep(subset: pd.DataFrame, f1_raw: np.ndarray, f2_raw: np.ndarray,
          f1_q: np.ndarray, f2_q: np.ndarray, label: str):
    """Print weight sweep table for this subset."""
    comp_arr = subset["comp_win_prob"].values
    elo_arr  = subset["elo_win_prob"].values

    print(f"\n{'='*72}")
    print(f"  {label}   n={len(subset):,}")
    print(f"{'='*72}")
    print(f"  {'Model':<26} {'Weight':>6}  {'Acc':>7}  {'Brier':>8}  {'LogLoss':>9}")
    print(f"  {'-'*60}")

    # Baselines
    print(f"  {'Elo-only':<26} {'—':>6}  {accuracy(elo_arr)*100:>6.2f}%  "
          f"{brier(elo_arr):>8.4f}  {logloss(elo_arr):>9.4f}")
    print(f"  {'Composite (w=0)':<26} {'0.0%':>6}  {accuracy(comp_arr)*100:>6.2f}%  "
          f"{brier(comp_arr):>8.4f}  {logloss(comp_arr):>9.4f}")
    print()

    best = {}   # method → (best_w, best_brier, best_probs)

    for method, m_f1, m_f2 in [
        ("Raw win-rate",  f1_raw, f2_raw),
        ("Quality (Elo)", f1_q,   f2_q),
    ]:
        print(f"  -- {method} --")
        best_b = float("inf")
        best_w = 0
        best_p = comp_arr

        for w in WEIGHTS[1:]:   # skip 0 — already shown as baseline
            probs = apply_form_blend(comp_arr, m_f1, m_f2, f1_raw, f2_raw, w)
            b = brier(probs)
            marker = " ◄" if b < best_b else ""
            if b < best_b:
                best_b, best_w, best_p = b, w, probs
            print(f"  {'':26} {w*100:>5.1f}%  {accuracy(probs)*100:>6.2f}%  "
                  f"{b:>8.4f}  {logloss(probs):>9.4f}{marker}")

        best[method] = (best_w, best_b, best_p)
        print(f"  >> Best weight: {best_w*100:.1f}%  "
              f"Brier {best_b:.4f}  vs comp {brier(comp_arr):.4f}  "
              f"Δ={brier(comp_arr)-best_b:+.4f}\n")

    return best


def surface_breakdown(subset: pd.DataFrame, f1_raw, f2_raw, f1_q, f2_q,
                      best_raw_w: float, best_q_w: float):
    """Print per-surface comparison at optimal weights."""
    comp_arr = subset["comp_win_prob"].values
    raw_adj  = apply_form_blend(comp_arr, f1_raw, f2_raw, f1_raw, f2_raw, best_raw_w)
    q_adj    = apply_form_blend(comp_arr, f1_q,   f2_q,   f1_raw, f2_raw, best_q_w)

    print(f"\n  Surface breakdown (raw w={best_raw_w*100:.1f}%  quality w={best_q_w*100:.1f}%):")
    print(f"  {'Surface':<8} {'n':>6}  {'Comp':>7}  {'Raw':>7}  {'Quality':>9}  "
          f"{'Elo':>7}  {'Best'}")
    print(f"  {'-'*65}")

    for surf in ["Hard", "Clay", "Grass"]:
        mask = (subset["surface"] == surf).values
        if mask.sum() < 10:
            continue
        b_comp = brier(comp_arr[mask])
        b_raw  = brier(raw_adj[mask])
        b_q    = brier(q_adj[mask])
        b_elo  = brier(subset.loc[subset["surface"] == surf, "elo_win_prob"].values)
        best_m = "Quality" if b_q < b_raw else "Raw"
        best_m = best_m if min(b_q, b_raw) < b_comp else "Comp"
        print(f"  {surf:<8} {mask.sum():>6,}  {b_comp:>7.4f}  {b_raw:>7.4f}  "
              f"{b_q:>9.4f}  {b_elo:>7.4f}  {best_m}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    results_path    = os.path.join(base, "backtest_results.csv")
    components_path = os.path.join(base, "backtest_components.csv")

    print("Loading CSVs...")
    res  = pd.read_csv(results_path)
    comp = pd.read_csv(components_path)
    assert len(res) == len(comp), "Row count mismatch between CSVs"
    comp = comp.drop(columns=["best_of"], errors="ignore")
    df   = pd.concat([res, comp], axis=1)
    df["year"] = pd.to_datetime(df["tourney_date"]).dt.year
    print(f"  {len(df):,} matches  ({df['year'].min()}–{df['year'].max()})")

    print("Building rolling quality-form scores (chronological replay)...")
    f1_q_all, f2_q_all = build_quality_form(df)
    n_matched = int(np.sum([v is not None for v in f1_q_all]))
    print(f"  Quality form matched: {n_matched:,} / {len(df):,} "
          f"({n_matched/len(df)*100:.1f}%)")

    f1_raw_all = df["form1"].values
    f2_raw_all = df["form2"].values

    # ── Full sample ────────────────────────────────────────────────────────────
    full_best = sweep(df, f1_raw_all, f2_raw_all, f1_q_all, f2_q_all,
                      "FULL SAMPLE")
    surface_breakdown(df, f1_raw_all, f2_raw_all, f1_q_all, f2_q_all,
                      full_best["Raw win-rate"][0], full_best["Quality (Elo)"][0])

    # ── OOS test ───────────────────────────────────────────────────────────────
    oos_mask   = df["year"] >= OOS_YEAR
    oos_df     = df[oos_mask].reset_index(drop=True)
    f1_q_oos   = f1_q_all[oos_mask.values]
    f2_q_oos   = f2_q_all[oos_mask.values]
    f1_raw_oos = f1_raw_all[oos_mask.values]
    f2_raw_oos = f2_raw_all[oos_mask.values]

    oos_best = sweep(oos_df, f1_raw_oos, f2_raw_oos, f1_q_oos, f2_q_oos,
                     f"OOS {OOS_YEAR}+")
    surface_breakdown(oos_df, f1_raw_oos, f2_raw_oos, f1_q_oos, f2_q_oos,
                      oos_best["Raw win-rate"][0], oos_best["Quality (Elo)"][0])

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  RECOMMENDATION (based on OOS Brier)")
    print(f"{'='*72}")
    raw_w, raw_b, _ = oos_best["Raw win-rate"]
    q_w,   q_b,   _ = oos_best["Quality (Elo)"]
    comp_b = brier(oos_df["comp_win_prob"].values)
    winner = "Quality (Elo)" if q_b <= raw_b else "Raw win-rate"
    best_w = q_w if winner == "Quality (Elo)" else raw_w
    best_b = min(q_b, raw_b)
    print(f"  Method:  {winner}")
    print(f"  Weight:  {best_w*100:.1f}%")
    print(f"  Brier:   {best_b:.4f}  (comp={comp_b:.4f}, Δ={comp_b-best_b:+.4f})")
    if best_b >= comp_b:
        print("  NOTE: Form adjustment does not improve on composite — consider weight=0")


if __name__ == "__main__":
    main()
