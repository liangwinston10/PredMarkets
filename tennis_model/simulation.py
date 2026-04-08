"""
Point-level Bernoulli simulation engine for tennis match prediction.
Models point → game → set → match through the scoring hierarchy.

Exports
-------
p_hold_game(p)                          - closed-form hold probability
simulate_tiebreak(p_a, p_b, ...)        - single tiebreak
simulate_set(p_a, p_b, ...)             - single set
simulate_match(p_a, p_b, best_of)       - single match
run_simulation(p_serve_a, p_serve_b, …) - Monte Carlo runner
"""

import random
import math
import collections
from scipy.optimize import brentq


# ── Game-level model (closed-form) ────────────────────────────────────────────

def p_hold_game(p: float) -> float:
    """
    Probability of holding serve given point-win rate p.
    Closed-form; no simulation needed.
    """
    q = 1 - p
    p_no_deuce = (
        p**4
        + 4  * p**4 * q
        + 10 * p**4 * q**2
    )
    p_deuce    = 20 * p**3 * q**3
    p_win_deuce = p**2 / (p**2 + q**2)
    return p_no_deuce + p_deuce * p_win_deuce


# ── SGW inversion ────────────────────────────────────────────────────────────

def sgw_to_point_prob(sgw: float) -> float:
    """
    Invert p_hold_game() to recover the point-win-on-serve probability
    from an observed service game hold rate.
    Solves: p_hold_game(p) = sgw  via Brent's method.
    """
    sgw = max(0.01, min(0.99, sgw))
    return brentq(lambda p: p_hold_game(p) - sgw, 0.01, 0.99)


# ── Tiebreak model (simulation) ───────────────────────────────────────────────

def simulate_tiebreak(p_a: float, p_b: float, a_serves_first: bool = True) -> bool:
    """
    Simulate a single tiebreak. Returns True if Player A wins.

    Serve pattern: A serves point 1, then alternate every 2 points.
    """
    score_a, score_b = 0, 0
    point_num = 0

    while True:
        if point_num == 0:
            a_serving = a_serves_first
        else:
            a_serving = ((point_num - 1) // 2) % 2 == (0 if a_serves_first else 1)

        p_server_wins = p_a if a_serving else p_b
        server_wins   = random.random() < p_server_wins

        if a_serving:
            if server_wins: score_a += 1
            else:           score_b += 1
        else:
            if server_wins: score_b += 1
            else:           score_a += 1

        point_num += 1

        if score_a >= 7 and score_a - score_b >= 2:
            return True
        if score_b >= 7 and score_b - score_a >= 2:
            return False


# ── Set model (simulation) ────────────────────────────────────────────────────

def simulate_set(p_a: float, p_b: float, a_serves_first: bool = True) -> dict:
    """
    Simulate a single set.

    Returns
    -------
    dict with keys:
        winner        : "A" or "B"
        score         : (games_a, games_b)
        tiebreak      : bool
        a_serves_next : bool  — who serves first in the next set
    """
    games_a, games_b = 0, 0
    a_serving = a_serves_first

    while True:
        if a_serving:
            hold = random.random() < p_hold_game(p_a)
            if hold: games_a += 1
            else:    games_b += 1
        else:
            hold = random.random() < p_hold_game(p_b)
            if hold: games_b += 1
            else:    games_a += 1

        a_serving = not a_serving

        if games_a >= 6 and games_a - games_b >= 2:
            return {"winner": "A", "score": (games_a, games_b),
                    "tiebreak": False, "a_serves_next": a_serving}
        if games_b >= 6 and games_b - games_a >= 2:
            return {"winner": "B", "score": (games_a, games_b),
                    "tiebreak": False, "a_serves_next": a_serving}

        if games_a == 6 and games_b == 6:
            a_wins_tb = simulate_tiebreak(p_a, p_b, a_serving)
            if a_wins_tb:
                return {"winner": "A", "score": (7, 6),
                        "tiebreak": True, "a_serves_next": not a_serving}
            else:
                return {"winner": "B", "score": (6, 7),
                        "tiebreak": True, "a_serves_next": not a_serving}


# ── Match model (simulation) ──────────────────────────────────────────────────

def simulate_match(p_a: float, p_b: float, best_of: int = 3) -> dict:
    """
    Simulate a single match.

    Returns
    -------
    dict with keys:
        winner      : "A" or "B"
        sets_a      : int
        sets_b      : int
        set_scores  : list of (games_a, games_b)
        total_games : int
    """
    sets_to_win = 2 if best_of == 3 else 3
    sets_a, sets_b = 0, 0
    set_scores = []
    a_serves_first = True

    while sets_a < sets_to_win and sets_b < sets_to_win:
        result = simulate_set(p_a, p_b, a_serves_first)
        set_scores.append(result["score"])
        if result["winner"] == "A":
            sets_a += 1
        else:
            sets_b += 1
        a_serves_first = result["a_serves_next"]

    winner = "A" if sets_a == sets_to_win else "B"
    total_games = sum(a + b for a, b in set_scores)

    return {
        "winner":      winner,
        "sets_a":      sets_a,
        "sets_b":      sets_b,
        "set_scores":  set_scores,
        "total_games": total_games,
    }


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def run_simulation(
    p_serve_a: float,
    p_serve_b: float,
    best_of:   int  = 3,
    n_sims:    int  = 10_000,
    seed:      int | None = None,
) -> dict:
    """
    Run n_sims match simulations and aggregate results.

    Parameters
    ----------
    p_serve_a : float  Player A's point-win prob on serve (opponent-adjusted).
    p_serve_b : float  Player B's point-win prob on serve (opponent-adjusted).
    best_of   : int    3 or 5.
    n_sims    : int    Number of simulations. 10K → ~±1pp precision.
    seed      : int|None  Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        p_match_a       : float
        p_match_b       : float
        set_score_dist  : dict  e.g. {"2-0": 0.49, "2-1": 0.25, …}
        avg_total_games : float
        avg_total_sets  : float
        confidence      : float  straight-set fraction of A's wins
        sim_std         : float  standard error of p_match_a
        p_serve_a       : float  (echo)
        p_serve_b       : float  (echo)
        n_sims          : int
    """
    if seed is not None:
        random.seed(seed)

    wins_a      = 0
    score_counts = collections.Counter()
    total_games  = 0
    total_sets   = 0

    for _ in range(n_sims):
        m = simulate_match(p_serve_a, p_serve_b, best_of)
        if m["winner"] == "A":
            wins_a += 1
        key = f"{m['sets_a']}-{m['sets_b']}"
        score_counts[key] += 1
        total_games += m["total_games"]
        total_sets  += m["sets_a"] + m["sets_b"]

    p_match_a = wins_a / n_sims
    p_match_b = 1.0 - p_match_a

    set_score_dist = {k: v / n_sims for k, v in score_counts.items()}

    straight_key   = "2-0" if best_of == 3 else "3-0"
    straight_wins  = score_counts.get(straight_key, 0) / n_sims
    confidence     = straight_wins / p_match_a if p_match_a > 0 else 0.0

    sim_std = math.sqrt(p_match_a * p_match_b / n_sims)

    return {
        "p_match_a":       p_match_a,
        "p_match_b":       p_match_b,
        "set_score_dist":  set_score_dist,
        "avg_total_games": total_games / n_sims,
        "avg_total_sets":  total_sets  / n_sims,
        "confidence":      confidence,
        "sim_std":         sim_std,
        "p_serve_a":       p_serve_a,
        "p_serve_b":       p_serve_b,
        "n_sims":          n_sims,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== p_hold_game checkpoints ===")
    for p, expected in [(0.60, 0.736), (0.65, 0.829), (0.70, 0.901)]:
        got = p_hold_game(p)
        print(f"  p={p}  got={got:.3f}  expected~{expected}")

    print("\n=== Simulation: Sinner(A) vs Cerundolo(B), p_serve_a=0.66, p_serve_b=0.61, Bo3 ===")
    r = run_simulation(0.66, 0.61, best_of=3, n_sims=50_000, seed=42)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
