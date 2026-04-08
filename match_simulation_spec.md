# Match Simulation Engine — Specification

This document specifies a point-level Bernoulli simulation engine for tennis match prediction. It replaces (or supplements) the current weighted-average composite in `live.py` with a mechanistic model that builds match win probability from point-level dynamics through the game → set → match hierarchy.

---

## 1. Motivation

The current `predict()` function in `live.py` computes `comp_p1` as a linear weighted average of component signals (Elo, surface Elo, ace rate, rank points, return quality). This is correlational — it fits a number to an outcome without modeling the mechanism that produces the outcome.

Tennis scoring is deeply nonlinear. A small edge in point-win rate compounds dramatically:
- 52% point-win rate → ~60% game-win rate → ~70%+ match-win rate (Bo3)
- The current linear blend can't express this compounding

The simulation models point → game → set → match and produces:
- A more accurate `p_match` (replaces `comp_p1`)
- A full outcome distribution (set scores, total games)
- A confidence signal derived from distribution shape, usable for sizing adjustments

---

## 2. Inputs

### Two numbers drive everything:

- **`p_serve_a`**: Probability Player A wins a point when A is serving (A's serve quality vs B's return quality)
- **`p_serve_b`**: Probability Player B wins a point when B is serving (B's serve quality vs A's return quality)

### Deriving `p_serve` from existing `live.py` stats

`build_player_stats()` already computes per-player rolling stats including `sgw` (service game win rate) and the raw components that feed it. The `compute_sgw()` function calculates:

```python
sgw = f_in * f_won + (1 - f_in) * s_won
```

where `f_in` = first serve in rate, `f_won` = first serve points won rate, `s_won` = second serve points won rate. This is effectively the point-win rate on serve (though it maps to game-level hold rate through a nonlinear transformation — see Section 3).

**Opponent-adjusted serve probability:**

Each player's effective `p_serve` in a specific matchup should blend their own serve quality with the opponent's return quality. `rgw` (return game win rate) is already computed in `predict()` as `1 - opponent_sgw`.

Proposed blending:

```python
# Player A's effective serve point-win rate against Player B
p_serve_a = (sgw_a + (1 - sgw_b)) / 2

# Player B's effective serve point-win rate against Player A  
p_serve_b = (sgw_b + (1 - sgw_a)) / 2
```

This is a starting point — a more sophisticated version could weight the blend differently or use surface-specific serve stats. The key property is that both players' abilities contribute to each serving scenario.

**Fallback for missing data:**

If a player has insufficient stat history (new to tour, surface-specific gaps), fall back to Elo-derived estimates:

```python
# Use Elo-expected win probability to anchor p_serve
# Average ATP point-win rate on serve is ~0.62-0.64
# Scale around that based on Elo advantage
elo_p = elo_expected(elo_a, elo_b)
p_serve_a = 0.60 + (elo_p - 0.50) * 0.15       # maps [0.3, 0.7] Elo range to ~[0.57, 0.63]
p_serve_b = 0.60 + ((1 - elo_p) - 0.50) * 0.15
```

These coefficients (0.60 baseline, 0.15 scaling) should be tuned against backtest data.

---

## 3. Game-Level Model (Closed-Form)

A service game is won when the server reaches 4 points with a 2-point lead at deuce. Given `p` = probability server wins a single service point, `q = 1 - p`:

```python
def p_hold_game(p: float) -> float:
    """Probability of holding serve given point-win rate p."""
    q = 1 - p
    # Probability of winning without reaching deuce
    p_no_deuce = (
        p**4                          # 4-0 (40-0, win)
        + 4 * p**4 * q               # 4-1 (40-15, win) — 4 paths
        + 10 * p**4 * q**2           # 4-2 (40-30, win) — 10 paths
    )
    # Probability of reaching deuce (3-3)
    p_deuce = 20 * p**3 * q**3
    # Probability of winning from deuce
    p_win_deuce = p**2 / (p**2 + q**2)
    
    return p_no_deuce + p_deuce * p_win_deuce
```

**Validation checkpoints:**
- `p_hold_game(0.60)` ≈ 0.736
- `p_hold_game(0.65)` ≈ 0.829
- `p_hold_game(0.70)` ≈ 0.901

This function is deterministic — no simulation needed at the game level.

---

## 4. Tiebreak Model (Simulation)

A tiebreak is played to 7 points with a 2-point lead, with serve alternating every 2 points after the first point. Given `p_a` (A's serve point-win rate) and `p_b` (B's serve point-win rate):

```python
def simulate_tiebreak(p_a: float, p_b: float, a_serves_first: bool = True) -> bool:
    """
    Simulate a single tiebreak. Returns True if Player A wins.
    
    Serve pattern: A serves point 1, then B serves points 2-3, 
    A serves points 4-5, B serves 6-7, etc.
    """
    score_a, score_b = 0, 0
    point_num = 0
    
    while True:
        # Determine who is serving
        if point_num == 0:
            a_serving = a_serves_first
        else:
            # After first point, serve alternates every 2 points
            a_serving = ((point_num - 1) // 2) % 2 == (0 if a_serves_first else 1)
        
        # Play point
        p_server_wins = p_a if a_serving else p_b
        server_wins = random.random() < p_server_wins
        
        if a_serving:
            if server_wins:
                score_a += 1
            else:
                score_b += 1
        else:
            if server_wins:
                score_b += 1
            else:
                score_a += 1
        
        point_num += 1
        
        # Check for tiebreak win: first to 7 with 2-point lead
        if score_a >= 7 and score_a - score_b >= 2:
            return True
        if score_b >= 7 and score_b - score_a >= 2:
            return False
```

---

## 5. Set-Level Model (Simulation)

A set alternates service games between players. A set is won when one player reaches 6 games with a 2-game lead, or via tiebreak at 6-6.

```python
def simulate_set(p_a: float, p_b: float, a_serves_first: bool = True) -> dict:
    """
    Simulate a single set. Returns dict with:
      - "winner": "A" or "B"
      - "score": (games_a, games_b)
      - "tiebreak": bool, whether set went to tiebreak
      - "a_serves_next": bool, who serves first in the next set
    """
    games_a, games_b = 0, 0
    a_serving = a_serves_first
    
    while True:
        # Play a service game
        if a_serving:
            hold = random.random() < p_hold_game(p_a)
            if hold:
                games_a += 1
            else:
                games_b += 1
        else:
            hold = random.random() < p_hold_game(p_b)
            if hold:
                games_b += 1
            else:
                games_a += 1
        
        a_serving = not a_serving  # alternate serve
        
        # Check for set win (6-x with 2-game lead, where x <= 4)
        if games_a >= 6 and games_a - games_b >= 2:
            return {"winner": "A", "score": (games_a, games_b), 
                    "tiebreak": False, "a_serves_next": a_serving}
        if games_b >= 6 and games_b - games_a >= 2:
            return {"winner": "B", "score": (games_a, games_b), 
                    "tiebreak": False, "a_serves_next": a_serving}
        
        # Tiebreak at 6-6
        if games_a == 6 and games_b == 6:
            a_wins_tb = simulate_tiebreak(p_a, p_b, a_serving)
            if a_wins_tb:
                return {"winner": "A", "score": (7, 6), 
                        "tiebreak": True, "a_serves_next": not a_serving}
            else:
                return {"winner": "B", "score": (6, 7), 
                        "tiebreak": True, "a_serves_next": not a_serving}
```

Note: After a tiebreak, the player who received first in the tiebreak serves first in the next set.

---

## 6. Match-Level Model (Simulation)

```python
def simulate_match(p_a: float, p_b: float, best_of: int = 3) -> dict:
    """
    Simulate a single match. Returns dict with:
      - "winner": "A" or "B"
      - "sets_a": int
      - "sets_b": int
      - "set_scores": list of (games_a, games_b) tuples
      - "total_games": int
    """
    sets_to_win = 2 if best_of == 3 else 3
    sets_a, sets_b = 0, 0
    set_scores = []
    a_serves_first = True  # or randomize with random.random() < 0.5
    
    while sets_a < sets_to_win and sets_b < sets_to_win:
        result = simulate_set(p_a, p_b, a_serves_first)
        set_scores.append(result["score"])
        if result["winner"] == "A":
            sets_a += 1
        else:
            sets_b += 1
        a_serves_first = result["a_serves_next"]
    
    total_games = sum(a + b for a, b in set_scores)
    
    return {
        "winner": "A",
        "sets_a": sets_a,
        "sets_b": sets_b,
        "set_scores": set_scores,
        "total_games": total_games,
    }
```

---

## 7. Monte Carlo Runner

```python
def run_simulation(
    p_serve_a: float,
    p_serve_b: float,
    best_of: int = 3,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> dict:
    """
    Run n_sims match simulations and aggregate results.
    
    Parameters
    ----------
    p_serve_a : float
        Player A's point-win probability on serve (opponent-adjusted).
    p_serve_b : float
        Player B's point-win probability on serve (opponent-adjusted).
    best_of : int
        3 or 5.
    n_sims : int
        Number of simulations to run. 10K gives ~1pp precision.
    seed : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    dict with keys:
        - "p_match_a": float, match win probability for A
        - "p_match_b": float, match win probability for B
        - "set_score_dist": dict mapping "X-Y" -> probability
            e.g., {"2-0": 0.49, "2-1": 0.25, "1-2": 0.17, "0-2": 0.09}
        - "avg_total_games": float
        - "avg_total_sets": float
        - "confidence": float, straight-set fraction of fav wins
        - "sim_std": float, standard error of p_match_a estimate
        - "p_serve_a": float, input echo
        - "p_serve_b": float, input echo
        - "n_sims": int
    """
```

**Implementation notes:**
- Use `random.seed(seed)` at the top for reproducibility
- Accumulate counts in a `Counter` for set score distributions
- `confidence` = `set_score_dist["2-0"] / p_match_a` (for Bo3) or `set_score_dist["3-0"] / p_match_a` (for Bo5). Higher values = more dominant favorite, suitable for fuller Kelly
- `sim_std` = `sqrt(p * (1-p) / n_sims)` — standard error of the binomial proportion
- 10K sims gives ±1pp precision at 95% confidence; 50K gives ±0.4pp if needed
- Consider using numpy for vectorized simulation if performance is an issue, but a pure-Python loop over 10K sims should complete in <1 second

---

## 8. Output Structure

For a Bo3 match, Sinner (A) vs Cerundolo (B), with `p_serve_a = 0.66`, `p_serve_b = 0.61`:

```python
{
    "p_match_a":      0.738,
    "p_match_b":      0.262,
    
    "set_score_dist": {
        "2-0": 0.491,      # A wins in straight sets
        "2-1": 0.247,      # A wins in 3 sets
        "1-2": 0.168,      # B wins in 3 sets
        "0-2": 0.094,      # B wins in straight sets
    },
    
    "avg_total_games": 19.7,
    "avg_total_sets":  2.42,
    
    "confidence":      0.665,   # 0.491 / 0.738 — 66.5% of A's wins are in straights
    "sim_std":         0.004,
    
    "p_serve_a":       0.66,
    "p_serve_b":       0.61,
    "n_sims":          10000,
}
```

---

## 9. Integration with `live.py`

### Replacing `comp_p1`

The simulation's `p_match_a` replaces (or supplements) the current `comp_p1` in the prediction pipeline:

```python
def predict(p1_name, p2_name, surface, best_of, market_p1,
            elo_ratings, surf_elo_by_name, rank_pts_map, get_stats, 
            last_date, h2h_map, today):
    
    # ... existing code to fetch stats, Elo, etc. ...
    
    s1 = get_stats(p1_name, surface)
    s2 = get_stats(p2_name, surface)
    
    # Derive opponent-adjusted serve probabilities
    p_serve_1 = (s1["sgw"] + (1 - s2["sgw"])) / 2
    p_serve_2 = (s2["sgw"] + (1 - s1["sgw"])) / 2
    
    # Run simulation
    sim = run_simulation(p_serve_1, p_serve_2, best_of=best_of, n_sims=10_000)
    
    # sim["p_match_a"] is the new mechanistic match win probability
    # Optionally blend with existing comp_p1 during transition period:
    # final_p1 = 0.5 * old_comp_p1 + 0.5 * sim["p_match_a"]
    
    # Or replace outright:
    raw_p1 = sim["p_match_a"]
    
    # Apply H2H shrinkage (same as current)
    raw_p1 = h2h_shrink(raw_p1, h2h_wins_p1, h2h_entry[1])
    
    # ... rest of existing logic ...
```

### Blending with existing signals

The serve-stats-derived simulation captures serve/return dynamics well but misses broader signals like Elo (career trajectory, consistency) and rank points (recent tournament performance). Two options:

**Option A — Simulation as primary, Elo as adjustment:**
Use `p_match_a` from the simulation as the base, then apply a small Elo-based adjustment (±2-3pp) when Elo diverges significantly from the simulation's estimate. This treats Elo as a "something the serve stats aren't capturing" correction.

**Option B — Weighted ensemble:**
Blend simulation output with the existing composite:
```python
final_p1 = w_sim * sim["p_match_a"] + w_comp * old_comp_p1
```
where `w_sim` and `w_comp` are weights that can be tuned via backtest. Start with 60/40 sim/composite and optimize.

Recommendation: Start with Option B during the transition period so you can validate the simulation against backtest data before fully replacing the composite. Once calibrated, move toward Option A.

### Display changes

Extend the `display()` function to show simulation output:

```
║  SIM MODEL             73.8%  /  26.2%           ║
║    Straight sets (2-0)   49.1%                    ║
║    Three sets (2-1)      24.7%                    ║
║    Confidence            66.5% (straights/wins)   ║
║    Avg total games       19.7                     ║
```

### Confidence-adjusted sizing

The `confidence` field from the simulation output can modulate the Kelly fraction in `size_bet()`:

```python
# In size_bet(), after computing kelly_frac:
if sim_confidence is not None:
    # Scale Kelly fraction: high confidence (>0.65) gets full fraction,
    # low confidence (<0.50) gets reduced
    conf_multiplier = 0.7 + 0.6 * min(max(sim_confidence - 0.40, 0), 0.30)
    # Maps: 0.40 confidence -> 0.7x, 0.55 -> 0.79x, 0.70 -> 1.0x
    kelly_frac *= conf_multiplier
```

---

## 10. Calibration & Validation

### Backtest procedure

Run the simulation on historical matches from `backtest.py` data and compare:
1. **Accuracy**: Does `p_match_a` from simulation predict outcomes better than `comp_p1`?
   Metric: Brier score. Lower is better.
2. **Calibration**: When the sim says 70%, does the favorite win ~70% of the time?
   Plot calibration curves for both models.
3. **Set score accuracy**: Does the predicted set score distribution match observed frequencies?
   This validates the game-level model, not just the match outcome.

### Known limitations to test for

- **Surface effects on serve dynamics**: Clay reduces ace rates and hold percentages vs hard court. The `p_serve` inputs should be surface-specific (which they will be if `get_stats()` is called with the correct surface). Validate that the simulation produces realistic set score distributions per surface.
- **Fatigue / momentum**: The simulation treats each point as independent (iid Bernoulli). Real matches have momentum shifts, fatigue in 3rd/5th sets, etc. This is a known simplification. If backtest shows the sim underestimates 3-set matches, consider adding a small "break-back boost" or "fatigue discount" in later sets.
- **Serve-stat sparsity**: Players with <20 matches of stat history will have noisy `sgw` estimates. The existing `STAT_PRIOR` shrinkage in `_shrink()` helps, but the simulation amplifies any bias in the inputs through the nonlinear scoring chain. Monitor calibration specifically for players with thin stat histories.

---

## 11. Performance Considerations

- 10K sims × ~20 games/match × point-level resolution ≈ 200K random draws per prediction
- Pure Python: ~0.5-1.0 seconds per match prediction
- NumPy vectorized: ~50-100ms per match prediction
- For CLI interactive use, pure Python is fine
- For batch/automated mode evaluating 16+ matches, consider NumPy vectorization

Start with pure Python for clarity, optimize later if batch mode needs it.

---

## 12. File Structure

Recommended: create `simulation.py` as a new module alongside `live.py`:

```
├── live.py              # existing — calls simulation.py
├── simulation.py        # NEW — game/set/match simulation engine
├── sizing.py            # NEW (from bet sizing spec) — position sizing
├── backtest.py          # existing — add simulation validation
├── player_elos.json     # existing
```

`simulation.py` exports: `p_hold_game()`, `simulate_tiebreak()`, `simulate_set()`, `simulate_match()`, `run_simulation()`

`live.py` imports `run_simulation` and calls it inside `predict()`.
