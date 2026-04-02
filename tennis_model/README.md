# Tennis Composite Win% Model

## Setup
pip install -r requirements.txt

## Step 1 — run the backtest (builds player_elos.json)
python backtest.py

## Step 2 — live predictions
python live.py

## Model weights (edit in backtest.py / live.py)
WEIGHTS = {
    "serve":  0.25,
    "return": 0.25,
    "bp":     0.15,
    "form":   0.20,
    "elo":    0.15,
}
