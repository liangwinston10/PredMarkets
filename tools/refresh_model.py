"""
Refresh ATP and WTA Elo ratings + rolling stats by re-running both backtests.
Pulls latest match data from Sackmann / TML GitHub repos.

Usage:
    python tools/refresh_model.py
"""

import sys
import os
import subprocess
import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATP_DIR   = os.path.join(REPO_ROOT, "tennis_model")
WTA_DIR   = os.path.join(REPO_ROOT, "tennis_model", "wta")
PYTHON    = sys.executable


def run(label: str, script: str, cwd: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [PYTHON, script],
        cwd=cwd,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ERROR: {label} exited with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"Refresh started: {start.strftime('%Y-%m-%d %H:%M')}")

    atp_rc = run("ATP backtest  →  player_elos.json", "backtest.py", ATP_DIR)
    wta_rc = run("WTA backtest  →  player_elos.json", "backtest.py", WTA_DIR)

    elapsed = (datetime.datetime.now() - start).seconds
    print(f"\nDone in {elapsed}s  |  ATP: {'OK' if atp_rc == 0 else 'FAIL'}  |  WTA: {'OK' if wta_rc == 0 else 'FAIL'}")
