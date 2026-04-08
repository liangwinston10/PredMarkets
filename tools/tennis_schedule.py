"""
Tennis Tournament Schedule — 2026 ATP/WTA 1000 + Grand Slams
Static calendar (hardcoded — ATP/WTA publish the full year in advance).
Provides get_upcoming_schedule() for use by the dashboard.
"""

import pandas as pd

# ── 2026 Grand Slams ───────────────────────────────────────────────────────────
GRAND_SLAMS_2026 = [
    {
        "name": "Australian Open",
        "tour": "ATP/WTA",
        "start_date": "2026-01-19",
        "end_date":   "2026-02-01",
        "surface":    "Hard",
        "draw_size":  128,
        "prize_money_usd": 86_500_000,
        "category":   "Grand Slam",
        "location":   "Melbourne, Australia",
        "confirmed":  True,
    },
    {
        "name": "Roland Garros",
        "tour": "ATP/WTA",
        "start_date": "2026-05-25",
        "end_date":   "2026-06-07",
        "surface":    "Clay",
        "draw_size":  128,
        "prize_money_usd": 56_000_000,
        "category":   "Grand Slam",
        "location":   "Paris, France",
        "confirmed":  True,
    },
    {
        "name": "Wimbledon",
        "tour": "ATP/WTA",
        "start_date": "2026-06-29",
        "end_date":   "2026-07-12",
        "surface":    "Grass",
        "draw_size":  128,
        "prize_money_usd": 50_000_000,
        "category":   "Grand Slam",
        "location":   "London, United Kingdom",
        "confirmed":  True,
    },
    {
        "name": "US Open",
        "tour": "ATP/WTA",
        "start_date": "2026-08-31",
        "end_date":   "2026-09-13",
        "surface":    "Hard",
        "draw_size":  128,
        "prize_money_usd": 65_000_000,
        "category":   "Grand Slam",
        "location":   "New York, USA",
        "confirmed":  True,
    },
]

# ── 2026 ATP Masters 1000 ──────────────────────────────────────────────────────
ATP_MASTERS_2026 = [
    {
        "name": "Indian Wells Masters",
        "tour": "ATP",
        "start_date": "2026-03-09",
        "end_date":   "2026-03-22",
        "surface":    "Hard",
        "draw_size":  96,
        "prize_money_usd": 9_007_450,
        "category":   "Masters 1000",
        "location":   "Indian Wells, USA",
        "confirmed":  True,
    },
    {
        "name": "Miami Open",
        "tour": "ATP",
        "start_date": "2026-03-23",
        "end_date":   "2026-04-05",
        "surface":    "Hard",
        "draw_size":  96,
        "prize_money_usd": 9_007_450,
        "category":   "Masters 1000",
        "location":   "Miami, USA",
        "confirmed":  True,
    },
    {
        "name": "Monte-Carlo Masters",
        "tour": "ATP",
        "start_date": "2026-04-12",
        "end_date":   "2026-04-19",
        "surface":    "Clay",
        "draw_size":  56,
        "prize_money_usd": 6_127_250,
        "category":   "Masters 1000",
        "location":   "Monte-Carlo, Monaco",
        "confirmed":  True,
    },
    {
        "name": "Madrid Open",
        "tour": "ATP",
        "start_date": "2026-04-26",
        "end_date":   "2026-05-10",
        "surface":    "Clay",
        "draw_size":  64,
        "prize_money_usd": 9_007_450,
        "category":   "Masters 1000",
        "location":   "Madrid, Spain",
        "confirmed":  True,
    },
    {
        "name": "Italian Open",
        "tour": "ATP",
        "start_date": "2026-05-11",
        "end_date":   "2026-05-24",
        "surface":    "Clay",
        "draw_size":  96,
        "prize_money_usd": 9_007_450,
        "category":   "Masters 1000",
        "location":   "Rome, Italy",
        "confirmed":  True,
    },
    {
        "name": "Canadian Open",
        "tour": "ATP",
        "start_date": "2026-08-10",
        "end_date":   "2026-08-16",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 6_661_825,
        "category":   "Masters 1000",
        "location":   "Montreal, Canada",
        "confirmed":  True,
    },
    {
        "name": "Western & Southern Open",
        "tour": "ATP",
        "start_date": "2026-08-17",
        "end_date":   "2026-08-23",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 6_661_825,
        "category":   "Masters 1000",
        "location":   "Cincinnati, USA",
        "confirmed":  True,
    },
    {
        "name": "Shanghai Masters",
        "tour": "ATP",
        "start_date": "2026-10-05",
        "end_date":   "2026-10-12",
        "surface":    "Hard",
        "draw_size":  96,
        "prize_money_usd": 8_800_000,
        "category":   "Masters 1000",
        "location":   "Shanghai, China",
        "confirmed":  True,
    },
    {
        "name": "Paris Masters",
        "tour": "ATP",
        "start_date": "2026-10-26",
        "end_date":   "2026-11-01",
        "surface":    "Hard (Indoor)",
        "draw_size":  48,
        "prize_money_usd": 5_980_000,
        "category":   "Masters 1000",
        "location":   "Paris, France",
        "confirmed":  True,
    },
]

# ── 2026 WTA 1000 ──────────────────────────────────────────────────────────────
WTA_1000_2026 = [
    {
        "name": "Dubai Tennis Championships",
        "tour": "WTA",
        "start_date": "2026-02-16",
        "end_date":   "2026-02-22",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 2_750_000,
        "category":   "WTA 1000",
        "location":   "Dubai, UAE",
        "confirmed":  True,
    },
    {
        "name": "Indian Wells (WTA)",
        "tour": "WTA",
        "start_date": "2026-03-09",
        "end_date":   "2026-03-22",
        "surface":    "Hard",
        "draw_size":  96,
        "prize_money_usd": 9_007_450,
        "category":   "WTA 1000",
        "location":   "Indian Wells, USA",
        "confirmed":  True,
    },
    {
        "name": "Miami Open (WTA)",
        "tour": "WTA",
        "start_date": "2026-03-23",
        "end_date":   "2026-04-05",
        "surface":    "Hard",
        "draw_size":  96,
        "prize_money_usd": 9_007_450,
        "category":   "WTA 1000",
        "location":   "Miami, USA",
        "confirmed":  True,
    },
    {
        "name": "Madrid Open (WTA)",
        "tour": "WTA",
        "start_date": "2026-04-26",
        "end_date":   "2026-05-10",
        "surface":    "Clay",
        "draw_size":  64,
        "prize_money_usd": 9_007_450,
        "category":   "WTA 1000",
        "location":   "Madrid, Spain",
        "confirmed":  True,
    },
    {
        "name": "Italian Open (WTA)",
        "tour": "WTA",
        "start_date": "2026-05-11",
        "end_date":   "2026-05-17",
        "surface":    "Clay",
        "draw_size":  56,
        "prize_money_usd": 4_115_012,
        "category":   "WTA 1000",
        "location":   "Rome, Italy",
        "confirmed":  True,
    },
    {
        "name": "Canadian Open (WTA)",
        "tour": "WTA",
        "start_date": "2026-08-10",
        "end_date":   "2026-08-16",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 4_115_012,
        "category":   "WTA 1000",
        "location":   "Toronto, Canada",
        "confirmed":  True,
    },
    {
        "name": "Western & Southern Open (WTA)",
        "tour": "WTA",
        "start_date": "2026-08-17",
        "end_date":   "2026-08-23",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 4_115_012,
        "category":   "WTA 1000",
        "location":   "Cincinnati, USA",
        "confirmed":  True,
    },
    {
        "name": "China Open",
        "tour": "WTA",
        "start_date": "2026-09-28",
        "end_date":   "2026-10-04",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 8_800_000,
        "category":   "WTA 1000",
        "location":   "Beijing, China",
        "confirmed":  True,
    },
    {
        "name": "Wuhan Open",
        "tour": "WTA",
        "start_date": "2026-10-12",
        "end_date":   "2026-10-18",
        "surface":    "Hard",
        "draw_size":  56,
        "prize_money_usd": 3_226_110,
        "category":   "WTA 1000",
        "location":   "Wuhan, China",
        "confirmed":  True,
    },
]

ALL_EVENTS_2026 = GRAND_SLAMS_2026 + ATP_MASTERS_2026 + WTA_1000_2026


def get_upcoming_schedule(days_ahead: int = 90, include_completed: bool = False) -> pd.DataFrame:
    """
    Return tournament schedule DataFrame filtered to the next N days.
    Columns: name, tour, start_date, end_date, surface, draw_size,
             prize_money_usd, category, location, status
    """
    today   = pd.Timestamp.today().normalize()
    cutoff  = today + pd.Timedelta(days=days_ahead)

    df = pd.DataFrame(ALL_EVENTS_2026)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])

    def _status(row: pd.Series) -> str:
        if row["start_date"] <= today <= row["end_date"]:
            return "Live"
        if row["start_date"] > today:
            return "Upcoming"
        return "Completed"

    df["status"] = df.apply(_status, axis=1)

    if not include_completed:
        df = df[df["status"] != "Completed"]

    mask = (df["end_date"] >= today) & (df["start_date"] <= cutoff)
    df = df[mask | (df["status"] == "Live")]

    return df.sort_values("start_date").reset_index(drop=True)


def get_live_tournaments() -> list[dict]:
    """Return any tournaments currently in progress."""
    df = get_upcoming_schedule(days_ahead=0, include_completed=False)
    live = df[df["status"] == "Live"]
    return live.to_dict("records")


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_upcoming_schedule(days_ahead=180)
    print(f"Upcoming events (next 180 days): {len(df)}")
    print(df[["name", "tour", "start_date", "surface", "category", "status"]].to_string(index=False))
    live = get_live_tournaments()
    if live:
        print(f"\nLive now: {[t['name'] for t in live]}")
    else:
        print("\nNo tournaments live right now.")
