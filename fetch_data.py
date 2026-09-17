"""
NBA Game Outcome Predictor — Phase 1: Data Collection
--------------------------------------------------------
Pulls regular-season game logs for multiple NBA seasons using nba_api
(a free Python wrapper around stats.nba.com) and saves them locally.

IMPORTANT: this saves RAW data — one row per team per game, including
that game's own final stats. We deliberately do NOT use those final
stats as prediction features later; that would be data leakage (using
information from the future to predict the past). Phase 2 will turn
this raw data into leakage-safe features (rolling averages of *prior*
games only).

Setup:
    pip install nba_api pandas

Run:
    python fetch_data.py
"""

import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Which seasons to pull. nba_api uses "YYYY-YY" format for seasons.
# Start with a few recent seasons — easy to add more later.
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

OUTPUT_FILE = "data/raw_games.csv"


def fetch_season(season):
    """
    Fetches every regular-season game for one season.
    Returns a DataFrame with one row PER TEAM PER GAME — meaning each
    actual game appears as two rows (one for each team), which makes
    building "team A vs team B" matchups straightforward later.
    """
    print(f"Fetching {season}...")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    return df


def main():
    all_seasons = []

    for season in SEASONS:
        try:
            df = fetch_season(season)
            all_seasons.append(df)
            print(f"  -> got {len(df)} rows")
        except Exception as e:
            print(f"  -> FAILED for {season}: {e}")

        # Be polite to NBA.com's servers — avoid hammering the API
        # with back-to-back requests.
        time.sleep(1)

    if not all_seasons:
        print("No data fetched — check your internet connection and try again.")
        return

    combined = pd.concat(all_seasons, ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(combined)} total rows to {OUTPUT_FILE}")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
