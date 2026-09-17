"""
build_roster_aware_features.py

Purpose
-------
Your existing raw_games.csv is TEAM-LEVEL: one row per team per game, with
box-score totals. It has no concept of *which players* were on the roster,
so it can't represent a trade like Giannis Antetokounmpo -> Miami Heat until
enough games have actually been played with him there.

This script fixes that by:
  1. Pulling PLAYER-level game logs (nba_api) for the same seasons, to know
     WHO was on each roster for each game.
  2. Computing a rolling, form-weighted player-impact rating per player
     (trailing-N-game average PLUS_MINUS, shifted to avoid leakage).
  3. Aggregating those ratings by team + game, so a roster change shows up
     as an immediate shift in team strength -- Giannis's rolling impact
     score follows him into Miami's aggregation the moment he's on their
     roster, with no dependency on Miami's past team-level history.
  4. Adding home/away, rest days, back-to-back flags, recent form (rolling
     win %), travel distance since the last game, and opponent-side
     versions of all of it (a team's strength only means something
     relative to who and what they're facing that night).
  5. Pulling CURRENT Vegas odds (The Odds API) and converting to implied
     win probability, since you only need this for predicting the game
     that's about to happen (not backfilling historical rows).
  6. Merging everything onto your raw_games.csv keyed on GAME_ID + TEAM_ID.

Run this LOCALLY (not in this sandbox) since it needs live access to
stats.nba.com and api.the-odds-api.com.

Install:
    pip install nba_api pandas requests
"""

import time
import datetime
import requests
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
def get_current_season_str(today=None):
    """
    NBA season labels flip in October, e.g. 2026-27 season starts Oct 2026.
    This auto-derives the current season string so you never hardcode it.
    """
    today = today or datetime.date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# Historical seasons stay fixed; current season is derived automatically
# so this list never needs manual updates.
HISTORICAL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
SEASONS = HISTORICAL_SEASONS + [get_current_season_str()]

ROLLING_WINDOW = 10          # trailing games used for "current form" rating
ODDS_API_KEY = "YOUR_ODDS_API_KEY_HERE"   # https://the-odds-api.com  (free tier: 500 req/mo)
RAW_GAMES_PATH = "/Users/bkobell33/Documents/nba-predictor/data/raw_games.csv"
OUTPUT_PATH = "/Users/bkobell33/Documents/nba-predictor/data/games_with_roster_features.csv"
PLAYER_LOGS_CACHE_PATH = "/Users/bkobell33/Documents/nba-predictor/data/player_logs_cache.csv"


# ---------------------------------------------------------------------------
# STEP 1: Pull player-level game logs
# ---------------------------------------------------------------------------
def pull_player_game_logs(seasons, cache_path=PLAYER_LOGS_CACHE_PATH):
    """
    Player-level game logs, one row per player per game.
    This is what lets us know WHO was on a roster on a given date,
    independent of the fixed team-level history.

    Caches HISTORICAL_SEASONS to disk since those never change once a
    season ends -- only the current season needs refetching each run.
    This also means adding a new season next year requires no code
    changes: get_current_season_str() picks it up automatically, and
    last year's "current" season just becomes historical and gets cached.
    """
    try:
        cached = pd.read_csv(cache_path)
        cached_seasons = set(cached["SEASON_YEAR"].unique())
    except FileNotFoundError:
        cached = pd.DataFrame()
        cached_seasons = set()

    current_season = get_current_season_str()
    all_logs = [cached] if not cached.empty else []

    for season in seasons:
        # Skip re-pulling a completed season we already have cached.
        # Always re-pull the current season since it's still in progress.
        if season in cached_seasons and season != current_season:
            continue
        print(f"Pulling player logs for {season}...")
        logs = playergamelogs.PlayerGameLogs(season_nullable=season)
        df = logs.get_data_frames()[0]
        df["SEASON_YEAR"] = season
        all_logs.append(df)
        time.sleep(1)  # be polite to the (unofficial) endpoint

    combined = pd.concat(all_logs, ignore_index=True)
    # CRITICAL: normalize GAME_ID to a consistent zero-padded string BEFORE
    # deduplicating. Fresh API pulls return GAME_ID as a string like
    # "0022500001"; once that round-trips through this CSV cache and gets
    # re-read, pandas infers it as a plain int (22500001, leading zeros
    # dropped). Without this normalization, the same real game ends up
    # represented two different ways across cached vs. freshly-pulled rows,
    # drop_duplicates fails to recognize them as duplicates, and a single
    # game's roster silently gets split into two separate groups downstream.
    combined["GAME_ID"] = combined["GAME_ID"].astype(str).str.zfill(10)
    combined = combined.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
    combined.to_csv(cache_path, index=False)
    return combined


# ---------------------------------------------------------------------------
# STEP 2: Rolling player-impact rating
# ---------------------------------------------------------------------------
def compute_player_impact(player_logs, window=ROLLING_WINDOW):
    """
    A simple, transparent impact metric: trailing-N-game average PLUS_MINUS,
    computed per player, sorted by date, shifted by 1 game to avoid leakage
    (only using info known BEFORE the game being predicted).
    """
    player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"])
    player_logs["ROLLING_IMPACT"] = (
        player_logs.groupby("PLAYER_ID")["PLUS_MINUS"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    return player_logs


# ---------------------------------------------------------------------------
# STEP 3: Aggregate to team-by-game roster strength
# ---------------------------------------------------------------------------
def aggregate_team_roster_strength(player_logs):
    """
    For each TEAM_ID + GAME_ID, sum minutes-weighted rolling impact across
    everyone who actually played. This is the piece that makes a trade
    show up immediately: Giannis's rolling impact score follows him into
    MIA's aggregation the moment he suits up for them, with no dependency
    on Miami's past team-level history.
    """
    player_logs["WEIGHTED_IMPACT"] = (
        player_logs["ROLLING_IMPACT"] * player_logs["MIN"]
    )
    team_strength = (
        player_logs.groupby(["GAME_ID", "TEAM_ID"])
        .agg(
            ROSTER_IMPACT_SUM=("WEIGHTED_IMPACT", "sum"),
            ROSTER_IMPACT_AVG=("ROLLING_IMPACT", "mean"),
            TOP_PLAYER_IMPACT=("ROLLING_IMPACT", "max"),
            PLAYERS_USED=("PLAYER_ID", "nunique"),
        )
        .reset_index()
    )
    return team_strength


# ---------------------------------------------------------------------------
# STEP 4: Home/away, rest days, back-to-backs, and opponent strength
# ---------------------------------------------------------------------------
def compute_rest_and_home_features(raw_games):
    """
    Adds, per team per game:
      - IS_HOME: derived from MATCHUP ("vs." = home, "@" = away)
      - DAYS_REST: days since that team's previous game. First game in
        the dataset for a team has no prior game, so this is filled with
        LEAGUE_AVG_REST_DAYS (2) rather than left as NaN.
      - IS_BACK_TO_BACK: DAYS_REST == 1 (played yesterday, no rest day)
      - IS_LONG_REST: DAYS_REST >= 3 (proxy for "well-rested," including
        the small number of true long layoffs, e.g. All-Star break)

    Rest and schedule spot are some of the most reliable, cheap signals
    in game-margin models -- back-to-backs alone shift scoring margin by
    a few points on average across the league.
    """
    raw_games = raw_games.copy()
    raw_games["GAME_DATE"] = pd.to_datetime(raw_games["GAME_DATE"])
    raw_games["IS_HOME"] = raw_games["MATCHUP"].str.contains("vs.", regex=False).astype(int)

    raw_games = raw_games.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    # Group by TEAM_ID *and* SEASON, not just TEAM_ID -- otherwise a team's
    # season opener gets DAYS_REST computed as the raw calendar gap since
    # their LAST GAME OF THE PREVIOUS SEASON (the whole offseason, often
    # 150-200+ days), rather than being treated as a fresh start. Grouping
    # by season too means .diff() correctly returns NaN at every season
    # boundary, which then gets filled with LEAGUE_AVG_REST_DAYS below.
    raw_games["DAYS_REST"] = (
        raw_games.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].diff().dt.days
    )
    # First game of each team's each season has no prior in-season game to
    # diff against, so DAYS_REST is NaN there. Fill with the league-average
    # rest (~2 days) rather than leaving it as NaN, which would silently
    # propagate into IS_BACK_TO_BACK (False) and REST_DIFF (NaN) downstream.
    LEAGUE_AVG_REST_DAYS = 2
    raw_games["DAYS_REST"] = raw_games["DAYS_REST"].fillna(LEAGUE_AVG_REST_DAYS)

    raw_games["IS_BACK_TO_BACK"] = (raw_games["DAYS_REST"] == 1).astype(int)
    raw_games["IS_LONG_REST"] = (raw_games["DAYS_REST"] >= 3).astype(int)

    return raw_games


def add_opponent_features(merged):
    """
    Every row so far describes one team's own strength/rest. For a margin
    or win-probability model, what matters just as much is the OPPONENT's
    strength and rest on the same night -- a team's roster impact means
    little without knowing who's standing across from them.

    This self-joins the merged frame against itself on GAME_ID, matching
    each row to the *other* team in that same game, and pulls over their
    roster-strength and rest columns with an OPP_ prefix.
    """
    opponent_cols = [
        "GAME_ID", "TEAM_ID", "ROSTER_IMPACT_SUM", "ROSTER_IMPACT_AVG",
        "TOP_PLAYER_IMPACT", "DAYS_REST", "IS_BACK_TO_BACK", "IS_HOME",
        "RECENT_WIN_PCT", "TRAVEL_DISTANCE_MILES",
    ]
    opponent_side = merged[opponent_cols].rename(
        columns={c: f"OPP_{c}" for c in opponent_cols if c != "GAME_ID"}
    )

    merged = merged.merge(
        opponent_side,
        on="GAME_ID",
        how="left",
    )
    # Drop the self-match (a team paired with itself) -- keep only rows
    # where the "opponent" row is a different TEAM_ID than the team row.
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    merged = merged.drop(columns=["OPP_TEAM_ID"])

    # Convenience differential features -- often more predictive than
    # either team's raw value alone.
    merged["IMPACT_DIFF"] = merged["ROSTER_IMPACT_SUM"] - merged["OPP_ROSTER_IMPACT_SUM"]
    merged["REST_DIFF"] = merged["DAYS_REST"] - merged["OPP_DAYS_REST"]
    merged["FORM_DIFF"] = merged["RECENT_WIN_PCT"] - merged["OPP_RECENT_WIN_PCT"]
    merged["TRAVEL_DIFF"] = merged["TRAVEL_DISTANCE_MILES"] - merged["OPP_TRAVEL_DISTANCE_MILES"]

    return merged


# ---------------------------------------------------------------------------
# STEP 4b: Recent form (rolling win %) and travel distance
# ---------------------------------------------------------------------------
def compute_recent_form(raw_games, window=10):
    """
    Trailing win percentage over the last `window` games, shifted by 1 to
    avoid leakage (only using results known BEFORE the game being
    predicted). Captures momentum/hot-streak effects that are separate
    from roster strength -- two teams with identical rosters can still be
    playing very differently depending on recent form.
    """
    raw_games = raw_games.copy()
    raw_games["WIN"] = (raw_games["WL"] == "W").astype(int)
    raw_games = raw_games.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    raw_games["RECENT_WIN_PCT"] = (
        raw_games.groupby(["TEAM_ID", "SEASON"])["WIN"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    # Early-season games with < 3 prior games get NaN -- fill with .500
    # (a neutral assumption: no evidence yet this team is hot or cold).
    raw_games["RECENT_WIN_PCT"] = raw_games["RECENT_WIN_PCT"].fillna(0.5)
    return raw_games


# Approximate arena coordinates (lat, lon) for each team, by TEAM_ID.
# Static reference data -- doesn't change year to year, so safe to hardcode.
TEAM_COORDS = {
    1610612737: (33.7573, -84.3963),   # Atlanta Hawks
    1610612738: (42.3663, -71.0622),   # Boston Celtics
    1610612751: (40.6826, -73.9754),   # Brooklyn Nets
    1610612766: (35.2251, -80.8392),   # Charlotte Hornets
    1610612741: (41.8807, -87.6742),   # Chicago Bulls
    1610612739: (41.4965, -81.6882),   # Cleveland Cavaliers
    1610612742: (32.7905, -96.8103),   # Dallas Mavericks
    1610612743: (39.7487, -105.0077),  # Denver Nuggets
    1610612765: (42.3410, -83.0552),   # Detroit Pistons
    1610612744: (37.7680, -122.3877),  # Golden State Warriors
    1610612745: (29.7508, -95.3621),   # Houston Rockets
    1610612754: (39.7640, -86.1555),   # Indiana Pacers
    1610612746: (34.0430, -118.2673),  # LA Clippers
    1610612747: (34.0430, -118.2673),  # LA Lakers
    1610612763: (35.1382, -90.0505),   # Memphis Grizzlies
    1610612748: (25.7814, -80.1870),   # Miami Heat
    1610612749: (43.0451, -87.9172),   # Milwaukee Bucks
    1610612750: (44.9795, -93.2760),   # Minnesota Timberwolves
    1610612740: (29.9490, -90.0821),   # New Orleans Pelicans
    1610612752: (40.7505, -73.9934),   # New York Knicks
    1610612760: (35.4634, -97.5151),   # Oklahoma City Thunder
    1610612753: (28.5392, -81.3839),   # Orlando Magic
    1610612755: (39.9012, -75.1720),   # Philadelphia 76ers
    1610612756: (33.4457, -112.0712),  # Phoenix Suns
    1610612757: (45.5316, -122.6668),  # Portland Trail Blazers
    1610612758: (38.6497, -121.5181),  # Sacramento Kings
    1610612759: (29.4269, -98.4375),   # San Antonio Spurs
    1610612761: (43.6435, -79.3791),   # Toronto Raptors
    1610612762: (40.7683, -111.9011),  # Utah Jazz
    1610612764: (38.8981, -77.0209),   # Washington Wizards
}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in miles."""
    R = 3958.8  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_travel_distance(raw_games):
    """
    Distance traveled since the previous game, in miles. A game's actual
    location is always the HOME team's arena -- so first we need to know,
    for every game, which TEAM_ID was home, then look up that team's
    coordinates as the game's location, then measure how far each team
    had to travel from their previous game's location to get there.

    This adds real signal a rest-days-only feature misses: a team flying
    coast-to-coast on a "normal" 2-day rest is in a different situation
    than a team playing a short in-city trip on the same amount of rest.
    """
    raw_games = raw_games.copy()

    # Determine each game's location: the home team's coordinates.
    home_team_per_game = (
        raw_games[raw_games["IS_HOME"] == 1][["GAME_ID", "TEAM_ID"]]
        .rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
    )
    raw_games = raw_games.merge(home_team_per_game, on="GAME_ID", how="left")
    raw_games["GAME_LAT"] = raw_games["HOME_TEAM_ID"].map(lambda t: TEAM_COORDS.get(t, (np.nan, np.nan))[0])
    raw_games["GAME_LON"] = raw_games["HOME_TEAM_ID"].map(lambda t: TEAM_COORDS.get(t, (np.nan, np.nan))[1])

    # For each team, find where their PREVIOUS game was played.
    raw_games = raw_games.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
    raw_games["PREV_GAME_LAT"] = raw_games.groupby(["TEAM_ID", "SEASON"])["GAME_LAT"].shift(1)
    raw_games["PREV_GAME_LON"] = raw_games.groupby(["TEAM_ID", "SEASON"])["GAME_LON"].shift(1)

    raw_games["TRAVEL_DISTANCE_MILES"] = haversine_miles(
        raw_games["PREV_GAME_LAT"], raw_games["PREV_GAME_LON"],
        raw_games["GAME_LAT"], raw_games["GAME_LON"],
    )
    # First game of each season has no previous location -- fill with 0
    # (treat season openers as "no unusual travel burden" by default).
    raw_games["TRAVEL_DISTANCE_MILES"] = raw_games["TRAVEL_DISTANCE_MILES"].fillna(0)

    raw_games = raw_games.drop(columns=["HOME_TEAM_ID", "GAME_LAT", "GAME_LON", "PREV_GAME_LAT", "PREV_GAME_LON"])
    return raw_games


# ---------------------------------------------------------------------------
# STEP 5: Vegas odds (current season only -- see note in module docstring)
# ---------------------------------------------------------------------------
def pull_current_odds(api_key=ODDS_API_KEY):
    """
    Pulls current NBA odds from The Odds API and converts to implied
    win probability. Only covers upcoming/live games -- free tier has
    no 5-year historical archive. Use this to enrich predictions for
    games going forward (e.g. Heat's 2026-27 season), not to backfill
    raw_games.csv.
    """
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",   # moneyline
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for game in data:
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    american_odds = outcome["price"]
                    implied_prob = american_to_implied_prob(american_odds)
                    rows.append({
                        "GAME_DATE": game["commence_time"][:10],
                        "TEAM_NAME": outcome["name"],
                        "BOOKMAKER": bookmaker["title"],
                        "AMERICAN_ODDS": american_odds,
                        "IMPLIED_WIN_PROB": implied_prob,
                    })
    odds_df = pd.DataFrame(rows)
    # average across bookmakers for a de-vigged-ish consensus number
    odds_summary = (
        odds_df.groupby(["GAME_DATE", "TEAM_NAME"])["IMPLIED_WIN_PROB"]
        .mean()
        .reset_index()
    )
    return odds_summary


def american_to_implied_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


# ---------------------------------------------------------------------------
# STEP 6: Merge onto raw_games.csv
# ---------------------------------------------------------------------------
def merge_features(raw_games_path=RAW_GAMES_PATH):
    raw_games = pd.read_csv(raw_games_path)
    raw_games = compute_rest_and_home_features(raw_games)
    raw_games = compute_recent_form(raw_games)
    raw_games = compute_travel_distance(raw_games)

    player_logs = pull_player_game_logs(SEASONS)
    player_logs = compute_player_impact(player_logs)
    team_strength = aggregate_team_roster_strength(player_logs)

    # nba_api's GAME_ID comes back as a zero-padded string (e.g. "0022100002"),
    # while pandas often infers raw_games.csv's GAME_ID as a plain int64
    # (silently dropping the leading zero). Force both to the same
    # zero-padded string format so the merge key actually matches.
    raw_games["GAME_ID"] = raw_games["GAME_ID"].astype(str).str.zfill(10)
    team_strength["GAME_ID"] = team_strength["GAME_ID"].astype(str).str.zfill(10)

    merged = raw_games.merge(
        team_strength,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
    )

    merged = add_opponent_features(merged)

    # Odds are optional / current-season-only -- merge if you have a key set
    if ODDS_API_KEY != "YOUR_ODDS_API_KEY_HERE":
        odds = pull_current_odds()
        merged = merged.merge(
            odds,
            left_on=["GAME_DATE", "TEAM_NAME"],
            right_on=["GAME_DATE", "TEAM_NAME"],
            how="left",
        )

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(merged)} rows to {OUTPUT_PATH}")
    return merged


if __name__ == "__main__":
    merge_features()