"""
live_game_predictions.py

Predicts today's NBA games using the model trained by train_baseline_model.py.

This is the piece that actually answers the original question this whole
project started with: "the Heat just got Giannis -- can the model account
for that?" Everything up to this point trained and validated on HISTORICAL
games. This script instead:

  1. Pulls TODAY's actual scheduled games (nba_api live scoreboard).
  2. For each team, pulls their CURRENT roster (nba_api commonteamroster --
     this reflects trades IMMEDIATELY, even before a traded player has
     played a single game for the new team).
  3. For each rostered player, looks up their most recent rolling impact
     rating from player_logs_cache.csv -- crucially, matched by PLAYER_ID
     across ANY team they've played for, not just their current one. This
     is what makes a brand-new trade work correctly: on Giannis's very
     first game as a Heat player, his rolling impact still reflects his
     Bucks performance (since that's all the history that exists yet),
     and it flows into Miami's team strength immediately via the current
     roster, without needing him to accumulate new games first.
  4. Computes rest days from each team's most recent actual game.
  5. Assembles the exact same feature vector the model was trained on and
     predicts tonight's margins.

Important limitations, stated plainly:
  - Since a future game hasn't happened, we don't know each player's
    ACTUAL minutes tonight. This uses each player's recent average
    minutes as a proxy, which won't be exactly right (e.g. it won't
    know about a coach's rotation change or a load-management night).
  - The roster is filtered to players with meaningful recent playing
    time (see MIN_RECENT_GAMES_THRESHOLD below) to approximate who will
    actually play, rather than every name on the full roster including
    two-way/G-League players who rarely suit up.
  - This does NOT know about same-day injury/inactive designations. A
    player who is questionable or ruled out tonight will still be
    included unless they've already been out of the lineup for a
    while (which the recent-games filter will naturally catch, just
    with a lag).
  - I could not test the actual live API calls (scoreboard, roster) in my
    environment -- only the aggregation/merging logic, using synthetic
    data. If the live calls error out, the traceback will likely point to
    a field name mismatch in the API response; paste it back and we'll
    adjust.

Run locally, ideally same-day before games start.

Install:
    pip install nba_api pandas joblib
"""

import datetime
import time
import pandas as pd
import numpy as np
import joblib

from nba_api.stats.endpoints import scoreboardv2, commonteamroster

from build_roster_aware_features import (
    get_current_season_str,
    ROLLING_WINDOW,
    PLAYER_LOGS_CACHE_PATH,
    RAW_GAMES_PATH,
)
from train_baseline_model import MODEL_PATH, FEATURE_COLS

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LEAGUE_AVG_REST_DAYS = 2
MIN_RECENT_GAMES_THRESHOLD = 2   # player must have played >= this many of their
                                  # last 10 logged games to count as "in the rotation"
FALLBACK_IMPACT = 0.0            # used for a rostered player with zero game history
                                  # (e.g. a rookie who hasn't debuted yet)
FALLBACK_MINUTES = 15.0          # same, for estimated minutes
API_TIMEOUT_SECONDS = 60         # bumped up from nba_api's default 30s -- stats.nba.com
                                  # can be slow, especially on some networks
API_MAX_RETRIES = 3
API_RETRY_DELAY_SECONDS = 3


def _call_with_retry(fn, description, max_retries=API_MAX_RETRIES, delay=API_RETRY_DELAY_SECONDS):
    """
    Wraps a live API call with a few retries on failure (timeouts,
    transient connection errors). stats.nba.com is an unofficial,
    sometimes-slow endpoint -- a single timeout shouldn't kill the whole
    prediction run, especially since this often gets run once a day
    rather than in a tight loop where retries would be wasteful.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            print(f"  [{description}] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# STEP 1: Today's scheduled games
# ---------------------------------------------------------------------------
def get_todays_games(game_date=None):
    """
    Returns a list of dicts: [{game_id, home_team_id, away_team_id,
    home_team_name, away_team_name, game_status_text}, ...] for the given
    date's NBA slate (defaults to today).

    Uses ScoreboardV2 (stats.nba.com) rather than the live scoreboard
    (cdn.nba.com) -- the live endpoint returned a 403 Access Denied from
    NBA's CDN in testing, while ScoreboardV2 uses the same stats.nba.com
    access pattern the rest of this pipeline already relies on
    successfully (player logs, rosters). Verified against real data:
    frame 0 (GameHeader) has GAME_ID/HOME_TEAM_ID/VISITOR_TEAM_ID/
    GAME_STATUS_TEXT; frame 1 (LineScore) has TEAM_ID/TEAM_CITY_NAME/
    TEAM_NAME for building readable team names.
    """
    game_date = game_date or datetime.date.today().strftime("%Y-%m-%d")
    sb = _call_with_retry(
        lambda: scoreboardv2.ScoreboardV2(game_date=game_date, timeout=API_TIMEOUT_SECONDS),
        description=f"ScoreboardV2({game_date})",
    )
    frames = sb.get_data_frames()
    game_header = frames[0]
    line_score = frames[1]

    if game_header.empty:
        return []

    # LineScore has one row per team per game -- dedupe to a clean
    # TEAM_ID -> "City Name" lookup.
    team_names = (
        line_score.drop_duplicates(subset=["TEAM_ID"])
        .set_index("TEAM_ID")
        .apply(lambda r: f"{r['TEAM_CITY_NAME']} {r['TEAM_NAME']}", axis=1)
        .to_dict()
    )

    todays_games = []
    for _, row in game_header.iterrows():
        home_id = row["HOME_TEAM_ID"]
        away_id = row["VISITOR_TEAM_ID"]
        todays_games.append({
            "game_id": row["GAME_ID"],
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": team_names.get(home_id, str(home_id)),
            "away_team_name": team_names.get(away_id, str(away_id)),
            "game_status_text": row["GAME_STATUS_TEXT"],
        })
    return todays_games


# ---------------------------------------------------------------------------
# STEP 2: Current roster per team
# ---------------------------------------------------------------------------
def get_current_roster(team_id, season=None):
    """
    Current roster via nba_api -- reflects trades immediately, even before
    a newly-acquired player has played a game for this team yet. Returns
    a list of PLAYER_IDs.
    """
    season = season or get_current_season_str()
    roster = _call_with_retry(
        lambda: commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=API_TIMEOUT_SECONDS),
        description=f"CommonTeamRoster(team_id={team_id})",
    )
    df = roster.get_data_frames()[0]
    return df["PLAYER_ID"].tolist()


# ---------------------------------------------------------------------------
# STEP 3: Player impact lookup (most recent value per player, any team)
# ---------------------------------------------------------------------------
def build_player_impact_lookup(cache_path=PLAYER_LOGS_CACHE_PATH, window=ROLLING_WINDOW):
    """
    For every player who has any game history, computes:
      - CURRENT_IMPACT: mean PLUS_MINUS over their most recent `window`
        games (their latest rolling-impact value, ready to be used for
        a hypothetical NEXT game -- unlike training, no shift is needed
        here since there's no "current row" to leak from; the game we're
        predicting doesn't exist in this data at all).
      - RECENT_AVG_MIN: mean minutes over that same recent window, used
        as a proxy for expected minutes tonight.
      - GAMES_IN_WINDOW: how many of their last `window` logged games they
        actually appeared in -- used downstream to filter out players who
        are barely playing (many DNPs) even if technically on the roster.

    Matched by PLAYER_ID only, deliberately NOT filtered by team -- this
    is what makes a same-day trade work correctly: a player's historical
    impact carries over regardless of which team those games were with.
    """
    logs = pd.read_csv(cache_path)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"])

    def last_n_stats(group):
        recent = group.tail(window)
        return pd.Series({
            "CURRENT_IMPACT": recent["PLUS_MINUS"].mean(),
            "RECENT_AVG_MIN": recent["MIN"].mean(),
            "GAMES_IN_WINDOW": len(recent),
        })

    lookup = logs.groupby("PLAYER_ID").apply(last_n_stats, include_groups=False)
    return lookup


# ---------------------------------------------------------------------------
# STEP 4: Aggregate a team's current roster into ROSTER_IMPACT features
# ---------------------------------------------------------------------------
def build_team_roster_features(team_id, impact_lookup, season=None):
    """
    Combines the current roster (step 2) with the impact lookup (step 3)
    to produce the same ROSTER_IMPACT_SUM / ROSTER_IMPACT_AVG / etc.
    features the model was trained on, but for TONIGHT's actual roster.
    """
    player_ids = get_current_roster(team_id, season)

    rows = []
    for pid in player_ids:
        if pid in impact_lookup.index:
            row = impact_lookup.loc[pid]
            # Skip players who are on the roster but barely play (likely
            # two-way/G-League/deep bench) -- they'd otherwise dilute the
            # estimate with near-zero-relevance rows.
            if row["GAMES_IN_WINDOW"] < MIN_RECENT_GAMES_THRESHOLD:
                continue
            impact = row["CURRENT_IMPACT"]
            minutes = row["RECENT_AVG_MIN"]
        else:
            # No game history at all (true rookie, or just signed) --
            # use neutral fallback values rather than dropping them
            # entirely, so the roster isn't silently undercounted.
            impact = FALLBACK_IMPACT
            minutes = FALLBACK_MINUTES
        rows.append({"PLAYER_ID": pid, "impact": impact, "minutes": minutes})

    roster_df = pd.DataFrame(rows)
    if roster_df.empty:
        # Extremely defensive fallback -- should not happen in practice.
        return {"ROSTER_IMPACT_SUM": 0.0, "ROSTER_IMPACT_AVG": 0.0, "TOP_PLAYER_IMPACT": 0.0}

    roster_df["weighted_impact"] = roster_df["impact"] * roster_df["minutes"]
    return {
        "ROSTER_IMPACT_SUM": roster_df["weighted_impact"].sum(),
        "ROSTER_IMPACT_AVG": roster_df["impact"].mean(),
        "TOP_PLAYER_IMPACT": roster_df["impact"].max(),
    }


# ---------------------------------------------------------------------------
# STEP 5: Rest days from each team's most recent actual game
# ---------------------------------------------------------------------------
def get_days_rest(team_id, raw_games_path=RAW_GAMES_PATH, as_of_date=None):
    """
    Looks up a team's most recent game date from raw_games.csv and
    computes days of rest heading into today. Falls back to
    LEAGUE_AVG_REST_DAYS if no recent game is found (e.g. season opener).
    """
    as_of_date = as_of_date or datetime.date.today()
    raw_games = pd.read_csv(raw_games_path)
    raw_games["GAME_DATE"] = pd.to_datetime(raw_games["GAME_DATE"])

    team_games = raw_games[raw_games["TEAM_ID"] == team_id]
    past_games = team_games[team_games["GAME_DATE"].dt.date < as_of_date]
    if past_games.empty:
        return LEAGUE_AVG_REST_DAYS

    last_game_date = past_games["GAME_DATE"].max().date()
    return (as_of_date - last_game_date).days


# ---------------------------------------------------------------------------
# STEP 6: Assemble features + predict
# ---------------------------------------------------------------------------
def predict_todays_games():
    model = joblib.load(MODEL_PATH)
    impact_lookup = build_player_impact_lookup()
    games = get_todays_games()

    if not games:
        print("No games scheduled today.")
        return None

    predictions = []
    for game in games:
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        home_roster = build_team_roster_features(home_id, impact_lookup)
        away_roster = build_team_roster_features(away_id, impact_lookup)

        home_rest = get_days_rest(home_id)
        away_rest = get_days_rest(away_id)

        # Build the feature row for the HOME team's perspective (matches
        # how the model was trained: one row per team, with OPP_ columns
        # for the opponent). Predicting from the away team's perspective
        # would just be the mirror image.
        features = {
            "IS_HOME": 1,
            "DAYS_REST": home_rest,
            "IS_BACK_TO_BACK": int(home_rest == 1),
            "IS_LONG_REST": int(home_rest >= 3),
            "ROSTER_IMPACT_SUM": home_roster["ROSTER_IMPACT_SUM"],
            "OPP_ROSTER_IMPACT_SUM": away_roster["ROSTER_IMPACT_SUM"],
            "OPP_DAYS_REST": away_rest,
            "OPP_IS_BACK_TO_BACK": int(away_rest == 1),
        }
        features["IMPACT_DIFF"] = features["ROSTER_IMPACT_SUM"] - features["OPP_ROSTER_IMPACT_SUM"]
        features["REST_DIFF"] = features["DAYS_REST"] - features["OPP_DAYS_REST"]

        X = pd.DataFrame([features])[FEATURE_COLS]  # enforce exact training column order
        predicted_margin = model.predict(X)[0]

        predicted_winner = game["home_team_name"] if predicted_margin > 0 else game["away_team_name"]

        predictions.append({
            "matchup": f"{game['away_team_name']} @ {game['home_team_name']}",
            "predicted_home_margin": round(predicted_margin, 1),
            "predicted_winner": predicted_winner,
        })

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    results = predict_todays_games()
    if results is not None:
        print("\nToday's predictions:")
        print(results.to_string(index=False))
