import pandas as pd

merged = pd.read_csv("/Users/bkobell33/Documents/nba-predictor/data/games_with_roster_features.csv")
raw = pd.read_csv("/Users/bkobell33/Documents/nba-predictor/data/raw_games.csv")

print("raw_games.csv row count:      ", len(raw))
print("merged output row count:      ", len(merged))
print()

# Check for duplicate (GAME_ID, TEAM_ID) pairs in the merged output --
# there should be EXACTLY one row per team per game.
dupe_check = merged.groupby(["GAME_ID", "TEAM_ID"]).size()
print("Max rows for a single (GAME_ID, TEAM_ID) pair:", dupe_check.max())
print("Number of (GAME_ID, TEAM_ID) pairs with >1 row:", (dupe_check > 1).sum())
print()

# Show a concrete example of a duplicated game/team so we can see what's repeating
if (dupe_check > 1).any():
    example_key = dupe_check[dupe_check > 1].index[0]
    print("Example duplicated (GAME_ID, TEAM_ID):", example_key)
    example_rows = merged[
        (merged["GAME_ID"] == example_key[0]) & (merged["TEAM_ID"] == example_key[1])
    ]
    print(example_rows[["GAME_ID", "TEAM_ID", "GAME_DATE", "ROSTER_IMPACT_SUM", "OPP_ROSTER_IMPACT_SUM"]].to_string())