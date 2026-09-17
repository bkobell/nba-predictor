import pandas as pd

df = pd.read_csv("/Users/bkobell33/Documents/nba-predictor/data/games_with_roster_features.csv")

print("Correlation DAYS_REST <-> OPP_DAYS_REST:", df["DAYS_REST"].corr(df["OPP_DAYS_REST"]))
print()

# What fraction of rows have DAYS_REST EXACTLY equal to OPP_DAYS_REST?
exact_match_pct = (df["DAYS_REST"] == df["OPP_DAYS_REST"]).mean()
print(f"Fraction of rows where DAYS_REST == OPP_DAYS_REST exactly: {exact_match_pct:.1%}")
print()

# Show a sample of rows to eyeball -- if this is a real bug, DAYS_REST and
# OPP_DAYS_REST will look suspiciously identical row after row. If it's a
# real schedule effect, they should differ often but trend together.
sample = df[["GAME_ID", "TEAM_ID", "TEAM_NAME", "GAME_DATE", "DAYS_REST", "OPP_DAYS_REST"]].sample(15, random_state=1)
print(sample.to_string(index=False))
print()

# Distribution check -- how many unique values does each column actually take?
print("DAYS_REST unique values:", sorted(df["DAYS_REST"].unique())[:15])
print("OPP_DAYS_REST unique values:", sorted(df["OPP_DAYS_REST"].unique())[:15])