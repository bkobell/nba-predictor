"""
Temporary test script -- NOT part of the pipeline, just for verifying
live_game_predictions.py works end to end using a real past date
(so we don't have to wait for the season to start to see it work).

Delete this file once you've confirmed things work.
"""

import live_game_predictions as lgp

TEST_DATE = "2026-04-01"  # a real date with games, confirmed earlier

# Monkey-patch get_todays_games to use the test date instead of "today"
original_get_todays_games = lgp.get_todays_games
lgp.get_todays_games = lambda: original_get_todays_games(TEST_DATE)

results = lgp.predict_todays_games()
if results is not None:
    print(f"\nTest predictions for {TEST_DATE}:")
    print(results.to_string(index=False))