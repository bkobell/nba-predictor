from nba_api.stats.endpoints import scoreboardv2

sb = scoreboardv2.ScoreboardV2(game_date="2026-04-01")
frames = sb.get_data_frames()

print("Number of frames:", len(frames))
for i, f in enumerate(frames):
    print(f"\n--- Frame {i} --- shape: {f.shape}")
    print("Columns:", list(f.columns))
    if not f.empty:
        print(f.head(2).to_string())