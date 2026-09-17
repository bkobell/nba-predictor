"""
train_baseline_model.py

Takes the feature-enriched output of build_roster_aware_features.py
(games_with_roster_features.csv) and trains a baseline single-game margin
predictor, using a TIME-BASED train/test split.

Why time-based, not random:
    A random split would let the model train on games from later in a
    season while testing on earlier games from that same season -- but
    ROLLING_IMPACT, DAYS_REST, etc. are all built from each team's prior
    games, so a random split can leak information about a team's future
    form into what looks like a held-out test set. Splitting strictly by
    date means every test-set game happens after every training-set game,
    which matches how you'd actually deploy this (predict today's game
    using only what's known before today).

Two baselines are included so you have an honest floor to compare against:
    1. NAIVE baseline: predict every game's margin as the training set's
       average home-court margin (no features at all). If your real model
       can't beat this, the features aren't adding value yet.
    2. LINEAR baseline: plain linear regression on the engineered features.
       This is the "does roster/rest data actually help" checkpoint before
       reaching for something more complex like gradient boosting.

Run locally after build_roster_aware_features.py has produced
games_with_roster_features.csv.

Install:
    pip install pandas scikit-learn
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

INPUT_PATH = "/Users/bkobell33/Documents/nba-predictor/data/games_with_roster_features.csv"
MODEL_PATH = "/Users/bkobell33/Documents/nba-predictor/data/trained_linear_model.joblib"

# Features available BEFORE the game is played -- no target leakage.
# NOTE: ROSTER_IMPACT_AVG, OPP_ROSTER_IMPACT_AVG, and TOP_PLAYER_IMPACT were
# dropped -- they're derived from the same underlying player-impact numbers
# as ROSTER_IMPACT_SUM (correlated at r=0.86-0.98, see check_collinearity),
# so keeping all of them added redundancy without new information and made
# coefficients unstable. ROSTER_IMPACT_SUM + OPP_ROSTER_IMPACT_SUM +
# IMPACT_DIFF capture the same signal more cleanly.
FEATURE_COLS = [
    "IS_HOME",
    "DAYS_REST",
    "IS_BACK_TO_BACK",
    "IS_LONG_REST",
    "ROSTER_IMPACT_SUM",
    "OPP_ROSTER_IMPACT_SUM",
    "OPP_DAYS_REST",
    "OPP_IS_BACK_TO_BACK",
    "IMPACT_DIFF",
    "REST_DIFF",
]

TARGET_COL = "PLUS_MINUS"   # the actual game margin for that team, already in raw_games.csv

TEST_FRACTION = 0.2         # most recent 20% of games, by date, held out for testing


# ---------------------------------------------------------------------------
# STEP 1: Time-based split
# ---------------------------------------------------------------------------
def time_based_split(df, date_col="GAME_DATE", test_fraction=TEST_FRACTION):
    """
    Sorts by date and takes the most recent `test_fraction` of games as the
    test set. Everything in train happened strictly before everything in
    test (modulo games sharing the exact same date, which land together).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df.iloc[cutoff_idx][date_col]

    train = df[df[date_col] < cutoff_date]
    test = df[df[date_col] >= cutoff_date]

    print(f"Train: {len(train)} rows ({train[date_col].min().date()} to {train[date_col].max().date()})")
    print(f"Test:  {len(test)} rows ({test[date_col].min().date()} to {test[date_col].max().date()})")

    return train, test


# ---------------------------------------------------------------------------
# STEP 2: Clean / prepare features
# ---------------------------------------------------------------------------
def prepare_features(df, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    """
    Drops rows with missing features (e.g. players with < 3 games so
    ROLLING_IMPACT is still NaN -- see min_periods=3 in
    compute_player_impact). For a baseline this is simplest; a more
    mature version could impute instead of dropping.
    """
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]
    return X, y


# ---------------------------------------------------------------------------
# STEP 3: Naive baseline (no features, just average margin)
# ---------------------------------------------------------------------------
def naive_baseline(y_train, y_test):
    """
    Predicts every test-set game's margin as the training set's mean
    margin. This is the floor -- any real model needs to clear this bar
    to be worth using at all.
    """
    mean_margin = y_train.mean()
    y_pred = np.full(len(y_test), mean_margin)
    return y_pred


# ---------------------------------------------------------------------------
# STEP 4: Linear baseline model
# ---------------------------------------------------------------------------
def train_linear_baseline(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def check_collinearity(X_train, threshold=0.7):
    """
    ROSTER_IMPACT_SUM, ROSTER_IMPACT_AVG, TOP_PLAYER_IMPACT, and IMPACT_DIFF
    are all derived from the same underlying player-impact numbers, so
    they're likely highly correlated with each other. When a linear model
    gets several correlated inputs at once, it can't cleanly attribute
    credit between them -- coefficients become unstable and can even flip
    sign, even though the model's overall predictions are still fine.
    This prints any feature pairs above the correlation threshold so you
    know which coefficients to distrust individually.
    """
    corr = X_train.corr()
    print("\nHighly correlated feature pairs (|r| > {:.1f}):".format(threshold))
    found_any = False
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                print(f"  {cols[i]:25s} <-> {cols[j]:25s}  r = {r:.2f}")
                found_any = True
    if not found_any:
        print("  None found.")


def train_linear_baseline_standardized(X_train, y_train, X_test):
    """
    Same linear model, but fit on STANDARDIZED features (mean 0, std 1).
    This makes coefficients directly comparable to each other -- with the
    raw (unstandardized) features, a feature ranging in the thousands
    (ROSTER_IMPACT_SUM) and a feature that's just 0/1 (IS_HOME) produce
    coefficients on totally different scales, so sorting by raw magnitude
    (as the original version of this script did) is misleading. After
    standardizing, coefficient magnitude ~= "how many points of margin
    per one standard deviation of this feature," which is a fair comparison.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# STEP 5: Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    """
    Reports both regression error (MAE/RMSE on predicted margin) and the
    derived win/loss accuracy (sign of predicted margin vs. sign of
    actual margin) -- since your end goal is game outcomes, not just
    a margin number in isolation.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    win_pred = (np.array(y_pred) > 0).astype(int)
    win_true = (np.array(y_true) > 0).astype(int)
    acc = accuracy_score(win_true, win_pred)

    print(f"\n{label}")
    print(f"  MAE (points):        {mae:.2f}")
    print(f"  RMSE (points):       {rmse:.2f}")
    print(f"  Win/Loss accuracy:   {acc:.1%}")
    return {"mae": mae, "rmse": rmse, "accuracy": acc}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(input_path=INPUT_PATH):
    df = pd.read_csv(input_path)

    train_df, test_df = time_based_split(df)

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # Naive baseline
    naive_preds = naive_baseline(y_train, y_test)
    evaluate(y_test, naive_preds, "Naive baseline (mean margin, no features)")

    # Linear baseline (raw features -- kept for MAE/RMSE/accuracy comparability)
    model = train_linear_baseline(X_train, y_train)
    linear_preds = model.predict(X_test)
    evaluate(y_test, linear_preds, "Linear regression baseline")

    # The model above was trained on X_train ONLY, so we could honestly
    # evaluate it against a held-out test set. But for actually predicting
    # future games (the live pipeline's job), you want every bit of
    # historical signal available -- so retrain one final time on the
    # FULL dataset (train + test combined) before persisting. This is
    # standard practice: use the split to validate the approach, then
    # refit on everything once you trust it.
    X_full, y_full = prepare_features(df)
    production_model = train_linear_baseline(X_full, y_full)
    joblib.dump(production_model, MODEL_PATH)
    print(f"\nProduction model (trained on all {len(X_full)} rows) saved to {MODEL_PATH}")

    # Collinearity check -- explains why some raw coefficients had
    # counterintuitive signs (e.g. ROSTER_IMPACT_AVG vs OPP_ROSTER_IMPACT_AVG)
    check_collinearity(X_train)

    # Standardized version -- THIS is the one to read for feature importance.
    # Same predictions/accuracy as the raw model above (standardizing doesn't
    # change what the model predicts, only how interpretable the coefficients
    # are), but now coefficient magnitude is directly comparable across features.
    std_model, X_train_scaled, X_test_scaled = train_linear_baseline_standardized(
        X_train, y_train, X_test
    )
    coef_report = pd.Series(std_model.coef_, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print("\nStandardized coefficients (points of margin per 1 std dev of feature):")
    print(coef_report.to_string())

    return model, (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    main()