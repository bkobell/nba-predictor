"""
walk_forward_validation.py

Your current train_baseline_model.py uses ONE train/test split (train on
2021-2025, test on 2025-26). That gives you a single MAE/accuracy number,
but no sense of whether that number was a bit lucky or unlucky for that
particular season -- some seasons have more roster chaos (trades,
injuries, tanking) than others.

Walk-forward validation instead trains on an EXPANDING window and tests on
each subsequent season in turn:

    Fold 1: Train [2021-22]                        -> Test [2022-23]
    Fold 2: Train [2021-22, 2022-23]                -> Test [2023-24]
    Fold 3: Train [2021-22, 2022-23, 2023-24]       -> Test [2024-25]
    Fold 4: Train [2021-22, ..., 2024-25]           -> Test [2025-26]

This mirrors how you'd actually use the model in practice (always
predicting the next, not-yet-seen season using everything before it), and
shows you the SPREAD of performance across folds, not just one number.

Run locally after build_roster_aware_features.py has produced
games_with_roster_features.csv.

Install:
    pip install pandas scikit-learn
"""

import pandas as pd
import numpy as np

from train_baseline_model import (
    INPUT_PATH,
    FEATURE_COLS,
    TARGET_COL,
    prepare_features,
    naive_baseline,
    train_linear_baseline,
    train_linear_baseline_standardized,
    evaluate,
)


# ---------------------------------------------------------------------------
# STEP 1: Build season-based folds
# ---------------------------------------------------------------------------
def build_season_folds(df, season_col="SEASON"):
    """
    Returns a list of (train_df, test_df, test_season_label) tuples, one
    per fold, using an expanding training window and a single held-out
    season as the test set each time.

    Requires at least 2 distinct seasons in the data (need at least 1
    season to train on before testing on the next).
    """
    seasons = sorted(df[season_col].unique())
    if len(seasons) < 2:
        raise ValueError(
            f"Need at least 2 seasons for walk-forward validation, found {len(seasons)}: {seasons}"
        )

    folds = []
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        train_df = df[df[season_col].isin(train_seasons)]
        test_df = df[df[season_col] == test_season]
        folds.append((train_df, test_df, test_season, train_seasons))

    return folds


# ---------------------------------------------------------------------------
# STEP 2: Run linear baseline across all folds
# ---------------------------------------------------------------------------
def run_walk_forward(input_path=INPUT_PATH):
    df = pd.read_csv(input_path)
    folds = build_season_folds(df)

    results = []
    coef_records = []

    for train_df, test_df, test_season, train_seasons in folds:
        print(f"\n{'='*70}")
        print(f"FOLD: train on {train_seasons} -> test on {test_season}")
        print(f"{'='*70}")

        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)

        print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")

        # Naive baseline for this fold
        naive_preds = naive_baseline(y_train, y_test)
        naive_metrics = evaluate(y_test, naive_preds, f"Naive baseline ({test_season})")

        # Linear baseline for this fold
        model = train_linear_baseline(X_train, y_train)
        linear_preds = model.predict(X_test)
        linear_metrics = evaluate(y_test, linear_preds, f"Linear baseline ({test_season})")

        # Standardized coefficients for this fold -- lets us check whether
        # IMPACT_DIFF's importance is stable across different seasons, or
        # whether it was a one-season fluke.
        std_model, _, _ = train_linear_baseline_standardized(X_train, y_train, X_test)
        coefs = dict(zip(FEATURE_COLS, std_model.coef_))
        coefs["test_season"] = test_season
        coef_records.append(coefs)

        results.append({
            "test_season": test_season,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "naive_mae": naive_metrics["mae"],
            "naive_accuracy": naive_metrics["accuracy"],
            "linear_mae": linear_metrics["mae"],
            "linear_rmse": linear_metrics["rmse"],
            "linear_accuracy": linear_metrics["accuracy"],
        })

    return pd.DataFrame(results), pd.DataFrame(coef_records)


# ---------------------------------------------------------------------------
# STEP 3: Summarize across folds
# ---------------------------------------------------------------------------
def summarize(results_df, coef_df):
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*70}\n")

    print(results_df.to_string(index=False))
    print()
    print(f"Linear MAE:      mean={results_df['linear_mae'].mean():.2f}  "
          f"std={results_df['linear_mae'].std():.2f}  "
          f"min={results_df['linear_mae'].min():.2f}  "
          f"max={results_df['linear_mae'].max():.2f}")
    print(f"Linear accuracy: mean={results_df['linear_accuracy'].mean():.1%}  "
          f"std={results_df['linear_accuracy'].std():.1%}  "
          f"min={results_df['linear_accuracy'].min():.1%}  "
          f"max={results_df['linear_accuracy'].max():.1%}")

    improvement = results_df["naive_mae"] - results_df["linear_mae"]
    print(f"\nMAE improvement over naive, per fold: "
          f"mean={improvement.mean():.2f}  std={improvement.std():.2f}")
    if (improvement > 0).all():
        print("-> Linear model beat naive in EVERY fold. Good sign of consistency.")
    else:
        losing_folds = results_df.loc[improvement <= 0, "test_season"].tolist()
        print(f"-> Linear model did NOT beat naive in: {losing_folds}. Worth investigating why.")

    print(f"\n{'='*70}")
    print("FEATURE STABILITY ACROSS FOLDS (standardized coefficients)")
    print(f"{'='*70}\n")
    coef_cols = [c for c in coef_df.columns if c != "test_season"]
    stability = coef_df[coef_cols].agg(["mean", "std"]).T
    stability["abs_mean"] = stability["mean"].abs()
    stability = stability.sort_values("abs_mean", ascending=False).drop(columns="abs_mean")
    print(stability.to_string())
    print("\nA feature with a large mean but small std relative to that mean is")
    print("stable across seasons. A feature whose std is large relative to its")
    print("mean (or that flips sign across folds) is not -- treat its raw-data")
    print("importance with more caution.")


if __name__ == "__main__":
    results_df, coef_df = run_walk_forward()
    summarize(results_df, coef_df)