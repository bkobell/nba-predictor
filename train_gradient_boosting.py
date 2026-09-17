"""
train_gradient_boosting.py

Same task as train_baseline_model.py (predict single-game margin), same
time-based split, same features -- but using XGBoost instead of plain
linear regression.

Why this might do better than the linear baseline:
    Linear regression assumes every feature's effect is constant and
    additive (e.g. "IMPACT_DIFF always adds the same margin, regardless
    of context"). Gradient boosted trees can capture interactions and
    non-linearities a linear model can't -- e.g. maybe a big roster
    mismatch matters MORE when the weaker team is also on a back-to-back,
    or the relationship between rest and margin isn't a straight line.
    Trees are also naturally more robust to the multicollinearity we
    cleaned up earlier, since each split only looks at one feature at a
    time rather than trying to jointly weight several correlated inputs.

What you lose vs. linear regression:
    Clean, signed coefficients ("N points of margin per std dev of this
    feature"). Instead you get FEATURE IMPORTANCE, which tells you how
    much a feature was used to reduce prediction error, but not the
    direction or size of its effect the way a coefficient does.

Run locally after build_roster_aware_features.py has produced
games_with_roster_features.csv.

Install:
    pip install pandas scikit-learn xgboost
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from train_baseline_model import (
    INPUT_PATH,
    FEATURE_COLS,
    TARGET_COL,
    time_based_split,
    prepare_features,
    naive_baseline,
    train_linear_baseline,
    evaluate,
)


# ---------------------------------------------------------------------------
# STEP 1: Train gradient boosting model
# ---------------------------------------------------------------------------
def train_xgboost_model(X_train, y_train):
    """
    Hyperparameters chosen conservatively given the dataset size (~10K
    training rows) to avoid overfitting:
      - max_depth=4: shallow trees, less prone to memorizing noise
      - n_estimators=300 with a low learning_rate=0.03: many small steps
        rather than a few large ones, generally more stable
      - subsample / colsample_bytree < 1: each tree sees a random subset
        of rows/features, which acts like a regularizer
    These are reasonable starting defaults, not exhaustively tuned --
    if this beats linear regression, tuning further (via grid search or
    a validation split within the training window) could squeeze out
    more, but isn't necessary to get an honest first read.
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# STEP 2: Feature importance
# ---------------------------------------------------------------------------
def report_feature_importance(model, feature_cols=FEATURE_COLS):
    """
    XGBoost's default importance is "gain" -- how much a feature reduced
    prediction error, on average, when used in a split. Unlike a linear
    coefficient, this has no direction (positive/negative) or "per unit"
    interpretation -- it only tells you how USEFUL a feature was overall.
    """
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\nXGBoost feature importance (gain-based, higher = more useful):")
    print(importance.to_string())
    return importance


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(input_path=INPUT_PATH):
    df = pd.read_csv(input_path)
    train_df, test_df = time_based_split(df)

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # Naive baseline (for reference)
    naive_preds = naive_baseline(y_train, y_test)
    evaluate(y_test, naive_preds, "Naive baseline (mean margin, no features)")

    # Linear baseline (for direct comparison -- same split, same features)
    linear_model = train_linear_baseline(X_train, y_train)
    linear_preds = linear_model.predict(X_test)
    evaluate(y_test, linear_preds, "Linear regression baseline")

    # Gradient boosting
    xgb_model = train_xgboost_model(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    evaluate(y_test, xgb_preds, "XGBoost gradient boosting")

    report_feature_importance(xgb_model)

    return xgb_model, (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    main()
