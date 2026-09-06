import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DATA_PATH = r"C:\Users\Sponge\Documents\Code\python\projects\FootballPredictor\data\ready_domestic_D.csv"
TARGET_COLUMN = "result"

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "learning_rate": 0.01,
    "max_depth": 3,
    "reg_alpha": 5.0,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def devig_odds(prob_or_odds_array, are_probabilities=False):
    arr = np.array(prob_or_odds_array)
    if are_probabilities:
        implied_probs = arr
    else:
        implied_probs = 1 / arr
    return implied_probs / np.sum(implied_probs, axis=1, keepdims=True)


def multiclass_brier_score(y_true, y_probs):
    classes = np.unique(y_true)
    y_true_onehot = np.eye(len(classes))[y_true]
    return np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))


def plot_ovr_calibration(y_test, preds_prob_adjusted, market_probs):
    plt.figure(figsize=(8, 6))
    for i in range(preds_prob_adjusted.shape[1]):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            (y_test == i).astype(int), preds_prob_adjusted[:, i], n_bins=10
        )
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            label=f"Class {i} Model",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.show()


# ==========================================
# 3. DATA LOADING & PREPROCESSING
# ==========================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(
        df[["homeW_probability", "draw_probability", "awayW_probability"]]
        .sum(axis=1)
        .describe()
    )
    sample = df[['homeW_probability','draw_probability','awayW_probability','result']].sample(10, random_state=1)
    print(sample)


    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

    y = df[TARGET_COLUMN].astype(int).values

    # Extract market odds/probabilities directly from your dataframe columns.
    # Update these column names if your CSV uses different headers (e.g., B365H, B365D, B365A or odds columns)
    odds_cols = ["home_odds", "draw_odds", "away_odds"]
    using_probabilities = not all(col in df.columns for col in odds_cols)
    if using_probabilities:
        odds_cols = ["homeW_probability", "draw_probability", "awayW_probability"]

    market_raw = df[odds_cols].values

    # Exclude target, metadata, and market raw columns from model features if needed
    ignore_cols = [
        TARGET_COLUMN,
        "match_date",
        "home_team",
        "away_team",
    ]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].values

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"Features used: {feature_cols}")
    return X, y, market_raw, feature_cols


# ==========================================
# 4. CHRONOLOGICAL SPLIT
# ==========================================
def create_splits(X, y, market_raw):
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)

    X_train, y_train, market_train = (
        X[:train_size],
        y[:train_size],
        market_raw[:train_size],
    )
    X_val, y_val, market_val = (
        X[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
        market_raw[train_size : train_size + val_size],
    )
    X_test, y_test, market_test = (
        X[train_size + val_size :],
        y[train_size + val_size :],
        market_raw[train_size + val_size :],
    )

    print(
        f"Train set: {len(X_train)} | Val set: {len(X_val)} | Test set:"
        f" {len(X_test)}"
    )
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        market_train,
        market_val,
        market_test,
    )


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    X, y, market_raw, feature_cols = load_data(DATA_PATH)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        market_train,
        market_val,
        market_test,
    ) = create_splits(X, y, market_raw)

    # Use actual test set odds/market data from the dataframe instead of random values
    test_market_data = market_test

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, "train"), (dval, "val")]

    print("\n--- Starting Model Training ---")
    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )

    print("\n--- Evaluating on Test Set ---")
    preds_prob = model.predict(dtest)

    preds_prob_adjusted = preds_prob.copy()
    preds_prob_adjuster = 1.0
    preds_prob_adjusted[:, 1] *= preds_prob_adjuster
    preds_prob_adjusted = preds_prob_adjusted / preds_prob_adjusted.sum(
        axis=1, keepdims=True
    )

    preds_class = np.argmax(preds_prob_adjusted, axis=1)

    acc = accuracy_score(y_test, preds_class)
    loss = log_loss(y_test, preds_prob_adjusted)
    print(f"acc:{acc}")
    print(f"test_loss: {loss}")

    draw_probs = preds_prob_adjusted[:, 1]
    actual_draws = (y_test == 1).astype(int)

    auc = roc_auc_score(actual_draws, draw_probs)
    print("Draw AUC:", auc)

    # 1. Check standard classification accuracy on test set
    accuracy = np.mean(preds_class == y_test)
    print(f"Model Classification Accuracy: {accuracy:.4f}")

    # 2. Convert dataframe market data to fair probabilities (if they are raw odds, devig them; if already probabilities, normalize)
    # Assuming test_market_data contains decimal odds; swap with direct assignment if they are already probabilities.
    market_probs = devig_odds(test_market_data, are_probabilities=True)

    # 3. Calculate and compare Multiclass Brier Scores using adjusted probabilities
    model_bs = multiclass_brier_score(y_test, preds_prob_adjusted)
    market_bs = multiclass_brier_score(y_test, market_probs)

    print(f"Model Brier Score:  {model_bs:.5f}")
    print(f"Market Brier Score: {market_bs:.5f}")

    if model_bs < market_bs:
        print(
            "-> Your model is better calibrated than the market overall (lower"
            " Brier score)."
        )
    else:
        print("-> The market is currently better calibrated overall.")

    # 4. Generate One-vs-Rest Reliability Diagrams
    plot_ovr_calibration(y_test, preds_prob_adjusted, market_probs)


if __name__ == "__main__":
    main()
