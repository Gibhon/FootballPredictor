import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, log_loss, classification_report

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DATA_PATH = r"C:\Users\Sponge\Documents\Code\python\projects\FootballMachine2\data\ready.csv"  # Update if needed
TARGET_COLUMN = "result"

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "learning_rate": 0.02,
    "max_depth": 4,
    "reg_alpha":5.0,
    "reg_lambda":5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_data(filepath):
    df = pd.read_csv(filepath)

    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

    y = df[TARGET_COLUMN].astype(int).values

    ignore_cols = [TARGET_COLUMN, "match_date", "home_team", "away_team"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].values

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"Features used: {feature_cols}")
    return X, y


# ==========================================
# 3. TRAIN / VAL / TEST SPLIT (CHRONOLOGICAL)
# ==========================================
def create_splits(X, y):
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = (
        X[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

    print(
        f"Train set: {len(X_train)} | Val set: {len(X_val)} | Test set: {len(X_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================================
# 4. TRAINING WITH CLASS IMBALANCE WEIGHTS
# ==========================================
def main():
    X, y = load_data(DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = create_splits(X, y)

    # Clean unweighted training to keep baseline performance stable
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
        verbose_eval=50
    )

    # ==========================================
    # 5. EVALUATION WITH CONTROLLED THRESHOLD
    # ==========================================
    print("\n--- Evaluating on Test Set ---")
    preds_prob = model.predict(dtest)
    
    # Gently nudge draw probabilities (index 1) at evaluation time only
    preds_prob_adjusted = preds_prob.copy()
    preds_prob_adjusted[:, 1] *= 1.12  # Fine-tune between 1.05 and 1.18
    
    preds_class = np.argmax(preds_prob_adjusted, axis=1)

    acc = accuracy_score(y_test, preds_class)
    loss = log_loss(y_test, preds_prob)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Multi-LogLoss: {loss:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds_class, target_names=["Home Win (0)", "Draw (1)", "Away Win (2)"]))

if __name__ == "__main__":
    main()
