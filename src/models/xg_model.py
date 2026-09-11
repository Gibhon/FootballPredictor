import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, classification_report

def multiclass_brier_score(y_true, y_prob):
    y_true_onehot = np.eye(y_prob.shape[1])[y_true]
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))

def prep_data():
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_df = pd.read_csv(processed_dir / "train_set.csv")
    val_df = pd.read_csv(processed_dir / "val_set.csv")
    test_df = pd.read_csv(processed_dir / "test_set.csv")

    X_train, y_train = train_df.drop(columns=["result"]), train_df["result"]
    X_val, y_val = val_df.drop(columns=["result"]), val_df["result"]
    X_test, y_test = test_df.drop(columns=["result"]), test_df["result"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_xgb(X_train, y_train, X_val, y_val):
    model_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": ["mlogloss", "merror"],
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 500
    }

    model = xgb.XGBClassifier(**model_params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    return model


def evaluate_xgb(model, X, y):
    print("=" * 60)
    print("XGBOOST EVALUATION RESULTS")
    print("=" * 60)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    loss = log_loss(y, probabilities)
    brier = multiclass_brier_score(y, probabilities)
    acc = accuracy_score(y, predictions)

    print(f"Log Loss: {loss}")
    print(f"Brier Score: {brier}")
    print(f"Accuracy: {acc}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report:")
    print(classification_report(y, predictions, target_names=["Away (0)", "Draw (1)", "Home (2)"], zero_division=0))

    return loss, brier, acc


def main():
    train_X, train_y, val_X, val_y, test_X, test_y = prep_data()
    model = train_xgb(train_X, train_y, val_X, val_y)
    print("\n--- Validation Set Evaluation ---")
    val_loss, val_brier, val_acc = evaluate_xgb(model, val_X, val_y)
    print("\n--- Test Set Evaluation ---")
    test_loss, test_brier, test_acc = evaluate_xgb(model, test_X, test_y)


if __name__ == "__main__":
    main()
