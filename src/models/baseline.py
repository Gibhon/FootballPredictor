"""
Baseline Model Evaluation Module for Football Match Outcome Prediction.

This module loads preprocessed train and validation datasets to evaluate a dummy
classifier baseline (prior-based probability matching). It computes probabilistic
metrics like Log-Loss and Multi-class Brier Score along with standard classification outputs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def multiclass_brier_score(y_true, y_prob):
    """
    Calculates the multi-class Brier score using one-hot encoded ground truth targets.
    """
    y_true_one_hot = np.eye(y_prob.shape[1])[y_true]
    return np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))


def evaluate_baseline(data_dir):
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train_set.csv")
    val_df = pd.read_csv(data_path / "val_set.csv")

    X_train, y_train = train_df.drop(columns=["result"]), train_df["result"]
    X_val, y_val = val_df.drop(columns=["result"]), val_df["result"]

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_val)
    y_prob = dummy.predict_proba(X_val)

    loss = log_loss(y_val, y_prob)
    brier = multiclass_brier_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)

    print("=" * 60)
    print("DUMMY CLASSIFIER BASELINE EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Probabilistic Metrics ---")
    print(f"Log-Loss           : {loss:.4f}")
    print(f"Multi-class Brier  : {brier:.4f}")

    print("\n--- Summary Metrics ---")
    print(f"Accuracy           : {acc:.4f} ({acc * 100:.2f}%)")

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n--- Detailed Classification Report ---")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=["Away (0)", "Draw (1)", "Home (2)"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    evaluate_baseline(processed_dir)
