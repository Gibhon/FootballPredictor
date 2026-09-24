"""XGBoost Model Training and Evaluation Pipeline.

This module loads processed training, validation, and test datasets, trains
an XGBoost multiclass classifier with early stopping, evaluates performance
across splits (Accuracy, Brier Score, Log Loss, and One-vs-Rest ROC-AUC), prints
a structured summary comparison table, and generates reliability calibration curves.
"""

from pathlib import Path
import sys
import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
import xgboost as xgb


# Resolve project root and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SEED

def get_class_auc(y_true: np.ndarray | pd.Series, proba: np.ndarray, idx: int = 1) -> float:
    """Calculates One-vs-Rest ROC-AUC for a specific target class.

    Args:
        y_true (np.ndarray | pd.Series): Ground truth target labels (0: Away, 1: Draw, 2: Home).
        proba (np.ndarray): Predicted probability matrix of shape (n_samples, n_classes).
        idx (int, optional): Target class index to evaluate. Defaults to 1 (Draw).

    Returns:
        float: One-vs-Rest ROC-AUC score for the specified class.
    """
    y_true_binary = (y_true == idx).astype(int)
    probabilities = proba[:, idx]
    return float(roc_auc_score(y_true_binary, probabilities))


def get_all_aucs(
    y_true: np.ndarray | pd.Series, proba: np.ndarray
) -> typing.Tuple[float, float, float]:
    """Calculates One-vs-Rest ROC-AUC for all three outcome classes.

    Args:
        y_true (np.ndarray | pd.Series): Ground truth target labels.
        proba (np.ndarray): Predicted probability matrix of shape (n_samples, 3).

    Returns:
        typing.Tuple[float, float, float]: Tuple containing (auc_away, auc_draw, auc_home).
    """
    auc_away = get_class_auc(y_true, proba, idx=0)
    auc_draw = get_class_auc(y_true, proba, idx=1)
    auc_home = get_class_auc(y_true, proba, idx=2)
    return auc_away, auc_draw, auc_home


def plot_reliability(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    class_names: list[str],
    out_path: Path | str,
    n_bins: int = 10,
) -> None:
    """Plots probability calibration reliability curves for each outcome class.

    Generates one calibration subplot per target class comparing predicted 
    quantile probabilities against observed proportions, including a reference 
    diagonal line for perfect calibration, and saves the plot figure to disk.

    Args:
        y_true (np.ndarray | pd.Series): Ground truth target array.
        proba (np.ndarray): Array of predicted class probabilities matching shape (n_samples, 3).
            Column order must align with class_names: [Away (0), Draw (1), Home (2)].
        class_names (list[str]): Names corresponding to target column indices.
        out_path (Path | str): Output destination path to write the saved figure.
        n_bins (int, optional): Number of probability bins for calibration curves. Defaults to 10.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 5), sharey=True)

    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        y_binary = (y_true == idx).astype(int)
        class_probs = proba[:, idx]

        prob_true, prob_pred = calibration_curve(
            y_binary, class_probs, n_bins=n_bins, strategy="quantile"
        )

        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

        ax.set_title(f"Reliability Curve: {class_name}")
        ax.set_xlabel("Mean Predicted Probability")
        if idx == 0:
            ax.set_ylabel("Fraction of Positives")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nReliability plot saved to {out_path}")


def multiclass_brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Calculates the multi-class Brier score using one-hot encoded targets.

    Args:
        y_true (np.ndarray | pd.Series): Ground truth target array.
        y_prob (np.ndarray): Predicted probability matrix of shape (n_samples, n_classes).

    Returns:
        float: Mean squared error across all class probability distributions.
    """
    y_true_onehot = np.eye(y_prob.shape[1])[y_true]
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


def prep_data() -> typing.Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Index
]:
    """Loads processed training, validation, and testing CSV datasets.

    Returns:
        typing.Tuple: Partitioned feature matrices and target series for train, val, test, 
            along with feature column names index.
    """
    processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    train_df = pd.read_csv(processed_dir / "train_set.csv")
    val_df = pd.read_csv(processed_dir / "val_set.csv")
    test_df = pd.read_csv(processed_dir / "test_set.csv")

    feature_only_df = train_df.drop(columns=["result"])
    features = feature_only_df.columns

    X_train, y_train = train_df.drop(columns=["result"]), train_df["result"]
    X_val, y_val = val_df.drop(columns=["result"]), val_df["result"]
    X_test, y_test = test_df.drop(columns=["result"]), test_df["result"]

    print("Data has been prepared for training!!!")

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def train_xgb(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> xgb.XGBClassifier:
    """Trains an XGBoost multi-class classifier using validation early stopping.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        X_val (pd.DataFrame): Validation feature matrix.
        y_val (pd.Series): Validation target labels.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model object.
    """
    model_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": ["mlogloss", "merror"],
        "learning_rate": 0.01,
        "max_depth": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "random_state": SEED,
        "n_estimators": 500,
    }
    
    model = xgb.XGBClassifier(**model_params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    return model


def evaluate_xgb(
    model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, split_name: str = ""
) -> typing.Tuple[float, float, float, np.ndarray]:
    """Evaluates the XGBoost classifier, returning core probability and error metrics.

    Args:
        model (xgb.XGBClassifier): Fitted XGBoost model instance.
        X (pd.DataFrame): Evaluation feature dataset.
        y (pd.Series): Evaluation ground truth targets.
        split_name (str, optional): Name of the data split for reporting. Defaults to "".

    Returns:
        typing.Tuple[float, float, float, np.ndarray]: Calculated log loss, 
            Brier score, accuracy score, and probability predictions array.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    loss = float(log_loss(y, probabilities))
    brier = multiclass_brier_score(y, probabilities)
    acc = float(accuracy_score(y, predictions))

    if split_name:
        print(f"\n--- {split_name} Classification Report ---")
        print(
            classification_report(
                y,
                predictions,
                target_names=["Away (0)", "Draw (1)", "Home (2)"],
                zero_division=0,
            )
        )
        print("Confusion Matrix:")
        print(confusion_matrix(y, predictions))

    return loss, brier, acc, probabilities


def get_feature_importance(
    model: xgb.XGBClassifier, feature_names: pd.Index | list[str]
) -> pd.DataFrame:
    """Extracts, normalizes, and prints model feature importances in descending order.

    Args:
        model (xgb.XGBClassifier): Fitted XGBoost model.
        feature_names (pd.Index | list[str]): Feature column names.

    Returns:
        pd.DataFrame: DataFrame containing feature names and relative percentage importances.
    """
    importances = model.feature_importances_

    total_importance = np.sum(importances)
    if total_importance > 0:
        importance_pct = (importances / total_importance) * 100
    else:
        importance_pct = importances

    df_importance = (
        pd.DataFrame({"feature": feature_names, "importance_pct": importance_pct})
        .sort_values(by="importance_pct", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Feature Importances ---")
    for _, row in df_importance.iterrows():
        print(f"{row['feature']:<35} {row['importance_pct']:6.2f}%")

    return df_importance


import json
from pathlib import Path
import xgboost as xgb


def save_model(model: xgb.XGBClassifier, feature_names: list[str], path: str) -> None:
    """Save the trained XGBoost model to <path>.json and the ordered
    feature names to a sidecar JSON file.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model object.
        feature_names (list[str]): List of ordered feature column names used during training.
        path (str): Target base file path (e.g., 'models/xgb_model.json').
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained XGBoost model binary/JSON representation
    model.save_model(model_path)

    # Create path for the feature names sidecar file (e.g., 'models/xgb_model_features.json')
    features_path = model_path.with_name(f"{model_path.stem}_features.json")

    # Save feature names to sidecar file
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)

    print(f"Model saved to '{model_path}' and feature names saved to '{features_path}'.")


def load_model(path: str) -> tuple[xgb.XGBClassifier, list[str]]:
    """Load the trained XGBoost model and its ordered feature list.

    Args:
        path (str): Target base file path of the saved model JSON.

    Returns:
        tuple[xgb.XGBClassifier, list[str]]: Loaded model and list of ordered feature names.
    """
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at '{model_path}'.")

    # Reconstruct empty model instance and load saved weights
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Locate and read sidecar feature names file
    features_path = model_path.with_name(f"{model_path.stem}_features.json")
    
    if not features_path.exists():
        raise FileNotFoundError(f"Feature sidecar file not found at '{features_path}'.")

    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return model, feature_names

def main() -> None:
    """Executes training, computes evaluation metrics, and displays a summary table."""
    # Load prepared feature datasets
    train_X, train_y, val_X, val_y, test_X, test_y, features = prep_data()

    # Train model with validation set early stopping
    model = train_xgb(train_X, train_y, val_X, val_y)

    # Compute predictions and primary metrics across splits
    train_loss, train_brier, train_acc, train_probs = evaluate_xgb(model, train_X, train_y)
    val_loss, val_brier, val_acc, val_probs = evaluate_xgb(model, val_X, val_y, split_name="Validation Set")
    test_loss, test_brier, test_acc, test_probs = evaluate_xgb(model, test_X, test_y, split_name="Test Set")

    # Compute class-wise AUC scores across splits
    train_auc_away, train_auc_draw, train_auc_home = get_all_aucs(train_y, train_probs)
    val_auc_away, val_auc_draw, val_auc_home = get_all_aucs(val_y, val_probs)
    test_auc_away, test_auc_draw, test_auc_home = get_all_aucs(test_y, test_probs)

    # Construct overall summary comparison table
    summary_data = {
        "split": ["train", "val", "test"],
        "acc": [train_acc, val_acc, test_acc],
        "brier": [train_brier, val_brier, test_brier],
        "logloss": [train_loss, val_loss, test_loss],
        "auc(draw)": [train_auc_draw, val_auc_draw, test_auc_draw],
        "auc(home)": [train_auc_home, val_auc_home, test_auc_home],
        "auc(away)": [train_auc_away, val_auc_away, test_auc_away],
    }

    df_metrics = pd.DataFrame(summary_data).set_index("split")

    print("\n========================= MODEL PERFORMANCE SUMMARY =========================")
    print(df_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=============================================================================\n")

    # Feature importance analysis
    get_feature_importance(model, features)

    # Export reliability calibration plots
    project_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = project_root / "reports" / "figures"
    class_labels = ["Away (A)", "Draw (D)", "Home (H)"]
    
    plot_reliability(
        y_true=test_y,
        proba=test_probs,
        class_names=class_labels,
        out_path=reports_dir / "reliability_curves.png",
        n_bins=10,
    )

    # Save trained model and feature ordering
    model_dir = project_root / "models"
    save_model(
        model=model,
        feature_names=features.tolist(),
        path=str(model_dir / "xgb_model.json")
    )

if __name__ == "__main__":
    main()