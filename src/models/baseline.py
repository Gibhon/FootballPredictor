"""Baseline Model Evaluation Module for Football Match Outcome Prediction."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

# Resolve project root and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# use absolute package imports
from src.models.xg_model import (
    evaluate_xgb,
    get_all_aucs,
    multiclass_brier_score,
    plot_reliability,
    prep_data,
)


def evaluate_baseline() -> None:
    # Load prepared feature datasets
    train_X, train_y, val_X, val_y, test_X, test_y, _ = prep_data()

    # Fit dummy classifier using empirical class prior distributions
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_X, train_y)

    # Compute predictions and probabilities across splits
    train_probs = dummy.predict_proba(train_X)
    train_preds = dummy.predict(train_X)
    train_loss = float(evaluate_xgb(dummy, train_X, train_y)[0])
    train_brier = multiclass_brier_score(train_y, train_probs)
    train_acc = float(np.mean(train_preds == train_y))

    val_loss, val_brier, val_acc, val_probs = evaluate_xgb(
        dummy, val_X, val_y, split_name="Dummy Baseline (Val Set)"
    )
    test_loss, test_brier, test_acc, test_probs = evaluate_xgb(
        dummy, test_X, test_y, split_name="Dummy Baseline (Test Set)"
    )

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

    print("\n==================== DUMMY BASELINE PERFORMANCE SUMMARY ====================")
    print(df_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
    print("============================================================================\n")

    # Export reliability curves for dummy baseline
    reports_dir = PROJECT_ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    class_labels = ["Away (A)", "Draw (D)", "Home (H)"]

    plot_reliability(
        y_true=test_y,
        proba=test_probs,
        class_names=class_labels,
        out_path=reports_dir / "dummy_baseline_reliability_curves.png",
        n_bins=10,
    )


if __name__ == "__main__":
    evaluate_baseline()