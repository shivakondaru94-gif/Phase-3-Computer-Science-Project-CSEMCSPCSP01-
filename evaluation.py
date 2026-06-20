"""
evaluation.py — Metrics computation and model comparison reporting.
"""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> tuple:
    """
    Evaluate one trained model on the test set.

    Returns:
        metrics_dict  — dict of scalar scores
        y_pred        — binary predictions array
        y_prob        — fraud probability scores array
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Avg Precision": average_precision_score(y_test, y_prob),
    }

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Fraudulent"],
        digits=4,
    ))
    print(f"  ROC-AUC:           {metrics['ROC-AUC']:.4f}")
    print(f"  Average Precision: {metrics['Avg Precision']:.4f}")

    return metrics, y_pred, y_prob


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Build and print a comparison table from a list of metrics dicts.
    Returns a DataFrame indexed by model name.
    """
    df = pd.DataFrame(metrics_list).set_index("Model").round(4)
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(df.to_string())
    print(f"{'='*70}\n")
    return df
