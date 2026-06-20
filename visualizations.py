"""
visualizations.py — Plotting functions. All figures are saved to disk (PNG).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_class_distribution(y, save_path: str = "results") -> None:
    """Bar chart of the raw dataset class imbalance."""
    _ensure(save_path)
    counts = y.value_counts().sort_index()
    labels = ["Legitimate", "Fraudulent"]
    colors = ["steelblue", "crimson"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution in Dataset", fontsize=13)
    ax.set_ylabel("Number of Transactions")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    plt.tight_layout()
    out = f"{save_path}/class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrices(pred_data: list, save_path: str = "results") -> None:
    """
    Side-by-side confusion matrices.

    pred_data: list of (model_name, y_pred, y_test) tuples
    """
    _ensure(save_path)
    n = len(pred_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred, y_test) in zip(axes, pred_data):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=["Legitimate", "Fraudulent"],
            yticklabels=["Legitimate", "Fraudulent"],
        )
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices — Model Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{save_path}/confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_roc_curves(prob_data: list, save_path: str = "results") -> None:
    """
    Overlaid ROC curves with AUC labels.

    prob_data: list of (model_name, y_test, y_prob) tuples
    """
    _ensure(save_path)
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, y_test, y_prob), color in zip(prob_data, colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = f"{save_path}/roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_metrics_comparison(comparison_df, save_path: str = "results") -> None:
    """Grouped bar chart comparing Precision, Recall, F1, ROC-AUC per model."""
    _ensure(save_path)
    metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    metrics = [m for m in metrics if m in comparison_df.columns]

    x = np.arange(len(comparison_df))
    width = 0.2
    colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, comparison_df[metric], width,
                      label=metric, color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{save_path}/metrics_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(model, feature_names: list, model_name: str,
                            top_n: int = 15, save_path: str = "results") -> None:
    """Bar chart of top-N feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        return
    _ensure(save_path)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices], color="steelblue", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance (Gini)")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    out = f"{save_path}/feature_importance_{safe_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
