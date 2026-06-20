"""
main.py — Credit Card Fraud Detection: full ML comparison pipeline.

Usage:
    python main.py --data creditcard.csv
    python main.py --data creditcard.csv --output results --no-smote

Dataset:
    Download creditcard.csv from Kaggle:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Place the file in the same directory as this script before running.
"""

import argparse
import os
import sys

from preprocessing import load_data, explore_data, preprocess, split_and_resample
from models import get_models, train_models
from evaluation import evaluate_model, compare_models
from visualizations import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_feature_importance,
)


def run(data_path: str, output_dir: str = "results", use_smote: bool = True):
    sep = "=" * 70

    print(sep)
    print("  CREDIT CARD FRAUD DETECTION — ML COMPARISON STUDY")
    print("  IU Portfolio Project: Computer Science (CSEMCSPCSP01)")
    print(sep)

    # ------------------------------------------------------------------ #
    # 1. Load and explore data
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading dataset ...")
    df = load_data(data_path)
    explore_data(df)
    plot_class_distribution(df["Class"], save_path=output_dir)

    # ------------------------------------------------------------------ #
    # 2. Preprocess
    # ------------------------------------------------------------------ #
    print("\n[2/5] Preprocessing ...")
    X, y = preprocess(df)

    # ------------------------------------------------------------------ #
    # 3. Split & resample
    # ------------------------------------------------------------------ #
    print("\n[3/5] Splitting and resampling ...")
    X_train, X_test, y_train, y_test = split_and_resample(
        X, y, use_smote=use_smote
    )

    # ------------------------------------------------------------------ #
    # 4. Train models
    # ------------------------------------------------------------------ #
    print("\n[4/5] Training models ...")
    models = get_models()
    trained = train_models(models, X_train, y_train)

    # ------------------------------------------------------------------ #
    # 5. Evaluate and visualize
    # ------------------------------------------------------------------ #
    print("\n[5/5] Evaluating models ...")
    all_metrics = []
    pred_data = []   # (name, y_pred, y_test) for confusion matrices
    prob_data = []   # (name, y_test, y_prob) for ROC curves
    feature_names = list(X.columns)

    for name, data in trained.items():
        model = data["model"]
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        pred_data.append((name, y_pred, y_test))
        prob_data.append((name, y_test, y_prob))
        plot_feature_importance(model, feature_names, name, save_path=output_dir)

    # Comparison table
    comparison_df = compare_models(all_metrics)
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv")
    print(f"  Saved: {output_dir}/model_comparison.csv")

    # Group plots
    plot_confusion_matrices(pred_data, save_path=output_dir)
    plot_roc_curves(prob_data, save_path=output_dir)
    plot_metrics_comparison(comparison_df, save_path=output_dir)

    print(f"\n{sep}")
    print(f"  Analysis complete.  All outputs saved to '{output_dir}/'")
    print(sep)
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — ML Comparison"
    )
    parser.add_argument(
        "--data", default="creditcard.csv",
        help="Path to creditcard.csv (default: ./creditcard.csv)"
    )
    parser.add_argument(
        "--output", default="results",
        help="Directory for output plots and CSV (default: results/)"
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE resampling (run on raw imbalanced data)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"\nERROR: Dataset not found at '{args.data}'")
        print("Download creditcard.csv from:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Then run:  python main.py --data creditcard.csv\n")
        sys.exit(1)

    run(args.data, output_dir=args.output, use_smote=not args.no_smote)


if __name__ == "__main__":
    main()
