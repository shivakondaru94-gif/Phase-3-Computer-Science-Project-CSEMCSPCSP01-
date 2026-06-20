# Credit Card Fraud Detection Using Machine Learning

A comparative study of Logistic Regression, Random Forest, and XGBoost for detecting credit card fraud, with SMOTE-based class imbalance handling.

**IU Portfolio Project — CSEMCSPCSP01**

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (Machine Learning Group, ULB).

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.172%) |
| Features | V1–V28 (PCA), Time, Amount |
| Target | Class (0 = legitimate, 1 = fraud) |
| Missing values | None |

### Download

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to the dataset page linked above and click **Download**
3. Extract and place `creditcard.csv` in this directory

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9 or higher is required.

---

## Usage

```bash
# Run full pipeline with SMOTE (recommended)
python main.py --data creditcard.csv

# Specify a custom output directory
python main.py --data creditcard.csv --output my_results

# Run without SMOTE (to observe effect of class imbalance)
python main.py --data creditcard.csv --no-smote
```

---

## Project Structure

```
code/
├── main.py              # Entry point — orchestrates the full pipeline
├── preprocessing.py     # Data loading, scaling, SMOTE resampling
├── models.py            # Model definitions (LR, RF, XGBoost) and training
├── evaluation.py        # Metrics computation and comparison table
├── visualizations.py    # All plots saved as PNG to results/
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Output

After running, `results/` contains:

| File | Description |
|---|---|
| `model_comparison.csv` | Precision, Recall, F1, ROC-AUC for all models |
| `class_distribution.png` | Dataset class imbalance bar chart |
| `confusion_matrices.png` | Side-by-side confusion matrices |
| `roc_curves.png` | ROC curves with AUC for all models |
| `metrics_comparison.png` | Grouped bar chart of all metrics |
| `feature_importance_Random_Forest.png` | Top 15 features (RF) |
| `feature_importance_XGBoost.png` | Top 15 features (XGBoost) |

---

## Representative Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.874 | 0.621 | 0.727 | 0.969 |
| Random Forest | 0.952 | 0.787 | 0.862 | 0.979 |
| **XGBoost** | **0.933** | **0.827** | **0.877** | **0.982** |

*Exact values vary slightly with random seed; XGBoost consistently achieves the highest F1 and recall.*

---

## Research Report

See `../Phase3_Final_Report.txt` for the full research report covering methodology, technical background, results analysis, and conclusions.
