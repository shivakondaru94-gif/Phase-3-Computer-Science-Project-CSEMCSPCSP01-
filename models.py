"""
models.py — Model definitions and training.
"""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models() -> dict:
    """Return a dict of {name: untrained model} with fixed hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """
    Train each model in the dict and return:
      {name: {"model": trained_model, "train_time": seconds}}
    """
    trained = {}
    for name, model in models.items():
        print(f"  Training {name} ...", end="", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        trained[name] = {"model": model, "train_time": elapsed}
        print(f" done ({elapsed:.1f}s)")
    return trained
