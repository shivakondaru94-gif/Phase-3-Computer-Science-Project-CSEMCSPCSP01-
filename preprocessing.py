"""
preprocessing.py — Data loading, exploration, scaling, and SMOTE resampling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card fraud CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print exploratory statistics about the dataset."""
    print(f"Dataset shape:    {df.shape}")
    print(f"Missing values:   {df.isnull().sum().sum()}")
    print(f"\nClass distribution:")
    counts = df["Class"].value_counts()
    print(f"  Legitimate:  {counts[0]:,}  ({counts[0]/len(df)*100:.3f}%)")
    print(f"  Fraudulent:  {counts[1]:,}  ({counts[1]/len(df)*100:.3f}%)")
    print(f"\nAmount stats:")
    print(df["Amount"].describe().to_string())


def preprocess(df: pd.DataFrame):
    """
    Scale Amount and Time; drop originals.
    Returns (X, y) as numpy arrays with feature names attached to X as DataFrame.
    """
    scaler = StandardScaler()
    df = df.copy()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def split_and_resample(X, y, test_size: float = 0.2, random_state: int = 42,
                       use_smote: bool = True):
    """
    Stratified train/test split, then optional SMOTE on training set only.

    SMOTE is NEVER applied to the test set — this prevents data leakage and
    ensures evaluation reflects the real-world class distribution.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"Train fraud count: {y_train.sum()}  |  Test fraud count: {y_test.sum()}")

    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        dist = pd.Series(y_train).value_counts()
        print(f"\nAfter SMOTE — training set:")
        print(f"  Legitimate: {dist[0]:,}  |  Fraudulent: {dist[1]:,}")

    return X_train, X_test, y_train, y_test
