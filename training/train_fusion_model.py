#!/usr/bin/env python3
"""Train and compare ML fusion models, saving the best one.

Compares:
    - Logistic regression (baseline)
    - k-Nearest Neighbours (non-parametric)
    - Multi-Layer Perceptron (small neural net)

Prints accuracy and confusion matrix for each, then saves the
best-performing model (by accuracy) to ``fusion/model.pkl``.

Usage
-----
    python -m training.train_fusion_model \\
        --dataset data/processed/dataset.npz \\
        --output fusion/model.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    X_train: np.ndarray = data["X_train"]
    X_test: np.ndarray = data["X_test"]
    y_train: np.ndarray = data["y_train"]
    y_test: np.ndarray = data["y_test"]
    col_names: list[str] = list(data["col_names"])
    label_names: list[str] = list(data["label_names"])
    print(f"[train_fusion] Loaded dataset: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    print(f"[train_fusion] Labels ({len(label_names)}): {label_names}")
    return X_train, X_test, y_train, y_test, col_names, label_names


def evaluate_model(name: str, clf, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  {'─' * 35}")
    print(f"  {name}")
    print(f"  {'─' * 35}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
    print(f"  CV accuracy (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare ML fusion models.")
    parser.add_argument("--dataset", default="data/processed/dataset.npz", help="Path to dataset.npz")
    parser.add_argument("--output", default="fusion/model.pkl", help="Where to save the best model")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"[train_fusion] Dataset not found: {args.dataset}")
        print("[train_fusion] Run `python -m training.prepare_dataset` first.")
        sys.exit(1)

    X_train, X_test, y_train, y_test, col_names, label_names = load_dataset(args.dataset)

    if X_train.shape[0] < 5:
        print("[train_fusion] Too few training samples — need at least 5")
        sys.exit(1)

    models: list[tuple[str, object]] = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("kNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
        (
            "MLP (2×32)",
            MLPClassifier(
                hidden_layer_sizes=(32, 32), activation="relu",
                max_iter=500, random_state=42, early_stopping=True,
            ),
        ),
    ]

    print(f"\n{'=' * 42}")
    print("  Model Comparison")
    print(f"{'=' * 42}")

    best_acc = -1.0
    best_clf = None
    best_name = ""

    for name, clf in models:
        acc = evaluate_model(name, clf, X_train, y_train, X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            best_name = name

    # Save best model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": best_clf,
        "model_name": best_name,
        "accuracy": best_acc,
        "label_names": label_names,
        "col_names": col_names,
    }
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  ✅ Best model: {best_name} (acc={best_acc:.4f}) → {output_path}")


if __name__ == "__main__":
    main()
