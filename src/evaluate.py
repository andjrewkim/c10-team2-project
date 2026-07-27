from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained gesture model")
    parser.add_argument("--model", default="models/best_model.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--features", default="data/processed/features.npz",
                        help="Path to test features")
    parser.add_argument("--output", default="results",
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    model_path = Path(args.model)
    features_path = Path(args.features)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: model not found: {model_path}")
        return
    if not features_path.exists():
        print(f"Error: features not found: {features_path}")
        return

    with open(model_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        pipeline = raw["pipeline"]
        gestures = raw.get("gestures", [])
        label_map = raw.get("label_map", {})
    else:
        pipeline = raw
        gestures = []
        label_map = {}

    data = np.load(features_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    if not gestures:
        gestures = data["gestures"].tolist() if "gestures" in data else []
    if not label_map:
        label_map = data["label_map"].item() if "label_map" in data else {}
    int_to_label = {v: k for k, v in label_map.items()}

    print(f"Loaded model: {model_path}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {len(gestures)}")

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix ({len(gestures)}x{len(gestures)}):")
    print("Rows: true, Columns: predicted")
    print("-" * 50)
    header = " " * 12 + "".join(f"{g:>10}" for g in gestures)
    print(header)
    for i, row in enumerate(cm):
        label = gestures[i] if i < len(gestures) else f"class_{i}"
        row_str = " ".join(f"{v:>10}" for v in row)
        print(f"{label:>10}  {row_str}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=gestures,
        zero_division=0,
    ))

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    plt.figure(figsize=(10, 8))
    if has_sns:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=gestures, yticklabels=gestures)
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(gestures))
        plt.xticks(tick_marks, gestures, rotation=45)
        plt.yticks(tick_marks, gestures)
        thresh = cm.max() / 2.0
        for i in range(len(gestures)):
            for j in range(len(gestures)):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved: {cm_path}")

    if y_prob is not None:
        plt.figure(figsize=(10, 6))
        n_classes = len(gestures)
        for i in range(min(n_classes, 4)):
            plt.plot(y_prob[:, i], label=f"{gestures[i]}" if i < len(gestures) else f"class_{i}")
        plt.title("Prediction Probabilities (first 4 classes)")
        plt.xlabel("Test sample")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        prob_path = figures_dir / "prediction_probabilities.png"
        plt.savefig(prob_path, dpi=150)
        print(f"Probability plot saved: {prob_path}")

    results = {
        "model": str(model_path),
        "num_test_samples": len(X_test),
        "num_classes": len(gestures),
        "classes": gestures,
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
