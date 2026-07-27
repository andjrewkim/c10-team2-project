from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
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

from src.extract_features import extract_mmwave_features, extract_window_features
from src.visualize import load_jsonl, load_session_csvs


def _load_frames(input_path: Path) -> dict[str, list[dict]]:
    if input_path.is_dir() and (input_path / "events.csv").exists():
        return load_session_csvs(input_path)
    if input_path.is_dir():
        session_dirs = sorted(input_path.glob("*/events.csv"))
        if session_dirs:
            result = {}
            for ev_csv in session_dirs:
                session_data = load_session_csvs(ev_csv.parent)
                for key, trial_frames in session_data.items():
                    unique_key = f"{ev_csv.parent.name}/{key}"
                    result[unique_key] = trial_frames
            return result
    if input_path.suffix == ".csv":
        frames_by_key: dict[str, list[dict]] = {}
        with open(input_path, newline="") as f:
            for row in csv.DictReader(f):
                frame = {
                    "timestamp": row["timestamp"],
                    "gesture": row["gesture"],
                    "trial": int(row["trial"]),
                    "elapsed": float(row["elapsed"]),
                }
                try:
                    points = json.loads(row.get("points", "[]")) if row.get("points") and row["points"] != "null" else []
                except json.JSONDecodeError:
                    points = []
                mm_data = {
                    "num_points": int(row.get("num_points", len(points))),
                    "points": points,
                    "range_profile": None,
                    "motion_score": float(row.get("motion_score", 0.0)),
                }
                frame["mmwave"] = {"data": mm_data, "confidence": float(row.get("confidence", 0.0)), "sensor_type": "mmwave"}
                key = f"{row['gesture']}_t{row['trial']}"
                frames_by_key.setdefault(key, []).append(frame)
        return frames_by_key
    if input_path.suffix == ".jsonl":
        frames = load_jsonl(input_path)
        gesture = frames[0].get("gesture", "?") if frames else "?"
        trial = frames[0].get("trial", 0) if frames else 0
        return {f"{gesture}_t{trial}": frames}
    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
        frames_list = data.get("frames", data if isinstance(data, list) else [])
        result = {}
        for frame in frames_list:
            g = frame.get("gesture", "?")
            t = frame.get("trial", 0)
            result.setdefault(f"{g}_t{t}", []).append(frame)
        return result
    raise ValueError(f"Unrecognized input: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained gesture model on session data")
    parser.add_argument("--model", default="models/best_model.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--input", default="data/raw",
                        help="Session folder, combined CSV, or combined JSON")
    parser.add_argument("--window", type=int, default=10,
                        help="Window size in frames")
    parser.add_argument("--stride", type=int, default=5,
                        help="Window stride")
    parser.add_argument("--output", default="results",
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"  Model type: {type(pipeline).__name__}")

    if hasattr(pipeline, "classes_"):
        model_classes = list(pipeline.classes_)
        print(f"  Model classes: {model_classes}")
    else:
        model_classes = None

    print(f"\nLoading data from: {args.input}")
    frames_by_trial = _load_frames(Path(args.input))
    print(f"  Loaded {len(frames_by_trial)} trials")

    all_X: list[np.ndarray] = []
    all_y: list[str] = []
    trial_labels: list[str] = []

    for trial_key, trial_frames in frames_by_trial.items():
        if not trial_frames:
            continue
        gesture = trial_frames[0].get("gesture", "unknown")
        X_win, y_win, _ = extract_window_features(trial_frames, window_size=args.window, stride=args.stride)
        if len(X_win) == 0:
            print(f"  Skipping {trial_key}: too few frames ({len(trial_frames)}) for window size {args.window}")
            continue
        all_X.append(X_win)
        all_y.extend(y_win)
        trial_labels.extend([trial_key] * len(X_win))

    if not all_X:
        print("No valid windows extracted.")
        return

    X = np.vstack(all_X)
    print(f"  Total windows: {len(X)}")

    label_set = sorted(set(all_y))
    label_to_int = {g: i for i, g in enumerate(label_set)}
    y_true = np.array([label_to_int[g] for g in all_y])

    print(f"\n--- Making predictions ---")
    y_pred = pipeline.predict(X)
    y_pred_labels = [label_set[int(p)] if isinstance(p, (int, np.integer)) else p for p in y_pred]

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)
    else:
        y_prob = None

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix ({len(label_set)}x{len(label_set)}):")
    print("Rows: true, Columns: predicted")
    print("-" * 50)
    header = " " * 12 + "".join(f"{g:>10}" for g in label_set)
    print(header)
    for i, row in enumerate(cm):
        label = label_set[i] if i < len(label_set) else f"class_{i}"
        row_str = " ".join(f"{v:>10}" for v in row)
        print(f"{label:>10}  {row_str}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_set, zero_division=0))

    gesture_counts = Counter(all_y)
    print(f"\nTest samples per gesture:")
    for g, c in sorted(gesture_counts.items()):
        print(f"  {g}: {c} windows")

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
                    xticklabels=label_set, yticklabels=label_set)
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(label_set))
        plt.xticks(tick_marks, label_set, rotation=45)
        plt.yticks(tick_marks, label_set)
        thresh = cm.max() / 2.0
        for i in range(len(label_set)):
            for j in range(len(label_set)):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nSaved: {cm_path}")

    if y_prob is not None:
        plt.figure(figsize=(10, 6))
        n_classes = y_prob.shape[1]
        for i in range(min(n_classes, min(4, n_classes))):
            label = label_set[i] if i < len(label_set) else f"class_{i}"
            plt.plot(y_prob[:, i], label=label)
        plt.title("Prediction Probabilities")
        plt.xlabel("Test sample")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        prob_path = figures_dir / "prediction_probabilities.png"
        plt.savefig(prob_path, dpi=150)
        print(f"Saved: {prob_path}")

    results = {
        "model": str(model_path),
        "input": str(args.input),
        "num_windows": len(X),
        "num_classes": len(label_set),
        "classes": label_set,
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "samples_per_class": dict(gesture_counts),
    }
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()