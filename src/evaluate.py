from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.extract_features import extract_window_features
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
        return _load_csv_as_trials(input_path)
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


def _load_csv_as_trials(csv_path: Path) -> dict[str, list[dict]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames or []
        base_cols = {"frame_index", "timestamp", "gesture", "trial", "elapsed", "dataset_source"}
        sensor_prefixes = set()
        for col in all_cols:
            if col.endswith("_confidence") and col not in base_cols:
                sensor_prefixes.add(col[:-len("_confidence")])

        frames_by_key: dict[str, list[dict]] = {}
        for row in reader:
            frame = {
                "timestamp": row["timestamp"],
                "gesture": row["gesture"],
                "trial": int(row["trial"]),
                "elapsed": float(row["elapsed"]),
            }
            for prefix in sorted(sensor_prefixes):
                data = {}
                for col in all_cols:
                    if col.startswith(prefix + "_") and col != f"{prefix}_confidence":
                        field_name = col[len(prefix) + 1:]
                        val = row.get(col, "")
                        if val == "" or val == "null":
                            data[field_name] = None
                        elif val.startswith(("[", "{")):
                            try:
                                data[field_name] = json.loads(val)
                            except json.JSONDecodeError:
                                data[field_name] = val
                        else:
                            try:
                                data[field_name] = int(val)
                            except ValueError:
                                try:
                                    data[field_name] = float(val)
                                except ValueError:
                                    data[field_name] = val
                confidence = float(row.get(f"{prefix}_confidence", 0.0))
                frame[prefix] = {
                    "data": data,
                    "confidence": confidence,
                    "sensor_type": prefix,
                }
            key = f"{row['gesture']}_t{row['trial']}"
            frames_by_key.setdefault(key, []).append(frame)
        return frames_by_key


_PRETTY_NAMES = {
    "random_forest": "Random Forest",
    "knn": "k-Nearest Neighbors",
    "svm_rbf": "SVM with rbf kernel",
    "svm_linear": "SVM Linear",
}


def _pretty_name(name: str) -> str:
    return _PRETTY_NAMES.get(name, name)


def _find_latest_model(models_dir: str = "models", pattern: str = "best_model.pkl") -> Path | None:
    models_path = Path(models_dir)
    candidates = sorted(models_path.glob(f"train_*/{pattern}"))
    return candidates[-1] if candidates else None


GENERATE_OPTIONS = {"cm", "probs", "per_class", "feature_importance", "tsne", "pca"}


def _parse_generate(value: str | list[str] | None) -> set[str]:
    if not value:
        return GENERATE_OPTIONS
    if isinstance(value, str):
        value = [value]
    value = set(value)
    if "all" in value:
        return GENERATE_OPTIONS
    return value & GENERATE_OPTIONS


def _plot_per_class_metrics(y_true, y_pred, label_set, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(label_set))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(label_set) * 0.5), 5))
    ax.bar(x - w, precisions, w, label="Precision")
    ax.bar(x, recalls, w, label="Recall")
    ax.bar(x + w, f1s, w, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(label_set, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, F1-Score")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "per_class_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_feature_importance(pipeline, feature_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clf = pipeline.named_steps.get("clf", pipeline)
    if not hasattr(clf, "feature_importances_"):
        print("  Skipping feature importance: model has no feature_importances_")
        return
    importances = clf.feature_importances_
    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"feat_{i}" for i in range(len(importances))]
    top_k = min(20, len(importances))
    idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
    ax.barh(range(top_k), importances[idx])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_tsne(X, y, label_set, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(2000, len(X))
    X_sub = X[:n]
    y_sub = y[:n]
    perplexity = min(30, max(5, n // 5))
    print(f"  Running t-SNE (perplexity={perplexity}, n={n})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="random")
    embed = tsne.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_set)))
    for i, label in enumerate(label_set):
        mask = y_sub == i
        if mask.any():
            ax.scatter(embed[mask, 0], embed[mask, 1], c=[colors[i]], label=label, s=10, alpha=0.7)
    ax.legend(fontsize=6, loc="best")
    ax.set_title("t-SNE Embedding of Test Features")
    fig.tight_layout()
    path = out_dir / "tsne.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_pca(X, y, label_set, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("  Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    embed = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_set)))
    for i, label in enumerate(label_set):
        mask = y == i
        if mask.any():
            ax.scatter(embed[mask, 0], embed[mask, 1], c=[colors[i]], label=label, s=10, alpha=0.7)
    ax.legend(fontsize=6, loc="best")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("PCA of Test Features")
    fig.tight_layout()
    path = out_dir / "pca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained gesture model on session data")
    parser.add_argument("--model", default=None,
                        help="Path to trained model pickle (default: latest train_*/best_model.pkl)")
    parser.add_argument("--input", default=None,
                        help="Features NPZ (from extract_features.py) or raw session data (dir, CSV, JSONL)")
    parser.add_argument("--window", type=int, default=10,
                        help="Window size in frames (used only when loading raw data)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Window stride (used only when loading raw data)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: results/evaluate_{features_timestamp})")
    parser.add_argument("--generate", nargs="+", default=["all"],
                        choices=sorted(GENERATE_OPTIONS | {"all"}),
                        help="Graphs to generate (default: all)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else _find_latest_model()
    if model_path is None:
        print("Error: no model specified and no train_*/best_model.pkl found in models/")
        return
    if not model_path.exists():
        print(f"Error: model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    with open(model_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        pipeline = raw["pipeline"]
        gestures = raw.get("gestures", [])
        label_map = raw.get("label_map", {})
        feature_names = raw.get("feature_names", [])
    else:
        pipeline = raw
        gestures = []
        label_map = {}
        feature_names = []
    int_to_label = {v: k for k, v in label_map.items()}

    input_path = Path(args.input) if args.input else None
    eval_timestamp = None
    if input_path and input_path.suffix == ".npz":
        # Load from features NPZ
        print(f"Loading features: {input_path}")
        data = np.load(input_path, allow_pickle=True)
        X_test = data["X_test"]
        y_test = data["y_test"]
        feature_names = data["feature_names"].tolist() if "feature_names" in data else []
        if not gestures:
            gestures = data["gestures"].tolist() if "gestures" in data else []
        if not label_map:
            label_map = data["label_map"].item() if "label_map" in data else {}
            int_to_label = {v: k for k, v in label_map.items()}
        stem = input_path.stem
        if stem.startswith("features_"):
            eval_timestamp = stem.removeprefix("features_")
    elif input_path and input_path.exists():
        # Load raw data and extract features
        print(f"Loading raw data: {input_path}")
        trials = _load_frames(input_path)
        all_frames = []
        for trial_frames in trials.values():
            all_frames.extend(trial_frames)
        print(f"  Loaded {len(all_frames)} frames across {len(trials)} trials")
        X_test, y_str, feature_names = extract_window_features(
            all_frames, window_size=args.window, stride=args.stride,
        )
        # Map string labels to ints using the model's label_map
        if not label_map:
            # Build label map from data if model has none
            unique = sorted(set(y_str))
            label_map = {g: i for i, g in enumerate(unique)}
            int_to_label = {i: g for g, i in label_map.items()}
        y_test = np.array([label_map.get(g, -1) for g in y_str])
        # Filter out labels unknown to the model
        known = y_test != -1
        if not known.all():
            print(f"  Warning: dropping {(~known).sum()} windows with unknown labels")
        X_test = X_test[known]
        y_test = y_test[known]
    else:
        print("Error: no input specified and no features NPZ found alongside model")
        print("Use --input to point to a features .npz file or raw session data")
        return

    if not eval_timestamp:
        from datetime import datetime, timezone
        eval_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path("results/figures") / f"evaluate_{eval_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {len(gestures)} ({', '.join(gestures) if gestures else 'unknown'})")

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)
    else:
        y_prob = None

    try:
        from src.collect import ALL_GESTURES
        _canonical = ALL_GESTURES
        present = set(gestures)
        ordered_gestures = [g for g in _canonical if g in present]
        ordered_gestures += [g for g in gestures if g not in ordered_gestures]
    except ImportError:
        ordered_gestures = gestures

    y_true = y_test
    label_set = [int_to_label.get(i, f"class_{i}") for i in sorted(set(y_true) | set(y_pred))]
    if ordered_gestures:
        present = set(label_set)
        label_set = [g for g in ordered_gestures if g in present] + [g for g in label_set if g not in ordered_gestures]

    generate_set = _parse_generate(args.generate)
    print(f"Generating graphs: {', '.join(sorted(generate_set))}")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    label_ints = [label_map[g] for g in ordered_gestures if g in label_map] if label_map else None
    cm = confusion_matrix(y_true, y_pred, labels=label_ints)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
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

    gesture_counts = Counter(y_true)
    print(f"\nTest samples per gesture:")
    for g, c in sorted(gesture_counts.items()):
        label = int_to_label.get(g, f"class_{g}")
        print(f"  {label}: {c} windows")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    print()

    if "cm" in generate_set:
        n = len(label_set)
        fig = plt.figure(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
        if has_sns:
            ax = sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                             xticklabels=label_set, yticklabels=label_set,
                             annot_kws={"fontsize": max(6, min(14, 14 - n * 0.3))})
            ax.set_xlabel("Predicted Gesture (%)")
            ax.set_ylabel("Actual Gesture (%)")
        else:
            plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()
            tick_marks = np.arange(n)
            plt.xticks(tick_marks, label_set, rotation=45, fontsize=max(6, min(12, 12 - n * 0.25)))
            plt.yticks(tick_marks, label_set, fontsize=max(6, min(12, 12 - n * 0.25)))
            thresh = cm_norm.max() / 2.0
            for i in range(n):
                for j in range(n):
                    plt.text(j, i, f"{cm_norm[i, j]:.1f}",
                             ha="center", va="center",
                             fontsize=max(6, min(12, 12 - n * 0.25)),
                             color="white" if cm_norm[i, j] > thresh else "black")
            plt.xlabel("Predicted Gesture (%)")
            plt.ylabel("Actual Gesture (%)")
        _model_title = _pretty_name(model_path.parent.name) if model_path.parent.name in _PRETTY_NAMES else ""
        if _model_title:
            plt.suptitle(_model_title, fontsize=14)
        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {cm_path}")

    if "probs" in generate_set and y_prob is not None:
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
        prob_path = out_dir / "prediction_probabilities.png"
        plt.savefig(prob_path, dpi=150)
        plt.close()
        print(f"  Saved: {prob_path}")

    if "per_class" in generate_set:
        _plot_per_class_metrics(y_true, y_pred, label_set, out_dir)

    if "feature_importance" in generate_set:
        _plot_feature_importance(pipeline, feature_names, out_dir)

    if "tsne" in generate_set:
        _plot_tsne(X_test, y_true, label_set, out_dir)

    if "pca" in generate_set:
        _plot_pca(X_test, y_true, label_set, out_dir)

    results = {
        "model": str(model_path),
        "input": str(args.input) if args.input else "",
        "num_windows": len(X_test),
        "num_classes": len(label_set),
        "classes": label_set,
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "samples_per_class": {int_to_label.get(int(k), k): int(v) for k, v in gesture_counts.items()},
    }
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
