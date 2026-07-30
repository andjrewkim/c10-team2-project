from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSIFIER_NAMES = ["random_forest", "knn", "svm_rbf", "svm_linear"]


def _make_classifier(name: str, **params) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            p=params.get("p", 2),
            n_jobs=-1,
        )
    elif name == "svm_rbf":
        return SVC(
            kernel="rbf", C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=True, random_state=params.get("random_state", 42),
        )
    elif name == "svm_linear":
        return SVC(
            kernel="linear", C=params.get("C", 1.0),
            probability=True, random_state=params.get("random_state", 42),
        )
    raise ValueError(f"Unknown classifier: {name}")


def _pretty_name(name: str) -> str:
    return {
        "random_forest": "Random Forest",
        "knn": "k-Nearest Neighbors",
        "svm_rbf": "SVM with rbf kernel",
        "svm_linear": "SVM Linear",
    }.get(name, name)


def _param_label(name: str, params: dict) -> str:
    if name == "random_forest":
        return f"rf_n{params['n_estimators']}_d{params['max_depth']}"
    elif name == "knn":
        return f"knn_n{params['n_neighbors']}_w{params['weights']}_p{params['p']}"
    elif name == "svm_rbf":
        g = params['gamma']
        return f"svm_rbf_c{params['C']}_g{g}"
    elif name == "svm_linear":
        return f"svm_linear_c{params['C']}"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture classification models")
    parser.add_argument("--input", default=None,
                        help="Input feature NPZ file (default: latest features_*.npz in data/processed)")
    parser.add_argument("--output", default="models",
                        help="Base output directory (a train_{timestamp} subfolder is created inside)")
    parser.add_argument("--output-dir", default=None,
                        help="Exact output directory (overrides --output, no timestamp subfolder added)")
    parser.add_argument("--output-name", default=None,
                        help="Optional prefix for model files (e.g. 'mmwave' → 'models/mmwave_rf.pkl')")
    parser.add_argument("--classifiers", nargs="+",
                        default=CLASSIFIER_NAMES,
                        choices=CLASSIFIER_NAMES,
                        help="Classifiers to train and compare")
    parser.add_argument("--sensors", nargs="+", default=None,
                        choices=["mmwave", "imu", "uwb"],
                        help="Filter to features from these sensors only (default: all)")

    # Random Forest params
    parser.add_argument("--rf-n-estimators", type=int, nargs="+", default=[100],
                        help="Number of trees (space-separated for sweep, default: 100)")
    parser.add_argument("--rf-max-depth", type=int, nargs="+", default=[10],
                        help="Max tree depth (space-separated for sweep, default: 10)")
    parser.add_argument("--rf-random-state", type=int, default=42,
                        help="Random state for random forest (default: 42)")

    # KNN params
    parser.add_argument("--knn-n-neighbors", type=int, nargs="+", default=[5],
                        help="Number of neighbors (space-separated for sweep, default: 5)")
    parser.add_argument("--knn-weights", nargs="+", default=["uniform"],
                        choices=["uniform", "distance"],
                        help="Weight function (space-separated for sweep, default: uniform)")
    parser.add_argument("--knn-p", type=int, nargs="+", default=[2], choices=[1, 2],
                        help="Distance metric (space-separated for sweep, default: 2)")

    # SVM params (shared by rbf and linear)
    parser.add_argument("--svm-c", type=float, nargs="+", default=[1.0],
                        help="Regularization C (space-separated for sweep, default: 1.0)")
    parser.add_argument("--svm-gamma", nargs="+", default=["scale"],
                        help="Kernel gamma (space-separated for sweep, default: scale)")
    parser.add_argument("--svm-random-state", type=int, default=42,
                        help="Random state for SVM (default: 42)")

    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(Path("data/processed").glob("features_*.npz"))
        if not candidates:
            print("Error: no features_*.npz found in data/processed/")
            print("Run extract_features.py first or specify --input")
            return
        input_path = candidates[-1]

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run extract_features.py first.")
        return

    data_ts = input_path.stem.removeprefix("features_")

    data = np.load(input_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist() if "feature_names" in data else []
    gestures = data["gestures"].tolist() if "gestures" in data else []
    label_map = data["label_map"].item() if "label_map" in data else {}

    if args.sensors:
        prefix_map = {"mmwave": "mm_", "imu": "imu_", "uwb": "uwb"}
        keep = [i for i, fn in enumerate(feature_names)
                if any(fn.startswith(prefix_map[s]) for s in args.sensors)]
        if not keep:
            print(f"Error: no features found for sensors {args.sensors}")
            return
        feature_names = [feature_names[i] for i in keep]
        X_train = X_train[:, keep]
        X_test = X_test[:, keep]
        print(f"Filtered to sensors {args.sensors}: {len(feature_names)} features")

    try:
        from src.collect import ALL_GESTURES
        _canonical = ALL_GESTURES
        present = set(gestures)
        ordered_gestures = [g for g in _canonical if g in present]
        ordered_gestures += [g for g in gestures if g not in ordered_gestures]
    except ImportError:
        ordered_gestures = gestures

    print(f"Training data: {X_train.shape}")
    print(f"Test data:     {X_test.shape}")
    print(f"Classes:       {len(gestures)} ({', '.join(gestures)})")
    print()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.output) / f"train_{data_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    best_score = 0.0
    best_name = ""
    best_classifier_name = ""
    best_pipeline = None

    for name in args.classifiers:
        if name not in CLASSIFIER_NAMES:
            print(f"  Unknown classifier: {name}, skipping")
            continue

        # Build param grid for this classifier
        if name == "random_forest":
            param_keys = ["n_estimators", "max_depth", "random_state"]
            param_grid = list(itertools.product(
                args.rf_n_estimators,
                args.rf_max_depth,
                [args.rf_random_state],
            ))
        elif name == "knn":
            param_keys = ["n_neighbors", "weights", "p"]
            param_grid = list(itertools.product(
                args.knn_n_neighbors,
                args.knn_weights,
                args.knn_p,
            ))
        elif name == "svm_rbf":
            param_keys = ["C", "gamma", "random_state"]
            param_grid = list(itertools.product(
                args.svm_c,
                args.svm_gamma,
                [args.svm_random_state],
            ))
        elif name == "svm_linear":
            param_keys = ["C", "random_state"]
            param_grid = list(itertools.product(
                args.svm_c,
                [args.svm_random_state],
            ))
        else:
            continue

        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {name}: {len(param_grid)} combination(s)")

        for combo in param_grid:
            params = dict(zip(param_keys, combo))
            label = _param_label(name, params)
            prefix = f"{args.output_name}_" if args.output_name else ""

            print(f"    Training {label}...")

            clf = _make_classifier(name, **params)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])

            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - t0

            train_acc = pipeline.score(X_train, y_train)
            test_acc = pipeline.score(X_test, y_test)

            if name not in results:
                results[name] = {"combos": [], "best_combo": ""}
            results[name]["combos"].append({
                "label": label,
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "train_time_s": round(train_time, 3),
                "params": str(clf.get_params()),
            })
            if test_acc > results[name].get("best_score", -1):
                results[name]["best_score"] = float(test_acc)
                results[name]["best_combo"] = label

            print(f"      Train accuracy: {train_acc:.4f}")
            print(f"      Test accuracy:  {test_acc:.4f}")
            print(f"      Time:           {train_time:.3f}s")

            # confusion matrix
            y_pred = pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=[label_map[g] for g in ordered_gestures if g in label_map])
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
            n = len(ordered_gestures)
            fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
            display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=ordered_gestures)
            display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f",
                         text_kw={"fontsize": max(6, min(14, 14 - n * 0.3))})
            ax.set_xlabel("Predicted Gesture (%)")
            ax.set_ylabel("Actual Gesture (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{_pretty_name(name)}\nTest accuracy: {test_acc:.3f}")
            fig.tight_layout()
            cm_path = model_dir / f"{prefix}{label}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=180)
            plt.close(fig)

            # save model
            # Infer window_size from delta feature count in the feature names.
            n_delta = sum(1 for fn in feature_names if 'delta' in fn)
            _inferred_window = (n_delta // 8) + 1 if n_delta > 0 else 1

            model_data = {
                "pipeline": pipeline,
                "gestures": gestures,
                "label_map": label_map,
                "feature_names": feature_names,
                "train_params": {
                    "window_size": _inferred_window,
                    "classifier": name,
                    "param_label": label,
                },
            }
            model_path = model_dir / f"{prefix}{label}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"      Saved: {model_path}")

            if test_acc > best_score:
                best_score = test_acc
                best_name = label
                best_classifier_name = name
                best_pipeline = pipeline

    # Save best model copy to top-level
    if best_pipeline is not None:
        # Infer window_size from delta feature count
        n_delta = sum(1 for fn in feature_names if 'delta' in fn)
        _inferred_window = (n_delta // 8) + 1 if n_delta > 0 else 1
        y_pred = best_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[label_map[g] for g in ordered_gestures if g in label_map])
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        n = len(ordered_gestures)
        fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
        display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=ordered_gestures)
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f",
                     text_kw={"fontsize": max(6, min(14, 14 - n * 0.3))})
        ax.set_xlabel("Predicted Gesture (%)")
        ax.set_ylabel("Actual Gesture (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"Best: {_pretty_name(best_classifier_name)}\nTest accuracy: {best_score:.3f}")
        fig.tight_layout()
        best_cm_path = out_dir / "best_model_confusion_matrix.png"
        fig.savefig(best_cm_path, dpi=180)
        plt.close(fig)
        print(f"\nBest model ({best_name}, {best_score:.4f}) confusion matrix: {best_cm_path}")

        model_data = {
            "pipeline": best_pipeline,
            "gestures": gestures,
            "label_map": label_map,
            "feature_names": feature_names,
            "train_params": {
                "window_size": _inferred_window,
                "classifier": best_classifier_name,
                "param_label": best_name,
            },
        }
        best_dst = out_dir / "best_model.pkl"
        with open(best_dst, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Best model saved: {best_dst}")

    results_meta = {
        "num_train": len(X_train),
        "num_test": len(X_test),
        "num_features": X_train.shape[1],
        "feature_names": feature_names,
        "gestures": gestures,
        "label_map": label_map,
        "results": results,
        "best_model": best_name,
        "best_score": best_score,
    }

    results_path = out_dir / "train_results.json"
    with open(results_path, "w") as f:
        json.dump(results_meta, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
