from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSIFIERS = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "svm_rbf": SVC(kernel="rbf", probability=True, random_state=42),
    "svm_linear": SVC(kernel="linear", probability=True, random_state=42),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture classification models")
    parser.add_argument("--input", default=None,
                        help="Input feature NPZ file (default: latest features_*.npz in data/processed)")
    parser.add_argument("--output", default="models",
                        help="Output directory for trained models")
    parser.add_argument("--output-name", default=None,
                        help="Optional prefix for model files (e.g. 'mmwave' → 'models/mmwave_rf.pkl')")
    parser.add_argument("--classifiers", nargs="+",
                        default=list(CLASSIFIERS.keys()),
                        choices=list(CLASSIFIERS.keys()),
                        help="Classifiers to train and compare")
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

    data = np.load(input_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist() if "feature_names" in data else []
    gestures = data["gestures"].tolist() if "gestures" in data else []
    label_map = data["label_map"].item() if "label_map" in data else {}

    print(f"Training data: {X_train.shape}")
    print(f"Test data:     {X_test.shape}")
    print(f"Classes:       {len(gestures)} ({', '.join(gestures)})")
    print()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    best_score = 0.0
    best_name = ""
    best_pipeline = None

    for name in args.classifiers:
        if name not in CLASSIFIERS:
            print(f"  Unknown classifier: {name}, skipping")
            continue

        print(f"  Training {name}...")
        clf = CLASSIFIERS[name]
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
        else:
            proba = None

        results[name] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_time_s": round(train_time, 3),
            "params": str(clf.get_params()),
        }

        print(f"    Train accuracy: {train_acc:.4f}")
        print(f"    Test accuracy:  {test_acc:.4f}")
        print(f"    Time:           {train_time:.3f}s")

        if test_acc > best_score:
            best_score = test_acc
            best_name = name
            best_pipeline = pipeline

        model_data = {
            "pipeline": pipeline,
            "gestures": gestures,
            "label_map": label_map,
            "feature_names": feature_names,
        }
        prefix = f"{args.output_name}_" if args.output_name else ""
        model_path = out_dir / f"{prefix}{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"    Saved: {model_path}")
        print()

    if best_name and best_pipeline is not None:
        model_data = {
            "pipeline": best_pipeline,
            "gestures": gestures,
            "label_map": label_map,
            "feature_names": feature_names,
        }
        prefix = f"{args.output_name}_" if args.output_name else ""
        best_dst = out_dir / f"{prefix}best_model.pkl"
        with open(best_dst, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Best model: {best_name} ({best_score:.4f}) -> {best_dst}")

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
