#!/usr/bin/env python3
"""Standalone gesture-model inference — one file, share anywhere.

Quickest way for someone else to use your trained model:
    pip install numpy scikit-learn
    python run_model.py --demo

All you need is this file + the .pkl model.  No other project files required.

Modes:
    --demo        Run with synthetic test data (works immediately, no hardware)
    --features    6.2 3.1 ...    Predict on space-separated feature values
    --csv data.csv               Predict on each row of a CSV (one feature per column)
    --info                       Print model metadata and exit
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: str | Path) -> tuple:
    """Load a trained model pickle and return (pipeline, gestures, feature_names)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        pipeline = raw["pipeline"]
        gestures = list(raw.get("gestures", []))
        feature_names = list(raw.get("feature_names", []))
    else:
        pipeline = raw
        gestures = []
        feature_names = []

    return pipeline, gestures, feature_names


def print_model_info(
    pipeline, gestures: list[str], feature_names: list[str], path: Path
) -> None:
    """Print a human-readable summary of the loaded model."""
    n_features = pipeline.n_features_in_
    classes = getattr(pipeline, "classes_", None)
    n_classes = (
        len(classes)
        if classes is not None
        else (len(gestures) if gestures else "?")
    )

    print(f"Model:         {path.name}")
    print(f"Pipeline:      {type(pipeline).__name__}")
    if hasattr(pipeline, "steps"):
        for name, step in pipeline.steps:
            print(f"  ├─ {name}: {type(step).__name__}")
    print(f"Features:      {n_features}")
    print(f"Gestures:      {n_classes}")
    if gestures:
        for g in gestures:
            print(f"  └─ {g}")
    if feature_names:
        print("Feature names: (first 5 shown)")
        for name in feature_names[:5]:
            print(f"  └─ {name}")
        if len(feature_names) > 5:
            print(f"  … and {len(feature_names) - 5} more")
    print()

    # Detect what sensor type the model likely expects
    sensor = "mmwave" if any("mm_" in f for f in feature_names) else "imu"
    print(f"Likely sensor: {sensor}")


# ---------------------------------------------------------------------------
# Demo / synthetic data
# ---------------------------------------------------------------------------


def generate_demo_features(n_features: int) -> np.ndarray:
    """Generate plausible synthetic feature values for a quick test."""
    rng = np.random.default_rng(42)
    # 50/50 chance of pattern vs. noise
    if rng.random() < 0.5:
        return rng.normal(0.5, 0.3, size=n_features).astype(np.float32)
    return rng.normal(-0.3, 0.3, size=n_features).astype(np.float32)


def generate_multi_demo(pipeline, gestures: list[str], n_features: int) -> None:
    """Run several demo predictions in sequence to show the model working."""
    print("── Demo runs ──")
    for i in range(5):
        feats = generate_demo_features(n_features)
        label, conf = classify(pipeline, feats, gestures)
        print(f"  #{i + 1}  →  {label:>12}  (conf={conf:.3f})")
        time.sleep(0.15)
    print()


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def classify(
    pipeline, features: np.ndarray, gestures: list[str]
) -> tuple[str, float]:
    """Run a single prediction and return (label_string, confidence)."""
    if features.ndim == 1:
        features = features.reshape(1, -1)

    pred = int(pipeline.predict(features)[0])

    # Map integer → gesture name
    if gestures and pred < len(gestures):
        label = gestures[pred]
    else:
        label = str(pred)

    # Confidence via predict_proba if available
    conf = 0.0
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)[0]
        conf = float(max(proba))

    return label, conf


# ---------------------------------------------------------------------------
# CSV inference
# ---------------------------------------------------------------------------


def predict_csv(
    pipeline, csv_path: Path, gestures: list[str], n_features: int
) -> None:
    """Run predictions on each row of a CSV file."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty.")
        return

    print(f"Predicting {len(rows)} rows from {csv_path} …")
    print(f"{'#':>4}  {'Prediction':>14}  {'Confidence':>10}")
    print("-" * 36)
    for i, row in enumerate(rows):
        values = []
        for v in row:
            try:
                values.append(float(v.strip()))
            except ValueError:
                continue
        if len(values) != n_features:
            print(
                f"  Row {i}: expected {n_features} features, got {len(values)} — skipping"
            )
            continue
        feats = np.array(values, dtype=np.float32)
        label, conf = classify(pipeline, feats, gestures)
        print(f"{i:>4}  {label:>14}  {conf:>10.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained gesture model. Share this file + the .pkl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_model.py --demo\n"
            "  python run_model.py --info\n"
            "  python run_model.py --features 6.2 3.1 -0.5 1.2 0.8 0.0\n"
            "  python run_model.py --csv my_features.csv\n"
        ),
    )
    parser.add_argument(
        "--model",
        default="models/best_model.pkl",
        help="Path to trained model .pkl file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic test data (no sensor needed)",
    )
    parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        metavar="F",
        help="Space-separated feature values to classify",
    )
    parser.add_argument(
        "--csv",
        type=str,
        metavar="FILE",
        help="CSV of feature values (one prediction per row)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model metadata and exit",
    )
    args = parser.parse_args(argv)

    # ── Load model ──────────────────────────────────────────────────────
    model_path = Path(args.model)
    try:
        pipeline, gestures, feature_names = load_model(model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    n_features = pipeline.n_features_in_

    # ── Info mode ──────────────────────────────────────────────────────
    if args.info:
        print_model_info(pipeline, gestures, feature_names, model_path)
        return 0

    # ── Print header ───────────────────────────────────────────────────
    print(f"Model:  {model_path.name}")
    print(f"Gestures: {', '.join(gestures) if gestures else '?'}")
    print(f"Features: {n_features}")
    print()

    # ── Demo mode ──────────────────────────────────────────────────────
    if args.demo:
        feats = generate_demo_features(n_features)
        label, conf = classify(pipeline, feats, gestures)
        print(f"Prediction:  {label}  (confidence: {conf:.3f})")
        print()
        generate_multi_demo(pipeline, gestures, n_features)
        return 0

    # ── Feature values from CLI ────────────────────────────────────────
    if args.features is not None:
        feats = np.array(args.features, dtype=np.float32)
        if feats.ndim == 0:
            feats = feats.reshape(1)
        if feats.shape[0] != n_features:
            print(
                f"Error: expected {n_features} features but got {feats.shape[0]}",
                file=sys.stderr,
            )
            return 1
        label, conf = classify(pipeline, feats, gestures)
        print(f"Prediction:  {label}  (confidence: {conf:.3f})")
        return 0

    # ── CSV mode ───────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
            return 1
        predict_csv(pipeline, csv_path, gestures, n_features)
        return 0

    # ── No action ──────────────────────────────────────────────────────
    parser.print_help()
    print(f"\nTip: run with --demo to see it work, or --info to inspect the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
