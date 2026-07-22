#!/usr/bin/env python3
"""Prepare a feature matrix from recorded sensor sessions for ML training.

Loads all session CSV files from ``data/raw/``, aligns observations across
sensors into time windows, and outputs a clean train/test split.

Usage
-----
    python -m training.prepare_dataset \\
        --data-dir data/raw \\
        --output-dir data/processed \\
        --window-s 1.0 \\
        --test-size 0.2

Each row in the output matrix = one time window with:
    - One column per sensor (its confidence value, NaN if silent in window)
    - A missing flag per sensor (0/1)
    - The activity label
    - session_id, participant_id, start_timestamp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def _parse_iso(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_sessions(data_dir: str) -> list[dict[str, Any]]:
    """Read all CSV recording files and return per-row dicts."""
    path = Path(data_dir)
    csv_files = sorted(path.glob("session_*.csv"))
    if not csv_files:
        print(f"[prepare_dataset] No session CSV files found in {data_dir}")
        return []

    all_rows: list[dict[str, Any]] = []
    for fpath in csv_files:
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_file"] = fpath.name
                all_rows.append(row)
    print(f"[prepare_dataset] Loaded {len(all_rows)} rows from {len(csv_files)} files")
    return all_rows


def build_feature_matrix(
    rows: list[dict[str, Any]],
    window_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Align observations into fixed-duration time windows.

    Returns
    -------
    X : np.ndarray  shape=(n_windows, n_sensors * 2)  — confidence + missing flag per sensor
    y : np.ndarray  shape=(n_windows,)                 — numeric label
    sensor_ids : list[str]                              — column names for X
    metadata : list[dict]                               — per-window session/participant info
    """
    if not rows:
        return np.array([]), np.array([]), [], []

    # Collect unique sensor ids across all sessions
    all_sensor_ids: list[str] = []
    seen: set[str] = set()
    for r in rows:
        sid = r.get("sensor_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            all_sensor_ids.append(sid)
    all_sensor_ids.sort()

    # Collect unique labels
    label_values: list[str] = []
    label_seen: set[str] = set()
    for r in rows:
        lbl = r.get("label", "")
        if lbl and lbl not in label_seen:
            label_seen.add(lbl)
            label_values.append(lbl)
    label_values.sort()
    label_to_int = {lbl: i for i, lbl in enumerate(label_values)}

    # Group rows by session
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sessions[r.get("session_id", "unknown")].append(r)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    for session_id, session_rows in sessions.items():
        session_rows.sort(key=lambda r: _parse_iso(r.get("timestamp", "")))
        if not session_rows:
            continue

        participant = session_rows[0].get("participant_id", "")
        label = session_rows[0].get("label", "")
        label_int = label_to_int.get(label, -1)
        if label_int < 0:
            continue

        # Determine window boundaries from session timestamps
        start_ts = _parse_iso(session_rows[0]["timestamp"])
        end_ts = _parse_iso(session_rows[-1]["timestamp"])
        window_delta = timedelta(seconds=window_s)

        current_start = start_ts
        while current_start < end_ts:
            current_end = current_start + window_delta
            # Collect confidences per sensor in this window
            window_data: dict[str, list[float]] = defaultdict(list)
            for r in session_rows:
                ts = _parse_iso(r["timestamp"])
                if current_start <= ts < current_end:
                    try:
                        conf = float(r.get("confidence", 0.0))
                    except (ValueError, TypeError):
                        conf = 0.0
                    window_data[r["sensor_id"]].append(conf)

            # Build feature vector: [conf_s1, flag_s1, conf_s2, flag_s2, ...]
            feat = []
            for sid in all_sensor_ids:
                vals = window_data.get(sid, [])
                if vals:
                    feat.append(np.mean(vals))
                    feat.append(0)  # not missing
                else:
                    feat.append(0.0)
                    feat.append(1)  # missing
            X_rows.append(np.array(feat, dtype=np.float64))
            y_rows.append(label_int)
            metadata_rows.append({
                "session_id": session_id,
                "participant_id": participant,
                "label": label,
                "window_start": current_start.isoformat(),
                "window_end": current_end.isoformat(),
            })
            current_start = current_end

    if not X_rows:
        return np.array([]), np.array([]), [], []

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    col_names: list[str] = []
    for sid in all_sensor_ids:
        col_names.append(f"{sid}_confidence")
        col_names.append(f"{sid}_missing")

    print(f"[prepare_dataset] Feature matrix: {X.shape[0]} windows × {X.shape[1]} features")
    print(f"[prepare_dataset] Classes: {label_values}")
    return X, y, col_names, metadata_rows


def save_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    col_names: list[str],
    label_names: list[str],
    meta_train: list[dict[str, Any]],
    meta_test: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Save train/test splits as compressed .npz + metadata JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "dataset.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        col_names=col_names,
        label_names=label_names,
    )

    with open(out / "metadata_train.json", "w") as f:
        json.dump(meta_train, f, indent=2)
    with open(out / "metadata_test.json", "w") as f:
        json.dump(meta_test, f, indent=2)

    print(f"[prepare_dataset] Saved to {out / 'dataset.npz'}")
    print(f"[prepare_dataset]   Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sensor dataset for ML training.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing session CSVs")
    parser.add_argument("--output-dir", default="data/processed", help="Where to write dataset.npz")
    parser.add_argument("--window-s", type=float, default=1.0, help="Time window in seconds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    args = parser.parse_args()

    rows = load_sessions(args.data_dir)
    if not rows:
        sys.exit(1)

    X, y, col_names, metadata = build_feature_matrix(rows, window_s=args.window_s)
    if X.size == 0:
        print("[prepare_dataset] No windows generated — check CSV data")
        sys.exit(1)

    label_names = sorted({m["label"] for m in metadata})

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata, test_size=args.test_size, stratify=y, random_state=42,
    )

    save_split(X_train, X_test, y_train, y_test, col_names, label_names, meta_train, meta_test, args.output_dir)


if __name__ == "__main__":
    main()
